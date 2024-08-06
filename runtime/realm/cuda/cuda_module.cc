/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_access.h"
#include "realm/cuda/cuda_internal.h"
#include "realm/cuda/cuda_memcpy.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/event_impl.h"
#include "realm/idx_impl.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_USE_CUDART_HIJACK
#include "realm/cuda/cudart_hijack.h"
#endif

#ifdef REALM_USE_DLFCN
  #include <dlfcn.h>
#endif

#include "realm/mutex.h"
#include "realm/utils.h"

#ifdef REALM_USE_VALGRIND_ANNOTATIONS
#include <valgrind/memcheck.h>
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

#if CUDA_VERSION < 11030
// Define cuGetProcAddress if it isn't defined, so we can query for it's existence later
typedef CUresult CUDAAPI (*PFN_cuGetProcAddress)(const char *, void **, int, int);
#define CU_GET_PROC_ADDRESS_DEFAULT 0
#endif

#define IS_DEFAULT_STREAM(stream)   \
  (((stream) == 0) || ((stream) == CU_STREAM_LEGACY) || ((stream) == CU_STREAM_PER_THREAD))

// The embedded fat binary that holds all the internal
// realm cuda kernels (see generated file realm_fatbin.c)
extern const unsigned char realm_fatbin[];

namespace Realm {

  extern Logger log_taskreg;

  namespace Cuda {

    enum CudaIpcResponseType
    {
      CUDA_IPC_RESPONSE_TYPE_IPC = 0,
      CUDA_IPC_RESPONSE_TYPE_FABRIC,
    };

    /// @brief Helper structure for describing a particular memory being imported via CUDA
    /// IPC
    struct CudaIpcResponseEntry {
      /// UUID of the source gpu this memory belongs to
      CUuuid src_gpu_uuid;
      /// Realm::Memory associated with this entry
      Memory mem;
      /// The base GPU address from the source GPU in the source process
      uintptr_t base_ptr = 0;
      size_t size = 0;
      CudaIpcResponseType type = CUDA_IPC_RESPONSE_TYPE_IPC;
      union CudaIpcHandle {
        /// IPC handle to be opened for the underlying memory
        CUipcMemHandle ipc_handle;
#if CUDA_VERSION >= 12030
        struct CudaFabricInfo {
          unsigned clique_id;
          CUuuid cluster_uuid;
          CUmemFabricHandle handle;
        } fabric;
#endif
      } data;
    };

    Logger log_gpu("gpu");
    Logger log_gpudma("gpudma");
    Logger log_cudart("cudart");
    Logger log_cudaipc("cudaipc");

    Logger log_stream("gpustream");
    bool nvml_api_fnptrs_loaded = false;
    bool nvml_initialized = false;
    CUresult cuda_init_code = CUDA_ERROR_UNKNOWN;

    bool cuda_api_fnptrs_loaded = false;

#if CUDA_VERSION >= 11030
// cuda 11.3+ gives us handy PFN_... types
#define DEFINE_FNPTR(name, ver) PFN_##name name##_fnptr = 0;
#else
// before cuda 11.3, we have to rely on typeof/decltype
#define DEFINE_FNPTR(name, ver) decltype(&name) name##_fnptr = 0;
#endif
    CUDA_DRIVER_APIS(DEFINE_FNPTR);
#undef DEFINE_FNPTR

    static unsigned ctz(uint64_t v) {
#ifdef REALM_ON_WINDOWS
      unsigned long index;
#ifdef _WIN64
      if (_BitScanForward64(&index, v)) return index;
#else
      unsigned v_lo = v;
      unsigned v_hi = v >> 32;
      if (_BitScanForward(&index, v_lo))
        return index;
      else if (_BitScanForward(&index, v_hi))
        return index + 32;
#endif
      else
        return 0;
#else
      return __builtin_ctzll(v);
#endif
    }

#define DEFINE_FNPTR(name) decltype(&name) name##_fnptr = 0;

    NVML_APIS(DEFINE_FNPTR);
#undef DEFINE_FNPTR

    // function pointers for cuda hook
    typedef void (*PFN_cuhook_register_callback)(void);
    typedef void (*PFN_cuhook_start_task)(GPUProcessor *gpu_proc);
    typedef void (*PFN_cuhook_end_task)(CUstream current_task_stream);

    static PFN_cuhook_register_callback cuhook_register_callback_fnptr = nullptr;
    static PFN_cuhook_start_task cuhook_start_task_fnptr = nullptr;
    static PFN_cuhook_end_task cuhook_end_task_fnptr = nullptr;
    static bool cuhook_enabled = false;

    namespace ThreadLocal {
      static REALM_THREAD_LOCAL GPUProcessor *current_gpu_proc = 0;
      static REALM_THREAD_LOCAL GPUStream *current_gpu_stream = 0;
      static REALM_THREAD_LOCAL std::set<GPUStream *> *created_gpu_streams = 0;
      static REALM_THREAD_LOCAL int context_sync_required = 0;
    }; // namespace ThreadLocal

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUStream

    GPUStream::GPUStream(GPU *_gpu, GPUWorker *_worker, int rel_priority /*= 0*/)
      : gpu(_gpu)
      , worker(_worker)
    {
      assert(worker != 0);
      // the math here is designed to balance the context's priority range
      //  around a relative priority of 0, favoring an extra negative (higher
      //  priority) option
      int abs_priority = (gpu->greatest_stream_priority +
                          rel_priority +
                          ((gpu->least_stream_priority -
                            gpu->greatest_stream_priority + 1) / 2));
      // CUDA promises to clamp to the actual range, so we don't have to
      CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamCreateWithPriority)
                (&stream, CU_STREAM_NON_BLOCKING, abs_priority) );
      log_stream.info() << "stream created: gpu=" << gpu
                        << " stream=" << stream << " priority=" << abs_priority;
    }

    GPUStream::~GPUStream(void)
    {
      // log_stream.info() << "CUDA stream " << stream << " destroyed - max copies = " 
      // 			<< pending_copies.capacity() << ", max events = " << pending_events.capacity();

      CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamDestroy)(stream) );
    }

    GPU *GPUStream::get_gpu(void) const
    {
      return gpu;
    }
    
    CUstream GPUStream::get_stream(void) const
    {
      return stream;
    }

    void GPUStream::add_fence(GPUWorkFence *fence)
    {
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, stream) );

      log_stream.debug() << "CUDA fence event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      add_event(e, fence, 0);
    } 

    void GPUStream::add_start_event(GPUWorkStart *start)
    {
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, stream) );

      log_stream.debug() << "CUDA start event " << e << " recorded on stream " << stream << " (GPU " << gpu << ")";

      // record this as a start event
      add_event(e, 0, 0, start);
    }

    void GPUStream::add_notification(GPUCompletionNotification *notification)
    {
      CUevent e = gpu->event_pool.get_event();

      CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, stream) );

      add_event(e, 0, notification);
    }

    void GPUStream::add_event(CUevent event, GPUWorkFence *fence, 
			      GPUCompletionNotification *notification, GPUWorkStart *start)
    {
      bool add_to_worker = false;
      {
	AutoLock<> al(mutex);

	// if we didn't already have work AND if there's not an active
	//  worker issuing copies, request attention
        add_to_worker = pending_events.empty();

        PendingEvent e;
        e.event = event;
        e.fence = fence;
        e.start = start;
        e.notification = notification;

        pending_events.push_back(e);
      }

      if(add_to_worker)
	worker->add_stream(this);
    }

    void GPUStream::wait_on_streams(const std::set<GPUStream*> &other_streams)
    {
      assert(!other_streams.empty());
      for (std::set<GPUStream*>::const_iterator it = 
            other_streams.begin(); it != other_streams.end(); it++)
      {
        if (*it == this)
          continue;
        CUevent e = gpu->event_pool.get_event();

        CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, (*it)->get_stream()) );

        log_stream.debug() << "CUDA stream " << stream << " waiting on stream " 
                           << (*it)->get_stream() << " (GPU " << gpu << ")";

        CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(stream, e, 0) );

        // record this event on our stream
        add_event(e, 0);
      }
    }

    bool GPUStream::has_work(void) const { return (!pending_events.empty()); }

    // atomically checks rate limit counters and returns true if 'bytes'
    //  worth of copies can be submitted or false if not (in which case
    //  the progress counter on the xd will be updated when it should try
    //  again)
    bool GPUStream::ok_to_submit_copy(size_t bytes, XferDes *xd)
    {
      return true;
    }

    bool GPUStream::reap_events(TimeLimit work_until)
    {
      // peek at the first event
      CUevent event;
      bool event_valid = false;
      {
	AutoLock<> al(mutex);

	if(pending_events.empty())
	  // no events left, but stream might have other work left
	  return has_work();

	event = pending_events.front().event;
	event_valid = true;
      }

      // we'll keep looking at events until we find one that hasn't triggered
      bool work_left = true;
      while(event_valid) {
	CUresult res = CUDA_DRIVER_FNPTR(cuEventQuery)(event);

	if(res == CUDA_ERROR_NOT_READY)
	  return true; // oldest event hasn't triggered - check again later

	// no other kind of error is expected
	if(res != CUDA_SUCCESS) {
	  const char *ename = 0;
	  const char *estr = 0;
	  CUDA_DRIVER_FNPTR(cuGetErrorName)(res, &ename);
	  CUDA_DRIVER_FNPTR(cuGetErrorString)(res, &estr);
	  log_gpu.fatal() << "CUDA error reported on GPU " << gpu->info->index << ": " << estr << " (" << ename << ")";
	  assert(0);
	}

	log_stream.debug() << "CUDA event " << event << " triggered on stream " << stream << " (GPU " << gpu << ")";

	// give event back to GPU for reuse
	gpu->event_pool.return_event(event);

	// this event has triggered, so figure out the fence/notification to trigger
	//  and also peek at the next event
	GPUWorkFence *fence = 0;
        GPUWorkStart *start = 0;
	GPUCompletionNotification *notification = 0;

	{
	  AutoLock<> al(mutex);

	  const PendingEvent &e = pending_events.front();
	  assert(e.event == event);
	  fence = e.fence;
          start = e.start;
	  notification = e.notification;
	  pending_events.pop_front();

	  if(pending_events.empty()) {
	    event_valid = false;
	    work_left = has_work();
	  } else
	    event = pending_events.front().event;
	}

        if (start) {
          start->mark_gpu_work_start();
        }
	if(fence)
	  fence->mark_finished(true /*successful*/);

	if(notification)
	  notification->request_completed();

	// don't repeat if we're out of time
	if(event_valid && work_until.is_expired())
	  return true;
      }

      // if we get here, we ran out of events, but there might have been
      //  other kinds of work that we need to let the caller know about
      return work_left;
    }

    void GPU::create_dma_channels(Realm::RuntimeImpl *r)
    {
      // we used to skip gpu dma channels when there was no fbmem, but in
      //  theory nvlink'd gpus can help move sysmem data even, so let's just
      //  always create the channels

      r->add_dma_channel(new GPUChannel(this, XFER_GPU_IN_FB, &r->bgwork));
      r->add_dma_channel(new GPUIndirectChannel(this, XFER_GPU_SC_IN_FB, &r->bgwork));
      r->add_dma_channel(new GPUfillChannel(this, &r->bgwork));
      r->add_dma_channel(new GPUreduceChannel(this, &r->bgwork));

      // treat managed mem like pinned sysmem on the assumption that most data
      //  is usually in system memory
      if(!pinned_sysmems.empty() || !managed_mems.empty()) {
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_TO_FB, &r->bgwork));
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_FROM_FB, &r->bgwork));
      } else {
        log_gpu.warning() << "GPU " << proc->me << " has no accessible system memories!?";
      }

      // only create a p2p channel if we have peers (and an fb)
      if(!peer_fbs.empty() || !cudaipc_mappings.empty()) {
        r->add_dma_channel(new GPUChannel(this, XFER_GPU_PEER_FB, &r->bgwork));
        r->add_dma_channel(new GPUIndirectChannel(this, XFER_GPU_SC_PEER_FB, &r->bgwork));
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUWorkFence

    GPUWorkFence::GPUWorkFence(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {}

    void GPUWorkFence::request_cancellation(void)
    {
      // ignored - no way to shoot down CUDA work
    }

    void GPUWorkFence::print(std::ostream& os) const
    {
      os << "GPUWorkFence";
    }

    void GPUWorkFence::enqueue_on_stream(GPUStream *stream)
    {
      if(stream->get_gpu()->module->config->cfg_fences_use_callbacks) {
	CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamAddCallback)(stream->get_stream(), &cuda_callback, (void *)this, 0) );
      } else {
	stream->add_fence(this);
      }
    }

    /*static*/ void GPUWorkFence::cuda_callback(CUstream stream, CUresult res, void *data)
    {
      GPUWorkFence *me = (GPUWorkFence *)data;

      assert(res == CUDA_SUCCESS);
      me->mark_finished(true /*succesful*/);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUWorkStart
    GPUWorkStart::GPUWorkStart(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {
    }

    void GPUWorkStart::print(std::ostream& os) const
    {
      os << "GPUWorkStart";
    }

    void GPUWorkStart::enqueue_on_stream(GPUStream *stream)
    {
      if(stream->get_gpu()->module->config->cfg_fences_use_callbacks) {
	CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamAddCallback)(stream->get_stream(), &cuda_start_callback, (void *)this, 0) );
      } else {
	stream->add_start_event(this);
      }
    }

    void GPUWorkStart::mark_gpu_work_start()
    {
      op->mark_gpu_work_start();
      mark_finished(true);
    }

    /*static*/ void GPUWorkStart::cuda_start_callback(CUstream stream, CUresult res, void *data)
    {
      GPUWorkStart *me = (GPUWorkStart *)data;
      assert(res == CUDA_SUCCESS);
      // record the real start time for the operation
      me->mark_gpu_work_start();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUEventPool

    GPUEventPool::GPUEventPool(int _batch_size)
      : batch_size(_batch_size), current_size(0), total_size(0), external_count(0)
    {
      // don't immediately fill the pool because we're not managing the context ourselves
    }

    // allocating the initial batch of events and cleaning up are done with
    //  these methods instead of constructor/destructor because we don't
    //  manage the GPU context in this helper class
    void GPUEventPool::init_pool(int init_size /*= 0 -- default == batch size */)
    {
      assert(available_events.empty());

      if(init_size == 0)
	init_size = batch_size;

      available_events.resize(init_size);

      current_size = init_size;
      total_size = init_size;

      for(int i = 0; i < init_size; i++)
	CHECK_CU( CUDA_DRIVER_FNPTR(cuEventCreate)(&available_events[i], CU_EVENT_DISABLE_TIMING) );
    }

    void GPUEventPool::empty_pool(void)
    {
      // shouldn't be any events running around still
      assert((current_size + external_count) == total_size);
      if(external_count)
        log_stream.warning() << "Application leaking " << external_count << " cuda events";

      for(int i = 0; i < current_size; i++)
	CHECK_CU( CUDA_DRIVER_FNPTR(cuEventDestroy)(available_events[i]) );

      current_size = 0;
      total_size = 0;

      // free internal vector storage
      std::vector<CUevent>().swap(available_events);
    }

    CUevent GPUEventPool::get_event(bool external)
    {
      AutoLock<> al(mutex);

      if(current_size == 0) {
	// if we need to make an event, make a bunch
	current_size = batch_size;
	total_size += batch_size;

	log_stream.info() << "event pool " << this << " depleted - adding " << batch_size << " events";
      
	// resize the vector (considering all events that might come back)
	available_events.resize(total_size);

	for(int i = 0; i < batch_size; i++)
	  CHECK_CU( CUDA_DRIVER_FNPTR(cuEventCreate)(&available_events[i], CU_EVENT_DISABLE_TIMING) );
      }

      if(external)
        external_count++;

      return available_events[--current_size];
    }

    void GPUEventPool::return_event(CUevent e, bool external)
    {
      AutoLock<> al(mutex);

      assert(current_size < total_size);

      if(external) {
        assert(external_count);
        external_count--;
      }

      available_events[current_size++] = e;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class ContextSynchronizer

    ContextSynchronizer::ContextSynchronizer(GPU *_gpu,
					     CUcontext _context,
					     CoreReservationSet& crs,
					     int _max_threads)
      : gpu(_gpu)
      , context(_context)
      , max_threads(_max_threads)
      , condvar(mutex)
      , shutdown_flag(false)
      , total_threads(0)
      , sleeping_threads(0)
      , syncing_threads(0)
    {
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
      params.set_ldst_usage(params.CORE_USAGE_MINIMAL);
      params.set_max_stack_size(1 << 20);

      std::string name = stringbuilder() << "GPU ctxsync " << context;

      core_rsrv = new Realm::CoreReservation(name, crs, params);
    }

    ContextSynchronizer::~ContextSynchronizer()
    {
      assert(total_threads == 0);
      delete core_rsrv;
    }

    void ContextSynchronizer::shutdown_threads()
    {
      // set the shutdown flag and wake up everybody
      {
	AutoLock<> al(mutex);
	shutdown_flag = true;
	if(sleeping_threads > 0)
	  condvar.broadcast();
      }

      for(int i = 0; i < total_threads; i++) {
	worker_threads[i]->join();
	delete worker_threads[i];
      }

      worker_threads.clear();
      total_threads = false;
      sleeping_threads = false;
      syncing_threads = false;
      shutdown_flag = false;
    }

    void ContextSynchronizer::add_fence(GPUWorkFence *fence)
    {
      bool start_new_thread = false;
      {
	AutoLock<> al(mutex);

	fences.push_back(fence);

	// if all the current threads are asleep or busy syncing, we
	//  need to do something
	if((sleeping_threads + syncing_threads) == total_threads) {
	  // is there a sleeping thread we can wake up to handle this?
	  if(sleeping_threads > 0) {
	    // just poke one of them
	    condvar.signal();
	  } else {
	    // can we start a new thread?  (if not, we'll just have to
	    //  be patient)
	    if(total_threads < max_threads) {
	      total_threads++;
	      syncing_threads++; // threads starts as if it's syncing
	      start_new_thread = true;
	    }
	  }
	}
      }

      if(start_new_thread) {
	Realm::ThreadLaunchParameters tlp;

	Thread *t = Realm::Thread::create_kernel_thread<ContextSynchronizer,
							&ContextSynchronizer::thread_main>(this,
											   tlp,
											   *core_rsrv,
											   0);
	// need the mutex to put this thread in the list
	{
	  AutoLock<> al(mutex);
	  worker_threads.push_back(t);
	}
      }
    }

    void ContextSynchronizer::thread_main()
    {
      while(true) {
	GPUWorkFence::FenceList my_fences;

	// attempt to get a non-empty list of fences to synchronize,
	//  sleeping when needed and paying attention to the shutdown
	//  flag
	{
	  AutoLock<> al(mutex);

	  syncing_threads--;

	  while(true) {
	    if(shutdown_flag)
	      return;

	    if(fences.empty()) {
	      // sleep until somebody tells us there's stuff to do
	      sleeping_threads++;
	      condvar.wait();
	      sleeping_threads--;
	    } else {
	      // grab everything (a single sync covers however much stuff
	      //  was pushed ahead of it)
	      syncing_threads++;
	      my_fences.swap(fences);
	      break;
	    }
	  }
	}

	// shouldn't get here with an empty list
	assert(!my_fences.empty());

	log_stream.debug() << "starting ctx sync: ctx=" << context;

	{
	  AutoGPUContext agc(gpu);

	  CUresult res = CUDA_DRIVER_FNPTR(cuCtxSynchronize)();

	  // complain loudly about any errors
	  if(res != CUDA_SUCCESS) {
	    const char *ename = 0;
	    const char *estr = 0;
	    CUDA_DRIVER_FNPTR(cuGetErrorName)(res, &ename);
	    CUDA_DRIVER_FNPTR(cuGetErrorString)(res, &estr);
	    log_gpu.fatal() << "CUDA error reported on GPU " << gpu->info->index << ": " << estr << " (" << ename << ")";
	    abort();
	  }
	}

	log_stream.debug() << "finished ctx sync: ctx=" << context;

	// mark all the fences complete
	while(!my_fences.empty()) {
	  GPUWorkFence *fence = my_fences.pop_front();
	  fence->mark_finished(true /*successful*/);
	}

	// and go back around for more...
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUTaskScheduler<T>

    // we want to subclass the scheduler to replace the execute_task method, but we also want to
    //  allow the use of user or kernel threads, so we apply a bit of template magic (which only works
    //  because the constructors for the KernelThreadTaskScheduler and UserThreadTaskScheduler classes
    //  have the same prototypes)

    template <typename T>
    class GPUTaskScheduler : public T {
    public:
      GPUTaskScheduler(Processor _proc, Realm::CoreReservation& _core_rsrv,
		       GPUProcessor *_gpu_proc);

      virtual ~GPUTaskScheduler(void);

    protected:
      virtual bool execute_task(Task *task);
      virtual void execute_internal_task(InternalTask *task);

      // might also need to override the thread-switching methods to keep TLS up to date

      GPUProcessor *gpu_proc;
    };

    template <typename T>
    GPUTaskScheduler<T>::GPUTaskScheduler(Processor _proc,
					  Realm::CoreReservation& _core_rsrv,
					  GPUProcessor *_gpu_proc)
      : T(_proc, _core_rsrv), gpu_proc(_gpu_proc)
    {
      // nothing else
    }

    template <typename T>
    GPUTaskScheduler<T>::~GPUTaskScheduler(void)
    {
    }

#ifdef REALM_USE_CUDART_HIJACK
    // this flag will be set on the first call into any of the hijack code in
    //  cudart_hijack.cc
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    /*extern*/ bool cudart_hijack_active = false;

    // for most CUDART API entry points, calling them from a non-GPU task is
    //  a fatal error - for others (e.g. cudaDeviceSynchronize), it's either
    //  silently permitted (0), warned (1), or a fatal error (2) based on this
    //  setting
    /*extern*/ int cudart_hijack_nongpu_sync = 2;
#endif

    template <typename T>
    bool GPUTaskScheduler<T>::execute_task(Task *task)
    {
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  CUDA RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // start record cuda calls if cuda book is enabled
      if(cuhook_enabled) {
        cuhook_start_task_fnptr(ThreadLocal::current_gpu_proc);
      }

      // push the CUDA context for this GPU onto this thread
      gpu_proc->gpu->push_context();

      // bump the current stream
      // TODO: sanity-check whether this even works right when GPU tasks suspend
      assert(ThreadLocal::current_gpu_stream == 0);
      GPUStream *s = gpu_proc->gpu->get_next_task_stream();
      ThreadLocal::current_gpu_stream = s;
      assert(!ThreadLocal::created_gpu_streams);

      // a task can force context sync on task completion either on or off during
      //  execution, so use -1 as a "no preference" value
      ThreadLocal::context_sync_required = -1;

      // we'll use a "work fence" to track when the kernels launched by this task actually
      //  finish - this must be added to the task _BEFORE_ we execute
      GPUWorkFence *fence = new GPUWorkFence(task);
      task->add_async_work_item(fence);

      // event to record the GPU start time for the task, if requested
      if(task->wants_gpu_work_start()) {
	GPUWorkStart *start = new GPUWorkStart(task);
	task->add_async_work_item(start);
	start->enqueue_on_stream(s);
      }

      bool ok = T::execute_task(task);

      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when GPU tasks suspend
      if(ThreadLocal::created_gpu_streams)
      {
        s->wait_on_streams(*ThreadLocal::created_gpu_streams);
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }

      // if this is our first task, we might need to decide whether
      //  full context synchronization is required for a task to be
      //  "complete"
      if(gpu_proc->gpu->module->config->cfg_task_context_sync < 0) {
        // if legacy stream sync was requested, default for ctxsync is off
        if(gpu_proc->gpu->module->config->cfg_task_legacy_sync) {
          gpu_proc->gpu->module->config->cfg_task_context_sync = 0;
        } else {
#ifdef REALM_USE_CUDART_HIJACK
          // normally hijack code will catch all the work and put it on the
          //  right stream, but if we haven't seen it used, there may be a
          //  static copy of the cuda runtime that's in use and foiling the
          //  hijack
          if(cudart_hijack_active) {
            gpu_proc->gpu->module->config->cfg_task_context_sync = 0;
          } else {
            if(!gpu_proc->gpu->module->config->cfg_suppress_hijack_warning)
              log_gpu.warning() << "CUDART hijack code not active"
                                << " - device synchronizations required after every GPU task!";
            gpu_proc->gpu->module->config->cfg_task_context_sync = 1;
          }
#else
          // without hijack or legacy sync requested, ctxsync is needed
          gpu_proc->gpu->module->config->cfg_task_context_sync = 1;
#endif
        }
      }

      // if requested, use a cuda event to couple legacy stream work into
      //  the current task's stream
      if(gpu_proc->gpu->module->config->cfg_task_legacy_sync) {
        CUevent e = gpu_proc->gpu->event_pool.get_event();
        CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, CU_STREAM_LEGACY) );
        CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(s->get_stream(), e, 0) );
        gpu_proc->gpu->event_pool.return_event(e);
      }

      if((ThreadLocal::context_sync_required > 0) ||
         ((ThreadLocal::context_sync_required < 0) &&
          gpu_proc->gpu->module->config->cfg_task_context_sync)) {
#if(CUDA_VERSION >= 12050)
        // If this driver supports retrieving an event for the context's current work,
        // retrieve and wait for it.  This will still over-synchronize with work from the
        // DMA engine, but at least this is completely asynchronous and doesn't require a
        // separate thread.
        if(CUDA_DRIVER_HAS_FNPTR(cuCtxRecordEvent)) {
          CUevent e = gpu_proc->gpu->event_pool.get_event();
          CHECK_CU(CUDA_DRIVER_FNPTR(cuCtxRecordEvent)(gpu_proc->gpu->context, e));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(s->get_stream(), e, 0));
          s->add_event(e, fence);
        } else
#endif
        {
          // Add the context sync to the thread
          gpu_proc->ctxsync.add_fence(fence);
        }
      } else {
        // Just wait for the fence
        fence->enqueue_on_stream(s);
      }

      // A useful debugging macro
#ifdef FORCE_GPU_STREAM_SYNCHRONIZE
      CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamSynchronize)(s->get_stream()) );
#endif

      // pop the CUDA context for this GPU back off
      gpu_proc->gpu->pop_context();

      // cuda stream sanity check and clear cuda hook calls
      // we only check against the current_gpu_stream because it is impossible to launch
      // tasks onto other realm gpu streams
      if(cuhook_enabled) {
        cuhook_end_task_fnptr(ThreadLocal::current_gpu_stream->get_stream());
      }

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;
      assert(ThreadLocal::current_gpu_stream == s);
      ThreadLocal::current_gpu_stream = 0;

      return ok;
    }

    template <typename T>
    void GPUTaskScheduler<T>::execute_internal_task(InternalTask *task)
    {
      // use TLS to make sure that the task can find the current GPU processor when it makes
      //  CUDA RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_gpu_proc == 0);
      ThreadLocal::current_gpu_proc = gpu_proc;

      // push the CUDA context for this GPU onto this thread
      gpu_proc->gpu->push_context();

      assert(ThreadLocal::current_gpu_stream == 0);
      GPUStream *s = gpu_proc->gpu->get_next_task_stream();
      ThreadLocal::current_gpu_stream = s;
      assert(!ThreadLocal::created_gpu_streams);

      // internal tasks aren't allowed to wait on events, so any cuda synch
      //  calls inside the call must be blocking
      gpu_proc->block_on_synchronize = true;

      // execute the internal task, whatever it is
      T::execute_internal_task(task);

      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when GPU tasks suspend
      if(ThreadLocal::created_gpu_streams)
      {
        s->wait_on_streams(*ThreadLocal::created_gpu_streams);
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }

      // we didn't use streams here, so synchronize the whole context
      CHECK_CU( CUDA_DRIVER_FNPTR(cuCtxSynchronize)() );
      gpu_proc->block_on_synchronize = false;

      // pop the CUDA context for this GPU back off
      gpu_proc->gpu->pop_context();

      assert(ThreadLocal::current_gpu_proc == gpu_proc);
      ThreadLocal::current_gpu_proc = 0;
      assert(ThreadLocal::current_gpu_stream == s);
      ThreadLocal::current_gpu_stream = 0;
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUProcessor

    GPUProcessor::GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::TOC_PROC)
      , gpu(_gpu)
      , block_on_synchronize(false)
      , ctxsync(_gpu, _gpu->context, crs, _gpu->module->config->cfg_max_ctxsync_threads)
    {
      Realm::CoreReservationParameters params;

      if (_gpu->info->has_numa_preference) {
        // Pick the first numa domain in the retrieved numa mask that is available
        // TODO: pass the mask directly to params instead of picking the first one
        const Realm::CoreMap::DomainMap& available_domains = crs.get_core_map()->by_domain;
        for (size_t numa_idx = 0; numa_idx < _gpu->info->MAX_NUMA_NODE_LEN; numa_idx++) {
          int numa_domain = 0;
          bool found_numa = false;
          for (size_t numa_offset = 0; numa_offset < sizeof(_gpu->info->numa_node_affinity[0]); numa_offset++) {
            numa_domain = numa_offset + numa_idx * sizeof(_gpu->info->numa_node_affinity[0]);
            if ((_gpu->info->numa_node_affinity[numa_idx] & (1UL << numa_offset)) &&
                available_domains.find(numa_domain) != available_domains.end()) {
              found_numa = true;
              break;
            }
          }
          if (found_numa) {
            params.set_numa_domain(numa_domain);
            break;
          }
        }
      }
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "GPU proc " << _me;

      core_rsrv = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS_FOR_GPU
      Realm::UserThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::UserThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#else
      Realm::KernelThreadTaskScheduler *sched = new GPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);
      // no config settings we want to tweak yet
#endif
      set_scheduler(sched);
    }

    GPUProcessor::~GPUProcessor(void)
    {
      delete core_rsrv;
    }

    GPUStream* GPU::find_stream(CUstream stream) const
    {
      for (std::vector<GPUStream*>::const_iterator it = 
            task_streams.begin(); it != task_streams.end(); it++)
        if ((*it)->get_stream() == stream)
          return *it;
      return NULL;
    }

    bool GPU::can_access_peer(const GPU *peer) const {
      return (peer != NULL) &&
             (info->peers.find(peer->info->device) != info->peers.end());
    }

    GPUStream* GPU::get_null_task_stream(void) const
    {
      GPUStream *stream = ThreadLocal::current_gpu_stream;
      assert(stream != NULL);
      return stream;
    }

    GPUStream* GPU::get_next_task_stream(bool create)
    {
      if(create && !ThreadLocal::created_gpu_streams)
      {
        // First time we get asked to create, user our current stream
        ThreadLocal::created_gpu_streams = new std::set<GPUStream*>();
        assert(ThreadLocal::current_gpu_stream);
        ThreadLocal::created_gpu_streams->insert(ThreadLocal::current_gpu_stream);
        return ThreadLocal::current_gpu_stream;
      }
      unsigned index = next_task_stream.fetch_add(1) % task_streams.size();
      GPUStream *result = task_streams[index];
      if (create)
        ThreadLocal::created_gpu_streams->insert(result);
      return result;
    }

    GPUStream *GPU::get_next_d2d_stream()
    {
      unsigned d2d_stream_index = (next_d2d_stream.fetch_add(1) %
                                   module->config->cfg_d2d_streams);
      return device_to_device_streams[d2d_stream_index];
    }

    static void launch_kernel(const Realm::Cuda::GPU::GPUFuncInfo &func_info, void *params,
                              size_t num_elems, GPUStream *stream)
    {
      unsigned int num_blocks = 0, num_threads = 0;
      void *args[] = {params};

      num_threads = std::min(static_cast<unsigned int>(func_info.occ_num_threads),
                             static_cast<unsigned int>(num_elems));
      num_blocks = std::min(
          static_cast<unsigned int>((num_elems + num_threads - 1) / num_threads),
          static_cast<unsigned int>(
              func_info.occ_num_blocks)); // Cap the grid based on the given volume

      CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchKernel)(func_info.func, num_blocks, 1, 1,
                                                 num_threads, 1, 1, 0,
                                                 stream->get_stream(), args, nullptr));
    }

    void GPU::launch_transpose_kernel(MemcpyTransposeInfo<size_t> &copy_info,
                                      size_t elem_size, GPUStream *stream)
    {
      size_t log_elem_size = std::min(static_cast<size_t>(ctz(elem_size)),
                                      CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES - 1);
      size_t num_elems = copy_info.extents[1] * copy_info.extents[2];
      assert((1ULL << log_elem_size) <= elem_size);

      GPUFuncInfo &func_info = transpose_kernels[log_elem_size];

      unsigned int num_blocks = 0, num_threads = 0;
      assert(copy_info.extents[0] <= CUDA_MAX_FIELD_BYTES);

      size_t chunks = copy_info.extents[0] / elem_size;
      copy_info.tile_size = static_cast<size_t>(
          static_cast<size_t>(std::sqrt(func_info.occ_num_threads) / chunks) * chunks);
      size_t shared_mem_bytes =
          (copy_info.tile_size * (copy_info.tile_size + 1)) * copy_info.extents[0];

      num_threads = copy_info.tile_size * copy_info.tile_size;
      num_blocks =
          std::min(static_cast<unsigned int>((num_elems + num_threads - 1) / num_threads),
                   static_cast<unsigned int>(func_info.occ_num_blocks));

      void *args[] = {&copy_info};
      CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchKernel)(func_info.func, num_blocks, 1, 1,
                                                 num_threads, 1, 1, shared_mem_bytes,
                                                 stream->get_stream(), args, NULL));
    }

    void GPU::launch_indirect_copy_kernel(void *copy_info, size_t dim, size_t addr_size,
                                          size_t field_size, size_t volume,
                                          GPUStream *stream)
    {
      size_t log_addr_size = std::min(static_cast<size_t>(ctz(addr_size)),
                                      CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES - 1);
      size_t log_field_size = std::min(static_cast<size_t>(ctz(field_size)),
                                       CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES - 1);

      assert((1ULL << log_field_size) <= field_size);
      assert(dim <= CUDA_MAX_DIM);
      assert(dim >= 1);

      GPUFuncInfo &func_info =
          indirect_copy_kernels[dim - 1][log_addr_size][log_field_size];
      launch_kernel(func_info, copy_info, volume, stream);
    }

    void GPU::launch_batch_affine_fill_kernel(void *fill_info, size_t dim,
                                              size_t elem_size, size_t volume,
                                              GPUStream *stream)
    {
      size_t log_elem_size = std::min(static_cast<size_t>(ctz(elem_size)),
                                      CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES - 1);

      assert((1ULL << log_elem_size) == elem_size);
      assert(dim <= REALM_MAX_DIM);
      assert(dim >= 1);

      // TODO: probably replace this
      // with a better data-structure
      GPUFuncInfo &func_info = batch_fill_affine_kernels[dim - 1][log_elem_size];
      launch_kernel(func_info, fill_info, volume, stream);
    }

    void GPU::launch_batch_affine_kernel(void *copy_info, size_t dim,
                                         size_t elem_size, size_t volume,
                                         GPUStream *stream) {
      size_t log_elem_size = std::min(static_cast<size_t>(ctz(elem_size)),
                                      CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES - 1);

      assert((1ULL << log_elem_size) == elem_size);
      assert(dim <= REALM_MAX_DIM);
      assert(dim >= 1);

      // TODO: probably replace this
      // with a better data-structure
      GPUFuncInfo &func_info = batch_affine_kernels[dim - 1][log_elem_size];
      launch_kernel(func_info, copy_info, volume, stream);
    }

    const GPU::CudaIpcMapping *GPU::find_ipc_mapping(Memory mem) const
    {
      for(const CudaIpcMapping &mapping : cudaipc_mappings) {
        if(mapping.mem == mem) {
          return &mapping;
        }
      }

      return nullptr;
    }

    bool GPUProcessor::register_task(Processor::TaskFuncID func_id,
				     CodeDescriptor& codedesc,
				     const ByteArrayRef& user_data)
    {
      // see if we have a function pointer to register
      const FunctionPointerImplementation *fpi = codedesc.find_impl<FunctionPointerImplementation>();

      // if we don't have a function pointer implementation, see if we can make one
      if(!fpi) {
	const std::vector<CodeTranslator *>& translators = get_runtime()->get_code_translators();
	for(std::vector<CodeTranslator *>::const_iterator it = translators.begin();
	    it != translators.end();
	    it++)
	  if((*it)->can_translate<FunctionPointerImplementation>(codedesc)) {
	    FunctionPointerImplementation *newfpi = (*it)->translate<FunctionPointerImplementation>(codedesc);
	    if(newfpi) {
	      log_taskreg.info() << "function pointer created: trans=" << (*it)->name << " fnptr=" << (void *)(newfpi->fnptr);
	      codedesc.add_implementation(newfpi);
	      fpi = newfpi;
	      break;
	    }
	  }
      }

      assert(fpi != 0);

      {
	RWLock::AutoWriterLock al(task_table_mutex);

	// first, make sure we haven't seen this task id before
	if(gpu_task_table.count(func_id) > 0) {
	  log_taskreg.fatal() << "duplicate task registration: proc=" << me << " func=" << func_id;
	  return false;
	}

	GPUTaskTableEntry &tte = gpu_task_table[func_id];

	// figure out what type of function we have
	if(codedesc.type() == TypeConv::from_cpp_type<Processor::TaskFuncPtr>()) {
	  tte.fnptr = (Processor::TaskFuncPtr)(fpi->fnptr);
	  tte.stream_aware_fnptr = 0;
	} else if(codedesc.type() == TypeConv::from_cpp_type<Cuda::StreamAwareTaskFuncPtr>()) {
	  tte.fnptr = 0;
	  tte.stream_aware_fnptr = (Cuda::StreamAwareTaskFuncPtr)(fpi->fnptr);
	} else {
	  log_taskreg.fatal() << "attempt to register a task function of improper type: " << codedesc.type();
	  assert(0);
	}

	tte.user_data = user_data;
      }

      log_taskreg.info() << "task " << func_id << " registered on " << me << ": " << codedesc;

      return true;
    }

    void GPUProcessor::execute_task(Processor::TaskFuncID func_id,
				    const ByteArrayRef& task_args)
    {
      if(func_id == Processor::TASK_ID_PROCESSOR_NOP)
	return;

      std::map<Processor::TaskFuncID, GPUTaskTableEntry>::const_iterator it;
      {
	RWLock::AutoReaderLock al(task_table_mutex);
	
	it = gpu_task_table.find(func_id);
	if(it == gpu_task_table.end()) {
	  log_taskreg.fatal() << "task " << func_id << " not registered on " << me;
	  assert(0);
	}
      }

      const GPUTaskTableEntry& tte = it->second;

      if(tte.stream_aware_fnptr) {
	// shouldn't be here without a valid stream
	assert(ThreadLocal::current_gpu_stream);
	CUstream stream = ThreadLocal::current_gpu_stream->get_stream();

	log_taskreg.debug() << "task " << func_id << " executing on " << me << ": " << ((void *)(tte.stream_aware_fnptr)) << " (stream aware)";
      
	(tte.stream_aware_fnptr)(task_args.base(), task_args.size(),
				 tte.user_data.base(), tte.user_data.size(),
				 me, stream);
      } else {
	assert(tte.fnptr);
	log_taskreg.debug() << "task " << func_id << " executing on " << me << ": " << ((void *)(tte.fnptr));
      
	(tte.fnptr)(task_args.base(), task_args.size(),
		    tte.user_data.base(), tte.user_data.size(),
		    me);
      }
    }

    void GPUProcessor::shutdown(void)
    {
      log_gpu.info("shutting down");

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();

      ctxsync.shutdown_threads();

      // synchronize the device so we can flush any printf buffers - do
      //  this after shutting down the threads so that we know all work is done
      {
	AutoGPUContext agc(gpu);

	CHECK_CU( CUDA_DRIVER_FNPTR(cuCtxSynchronize)() );
      }
    }

    GPUWorker::GPUWorker(void)
      : BackgroundWorkItem("gpu worker")
      , condvar(lock)
      , core_rsrv(0), worker_thread(0)
      , thread_sleeping(false)
      , worker_shutdown_requested(false)
    {}

    GPUWorker::~GPUWorker(void)
    {
      // shutdown should have already been called
      assert(worker_thread == 0);
    }

    void GPUWorker::start_background_thread(Realm::CoreReservationSet &crs,
					    size_t stack_size)
    {
      // shouldn't be doing this if we've registered as a background work item
      assert(manager == 0);

      core_rsrv = new Realm::CoreReservation("GPU worker thread", crs,
					     Realm::CoreReservationParameters());

      Realm::ThreadLaunchParameters tlp;

      worker_thread = Realm::Thread::create_kernel_thread<GPUWorker,
							  &GPUWorker::thread_main>(this,
										   tlp,
										   *core_rsrv,
										   0);
    }

    void GPUWorker::shutdown_background_thread(void)
    {
      {
	AutoLock<> al(lock);
	worker_shutdown_requested.store(true);
	if(thread_sleeping) {
	  thread_sleeping = false;
	  condvar.broadcast();
	}
      }

      worker_thread->join();
      delete worker_thread;
      worker_thread = 0;

      delete core_rsrv;
      core_rsrv = 0;
    }

    void GPUWorker::add_stream(GPUStream *stream)
    {
      bool was_empty = false;
      {
	AutoLock<> al(lock);

#ifdef DEBUG_REALM
	// insist that the caller de-duplicate these
	for(ActiveStreamQueue::iterator it = active_streams.begin();
	    it != active_streams.end();
	    ++it)
	  assert(*it != stream);
#endif
	was_empty = active_streams.empty();
	active_streams.push_back(stream);

	if(thread_sleeping) {
	  thread_sleeping = false;
	  condvar.broadcast();
	}
      }

      // if we're a background work item, request attention if needed
      if(was_empty && (manager != 0))
	make_active();
    }

    bool GPUWorker::do_work(TimeLimit work_until)
    {
      // pop the first stream off the list and immediately become re-active
      //  if more streams remain
      GPUStream *stream = 0;
      bool still_not_empty = false;
      {
        AutoLock<> al(lock);

        assert(!active_streams.empty());
        stream = active_streams.front();
        active_streams.pop_front();
        still_not_empty = !active_streams.empty();
      }
      if(still_not_empty) {
        make_active();
      }

      // do work for the stream we popped, paying attention to the cutoff
      //  time
      bool was_empty = false;
      if(stream->reap_events(work_until)) {
        AutoLock<> al(lock);

        was_empty = active_streams.empty();
        active_streams.push_back(stream);
      }
      // note that we can need requeueing even if we called make_active above!
      return was_empty;
    }

    bool GPUWorker::process_streams(bool sleep_on_empty)
    {
      GPUStream *cur_stream = 0;
      GPUStream *first_stream = 0;
      bool requeue_stream = false;

      while(true) {
	// grab the front stream in the list
	{
	  AutoLock<> al(lock);

	  // if we didn't finish work on the stream from the previous
	  //  iteration, add it back to the end
	  if(requeue_stream)
	    active_streams.push_back(cur_stream);

	  while(active_streams.empty()) {
	    // sleep only if this was the first attempt to get a stream
	    if(sleep_on_empty && (first_stream == 0) &&
	       !worker_shutdown_requested.load()) {
	      thread_sleeping = true;
	      condvar.wait();
	    } else
	      return false;
	  }

	  cur_stream = active_streams.front();
	  // did we wrap around?  if so, stop for now
	  if(cur_stream == first_stream)
	    return true;

	  active_streams.pop_front();
	  if(!first_stream)
	    first_stream = cur_stream;
	}

	// and do some work for it
	requeue_stream = false;

        //  reap_events report whether any kind of work
        //  remains, so we have to be careful to avoid double-requeueing -
        //  if the first call returns false, we can't try the second one
        //  because we may be doing (or failing to do and then requeuing)
        //  somebody else's work
        if(!cur_stream->reap_events(TimeLimit()))
          continue;

        // if we fall all the way through, the queues never went empty at
        //  any time, so it's up to us to requeue
        requeue_stream = true;
      }
    }

    void GPUWorker::thread_main(void)
    {
      // TODO: consider busy-waiting in some cases to reduce latency?
      while(!worker_shutdown_requested.load()) {
	bool work_left = process_streams(true);

	// if there was work left, yield our thread for now to avoid a tight spin loop
	// TODO: enqueue a callback so we can go to sleep and wake up sooner than a kernel
	//  timeslice?
	if(work_left)
	  Realm::Thread::yield();
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class BlockingCompletionNotification

    class BlockingCompletionNotification : public GPUCompletionNotification {
    public:
      BlockingCompletionNotification(void);
      virtual ~BlockingCompletionNotification(void);

      virtual void request_completed(void);

      virtual void wait(void);

    public:
      atomic<bool> completed;
    };

    BlockingCompletionNotification::BlockingCompletionNotification(void)
      : completed(false)
    {}

    BlockingCompletionNotification::~BlockingCompletionNotification(void)
    {}

    void BlockingCompletionNotification::request_completed(void)
    {
      // no condition variable needed - the waiter is spinning
      completed.store(true);
    }

    void BlockingCompletionNotification::wait(void)
    {
      // blocking completion is horrible and should die as soon as possible
      // in the mean time, we need to assist with background work to avoid
      //  the risk of deadlock
      // note that this means you can get NESTED blocking completion
      //  notifications, which is just one of the ways this is horrible
      BackgroundWorkManager::Worker worker;

      worker.set_manager(&(get_runtime()->bgwork));

      while(!completed.load())
	worker.do_work(-1 /* as long as it takes */,
		       &completed /* until this is set */);
    }
	

    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUFBMemory

    GPUFBMemory::GPUFBMemory(Memory _me, GPU *_gpu, CUdeviceptr _base, size_t _size)
      : LocalManagedMemory(_me, _size, MKIND_GPUFB, 512, Memory::GPU_FB_MEM, 0)
      , gpu(_gpu)
      , base(_base)
    {
      // mark what context we belong to
      add_module_specific(new CudaDeviceMemoryInfo(gpu->context));

      // advertise for potential gpudirect support
      local_segment.assign(NetworkSegmentInfo::CudaDeviceMem,
			   reinterpret_cast<void *>(base), size,
			   reinterpret_cast<uintptr_t>(gpu));
      segment = &local_segment;
    }

    GPUFBMemory::~GPUFBMemory(void) {}

    // these work, but they are SLOW
    void GPUFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyDtoH)
                  (dst, reinterpret_cast<CUdeviceptr>(base + offset), size) );
      }
    }

    void GPUFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyHtoD)
                  (reinterpret_cast<CUdeviceptr>(base + offset), src, size) );
      }
    }

    void *GPUFBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (void *)(base + offset);
    }

    // GPUFBMemory supports ExternalCudaMemoryResource and
    //  ExternalCudaArrayResource
    bool GPUFBMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                         size_t& inst_offset)
    {
      {
        ExternalCudaMemoryResource *res = dynamic_cast<ExternalCudaMemoryResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = res->base - base; // offset relative to our base
          return true;
        }
      }

      {
        ExternalCudaArrayResource *res = dynamic_cast<ExternalCudaArrayResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = 0;
          CUarray array = reinterpret_cast<CUarray>(res->array);
          inst->metadata.add_mem_specific(new MemSpecificCudaArray(array));
          return true;
        }
      }

      // not a kind we recognize
      return false;
    }

    void GPUFBMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // TODO: clean up surface/texture objects
      MemSpecificCudaArray *ms = inst->metadata.find_mem_specific<MemSpecificCudaArray>();
      if(ms) {
        ms->array = 0;
      }
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUFBMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                  const IndexSpaceGeneric *subspace,
                                                                  span<const FieldID> fields,
                                                                  bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      uintptr_t abs_base = (this->base + inst->metadata.inst_offset + rel_base);

      return new ExternalCudaMemoryResource(gpu->info->index,
                                            abs_base, extent, read_only);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUDynamicMemory

    GPUDynamicFBMemory::GPUDynamicFBMemory(Memory _me, GPU *_gpu,
                                           size_t _max_size)
      : MemoryImpl(_me, _max_size, MKIND_GPUFB, Memory::GPU_DYNAMIC_MEM, 0)
      , gpu(_gpu)
      , cur_size(0)
    {
      // mark what context we belong to
      add_module_specific(new CudaDeviceMemoryInfo(gpu->context));

      // advertise for potential (on-demand) gpudirect support
      local_segment.assign(NetworkSegmentInfo::CudaDeviceMem, 0 /*base*/, 0 /*size*/,
                           reinterpret_cast<uintptr_t>(gpu),
                           NetworkSegmentInfo::OptionFlags::OnDemandRegistration);
      segment = &local_segment;
    }

    GPUDynamicFBMemory::~GPUDynamicFBMemory(void)
    {
      cleanup();
    }

    void GPUDynamicFBMemory::cleanup(void)
    {
      AutoLock<> al(mutex);
      if(alloc_bases.empty())
        return;
      // free any remaining allocations
      AutoGPUContext agc(gpu);
      for(std::map<RegionInstance, std::pair<CUdeviceptr, size_t>>::const_iterator it =
              alloc_bases.begin();
          it != alloc_bases.end(); ++it)
        if(it->second.first)
          CHECK_CU(CUDA_DRIVER_FNPTR(cuMemFree)(it->second.first));
      alloc_bases.clear();
    }

    MemoryImpl::AllocationResult GPUDynamicFBMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
                                                                                bool need_alloc_result,
                                                                                bool poisoned,
                                                                                TimeLimit work_until)
    {
      // poisoned allocations are cancellled
      if(poisoned) {
        inst->notify_allocation(ALLOC_CANCELLED,
                                RegionInstanceImpl::INSTOFFSET_FAILED,
                                work_until);
        return ALLOC_CANCELLED;
      }

      // attempt cuMemAlloc, except for bytes=0 allocations
      size_t bytes = inst->metadata.layout->bytes_used;
      CUdeviceptr base = 0;
      if(bytes > 0) {
        // before we attempt an allocation with cuda, make sure we're not
        //  going over our usage limit
        bool limit_ok;
        size_t cur_snapshot;
        {
          AutoLock<> al(mutex);
          cur_snapshot = cur_size;
          if((cur_size + bytes) <= size) {
            cur_size += bytes;
            limit_ok = true;
          } else
            limit_ok = false;
        }

        if(!limit_ok) {
          log_gpu.warning() << "dynamic allocation limit reached: mem=" << me
                            << " cur_size=" << cur_snapshot
                            << " bytes=" << bytes << " limit=" << size;
          inst->notify_allocation(ALLOC_INSTANT_FAILURE,
                                  RegionInstanceImpl::INSTOFFSET_FAILED,
                                  work_until);
          return ALLOC_INSTANT_FAILURE;
        }

        CUresult ret = CUDA_SUCCESS;
        {
          AutoGPUContext agc(gpu);
          // TODO: handle large alignments?
          ret = CUDA_DRIVER_FNPTR(cuMemAlloc)(&base, bytes);
          if((ret != CUDA_SUCCESS) && (ret != CUDA_ERROR_OUT_OF_MEMORY)) {
            REPORT_CU_ERROR(Logger::LEVEL_ERROR, "cuMemAlloc", ret);
            abort();
          }
        }
        if(ret == CUDA_ERROR_OUT_OF_MEMORY) {
          log_gpu.warning() << "out of memory in cuMemAlloc: bytes=" << bytes;
          inst->notify_allocation(ALLOC_INSTANT_FAILURE,
                                  RegionInstanceImpl::INSTOFFSET_FAILED,
                                  work_until);
          return ALLOC_INSTANT_FAILURE;
        }
      }

      // insert entry into our alloc_bases map
      {
        AutoLock<> al(mutex);
        alloc_bases[inst->me] = std::make_pair(base, bytes);
      }

      inst->notify_allocation(ALLOC_INSTANT_SUCCESS, base, work_until);
      return ALLOC_INSTANT_SUCCESS;
    }

    void GPUDynamicFBMemory::release_storage_immediate(RegionInstanceImpl *inst,
                                                       bool poisoned,
                                                       TimeLimit work_until)
    {
      // ignore poisoned releases
      if(poisoned)
        return;

      // for external instances, all we have to do is ack the destruction
      if(inst->metadata.ext_resource != 0) {
        unregister_external_resource(inst);
        inst->notify_deallocation();
	return;
      }

      CUdeviceptr base;
      {
        AutoLock<> al(mutex);
        std::map<RegionInstance, std::pair<CUdeviceptr, size_t> >::iterator it = alloc_bases.find(inst->me);
        if(it == alloc_bases.end()) {
          log_gpu.fatal() << "attempt to release unknown instance: inst=" << inst->me;
          abort();
        }
        base = it->second.first;
        assert(cur_size >= it->second.second);
        cur_size -= it->second.second;
        alloc_bases.erase(it);
      }

      if(base != 0) {
        AutoGPUContext agc(gpu);
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemFree)(base) );
      }

      inst->notify_deallocation();
    }

    // these work, but they are SLOW
    void GPUDynamicFBMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyDtoH)
                  (dst, CUdeviceptr(offset), size) );
      }
    }

    void GPUDynamicFBMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        AutoGPUContext agc(gpu);
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyHtoD)
                  (CUdeviceptr(offset), src, size) );
      }
    }

    void *GPUDynamicFBMemory::get_direct_ptr(off_t offset, size_t size)
    {
      // offset 'is' the pointer for instances in this memory
      return reinterpret_cast<void *>(offset);
    }

    // GPUFBMemory supports ExternalCudaMemoryResource and
    //  ExternalCudaArrayResource
    bool GPUDynamicFBMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                                size_t& inst_offset)
    {
      {
        ExternalCudaMemoryResource *res = dynamic_cast<ExternalCudaMemoryResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = res->base; // "offsets" are absolute in dynamic fbmem
          return true;
        }
      }

      {
        ExternalCudaArrayResource *res = dynamic_cast<ExternalCudaArrayResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success
          inst_offset = 0;
          CUarray array = reinterpret_cast<CUarray>(res->array);
          inst->metadata.add_mem_specific(new MemSpecificCudaArray(array));
          return true;
        }
      }

      // not a kind we recognize
      return false;
    }

    void GPUDynamicFBMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // TODO: clean up surface/texture objects
      MemSpecificCudaArray *ms = inst->metadata.find_mem_specific<MemSpecificCudaArray>();
      if(ms) {
        ms->array = 0;
      }
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUDynamicFBMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                         const IndexSpaceGeneric *subspace,
                                                                         span<const FieldID> fields,
                                                                         bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      uintptr_t abs_base = (inst->metadata.inst_offset + rel_base);

      return new ExternalCudaMemoryResource(gpu->info->index,
                                            abs_base, extent, read_only);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUZCMemory

    GPUZCMemory::GPUZCMemory(GPU *gpu, Memory _me, CUdeviceptr _gpu_base, void *_cpu_base,
                             size_t _size, MemoryKind _kind, Memory::Kind _lowlevel_kind)
      : LocalManagedMemory(_me, _size, _kind, 256, _lowlevel_kind, 0)
      , gpu_base(_gpu_base)
      , cpu_base((char *)_cpu_base)
    {
      add_module_specific(new CudaDeviceMemoryInfo(gpu->context));
      // advertise ourselves as a host or managed memory, as appropriate
      NetworkSegmentInfo::MemoryType mtype;
      if(_kind == MemoryImpl::MKIND_MANAGED)
        mtype = NetworkSegmentInfo::CudaManagedMem;
      else
        mtype = NetworkSegmentInfo::HostMem;
      local_segment.assign(mtype, cpu_base, size);
      segment = &local_segment;
    }

    GPUZCMemory::~GPUZCMemory(void) {}

    void GPUZCMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      memcpy(dst, cpu_base+offset, size);
    }

    void GPUZCMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      memcpy(cpu_base+offset, src, size);
    }

    void *GPUZCMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (cpu_base + offset);
    }

    // GPUZCMemory supports ExternalCudaPinnedHostResource
    bool GPUZCMemory::attempt_register_external_resource(RegionInstanceImpl *inst,
                                                         size_t& inst_offset)
    {
      {
        ExternalCudaPinnedHostResource *res = dynamic_cast<ExternalCudaPinnedHostResource *>(inst->metadata.ext_resource);
        if(res) {
          // automatic success - offset relative to our base
          inst_offset = res->base - reinterpret_cast<uintptr_t>(cpu_base);
          return true;
        }
      }

      // not a kind we recognize
      return false;
    }

    void GPUZCMemory::unregister_external_resource(RegionInstanceImpl *inst)
    {
      // nothing actually to clean up
    }

    // for re-registration purposes, generate an ExternalInstanceResource *
    //  (if possible) for a given instance, or a subset of one
    ExternalInstanceResource *GPUZCMemory::generate_resource_info(RegionInstanceImpl *inst,
                                                                  const IndexSpaceGeneric *subspace,
                                                                  span<const FieldID> fields,
                                                                  bool read_only)
    {
      // compute the bounds of the instance relative to our base
      assert(inst->metadata.is_valid() &&
             "instance metadata must be valid before accesses are performed");
      assert(inst->metadata.layout);
      InstanceLayoutGeneric *ilg = inst->metadata.layout;
      uintptr_t rel_base, extent;
      if(subspace == 0) {
        // want full instance
        rel_base = 0;
        extent = ilg->bytes_used;
      } else {
        assert(!fields.empty());
        uintptr_t limit;
        for(size_t i = 0; i < fields.size(); i++) {
          uintptr_t f_base, f_limit;
          if(!subspace->impl->compute_affine_bounds(ilg, fields[i], f_base, f_limit))
            return 0;
          if(i == 0) {
            rel_base = f_base;
            limit = f_limit;
          } else {
            rel_base = std::min(rel_base, f_base);
            limit = std::max(limit, f_limit);
          }
        }
        extent = limit - rel_base;
      }

      void *mem_base = (this->cpu_base +
                        inst->metadata.inst_offset +
                        rel_base);

      return new ExternalCudaPinnedHostResource(reinterpret_cast<uintptr_t>(mem_base),
                                                extent, read_only);
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPUFBIBMemory

    GPUFBIBMemory::GPUFBIBMemory(Memory _me, GPU *_gpu,
                                 CUdeviceptr _base, size_t _size)
      : IBMemory(_me, _size, MKIND_GPUFB, Memory::GPU_FB_MEM,
                 reinterpret_cast<void *>(_base), 0)
      , gpu(_gpu)
      , base(_base)
    {
      add_module_specific(new CudaDeviceMemoryInfo(gpu->context));
      // advertise for potential gpudirect support
      local_segment.assign(NetworkSegmentInfo::CudaDeviceMem,
			   reinterpret_cast<void *>(_base), _size,
			   reinterpret_cast<uintptr_t>(_gpu));
      segment = &local_segment;
    }


    // Helper methods for emulating the cuda runtime
    /*static*/ GPUProcessor* GPUProcessor::get_current_gpu_proc(void)
    {
      return ThreadLocal::current_gpu_proc;
    }

#ifdef REALM_USE_CUDART_HIJACK
    void GPUProcessor::push_call_configuration(dim3 grid_dim, dim3 block_dim,
                                               size_t shared_size, void *stream)
    {
      call_configs.push_back(CallConfig(grid_dim, block_dim,
                                        shared_size, (CUstream)stream));
    }

    void GPUProcessor::pop_call_configuration(dim3 *grid_dim, dim3 *block_dim,
                                              size_t *shared_size, void *stream)
    {
      assert(!call_configs.empty());
      const CallConfig &config = call_configs.back();
      *grid_dim = config.grid;
      *block_dim = config.block;
      *shared_size = config.shared;
      *((CUstream*)stream) = config.stream;
      call_configs.pop_back();
    }
#endif

    void GPUProcessor::stream_wait_on_event(CUstream stream, CUevent event)
    {
      if(IS_DEFAULT_STREAM(stream))
        CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(
              ThreadLocal::current_gpu_stream->get_stream(), event, 0) );
      else
        CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(stream, event, 0) );
    }

    void GPUProcessor::stream_synchronize(CUstream stream)
    {
      // same as device_synchronize if stream is zero
      if(!IS_DEFAULT_STREAM(stream))
      {
        if(!block_on_synchronize) {
          GPUStream *s = gpu->find_stream(stream);
          if(s) {
            // We don't actually want to block the GPU processor
            // when synchronizing, so we instead register a cuda
            // event on the stream and then use it triggering to
            // indicate that the stream is caught up
            // Make a completion notification to be notified when
            // the event has actually triggered
            GPUPreemptionWaiter waiter(gpu);
            // Register the waiter with the stream 
            s->add_notification(&waiter); 
            // Perform the wait, this will preempt the thread
            waiter.preempt();
          } else {
            log_gpu.warning() << "WARNING: Detected unknown CUDA stream "
              << stream << " that Realm did not create which suggests "
              << "that there is another copy of the CUDA runtime "
              << "somewhere making its own streams... be VERY careful.";
            CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamSynchronize)(stream) );
          }
        } else {
          // oh well...
          CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamSynchronize)(stream) );
        }
      }
      else
        device_synchronize();
    }

    GPUPreemptionWaiter::GPUPreemptionWaiter(GPU *g) : gpu(g)
    {
      GenEventImpl *impl = GenEventImpl::create_genevent();
      wait_event = impl->current_event();
    }

    void GPUPreemptionWaiter::request_completed(void)
    {
      GenEventImpl::trigger(wait_event, false/*poisoned*/);
    }

    void GPUPreemptionWaiter::preempt(void)
    {
      // Realm threads don't obey a stack discipline for
      // preemption so we can't leave our context on the stack
      gpu->pop_context();
      wait_event.wait();
      // When we wake back up, we have to push our context again
      gpu->push_context();
    }

    void GPUProcessor::device_synchronize(void)
    {
      GPUStream *current = ThreadLocal::current_gpu_stream;

      if(ThreadLocal::created_gpu_streams)
      {
        current->wait_on_streams(*ThreadLocal::created_gpu_streams); 
        delete ThreadLocal::created_gpu_streams;
        ThreadLocal::created_gpu_streams = 0;
      }

      if(!block_on_synchronize) {
	// We don't actually want to block the GPU processor
	// when synchronizing, so we instead register a cuda
	// event on the stream and then use it triggering to
	// indicate that the stream is caught up
	// Make a completion notification to be notified when
	// the event has actually triggered
	GPUPreemptionWaiter waiter(gpu);
	// Register the waiter with the stream 
	current->add_notification(&waiter); 
	// Perform the wait, this will preempt the thread
	waiter.preempt();
      } else {
	// oh well...
	CHECK_CU( CUDA_DRIVER_FNPTR(cuStreamSynchronize)(current->get_stream()) );
      }
    }
    
#ifdef REALM_USE_CUDART_HIJACK
    void GPUProcessor::event_record(CUevent event, CUstream stream)
    {
      CUevent e = event;
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_CU( CUDA_DRIVER_FNPTR(cuEventRecord)(e, stream) );
    }
      
    GPUProcessor::LaunchConfig::LaunchConfig(dim3 _grid, dim3 _block, size_t _shared)
      : grid(_grid), block(_block), shared(_shared)
    {}

    GPUProcessor::CallConfig::CallConfig(dim3 _grid, dim3 _block, 
                                         size_t _shared, CUstream _stream)
      : LaunchConfig(_grid, _block, _shared), stream(_stream)
    {}

    void GPUProcessor::configure_call(dim3 grid_dim,
				      dim3 block_dim,
				      size_t shared_mem,
				      CUstream stream)
    {
      launch_configs.push_back(CallConfig(grid_dim, block_dim, shared_mem, stream));
    }

    void GPUProcessor::setup_argument(const void *arg,
				      size_t size, size_t offset)
    {
      size_t required = offset + size;

      if(required > kernel_args.size())
	kernel_args.resize(required);

      memcpy(&kernel_args[offset], arg, size);
    }

    void GPUProcessor::launch(const void *func)
    {
      // make sure we have a launch config
      assert(!launch_configs.empty());
      CallConfig &config = launch_configs.back();

      // Find our function
      CUfunction f = gpu->lookup_function(func);

      size_t arg_size = kernel_args.size();
      void *extra[] = { 
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kernel_args[0],
        CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        CU_LAUNCH_PARAM_END
      };

      if(IS_DEFAULT_STREAM(config.stream))
        config.stream = ThreadLocal::current_gpu_stream->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << config.stream;

      // Launch the kernel on our stream dammit!
      CHECK_CU( cuLaunchKernel(f, 
			       config.grid.x, config.grid.y, config.grid.z,
                               config.block.x, config.block.y, config.block.z,
                               config.shared,
                               config.stream,
			       NULL, extra) );

      // pop the config we just used
      launch_configs.pop_back();

      // clear out the kernel args
      kernel_args.clear();
    }

    void GPUProcessor::launch_kernel(const void *func,
                                     dim3 grid_dim,
                                     dim3 block_dim,
                                     void **args,
                                     size_t shared_memory,
                                     CUstream stream,
                                     bool cooperative /*=false*/)
    {
      // Find our function
      CUfunction f = gpu->lookup_function(func);

      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      log_stream.debug() << "kernel " << func << " added to stream " << stream;

      if (cooperative) {
#if CUDA_VERSION >= 9000
        CHECK_CU( cuLaunchCooperativeKernel(f,
                                 grid_dim.x, grid_dim.y, grid_dim.z,
                                 block_dim.x, block_dim.y, block_dim.z,
                                 shared_memory,
                                 stream,
                                 args) );
#else
	log_gpu.fatal() << "attempt to launch cooperative kernel on CUDA < 9.0!";
	abort();
#endif
      } else {
        CHECK_CU( cuLaunchKernel(f,
                                 grid_dim.x, grid_dim.y, grid_dim.z,
                                 block_dim.x, block_dim.y, block_dim.z,
                                 shared_memory,
                                 stream,
                                 args, NULL) );
      }
    }
#endif

    void GPUProcessor::gpu_memcpy(void *dst, const void *src, size_t size)
    {
      CUstream current = ThreadLocal::current_gpu_stream->get_stream();
      // the synchronous copy still uses cuMemcpyAsync so that we can limit the
      //  synchronization to just the right stream
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                ((CUdeviceptr)dst, (CUdeviceptr)src, size, current) );
      stream_synchronize(current);
    }

    void GPUProcessor::gpu_memcpy_async(void *dst, const void *src, size_t size, CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                ((CUdeviceptr)dst, (CUdeviceptr)src, size, stream) );
      // no synchronization here
    }

#ifdef REALM_USE_CUDART_HIJACK
    void GPUProcessor::gpu_memcpy2d(void *dst, size_t dpitch, const void *src, 
                                    size_t spitch, size_t width, size_t height)
    {
      CUstream current = ThreadLocal::current_gpu_stream->get_stream();
      CUDA_MEMCPY2D copy_info;
      copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
      copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
      copy_info.srcDevice = (CUdeviceptr)src;
      copy_info.srcHost = src;
      copy_info.srcPitch = spitch;
      copy_info.srcY = 0;
      copy_info.srcXInBytes = 0;
      copy_info.dstDevice = (CUdeviceptr)dst;
      copy_info.dstHost = dst;
      copy_info.dstPitch = dpitch;
      copy_info.dstY = 0;
      copy_info.dstXInBytes = 0;
      copy_info.WidthInBytes = width;
      copy_info.Height = height;
      // the synchronous copy still uses cuMemcpyAsync so that we can limit the
      //  synchronization to just the right stream
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)
                (&copy_info, current) );
      stream_synchronize(current);
    }

    void GPUProcessor::gpu_memcpy2d_async(void *dst, size_t dpitch, const void *src, 
                                          size_t spitch, size_t width, size_t height, 
                                          CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CUDA_MEMCPY2D copy_info;
      copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
      copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
      copy_info.srcDevice = (CUdeviceptr)src;
      copy_info.srcHost = src;
      copy_info.srcPitch = spitch;
      copy_info.srcY = 0;
      copy_info.srcXInBytes = 0;
      copy_info.dstDevice = (CUdeviceptr)dst;
      copy_info.dstHost = dst;
      copy_info.dstPitch = dpitch;
      copy_info.dstY = 0;
      copy_info.dstXInBytes = 0;
      copy_info.WidthInBytes = width;
      copy_info.Height = height;
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)
                (&copy_info, stream) );
      // no synchronization here
    }

    void GPUProcessor::gpu_memcpy_to_symbol(const void *dst, const void *src,
					    size_t size, size_t offset)
    {
      CUstream current = ThreadLocal::current_gpu_stream->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(dst);
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                (var_base + offset,
                 (CUdeviceptr)src, size, current) );
      stream_synchronize(current);
    }

    void GPUProcessor::gpu_memcpy_to_symbol_async(const void *dst, const void *src,
						  size_t size, size_t offset, CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(dst);
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                (var_base + offset,
                 (CUdeviceptr)src, size, stream) );
      // no synchronization here
    }

    void GPUProcessor::gpu_memcpy_from_symbol(void *dst, const void *src,
					      size_t size, size_t offset)
    {
      CUstream current = ThreadLocal::current_gpu_stream->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(src);
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                ((CUdeviceptr)dst,
                 var_base + offset,
                 size, current) );
      stream_synchronize(current);
    }

    void GPUProcessor::gpu_memcpy_from_symbol_async(void *dst, const void *src,
						    size_t size, size_t offset,
						    CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CUdeviceptr var_base = gpu->lookup_variable(src);
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemcpyAsync)
                ((CUdeviceptr)dst,
                 var_base + offset,
                 size, stream) );
      // no synchronization here
    }
#endif

    void GPUProcessor::gpu_memset(void *dst, int value, size_t count)
    {
      CUstream current = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD8Async)
                ((CUdeviceptr)dst, (unsigned char)value,
                 count, current) );
    }

    void GPUProcessor::gpu_memset_async(void *dst, int value, 
                                        size_t count, CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream))
        stream = ThreadLocal::current_gpu_stream->get_stream();
      CHECK_CU( CUDA_DRIVER_FNPTR(cuMemsetD8Async)
                ((CUdeviceptr)dst, (unsigned char)value,
                 count, stream) );
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class GPU

    GPU::GPU(CudaModule *_module, GPUInfo *_info, GPUWorker *_worker, CUcontext _context)
      : module(_module)
      , info(_info)
      , worker(_worker)
      , context(_context)
    {
      push_context();

      CHECK_CU( CUDA_DRIVER_FNPTR(cuCtxGetStreamPriorityRange)
                (&least_stream_priority, &greatest_stream_priority) );

      event_pool.init_pool();

      host_to_device_stream = new GPUStream(this, worker);
      device_to_host_stream = new GPUStream(this, worker);

      CUdevice dev;
      int numSMs;

      CHECK_CU(CUDA_DRIVER_FNPTR(cuCtxGetDevice)(&dev));
      CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
          &numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

      CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleLoadDataEx)(
          &device_module, realm_fatbin, 0, NULL, NULL));
      for(unsigned int log_bit_sz = 0; log_bit_sz < CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES;
          log_bit_sz++) {
        const unsigned int bit_sz = 8U << log_bit_sz;
        GPUFuncInfo func_info;
        char name[30];
        std::snprintf(name, sizeof(name), "memcpy_transpose%u", bit_sz);
        CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&func_info.func,
                                                        device_module, name));

        auto blocksize_to_sharedmem = [](int block_size) -> size_t {
          int tile_size = sqrt(block_size);
          return static_cast<size_t>(tile_size * (tile_size + 1) * CUDA_MAX_FIELD_BYTES);
        };

        CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxPotentialBlockSize)(
            &func_info.occ_num_blocks, &func_info.occ_num_threads, func_info.func,
            blocksize_to_sharedmem, 0, 0));

        // func_info.occ_num_blocks *=
        //  numSMs; // Fill up the GPU with the number of blocks if possible
        transpose_kernels[log_bit_sz] = func_info;

        for(unsigned int d = 1; d <= CUDA_MAX_DIM; d++) {
          std::snprintf(name, sizeof(name), "memcpy_affine_batch%uD_%u", d, bit_sz);
          CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&func_info.func, device_module,
                                                          name));
          // Here, we don't have a constraint on the block size, so allow
          // the driver to decide the best combination we can launch
          CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxPotentialBlockSize)(
              &func_info.occ_num_blocks, &func_info.occ_num_threads, func_info.func, 0, 0,
              0));
          batch_affine_kernels[d - 1][log_bit_sz] = func_info;

          std::snprintf(name, sizeof(name), "fill_affine_large%uD_%u", d, bit_sz);
          CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&func_info.func, device_module,
                                                          name));
          // Here, we don't have a constraint on the block size, so allow
          // the driver to decide the best combination we can launch
          CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxPotentialBlockSize)(
              &func_info.occ_num_blocks, &func_info.occ_num_threads, func_info.func, 0, 0,
              0));
          fill_affine_large_kernels[d - 1][log_bit_sz] = func_info;

          std::snprintf(name, sizeof(name), "fill_affine_batch%uD_%u", d, bit_sz);
          CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&func_info.func, device_module,
                                                          name));
          // Here, we don't have a constraint on the block size, so allow
          // the driver to decide the best combination we can launch
          CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxPotentialBlockSize)(
              &func_info.occ_num_blocks, &func_info.occ_num_threads, func_info.func, 0, 0,
              0));
          batch_fill_affine_kernels[d - 1][log_bit_sz] = func_info;

          for(unsigned int log_addr_bit_sz = 2; log_addr_bit_sz < 4; log_addr_bit_sz++) {
            const unsigned int addr_bit_sz = 8U << log_addr_bit_sz;
            std::snprintf(name, sizeof(name), "memcpy_indirect%uD_%u%u", d, bit_sz,
                          addr_bit_sz);

            CHECK_CU(CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&func_info.func,
                                                            device_module, name));

            CHECK_CU(CUDA_DRIVER_FNPTR(cuOccupancyMaxPotentialBlockSize)(
                &func_info.occ_num_blocks, &func_info.occ_num_threads, func_info.func, 0,
                0, 0));

            indirect_copy_kernels[d - 1][log_addr_bit_sz][log_bit_sz] = func_info;
          }
        }
      }

      device_to_device_streams.resize(module->config->cfg_d2d_streams, 0);
      for(unsigned i = 0; i < module->config->cfg_d2d_streams; i++) {
        device_to_device_streams[i] =
            new GPUStream(this, worker, module->config->cfg_d2d_stream_priority);
      }

      // only create p2p streams for devices we can talk to
      peer_to_peer_streams.resize(module->gpu_info.size(), 0);
      for(std::vector<GPUInfo *>::const_iterator it = module->gpu_info.begin();
	  it != module->gpu_info.end();
	  ++it)
	if(info->peers.count((*it)->index) != 0)
	  peer_to_peer_streams[(*it)->index] = new GPUStream(this, worker);

      task_streams.resize(module->config->cfg_task_streams);
      for(unsigned i = 0; i < module->config->cfg_task_streams; i++)
	task_streams[i] = new GPUStream(this, worker);

      pop_context();

#ifdef REALM_USE_CUDART_HIJACK
      // now hook into the cuda runtime fatbin/etc. registration path
      GlobalRegistrations::add_gpu_context(this);
#endif
    }

    GPU::~GPU(void)
    {
      push_context();

      // Free up all the allocations for this GPU
      allocations.clear();

      event_pool.empty_pool();

      // destroy streams
      delete host_to_device_stream;
      delete device_to_host_stream;

      delete_container_contents(device_to_device_streams);

      for(std::vector<GPUStream *>::iterator it = peer_to_peer_streams.begin();
	  it != peer_to_peer_streams.end();
	  ++it)
	if(*it)
	  delete *it;

      for(std::map<NodeID, GPUStream *>::iterator it = cudaipc_streams.begin();
          it != cudaipc_streams.end();
	  ++it)
        delete it->second;

      delete_container_contents(task_streams);

      if (fb_dmem) {
        fb_dmem->cleanup();
      }

      CHECK_CU( CUDA_DRIVER_FNPTR(cuDevicePrimaryCtxRelease)(info->device) );
    }

    void GPU::push_context(void)
    {
      CHECK_CU( CUDA_DRIVER_FNPTR(cuCtxPushCurrent)(context) );
    }

    void GPU::pop_context(void)
    {
      // the context we pop had better be ours...
      CUcontext popped;
      CHECK_CU( CUDA_DRIVER_FNPTR(cuCtxPopCurrent)(&popped) );
      assert(popped == context);
    }

    GPUAllocation &GPU::add_allocation(GPUAllocation &&alloc)
    {
      AutoLock<> al(alloc_mutex);
      assert(((!!alloc) && (alloc.get_dptr() != 0)) && "Given allocation is not valid!");
      return allocations.emplace(std::make_pair(alloc.get_dptr(), std::move(alloc)))
          .first->second;
    }

    void GPU::create_processor(RuntimeImpl *runtime, size_t stack_size)
    {
      Processor p = runtime->next_local_processor_id();
      proc = new GPUProcessor(this, p, runtime->core_reservation_set(), stack_size);
      runtime->add_processor(proc);

      // this processor is able to access its own FB and the ZC mem (if any)
      if(fbmem) {
        Machine::ProcessorMemoryAffinity pma;
        pma.p = p;
        pma.m = fbmem->me;
        pma.bandwidth = info->logical_peer_bandwidth[info->index];
        pma.latency   = info->logical_peer_latency[info->index];
        runtime->add_proc_mem_affinity(pma);
      }

      for(std::set<Memory>::const_iterator it = pinned_sysmems.begin();
          it != pinned_sysmems.end(); ++it) {
        // no processor affinity to IB memories
        if(!ID(*it).is_memory())
          continue;

        Machine::ProcessorMemoryAffinity pma;
        pma.p = p;
        pma.m = *it;
        pma.bandwidth = info->pci_bandwidth;
        pma.latency = 200; // "bad"
        runtime->add_proc_mem_affinity(pma);
      }

      for(std::set<Memory>::const_iterator it = managed_mems.begin();
          it != managed_mems.end(); ++it) {
        // no processor affinity to IB memories
        if(!ID(*it).is_memory())
          continue;

        Machine::ProcessorMemoryAffinity pma;
        pma.p = p;
        pma.m = *it;
        pma.bandwidth = info->pci_bandwidth; // Not quite correct, but be pessimistic here
        pma.latency = 300;                   // "worse" (pessimistically assume faults)
        runtime->add_proc_mem_affinity(pma);
      }

      // peer access
      for(size_t i = 0; i < module->gpus.size(); i++) {
        GPU *peer_gpu = module->gpus[i];
        // ignore ourselves
        if(peer_gpu == this)
          continue;

        // ignore gpus that we don't expect to be able to peer with
        if(info->peers.count(peer_gpu->info->index) == 0)
          continue;

        // ignore gpus with no fb
        if(peer_gpu->fbmem == nullptr)
          continue;

        // enable peer access (it's ok if it's already been enabled)
        //  (don't try if it's the same physical device underneath)
        if(info != peer_gpu->info) {
          AutoGPUContext agc(this);

          CUresult ret = CUDA_DRIVER_FNPTR(cuCtxEnablePeerAccess)(peer_gpu->context, 0);
          if((ret != CUDA_SUCCESS) && (ret != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)) {
            REPORT_CU_ERROR(Logger::LEVEL_ERROR,
                            "cuCtxEnablePeerAccess(peer_gpu->context, 0)", ret);
            abort();
          }
        }
        log_gpu.info() << "peer access enabled from GPU " << p << " to FB "
                       << peer_gpu->fbmem->me;
        peer_fbs.insert(peer_gpu->fbmem->me);
        
        {
          Machine::ProcessorMemoryAffinity pma;
          pma.p = p;
          pma.m = peer_gpu->fbmem->me;
          pma.bandwidth = info->logical_peer_bandwidth[i];
          pma.latency = info->logical_peer_latency[i];
          runtime->add_proc_mem_affinity(pma);
        }

        if(peer_gpu->fb_ibmem != nullptr) {
          // Don't add fb_ibmem to affinity topology as this is an internal
          // memory
          peer_fbs.insert(peer_gpu->fb_ibmem->me);
        }
      }

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        CudaDeviceMemoryInfo *cdm = (*it)->find_module_specific<CudaDeviceMemoryInfo>();
        if(!cdm)
          continue;
        if(cdm->gpu && (info->peers.count(cdm->gpu->info->index) > 0)) {
          Machine::ProcessorMemoryAffinity pma;
          pma.p = p;
          pma.m = (*it)->me;
          pma.bandwidth = info->logical_peer_bandwidth[cdm->gpu->info->index];
          pma.latency = info->logical_peer_latency[cdm->gpu->info->index];

          runtime->add_proc_mem_affinity(pma);
        }
      }
    }

    static CUdeviceptr allocate_device_memory(GPU *gpu, size_t size)
    {
      GPUAllocation *alloc = nullptr;
      // The total fb size requested to allocate
      // TODO: consider padding these sizes to 2MiB alignment
#if CUDA_VERSION >= 11050
      int mmap_supported = 0, mmap_supports_rdma = 0, rdma_supported = 0;

      CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
      (&mmap_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
       gpu->info->device);
      CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
      (&rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, gpu->info->device);
      CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
      (&mmap_supports_rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
       gpu->info->device);

      // To prevent bit-rot, and because there's no advantage to not using the
      // cuMemMap APIs, use them by default unless we need a feature they
      // don't support.
      if((!gpu->module->config->cfg_use_cuda_ipc || gpu->info->fabric_supported) &&
         mmap_supported && !(rdma_supported && !mmap_supports_rdma)) {
        CUmemAllocationProp mem_prop;
        memset(&mem_prop, 0, sizeof(mem_prop));
        mem_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        mem_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        mem_prop.location.id = gpu->info->index;
        mem_prop.win32HandleMetaData = GPUAllocation::get_win32_shared_attributes();
        mem_prop.allocFlags.compressionType = 0;
        // TODO: check if fb_mem actually needs to be rdma capable
        mem_prop.allocFlags.gpuDirectRDMACapable = mmap_supports_rdma;
        mem_prop.allocFlags.usage = 0;
        // Try fabric first.  This can fail for a number of reasons, but most commonly,
        // it's because the application isn't bound to an IMEX channel
        if(gpu->info->fabric_supported != 0) {
#if CUDA_VERSION >= 12030
          mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#endif
          alloc = GPUAllocation::allocate_mmap(gpu, mem_prop, size, 0,
                                               /*peer_enabled=*/true);
        }
        if(alloc == nullptr) {
#if defined(REALM_ON_WINDOWS)
          mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
#else
          mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
          alloc = GPUAllocation::allocate_mmap(gpu, mem_prop, size, 0,
                                               /*peer_enabled=*/true);
        }
      } else
#endif
      {
        alloc = GPUAllocation::allocate_dev(gpu, size, /*peer_enabled=*/true,
                                            /*shareable=*/true);
      }

      if(alloc == nullptr) {
        log_gpu.fatal() << "Failed to allocate GPU memory of size " << size;
        abort();
      }

      return alloc->get_dptr();
    }

    void GPU::create_fb_memory(RuntimeImpl *runtime, size_t size, size_t ib_size)
    {

      if(size > 0) {
        Memory m = runtime->next_local_memory_id();
        fbmem_base = allocate_device_memory(this, size);
        fbmem = new GPUFBMemory(m, this, fbmem_base, size);
        runtime->add_memory(fbmem);
      }

      if(ib_size > 0) {
        Memory m = runtime->next_local_ib_memory_id();
        fb_ibmem_base = allocate_device_memory(this, ib_size);
        fb_ibmem = new GPUFBIBMemory(m, this, fb_ibmem_base, ib_size);
        runtime->add_ib_memory(fb_ibmem);
      }
    }

    void GPU::create_dynamic_fb_memory(RuntimeImpl *runtime, size_t max_size)
    {
      // if the max_size is non-zero, also limit by what appears to be
      //  currently available
      if(max_size > 0) {
	AutoGPUContext agc(this);

        size_t free_bytes, total_bytes;
        CHECK_CU( CUDA_DRIVER_FNPTR(cuMemGetInfo)(&free_bytes, &total_bytes) );
        if(total_bytes < max_size)
          max_size = total_bytes;
      }

      Memory m = runtime->next_local_memory_id();
      // TODO(apryakhin@): Determine if we need to keep the pointer.
      fb_dmem = new GPUDynamicFBMemory(m, this, max_size);
      runtime->add_memory(fb_dmem);
    }

#ifdef REALM_USE_CUDART_HIJACK
    void GPU::register_fat_binary(const FatBin *fatbin)
    {
      AutoGPUContext agc(this);

      log_gpu.info() << "registering fat binary " << fatbin << " with GPU " << this;

      // have we see this one already?
      if(device_modules.count(fatbin) > 0) {
	log_gpu.warning() << "duplicate registration of fat binary data " << fatbin;
	return;
      }

      if(fatbin->data != 0) {
	// binary data to be loaded with cuModuleLoad(Ex)
	CUmodule module = load_cuda_module(fatbin->data);
	device_modules[fatbin] = module;
	return;
      }

      assert(0);
    }
    
    void GPU::register_variable(const RegisteredVariable *var)
    {
      AutoGPUContext agc(this);

      log_gpu.debug() << "registering variable " << var->device_name << " (" << var->host_var << ") with GPU " << this;

      // have we seen it already?
      if(device_variables.count(var->host_var) > 0) {
	log_gpu.warning() << "duplicate registration of variable " << var->device_name;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(var->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUdeviceptr ptr;
      size_t size;
      CHECK_CU( cuModuleGetGlobal(&ptr, &size, module, var->device_name) );
      device_variables[var->host_var] = ptr;

      // if this is a managed variable, the "host_var" is actually a pointer
      //  we need to fill in, so do that now
      if(var->managed) {
        CUdeviceptr *indirect = const_cast<CUdeviceptr *>(static_cast<const CUdeviceptr *>(var->host_var));
        if(*indirect) {
          // it's already set - make sure we're consistent (we're probably not)
          if(*indirect != ptr) {
            log_gpu.fatal() << "__managed__ variables are not supported when using multiple devices with CUDART hijack enabled";
            abort();
          }
        } else {
          *indirect = ptr;
        }
      }
    }
    
    void GPU::register_function(const RegisteredFunction *func)
    {
      AutoGPUContext agc(this);

      log_gpu.debug() << "registering function " << func->device_fun << " (" << func->host_fun << ") with GPU " << this;

      // have we seen it already?
      if(device_functions.count(func->host_fun) > 0) {
	log_gpu.warning() << "duplicate registration of function " << func->device_fun;
	return;
      }

      // get the module it lives in
      std::map<const FatBin *, CUmodule>::const_iterator it = device_modules.find(func->fat_bin);
      assert(it != device_modules.end());
      CUmodule module = it->second;

      CUfunction f;
      // the cuda runtime apparently permits calls to __cudaRegisterFunction
      //  that name a symbol that does not actually exist in the module - since
      //  we are doing eager lookup, we need to tolerate CUDA_ERROR_NOT_FOUND
      //  results here
      CUresult res = CUDA_DRIVER_FNPTR(cuModuleGetFunction)(&f, module, func->device_fun);
      switch(res) {
      case CUDA_SUCCESS:
        {
          device_functions[func->host_fun] = f;
          break;
        }
      case CUDA_ERROR_NOT_FOUND:
        {
          // just an informational message here - an actual attempt to invoke
          //  this kernel will be a fatal error at the call site
          log_gpu.info() << "symbol '" << func->device_fun
                         << "' not found in module " << module;
          break;
        }
      default:
        {
          const char *name, *str;
          CUDA_DRIVER_FNPTR(cuGetErrorName)(res, &name);
          CUDA_DRIVER_FNPTR(cuGetErrorString)(res, &str);
          log_gpu.fatal() << "unexpected error when looking up device function '"
                          << func->device_fun << "' in module " << module
                          << ": " << str << " (" << name << ")";
          abort();
        }
      }
    }

    CUfunction GPU::lookup_function(const void *func)
    {
      std::map<const void *, CUfunction>::iterator finder = device_functions.find(func);
      assert(finder != device_functions.end());
      return finder->second;
    }

    CUdeviceptr GPU::lookup_variable(const void *var)
    {
      std::map<const void *, CUdeviceptr>::iterator finder = device_variables.find(var);
      assert(finder != device_variables.end());
      return finder->second;
    }
#endif

    CUmodule GPU::load_cuda_module(const void *data)
    {
      const unsigned num_options = 4;
      CUjit_option jit_options[num_options];
      void*        option_vals[num_options];
      const size_t buffer_size = 16384;
      char* log_info_buffer = (char*)malloc(buffer_size);
      char* log_error_buffer = (char*)malloc(buffer_size);
      jit_options[0] = CU_JIT_INFO_LOG_BUFFER;
      jit_options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
      jit_options[2] = CU_JIT_ERROR_LOG_BUFFER;
      jit_options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      option_vals[0] = log_info_buffer;
      option_vals[1] = (void*)buffer_size;
      option_vals[2] = log_error_buffer;
      option_vals[3] = (void*)buffer_size;
      CUmodule module;
      CUresult result = CUDA_DRIVER_FNPTR(cuModuleLoadDataEx)(&module,
                                                              data,
                                                              num_options,
                                                              jit_options,
                                                              option_vals);
      if (result != CUDA_SUCCESS)
      {
#ifdef REALM_ON_MACOS
        if (result == CUDA_ERROR_OPERATING_SYSTEM) {
          log_gpu.error("ERROR: Device side asserts are not supported by the "
                              "CUDA driver for MAC OSX, see NVBugs 1628896.");
        } else
#endif
        if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
          log_gpu.error("ERROR: The binary was compiled for the wrong GPU "
                              "architecture. Update the 'GPU_ARCH' flag at the top "
                              "of runtime/runtime.mk to match/include your current GPU "
			      "architecture (%d).",
			(info->major * 10 + info->minor));
        } else {
	  log_gpu.error("Failed to load CUDA module! Error log: %s", 
			log_error_buffer);
#if CUDA_VERSION >= 6050
	  const char *name, *str;
	  CHECK_CU( CUDA_DRIVER_FNPTR(cuGetErrorName)(result, &name) );
	  CHECK_CU( CUDA_DRIVER_FNPTR(cuGetErrorString)(result, &str) );
	  fprintf(stderr,"CU: cuModuleLoadDataEx = %d (%s): %s\n",
		  result, name, str);
#else
	  fprintf(stderr,"CU: cuModuleLoadDataEx = %d\n", result);
#endif
	}
	abort();
      }
      else
        log_gpu.info("Loaded CUDA Module. JIT Output: %s", log_info_buffer);
      free(log_info_buffer);
      free(log_error_buffer);
      return module;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class AutoGPUContext

    AutoGPUContext::AutoGPUContext(GPU& _gpu)
      : gpu(&_gpu)
    {
      gpu->push_context();
    }

    AutoGPUContext::AutoGPUContext(GPU *_gpu)
      : gpu(_gpu)
    {
      if(gpu)
        gpu->push_context();
    }

    AutoGPUContext::~AutoGPUContext(void)
    {
      if(gpu)
        gpu->pop_context();
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class CudaModuleConfig

    CudaModuleConfig::CudaModuleConfig(void)
      : ModuleConfig("cuda")
    {
      config_map.insert({"gpu", &cfg_num_gpus});
      config_map.insert({"zcmem", &cfg_zc_mem_size});
      config_map.insert({"fbmem", &cfg_fb_mem_size});
      config_map.insert({"ib_fbmem", &cfg_fb_ib_size});
      config_map.insert({"ib_zcmem", &cfg_zc_ib_size});
      config_map.insert({"uvmem", &cfg_uvm_mem_size});
      config_map.insert({"use_dynamic_fb", &cfg_use_dynamic_fb});
      config_map.insert({"dynfb_max_size", &cfg_dynfb_max_size});
      config_map.insert({"task_streams", &cfg_task_streams});
      config_map.insert({"d2d_streams", &cfg_d2d_streams});

      resource_map.insert({"gpu", &res_num_gpus});
      resource_map.insert({"fbmem", &res_min_fbmem_size});
    }

    bool CudaModuleConfig::discover_resource(void)
    {
      CUresult ret = CUDA_DRIVER_FNPTR(cuInit)(0);
      cuda_init_code = ret;
      if (ret != CUDA_SUCCESS) {
        const char *err_name, *err_str;
        CUDA_DRIVER_FNPTR(cuGetErrorName)(ret, &err_name);
        CUDA_DRIVER_FNPTR(cuGetErrorString)(ret, &err_str);
        log_gpu.warning() << "cuInit(0) returned " << ret << " ("
                          << err_name << "): " << err_str
                          << ", resource discovery failed";
      } else {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetCount)(&res_num_gpus));
        res_fbmem_sizes.resize(res_num_gpus);
        for(int i = 0; i < res_num_gpus; i++) {
          CUdevice device;
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGet)(&device, i));
          CHECK_CU(
              CUDA_DRIVER_FNPTR(cuDeviceTotalMem)(&res_fbmem_sizes[i], device));
        }
        res_min_fbmem_size =
            *std::min_element(res_fbmem_sizes.begin(), res_fbmem_sizes.end());
        resource_discover_finished = true;
      }
      return resource_discover_finished;
    }

    void CudaModuleConfig::configure_from_cmdline(std::vector<std::string>& cmdline)
    {
      assert(finish_configured == false);
      // first order of business - read command line parameters
      CommandLineParser cp;

      cp.add_option_int_units("-ll:fsize", cfg_fb_mem_size, 'm')
        .add_option_int_units("-ll:zsize", cfg_zc_mem_size, 'm')
        .add_option_int_units("-ll:ib_fsize", cfg_fb_ib_size, 'm')
        .add_option_int_units("-ll:ib_zsize", cfg_zc_ib_size, 'm')
        .add_option_int_units("-ll:msize", cfg_uvm_mem_size, 'm')
        .add_option_int("-cuda:dynfb", cfg_use_dynamic_fb)
        .add_option_int_units("-cuda:dynfb_max", cfg_dynfb_max_size, 'm')
        .add_option_int("-ll:gpu", cfg_num_gpus)
        .add_option_string("-ll:gpu_ids", cfg_gpu_idxs)
        .add_option_int("-ll:streams", cfg_task_streams)
        .add_option_int("-ll:d2d_streams", cfg_d2d_streams)
        .add_option_int("-ll:d2d_priority", cfg_d2d_stream_priority)
        .add_option_int("-ll:gpuworkthread", cfg_use_worker_threads)
        .add_option_int("-ll:gpuworker", cfg_use_shared_worker)
        .add_option_int("-ll:pin", cfg_pin_sysmem)
        .add_option_bool("-cuda:callbacks", cfg_fences_use_callbacks)
        .add_option_bool("-cuda:nohijack", cfg_suppress_hijack_warning)
        .add_option_int("-cuda:skipgpus", cfg_skip_gpu_count)
        .add_option_bool("-cuda:skipbusy", cfg_skip_busy_gpus)
        .add_option_int_units("-cuda:minavailmem", cfg_min_avail_mem, 'm')
        .add_option_int("-cuda:legacysync", cfg_task_legacy_sync)
        .add_option_int("-cuda:contextsync", cfg_task_context_sync)
        .add_option_int("-cuda:maxctxsync", cfg_max_ctxsync_threads)
        .add_option_int("-cuda:lmemresize", cfg_lmem_resize_to_max)
        .add_option_int("-cuda:mtdma", cfg_multithread_dma)
        .add_option_int_units("-cuda:hostreg", cfg_hostreg_limit, 'm')
        .add_option_int("-cuda:ipc", cfg_use_cuda_ipc);
  #ifdef REALM_USE_CUDART_HIJACK
        cp.add_option_int("-cuda:nongpusync", Cuda::cudart_hijack_nongpu_sync);
  #endif

      bool ok = cp.parse_command_line(cmdline);
      if(!ok) {
        printf("error reading CUDA command line parameters\n");
        exit(1);
      }
    }


    ////////////////////////////////////////////////////////////////////////
    //
    // class CudaModule

    // our interface to the rest of the runtime

    CudaModule *cuda_module_singleton = 0;

    CudaModule::CudaModule(RuntimeImpl *_runtime)
      : Module("cuda")
      , config(nullptr)
      , runtime(_runtime)
      , shared_worker(0)
      , zcmem_cpu_base(0)
      , zcib_cpu_base(0)
      , zcmem(0)
      , uvm_base(0)
      , uvmmem(0)
      , initialization_complete(false)
      , cudaipc_condvar(cudaipc_mutex)
      , cudaipc_responses_received(0)
    {
      assert(!cuda_module_singleton);
      cuda_module_singleton = this;
      rh_listener = new GPUReplHeapListener(this);
    }
      
    CudaModule::~CudaModule(void)
    {
      assert(config != nullptr);
      config = nullptr;
      delete_container_contents(gpu_info);
      assert(cuda_module_singleton == this);
      cuda_module_singleton = 0;
      cuhook_register_callback_fnptr = nullptr;
      cuhook_start_task_fnptr = nullptr;
      cuhook_end_task_fnptr = nullptr;
      cuhook_enabled = false;
      delete rh_listener;
    }

    static std::string convert_uuid(CUuuid& cu_uuid)
    {
      stringbuilder ss;
      ss << "GPU-";
      for (size_t i = 0; i < 16; i++) {
        switch (i) {
        case 4:
        case 6:
        case 8:
        case 10:
          ss << '-';
        }
        ss << std::hex << std::setfill('0') << std::setw(2) << (0xFF & (int)cu_uuid.bytes[i]);
      }
      return ss;
    }

#ifndef STRINGIFY
#define STRINGIFY(s) #s
#endif

    template <typename Fn>
    static void cuGetProcAddress_stable(PFN_cuGetProcAddress loader, Fn &fnptr,
                                        const char *name, int version,
                                        const char *err_msg)
    {
      CUresult ret = CUDA_SUCCESS;
      // When using cuGetProcAddress, we need to make sure to specify the either the
      // version the API was introduced, or the current compilation version, whichever is
      // newer.  This is to deal with CUDA changing the API signature for the same API,
      // but allows us to retrieve APIs from drivers newer than what we're compiling with
#if CUDA_VERSION < 12000
      ret = (loader)(name, reinterpret_cast<void **>(&fnptr),
                     std::max(CUDA_VERSION, version), CU_GET_PROC_ADDRESS_DEFAULT);
#else
      // cuGetProcAddress changed signature in 12.0+ to include more diagnostic
      // information we don't need.
      ret =
          (loader)(name, reinterpret_cast<void **>(&fnptr),
                   std::max(CUDA_VERSION, version), CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
#endif
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, err_msg, ret);
      }
    }

    static bool resolve_cuda_api_fnptrs(void)
    {
      if(cuda_api_fnptrs_loaded) {
        return true;
      }

      PFN_cuGetProcAddress cuGetProcAddress_fnptr = nullptr;

#if defined(REALM_USE_LIBDL)
      log_gpu.info() << "dynamically loading libcuda.so";
      void *libcuda = dlopen("libcuda.so.1", RTLD_NOW);
      if(!libcuda) {
        log_gpu.info() << "could not open libcuda.so: " << strerror(errno);
        return false;
      }
      // Use the symbol we get from the dynamically loaded library
      cuGetProcAddress_fnptr = reinterpret_cast<decltype(cuGetProcAddress_fnptr)>(
          dlsym(libcuda, STRINGIFY(cuGetProcAddress)));
#elif CUDA_VERSION >= 11030
      // Use the statically available symbol
      cuGetProcAddress_fnptr = &cuGetProcAddress;
#endif

      if(cuGetProcAddress_fnptr != nullptr) {
#define DRIVER_GET_FNPTR(name, ver)                                                      \
  cuGetProcAddress_stable(cuGetProcAddress_fnptr, name##_fnptr, #name, ver,              \
                          "Could not retrieve symbol " #name);

        CUDA_DRIVER_APIS(DRIVER_GET_FNPTR);
#undef DRIVER_GET_FNPTR
      } else {
#if defined(REALM_USE_LIBDL)
#define DRIVER_GET_FNPTR(name, ver)                                                      \
  if(CUDA_SUCCESS != (nullptr != (name##_fnptr = reinterpret_cast<decltype(&name)>(      \
                                      dlsym(libcuda, STRINGIFY(name)))))) {              \
    log_gpu.info() << "Could not retrieve symbol " #name;                                \
  }
        CUDA_DRIVER_APIS(DRIVER_GET_FNPTR)
#undef DRIVER_GET_FNPTR
#else
#define DRIVER_GET_FNPTR(name, ver) name##_fnptr = &name;
        // Only enumerate the driver apis for the base toolkit version, extra features
        // cannot be enumerated
        CUDA_DRIVER_APIS_BASE(DRIVER_GET_FNPTR);
#undef DRIVER_GET_FNPTR
#endif /* REALM_USE_LIBDL */
      }

      cuda_api_fnptrs_loaded = true;

      return true;
    }

    static bool resolve_nvml_api_fnptrs()
    {
#ifdef REALM_USE_LIBDL
      void *libnvml = NULL;
      if (nvml_api_fnptrs_loaded)
        return true;
      log_gpu.info() << "dynamically loading libnvidia-ml.so";
      libnvml = dlopen("libnvidia-ml.so.1", RTLD_NOW);
      if (libnvml == NULL) {
        log_gpu.info() << "could not open libnvidia-ml.so" << strerror(errno);
        return false;
      }

#define DRIVER_GET_FNPTR(name)                                                           \
  do {                                                                                   \
    void *sym = dlsym(libnvml, STRINGIFY(name));                                         \
    if(!sym) {                                                                           \
      log_gpu.info() << "symbol '" STRINGIFY(name) " missing from libnvidia-ml.so!";     \
    }                                                                                    \
    name##_fnptr = reinterpret_cast<decltype(&name)>(sym);                               \
  } while(0)

      NVML_APIS(DRIVER_GET_FNPTR);
#undef DRIVER_GET_FNPTR

      nvml_api_fnptrs_loaded = true;
      return true;
#else
      return false;
#endif
    }

    /*static*/ ModuleConfig *CudaModule::create_module_config(RuntimeImpl *runtime)
    {
      CudaModuleConfig *config = new CudaModuleConfig();
      // load the cuda lib
      if(!resolve_cuda_api_fnptrs()) {
        // warning was printed in resolve function
        delete config;
        return nullptr;
      }
      if(!config->discover_resource()) {
        log_gpu.error("We are not able to discover the CUDA resources.");
      }
      return config;
    }

    template <typename T>
    static void get_nvml_field_value(const nvmlFieldValue_t &field_value, T &value)
    {
      if(field_value.nvmlReturn != NVML_SUCCESS) {
        return;
      }
      switch(field_value.valueType) {
      case NVML_VALUE_TYPE_DOUBLE:
        value = static_cast<T>(field_value.value.dVal);
        break;
#if CUDA_VERSION >= 12020
      case NVML_VALUE_TYPE_SIGNED_INT:
        value = static_cast<T>(field_value.value.siVal);
        break;
#endif
      case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
        value = static_cast<T>(field_value.value.sllVal);
        break;
      case NVML_VALUE_TYPE_UNSIGNED_INT:
        value = static_cast<T>(field_value.value.uiVal);
        break;
      case NVML_VALUE_TYPE_UNSIGNED_LONG:
        value = static_cast<T>(field_value.value.ulVal);
        break;
      case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        value = static_cast<T>(field_value.value.ullVal);
        break;
      default:
        log_gpu.info("Unknown nvml field value %d",
                     static_cast<int>(field_value.valueType));
        break;
      }
    }

    /*static*/ Module *CudaModule::create_module(RuntimeImpl *runtime)
    {
      ModuleConfig *uncasted_config = runtime->get_module_config("cuda");
      if(!uncasted_config) {
        return nullptr;
      }

      CudaModule *m = new CudaModule(runtime);

      CudaModuleConfig *config = checked_cast<CudaModuleConfig *>(uncasted_config);
      assert(config != nullptr);
      assert(config->finish_configured);
      assert(m->name == config->get_name());
      assert(m->config == nullptr);
      m->config = config;

      // check if gpus have been requested
      bool init_required = ((m->config->cfg_num_gpus > 0) || !m->config->cfg_gpu_idxs.empty());

      // check if cuda can be initialized
      if(cuda_init_code != CUDA_SUCCESS && init_required) {
        // failure to initialize the driver is a fatal error if we know gpus
        //  have been requested
        log_gpu.warning() << "gpus requested, but cuInit(0) returned " << cuda_init_code;
        init_required = false;
      }

      // do not create the module if previous steps are failed
      if(!init_required) {
        const char *err_name, *err_str;
        CUDA_DRIVER_FNPTR(cuGetErrorName)(cuda_init_code, &err_name);
        CUDA_DRIVER_FNPTR(cuGetErrorString)(cuda_init_code, &err_str);
        log_gpu.info() << "cuda module is not loaded, cuInit(0) returned:" << err_name
                       << " (" << err_str << ")";
        delete m;
        return nullptr;
      }

      CHECK_CU(CUDA_DRIVER_FNPTR(cuDriverGetVersion)(&m->cuda_api_version));

      // check if nvml can be initialized
      // we will continue create cuda module even if nvml can not be initialized
      if(!nvml_initialized && resolve_nvml_api_fnptrs()) {
        nvmlReturn_t res = NVML_FNPTR(nvmlInit)();
        if(res == NVML_SUCCESS) {
          nvml_initialized = true;
        } else {
          log_gpu.info() << "Unable to initialize nvml: Error(" << (unsigned long long)res
                         << ')';
        }
      }

      // create GPUInfo
      std::vector<GPUInfo *> infos;
      {
        for(int i = 0; i < config->res_num_gpus; i++) {
          GPUInfo *info = new GPUInfo;
          int attribute_value = 0;

          info->index = i;
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGet)(&info->device, i));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetName)(info->name, sizeof(info->name),
                                                      info->device));
          CHECK_CU(
              CUDA_DRIVER_FNPTR(cuDeviceTotalMem)(&info->totalGlobalMem, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetUuid)(&info->uuid, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
              &info->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
              &info->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
              &info->pci_busid, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
              &info->pci_deviceid, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, info->device));
          CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
              &info->pci_domainid, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, info->device));
          CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
          (&attribute_value, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
           info->device);
          info->host_gpu_same_va = !!attribute_value;
#if CUDA_VERSION >= 12030
          CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
          (&attribute_value, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
           info->device);
#else
          attribute_value = 0;
#endif
          info->fabric_supported = !!attribute_value;
          // Assume x16 PCI-e 2.0 = 8000 MB/s, which is reasonable for most
          // systems
          info->pci_bandwidth = 8000;
          info->logical_peer_bandwidth.resize(config->res_num_gpus, 0);
          info->logical_peer_latency.resize(config->res_num_gpus, SIZE_MAX);

          info->has_numa_preference = false;
          memset(info->numa_node_affinity, 0xff, sizeof(info->numa_node_affinity));

          if(nvml_initialized) {
            // Convert uuid bytes to uuid string for nvml
            std::string uuid = convert_uuid(info->uuid);
            if(NVML_SUCCESS !=
               NVML_FNPTR(nvmlDeviceGetHandleByUUID)(uuid.c_str(), &info->nvml_dev)) {
              // Unfortunately, CUDA doesn't provide a way to query if a device is in MIG
              // or not, and for some god awful reason NVML decided it must prefix the
              // UUID with either GPU- or MIG-.  So try 'GPU-' first, if it fails, try
              // 'MIG-'.
              uuid[0] = 'M';
              uuid[1] = 'I';
              uuid[2] = 'G';
              CHECK_NVML(
                  NVML_FNPTR(nvmlDeviceGetHandleByUUID)(uuid.c_str(), &info->nvml_dev));
              // Then translate it to a physical device handle since that's all we'll
              // really be caring about for the following queries
              CHECK_NVML(NVML_FNPTR(nvmlDeviceGetDeviceHandleFromMigDeviceHandle)(
                  info->nvml_dev, &info->nvml_dev));
            }
            unsigned int gen, buswidth;
            // Rates in MB/s from https://en.wikipedia.org/wiki/PCI_Express
            static const unsigned int rates[] = {250, 500, 985, 1969, 3938, 7563, 15125};
            static const unsigned int rates_len = sizeof(rates) / sizeof(rates[0]);
            // Use the max pcie link information here, as when the GPU is not in use,
            // the OS may power down some links to conserve power, but we want to
            // estimate the bandwidth when in use.
            CHECK_NVML(
                NVML_FNPTR(nvmlDeviceGetMaxPcieLinkGeneration)(info->nvml_dev, &gen));
            CHECK_NVML(
                NVML_FNPTR(nvmlDeviceGetMaxPcieLinkWidth)(info->nvml_dev, &buswidth));
            if(gen >= sizeof(rates) / sizeof(rates[0])) {
              log_gpu.warning() << "Unknown PCIe generation version '" << gen
                                << "', assuming '" << rates_len << '\'';
              gen = rates_len;
            }
            info->pci_bandwidth = (rates[gen - 1] * buswidth);

#if NVML_API_VERSION >= 11
            {
              memset(info->numa_node_affinity, 0, sizeof(info->numa_node_affinity));
              nvmlReturn_t ret = NVML_FNPTR(nvmlDeviceGetMemoryAffinity)(
                  info->nvml_dev, info->MAX_NUMA_NODE_LEN, info->numa_node_affinity,
                  NVML_AFFINITY_SCOPE_NODE);
              if(ret != NVML_SUCCESS) {
                memset(info->numa_node_affinity, -1, sizeof(info->numa_node_affinity));
              }
            }
#endif
#if NVML_API_VERSION >= 12
            {
              nvmlGpuFabricInfo_t fabric_info;
              if((NVML_FNPTR(nvmlDeviceGetGpuFabricInfo) != nullptr) &&
                 (NVML_SUCCESS ==
                  NVML_FNPTR(nvmlDeviceGetGpuFabricInfo)(info->nvml_dev, &fabric_info))) {
                if(fabric_info.status != NVML_SUCCESS) {
                  log_gpu.info() << "Unable to retrieve fabric information from NVML, "
                                    "fabric import/export may not work properly";
                } else {
                  // NVML decided to change the name of this field between minor versions,
                  // but NVML doesn't have a way to distinguish between minor versions, so
                  // we have to use CUDA_VERSION...
#if CUDA_VERSION >= 12030
                  info->fabric_clique = fabric_info.cliqueId;
#else
                  info->fabric_clique = fabric_info.partitionId;
#endif
                  memcpy(info->fabric_uuid.bytes, fabric_info.clusterUuid,
                         sizeof(fabric_info.clusterUuid));
                }
              }
            }
#endif
          }

          // For fast lookups, check if we actually have a numa preference
          for(size_t i = 0; i < info->MAX_NUMA_NODE_LEN; i++) {
            if(info->numa_node_affinity[i] != (unsigned long)-1) {
              info->has_numa_preference = true;
              break;
            }
          }

          log_gpu.info() << "GPU #" << i << ": " << info->name << " (" << info->major
                         << '.' << info->minor << ") " << (info->totalGlobalMem >> 20)
                         << " MB";

          infos.push_back(info);
        }

        if (nvml_initialized) {
          for(GPUInfo *info : infos) {
            nvmlFieldValue_t values[] = {
#if CUDA_VERSION >= 12000
              {NVML_FI_DEV_NVLINK_GET_SPEED, 0},
#else
              {NVML_FI_DEV_NVLINK_SPEED_MBPS_L0, 0}
#endif
#if CUDA_VERSION >= 12030
              {NVML_FI_DEV_C2C_LINK_GET_STATUS, 0},
              {NVML_FI_DEV_C2C_LINK_COUNT, 0},
              {NVML_FI_DEV_C2C_LINK_GET_MAX_BW, 0},
#endif
            };

            CHECK_NVML(NVML_FNPTR(nvmlDeviceGetFieldValues)(
                info->nvml_dev, sizeof(values) / sizeof(values[0]), values));
#if CUDA_VERSION >= 12030
            int c2c_status = 0;
            get_nvml_field_value(values[1], c2c_status);
            if(c2c_status != 0) {
              size_t c2c_rate = 0;
              size_t c2c_count = 0;
              get_nvml_field_value(values[2], c2c_count);
              get_nvml_field_value(values[3], c2c_rate);
              info->c2c_bandwidth = c2c_count * c2c_rate;
            }
#endif

            size_t nvlink_rate = 0;
            get_nvml_field_value(values[0], nvlink_rate);

            // Iterate each of the links for this GPU and find what's on the other end
            // of the link, adding this link's bandwidth to the accumulated peer pair
            // bandwidth.
            for(size_t i = 0; i < NVML_NVLINK_MAX_LINKS; i++) {
              nvmlIntNvLinkDeviceType_t dev_type;
              nvmlEnableState_t link_state;
              nvmlPciInfo_t pci_info;
              nvmlReturn_t status =
                  NVML_FNPTR(nvmlDeviceGetNvLinkState)(info->nvml_dev, i, &link_state);
              if(status != NVML_SUCCESS || link_state != NVML_FEATURE_ENABLED) {
                continue;
              }

              if(NVML_FNPTR(nvmlDeviceGetNvLinkRemoteDeviceType) != nullptr) {
                CHECK_NVML(NVML_FNPTR(nvmlDeviceGetNvLinkRemoteDeviceType)(info->nvml_dev,
                                                                          i, &dev_type));
              } else {
                // GetNvLinkRemoteDeviceType not found, probably an older nvml driver, so
                // assume GPU
                dev_type = NVML_NVLINK_DEVICE_TYPE_GPU;
              }

              if(dev_type == NVML_NVLINK_DEVICE_TYPE_GPU) {
                CHECK_NVML(NVML_FNPTR(nvmlDeviceGetNvLinkRemotePciInfo)(info->nvml_dev, i,
                                                                        &pci_info));
                // Unfortunately NVML doesn't give a way to return a GPU handle for a remote
                // end point, so we have to search for the remote GPU using the PCIe
                // information...
                int peer_gpu_idx = 0;
                for(peer_gpu_idx = 0; peer_gpu_idx < config->res_num_gpus; peer_gpu_idx++) {
                  if(infos[peer_gpu_idx]->pci_busid == static_cast<int>(pci_info.bus) &&
                    infos[peer_gpu_idx]->pci_deviceid == static_cast<int>(pci_info.device) &&
                    infos[peer_gpu_idx]->pci_domainid == static_cast<int>(pci_info.domain)) {
                    // Found the peer device on the other end of the link!  Add this link's
                    // bandwidth to the logical peer link
                    info->logical_peer_bandwidth[peer_gpu_idx] += nvlink_rate;
                    info->logical_peer_latency[peer_gpu_idx] = 100;
                    break;
                  }
                }

                if(peer_gpu_idx == config->res_num_gpus) {
                  // We can't make any assumptions about this link, since we don't know
                  // what's on the other side.  This could be a GPU that was removed via
                  // CUDA_VISIBLE_DEVICES, or NVSWITCH / P9 NPU on a system with an slightly
                  // older driver that doesn't support "GetNvlinkRemotePciInfo"
                  log_gpu.info() << "GPU " << info->index
                                << " has active NVLINK to unknown device "
                                << pci_info.busId << "(" << std::hex
                                << pci_info.pciDeviceId << "), ignoring...";
                }
              } else if((info == infos[0]) && (dev_type == NVML_NVLINK_DEVICE_TYPE_SWITCH)) {
                // Accumulate the link bandwidth for one gpu and assume symmetry
                // across all GPUs, and all GPus have access to the NVSWITCH fabric
                info->nvswitch_bandwidth += nvlink_rate;
              } else if((info == infos[0]) && (dev_type == NVML_NVLINK_DEVICE_TYPE_IBMNPU)) {
                // TODO: use the npu_bandwidth for sysmem affinities
                // npu_bandwidth += nvlink_bandwidth;
              }
            }
          }
        }

        // query peer-to-peer access (all pairs)
        for(size_t i = 0; i < infos.size(); i++) {
          // two contexts on the same device can always "peer to peer"
          infos[i]->peers.insert(infos[i]->index);
          {
            // Gather the framebuffer bandwidth and latency from CUDA
            int memclk /*kHz*/, buswidth;
            CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
                &memclk, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                infos[i]->device));
            CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
                &buswidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                infos[i]->device));
            // Account for double-data rate memories
            infos[i]->logical_peer_bandwidth[i] =
                (250ULL * memclk * buswidth) / 1000000ULL;
            infos[i]->logical_peer_latency[i] =
                std::max(1ULL, 10000000ULL / memclk);
            log_gpu.info() << "GPU #" << i << " local memory: "
                           << infos[i]->logical_peer_bandwidth[i] << " MB/s, "
                           << infos[i]->logical_peer_latency[i] << " ns";
          }
          for (size_t j = 0; j < infos.size(); j++) {
            int can_access;
            if (i == j) {
              continue;
            }
            CHECK_CU(CUDA_DRIVER_FNPTR(cuDeviceCanAccessPeer)(
                &can_access, infos[i]->device, infos[j]->device));
            if (can_access) {
              infos[i]->peers.insert(infos[j]->index);
              if (infos[i]->logical_peer_bandwidth[j] == 0) {
                // Not nvlink (otherwise this would have been enumerated
                // earlier), so assume this is NVSWITCH (if we detected nvswitch
                // earlier) or PCIe
                infos[i]->logical_peer_bandwidth[j] =
                    std::max(infos[i]->nvswitch_bandwidth,
                             std::min(infos[i]->pci_bandwidth, infos[j]->pci_bandwidth));
                infos[i]->logical_peer_latency[j] = 400;
              }
              log_gpu.info()
                  << "p2p access from device " << infos[i]->index
                  << " to device " << infos[j]->index
                  << " bandwidth: " << infos[i]->logical_peer_bandwidth[j]
                  << " MB/s"
                  << " latency: " << infos[i]->logical_peer_latency[j] << " ns";
            }
          }
        }

        // give the gpu info we assembled to the module
        m->gpu_info.swap(infos);
      }

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void CudaModule::initialize(RuntimeImpl *runtime)
    {
      assert(config != NULL);
      Module::initialize(runtime);

      // if we are using a shared worker, create that next
      if(config->cfg_use_shared_worker) {
        shared_worker = new GPUWorker;

        if(config->cfg_use_worker_threads)
          shared_worker->start_background_thread(runtime->core_reservation_set(),
                                                 1 << 20); // hardcoded worker stack size
        else
          shared_worker->add_to_manager(&(runtime->bgwork));
      }

      // decode specific device id list if given
      std::vector<unsigned> fixed_indices;
      if(!config->cfg_gpu_idxs.empty()) {
        const char *p = config->cfg_gpu_idxs.c_str();
        while(true) {
          if(!isdigit(*p)) {
            log_gpu.fatal() << "invalid number in cuda device list: '" << p << "'";
            abort();
          }
          unsigned v = 0;
          do {
            v = (v * 10) + (*p++ - '0');
          } while(isdigit(*p));
          if(v >= gpu_info.size()) {
            log_gpu.fatal() << "requested cuda device id out of range: " << v
                            << " >= " << gpu_info.size();
            abort();
          }
          fixed_indices.push_back(v);
          if(!*p)
            break;
          if(*p == ',') {
            p++; // skip comma and parse another integer
          } else {
            log_gpu.fatal() << "invalid separator in cuda device list: '" << p << "'";
            abort();
          }
        }
        // if num_gpus was specified, they should match
        if(config->cfg_num_gpus > 0) {
          if(config->cfg_num_gpus != static_cast<int>(fixed_indices.size())) {
            log_gpu.fatal() << "mismatch between '-ll:gpu' and '-ll:gpu_ids'";
            abort();
          }
        } else
          config->cfg_num_gpus = fixed_indices.size();
        // also disable skip count and skip busy options
        config->cfg_skip_gpu_count = 0;
        config->cfg_skip_busy_gpus = false;
      }

      gpus.resize(config->cfg_num_gpus);
      unsigned gpu_count = 0;
      // try to get cfg_num_gpus, working through the list in order
      for(size_t i = config->cfg_skip_gpu_count;
          (i < gpu_info.size()) && (static_cast<int>(gpu_count) < config->cfg_num_gpus); i++) {
        int idx = (fixed_indices.empty() ? i : fixed_indices[i]);

        // try to create a context and possibly check available memory - in order
        //  to be compatible with an application's use of the cuda runtime, we
        //  need this to be the device's "primary context"

        // set context flags before we create it, but it's ok to be told that
        //  it's too late (unless lmem resize is wrong)
        {
          unsigned flags = CU_CTX_SCHED_BLOCKING_SYNC;
          if(config->cfg_lmem_resize_to_max)
            flags |= CU_CTX_LMEM_RESIZE_TO_MAX;

          CUresult res =
              CUDA_DRIVER_FNPTR(cuDevicePrimaryCtxSetFlags)(gpu_info[idx]->device, flags);
          if(res != CUDA_SUCCESS) {
            bool lmem_ok;
            if(res == CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) {
              if(config->cfg_lmem_resize_to_max) {
                unsigned act_flags = 0;
                CHECK_CU(CUDA_DRIVER_FNPTR(cuCtxGetFlags)(&act_flags));
                lmem_ok = ((act_flags & CU_CTX_LMEM_RESIZE_TO_MAX) != 0);
              } else
                lmem_ok = true;
            } else
              lmem_ok = false;

            if(!lmem_ok) {
              REPORT_CU_ERROR(Logger::LEVEL_ERROR, "cuDevicePrimaryCtxSetFlags", res);
              abort();
            }
          }
        }

        CUcontext context;
        CUresult res =
            CUDA_DRIVER_FNPTR(cuDevicePrimaryCtxRetain)(&context, gpu_info[idx]->device);
        // a busy GPU might return INVALID_DEVICE or OUT_OF_MEMORY here
        if((res == CUDA_ERROR_INVALID_DEVICE) || (res == CUDA_ERROR_OUT_OF_MEMORY)) {
          if(config->cfg_skip_busy_gpus) {
            log_gpu.info() << "GPU " << gpu_info[idx]->device
                           << " appears to be busy (res=" << res << ") - skipping";
            continue;
          } else {
            log_gpu.fatal() << "GPU " << gpu_info[idx]->device
                            << " appears to be in use - use CUDA_VISIBLE_DEVICES, "
                               "-cuda:skipgpus, or -cuda:skipbusy to select other GPUs";
            abort();
          }
        }
        // any other error is a (unknown) problem
        CHECK_CU(res);

        if(config->cfg_min_avail_mem > 0) {
          size_t total_mem, avail_mem;
          {
            CHECK_CU(CUDA_DRIVER_FNPTR(cuCtxPushCurrent)(context));
            CHECK_CU(CUDA_DRIVER_FNPTR(cuMemGetInfo)(&avail_mem, &total_mem));
            CUcontext popped;
            CHECK_CU(CUDA_DRIVER_FNPTR(cuCtxPopCurrent)(&popped));
            assert(popped == context);
          }
          if(avail_mem < config->cfg_min_avail_mem) {
            log_gpu.info() << "GPU " << gpu_info[idx]->device
                           << " does not have enough available memory (" << avail_mem
                           << " < " << config->cfg_min_avail_mem << ") - skipping";
            CHECK_CU(CUDA_DRIVER_FNPTR(cuDevicePrimaryCtxRelease)(gpu_info[idx]->device));
            continue;
          }
        }

        // either create a worker for this GPU or use the shared one
        GPUWorker *worker;
        if(config->cfg_use_shared_worker) {
          worker = shared_worker;
        } else {
          worker = new GPUWorker;

          if(config->cfg_use_worker_threads)
            worker->start_background_thread(runtime->core_reservation_set(),
                                            1 << 20); // hardcoded worker stack size
          else
            worker->add_to_manager(&(runtime->bgwork));
        }

        GPU *g = new GPU(this, gpu_info[idx], worker, context);

        if(!config->cfg_use_shared_worker)
          dedicated_workers[g] = worker;

        gpus[gpu_count++] = g;
      }

      // did we actually get the requested number of GPUs?
      if(static_cast<int>(gpu_count) < config->cfg_num_gpus) {
        log_gpu.fatal() << config->cfg_num_gpus << " GPUs requested, but only " << gpu_count
                        << " available!";
        assert(false);
      }

      // make sure we hear about any changes to the size of the replicated
      //  heap
      runtime->repl_heap.add_listener(rh_listener);
#ifdef REALM_USE_LIBDL
      cuhook_register_callback_fnptr =
          (PFN_cuhook_register_callback)dlsym(NULL, "cuhook_register_callback");
      cuhook_start_task_fnptr = (PFN_cuhook_start_task)dlsym(NULL, "cuhook_start_task");
      cuhook_end_task_fnptr = (PFN_cuhook_end_task)dlsym(NULL, "cuhook_end_task");
      if(cuhook_register_callback_fnptr && cuhook_start_task_fnptr &&
         cuhook_end_task_fnptr) {
        cuhook_register_callback_fnptr();
        cuhook_enabled = true;
      }
#endif
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void CudaModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      // each GPU needs its FB memory
      if(config->cfg_fb_mem_size > 0)
        for(std::vector<GPU *>::iterator it = gpus.begin(); it != gpus.end(); it++)
          (*it)->create_fb_memory(runtime, config->cfg_fb_mem_size, config->cfg_fb_ib_size);

      if(config->cfg_use_dynamic_fb)
        for(std::vector<GPU *>::iterator it = gpus.begin(); it != gpus.end(); it++)
          (*it)->create_dynamic_fb_memory(runtime, config->cfg_dynfb_max_size);

      // Allocate and assign sysmem memories
      if(!gpus.empty()) {
        IBMemory *ib_mem = nullptr;

        if(config->cfg_zc_mem_size > 0) {
          // In order to work around a bug in 12.3 and 12.4 drivers with shareable host
          // allocations, disable the shareable path if the driver we're running on does
          // not support at least 12.5
          GPUAllocation *alloc = GPUAllocation::allocate_host(
              gpus[0], config->cfg_zc_mem_size, /*peer_enabled=*/true,
              /*shareable=*/cuda_api_version >= 12050,
              /*same_va=*/true);
          if(alloc == nullptr) {
            log_gpu.fatal() << "Insufficient zero-copy device-mappable host memory: "
                            << config->cfg_zc_mem_size << " total bytes needed";
            abort();
          }
          Memory m = runtime->next_local_memory_id();
          zcmem = new GPUZCMemory(gpus[0], m, alloc->get_dptr(), alloc->get_hptr(),
                                  config->cfg_zc_mem_size, MemoryImpl::MKIND_ZEROCOPY,
                                  Memory::Kind::Z_COPY_MEM);
          runtime->add_memory(zcmem);
        }

        if(config->cfg_zc_ib_size > 0) {
          // In order to work around a bug in 12.3 and 12.4 drivers with shareable host
          // allocations, disable the shareable path if the driver we're running on does
          // not support at least 12.5
          GPUAllocation *alloc = GPUAllocation::allocate_host(
              gpus[0], config->cfg_zc_ib_size, /*peer_enabled=*/true,
              /*shareable=*/cuda_api_version >= 12050,
              /*same_va=*/false);
          if(alloc == nullptr) {
            log_gpu.fatal() << "Insufficient ib device-mappable host memory: "
                            << config->cfg_zc_ib_size << " total bytes needed";
            abort();
          }
          Memory m = runtime->next_local_ib_memory_id();
          ib_mem = new IBMemory(m, config->cfg_zc_ib_size, MemoryImpl::MKIND_ZEROCOPY,
                                Memory::Z_COPY_MEM, alloc->get_hptr(), nullptr);
          ib_mem->add_module_specific(new CudaDeviceMemoryInfo(gpus[0]->context));
          runtime->add_ib_memory(ib_mem);
        }

        // add the new memories as a pinned memory to all GPUs that support unified
        // addressing (as they'll all have the same VA on the device and on the host)
        for(GPU *gpu : gpus) {
          int uva_supported = 0;
          CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
          (&uva_supported, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, gpu->info->device);
          if(uva_supported != 0) {
            if(zcmem != nullptr) {
              gpu->pinned_sysmems.insert(zcmem->me);
            }
            if(ib_mem != nullptr) {
              gpu->pinned_sysmems.insert(ib_mem->me);
            }
          } else {
            log_gpu.warning() << "GPU #" << gpu->info->index
                              << " does not support unified addressing and thus cannot "
                                 "access allocated sysmem";
          }
        }
      }

      // a single unified (managed) memory for everybody
      if((config->cfg_uvm_mem_size > 0) && !gpus.empty()) {
        CUdeviceptr uvm_gpu_base = 0;
        {
          GPUAllocation *alloc =
              GPUAllocation::allocate_managed(gpus[0], config->cfg_uvm_mem_size);
          if(alloc == nullptr) {
            log_gpu.fatal() << "Insufficient managed memory: " << config->cfg_uvm_mem_size
                            << " total bytes needed";
            abort();
          }
          uvm_gpu_base = alloc->get_dptr();
        }

        uvm_base = reinterpret_cast<void *>(uvm_gpu_base);
        Memory m = runtime->next_local_memory_id();
        uvmmem =
            new GPUZCMemory(gpus[0], m, uvm_gpu_base, uvm_base, config->cfg_uvm_mem_size,
                            MemoryImpl::MKIND_MANAGED, Memory::Kind::GPU_MANAGED_MEM);
        runtime->add_memory(uvmmem);

        // add the managed memory to any GPU capable of concurrent access
        for(GPU *gpu : gpus) {
          int concurrent_access = 0;
          CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)
          (&concurrent_access, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
           gpu->info->device);

          if(concurrent_access) {
            gpu->managed_mems.insert(uvmmem->me);
          } else {
            log_gpu.warning()
                << "GPU #" << gpu->info->index
                << " is not capable of concurrent access to managed memory!";
          }
        }
      }
    }

    template <typename Container>
    static void enumerate_ipc_entries(std::vector<CudaIpcResponseEntry> &entries,
                                      Container container)
    {
      for(typename Container::value_type const mem : container) {
        CudaIpcResponseEntry entry;
        const CudaDeviceMemoryInfo *cdm =
            mem->template find_module_specific<CudaDeviceMemoryInfo>();
        if((cdm == nullptr) || (mem->size == 0)) {
          continue;
        }
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(mem->get_direct_ptr(0, 0));
        if(dptr != 0) {
          GPUAllocation &alloc = cdm->gpu->allocations[dptr];
          entry.src_gpu_uuid = cdm->gpu->info->uuid;
          entry.mem = mem->me;
          entry.base_ptr = dptr;
          entry.size = mem->size;
          if(alloc.get_ipc_handle(entry.data.ipc_handle)) {
            entry.type = CUDA_IPC_RESPONSE_TYPE_IPC;
          }
#if CUDA_VERSION >= 12030
          else if(alloc.get_fabric_handle(entry.data.fabric.handle)) {
            entry.type = CUDA_IPC_RESPONSE_TYPE_FABRIC;
            entry.data.fabric.clique_id = cdm->gpu->info->fabric_clique;
            memcpy(entry.data.fabric.cluster_uuid.bytes,
                   cdm->gpu->info->fabric_uuid.bytes,
                   sizeof(cdm->gpu->info->fabric_uuid.bytes));
          }
#endif
          else {
            continue;
          }

          entries.push_back(entry);
        }
      }
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void CudaModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      // each GPU needs a processor
      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++)
	(*it)->create_processor(runtime,
				2 << 20); // TODO: don't use hardcoded stack size...
    }

    // create any DMA channels provided by the module (default == do nothing)
    void CudaModule::create_dma_channels(RuntimeImpl *runtime)
    {
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with CUDA
      if(config->cfg_pin_sysmem && !gpus.empty()) {
        const std::vector<MemoryImpl *> &local_mems =
            runtime->nodes[Network::my_node_id].memories;
        // <NEW_DMA> also add intermediate buffers into local_mems
        const std::vector<IBMemory *> &local_ib_mems =
            runtime->nodes[Network::my_node_id].ib_memories;
        std::set<MemoryImpl *> all_local_mems;
        all_local_mems.insert(local_mems.begin(), local_mems.end());
        all_local_mems.insert(local_ib_mems.begin(), local_ib_mems.end());
        // </NEW_DMA>
        assert(all_local_mems.size() == (local_ib_mems.size() + local_mems.size()));
        for(MemoryImpl *mem_impl : all_local_mems) {
          // ignore FB/ZC/managed memories or anything that doesn't have a
          //   "direct" pointer
          if((mem_impl->kind == MemoryImpl::MKIND_GPUFB) ||
             (mem_impl->kind == MemoryImpl::MKIND_ZEROCOPY) ||
             (mem_impl->kind == MemoryImpl::MKIND_MANAGED))
            continue;

          // skip any memory that's over the max size limit for host
          //  registration
          if((config->cfg_hostreg_limit > 0) &&
             (mem_impl->size > config->cfg_hostreg_limit)) {
            log_gpu.info() << "memory " << mem_impl->me
                           << " is larger than hostreg limit (" << mem_impl->size << " > "
                           << config->cfg_hostreg_limit << ") - skipping registration";
            continue;
          }

          void *base = mem_impl->get_direct_ptr(0, mem_impl->size);
          if(base == 0) {
            continue;
          }

          GPUAllocation *alloc =
              GPUAllocation::register_allocation(gpus[0], base, mem_impl->size);
          if(alloc == nullptr) {
            log_gpu.info() << "failed to register mem " << mem_impl->me << " (" << base
                           << " + " << mem_impl->size << ")";
            continue;
          }

          // Make sure each gpu knows that this is a pinned sysmem it can use
          for(GPU *gpu : gpus) {
            gpu->pinned_sysmems.insert(mem_impl->me);
          }
        }
      }

      // ask any ipc-able nodes to share handles with us
      if(config->cfg_use_cuda_ipc && !gpus.empty()) {
        NodeSet ipc_peers = Network::shared_peers;
        // If this is a fabric enabled system, then we need to query all ranks in the
        // system for mappings (assume all GPUs are fabric enabled)
        if(gpus[0]->info->fabric_supported) {
          ipc_peers = Network::all_peers;
        }

        if(!ipc_peers.empty()) {
          log_cudaipc.info() << "Sending cuda ipc handles to " << ipc_peers.size()
                             << " peers";
          const Node &n = get_runtime()->nodes[Network::my_node_id];
          std::vector<CudaIpcResponseEntry> entries;
          // Find all the memories that are exportable via CUDA-IPC
          enumerate_ipc_entries(entries, n.memories);
          enumerate_ipc_entries(entries, n.ib_memories);

          // Broadcast all the IPC handles to all my peers
          // TODO: this could be replaced with ipc_mailbox
          size_t datalen = entries.size() * sizeof(entries[0]);
          ActiveMessage<CudaIpcImportRequest> amsg(
              ipc_peers,
              ActiveMessage<CudaIpcImportRequest>::recommended_max_payload(
                  ipc_peers, entries.data(), HOST_NAME_MAX + datalen, 1, datalen, true));
#if !defined(REALM_IS_WINDOWS)
          amsg->hostid = gethostid();
          char hostname[HOST_NAME_MAX];
          gethostname(hostname, sizeof(hostname));
          amsg.add_payload(hostname, sizeof(hostname));
#endif
          amsg->count = entries.size();
          amsg.add_payload(entries.data(), datalen);
          amsg.commit();

          log_cudaipc.debug() << "Sent " << entries.size() << " IPC entries";

          {
            // Wait for all the ipc_peers to send their IPC handles back to us
            AutoLock<> al(cudaipc_mutex);
            log_cudaipc.debug() << "Waiting for cudaipc responses...";

            cudaipc_responses_received.store_release(ipc_peers.size());
            initialization_complete.store_release(true);
            cudaipc_condvar.broadcast();
            while(cuda_module_singleton->cudaipc_responses_received.load_acquire() != 0) {
              cudaipc_condvar.wait();
            }
          }
        }
      }

      // now actually let each GPU make its channels
      for(GPU *gpu : gpus) {
        gpu->create_dma_channels(runtime);
      }

      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void CudaModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // if a module has to do cleanup that involves sending messages to other
    //  nodes, this must be done in the pre-detach cleanup
    void CudaModule::pre_detach_cleanup(void) {}

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void CudaModule::cleanup(void)
    {
      // clean up worker(s)
      if(shared_worker) {
#ifdef DEBUG_REALM
	shared_worker->shutdown_work_item();
#endif
	if(config->cfg_use_worker_threads)
	  shared_worker->shutdown_background_thread();

	delete shared_worker;
	shared_worker = 0;
      }
      for(std::map<GPU *, GPUWorker *>::iterator it = dedicated_workers.begin();
	  it != dedicated_workers.end();
	  it++) {
	GPUWorker *worker = it->second;

#ifdef DEBUG_REALM
	worker->shutdown_work_item();
#endif
	if(config->cfg_use_worker_threads)
	  worker->shutdown_background_thread();

	delete worker;
      }
      dedicated_workers.clear();

      // and clean up anything that was needed for the replicated heap
      runtime->repl_heap.remove_listener(rh_listener);

      for(std::vector<GPU *>::iterator it = gpus.begin();
	  it != gpus.end();
	  it++) {
#ifdef REALM_USE_CUDART_HIJACK
        GlobalRegistrations::remove_gpu_context(*it);
#endif
	delete *it;
      }
      gpus.clear();
      
      Module::cleanup();
    }

    CUstream_st *CudaModule::get_task_cuda_stream()
    {
      // if we're not in a gpu task, this'll be null
      if(ThreadLocal::current_gpu_stream)
	return ThreadLocal::current_gpu_stream->get_stream();
      else
	return 0;
    }

    void CudaModule::set_task_ctxsync_required(bool is_required)
    {
      // if we're not in a gpu task, setting this will have no effect
      ThreadLocal::context_sync_required = (is_required ? 1 : 0);
    }

    static void CUDA_CB event_trigger_callback(void *userData) {
      UserEvent realm_event;
      realm_event.id = reinterpret_cast<Realm::Event::id_t>(userData);
      realm_event.trigger();
    }

    Event CudaModule::make_realm_event(CUevent_st *cuda_event)
    {
      CUresult res = CUDA_DRIVER_FNPTR(cuEventQuery)(cuda_event);
      if(res == CUDA_SUCCESS) {
        // This CUDA event is already completed, no need to create a new event.
        return Event::NO_EVENT;
      } else if(res != CUDA_ERROR_NOT_READY) {
        CHECK_CU(res);
      }
      UserEvent realm_event = UserEvent::create_user_event();
      bool free_stream = false;
      CUstream cuda_stream = 0;
      if(ThreadLocal::current_gpu_stream != nullptr) {
        cuda_stream = ThreadLocal::current_gpu_stream->get_stream();
      } else {
        // Create a temporary stream to push the signaling onto.  This will ensure there's
        // no direct dependency on the signaling other than the event
        CHECK_CU(CUDA_DRIVER_FNPTR(cuStreamCreate)(&cuda_stream, CU_STREAM_NON_BLOCKING));
        free_stream = true;
      }
      CHECK_CU(CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(cuda_stream, cuda_event,
                                                    CU_EVENT_WAIT_DEFAULT));
      CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchHostFunc)(
          cuda_stream, event_trigger_callback, reinterpret_cast<void *>(realm_event.id)));
      if(free_stream) {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuStreamDestroy)(cuda_stream));
      }
      
      return realm_event;
    }

    Event CudaModule::make_realm_event(CUstream_st *cuda_stream)
    {
      CUresult res = CUDA_DRIVER_FNPTR(cuStreamQuery)(cuda_stream);
      if (res == CUDA_SUCCESS) {
        // This CUDA stream is already completed, no need to create a new event.
        return Event::NO_EVENT;
      }
      else if (res != CUDA_ERROR_NOT_READY) {
        CHECK_CU(res);
      }
      UserEvent realm_event = UserEvent::create_user_event();
      CHECK_CU(CUDA_DRIVER_FNPTR(cuLaunchHostFunc)(
          cuda_stream, event_trigger_callback,
          reinterpret_cast<void *>(realm_event.id)));
      return realm_event;
    }

    bool CudaModule::get_cuda_device_uuid(Processor p, Uuid *uuid) const
    {
      for(const GPU *gpu : gpus) {
        if((gpu->proc->me) == p) {
          static_assert(sizeof(CUuuid) == sizeof(char) * UUID_SIZE,
                        "UUID_SIZE is not set correctly");
          memcpy(uuid, &(gpu->info->uuid), sizeof(CUuuid));
          return true;
        }
      }
      // can not find the Processor p, so return false
      return false;
    }

    bool CudaModule::get_cuda_device_id(Processor p, int *device) const
    {
      for(const GPU *gpu : gpus) {
        if((gpu->proc->me) == p) {
          *device = gpu->info->index;
          return true;
        }
      }
      // can not find the Processor p, so return false
      return false;
    }

#ifdef REALM_USE_CUDART_HIJACK
    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredFunction

    RegisteredFunction::RegisteredFunction(const FatBin *_fat_bin, const void *_host_fun,
					   const char *_device_fun)
      : fat_bin(_fat_bin), host_fun(_host_fun), device_fun(_device_fun)
    {}
     
    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredVariable

    RegisteredVariable::RegisteredVariable(const FatBin *_fat_bin, const void *_host_var,
					   const char *_device_name, bool _external,
					   int _size, bool _constant, bool _global,
                                           bool _managed)
      : fat_bin(_fat_bin), host_var(_host_var), device_name(_device_name),
	external(_external), size(_size), constant(_constant), global(_global),
        managed(_managed)
    {}


    ////////////////////////////////////////////////////////////////////////
    //
    // class GlobalRegistrations

    GlobalRegistrations::GlobalRegistrations(void)
    {}

    GlobalRegistrations::~GlobalRegistrations(void)
    {
      delete_container_contents(variables);
      delete_container_contents(functions);
      // we don't own fat binary pointers, but we can forget them
      fat_binaries.clear();
    }

    /*static*/ GlobalRegistrations& GlobalRegistrations::get_global_registrations(void)
    {
      static GlobalRegistrations reg;
      return reg;
    }

    // called by a GPU when it has created its context - will result in calls back
    //  into the GPU for any modules/variables/whatever already registered
    /*static*/ void GlobalRegistrations::add_gpu_context(GPU *gpu)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      // add this gpu to the list
      assert(g.active_gpus.count(gpu) == 0);
      g.active_gpus.insert(gpu);

      // and now tell it about all the previous-registered stuff
      for(std::vector<FatBin *>::iterator it = g.fat_binaries.begin();
	  it != g.fat_binaries.end();
	  it++)
	gpu->register_fat_binary(*it);

      for(std::vector<RegisteredVariable *>::iterator it = g.variables.begin();
	  it != g.variables.end();
	  it++)
	gpu->register_variable(*it);

      for(std::vector<RegisteredFunction *>::iterator it = g.functions.begin();
	  it != g.functions.end();
	  it++)
	gpu->register_function(*it);
    }

    /*static*/ void GlobalRegistrations::remove_gpu_context(GPU *gpu)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      assert(g.active_gpus.count(gpu) > 0);
      g.active_gpus.erase(gpu);
    }

    // called by __cuda(un)RegisterFatBinary
    /*static*/ void GlobalRegistrations::register_fat_binary(FatBin *fatbin)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      // add the fat binary to the list and tell any gpus we know of about it
      g.fat_binaries.push_back(fatbin);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_fat_binary(fatbin);
    }

    /*static*/ void GlobalRegistrations::unregister_fat_binary(FatBin *fatbin)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      // remove the fatbin from the list - don't bother telling gpus
      std::vector<FatBin *>::iterator it = g.fat_binaries.begin();
      while(it != g.fat_binaries.end())
	if(*it == fatbin)
	  it = g.fat_binaries.erase(it);
	else
	  it++;
    }

    // called by __cudaRegisterVar
    /*static*/ void GlobalRegistrations::register_variable(RegisteredVariable *var)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      // add the variable to the list and tell any gpus we know
      g.variables.push_back(var);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_variable(var);
    }

    // called by __cudaRegisterFunction
    /*static*/ void GlobalRegistrations::register_function(RegisteredFunction *func)
    {
      GlobalRegistrations& g = get_global_registrations();

      AutoLock<> al(g.mutex);

      // add the function to the list and tell any gpus we know
      g.functions.push_back(func);

      for(std::set<GPU *>::iterator it = g.active_gpus.begin();
	  it != g.active_gpus.end();
	  it++)
	(*it)->register_function(func);
    }
#endif

    ////////////////////////////////////////////////////////////////////////
    //
    // struct CudaIpcImportRequest

    /*static*/ void CudaIpcImportRequest::handle_message(NodeID sender,
                                                         const CudaIpcImportRequest &args,
                                                         const void *data, size_t datalen)
    {
      const CudaIpcResponseEntry *entries = nullptr;
      assert(cuda_module_singleton != nullptr);
      bool allow_ipc = true;

      log_cudaipc.debug() << "IPC request from " << sender;

      {
        // Make sure initialization of the cuda module is complete before servicing the
        // active message. This will always be skipped, except in the rare case the
        // network module is initialized before the cuda module.
        AutoLock<> al(cuda_module_singleton->cudaipc_mutex);
        log_cudaipc.debug(
            "Waiting for cuda module initialization before processing IPC request");
        while(!cuda_module_singleton->initialization_complete.load_acquire()) {
          cuda_module_singleton->cudaipc_condvar.wait();
        }
        log_cudaipc.debug("Module initialized, processing IPC request");
      }

      if(args.count == 0) {
        log_cudaipc.info("Sender sent no entries to import, skipping import...");
        goto Done;
      }

#if !defined(REALM_IS_WINDOWS)
      // If shared_peers_use_network_module == false, we can safely rely on
      //  the Network::shared_peers created by ipc mailbox. Otherwise,
      //  the shared_peers could be all_peers, therefore, we try to use
      //  hostid and hostname to check if two peers are on the same physical
      //  node. However, the hostid and hostname are likely the same on containers
      //  that are not cuda ipc capable. This can lead to a failure when opening
      //  the ipc handle from another physical node, or it could succeed and we
      //  could end up getting corrupted results.
      if(Realm::get_runtime()->shared_peers_use_network_module) {
        char local_hostname[HOST_NAME_MAX] = {0};
        assert(datalen > sizeof(local_hostname));
        gethostname(local_hostname, sizeof(local_hostname));
        if((strncmp(local_hostname, static_cast<const char *>(data),
                    sizeof(local_hostname)) != 0) &&
           (args.hostid != gethostid())) {
          log_cudaipc.info() << "Sender " << sender
                             << " is not an ipc-capable node, skipping ipc import";
          allow_ipc = false;
        }
      }
      data = static_cast<const char *>(data) + HOST_NAME_MAX;
      datalen -= HOST_NAME_MAX;
#endif

      entries = static_cast<const CudaIpcResponseEntry *>(data);
      assert(datalen == (args.count * sizeof(CudaIpcResponseEntry)));

      // For each entry
      //  if type == fabric: find a gpu with the same clique/cluster, import it and peer
      //  access for everyone with the same clique/cluster
      //    for each gpu: add an ipc mapping
      //  if type == ipc and allow IPC:
      //    for each gpu: open_ipc(), add an ipc mapping.
      for(GPU *gpu : cuda_module_singleton->gpus) {
        for(size_t i = 0; i < args.count; i++) {
          const CudaIpcResponseEntry &entry = entries[i];
          GPU::CudaIpcMapping mapping;
          GPUAllocation *alloc = nullptr;

          mapping.src_gpu = nullptr;
          mapping.owner = sender;
          mapping.mem = entry.mem;

          for(GPU *mapping_gpu : cuda_module_singleton->gpus) {
            if(memcmp(&mapping_gpu->info->uuid, &entries[i].src_gpu_uuid,
                      sizeof(mapping_gpu->info->uuid)) == 0) {
              mapping.src_gpu = mapping_gpu;
            }
          }

          switch(entry.type) {
          case CUDA_IPC_RESPONSE_TYPE_IPC:
            if(allow_ipc) {
              alloc = GPUAllocation::open_ipc(gpu, entry.data.ipc_handle);
            }
            break;
#if CUDA_VERSION >= 12030
          case CUDA_IPC_RESPONSE_TYPE_FABRIC:
            // TODO(cperry): Seperate this out such that we can leverage the same VA
            // across all GPUs that can import this memory to reduce the import time cost
            if((gpu->info->fabric_clique == entry.data.fabric.clique_id) &&
               (memcmp(gpu->info->fabric_uuid.bytes, entry.data.fabric.cluster_uuid.bytes,
                       sizeof(gpu->info->fabric_uuid.bytes)) == 0)) {
              alloc =
                  GPUAllocation::open_fabric(gpu, entry.data.fabric.handle, entry.size,
                                             false, false); // Should be allow_ipc
            }
            break;
#endif
          default:
            log_cudaipc.fatal("Invalid ipc entry type received: %d",
                              static_cast<int>(entry.type));
            break;
          }

          if(alloc != nullptr) {
            AutoLock<> al(cuda_module_singleton->cudaipc_mutex);

            if(alloc->get_hptr() != nullptr) {
              gpu->pinned_sysmems.insert(entry.mem);
            }

            mapping.local_base = alloc->get_dptr();
            mapping.address_offset = entry.base_ptr - alloc->get_dptr();

            gpu->cudaipc_mappings.push_back(mapping);

            // do we have a stream for this target?
            if(gpu->cudaipc_streams.count(sender) == 0) {
              AutoGPUContext ag(gpu);
              gpu->cudaipc_streams[sender] = new GPUStream(gpu, gpu->worker);
            }
          } else {
            log_cudaipc.info("Unable to import given memory from sender %u",
                             static_cast<unsigned>(sender));
          }
        }
      }

    Done:
    {
      // Count the number of peers that have been received and signal to continue
      // initialization when all of them have been recieved.  This needs to be done for
      // every message received in order to unblock module initialization
      AutoLock<> al(cuda_module_singleton->cudaipc_mutex);
      if((cuda_module_singleton->cudaipc_responses_received.fetch_sub_acqrel(1)) == 1) {
        log_cudaipc.debug() << "Signalling completion!";
        cuda_module_singleton->cudaipc_condvar.signal();
      }
    }
    }

    ActiveMessageHandlerReg<CudaIpcImportRequest> cuda_ipc_request_handler;

    GPUAllocation::GPUAllocation(GPUAllocation &&other) noexcept
      : gpu(other.gpu)
      , dev_ptr(other.dev_ptr)
      , host_ptr(other.host_ptr)
      , size(other.size)
      , deleter(other.deleter)
#if CUDA_VERSION >= 11000
      , mmap_handle(other.mmap_handle)
      , owns_va(other.owns_va)
#endif
      , has_ipc_handle(other.has_ipc_handle)
      , ipc_handle(other.ipc_handle)
    {
      other.gpu = nullptr;
      other.dev_ptr = 0;
      other.host_ptr = nullptr;
      other.size = 0;
      other.deleter = nullptr;
#if CUDA_VERSION >= 11000
      other.mmap_handle = 0;
      other.owns_va = false;
#endif
      other.has_ipc_handle = false;
    }

    GPUAllocation &GPUAllocation::operator=(GPUAllocation &&other) noexcept
    {
      if(this != &other) {
        if(deleter != nullptr) {
          deleter(*this);
        }

        gpu = other.gpu;
        dev_ptr = other.dev_ptr;
        host_ptr = other.host_ptr;
        size = other.size;
        deleter = other.deleter;
#if CUDA_VERSION >= 11000
        mmap_handle = other.mmap_handle;
        owns_va = other.owns_va;
#endif
        has_ipc_handle = other.has_ipc_handle;
        ipc_handle = other.ipc_handle;

        other.gpu = nullptr;
        other.dev_ptr = 0;
        other.host_ptr = nullptr;
        other.size = 0;
        other.deleter = nullptr;
#if CUDA_VERSION >= 11000
        other.mmap_handle = 0;
        other.owns_va = false;
#endif
        other.has_ipc_handle = false;
      }
      return *this;
    }

    GPUAllocation::~GPUAllocation()
    {
      if(deleter != nullptr) {
        deleter(*this);
      }
    }

    OsHandle GPUAllocation::get_os_handle() const
    {
      OsHandle handle = Realm::INVALID_OS_HANDLE;
#if CUDA_VERSION >= 11000
      if(mmap_handle != 0) {
        CUresult res = CUDA_SUCCESS;
#if defined(REALM_ON_WINDOWS)
        CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_WIN32;
#else
        CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
        res = CUDA_DRIVER_FNPTR(cuMemExportToShareableHandle)(&handle, mmap_handle,
                                                              handle_type, 0);
        if(res == CUDA_SUCCESS) {
          REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemExportToShareableHandle", res);
        }
      }
#endif
      return handle;
    }

#if CUDA_VERSION >= 12030
    bool GPUAllocation::get_fabric_handle(CUmemFabricHandle &handle) const
    {
      CUresult res = CUDA_SUCCESS;
      if(mmap_handle == 0) {
        return false;
      }
      res = CUDA_DRIVER_FNPTR(cuMemExportToShareableHandle)(&handle, mmap_handle,
                                                            CU_MEM_HANDLE_TYPE_FABRIC, 0);
      if(res != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemExportToShareableHandle", res);
      }
      return res == CUDA_SUCCESS;
    }
#endif

    /*static*/ void *GPUAllocation::get_win32_shared_attributes(void)
    {
#if defined(REALM_IS_WINDOWS)
      // If we require a win32 handle type, then set the security descriptor
      static OBJECT_ATTRIBUTES objAttributes;
      static bool objAttributesConfigured = false;

      // TODO: Should put this in a std::call_once?  Current uses are only
      // single-threaded.
      if(!objAttributesConfigured) {
        // This default security descriptor is fairly permissive and should be tuned
        // properly
        const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
        PSECURITY_DESCRIPTOR secDesc;

        if(!ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1,
                                                                 &secDesc, NULL)) {
          log_gpu.info("Failed to create security descriptor: (%d)", GetLastError());
          return nullptr;
        }

        InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);
        objAttributesConfigured = true;
      }
      return &objAttributes;
#else
      return nullptr;
#endif
    }

    /*static*/ GPUAllocation *
    GPUAllocation::allocate_dev(GPU *gpu, size_t size, bool peer_enabled, bool shareable)
    {
      GPUAllocation alloc;
      AutoGPUContext agc(gpu);
      alloc.size = size;
      alloc.gpu = gpu;
      alloc.deleter = &GPUAllocation::cuda_malloc_free;
      if(CUDA_DRIVER_FNPTR(cuMemAlloc)(&alloc.dev_ptr, alloc.size) != CUDA_SUCCESS) {
        return nullptr;
      }

      if(shareable) {
        alloc.has_ipc_handle = true;
        CHECK_CU(CUDA_DRIVER_FNPTR(cuIpcGetMemHandle)(&alloc.ipc_handle, alloc.dev_ptr));
      }

      return &gpu->add_allocation(std::move(alloc));
    }

#if CUDA_VERSION >= 11000
    /*static*/ GPUAllocation *
    GPUAllocation::allocate_mmap(GPU *gpu, const CUmemAllocationProp &prop, size_t size,
                                 CUdeviceptr vaddr /*= 0*/, bool peer_enabled /*= true*/)
    {
      CUresult ret = CUDA_SUCCESS;
      GPUAllocation alloc;
#if CUDA_VERSION >= 12030
      bool map_host = prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA;
#else
      bool map_host = false;
#endif

      size = GPUAllocation::align_size(prop, size);

      ret = CUDA_DRIVER_FNPTR(cuMemCreate)(&alloc.mmap_handle, size, &prop, 0);
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemCreate", ret);
        return nullptr;
      }

      ret = alloc.map_allocation(gpu, alloc.mmap_handle, size, vaddr, 0, peer_enabled,
                                 map_host);
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemMap", ret);
        return nullptr;
      }

      return &gpu->add_allocation(std::move(alloc));
    }
#endif

    /*static*/ GPUAllocation *GPUAllocation::allocate_host(GPU *gpu, size_t size,
                                                           bool peer_enabled /*= true*/,
                                                           bool shareable /*= true*/,
                                                           bool same_va /*= true*/)
    {
      CUresult ret = CUDA_SUCCESS;
      GPUAllocation alloc;
      AutoGPUContext ac(gpu);
      unsigned int cuda_flags = CU_MEMHOSTALLOC_DEVICEMAP;

      if(shareable) {
#if CUDA_VERSION >= 12030
        int numa = -1;
        GPUAllocation *shared_alloc = nullptr;
        if((CUDA_SUCCESS ==
            CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(
                &numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, gpu->info->device)) &&
           (numa >= 0)) {
          CUmemAllocationProp mem_prop;
          memset(&mem_prop, 0, sizeof(mem_prop));
          mem_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
          mem_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
          mem_prop.location.id = numa;
          mem_prop.win32HandleMetaData = GPUAllocation::get_win32_shared_attributes();

          // Try fabric first.  This can fail for a number of reasons, but most commonly,
          // it's because the application isn't bound to an IMEX channel
          if(gpu->info->fabric_supported) {
#if CUDA_VERSION >= 12030
            mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#endif
            shared_alloc = allocate_mmap(gpu, mem_prop, size, 0, peer_enabled);
            if(shared_alloc != nullptr) {
              return shared_alloc;
            }
          }

          // Fallback to OS handle.
#if defined(REALM_ON_WINDOWS)
          mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
#else
          mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

          shared_alloc = allocate_mmap(gpu, mem_prop, size, 0, peer_enabled);
          if(shared_alloc != nullptr) {
            return shared_alloc;
          }
          // Fallback to non-shareable if, for some reason, the platform cannot allocate
          // the memory
        }
#endif
      }

      if(peer_enabled) {
        cuda_flags |= CU_MEMHOSTALLOC_PORTABLE;
      }
      alloc.size = size;
      alloc.gpu = gpu;
      alloc.deleter = &GPUAllocation::cuda_malloc_host_free;

      ret = CUDA_DRIVER_FNPTR(cuMemHostAlloc)(&alloc.host_ptr, size, cuda_flags);
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemHostAlloc", ret);
        return nullptr;
      }
      CHECK_CU(CUDA_DRIVER_FNPTR(cuMemHostGetDevicePointer)(&alloc.dev_ptr,
                                                            alloc.host_ptr, 0));

      return &gpu->add_allocation(std::move(alloc));
    }

    /*static*/ GPUAllocation *GPUAllocation::allocate_managed(GPU *gpu, size_t size)
    {
      CUresult ret = CUDA_SUCCESS;
      GPUAllocation alloc;
      AutoGPUContext ac(gpu);
      alloc.size = size;
      alloc.gpu = gpu;
      alloc.deleter = &GPUAllocation::cuda_malloc_free;

      ret = CUDA_DRIVER_FNPTR(cuMemAllocManaged)(&alloc.dev_ptr, size,
                                                 CU_MEM_ATTACH_GLOBAL);
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemAllocManaged", ret);
        return nullptr;
      }
      alloc.host_ptr = reinterpret_cast<void *>(alloc.dev_ptr);

      return &gpu->add_allocation(std::move(alloc));
    }

    /*static*/ GPUAllocation *
    GPUAllocation::register_allocation(GPU *gpu, void *ptr, size_t size,
                                       bool peer_enabled /*= true*/)
    {
      CUresult ret = CUDA_SUCCESS;
      CUdeviceptr dev_ptr = 0;
      GPUAllocation alloc;
      AutoGPUContext ac(gpu);
      unsigned int cuda_flags = CU_MEMHOSTREGISTER_DEVICEMAP;

      if(peer_enabled) {
        cuda_flags |= CU_MEMHOSTREGISTER_PORTABLE;
      }

      alloc.size = size;
      alloc.gpu = gpu;
      alloc.deleter = &GPUAllocation::cuda_register_free;

      ret = CUDA_DRIVER_FNPTR(cuMemHostGetDevicePointer)(&dev_ptr, ptr, 0);
      if(ret != CUDA_SUCCESS) {
        // Not a CUDA registered address, so register it!
        ret = CUDA_DRIVER_FNPTR(cuMemHostRegister)(ptr, size, cuda_flags);
        if(ret != CUDA_SUCCESS) {
          REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemHostRegister", ret);
          return nullptr;
        }
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemHostGetDevicePointer)(&alloc.dev_ptr, ptr, 0));
        alloc.host_ptr = ptr;
        alloc.owns_va = true;
      } else {
        // This allocation is already registered, so check if the allocation range is
        // actually pinned, then track it without releasing it ourselves (weak_ptr
        // semantics)
        size_t registered_size = 0;
        CUdeviceptr base_addr = 0;
        void *values[] = {(void *)&base_addr, (void *)&size};
        CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                       CU_POINTER_ATTRIBUTE_RANGE_SIZE};
        ret = CUDA_DRIVER_FNPTR(cuPointerGetAttributes)(sizeof(attrs) / sizeof(attrs[0]),
                                                        attrs, values, alloc.dev_ptr);
        if(ret != CUDA_SUCCESS) {
          REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuPointerGetAttributes", ret);
          return nullptr;
        }
        if((registered_size - (alloc.dev_ptr - base_addr)) < size) {
          log_gpu.info()
              << "Requested registered memory is already mapped, but requested "
                 "size is too large";
          return nullptr;
        }
        alloc.dev_ptr = dev_ptr;
        alloc.host_ptr = ptr;
        alloc.owns_va = false; // Don't free this VA, as it is managed externally
      }

      return &gpu->add_allocation(std::move(alloc));
    }

    /*static*/ GPUAllocation *GPUAllocation::open_ipc(GPU *gpu,
                                                      const CUipcMemHandle &mem_hdl)
    {
      CUresult ret = CUDA_SUCCESS;
      GPUAllocation alloc;
      AutoGPUContext ac(gpu);
      alloc.deleter = &GPUAllocation::cuda_ipc_free;
      alloc.gpu = gpu;
      ret = CUDA_DRIVER_FNPTR(cuIpcOpenMemHandle)(&alloc.dev_ptr, mem_hdl,
                                                  CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
      if(ret != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuIpcOpenMemHandle", ret);
        return nullptr;
      }
      CHECK_CU(
          CUDA_DRIVER_FNPTR(cuMemGetAddressRange)(nullptr, &alloc.size, alloc.dev_ptr));

      return &gpu->add_allocation(std::move(alloc));
    }

    /*static*/ GPUAllocation *GPUAllocation::open_handle(GPU *gpu, OsHandle hdl,
                                                         size_t size,
                                                         bool peer_enabled /*= true*/)
    {
#if CUDA_VERSION >= 11000
      GPUAllocation alloc;
      CUmemGenericAllocationHandle cuda_hdl;
      CUmemAllocationProp mem_prop{};
      bool map_host = false;
      void *casted_handle = reinterpret_cast<void *>(static_cast<uintptr_t>(hdl));
#if defined(REALM_ON_WINDOWS)
      CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_WIN32;
#else
      CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

      CHECK_CU(CUDA_DRIVER_FNPTR(cuMemImportFromShareableHandle)(&cuda_hdl, casted_handle,
                                                                 handle_type));
      CHECK_CU(
          CUDA_DRIVER_FNPTR(cuMemGetAllocationPropertiesFromHandle)(&mem_prop, cuda_hdl));
#if CUDA_VERSION >= 12030
      map_host = (mem_prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA);
#endif

      if(alloc.map_allocation(gpu, cuda_hdl, size, 0, 0, peer_enabled, map_host) !=
         CUDA_SUCCESS) {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemRelease)(cuda_hdl));
        return nullptr;
      }

      alloc.mmap_handle = cuda_hdl;

      return &gpu->add_allocation(std::move(alloc));
#else
      // TODO Add in a path for supporting SharedMemoryInfo + cuMemHostRegister paths as a
      // fallback
      return nullptr;
#endif
    }

#if CUDA_VERSION >= 12030
    /*static*/ GPUAllocation *
    GPUAllocation::open_fabric(GPU *gpu, const CUmemFabricHandle &hdl, size_t size,
                               bool peer_enabled /*= true*/, bool is_local /* = false*/)
    {
      GPUAllocation alloc;
      CUmemGenericAllocationHandle cuda_hdl;
      CUmemAllocationProp mem_prop{};
      CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC;

      CHECK_CU(CUDA_DRIVER_FNPTR(cuMemImportFromShareableHandle)(
          &cuda_hdl, const_cast<void *>(reinterpret_cast<const void *>(&hdl)),
          handle_type));

      CHECK_CU(
          CUDA_DRIVER_FNPTR(cuMemGetAllocationPropertiesFromHandle)(&mem_prop, cuda_hdl));

      // Unfortunately HOST_NUMA memory cannot be mapped to the CPU if it exists on
      // another node, only to the GPU.

      if(alloc.map_allocation(
             gpu, cuda_hdl, size, 0, 0, peer_enabled,
             is_local && (mem_prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA)) !=
         CUDA_SUCCESS) {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemRelease)(cuda_hdl));
        return nullptr;
      }

      alloc.mmap_handle = cuda_hdl;

      return &gpu->add_allocation(std::move(alloc));
    }
#endif

#if CUDA_VERSION >= 11000
    CUresult GPUAllocation::map_allocation(GPU *gpu, CUmemGenericAllocationHandle handle,
                                           size_t size, CUdeviceptr vaddr /*= 0*/,
                                           size_t offset /*= 0*/,
                                           bool peer_enabled /*= false*/,
                                           bool map_host /*= false*/)
    {
      CUresult res = CUDA_SUCCESS;
      std::vector<CUmemAccessDesc> desc(1);

      this->gpu = gpu;
      this->size = size;
      dev_ptr = vaddr;
      owns_va = vaddr == 0;
      deleter = &GPUAllocation::cuda_memmap_free;

      if(vaddr == 0) {
        res = CUDA_DRIVER_FNPTR(cuMemAddressReserve)(&dev_ptr, size, 0ULL, 0ULL, 0ULL);
        if(res != CUDA_SUCCESS) {
          REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemAddressReserve", res);
          goto Done;
        }
      }

      res = CUDA_DRIVER_FNPTR(cuMemMap)(dev_ptr, size, 0, handle, offset);
      if(res != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemMap", res);
        goto Done;
      }

      desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      desc[0].location.id = gpu->info->index;
      if(peer_enabled) {
        size_t peer_offset = 1;
        desc.resize(gpu->info->peers.size() + 1);
        for(int peer_idx : gpu->info->peers) {
          desc[peer_offset].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          desc[peer_offset].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          desc[peer_offset].location.id = peer_idx;
          peer_offset++;
        }
      }
      if(map_host) {
#if CUDA_VERSION >= 12030
        // Map this to the CPU as well!
        desc.push_back({});
        desc.back().flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc.back().location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        desc.back().location.id = 0;
#else
        res = CUDA_ERROR_NOT_SUPPORTED;
        goto Done;
#endif // CUDA_VERSION >= 12000
      }

      res = CUDA_DRIVER_FNPTR(cuMemSetAccess)(dev_ptr, size, desc.data(), desc.size());
      if(res != CUDA_SUCCESS) {
        REPORT_CU_ERROR(Logger::LEVEL_INFO, "cuMemSetAccess", res);
        goto Done;
      }

      if(map_host) {
        host_ptr = reinterpret_cast<void *>(dev_ptr);
      }

    Done:
      if(res != CUDA_SUCCESS) {
        deleter(*this);
      }

      return res;
    }

    /*static*/ size_t GPUAllocation::align_size(const CUmemAllocationProp &prop,
                                                size_t size)
    {
      size_t granularity = 0;
      CHECK_CU(CUDA_DRIVER_FNPTR(cuMemGetAllocationGranularity)(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
      // Round up size to the recommended granularity
      return (size + granularity - 1) & ~(granularity - 1);
    }
#endif // CUDA_VERSION >= 11000

    /*static*/ void GPUAllocation::cuda_malloc_free(GPUAllocation &alloc)
    {
      if(alloc.dev_ptr != 0) {
        AutoGPUContext ac(alloc.gpu);
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemFree)(alloc.dev_ptr));
        alloc.host_ptr = nullptr;
        alloc.dev_ptr = 0;
      }
    }
    /*static*/ void GPUAllocation::cuda_malloc_host_free(GPUAllocation &alloc)
    {
      if(alloc.host_ptr != nullptr) {
        AutoGPUContext ac(alloc.gpu);
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemFreeHost)(alloc.host_ptr));
        alloc.host_ptr = nullptr;
        alloc.dev_ptr = 0;
      }
    }
    /*static*/ void GPUAllocation::cuda_register_free(GPUAllocation &alloc)
    {
      if((alloc.host_ptr != nullptr) && alloc.owns_va) {
        AutoGPUContext ac(alloc.gpu);
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemHostUnregister)(alloc.host_ptr));
        alloc.host_ptr = nullptr;
        alloc.dev_ptr = 0;
      }
    }
    /*static*/ void GPUAllocation::cuda_ipc_free(GPUAllocation &alloc)
    {
      if(alloc.dev_ptr != 0) {
        AutoGPUContext ac(alloc.gpu);
        CHECK_CU(CUDA_DRIVER_FNPTR(cuIpcCloseMemHandle)(alloc.dev_ptr));
        alloc.dev_ptr = 0;
      }
    }
#if CUDA_VERSION >= 11000
    /*static*/ void GPUAllocation::cuda_memmap_free(GPUAllocation &alloc)
    {
      if(alloc.mmap_handle != 0) {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemRelease)(alloc.mmap_handle));
        alloc.mmap_handle = 0;
      }
      if(alloc.dev_ptr != 0) {
        CHECK_CU(CUDA_DRIVER_FNPTR(cuMemUnmap)(alloc.dev_ptr, alloc.size));
        if(alloc.owns_va) {
          CHECK_CU(CUDA_DRIVER_FNPTR(cuMemAddressFree)(alloc.dev_ptr, alloc.size));
        }
        alloc.dev_ptr = 0;
        alloc.host_ptr = nullptr;
      }
    }
#endif

  }; // namespace Cuda
}; // namespace Realm
