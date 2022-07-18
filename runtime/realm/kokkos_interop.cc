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

// Realm+Kokkos interop support

#include "realm/kokkos_interop.h"

#include "realm/mutex.h"
#include "realm/processor.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_internal.h"

#include <cuda_runtime.h>
#endif

// some compilers (e.g. clang++ 10) will hide symbols that you want to be
//  public if any template parameters have hidden visibility, even if they
//  come from an "external" header file...
// work around this by declaring any of the kokkos execution space types
//  that we might use below (we don't get the defines to say whether we
//  actually use them until we include Kokkos_Core.hpp, at which point it's
//  too late to try to change the visibility)
namespace Kokkos {
  class REALM_PUBLIC_API Serial;
  class REALM_PUBLIC_API OpenMP;
  class REALM_PUBLIC_API Cuda;
};

#include <Kokkos_Core.hpp>

// during the development of Kokkos 3.7.00, initialization data structures
//  were changed - detect the presence of a new header (included indirectly
//  via Kokkos_Core.hpp)
#ifdef KOKKOS_INITIALIZATION_SETTINGS_HPP
  #define REALM_USE_KOKKOS_INITIALIZATION_SETTINGS
#endif

#include <stdlib.h>

namespace Realm {

  Logger log_kokkos("kokkos");

  namespace KokkosInterop {

    bool is_kokkos_cuda_enabled(void)
    {
#ifdef KOKKOS_ENABLE_CUDA
      return true;
#else
      return false;
#endif
    }

    bool is_kokkos_openmp_enabled(void)
    {
#ifdef KOKKOS_ENABLE_OPENMP
      return true;
#else
      return false;
#endif
    }

    class KokkosInternalTask : public InternalTask {
    public:
      KokkosInternalTask()
	: done(false), condvar(mutex) {}

      void mark_done()
      {
	AutoLock<> al(mutex);
	done = true;
	condvar.broadcast();
      }

      void wait_done()
      {
	AutoLock<> al(mutex);
	while(!done) condvar.wait();
      }

      bool done;
      Mutex mutex;
      Mutex::CondVar condvar;
    };

#ifdef KOKKOS_ENABLE_OPENMP
    std::vector<ProcessorImpl *> kokkos_omp_procs;
    
    class KokkosOpenMPInitializer : public KokkosInternalTask {
    public:
      virtual void execute_on_processor(Processor p)
      {
	log_kokkos.info() << "doing openmp init on proc " << p;
#ifdef REALM_USE_KOKKOS_INITIALIZATION_SETTINGS
        Kokkos::InitializationSettings init_settings;
	init_settings.set_num_threads(-1); // todo - get from proc
        Kokkos::OpenMP::impl_initialize(init_settings);
#else
	int thread_count = -1; // todo - get from proc
	Kokkos::OpenMP::impl_initialize(thread_count);
#endif
	mark_done();
      }
    };
    
    class KokkosOpenMPFinalizer : public KokkosInternalTask {
    public:
      virtual void execute_on_processor(Processor p)
      {
	log_kokkos.info() << "doing openmp finalize on proc " << p;
	Kokkos::OpenMP::impl_finalize();
	mark_done();
      }
    };
#endif

#ifdef KOKKOS_ENABLE_CUDA
    std::vector<ProcessorImpl *> kokkos_cuda_procs;

    Mutex cuda_instance_map_mutex;
    std::map<std::pair<Processor, cudaStream_t>, Kokkos::Cuda *> cuda_instance_map;

    class KokkosCudaInitializer : public KokkosInternalTask {
    public:
      virtual void execute_on_processor(Processor p)
      {
	log_kokkos.info() << "doing cuda init on proc " << p;

	ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
	assert(impl->kind == Processor::TOC_PROC);
	Cuda::GPUProcessor *gpu = checked_cast<Cuda::GPUProcessor *>(impl);

#ifdef REALM_USE_KOKKOS_INITIALIZATION_SETTINGS
        Kokkos::InitializationSettings init_settings;
        init_settings.set_device_id(gpu->gpu->info->index);
        init_settings.set_num_devices(1);
        Kokkos::Cuda::impl_initialize(init_settings);
#else
	int cuda_device_id = gpu->gpu->info->index;
	int num_instances = 1; // unused in kokkos?

	Kokkos::Cuda::impl_initialize(Kokkos::Cuda::SelectDevice(cuda_device_id),
				      num_instances);
#endif
	{
	  // some init is deferred until an instance is created
	  Kokkos::Cuda dummy;
	}
	mark_done();
      }
    };
    
    class KokkosCudaFinalizer : public KokkosInternalTask {
    public:
      virtual void execute_on_processor(Processor p)
      {
	log_kokkos.info() << "doing cuda finalize on proc " << p;

	// delete all the cuda instances from this proc that we've cached
	for(std::map<std::pair<Processor, cudaStream_t>, Kokkos::Cuda *>::iterator it = cuda_instance_map.begin();
	    it != cuda_instance_map.end();
	    ++it)
	  if(it->first.first == p)
	    delete it->second;

	Kokkos::Cuda::impl_finalize();
	mark_done();
      }
    };
#endif

    void kokkos_initialize(const std::vector<ProcessorImpl *>& local_procs)
    {
      // use Kokkos::Impl::{pre,post}_initialize to allow us to do our own
      //  execution space initialization
#ifdef REALM_USE_KOKKOS_INITIALIZATION_SETTINGS
      Kokkos::InitializationSettings kokkos_init_args;
#else
      Kokkos::InitArguments kokkos_init_args;
#endif
      log_kokkos.info() << "doing general pre-initialization";
      Kokkos::Impl::pre_initialize(kokkos_init_args);

#ifdef KOKKOS_ENABLE_SERIAL
      // nothing thread-specific for serial execution space, so just call it
      //  here
#ifdef REALM_USE_KOKKOS_INITIALIZATION_SETTINGS
      Kokkos::Serial::impl_initialize(kokkos_init_args);
#else
      Kokkos::Serial::impl_initialize();
#endif
#endif

#ifdef KOKKOS_ENABLE_OPENMP
      // need to initialize the Kokkos openmp execution space...
#ifdef REALM_USE_OPENMP
      // ... from an openmp proc
      {
	// if we're providing openmp goodness, set environment variable to shut
	//  off some kokkos warnings that don't mean anything
	setenv("OMP_PROC_BIND", "false", 0 /*!overwrite*/);

	size_t count = 0;
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    ++it)
	  if((*it)->kind == Processor::OMP_PROC) {
	    count++;
	    if(count > 1) continue; // we'll complain below
	    KokkosOpenMPInitializer ompinit;
	    (*it)->add_internal_task(&ompinit);
	    ompinit.wait_done();
	    kokkos_omp_procs.push_back(*it);
	  }
	if(count != 1) {
	  log_kokkos.fatal() << "Kokkos OpenMP support requires exactly 1 omp proc (found " << count << ") - suggest -ll:ocpu 1 -ll:onuma 0";
	  abort();
	}
      }
#else
      // ... from normal CPU procs since we don't have anything better
      {
	size_t count = 0;
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    ++it)
	  if((*it)->kind == Processor::LOC_PROC) {
	    count++;
	    if(count > 1) continue; // we'll complain below
	    KokkosOpenMPInitializer ompinit;
	    (*it)->add_internal_task(&ompinit);
	    ompinit.wait_done();
	    kokkos_omp_procs.push_back(*it);
	  }
	if(count != 1) {
	  log_kokkos.fatal() << "Kokkos OpenMP support without realm OpenMP requires exactly 1 cpu proc (found " << count << ") - suggest -ll:cpu 1";
	  abort();
	}
      }
#endif
#endif

#ifdef KOKKOS_ENABLE_CUDA
      {
	size_t count = 0;
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    ++it)
	  if((*it)->kind == Processor::TOC_PROC) {
	    count++;
	    if(count > 1) continue; // we'll complain below
	    KokkosCudaInitializer cudainit;
	    (*it)->add_internal_task(&cudainit);
	    cudainit.wait_done();
	    kokkos_cuda_procs.push_back(*it);
	  }
	if(count != 1) {
	  log_kokkos.fatal() << "Kokkos Cuda support requires exactly 1 gpu proc (found " << count << ") - suggest -ll:gpu 1";
	  abort();
	}
      }
#endif

      // TODO: warn if Kokkos has other execution spaces enabled that we're not
      //  willing/able to initialize?

      log_kokkos.info() << "doing general post-initialization";
      Kokkos::Impl::post_initialize(kokkos_init_args);
    }
    
    void kokkos_finalize(const std::vector<ProcessorImpl *>& local_procs)
    {
      // per processor finalization on the correct threads
#ifdef KOKKOS_ENABLE_OPENMP
      for(std::vector<ProcessorImpl *>::const_iterator it = kokkos_omp_procs.begin();
	  it != kokkos_omp_procs.end();
	  ++it)
	{
	  KokkosOpenMPFinalizer ompfinal;
	  (*it)->add_internal_task(&ompfinal);
	  ompfinal.wait_done();
	}
#endif
      
#ifdef KOKKOS_ENABLE_CUDA
      for(std::vector<ProcessorImpl *>::const_iterator it = kokkos_cuda_procs.begin();
	  it != kokkos_cuda_procs.end();
	  ++it)
	{
	  KokkosCudaFinalizer cudafinal;
	  (*it)->add_internal_task(&cudafinal);
	  cudafinal.wait_done();
	}
#endif
      
      log_kokkos.info() << "doing general finalization";
      Kokkos::finalize();
    }

  };

  // execution space instance conversions from processor.h
#ifdef KOKKOS_ENABLE_SERIAL
  template <>
  Processor::KokkosExecInstance::operator Kokkos::Serial() const
  {
    return Kokkos::Serial();
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  template <>
  Processor::KokkosExecInstance::operator Kokkos::OpenMP() const
  {
    return Kokkos::OpenMP();
  }
#endif

#ifdef KOKKOS_ENABLE_CUDA
  template <>
  Processor::KokkosExecInstance::operator Kokkos::Cuda() const
  {
#ifdef REALM_USE_CUDA
    ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
    assert(impl->kind == Processor::TOC_PROC);
    Cuda::GPUProcessor *gpu = checked_cast<Cuda::GPUProcessor *>(impl);
    cudaStream_t stream = gpu->gpu->get_null_task_stream()->get_stream();
    log_kokkos.info() << "handing back stream " << stream;
    Kokkos::Cuda *inst = 0;
    {
      AutoLock<> al(KokkosInterop::cuda_instance_map_mutex);
      std::pair<Processor, cudaStream_t> key(p, stream);
      std::map<std::pair<Processor, cudaStream_t>, Kokkos::Cuda *>::iterator it = KokkosInterop::cuda_instance_map.find(key);
      if(it != KokkosInterop::cuda_instance_map.end()) {
	inst = it->second;
      } else {
	// creating a Kokkos::Cuda instance does some blocking calls, but we're
	//  not re-entrant here, so enable the scheduler lock
	Processor::enable_scheduler_lock();
	inst = new Kokkos::Cuda(stream);
	Processor::disable_scheduler_lock();
	KokkosInterop::cuda_instance_map[key] = inst;
      }
    }
    return *inst;
#else
    // we're oblivious to the application's use of CUDA
    return Kokkos::Cuda();
#endif
  }
#endif

};
