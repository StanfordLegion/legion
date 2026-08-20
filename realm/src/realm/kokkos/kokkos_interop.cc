/*
 * Copyright 2026 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
 * SPDX-License-Identifier: Apache-2.0
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

#include "realm/kokkos/kokkos_interop.h"

#include "realm/mutex.h"
#include "realm/processor.h"
#include "realm/runtime_impl.h"
#include "realm/logging.h"

#ifdef REALM_USE_OPENMP
#include "realm/openmp/openmp_internal.h"
#endif

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_internal.h"

#include <cuda_runtime.h>
#endif

#ifdef REALM_USE_HIP
#include "realm/hip/hip_internal.h"

#include <hip/hip_runtime.h>
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
  class REALM_PUBLIC_API HIP;
}; // namespace Kokkos

#include <Kokkos_Core.hpp>

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

    bool is_kokkos_hip_enabled(void)
    {
#ifdef KOKKOS_ENABLE_HIP
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
        : done(false)
        , condvar(mutex)
      {}

      void mark_done()
      {
        AutoLock<> al(mutex);
        done = true;
        condvar.broadcast();
      }

      void wait_done()
      {
        AutoLock<> al(mutex);
        while(!done)
          condvar.wait();
      }

      bool done;
      Mutex mutex;
      Mutex::CondVar condvar;
    };

#ifdef KOKKOS_ENABLE_OPENMP
    std::vector<ProcessorImpl *> kokkos_omp_procs;

    Mutex omp_instance_map_mutex;
    std::map<Processor, Kokkos::OpenMP *> omp_instance_map;

    class KokkosOpenMPInitializer : public KokkosInternalTask {
      bool is_first;

    public:
      KokkosOpenMPInitializer(bool first)
        : is_first(first)
      {}

      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing openmp init on proc " << p;
        if(!is_first) {
          // only the first proc initializes the backend
          mark_done();
          return;
        }
        ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
        int num_threads =
            (impl->kind == Processor::OMP_PROC)
                ? checked_cast<LocalOpenMPProcessor *>(impl)->get_num_threads()
                : 1;
        Kokkos::InitializationSettings init_settings;
        init_settings.set_num_threads(num_threads);
        Kokkos::OpenMP::impl_initialize(init_settings);
        mark_done();
      }
    };

    class KokkosOpenMPFinalizer : public KokkosInternalTask {
      bool is_last;

    public:
      KokkosOpenMPFinalizer(bool last)
        : is_last(last)
      {}

      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing openmp finalize on proc " << p;

        // delete all the omp instances from this proc that we've cached
        for(std::map<Processor, Kokkos::OpenMP *>::iterator it = omp_instance_map.begin();
            it != omp_instance_map.end(); ++it)
          if(it->first == p)
            delete it->second;
        if(!is_last) {
          // only the last proc finalizes the backend
          mark_done();
          return;
        }
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
      bool is_first;

    public:
      KokkosCudaInitializer(bool first)
        : is_first(first)
      {}

      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing cuda init on proc " << p;

        if(!is_first) {
          // only the first gpu proc initializes the backend
          mark_done();
          return;
        }
        ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
        assert(impl != nullptr && "invalid processor handle");
        assert(impl->kind == Processor::TOC_PROC);
        Cuda::GPUProcessor *gpu = checked_cast<Cuda::GPUProcessor *>(impl);

        // initialize Kokkos's Cuda default instance on this processor's own
        //  device - without an explicit device id Kokkos would pick the
        //  first visible device, which need not be this processor's
        CUcontext entry_ctx = nullptr;
        cuCtxGetCurrent(&entry_ctx);
        Kokkos::InitializationSettings init_settings;
        init_settings.set_device_id(gpu->gpu->info->index);
        Kokkos::Cuda::impl_initialize(init_settings);
        // no instance creation here: Kokkos::Cuda::impl_initialize fully
        //  initializes the default instance itself (Kokkos 4 and 5), and
        //  constructing an execution space instance is not permitted until
        //  Kokkos::Impl::post_initialize has run (Kokkos 5 aborts on it)

        // Kokkos's initialization binds the device's primary context and
        //  (in Kokkos 5) leaves an extra context stack entry from the
        //  default-instance creation; restore the context this processor's
        //  scheduler had established
        CUcontext ctx;
        cuCtxPopCurrent(&ctx);
        cuCtxSetCurrent(entry_ctx);
        mark_done();
      }
    };

    class KokkosCudaFinalizer : public KokkosInternalTask {
      bool is_last;

    public:
      KokkosCudaFinalizer(bool last)
        : is_last(last)
      {}

      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing cuda finalize on proc " << p;

        // deleting cached instances or finalizing the backend can disturb
        //  this thread's CUDA context; restore on every path out
        CUcontext entry_ctx = nullptr;
        cuCtxGetCurrent(&entry_ctx);
        // delete all the cuda instances from this proc that we've cached
        for(std::map<std::pair<Processor, cudaStream_t>, Kokkos::Cuda *>::iterator it =
                cuda_instance_map.begin();
            it != cuda_instance_map.end(); ++it)
          if(it->first.first == p)
            delete it->second;
        if(!is_last) {
          // only the last gpu proc finalizes the backend
          cuCtxSetCurrent(entry_ctx);
          mark_done();
          return;
        }
        Kokkos::Cuda::impl_finalize();
        // Kokkos's finalization sets each initialized device's context
        //  current in turn; restore the context this processor's scheduler
        //  had established
        cuCtxSetCurrent(entry_ctx);
        mark_done();
      }
    };
#endif

#ifdef KOKKOS_ENABLE_HIP
    std::vector<ProcessorImpl *> kokkos_hip_procs;

    Mutex hip_instance_map_mutex;
    std::map<std::pair<Processor, hipStream_t>, Kokkos::HIP *> hip_instance_map;

    class KokkosHipInitializer : public KokkosInternalTask {
    public:
      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing hip init on proc " << p;

        ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
        assert(impl != nullptr && "invalid processor handle");
        assert(impl->kind == Processor::TOC_PROC);
        Hip::GPUProcessor *gpu = checked_cast<Hip::GPUProcessor *>(impl);

        Kokkos::InitializationSettings init_settings;
        init_settings.set_device_id(gpu->gpu->info->index);
        Kokkos::HIP::impl_initialize(init_settings);
        // no instance creation here: Kokkos::HIP::impl_initialize fully
        //  initializes the default instance itself (Kokkos 4 and 5), and
        //  constructing an execution space instance is not permitted until
        //  Kokkos::Impl::post_initialize has run (Kokkos 5 aborts on it)
        mark_done();
      }
    };

    class KokkosHipFinalizer : public KokkosInternalTask {
      bool is_last;

    public:
      KokkosHipFinalizer(bool last)
        : is_last(last)
      {}

      virtual void execute_on_processor(Processor p)
      {
        log_kokkos.info() << "doing hip finalize on proc " << p;

        // delete all the hip instances from this proc that we've cached
        for(std::map<std::pair<Processor, hipStream_t>, Kokkos::HIP *>::iterator it =
                hip_instance_map.begin();
            it != hip_instance_map.end(); ++it)
          if(it->first.first == p)
            delete it->second;

        if(!is_last) {
          // only the last gpu proc finalizes the backend
          mark_done();
          return;
        }
        Kokkos::HIP::impl_finalize();
        mark_done();
      }
    };
#endif

    REALM_PUBLIC_API void kokkos_initialize(
        const std::vector<ProcessorImpl *> &local_procs) // needed by librealm.so
    {
      // use Kokkos::Impl::{pre,post}_initialize to allow us to do our own
      //  execution space initialization
      Kokkos::InitializationSettings kokkos_init_args;
      log_kokkos.info() << "doing general pre-initialization";
      Kokkos::Impl::pre_initialize(kokkos_init_args);

#ifdef KOKKOS_ENABLE_SERIAL
      // nothing thread-specific for serial execution space, so just call it
      //  here
      Kokkos::Serial::impl_initialize(kokkos_init_args);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
      // need to initialize the Kokkos openmp execution space...
#ifdef REALM_USE_OPENMP
      // ... from an openmp proc
      {
        // if we're providing openmp goodness, set environment variable to shut
        //  off some kokkos warnings that don't mean anything
        setenv("OMP_PROC_BIND", "false", 0 /*!overwrite*/);

        int count = 0;
        for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
            it != local_procs.end(); ++it)
          if((*it)->kind == Processor::OMP_PROC) {
            count++;
            KokkosOpenMPInitializer ompinit(count == 1);
            (*it)->add_internal_task(&ompinit);
            ompinit.wait_done();
            kokkos_omp_procs.push_back(*it);
#ifndef REALM_OPENMP_SYSTEM_RUNTIME
            LocalOpenMPProcessor *omp = checked_cast<LocalOpenMPProcessor *>(*it);
            int num_threads = omp->get_num_threads();
            if(num_threads != 1) {
              log_kokkos.fatal() << "Kokkos OpenMP support under Realm OpenMP requires "
                                    "exactly 1 thread per proc (found "
                                 << num_threads << ") - suggest -ll:othr 1";
              abort();
            }
#endif
          }
      }
#else
      // ... from normal CPU procs since we don't have anything better
      {
        int count = 0;
        for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
            it != local_procs.end(); ++it)
          if((*it)->kind == Processor::LOC_PROC) {
            count++;
            if(count > 1)
              continue;
            KokkosOpenMPInitializer ompinit(count == 1);
            (*it)->add_internal_task(&ompinit);
            ompinit.wait_done();
            kokkos_omp_procs.push_back(*it);
          }
        if(count != 1) {
          log_kokkos.fatal() << "Kokkos OpenMP support without realm OpenMP requires "
                                "exactly 1 cpu proc (found "
                             << count << ") - suggest -ll:cpu 1";
          abort();
        }
      }
#endif
#endif

#ifdef KOKKOS_ENABLE_CUDA
      {
        size_t count = 0;
        for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
            it != local_procs.end(); ++it)
          if((*it)->kind == Processor::TOC_PROC) {
            count++;
            KokkosCudaInitializer cudainit(count == 1);
            (*it)->add_internal_task(&cudainit);
            cudainit.wait_done();
            kokkos_cuda_procs.push_back(*it);
          }
      }
#endif

#ifdef KOKKOS_ENABLE_HIP
      {
        size_t count = 0;
        for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
            it != local_procs.end(); ++it)
          if((*it)->kind == Processor::TOC_PROC) {
            count++;
            if(count > 1)
              continue; // we'll complain below
            KokkosHipInitializer hipinit;
            (*it)->add_internal_task(&hipinit);
            hipinit.wait_done();
            kokkos_hip_procs.push_back(*it);
          }
        if(count != 1) {
          log_kokkos.fatal() << "Kokkos Hip support requires exactly 1 gpu proc (found "
                             << count << ") - suggest -ll:gpu 1";
          abort();
        }
      }
#endif

      // TODO: warn if Kokkos has other execution spaces enabled that we're not
      //  willing/able to initialize?

      log_kokkos.info() << "doing general post-initialization";
      Kokkos::Impl::post_initialize(kokkos_init_args);
    }

    REALM_PUBLIC_API void kokkos_finalize(
        const std::vector<ProcessorImpl *> &local_procs) // needed by librealm.so
    {
      Kokkos::Impl::pre_finalize();

      // per processor finalization on the correct threads
#ifdef KOKKOS_ENABLE_OPENMP
      ProcessorImpl *last_omp_proc =
          kokkos_omp_procs.empty() ? nullptr : kokkos_omp_procs.back();
      for(std::vector<ProcessorImpl *>::const_iterator it = kokkos_omp_procs.begin();
          it != kokkos_omp_procs.end(); ++it) {
        KokkosOpenMPFinalizer ompfinal(*it == last_omp_proc);
        (*it)->add_internal_task(&ompfinal);
        ompfinal.wait_done();
      }
#endif

#ifdef KOKKOS_ENABLE_CUDA
      ProcessorImpl *last_cuda_proc =
          kokkos_cuda_procs.empty() ? nullptr : kokkos_cuda_procs.back();
      for(std::vector<ProcessorImpl *>::const_iterator it = kokkos_cuda_procs.begin();
          it != kokkos_cuda_procs.end(); ++it) {
        KokkosCudaFinalizer cudafinal(*it == last_cuda_proc);
        (*it)->add_internal_task(&cudafinal);
        cudafinal.wait_done();
      }
#endif

#ifdef KOKKOS_ENABLE_HIP
      ProcessorImpl *last_hip_proc =
          kokkos_hip_procs.empty() ? nullptr : kokkos_hip_procs.back();
      for(std::vector<ProcessorImpl *>::const_iterator it = kokkos_hip_procs.begin();
          it != kokkos_hip_procs.end(); ++it) {
        KokkosHipFinalizer hipfinal(*it == last_hip_proc);
        (*it)->add_internal_task(&hipfinal);
        hipfinal.wait_done();
      }
#endif

#ifdef KOKKOS_ENABLE_SERIAL
      // match the Serial::impl_initialize in kokkos_initialize - without
      //  this the Serial singleton lives until static destruction, and
      //  releasing any scratch it acquired then performs a global fence
      //  that touches an already-unloaded CUDA runtime and terminates
      Kokkos::Serial::impl_finalize();
#endif

      log_kokkos.info() << "doing general finalization";
      Kokkos::Impl::post_finalize();
    }

  }; // namespace KokkosInterop

    // execution space instance conversions from processor.h
#ifdef KOKKOS_ENABLE_SERIAL
  template <>
  REALM_PUBLIC_API Processor::KokkosExecInstance::operator Kokkos::Serial() const
  {
    return Kokkos::Serial();
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  template <>
  REALM_PUBLIC_API Processor::KokkosExecInstance::operator Kokkos::OpenMP() const
  {
    ProcessorImpl *impl = get_runtime()->get_processor_impl(p);
    LocalOpenMPProcessor *omp = checked_cast<LocalOpenMPProcessor *>(impl);
    Kokkos::OpenMP *inst = nullptr;
    {
      AutoLock<> al(KokkosInterop::omp_instance_map_mutex);
      std::map<Processor, Kokkos::OpenMP *>::iterator it =
          KokkosInterop::omp_instance_map.find(p);
      if(it != KokkosInterop::omp_instance_map.end()) {
        inst = it->second;
      } else {
        Processor::enable_scheduler_lock(); // TODO: remove?
        inst = new Kokkos::OpenMP(omp->get_num_threads());
        Processor::disable_scheduler_lock();
        KokkosInterop::omp_instance_map[p] = inst;
      }
    }
    return *inst;
  }
#endif

#ifdef KOKKOS_ENABLE_CUDA
  template <>
  REALM_PUBLIC_API Processor::KokkosExecInstance::operator Kokkos::Cuda() const
  {
#ifdef REALM_USE_CUDA
    cudaStream_t stream = Cuda::get_task_cuda_stream();
    log_kokkos.info() << "handing back stream " << stream;
    Kokkos::Cuda *inst = nullptr;
    {
      AutoLock<> al(KokkosInterop::cuda_instance_map_mutex);
      std::pair<Processor, cudaStream_t> key(p, stream);
      std::map<std::pair<Processor, cudaStream_t>, Kokkos::Cuda *>::iterator it =
          KokkosInterop::cuda_instance_map.find(key);
      if(it != KokkosInterop::cuda_instance_map.end()) {
        inst = it->second;
      } else {
        // creating a Kokkos::Cuda instance does some blocking calls, but we're
        //  not re-entrant here, so enable the scheduler lock
        Processor::enable_scheduler_lock();
        inst = new Kokkos::Cuda(stream);
        CUcontext ctx;
        cuCtxPopCurrent(&ctx);
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

#ifdef KOKKOS_ENABLE_HIP
  template <>
  REALM_PUBLIC_API Processor::KokkosExecInstance::operator Kokkos::HIP() const
  {
#ifdef REALM_USE_HIP
    hipStream_t stream = Hip::get_task_hip_stream();
    log_kokkos.info() << "handing back stream " << stream;
    Kokkos::HIP *inst = nullptr;
    {
      AutoLock<> al(KokkosInterop::hip_instance_map_mutex);
      std::pair<Processor, hipStream_t> key(p, stream);
      std::map<std::pair<Processor, hipStream_t>, Kokkos::HIP *>::iterator it =
          KokkosInterop::hip_instance_map.find(key);
      if(it != KokkosInterop::hip_instance_map.end()) {
        inst = it->second;
      } else {
        // creating a Kokkos::HIP instance does some blocking calls, but we're
        //  not re-entrant here, so enable the scheduler lock
        Processor::enable_scheduler_lock();
        inst = new Kokkos::HIP(stream);
        Processor::disable_scheduler_lock();
        KokkosInterop::hip_instance_map[key] = inst;
      }
    }
    return *inst;
#else
    // we're oblivious to the application's use of HIP
    return Kokkos::HIP();
#endif
  }
#endif

}; // namespace Realm
