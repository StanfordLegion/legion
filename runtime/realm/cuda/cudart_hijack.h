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

// helper objects related to Realm's hijacking of the CUDA runtime API

#ifndef REALM_CUDART_HIJACK_H
#define REALM_CUDART_HIJACK_H

// so that we get types and stuff right
#include <cuda_runtime.h>

#include "realm/mutex.h"

#include <set>
#include <vector>
#include <unordered_map>

struct CUmod_st;
struct CUfunc_st;

namespace Realm {
  namespace Cuda {

    // this flag will be set on the first call into any of this hijack code - if
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    extern bool cudart_hijack_active;

    // for most CUDART API entry points, calling them from a non-GPU task is
    //  a fatal error - for others (e.g. cudaDeviceSynchronize), it's either
    //  silently permitted (0), warned (1), or a fatal error (2) based on this
    //  setting
    extern int cudart_hijack_nongpu_sync;

    // files compiled with nvcc will use global registrations of modules, variables, etc.
    //  that get broadcast to all contexts

    // Reverse engineered structure contents from cudart
    struct FatBin {
      int magic; // magic number
      int version;
      const unsigned long long *data;
      void *filename_or_fatbins;
    };

    struct RegisteredModule {
      const FatBin *fat_bin = nullptr;
      std::vector<CUmod_st *> gpu_modules;
    };

    struct RegisteredFunction {
      const FatBin *fat_bin = nullptr;
      const void *host_fun = nullptr;
      const char *device_fun = nullptr;
      std::vector<CUfunc_st *> gpu_functions;

      RegisteredFunction() = default;
      RegisteredFunction(const FatBin *_fat_bin, const void *_host_fun,
			 const char *_device_fun);
    };
     
    struct RegisteredVariable {
      const FatBin *fat_bin = nullptr;
      const void *host_var = nullptr;
      const char *device_name = nullptr;
      bool external = false;
      int size = 0;
      bool constant = false;
      bool global = false;
      bool managed = false;
      std::vector<uintptr_t> gpu_addresses;

      RegisteredVariable() = default;
      RegisteredVariable(const FatBin *_fat_bin, const void *_host_var,
                         const char *_device_name, bool _external, int _size,
                         bool _constant, bool _global, bool _managed);
    };

    class GPU;

    class GlobalRegistrations {
    protected:
      GlobalRegistrations(void);
      ~GlobalRegistrations(void);

      static GlobalRegistrations& get_global_registrations(void);

    public:
      // called by a GPU when it has created its context - will result in calls back
      //  into the GPU for any modules/variables/whatever already registered
      static void add_gpu_context(GPU *gpu);
      static void remove_gpu_context(GPU *gpu);

      // called by __cuda(un)RegisterFatBinary
      static void register_fat_binary(const FatBin *fatbin);
      static void unregister_fat_binary(const FatBin *fatbin);

      // called by __cudaRegisterVar
      static void register_variable(const RegisteredVariable &var);

      // called by __cudaRegisterFunction
      static void register_function(const RegisteredFunction &func);

      static CUfunc_st *lookup_function(const void *func, GPU *gpu);
      static uintptr_t lookup_variable(const void *var, GPU *gpu);

    protected:
      void register_variable_under_lock(RegisteredVariable &var, GPU *gpu);
      void register_function_under_lock(RegisteredFunction &func, GPU *gpu);
      void load_module_under_lock(RegisteredModule &mod, GPU *gpu);

      RWLock rwlock;
      std::set<GPU *> active_gpus;
      std::unordered_map<const FatBin *, RegisteredModule> modules;
      std::unordered_map<const void *, RegisteredVariable> variables;
      std::unordered_map<const void *, RegisteredFunction> functions;
    };
  };
};

#endif
