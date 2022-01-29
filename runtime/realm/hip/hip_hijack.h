/* Copyright 2022 Stanford University, NVIDIA Corporation
 *                Los Alamos National Laboratory
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

// helper objects related to Realm's hijacking of the HIP runtime API

#ifndef REALM_HIP_HIJACK_H
#define REALM_HIP_HIJACK_H


#include "realm/mutex.h"

#include <set>
#include <vector>

namespace Realm {
  namespace Hip {

    // this flag will be set on the first call into any of this hijack code - if
    //  an application is linked with -lcudart, we will NOT be hijacking the
    //  application's calls, and the cuda module needs to know that)
    extern bool cudart_hijack_active;
    
    // files compiled with nvcc will use global registrations of modules, variables, etc.
    //  that get broadcast to all contexts

    struct FatBin {
      int magic; // magic number
      int version;
      const unsigned long long *data;
      void *filename_or_fatbins;
    };

    struct RegisteredFunction {
      const FatBin *fat_bin;
      const void *host_fun;
      const char *device_fun;

      RegisteredFunction(const FatBin *_fat_bin, const void *_host_fun,
			 const char *_device_fun);
    };
     
    struct RegisteredVariable {
      const FatBin *fat_bin;
      const void *host_var;
      const char *device_name;
      bool external;
      int size;
      bool constant;
      bool global;

      RegisteredVariable(const FatBin  *_fat_bin, const void *_host_var,
			 const char *_device_name, bool _external,
			 int _size, bool _constant, bool _global);
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
      static void register_fat_binary(FatBin *fatbin);
      static void unregister_fat_binary(FatBin *fatbin);

      // called by __cudaRegisterVar
      static void register_variable(RegisteredVariable *var);

      // called by __cudaRegisterFunction
      static void register_function(RegisteredFunction *func);

    protected:
      Mutex mutex;
      std::set<GPU *> active_gpus;
      std::vector<FatBin *> fat_binaries;
      std::vector<RegisteredVariable *> variables;
      std::vector<RegisteredFunction *> functions;
    };
  };
};

#endif
