/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "activemsg.h"

#include <set>
#include <vector>

namespace Realm {
  namespace Cuda {

    // files compiled with nvcc will use global registrations of modules, variables, etc.
    //  that get broadcast to all contexts

    struct FatBin {
      int magic; // magic number
      int version;
      const unsigned long long *data;
      void *filename_or_fatbins;
    };

    struct RegisteredFunction {
      void **handle;
      const char *host_fun;
      const char *device_fun;

      RegisteredFunction(void **_handle, const char *_host_fun,
			 const char *_device_fun);
    };
     
    struct RegisteredVariable {
      void **handle;
      char *host_var;
      const char *device_name;
      bool external;
      int size;
      bool constant;
      bool global;

      RegisteredVariable(void **_handle, char *_host_var,
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
      GASNetHSL mutex;
      std::set<GPU *> active_gpus;
      std::vector<FatBin *> fat_binaries;
      std::vector<RegisteredVariable *> variables;
      std::vector<RegisteredFunction *> functions;
    };
  };
};

#endif
