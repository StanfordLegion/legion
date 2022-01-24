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

#ifndef REALM_KOKKOS_INTEROP_H
#define REALM_KOKKOS_INTEROP_H

#include "realm/realm_config.h"
#include "realm/proc_impl.h"

#include <vector>

namespace Realm {
  
  namespace KokkosInterop {

    bool is_kokkos_cuda_enabled(void);
    bool is_kokkos_openmp_enabled(void);

    // initializes the kokkos runtime, using the threads from local processors
    //  to initialize the various kokkos execution spaces
    void kokkos_initialize(const std::vector<ProcessorImpl *>& local_procs);
    
    void kokkos_finalize(const std::vector<ProcessorImpl *>& local_procs);
    
  };

};

#endif
