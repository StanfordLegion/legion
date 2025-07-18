
/* Copyright 2024 NVIDIA Corporation
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

// UCP bootstrap internal

#ifndef BOOTSTRAP_INTERNAL_H
#define BOOTSTRAP_INTERNAL_H

#include <cstddef>
#include "realm/ucx/bootstrap/bootstrap.h"

#define BOOTSTRAP_MPI_PLUGIN "realm_ucp_bootstrap_mpi.so"
#define BOOTSTRAP_P2P_PLUGIN "realm_ucp_bootstrap_p2p.so"
namespace Realm {
namespace UCP {

  enum BootstrapMode
  {
    BOOTSTRAP_MPI,
    BOOTSTRAP_P2P,
    BOOTSTRAP_PLUGIN
  };

  struct BootstrapConfig {
    BootstrapMode mode;
    char *plugin_name;
  };

  int bootstrap_init(const BootstrapConfig *config, bootstrap_handle_t *handle);
  int bootstrap_finalize(bootstrap_handle_t *handle);

}; // namespace UCP

}; // namespace Realm

#endif
