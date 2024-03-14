
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

// UCP network module internals

#include "realm/logging.h"
#include "realm/ucx/bootstrap/bootstrap_internal.h"
#include "realm/ucx/bootstrap/bootstrap_loader.h"

namespace Realm {

  // defined in ucp_module.cc
  extern Logger log_ucp;

namespace UCP {

  int bootstrap_init(const BootstrapConfig *config, bootstrap_handle_t *handle) {
    int status = 0;

    switch (config->mode) {
      case BOOTSTRAP_MPI:
        if (config->plugin_name != NULL) {
          status = bootstrap_loader_init(config->plugin_name, NULL, handle);
        } else {
          status = bootstrap_loader_init(BOOTSTRAP_MPI_PLUGIN, NULL, handle);
        }
        if (status != 0) {
          log_ucp.error() << "bootstrap_loader_init failed";
        }
        break;
      case BOOTSTRAP_PLUGIN:
        status = bootstrap_loader_init(config->plugin_name, NULL, handle);
        if (status != 0) {
          log_ucp.error() << "bootstrap_loader_init failed";
        }
        break;
      default:
        status = BOOTSTRAP_ERROR_INTERNAL;
        log_ucp.error() << ("invalid bootstrap mode");
    }

    return status;
  }

  int bootstrap_finalize(bootstrap_handle_t *handle) {
    int status = bootstrap_loader_finalize(handle);
    if (status != 0) {
      log_ucp.error() << "bootstrap_finalize failed";
    }

    return status;
}

}; // namespace UCP

}; // namespace Realm
