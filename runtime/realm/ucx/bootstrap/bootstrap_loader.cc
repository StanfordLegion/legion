
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

// UCP bootstrap dynamic loader

#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "realm/realm_config.h"
#include "realm/ucx/bootstrap/bootstrap_util.h"
#include "realm/ucx/bootstrap/bootstrap_internal.h"

#define GET_SYMBOL(lib_handle, name, var, status)                         \
  do {                                                                    \
    void **var_ptr = (void **)&(var);                                     \
    void *tmp = (void *)dlsym(lib_handle, name);                          \
    BOOTSTRAP_NULL_ERROR_JMP(tmp, status, BOOTSTRAP_ERROR_INTERNAL, out,  \
        "Bootstrap failed to get symbol '%s'\n\t%s\n", name, dlerror());  \
    *var_ptr = tmp;                                                       \
  } while (0)

static void *plugin_hdl;
static char *plugin_name;

namespace Realm {
namespace UCP {

  int bootstrap_loader_finalize(bootstrap_handle_t *handle) {
    int status = handle->finalize(handle);

    if (status != 0)
      BOOTSTRAP_ERROR_PRINT("Bootstrap plugin finalize failed for '%s'\n", plugin_name);

    dlclose(plugin_hdl);
    free(plugin_name);

    return 0;
  }

  int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle) {
    int (*bootstrap_plugin_init)(void *arg, bootstrap_handle_t *handle);
    int status = 0;

    dlerror(); /* Clear any existing error */
    plugin_name = strdup(plugin);
    plugin_hdl  = dlopen(plugin, RTLD_NOW);
    BOOTSTRAP_NULL_ERROR_JMP(plugin_hdl, status, -1, error,
        "Bootstrap unable to load '%s'\n\t%s\n", plugin, dlerror());

    dlerror(); /* Clear any existing error */
    GET_SYMBOL(plugin_hdl, "realm_ucp_bootstrap_plugin_init", bootstrap_plugin_init, status);

    status = bootstrap_plugin_init(arg, handle);
    BOOTSTRAP_NZ_ERROR_JMP(status, BOOTSTRAP_ERROR_INTERNAL, error,
        "Bootstrap plugin init failed for '%s'\n", plugin);

    goto out;

error:
    if (plugin_hdl != NULL)
      dlclose(plugin_hdl);
    if (plugin_name)
      free(plugin_name);

out:
    return status;
  }

}; // namespace UCP

}; // namespace Realm
