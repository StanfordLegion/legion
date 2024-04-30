
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

// UCP bootstrap loader

#ifndef BOOTSTRAP_LOADER_H
#define BOOTSTRAP_LOADER_H

#include "realm/ucx/bootstrap/bootstrap.h"

namespace Realm {
namespace UCP {

  int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle);
  int bootstrap_loader_finalize(bootstrap_handle_t *handle);

}; // namespace UCP

}; // namespace Realm

#endif
