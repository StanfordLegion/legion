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

#include "realm/realm_c.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

static realm_status_t REALM_FNPTR retrieve_processor(realm_processor_t p, void *user_data)
{
  realm_processor_t *rp = (realm_processor_t *)(user_data);
  *rp = p;
  return REALM_SUCCESS;
}

static void realm_processor_query_first(realm_processor_query_t query,
                                        realm_processor_t *proc)
{
  realm_status_t status = realm_processor_query_iter(query, retrieve_processor, proc, 1);
  assert(status == REALM_SUCCESS);
}
