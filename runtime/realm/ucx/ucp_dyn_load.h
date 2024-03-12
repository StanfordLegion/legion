
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

#ifndef UCP_DYN_LOAD_H
#define UCP_DYN_LOAD_H

namespace Realm {
namespace UCP {

#ifdef REALM_UCX_DYNAMIC_LOAD
#define UCP_FNPTR(name) (name##_fnptr)

#define UCP_APIS(__op__)                            \
  __op__(ucp_am_data_release);                      \
  __op__(ucp_am_recv_data_nbx);                     \
  __op__(ucp_am_send_nbx);                          \
  __op__(ucp_cleanup);                              \
  __op__(ucp_config_modify);                        \
  __op__(ucp_config_read);                          \
  __op__(ucp_config_release);                       \
  __op__(ucp_context_query);                        \
  __op__(ucp_ep_close_nbx);                         \
  __op__(ucp_ep_create);                            \
  __op__(ucp_ep_flush_nbx);                         \
  __op__(ucp_ep_rkey_unpack);                       \
  __op__(ucp_get_version);                          \
  __op__(ucp_init_version);                         \
  __op__(ucp_mem_map);                              \
  __op__(ucp_mem_query);                            \
  __op__(ucp_mem_unmap);                            \
  __op__(ucp_put_nbx);                              \
  __op__(ucp_request_check_status);                 \
  __op__(ucp_request_free);                         \
  __op__(ucp_rkey_buffer_release);                  \
  __op__(ucp_rkey_destroy);                         \
  __op__(ucp_rkey_pack);                            \
  __op__(ucp_worker_arm);                           \
  __op__(ucp_worker_create);                        \
  __op__(ucp_worker_destroy);                       \
  __op__(ucp_worker_get_address);                   \
  __op__(ucp_worker_get_efd);                       \
  __op__(ucp_worker_progress);                      \
  __op__(ucp_worker_query);                         \
  __op__(ucp_worker_release_address);               \
  __op__(ucp_worker_set_am_recv_handler);

#define DECL_FNPTR_EXTERN(name) extern decltype(&name) name##_fnptr;
#include <ucp/api/ucp.h>
UCP_APIS(DECL_FNPTR_EXTERN);
#undef DECL_FNPTR_EXTERN

#else
#define UCP_FNPTR(name) (name)
#endif

}; // namespace UCP

}; // namespace Realm

#endif
