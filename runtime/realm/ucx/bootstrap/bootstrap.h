
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

// UCP bootstrap

#ifndef UCP_BOOTSTRAP_H
#define UCP_BOOTSTRAP_H

#define BOOTSTRAP_ERROR_INTERNAL -1

enum reduction_op
{
  REDUCTION_SUM = 0x0,
  REDUCTION_LAST,
  REDUCTION_MAX = 0xFFFFFFFF
};

typedef struct bootstrap_handle {
  int pg_rank;
  int pg_size;
  int *shared_ranks;
  int num_shared_ranks;
  int (*barrier)(struct bootstrap_handle *handle);
  int (*bcast)(void *buf, int bytes, int root,
               struct bootstrap_handle *handle);
  int (*gather)(const void *sendbuf, void *recvbuf, int bytes, int root,
                struct bootstrap_handle *handle);
  int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                   struct bootstrap_handle *handle);
  int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes,
                   struct bootstrap_handle *handle);
  int (*allreduce_ull)(const void *sendbuf, void *recvbuf, int count,
                       enum reduction_op op, struct bootstrap_handle *handle);
  int (*allgatherv)(const void *sendbuf, void *recvbuf, int *sizes, int *offsets,
                    struct bootstrap_handle *handle);
  int (*finalize)(struct bootstrap_handle *handle);
} bootstrap_handle_t;

#ifdef __cplusplus
extern "C" {
extern int realm_ucp_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle)
    __attribute__((visibility("default")));
}
#else
__attribute__((visibility("default"))) int
realm_ucp_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle);
#endif

#endif
