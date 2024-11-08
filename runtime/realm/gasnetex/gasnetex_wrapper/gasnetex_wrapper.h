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

#ifndef GASNETEX_WRAPPER_H
#define GASNETEX_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifndef GEX_NO_PROTOTYPES
#include "gex_export.h"
#endif

typedef uint32_t gex_flags_t;
typedef uint32_t gex_rank_t;
typedef uint16_t gex_ep_index_t;
typedef uint32_t gex_dt_t;
typedef uint32_t gex_op_t;
typedef int32_t gex_am_arg_t;
typedef uint32_t gex_ep_capabilities_t;

typedef void *gex_client_opaque_t;
typedef void *gex_ep_opaque_t;
typedef void *gex_tm_opaque_t;
typedef void *gex_segment_opaque_t;
typedef void *gex_mk_opaque_t;
typedef void *gex_addr_opaque_t;
typedef void *gex_event_opaque_t;
typedef void *gex_am_src_desc_opaque_t;

#define __GASNET_ERR_BASE 10000
#define GEX_WRAPPER_OK 0
#define GEX_WRAPPER_NOT_VALID -1
#define GEX_WRAPPER_ERR_NOT_READY (__GASNET_ERR_BASE + 4)
#define GEX_WRAPPER_EP_INVALID ((gex_ep_opaque_t)(uintptr_t)0)
#define GEX_WRAPPER_MK_INVALID ((gex_mk_opaque_t)(uintptr_t)0)
#define GEX_WRAPPER_SEGMENT_INVALID ((gex_segment_opaque_t)(uintptr_t)0)
#define GEX_WRAPPER_EVENT_INVALID ((gex_event_opaque_t)(uintptr_t)0)
#define GEX_WRAPPER_EVENT_NO_OP ((gex_event_opaque_t)(uintptr_t)1)
#define GEX_WRAPPER_EVENT_NOW ((gex_event_opaque_t *)(uintptr_t)1)
#define GEX_WRAPPER_EVENT_DEFER ((gex_event_opaque_t *)(uintptr_t)2)
#define GEX_WRAPPER_EVENT_GROUP ((gex_event_opaque_t *)(uintptr_t)3)
#define GEX_WRAPPER_AM_SRCDESC_NO_OP NULL
#define GEX_WRAPPER_DT_U64 1 << 3
#define GEX_WRAPPER_OP_ADD 1 << 3
#define GEX_WRAPPER_EP_CAPABILITY_RMA (1U << 0)
#define GEX_WRAPPER_MK_HOST ((gex_mk_opaque_t)(uintptr_t)1)
#define GEX_WRAPPER_FLAG_IMMEDIATE (1U << 0)
#define GEX_WRAPPER_FLAG_AM_PREPARE_LEAST_CLIENT (1U << 11)
#define GEX_WRAPPER_FLAG_AM_PREPARE_LEAST_ALLOC (1U << 12)

#define GEX_WRAPPER_CONDUIT_ARIES 0
#define GEX_WRAPPER_CONDUIT_IBV 1
#define GEX_WRAPPER_CONDUIT_MPI 2
#define GEX_WRAPPER_CONDUIT_OFI 3
#define GEX_WRAPPER_CONDUIT_SMP 4
#define GEX_WRAPPER_CONDUIT_UCX 5
#define GEX_WRAPPER_CONDUIT_UDP 6

/// @brief Handle structure that contains the full API for the wrapper
/// @note This structure is versioned based on size, do not add fields or APIs in the
/// middle of this structure, only at the end.
typedef struct gex_wrapper_handle_s {
  /* variable */
  size_t handle_size;
  uint16_t max_eps;
  int conduit;
  size_t AM_LUBRequestMedium;
  int GEX_RELEASE;
  int GEX_API;
  bool GEX_RMA_HONORS_IMMEDIATE_FLAG;
  bool GEX_NATIVE_NP_ALLOC_REQ_MEDIUM;
  bool GEX_HAVE_MK_CLASS_CUDA_UVA;
  bool GEX_HAVE_MK_CLASS_HIP;

  /* combined */

  int (*gex_client_init)(gex_client_opaque_t *client_p, /*INOUT*/
                         gex_ep_opaque_t *ep_p,         /*INOUT*/
                         gex_tm_opaque_t *tm_p,         /*INOUT*/
                         gex_rank_t *rank,              /*OUT*/
                         gex_rank_t *size,              /*OUT*/
                         const char *clientName,        /*IN*/
                         int *argc,                     /*IN*/
                         char ***argv,                  /*IN*/
                         gex_flags_t flags,             /*IN*/
                         const void *val                /*IN*/
  );

  int (*gex_ep_create)(gex_ep_opaque_t *ep_p,              /*INOUT*/
                       gex_ep_index_t *ep_index,           /*OUT*/
                       gex_client_opaque_t client,         /*IN*/
                       gex_ep_capabilities_t capabilities, /*IN*/
                       gex_flags_t flags,                  /*IN*/
                       const void *val                     /*IN*/
  );

  void *(*gex_segment_attach_query_addr)(gex_segment_opaque_t *segment_p,
                                         gex_tm_opaque_t tm, uintptr_t size);

  int (*gex_mk_create_cuda)(gex_mk_opaque_t *memkind_p, gex_client_opaque_t client,
                            int device, gex_flags_t flags);

  int (*gex_mk_create_hip)(gex_mk_opaque_t *memkind_p, gex_client_opaque_t client,
                           int device, gex_flags_t flags);

  int (*gex_segment_create)(gex_segment_opaque_t *segment_p, // OUT
                            gex_client_opaque_t client, gex_addr_opaque_t address,
                            uintptr_t length, gex_mk_opaque_t kind, gex_flags_t flags);

  void (*gex_ep_bind_segment)(gex_ep_opaque_t ep, gex_segment_opaque_t segment,
                              gex_flags_t flags);

  void (*gex_query_shared_peers)(gex_rank_t *num_shared_ranks, gex_rank_t **shared_ranks);

  void (*gex_nbi_wait_ec_am)(gex_flags_t flags);

  /* wrapper */

  const char *(*gex_error_name)(int error);

  const char *(*gex_error_desc)(int error);

  int (*gex_ep_publish_bound_segment)(gex_tm_opaque_t tm,
                                      gex_ep_opaque_t *eps, // IN
                                      size_t num_eps, gex_flags_t flags);

  int (*gex_event_test)(gex_event_opaque_t event);

  size_t (*gex_am_src_desc_size)(gex_am_src_desc_opaque_t sd);

  void *(*gex_am_src_desc_addr)(gex_am_src_desc_opaque_t sd);

  int (*gex_am_poll)(void);

  /* rma */

  gex_event_opaque_t (*gex_rma_iget)(gex_ep_opaque_t local_ep,
                                     gex_ep_index_t remote_ep_index, void *dest,
                                     gex_rank_t rank, void *src, size_t nbytes,
                                     gex_flags_t flags);

  gex_event_opaque_t (*gex_rma_iput)(gex_ep_opaque_t local_ep,
                                     gex_ep_index_t remote_ep_index, gex_rank_t rank,
                                     void *dest, const void *src, size_t nbytes,
                                     gex_event_opaque_t *lc_opt, gex_flags_t flags);

  /* collective */

  void (*gex_coll_barrier)(gex_tm_opaque_t tm, gex_flags_t flags);

  void (*gex_coll_bcast)(gex_tm_opaque_t tm, gex_rank_t root, void *dst, const void *src,
                         size_t nbytes, gex_flags_t flags);

  void (*gex_coll_gather)(gex_tm_opaque_t tm, gex_rank_t root, const void *val_in,
                          void *vals_out, size_t bytes, gex_flags_t flags);

  void (*gex_coll_allgather)(gex_tm_opaque_t tm, const void *val_in, void *vals_out,
                             size_t bytes, gex_flags_t flags);

  void (*gex_coll_allgatherv)(gex_tm_opaque_t tm, const void *val_in, void *vals_out,
                              int *bytes, int *offsets, gex_flags_t flags);

  gex_event_opaque_t (*gex_coll_ireduce)(gex_tm_opaque_t tm, void *dst, const void *src,
                                         gex_dt_t dt, size_t dt_sz, size_t dt_cnt,
                                         gex_op_t op, gex_flags_t flags);

  /* gasnetex_handlers */

  int (*send_completion_reply)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                               gex_ep_index_t tgt_ep_index, const gex_am_arg_t *args,
                               size_t nargs, gex_flags_t flags);

  int (*send_request_short)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                            gex_ep_index_t tgt_ep_index, gex_am_arg_t arg0,
                            const void *hdr, size_t hdr_bytes, gex_flags_t flags);

  size_t (*max_request_medium)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                               gex_ep_index_t tgt_ep_index, size_t hdr_bytes,
                               gex_event_opaque_t *lc_opt, gex_flags_t flags);

  int (*send_request_medium)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                             gex_ep_index_t tgt_ep_index, gex_am_arg_t arg0,
                             const void *hdr, size_t hdr_bytes, const void *data,
                             size_t data_bytes, gex_event_opaque_t *lc_opt,
                             gex_flags_t flags);

  gex_am_src_desc_opaque_t (*prepare_request_medium)(
      gex_ep_opaque_t src_ep, gex_rank_t tgt_rank, gex_ep_index_t tgt_ep_index,
      size_t hdr_bytes, const void *data, size_t min_data_bytes, size_t max_data_bytes,
      gex_event_opaque_t *lc_opt, gex_flags_t flags);

  void (*commit_request_medium)(gex_am_src_desc_opaque_t srcdesc, gex_am_arg_t arg0,
                                const void *hdr, size_t hdr_bytes, size_t data_bytes);

  size_t (*max_request_long)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                             gex_ep_index_t tgt_ep_index, size_t hdr_bytes,
                             gex_event_opaque_t *lc_opt, gex_flags_t flags);

  int (*send_request_long)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                           gex_ep_index_t tgt_ep_index, gex_am_arg_t arg0,
                           const void *hdr, size_t hdr_bytes, const void *data,
                           size_t data_bytes, gex_event_opaque_t *lc_opt,
                           gex_flags_t flags, uintptr_t dest_addr);

  // sends a long as a "reverse get": the _header_ is sent as the
  //  payload of a medium that contains the src ep_index/ptr so that
  //  the target can do an RMA get of the actual payload
  int (*send_request_rget)(gex_tm_opaque_t prim_tm, gex_rank_t tgt_rank,
                           gex_ep_index_t tgt_ep_index, gex_am_arg_t arg0,
                           const void *hdr, size_t hdr_bytes,
                           gex_ep_index_t payload_ep_index, const void *payload,
                           size_t payload_bytes, gex_event_opaque_t *lc_opt,
                           gex_flags_t flags, uintptr_t dest_addr);

  // send the header of a long after the payload has been delivered via
  //  an RMA put: header is sent as a medium
  int (*send_request_put_header)(gex_ep_opaque_t src_ep, gex_rank_t tgt_rank,
                                 gex_ep_index_t tgt_ep_index, gex_am_arg_t arg0,
                                 const void *hdr, size_t hdr_bytes, uintptr_t dest_addr,
                                 size_t payload_bytes, gex_event_opaque_t *lc_opt,
                                 gex_flags_t flags);

  gex_am_src_desc_opaque_t (*prepare_request_batch)(
      gex_ep_opaque_t src_ep, gex_rank_t tgt_rank, gex_ep_index_t tgt_ep_index,
      const void *data, size_t min_data_bytes, size_t max_data_bytes,
      gex_event_opaque_t *lc_opt, gex_flags_t flags);

  void (*commit_request_batch)(gex_am_src_desc_opaque_t srcdesc, gex_am_arg_t arg0,
                               gex_am_arg_t cksum, size_t data_bytes);

} gex_wrapper_handle_t;

/* callback table for handlers, this is used for handlers to callback the functions inside
 * the GASNetEXInternal */

typedef struct gex_callback_handle_s {
  void *gex_internal; // this is a pointer to the GASNetEXInternal object
  gex_am_arg_t (*handle_short)(void *gex_internal, gex_rank_t srcrank, gex_am_arg_t arg0,
                               const void *hdr, size_t hdr_bytes);
  gex_am_arg_t (*handle_medium)(void *gex_internal, gex_rank_t srcrank, gex_am_arg_t arg0,
                                const void *hdr, size_t hdr_bytes, const void *data,
                                size_t data_bytes);
  gex_am_arg_t (*handle_long)(void *gex_internal, gex_rank_t srcrank, gex_am_arg_t arg0,
                              const void *hdr, size_t hdr_bytes, const void *data,
                              size_t data_bytes);
  void (*handle_reverse_get)(void *gex_internal, gex_rank_t srcrank,
                             gex_ep_index_t src_ep_index, gex_ep_index_t tgt_ep_index,
                             gex_am_arg_t arg0, const void *hdr, size_t hdr_bytes,
                             uintptr_t src_ptr, uintptr_t tgt_ptr, size_t payload_bytes);
  size_t (*handle_batch)(void *gex_internal, gex_rank_t srcrank, gex_am_arg_t arg0,
                         gex_am_arg_t cksum, const void *data, size_t data_bytes,
                         gex_am_arg_t *comps);
  void (*handle_completion_reply)(void *gex_internal, gex_rank_t srcrank,
                                  const gex_am_arg_t *args, size_t nargs);

  // ^--------- ADD NEW APIS AND FIELDS HERE -------------^

} gex_callback_handle_t;

typedef int (*gex_wrapper_init_pfn)(gex_wrapper_handle_t *handle);

#ifdef __cplusplus
extern "C" {
#endif
#ifndef GEX_NO_PROTOTYPES
int GEX_EXPORT realm_gex_wrapper_init(gex_wrapper_handle_t *handle);
#endif /*GEX_PROTOTYPES*/
#ifdef __cplusplus
}
#endif

#endif
