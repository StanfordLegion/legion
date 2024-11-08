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

#include "gasnetex_wrapper.h"
#include "gasnetex_wrapper_internal.h"

#include <gasnet_mk.h>
#include <gasnet_coll.h>

#include <assert.h>
#include <vector>

namespace Realm {

  namespace GASNetEXWrapper {

    ////////////////////////////////////////////////////////////////////////
    //
    // combined
    //

    static int gex_client_init(gex_client_opaque_t *client_p, /*INOUT*/
                               gex_ep_opaque_t *ep_p,         /*INOUT*/
                               gex_tm_opaque_t *tm_p,         /*INOUT*/
                               gex_rank_t *rank,              /*OUT*/
                               gex_rank_t *size,              /*OUT*/
                               const char *clientName,        /*IN*/
                               int *argc,                     /*IN*/
                               char ***argv,                  /*IN*/
                               gex_flags_t flags,             /*IN*/
                               const void *val                /*IN*/
    )
    {
      int status = gex_Client_Init(
          reinterpret_cast<gex_Client_t *>(client_p), reinterpret_cast<gex_EP_t *>(ep_p),
          reinterpret_cast<gex_TM_t *>(tm_p), clientName, argc, argv, flags);

      if(status == GASNET_OK) {
        gex_TM_t tm = *(reinterpret_cast<gex_TM_t *>(tm_p));
        *rank = gex_TM_QueryRank(tm);
        *size = gex_TM_QuerySize(tm);
        gex_EP_t ep = *(reinterpret_cast<gex_EP_t *>(ep_p));
        status = gex_EP_RegisterHandlers(static_cast<gex_EP_t>(ep),
                                         GASNetEXHandlers::handler_table,
                                         GASNetEXHandlers::handler_table_size);
        gex_EP_SetCData(ep, val);

#if REALM_GEX_API >= 1300
        // once we've done the basic init, shut off verbose errors from GASNet
        //  and we'll report failures ourselves
        gex_System_SetVerboseErrors(0);
#endif
      }
      return status;
    }

    static int gex_ep_create(gex_ep_opaque_t *ep_p,              /*INOUT*/
                             gex_ep_index_t *ep_index,           /*OUT*/
                             gex_client_opaque_t client,         /*IN*/
                             gex_ep_capabilities_t capabilities, /*IN*/
                             gex_flags_t flags,                  /*IN*/
                             const void *val                     /*IN*/
    )
    {
      int ret =
          gex_EP_Create(reinterpret_cast<gex_EP_t *>(ep_p),
                        static_cast<gex_Client_t>(client), capabilities, 0 /*flags*/);
      if(ret == GASNET_OK) {
        gex_EP_t ep = *(reinterpret_cast<gex_EP_t *>(ep_p));
        *ep_index = gex_EP_QueryIndex(ep);
        gex_EP_SetCData(ep, val);
      }
      return ret;
    }

    void *gex_segment_attach_query_addr(gex_segment_opaque_t *segment_p,
                                        gex_tm_opaque_t tm, uintptr_t size)
    {
      CHECK_GEX(gex_Segment_Attach(reinterpret_cast<gex_Segment_t *>(segment_p),
                                   static_cast<gex_TM_t>(tm), size));
      gex_Segment_t segment = *(reinterpret_cast<gex_Segment_t *>(segment_p));
      return gex_Segment_QueryAddr(static_cast<gex_Segment_t>(segment));
    }

    static int gex_mk_create_cuda(gex_mk_opaque_t *memkind_p, gex_client_opaque_t client,
                                  int device, gex_flags_t flags)
    {
#if defined(GASNET_HAVE_MK_CLASS_CUDA_UVA)
      gex_MK_Create_args_t args;
      args.gex_flags = 0;
      args.gex_class = GEX_MK_CLASS_CUDA_UVA;
      args.gex_args.gex_class_cuda_uva.gex_CUdevice = device;
      int ret = gex_MK_Create(reinterpret_cast<gex_MK_t *>(memkind_p),
                              static_cast<gex_Client_t>(client), &args, flags);
      return ret;
#else
      return GEX_WRAPPER_NOT_VALID;
#endif
    }

    static int gex_mk_create_hip(gex_mk_opaque_t *memkind_p, gex_client_opaque_t client,
                                 int device, gex_flags_t flags)
    {
#if defined(GASNET_HAVE_MK_CLASS_HIP)
      gex_MK_Create_args_t args;
      args.gex_flags = 0;
      args.gex_class = GEX_MK_CLASS_HIP;
      args.gex_args.gex_class_hip.gex_hipDevice = device;
      int ret = gex_MK_Create(reinterpret_cast<gex_MK_t *>(memkind_p),
                              static_cast<gex_Client_t>(client), &args, flags);
      return ret;
#else
      return GEX_WRAPPER_NOT_VALID;
#endif
    }

    static int gex_segment_create(gex_segment_opaque_t *segment_p, // OUT
                                  gex_client_opaque_t client, gex_addr_opaque_t address,
                                  uintptr_t length, gex_mk_opaque_t kind,
                                  gex_flags_t flags)
    {
      return gex_Segment_Create(reinterpret_cast<gex_Segment_t *>(segment_p),
                                static_cast<gex_Client_t>(client), address, length,
                                static_cast<gex_MK_t>(kind), flags);
    }

    static void gex_ep_bind_segment(gex_ep_opaque_t ep, gex_segment_opaque_t segment,
                                    gex_flags_t flags)
    {
      gex_System_SetVerboseErrors(1);
      gex_EP_BindSegment(static_cast<gex_EP_t>(ep), static_cast<gex_Segment_t>(segment),
                         flags);
      if(gex_EP_QuerySegment(static_cast<gex_EP_t>(ep)) !=
         static_cast<gex_Segment_t>(segment)) {
        fprintf(stderr, "failed to bind segment");
        fflush(stderr);
        abort();
      }
      gex_System_SetVerboseErrors(0);
    }

    static void gex_query_shared_peers(gex_rank_t *num_shared_ranks,
                                       gex_rank_t **shared_ranks)
    {
      gex_RankInfo_t *neighbor_array = nullptr;
      gex_System_QueryNbrhdInfo(&neighbor_array, num_shared_ranks, nullptr);
      // if PSHM module is disabled, gex_System_QueryNbrhdInfo returns size one
      // then fall back to use gex_System_QueryHostInfo
      if(*num_shared_ranks == 1) {
        gex_System_QueryHostInfo(&neighbor_array, num_shared_ranks, nullptr);
      }
      *shared_ranks = (gex_Rank_t *)malloc(sizeof(gex_Rank_t) * (*num_shared_ranks));
      for(gex_Rank_t r = 0; r < *num_shared_ranks; r++) {
        (*shared_ranks)[r] = neighbor_array[r].gex_jobrank;
      }
    }

    static void gex_nbi_wait_ec_am(gex_flags_t flags) { gex_NBI_Wait(GEX_EC_AM, flags); }

    /* wrapper */

    static const char *gex_error_name(int error) { return gasnet_ErrorName(error); }

    static const char *gex_error_desc(int error) { return gasnet_ErrorDesc(error); }

    static int gex_ep_publish_bound_segment(gex_tm_opaque_t tm,
                                            gex_ep_opaque_t *eps, // IN
                                            size_t num_eps, gex_flags_t flags)
    {
      return gex_EP_PublishBoundSegment(
          static_cast<gex_TM_t>(tm), reinterpret_cast<gex_EP_t *>(eps), num_eps, flags);
    }

    static int gex_event_test(gex_event_opaque_t event)
    {
      return gex_Event_Test(static_cast<gex_Event_t>(event));
    }

    static size_t gex_am_src_desc_size(gex_am_src_desc_opaque_t sd)
    {
      return gex_AM_SrcDescSize(static_cast<gex_AM_SrcDesc_t>(sd));
    }

    static void *gex_am_src_desc_addr(gex_am_src_desc_opaque_t sd)
    {
      return gex_AM_SrcDescAddr(static_cast<gex_AM_SrcDesc_t>(sd));
    }

    static int gex_am_poll(void) { return gasnet_AMPoll(); }

    /* rma */

    static gex_event_opaque_t gex_rma_iget(gex_ep_opaque_t local_ep,
                                           gex_ep_index_t remote_ep_index, void *dest,
                                           gex_rank_t rank, void *src, size_t nbytes,
                                           gex_flags_t flags)
    {
      gex_TM_t pair = gex_TM_Pair(static_cast<gex_EP_t>(local_ep), remote_ep_index);
      gex_Event_t rc_event = gex_RMA_GetNB(pair, dest, rank, src, nbytes, flags);
      return static_cast<gex_event_opaque_t>(rc_event);
    }

    static gex_event_opaque_t gex_rma_iput(gex_ep_opaque_t local_ep,
                                           gex_ep_index_t remote_ep_index,
                                           gex_rank_t rank, void *dest, const void *src,
                                           size_t nbytes, gex_event_opaque_t *lc_opt,
                                           gex_flags_t flags)
    {
      gex_TM_t pair = gex_TM_Pair(static_cast<gex_EP_t>(local_ep), remote_ep_index);
      gex_Event_t rc_event =
          gex_RMA_PutNB(pair, rank, dest, (void *)src, nbytes,
                        reinterpret_cast<gex_Event_t *>(lc_opt), flags);
      return static_cast<gex_event_opaque_t>(rc_event);
    }

    /* collective */

    static void gex_coll_barrier(gex_tm_opaque_t tm, gex_flags_t flags)
    {
      gex_Event_t done = gex_Coll_BarrierNB(static_cast<gex_TM_t>(tm), flags);
      gex_Event_Wait(done);
    }

    static void gex_coll_bcast(gex_tm_opaque_t tm, // The team
                               gex_rank_t root,    // Root rank (single-valued)
                               void *dst,          // Destination (all ranks)
                               const void *src,    // Source (root rank only)
                               size_t nbytes,      // Length of data (single-valued)
                               gex_flags_t flags)  // Flags (partially single-valued)
    {
      gex_Event_t done =
          gex_Coll_BroadcastNB(static_cast<gex_TM_t>(tm), root, dst, src, nbytes, flags);
      gex_Event_Wait(done);
    }

    static void gex_coll_gather(gex_tm_opaque_t tm, gex_rank_t root, const void *val_in,
                                void *vals_out, size_t bytes, gex_flags_t flags)
    {
      // GASNetEX doesn't have a gather collective?
      // this isn't performance critical right now, so cobble it together from
      //  a bunch of broadcasts
      gex_Rank_t prim_rank = gex_TM_QueryRank(static_cast<gex_TM_t>(tm));
      gex_Rank_t prim_size = gex_TM_QuerySize(static_cast<gex_TM_t>(tm));
      void *dummy = (root == prim_rank) ? 0 : alloca(bytes);
      for(gex_rank_t i = 0; i < prim_size; i++) {
        void *dst;
        if(root == prim_rank)
          dst = static_cast<char *>(vals_out) + (i * bytes);
        else
          dst = dummy;
        gex_Event_t done =
            gex_Coll_BroadcastNB(static_cast<gex_TM_t>(tm), i, dst, val_in, bytes, flags);
        gex_Event_Wait(done);
      }
    }

    static void gex_coll_allgather(gex_tm_opaque_t tm, const void *val_in, void *vals_out,
                                   size_t bytes, gex_flags_t flags)
    {
      constexpr size_t max_in_flight = 16;
      gex_Rank_t prim_size = gex_TM_QuerySize(static_cast<gex_TM_t>(tm));
      std::vector<gex_Event_t> events(max_in_flight);
      // Have everyone send each other their sizes
      for(gex_Rank_t i = 0; i < prim_size; i++) {
        void *dst = static_cast<char *>(vals_out) + (i * bytes);
        events[i % max_in_flight] =
            gex_Coll_BroadcastNB(static_cast<gex_TM_t>(tm), i, dst, val_in, bytes, flags);
        if(i % max_in_flight == max_in_flight - 1) {
          gex_Event_WaitAll(events.data(), max_in_flight, 0);
        }
      }
      // Wait for all these to complete, as we'll need their results
      gex_Event_WaitAll(events.data(), prim_size % max_in_flight, 0);
    }

    static void gex_coll_allgatherv(gex_tm_opaque_t tm, const void *val_in,
                                    void *vals_out, int *bytes, int *offsets,
                                    gex_flags_t flags)
    {
      constexpr size_t max_in_flight = 16;
      gex_Rank_t prim_size = gex_TM_QuerySize(static_cast<gex_TM_t>(tm));
      std::vector<gex_Event_t> events(prim_size);
      // Have everyone send each other their sizes
      for(gex_Rank_t i = 0; i < prim_size; i++) {
        void *dst = static_cast<char *>(vals_out) + offsets[i];
        events[i % max_in_flight] = gex_Coll_BroadcastNB(static_cast<gex_TM_t>(tm), i,
                                                         dst, val_in, bytes[i], flags);
        if(i % max_in_flight == max_in_flight - 1) {
          gex_Event_WaitAll(events.data(), max_in_flight, 0);
        }
      }
      // Wait for all these to complete, as we'll need their results
      gex_Event_WaitAll(events.data(), prim_size % max_in_flight, 0);
    }

    static gex_event_opaque_t gex_coll_ireduce(gex_tm_opaque_t tm, void *dst,
                                               const void *src, gex_dt_t dt, size_t dt_sz,
                                               size_t dt_cnt, gex_op_t op,
                                               gex_flags_t flags)
    {
      return static_cast<gex_event_opaque_t>(
          gex_Coll_ReduceToAllNB(static_cast<gex_TM_t>(tm), dst, src, dt, dt_sz, dt_cnt,
                                 op, nullptr, nullptr, flags));
    }

    // check the size of typedefed structs and defines
    static void check_size(void)
    {
      static_assert(sizeof(gex_flags_t) == sizeof(gex_Flags_t));
      static_assert(sizeof(gex_rank_t) == sizeof(gex_Rank_t));
      static_assert(sizeof(gex_ep_index_t) == sizeof(gex_EP_Index_t));
      static_assert(sizeof(gex_dt_t) == sizeof(gex_DT_t));
      static_assert(sizeof(gex_op_t) == sizeof(gex_OP_t));
      static_assert(sizeof(gex_am_arg_t) == sizeof(gex_AM_Arg_t));
      static_assert(sizeof(gex_ep_capabilities_t) == sizeof(gex_EP_Capabilities_t));

      static_assert(sizeof(gex_client_opaque_t) == sizeof(gex_Client_t));
      static_assert(sizeof(gex_ep_opaque_t) == sizeof(gex_EP_t));
      static_assert(sizeof(gex_tm_opaque_t) == sizeof(gex_TM_t));
      static_assert(sizeof(gex_segment_opaque_t) == sizeof(gex_Segment_t));
      static_assert(sizeof(gex_mk_opaque_t) == sizeof(gex_MK_t));
      static_assert(sizeof(gex_addr_opaque_t) == sizeof(gex_Addr_t));
      static_assert(sizeof(gex_event_opaque_t) == sizeof(gex_Event_t));
      static_assert(sizeof(gex_am_src_desc_opaque_t) == sizeof(gex_AM_SrcDesc_t));

      static_assert(GEX_WRAPPER_OK == GASNET_OK);
      static_assert(GEX_WRAPPER_ERR_NOT_READY == GASNET_ERR_NOT_READY);
      assert((uintptr_t)GEX_WRAPPER_EP_INVALID == (uintptr_t)GEX_EP_INVALID);
      assert((uintptr_t)GEX_WRAPPER_MK_INVALID == (uintptr_t)GEX_MK_INVALID);
      assert((uintptr_t)GEX_WRAPPER_SEGMENT_INVALID == (uintptr_t)GEX_SEGMENT_INVALID);
      assert((uintptr_t)GEX_WRAPPER_EVENT_INVALID == (uintptr_t)GEX_EVENT_INVALID);
      assert((uintptr_t)GEX_WRAPPER_EVENT_NO_OP == (uintptr_t)GEX_EVENT_NO_OP);
      assert((uintptr_t)GEX_WRAPPER_EVENT_NOW == (uintptr_t)GEX_EVENT_NOW);
      assert((uintptr_t)GEX_WRAPPER_EVENT_DEFER == (uintptr_t)GEX_EVENT_DEFER);
      assert((uintptr_t)GEX_WRAPPER_EVENT_GROUP == (uintptr_t)GEX_EVENT_GROUP);
      static_assert(GEX_WRAPPER_DT_U64 == GEX_DT_U64);
      static_assert(GEX_WRAPPER_OP_ADD == GEX_OP_ADD);
      static_assert(GEX_WRAPPER_EP_CAPABILITY_RMA == GEX_EP_CAPABILITY_RMA);
      assert((uintptr_t)GEX_WRAPPER_MK_HOST == (uintptr_t)GEX_MK_HOST);
      static_assert(GEX_WRAPPER_FLAG_IMMEDIATE == GEX_FLAG_IMMEDIATE);
      static_assert(GEX_WRAPPER_FLAG_AM_PREPARE_LEAST_CLIENT ==
                    GEX_FLAG_AM_PREPARE_LEAST_CLIENT);
      static_assert(GEX_WRAPPER_FLAG_AM_PREPARE_LEAST_ALLOC ==
                    GEX_FLAG_AM_PREPARE_LEAST_ALLOC);

      static_assert(std::is_pod<gex_callback_handle_t>::value);
    }

  }; // namespace GASNetEXWrapper

}; // namespace Realm

extern "C" {

int realm_gex_wrapper_init(gex_wrapper_handle_t *handle)
{
  // Invalid argument
  if(handle == nullptr) {
    return -1;
  }

  // Failed verison check
  if(handle->handle_size < sizeof(*handle)) {
    return -1;
  }

  // set some gasnet-related environment variables, taking care not to
  //  overwrite anything explicitly set by the user

  // do not probe amount of pinnable memory
  setenv("GASNET_PHYSMEM_PROBE", "0", 0 /*no overwrite*/);

  // do not comment about on-demand-paging, which we are uninterested in
  setenv("GASNET_ODP_VERBOSE", "0", 0 /*no overwrite*/);

  // if we are using the ibv conduit with multiple-hca support, we need
  //  to enable fenced puts to work around gasnet bug 3447
  //  (https://gasnet-bugs.lbl.gov/bugzilla/show_bug.cgi?id=3447), but
  //  we can't set the flag if gasnet does NOT have multiple-hca support
  //  because it'll print warnings
  // in 2021.3.0 and earlier releases, there is no official way to detect
  //  this, and we can't even see the internal GASNETC_HAVE_FENCED_PUTS
  //  define, so we use the same condition that's used to set that in
  //  gasnet_core_internal.h and hope it doesn't change
  // releases after 2021.3.0 will define/expose GASNET_IBV_MULTIRAIL for us
  //  to look at
#if GASNET_IBV_MULTIRAIL || GASNETC_IBV_MAX_HCAS_CONFIGURE
  setenv("GASNET_USE_FENCED_PUTS", "1", 0 /*no overwrite*/);
#endif

  Realm::GASNetEXWrapper::check_size();

  handle->max_eps = GASNET_MAXEPS;
  handle->conduit = -1;
#if defined(GASNET_CONDUIT_ARIES)
  handle->conduit = GEX_WRAPPER_CONDUIT_ARIES;
#elif defined(GASNET_CONDUIT_IBV)
  handle->conduit = GEX_WRAPPER_CONDUIT_IBV;
#elif defined(GASNET_CONDUIT_MPI)
  handle->conduit = GEX_WRAPPER_CONDUIT_MPI;
#elif defined(GASNET_CONDUIT_OFI)
  handle->conduit = GEX_WRAPPER_CONDUIT_OFI;
#elif defined(GASNET_CONDUIT_SMP)
  handle->conduit = GEX_WRAPPER_CONDUIT_SMP;
#elif defined(GASNET_CONDUIT_UCX)
  handle->conduit = GEX_WRAPPER_CONDUIT_UCX;
#elif defined(GASNET_CONDUIT_UDP)
  handle->conduit = GEX_WRAPPER_CONDUIT_UDP;
#endif
  assert(handle->conduit != -1);

  handle->AM_LUBRequestMedium = gex_AM_LUBRequestMedium();
  handle->GEX_RELEASE = REALM_GEX_RELEASE;
  handle->GEX_API = REALM_GEX_API;

#ifdef REALM_GEX_RMA_HONORS_IMMEDIATE_FLAG
  handle->GEX_RMA_HONORS_IMMEDIATE_FLAG = true;
#else
  handle->GEX_RMA_HONORS_IMMEDIATE_FLAG = false;
#endif

#ifdef GASNET_NATIVE_NP_ALLOC_REQ_MEDIUM
  handle->GEX_NATIVE_NP_ALLOC_REQ_MEDIUM = true;
#else
  handle->GEX_NATIVE_NP_ALLOC_REQ_MEDIUM = false;
#endif

#if defined(GASNET_HAVE_MK_CLASS_CUDA_UVA)
  handle->GEX_HAVE_MK_CLASS_CUDA_UVA = true;
#else
  handle->GEX_HAVE_MK_CLASS_CUDA_UVA = false;
#endif

#if defined(GASNET_HAVE_MK_CLASS_HIP)
  handle->GEX_HAVE_MK_CLASS_HIP = true;
#else
  handle->GEX_HAVE_MK_CLASS_HIP = false;
#endif

  /* handlers */
  Realm::GASNetEXHandlers::init_gex_handler_fnptr(handle);

  /* combined */
  handle->gex_client_init = Realm::GASNetEXWrapper::gex_client_init;
  handle->gex_ep_create = Realm::GASNetEXWrapper::gex_ep_create;
  handle->gex_segment_attach_query_addr =
      Realm::GASNetEXWrapper::gex_segment_attach_query_addr;
  handle->gex_mk_create_cuda = Realm::GASNetEXWrapper::gex_mk_create_cuda;
  handle->gex_mk_create_hip = Realm::GASNetEXWrapper::gex_mk_create_hip;
  handle->gex_segment_create = Realm::GASNetEXWrapper::gex_segment_create;
  handle->gex_ep_bind_segment = Realm::GASNetEXWrapper::gex_ep_bind_segment;
  handle->gex_query_shared_peers = Realm::GASNetEXWrapper::gex_query_shared_peers;
  handle->gex_nbi_wait_ec_am = Realm::GASNetEXWrapper::gex_nbi_wait_ec_am;

  /* wrapper */
  handle->gex_error_name = Realm::GASNetEXWrapper::gex_error_name;
  handle->gex_error_desc = Realm::GASNetEXWrapper::gex_error_desc;
  handle->gex_ep_publish_bound_segment =
      Realm::GASNetEXWrapper::gex_ep_publish_bound_segment;
  handle->gex_event_test = Realm::GASNetEXWrapper::gex_event_test;
  handle->gex_am_src_desc_size = Realm::GASNetEXWrapper::gex_am_src_desc_size;
  handle->gex_am_src_desc_addr = Realm::GASNetEXWrapper::gex_am_src_desc_addr;
  handle->gex_am_poll = Realm::GASNetEXWrapper::gex_am_poll;

  /* rma */
  handle->gex_rma_iget = Realm::GASNetEXWrapper::gex_rma_iget;
  handle->gex_rma_iput = Realm::GASNetEXWrapper::gex_rma_iput;

  /* collective */
  handle->gex_coll_barrier = Realm::GASNetEXWrapper::gex_coll_barrier;
  handle->gex_coll_bcast = Realm::GASNetEXWrapper::gex_coll_bcast;
  handle->gex_coll_gather = Realm::GASNetEXWrapper::gex_coll_gather;
  handle->gex_coll_allgather = Realm::GASNetEXWrapper::gex_coll_allgather;
  handle->gex_coll_allgatherv = Realm::GASNetEXWrapper::gex_coll_allgatherv;
  handle->gex_coll_ireduce = Realm::GASNetEXWrapper::gex_coll_ireduce;

  return 0;
}
}
