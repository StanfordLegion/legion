
/* Copyright 2023 NVIDIA Corporation
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

#include "realm/logging.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"
#endif

#include "ucp_context.h"
#include "ucp_utils.h"

#include <ucp/api/ucp.h>
#include <unordered_map>
#include <sys/epoll.h>
#include <unistd.h>

namespace Realm {

  // defined in ucp_module.cc
  extern Logger log_ucp;

namespace UCP {

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPContext
  //

  bool UCPContext::open_context(int ep_nums_est, const ucp_config_t *config)
  {
    ucp_params_t ucp_params;
    ucp_context_attr_t context_attr;
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    // Initialize UCX context
    ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                                   UCP_PARAM_FIELD_MT_WORKERS_SHARED;

    ucp_params.features          = UCP_FEATURE_AM  |
                                   UCP_FEATURE_RMA |
                                   UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = 0; // We use only one worker for now.

    if (ep_nums_est != -1) {
      ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
      ucp_params.estimated_num_eps = ep_nums_est;
    }

    status = ucp_init(&ucp_params, config, &context);

    CHKERR_JUMP(status != UCS_OK, "ucp_init failed", log_ucp, err);

    context_attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
    status = ucp_context_query(context, &context_attr);
    CHKERR_JUMP(status != UCS_OK, "ucp_context_query failed",
        log_ucp, err_cleanup_context);

    request_size = context_attr.request_size;

    return true;

err_cleanup_context:
    ucp_cleanup(context);
err:
    return false;
  }

  bool UCPContext::create_worker()
  {
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t worker_attr;
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    // TODO: what thread mode do we actually need?
    worker_params.field_mask   = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                 UCP_WORKER_PARAM_FIELD_AM_ALIGNMENT;
    worker_params.thread_mode  = UCS_THREAD_MODE_MULTI;
    worker_params.am_alignment = am_alignment;

    status = ucp_worker_create(context, &worker_params, &worker);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_create failed", log_ucp, err);

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE |
                             UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

    status = ucp_worker_query(worker, &worker_attr);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_query failed", log_ucp, err_destroy_worker);

    if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
      log_ucp.error() << "UCP worker does not support multiple-thread mode";
      goto err_destroy_worker;
    }

    max_am_header = worker_attr.max_am_header;

    return true;

err_destroy_worker:
    ucp_worker_destroy(worker);
err:
    return false;
  }

  bool UCPContext::setup_worker_efd()
  {
    int ret = 0;
    int efd = 0;
    struct epoll_event ev;
    ev.data.u64 = 0;
    ucs_status_t status;

    status = ucp_worker_get_efd(worker, &efd);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_get_efd failed", log_ucp, err);

    worker_efd = epoll_create(1);
    CHKERR_JUMP(worker_efd < 0, "epoll_create failed", log_ucp, err);

    ev.data.fd = efd;
    ev.events  = EPOLLIN;
    ret        = epoll_ctl(worker_efd, EPOLL_CTL_ADD, efd, &ev);
    CHKERR_JUMP(ret < 0, "epoll_ctl ADD failed", log_ucp, err_close_efd);

    status = ucp_worker_arm(worker);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_arm failed", log_ucp, err_close_efd);

    log_ucp.info() << "armed ucp worker " << worker << " context " << context;
    return true;

err_close_efd:
    close(worker_efd);
err:
    return false;
  }

  bool UCPContext::needs_progress()
  {
    // do a non-blocking wait on worker efd and
    // return true only if there are new events.
    struct epoll_event ev;

    if (have_residual_events) return true;

    int count = epoll_wait(worker_efd, &ev, 1, 0);
    if (count == -1) {
      log_ucp.error() << "epoll_wait failed"
                      << " context "    << context
                      << " worker "     << worker
                      << " worker_efd " << worker_efd;
    }

    return (count > 0);
  }

  bool UCPContext::progress_with_wakeup()
  {
    ucs_status_t status;
#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif
    if (!ucp_worker_progress(worker)) {
      status = ucp_worker_arm(worker);
      if (status == UCS_OK) {
        have_residual_events = false;
      } else if (status == UCS_ERR_BUSY) {
        have_residual_events = true;
      } else {
        CHKERR_JUMP(true, "ucp_worker_arm failed", log_ucp, err);
      }
    } else {
      have_residual_events = true;
    }

    return true;
err:
    return false;
  }

  bool UCPContext::progress_without_wakeup()
  {
#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif
    (void) ucp_worker_progress(worker);

    return true;
  }

  bool UCPContext::progress()
  {
    // yield to submitters for a few tries
    if (scount.load() > pcount.load() && prog_boff_count++ < prog_boff_max) {
        return true;
    }

    prog_boff_count = 0;

    // first release one rdesc if there are any
    void *am_rdesc = nullptr;
    {
      AutoLock<> al(am_rdesc_q_mutex);
      if (!am_rdesc_q.empty()) {
        am_rdesc = am_rdesc_q.front();
        am_rdesc_q.pop();
      }
    }
    if (am_rdesc) {
        ucp_am_data_release(worker, am_rdesc);
    }

    return use_wakeup ? progress_with_wakeup()
                      : progress_without_wakeup();
  }

  void UCPContext::return_am_rdesc(void *rdesc)
  {
    AutoLock<> al(am_rdesc_q_mutex);
    am_rdesc_q.push(rdesc);
  }

  bool UCPContext::am_send_fast_path(ucp_ep_h ep, unsigned am_id,
      const void *header, size_t header_size,
      const void *payload, size_t payload_size,
      ucs_memory_type_t memtype)
  {
    ucp_request_param_t param;
    ucs_status_ptr_t status_ptr;

    param.op_attr_mask = UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL |
                         UCP_OP_ATTR_FIELD_FLAGS;

    param.flags        = UCP_AM_SEND_FLAG_COPY_HEADER |
                         UCP_AM_SEND_FLAG_REPLY;

    scount.fetch_add_acqrel(1);
    status_ptr = ucp_am_send_nbx(ep, am_id,
        header, header_size, payload, payload_size, &param);
    pcount.fetch_add_acqrel(1);

    return status_ptr == NULL;
  }

  bool UCPContext::submit_req(Request *req)
  {
    ucp_request_param_t param;
    ucs_status_ptr_t status_ptr;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_REQUEST     |
                         UCP_OP_ATTR_FIELD_CALLBACK    |
                         UCP_OP_ATTR_FIELD_USER_DATA   |
                         UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.request      = req;
    param.cb.send      = req->cb,
    param.user_data    = req->args,
    param.memory_type  = req->memtype,
    param.flags        = req->flags |
                         UCP_AM_SEND_FLAG_COPY_HEADER |
                         UCP_AM_SEND_FLAG_REPLY;

    scount.fetch_add_acqrel(1);

    switch (req->op_type) {
      case AM_SEND:
        status_ptr = ucp_am_send_nbx(req->ep, req->am.id,
            req->am.header, req->am.header_size,
            req->payload, req->payload_size, &param);
        CHKERR_JUMP(UCS_PTR_IS_ERR(status_ptr),
            "ucp_am_send_nbx failed", log_ucp, err);
        break;
      case PUT:
        status_ptr = ucp_put_nbx(req->ep,
            req->payload, req->payload_size,
            req->rma.remote_addr, req->rma.rkey, &param);
        CHKERR_JUMP(UCS_PTR_IS_ERR(status_ptr),
            "ucp_put_nbx failed", log_ucp, err);
        break;
      case EP_FLUSH:
        status_ptr = ucp_ep_flush_nbx(req->ep, &param);
        CHKERR_JUMP(UCS_PTR_IS_ERR(status_ptr),
            "ucp_ep_flush_nbx failed", log_ucp, err);
        break;
      default:
        CHKERR_JUMP(true, "invalid ucp operation request", log_ucp, err);
    }

    pcount.fetch_add_acqrel(1);

    if (status_ptr == NULL) {
      // immediate completion
      req->cb(req, UCS_OK, req->args);
    }

    return true;

err:
    pcount.fetch_add_acqrel(1);
    return false;
  }

  UCPContext::UCPContext()
  {}

  UCPContext::~UCPContext()
  {}

  void *UCPContext::pbuf_chunk_alloc(size_t size, void *arg)
  {
    UCPContext *context = static_cast<UCPContext*>(arg);
    ucp_mem_h mem_h;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                         UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                         UCP_MEM_MAP_PARAM_FIELD_FLAGS   |
                         UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.address     = NULL;
    params.length      = size;
    params.flags       = UCP_MEM_MAP_ALLOCATE;
    params.memory_type = UCS_MEMORY_TYPE_HOST;

    status = ucp_mem_map(context->context, &params, &mem_h);
    CHKERR_JUMP(status != UCS_OK,
        "pbuf_chunk_alloc mem_map failed", log_ucp, err);

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                      UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(mem_h, &attr);
    CHKERR_JUMP(status != UCS_OK,
        "pbuf_chunk_alloc mem_query failed", log_ucp, err);

    assert(context->pbuf_mp_mem_hs.count(attr.address) == 0);
    context->pbuf_mp_mem_hs[attr.address] = mem_h;
    return attr.address;

err:
    return nullptr;
  }

  void UCPContext::pbuf_chunk_release(void *chunk, void *arg)
  {
    UCPContext *context = static_cast<UCPContext*>(arg);
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    auto iter = context->pbuf_mp_mem_hs.find(chunk);
    assert(iter != context->pbuf_mp_mem_hs.end());
    status = ucp_mem_unmap(context->context, iter->second);
    if (status != UCS_OK) {
      log_ucp.error() << "pbuf_chunk_release mem_unmap failed";
    }
    context->pbuf_mp_mem_hs.erase(iter);
  }

  bool UCPContext::init(const InitParams &init_params,
      const std::unordered_map<std::string, std::string> &ev_map)
  {
    am_alignment                    = init_params.am_alignment;
    use_wakeup                      = init_params.use_wakeup;
    prog_boff_max                   = init_params.prog_boff_max;
    pbuf_max_size                   = init_params.pbuf_max_size;
    MPool::InitParams req_mp_params = init_params.req_mp_params;
    MPool::InitParams pbuf_mp_params;
    VMPool::InitParams mmp_params;

    for (const auto &kv : ev_map) {
      // "UCX_" should be removed from config name
      const char *config_name = &kv.first.c_str()[4];

      ucs_status_t status = ucp_config_modify(init_params.ucp_config,
          config_name, kv.second.c_str());
      if (status != UCS_OK) {
        log_ucp.error() << "ucp_config_modify failed "
                        << kv.first << " " << kv.second;
        return false;
      } else{
        log_ucp.info() << kv.first << " modified to " << kv.second
                       << " for context " << this;
      }
    }

    bool ok = open_context(init_params.ep_nums_est, init_params.ucp_config);
    CHKERR_JUMP(!ok, "", log_ucp, err);

    ok = create_worker();
    CHKERR_JUMP(!ok, "", log_ucp, err_cleanup_context);

    if (use_wakeup) {
      ok = setup_worker_efd();
      CHKERR_JUMP(!ok, "", log_ucp, err_destroy_worker);
    }

    // update request mpool object size to account for ucp request
    req_mp_params.obj_size         += request_size;
    req_mp_params.alignment_offset += request_size;
    request_mp = new MPool(req_mp_params);

    // pre-registered payload buffer mpool
    pbuf_mp_params.obj_size          = init_params.pbuf_max_size;
    pbuf_mp_params.max_chunk_size    = init_params.pbuf_max_chunk_size;
    pbuf_mp_params.max_objs          = init_params.pbuf_max_count;
    pbuf_mp_params.init_num_objs     = init_params.pbuf_init_count;
    pbuf_mp_params.alignment         = am_alignment;
    pbuf_mp_params.alignment_offset  = 0;
    pbuf_mp_params.name              = "pbuf_mp";
    pbuf_mp_params.chunk_alloc       = &UCPContext::pbuf_chunk_alloc;
    pbuf_mp_params.chunk_release     = &UCPContext::pbuf_chunk_release;
    pbuf_mp_params.chunk_alloc_arg   = this;
    pbuf_mp_params.chunk_release_arg = this;

    pbuf_mp = new MPool(pbuf_mp_params);

    // simple malloc mpool
    mmp_params.max_obj_size = init_params.mmp_max_obj_size;
    mmp_params.alignment    = am_alignment;
    mmp_params.name         = "internal_malloc_mp";

    mmp = new VMPool(mmp_params);

    initialized = true;
    log_ucp.info() << "initialized ucp context " << this
#ifdef REALM_USE_CUDA
                   << (gpu ? " dev_index " : "")
                   << (gpu ? std::to_string(gpu->info->index) : "")
#endif
                   << " max_am_header " << max_am_header;

    return true;

err_destroy_worker:
    ucp_worker_destroy(worker);
err_cleanup_context:
    ucp_cleanup(context);
err:
    return false;

  }

  void UCPContext::finalize()
  {
    if (!initialized) return;

    log_ucp.debug() << "finalizing ucp context " << this;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    delete request_mp;
    delete pbuf_mp;
    delete mmp;
    if (use_wakeup) close(worker_efd);
    ucp_worker_destroy(worker);
    log_ucp.debug() << "destroyed ucp worker " << worker;
    ucp_cleanup(context);
    log_ucp.debug() << "cleaned up ucp context " << this;
    initialized = false;
  }

  bool UCPContext::mem_map(const ucp_mem_map_params_t *params,
      ucp_mem_h *mem_h_ptr)
  {
#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    ucs_status_t status = ucp_mem_map(context, params, mem_h_ptr);
    if (status != UCS_OK) return false;

    mem_hs.push_back(*mem_h_ptr);
    return true;
  }

  void UCPContext::mem_unmap_all()
  {
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    while (!mem_hs.empty()) {
      status = ucp_mem_unmap(context, mem_hs.back());
      if (status != UCS_OK) {
        log_ucp.info() << "ucp_mem_unmap failed";
      }
      mem_hs.pop_back();
    }
  }

  bool UCPContext::ep_add(int target, ucp_address_t *addr, int remote_dev_index)
  {
    ucp_ep_h ep;
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = addr;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    ucs_status_t status = ucp_ep_create(worker, &ep_params, &ep);
    CHKERR_JUMP(status != UCS_OK, "ucp_ep_create failed", log_ucp, err);

    // make sure we're not overwriting an existing ep
    assert(!eps[target].count(remote_dev_index));
    eps[target][remote_dev_index] = ep;

    return true;
err:
    return false;
  }

  bool UCPContext::ep_get(int target, int remote_dev_index, ucp_ep_h *ep) const
  {
    auto iter1 = eps.find(target);
    assert(iter1 != eps.end());

    auto iter2 = iter1->second.find(remote_dev_index);
    assert(iter2 != iter1->second.end());

    *ep = iter2->second;

    return true;
  }

  bool UCPContext::ep_close(ucp_ep_h ep)
  {
    ucp_request_param_t param = {0};
    ucs_status_t status;
    ucs_status_ptr_t close_req;

    close_req = ucp_ep_close_nbx(ep, &param);
    if (UCS_PTR_IS_ERR(close_req)) return false;

    if (close_req != NULL) {
      do {
          ucp_worker_progress(worker);
          status = ucp_request_check_status(close_req);
      } while (status == UCS_INPROGRESS);
      ucp_request_free(close_req);
    }

    return true;
  }

  size_t UCPContext::num_eps()
  {
    size_t count = 0;
    for (const auto &kv1 : eps) {
      count += kv1.second.size();
    }
    return count;
  }

  void *UCPContext::request_get()
  {
    char *buf;

    {
      AutoLock<> al(req_mp_mutex);
      buf = reinterpret_cast<char*>(request_mp->get());
    }
    if (!buf) return nullptr;

    return buf + request_size;
  }

  void UCPContext::request_release(void *req)
  {
    char *buf = reinterpret_cast<char*>(req) - request_size;
    AutoLock<> al(req_mp_mutex);
    MPool::put(buf);
  }

  void *UCPContext::pbuf_get(size_t size)
  {
    assert(size <= pbuf_max_size);

    AutoLock<> al(pbuf_mp_mutex);
    return pbuf_mp->get();
  }

  void UCPContext::pbuf_release(void *buf)
  {
    AutoLock<> al(pbuf_mp_mutex);
    MPool::put(buf);
  }

  void *UCPContext::mmp_get(size_t size)
  {
    AutoLock<> al(mmp_mutex);
    return mmp->get(size);
  }

  void UCPContext::mmp_release(void *buf)
  {
    AutoLock<> al(mmp_mutex);
    VMPool::put(buf);
  }

}; // namespace UCP

}; // namespace Realm
