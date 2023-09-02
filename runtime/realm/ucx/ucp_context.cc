
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

#include "realm/ucx/ucp_context.h"
#include "realm/ucx/ucp_utils.h"

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
  // class UCPWorker
  //

  bool UCPWorker::setup_worker_efd()
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

  bool UCPWorker::needs_progress() const
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

  bool UCPWorker::progress_with_wakeup()
  {
    ucs_status_t status;

    if (!needs_progress()) return true;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    int c = prog_itr_max;
    while (c-- && ucp_worker_progress(worker));

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

  bool UCPWorker::progress_without_wakeup()
  {
#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif
    int c = prog_itr_max;
    while (c-- && ucp_worker_progress(worker));

    return true;
  }

  bool UCPWorker::progress()
  {
    if (type == WORKER_RX) {
      // rx workers should first release some rdesc (persistent UCX buffers)
      if (am_rdesc_q_spinlock.trylock()) {
        int c = rdesc_rel_max;
        while (c-- && !am_rdesc_q.empty()) {
          ucp_am_data_release(worker, am_rdesc_q.front());
          am_rdesc_q.pop();
        }
        am_rdesc_q_spinlock.unlock();
      }
    } else {
      // tx workers should yield to submitters for a few tries
      if (scount.load() > pcount.load() && prog_boff_count++ < prog_boff_max) {
        return true;
      }
      prog_boff_count = 0;
    }

    return use_wakeup ? progress_with_wakeup()
                      : progress_without_wakeup();
  }

  bool UCPWorker::set_am_handler(unsigned am_id,
      ucp_am_recv_callback_t cb, void *args)
  {
    ucp_am_handler_param_t param;
    ucs_status_t status;

    assert(type == WORKER_RX);

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = am_id;
    param.cb         = cb;
    param.arg        = args;

    status           = ucp_worker_set_am_recv_handler(worker, &param);
    if (status != UCS_OK) return false;

    return true;
  }

  void UCPWorker::return_am_rdesc(void *rdesc)
  {
    assert(type == WORKER_RX);
    AutoLock<SpinLock> al(am_rdesc_q_spinlock);
    am_rdesc_q.push(rdesc);
  }

  bool UCPWorker::am_send_fast_path(ucp_ep_h ep, unsigned am_id,
      const void *header, size_t header_size,
      const void *payload, size_t payload_size,
      ucs_memory_type_t memtype)
  {
    ucp_request_param_t param;
    ucs_status_ptr_t status_ptr;

    assert(type == WORKER_TX);

    param.op_attr_mask = UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL |
                         UCP_OP_ATTR_FIELD_MEMORY_TYPE   |
                         UCP_OP_ATTR_FIELD_FLAGS;

    param.memory_type  = memtype;
    param.flags        = UCP_AM_SEND_FLAG_COPY_HEADER;

    scount.fetch_add_acqrel(1);
    status_ptr = ucp_am_send_nbx(ep, am_id,
        header, header_size, payload, payload_size, &param);
    pcount.fetch_add_acqrel(1);

    return status_ptr == NULL;
  }

  bool UCPWorker::submit_req(Request *req)
  {
    ucp_request_param_t param;
    ucs_status_ptr_t status_ptr;

    assert(type == WORKER_TX);

    param.op_attr_mask = UCP_OP_ATTR_FIELD_REQUEST     |
                         UCP_OP_ATTR_FIELD_CALLBACK    |
                         UCP_OP_ATTR_FIELD_USER_DATA   |
                         UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.request      = req;
    param.cb.send      = req->cb,
    param.user_data    = req->args,
    param.memory_type  = req->memtype,
    param.flags        = req->flags;

    scount.fetch_add_acqrel(1);

    switch (req->op_type) {
      case AM_SEND:
        param.flags |= UCP_AM_SEND_FLAG_COPY_HEADER;
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

  UCPWorker::UCPWorker(const UCPContext *context_, Type type_,
      size_t am_alignment_, bool use_wakeup_,
      unsigned prog_boff_max_, int prog_itr_max_, int rdesc_rel_max_,
      ucs_thread_mode_t thread_mode_,
      size_t user_req_size_, size_t user_req_alignment_,
      size_t pbuf_max_size_, size_t pbuf_max_chunk_size_,
      size_t pbuf_max_count_, size_t pbuf_init_count_,
      size_t mmp_max_obj_size_, bool leak_check_)
    : context(context_)
    , type(type_)
    , ucp_req_size(context->get_ucp_req_size())
    , am_alignment(am_alignment_)
    , use_wakeup(use_wakeup_)
    , prog_boff_max(prog_boff_max_)
    , prog_itr_max(prog_itr_max_)
    , rdesc_rel_max(rdesc_rel_max_)
    , thread_mode(thread_mode_)
    , pbuf_max_size(pbuf_max_size_)
  {
    // request mpoll
    request_mp = new MPool("request_mp", leak_check_,
        user_req_size_ + ucp_req_size, user_req_alignment_, ucp_req_size);

    // pre-registered payload buffer mpool
    pbuf_mp = new MPool("pbuf_mp", leak_check_,
        pbuf_max_size_, am_alignment_, 0, 1024,
        pbuf_init_count_, pbuf_max_count_, pbuf_max_chunk_size_, 1.5,
        &UCPWorker::pbuf_chunk_alloc, this,
        &UCPWorker::pbuf_chunk_release, this,
        nullptr, nullptr, nullptr, nullptr);

    // simple malloc mpool
    mmp = new VMPool("internal_malloc_mp", leak_check_,
        mmp_max_obj_size_, am_alignment_);
  }

  UCPWorker::~UCPWorker()
  {
    delete request_mp;
    delete pbuf_mp;
    delete mmp;
  }

  bool UCPWorker::init()
  {
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t worker_attr;
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    // TODO: what thread mode do we actually need?
    worker_params.field_mask   = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                 UCP_WORKER_PARAM_FIELD_AM_ALIGNMENT;
    worker_params.thread_mode  = thread_mode;
    worker_params.am_alignment = am_alignment;

    status = ucp_worker_create(context->get_ucp_context(),
        &worker_params, &worker);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_create failed", log_ucp, err);

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE |
                             UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

    status = ucp_worker_query(worker, &worker_attr);
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_query failed", log_ucp, err_destroy_worker);

    if (thread_mode == UCS_THREAD_MODE_MULTI &&
        worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
      log_ucp.error() << "UCP worker does not support multiple-thread mode";
      goto err_destroy_worker;
    }

    max_am_header = worker_attr.max_am_header;

    if (use_wakeup) {
      CHKERR_JUMP(!setup_worker_efd(), "", log_ucp, err_destroy_worker);
    }

    initialized = true;
    log_ucp.info() << "initialized ucp worker " << this
                   << " context "               << context
                   << " max_am_header "         << max_am_header;
    return true;

err_destroy_worker:
    ucp_worker_destroy(worker);
err:
    return false;
  }

  void UCPWorker::finalize()
  {
    assert(initialized);

    log_ucp.debug() << "finalizing ucp worker" << this;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    if (use_wakeup) close(worker_efd);
    ucp_worker_destroy(worker);
    log_ucp.debug() << "destroyed ucp worker " << worker;
    initialized = false;
  }

  void *UCPWorker::pbuf_chunk_alloc(size_t size, void *arg)
  {
    UCPWorker *worker = static_cast<UCPWorker*>(arg);
    ucp_mem_h mem_h;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(worker->get_context()->gpu);
#endif

    params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                         UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                         UCP_MEM_MAP_PARAM_FIELD_FLAGS   |
                         UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.address     = NULL;
    params.length      = size;
    params.flags       = UCP_MEM_MAP_ALLOCATE;
    params.memory_type = UCS_MEMORY_TYPE_HOST;

    CHKERR_JUMP(!worker->context->mem_map(&params, &mem_h),
        "pbuf_chunk_alloc mem_map failed", log_ucp, err);

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                      UCP_MEM_ATTR_FIELD_LENGTH;
    CHKERR_JUMP(ucp_mem_query(mem_h, &attr) != UCS_OK,
        "pbuf_chunk_alloc mem_query failed", log_ucp, err);

    assert(worker->pbuf_mp_mem_hs.count(attr.address) == 0);
    worker->pbuf_mp_mem_hs[attr.address] = mem_h;
    return attr.address;

err:
    return nullptr;
  }

  void UCPWorker::pbuf_chunk_release(void *chunk, void *arg)
  {
    UCPWorker *worker = static_cast<UCPWorker*>(arg);
    auto iter         = worker->pbuf_mp_mem_hs.find(chunk);
    assert(iter != worker->pbuf_mp_mem_hs.end());

    if (worker->context->mem_unmap(iter->second)) {
      worker->pbuf_mp_mem_hs.erase(iter);
    } else {
      log_ucp.error() << "pbuf_chunk_release failed for chunk " << chunk;
    }
  }

  bool UCPWorker::ep_add(int target, ucp_address_t *addr, int remote_dev_index)
  {
    ucp_ep_h ep;
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = addr;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
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

  bool UCPWorker::ep_get(int target, int remote_dev_index, ucp_ep_h *ep) const
  {
    auto iter1 = eps.find(target);
    assert(iter1 != eps.end());

    auto iter2 = iter1->second.find(remote_dev_index);
    assert(iter2 != iter1->second.end());

    *ep = iter2->second;

    return true;
  }

  bool UCPWorker::ep_close(ucp_ep_h ep)
  {
    ucp_request_param_t param = {0};
    ucs_status_t status;
    ucs_status_ptr_t close_req;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

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

  size_t UCPWorker::num_eps() const
  {
    size_t count = 0;
    for (const auto &kv1 : eps) {
      count += kv1.second.size();
    }
    return count;
  }

  void *UCPWorker::request_get()
  {
    char *buf;

    {
      AutoLock<SpinLock> al(req_mp_spinlock);
      buf = reinterpret_cast<char*>(request_mp->get());
    }
    if (!buf) return nullptr;

    return buf + ucp_req_size;
  }

  void UCPWorker::request_release(void *req)
  {
    char *buf = reinterpret_cast<char*>(req) - ucp_req_size;
    AutoLock<SpinLock> al(req_mp_spinlock);
    MPool::put(buf);
  }

  void *UCPWorker::pbuf_get(size_t size)
  {
    assert(size <= pbuf_max_size);

    AutoLock<SpinLock> al(pbuf_mp_spinlock);
    return pbuf_mp->get();
  }

  void UCPWorker::pbuf_release(void *buf)
  {
    AutoLock<SpinLock> al(pbuf_mp_spinlock);
    MPool::put(buf);
  }

  void *UCPWorker::mmp_get(size_t size)
  {
    AutoLock<SpinLock> al(mmp_spinlock);
    return mmp->get(size);
  }

  void UCPWorker::mmp_release(void *buf)
  {
    AutoLock<SpinLock> al(mmp_spinlock);
    VMPool::put(buf);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPContext
  //

  UCPContext::UCPContext(int ep_nums_est_)
    : ep_nums_est(ep_nums_est_)
  {}

  UCPContext::~UCPContext()
  {}

  bool UCPContext::init(
      const std::unordered_map<std::string, std::string> &ev_map)
  {
    ucp_config_t *ucp_config;
    ucp_params_t ucp_params;
    ucp_context_attr_t context_attr;
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    // read ucp config once for all contexts
    status = ucp_config_read(NULL, NULL, &ucp_config);
    CHKERR_JUMP(status != UCS_OK, "ucp_config_read failed", log_ucp, err);

    for (const auto &kv : ev_map) {
      // "UCX_" should be removed from config name
      const char *config_name = &kv.first.c_str()[4];

      status = ucp_config_modify(ucp_config, config_name, kv.second.c_str());
      if (status != UCS_OK) {
        log_ucp.error() << "ucp_config_modify failed "
                        << kv.first << " " << kv.second;
        goto err_rel_config;
      } else{
        log_ucp.info() << kv.first << " modified to " << kv.second
                       << " for context " << this;
      }
    }

    // Initialize UCX context
    ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                                   UCP_PARAM_FIELD_MT_WORKERS_SHARED;

    ucp_params.features          = UCP_FEATURE_AM  |
                                   UCP_FEATURE_RMA |
                                   UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = 1;

    if (ep_nums_est != -1) {
      ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
      ucp_params.estimated_num_eps = ep_nums_est;
    }

    status = ucp_init(&ucp_params, ucp_config, &context);

    CHKERR_JUMP(status != UCS_OK, "ucp_init failed", log_ucp, err_rel_config);

    context_attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
    status = ucp_context_query(context, &context_attr);
    CHKERR_JUMP(status != UCS_OK, "ucp_context_query failed",
        log_ucp, err_cleanup_context);

    ucp_req_size = context_attr.request_size;

    ucp_config_release(ucp_config);

    initialized = true;
    log_ucp.info() << "initialized ucp context " << this
#ifdef REALM_USE_CUDA
                   << (gpu ? " dev_index " : "")
                   << (gpu ? std::to_string(gpu->info->index) : "")
#endif
                   << " ep_nums_est " << ep_nums_est;

    return true;

err_cleanup_context:
    ucp_cleanup(context);
err_rel_config:
    ucp_config_release(ucp_config);
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

    ucp_cleanup(context);

    log_ucp.debug() << "cleaned up ucp context " << this;
    initialized = false;
  }

  bool UCPContext::mem_map(const ucp_mem_map_params_t *params,
      ucp_mem_h *mem_h_ptr) const
  {
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    status = ucp_mem_map(context, params, mem_h_ptr);
    if (status != UCS_OK) {
      log_ucp.error() << "ucp_mem_map failed";
      return false;
    }

    return true;
  }

  bool UCPContext::mem_unmap(ucp_mem_h mem_h) const
  {
    ucs_status_t status;

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(gpu);
#endif

    status = ucp_mem_unmap(context, mem_h);
    if (status != UCS_OK) {
      log_ucp.error() << "ucp_mem_unmap failed";
      return false;
    }

    return true;
  }

}; // namespace UCP

}; // namespace Realm
