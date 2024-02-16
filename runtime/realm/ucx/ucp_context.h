
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

#ifndef UCP_CONTEXT_H
#define UCP_CONTEXT_H

#include "realm/ucx/ucp_utils.h"
#include "realm/ucx/mpool.h"
#include "realm/ucx/spinlock.h"

#include "realm/atomics.h"
#include "realm/mutex.h"
#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#endif

#include <ucp/api/ucp.h>
#include <unordered_map>
#include <queue>

namespace Realm {
namespace UCP {

  class UCPContext;

  class UCPWorker {
  public:
    enum Type {
      WORKER_TX,
      WORKER_RX,
      WORKER_LAST
    };
    enum OpType{
      AM_SEND,
      PUT,
      EP_FLUSH,
    };
    struct Request {
      OpType                    op_type;
      ucp_ep_h                  ep;
      uint32_t                  flags;
      void                      *args;
      void                      *payload;
      size_t                    payload_size;
      ucs_memory_type_t         memtype;
      ucp_send_nbx_callback_t   cb;

      union {
        struct {
          unsigned              id;
          void                  *header;
          size_t                header_size;
        } am;

        struct {
          ucp_rkey_h            rkey;
          uint64_t              remote_addr;
        } rma;
      };

      // Should be allocated/freed only through get/release methods because
      // it must always have UCP-request-size bytes of space before itself.
      Request()  = delete;
      ~Request() = delete;
    };

    UCPWorker(const UCPContext *context, Type type,
        size_t am_alignment, bool use_wakeup,
        unsigned prog_boff_max /*progress thread maximum backoff*/,
        int prog_itr_max, int rdesc_rel_max,
        ucs_thread_mode_t thread_mode,
        size_t user_req_size, size_t user_req_alignment,
        size_t pbuf_max_size, size_t pbuf_max_chunk_size,
        size_t pbuf_max_count, size_t pbuf_init_count,
        size_t mmp_max_obj_size, bool leak_check);
    ~UCPWorker();
    bool init();
    void finalize();
    bool ep_add(int target, ucp_address_t *addr, int remote_dev_index);
    bool ep_get(int target, int remote_dev_index, ucp_ep_h *ep) const;
    void *request_get();
    void request_release(void *req);
    void *pbuf_get(size_t size);
    void pbuf_release(void *buf);
    void *mmp_get(size_t size);
    void mmp_release(void *buf);
    size_t get_max_am_header() const { return max_am_header; }
    const UCPContext *get_context() const { return context; }
    ucp_worker_h get_ucp_worker() const { return worker; }
    bool set_am_handler(unsigned am_id, ucp_am_recv_callback_t cb, void *args);
    bool progress();
    void return_am_rdesc(void *rdesc);
    bool am_send_fast_path(ucp_ep_h ep, unsigned am_id,
        const void *header, size_t header_size,
        const void *payload, size_t payload_size,
        ucs_memory_type_t memtype);
    bool submit_req(Request *req);
    size_t num_eps() const;

  private:
    bool setup_worker_efd();
    bool ep_close(ucp_ep_h ep);
    bool needs_progress() const;
    bool progress_with_wakeup();
    bool progress_without_wakeup();
    static void *pbuf_chunk_alloc(size_t size, void *arg);
    static void pbuf_chunk_release(void *chunk, void *arg);

    const UCPContext *context;
    ucp_worker_h worker;
    bool initialized{false};
    bool have_residual_events{false};
    int  worker_efd;
    Type type;
    size_t ucp_req_size;
    size_t am_alignment;
    bool use_wakeup;
    unsigned prog_boff_max;
    int prog_itr_max;
    int rdesc_rel_max;
    ucs_thread_mode_t thread_mode;
    size_t pbuf_max_size;
    MPool *request_mp;
    MPool *pbuf_mp;
    VMPool *mmp;
    SpinLock req_mp_spinlock;
    SpinLock pbuf_mp_spinlock;
    std::queue<void*> am_rdesc_q;
    SpinLock am_rdesc_q_spinlock;
    SpinLock mmp_spinlock;
    std::unordered_map<void*, ucp_mem_h> pbuf_mp_mem_hs;
    std::unordered_map<int, std::unordered_map<int, ucp_ep_h>> eps;
    size_t max_am_header;
    atomic<uint64_t> scount{0};
    atomic<uint64_t> pcount{0};
    unsigned prog_boff_count{0};
  };

  class UCPContext {
  public:
    UCPContext(int ep_nums_est = -1);
    ~UCPContext();
    bool init(const std::unordered_map<std::string, std::string> &ev_map);
    void finalize();
    bool mem_map(const ucp_mem_map_params_t *params, ucp_mem_h *mem_h_ptr) const;
    bool mem_unmap(ucp_mem_h mem_h) const;
    ucp_context_h get_ucp_context() const { return context; }
    size_t get_ucp_req_size() const { return context_attr.request_size; }
    uint64_t supported_memtypes() const { return context_attr.memory_types; }

#ifdef REALM_USE_CUDA
    Cuda::GPU *gpu{nullptr};
#endif
  private:
    bool initialized{false};
    ucp_context_h context;
    ucp_context_attr_t context_attr;
    std::unordered_map<int, std::unordered_map<int, ucp_ep_h>> eps;
    int ep_nums_est;
  };

}; // namespace UCP

}; // namespace Realm

#endif
