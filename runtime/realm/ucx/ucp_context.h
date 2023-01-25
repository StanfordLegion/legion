
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

#include "ucp_utils.h"
#include "mpool.h"

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

  class UCPContext {
  public:
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
    struct InitParams {
      ucp_config_t *ucp_config;
      int ep_nums_est{-1};
      MPool::InitParams req_mp_params;
      size_t am_alignment{1};
      size_t pbuf_max_size{8 << 10 /* 8K */};
      size_t pbuf_max_chunk_size{4 << 20 /* 4M */};
      size_t pbuf_max_count{SIZE_MAX};
      size_t pbuf_init_count{1024};
      size_t mmp_max_obj_size{2 << 10 /* 2K */}; //malloc mpool max object size
      bool use_wakeup{false};
      unsigned prog_boff_max{4}; // progress thread maximum backoff
    };
    UCPContext();
    ~UCPContext();
    bool init(const InitParams &init_params,
      const std::unordered_map<std::string, std::string> &ev_map);
    void finalize();
    bool mem_map(const ucp_mem_map_params_t *params, ucp_mem_h *mem_h_ptr);
    void mem_unmap_all();
    bool ep_add(int target, ucp_address_t *addr, int remote_dev_index);
    bool ep_get(int target, int remote_dev_index, ucp_ep_h *ep) const;
    void *request_get();
    void request_release(void *req);
    void *pbuf_get(size_t size);
    void pbuf_release(void *buf);
    void *mmp_get(size_t size);
    void mmp_release(void *buf);
    ucp_worker_h get_ucp_worker() { return worker; }
    size_t get_max_am_header() const { return max_am_header; }
    bool progress();
    void return_am_rdesc(void *rdesc);
    bool am_send_fast_path(ucp_ep_h ep, unsigned am_id,
        const void *header, size_t header_size,
        const void *payload, size_t payload_size,
        ucs_memory_type_t memtype);
    bool submit_req(Request *req);
    size_t num_eps();

  private:
    bool open_context(int ep_nums_est, const ucp_config_t *config);
    bool create_worker();
    bool setup_worker_efd();
    bool ep_close(ucp_ep_h ep);
    bool needs_progress();
    bool progress_with_wakeup();
    bool progress_without_wakeup();
    static void *pbuf_chunk_alloc(size_t size, void *arg);
    static void pbuf_chunk_release(void *chunk, void *arg);

  public:
    ucp_context_h     context;
    ucp_worker_h      worker;
#ifdef REALM_USE_CUDA
    Cuda::GPU         *gpu{nullptr};
#endif
  private:
    bool initialized{false};
    bool use_wakeup{false};
    bool have_residual_events{false};
    int  worker_efd;
    size_t am_alignment;
    size_t request_size;
    MPool *request_mp;
    MPool *pbuf_mp;
    VMPool *mmp;
    Mutex req_mp_mutex;
    Mutex pbuf_mp_mutex;
    std::queue<void*> am_rdesc_q;
    Mutex am_rdesc_q_mutex;
    Mutex mmp_mutex;
    std::unordered_map<void*, ucp_mem_h> pbuf_mp_mem_hs;
    std::vector<ucp_mem_h> mem_hs;
    std::unordered_map<int, std::unordered_map<int, ucp_ep_h>> eps;
    size_t pbuf_max_size;
    size_t max_am_header;
    atomic<uint64_t> scount{0};
    atomic<uint64_t> pcount{0};
    unsigned prog_boff_count{0};
    unsigned prog_boff_max{0};
  };

}; // namespace UCP

}; // namespace Realm

#endif
