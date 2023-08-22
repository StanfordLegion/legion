
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

// UCP network module internals

#ifndef UCP_INTERNAL_H
#define UCP_INTERNAL_H

#include "realm/mem_impl.h"
#include "realm/activemsg.h"
#include "realm/atomics.h"
#include "realm/mutex.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"
#endif

#include "ucp_module.h"
#include "ucp_context.h"
#include "bootstrap/bootstrap_internal.h"

#include <ucp/api/ucp.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace Realm {
namespace UCP {

  enum AmWithRemoteAddrMode {
    AM_WITH_REMOTE_ADDR_MODE_AUTO,
    AM_WITH_REMOTE_ADDR_MODE_PUT,
    AM_WITH_REMOTE_ADDR_MODE_AM
  };

  enum PayloadBaseType {
    PAYLOAD_BASE_INLINE,
    PAYLOAD_BASE_INTERNAL,
    PAYLOAD_BASE_EXTERNAL,
    PAYLOAD_BASE_LAST
  };

  struct CompList;
  struct RemoteComp;
  struct UCPRDMAInfo;
  struct Request;

  struct UCPMsgHdr {
    uint32_t       crc;
    NodeID         src;
    unsigned short msgid;
    RemoteComp     *remote_comp;
    void           *rdma_payload_addr;
    size_t         rdma_payload_size;
    char           realm_hdr[0];
  } __attribute__ ((packed)); // gcc-specific

  struct SegmentInfo {
    uintptr_t base, limit;
    NetworkSegmentInfo::MemoryType memtype;
    NetworkSegmentInfo::MemoryTypeExtraData memextra;
  };

  class UCPPoller : public BackgroundWorkItem {
  public:
    UCPPoller();
    ~UCPPoller() = default;
    void begin_polling();
    void end_polling();
    void wait_polling();
    bool do_work(TimeLimit work_until);
    void add_worker(UCPWorker *worker);

  private:
    std::vector<UCPWorker*> workers;
    Mutex shutdown_mutex;
    // set and cleared inside mutex, but tested outside
    atomic<bool> shutdown_flag;
    Mutex::CondVar shutdown_cond;
    // set and cleared inside mutex, but tested outside
    Mutex poll_notify_mutex;
    atomic<bool> poll_notify_flag;
    Mutex::CondVar poll_notify_cond;
  };

  class UCPInternal {
  public:
    friend class UCPMessageImpl;
    struct Config {
      AmWithRemoteAddrMode am_wra_mode{AM_WITH_REMOTE_ADDR_MODE_AUTO};
      bool bind_hostmem{true};
      int pollers_max{2};
      int prog_boff_max{4}; //progress thread maximum backoff
      int prog_itr_max{16};
      int rdesc_rel_max{16};
      bool mpool_leakcheck{false};
      bool crc_check{false};
      bool hbuf_malloc{false};
      bool pbuf_malloc{false};
      bool use_wakeup{false};
      size_t fp_max{2 << 10 /* 2K */}; //fast path max message size
      size_t pbuf_max_size{8 << 10 /* 8K */};
      size_t pbuf_max_chunk_size{4 << 20 /* 4M */};
      size_t pbuf_max_count{SIZE_MAX};
      size_t pbuf_init_count{1024};
      size_t pbuf_mp_thresh{2 << 10 /* 2K */};
      size_t mmp_max_obj_size{2 << 10 /* 2K */}; //malloc mpool max object size
      size_t outstanding_reqs_limit{1 << 17 /* 128K (count) */};
      std::string ib_seg_size;
      std::string host_nics;
      std::string zcopy_thresh_host;
      std::string tls_host;
#ifdef REALM_USE_CUDA
      bool bind_cudamem{true};
      std::string gpu_nics;
      std::string zcopy_thresh_dev;
      std::string tls_dev;
#endif
    };

    UCPInternal(Realm::UCPModule *_module, Realm::RuntimeImpl *_runtime);
    ~UCPInternal();
    bool bootstrap();
    bool init(const Config &config);
    void finalize();
    void attach(std::vector<NetworkSegment *>& segments);
    void detach(std::vector<NetworkSegment *>& segments);
    void barrier();
    void broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes);
    void gather(NodeID root, const void *val_in, void *vals_out, size_t bytes);
    size_t sample_messages_received_count();
    bool check_for_quiescence(size_t sampled_receive_count);
    size_t recommended_max_payload(const RemoteAddress *dest_payload_addr,
        bool with_congestion, size_t header_size);
    size_t recommended_max_payload(NodeID target,
        const RemoteAddress *dest_payload_addr,
        bool with_congestion, size_t header_size);
    size_t recommended_max_payload(const RemoteAddress *dest_payload_addr,
        const void *data, size_t bytes_per_line,
        size_t lines, size_t line_stride,
        bool with_congestion, size_t header_size);
    size_t recommended_max_payload(NodeID target,
        const RemoteAddress *dest_payload_addr,
        const void *data, size_t bytes_per_line,
        size_t lines, size_t line_stride,
        bool with_congestion, size_t header_size);

    bool get_ucp_ep(const UCPWorker *worker,
        NodeID target, const UCPRDMAInfo *rdma_info, ucp_ep_h *ep) const;

    Request *request_get(UCPWorker *worker);
    void request_release(Request *req);

    void *hbuf_get(UCPWorker *worker, size_t size);
    void hbuf_release(UCPWorker *worker, void *buf);

    void *pbuf_get(UCPWorker *worker, size_t size);
    void pbuf_release(UCPWorker *worker, void *buf);

    void notify_msg_sent(uint64_t count);

    const SegmentInfo *find_segment(const void *srcptr) const;

    const UCPContext *get_context(const SegmentInfo *seg_info) const;
    UCPWorker *get_worker(const UCPContext *context) const;

    size_t num_eps(const UCPContext &context) const;

  protected:
    UCPModule   *module;
    RuntimeImpl *runtime;

  private:
    struct AmHandlersArgs {
      UCPInternal *internal;
      UCPWorker   *worker;
    };
    struct SegmentInfoSorter;

#ifdef REALM_USE_CUDA
  bool init_ucp_contexts(const std::unordered_set<Realm::Cuda::GPU*> &gpus);
#else
  bool init_ucp_contexts();
#endif

    bool create_workers();
    void destroy_workers();
    bool set_am_handlers();
    bool create_eps();
    bool create_pollers();
    const UCPContext *get_context_host() const;
#if defined(REALM_USE_CUDA)
    const UCPContext *get_context_device(int dev_index) const;
#endif
    bool is_congested();
    bool add_rdma_info(NetworkSegment *segment,
        const UCPContext *context, ucp_mem_h mem_h);
    bool am_msg_recv_data_ready(UCPInternal *internal,
        UCPWorker *worker, ucp_ep_h reply_ep,
        const UCPMsgHdr *ucp_msg_hdr, size_t header_size,
        void *payload, size_t payload_size, int payload_mode);
    static ucs_status_t am_remote_comp_handler(void *arg,
        const void *header, size_t header_size,
        void *data, size_t data_size,
        const ucp_am_recv_param_t *param);
    static void am_send_reply_comp_handler(void *request,
        ucs_status_t status, void *user_data);
    static void am_realm_comp_handler(NodeID sender,
        IncomingMessageManager::CallbackData cb_data1,
        IncomingMessageManager::CallbackData cb_data2);
    static void am_rndv_recv_data_handler(void *request, ucs_status_t status,
        size_t length, void *user_data);
    static ucs_status_t am_msg_recv_handler(
        void *arg, const void *header, size_t header_size,
        void *payload, size_t payload_size,
        const ucp_am_recv_param_t *param);
    static ucs_status_t am_rdma_msg_recv_handler(
        void *arg, const void *header, size_t header_size,
        void *payload, size_t payload_size,
        const ucp_am_recv_param_t *param);

    using WorkersMap = std::unordered_map<const UCPContext*, UCPWorker*>;
    using AttachMap  = std::unordered_map<const UCPContext*, std::vector<ucp_mem_h>>;

    bool                                    initialized_boot{false};
    bool                                    initialized_ucp{false};
    Config                                  config;
    bootstrap_handle_t                      boot_handle;
    std::list<UCPContext>                   ucp_contexts;
#ifdef REALM_USE_CUDA
    std::unordered_map<int, UCPContext*>    dev_ctx_map;
#endif
    WorkersMap                              workers;
    std::list<UCPPoller>                    pollers;
    std::list<AmHandlersArgs>               am_handlers_args;
    AttachMap                               attach_mem_hs;
    atomic<uint64_t>                        total_msg_sent;
    atomic<uint64_t>                        total_msg_received;
    atomic<uint64_t>                        total_rcomp_sent;
    atomic<uint64_t>                        total_rcomp_received;
    atomic<uint64_t>                        outstanding_reqs;
    MPool                                   *rcba_mp;
    Mutex                                   rcba_mp_mutex;
    size_t                                  ib_seg_size;
    // this list is sorted by address to enable quick address lookup
    std::vector<SegmentInfo>                segments_by_addr;
  };

  class UCPMessageImpl : public ActiveMessageImpl {
  public:
    UCPMessageImpl(
      UCPInternal *internal,
      NodeID target,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      size_t storage_size);

    UCPMessageImpl(
      UCPInternal *internal,
      NodeID target,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      const RemoteAddress& dest_payload_addr,
      size_t storage_size);

    UCPMessageImpl(
      UCPInternal *internal,
      const NodeSet &targets,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      size_t storage_size);

    virtual ~UCPMessageImpl();

    virtual void *add_local_completion(size_t size);
    virtual void *add_remote_completion(size_t size);

    virtual void commit(size_t act_payload_size);
    virtual void cancel();

  private:
    void constructor_common(
      unsigned short _msgid,
      size_t _header_size,
      size_t _max_payload_size,
      size_t _storage_size);
    bool set_inline_payload_base();
    bool commit_with_rma(ucp_ep_h ep);
    bool commit_unicast(size_t act_payload_size);
    bool commit_multicast(size_t act_payload_size);
    bool send_fast_path(ucp_ep_h ep, size_t act_payload_size);
    bool send_slow_path(ucp_ep_h ep, size_t act_payload_size,
        uint32_t flags);
    Request *make_request(size_t act_payload_size);
    static void cleanup_request(Request *req, UCPInternal *internal);
    static bool send_request(Request *req, unsigned am_id);
    static void am_local_failure_handler(Request *req, UCPInternal *internal);
    static void am_local_comp_handler(void *request,
        ucs_status_t status, void *user_data);
    static void am_put_comp_handler(void *request,
        ucs_status_t status, void *user_data);
    static void am_put_flush_comp_handler(void *request,
        ucs_status_t status, void *user_data);

    UCPInternal *internal;
    UCPWorker   *worker;
    NodeID target;
    NodeSet targets;
    const void *src_payload_addr;
    size_t src_payload_lines;
    size_t src_payload_line_stride;
    size_t header_size;
    PayloadBaseType payload_base_type;
    UCPRDMAInfo *dest_payload_rdma_info{nullptr};
    CompList *local_comp{nullptr};
    RemoteComp *remote_comp{nullptr};
    bool is_multicast{false};
    ucs_memory_type_t memtype;

    UCPMsgHdr ucp_msg_hdr;
    // nothing should be added after 'ucp_msg_hdr'
  };

  class UCPRemoteMemoryCommon {
  public:
    UCPRemoteMemoryCommon(const ByteArray& rdma_info_ba);

    virtual ~UCPRemoteMemoryCommon();

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);

  private:
    UCPRDMAInfo *rdma_info;
    size_t rdma_info_size;
  };

  /* A block of memory on a remote process
   * Parent class RemoteMemory has node id in me.memory_owner_node()
   */
  class UCPRemoteMemory : public UCPRemoteMemoryCommon, public RemoteMemory {
  public:
    UCPRemoteMemory(Memory me,
        size_t size,
        Memory::Kind kind,
        const ByteArray& rdma_info_ba,
        UCPInternal *internal);

    virtual void get_bytes(off_t offset, void *dst, size_t size);
    virtual void put_bytes(off_t offset, const void *src, size_t size);
    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);
  };

  // Intermediate buffer memory
  class UCPIBMemory : public UCPRemoteMemoryCommon, public IBMemory {
  public:
    UCPIBMemory(Memory me,
        size_t size,
        Memory::Kind kind,
        const ByteArray& rdma_info_ba,
        UCPInternal *internal);

    virtual bool get_remote_addr(off_t offset, RemoteAddress& remote_addr);
  };

}; // namespace UCP

}; // namespace Realm

#endif
