
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

#include "realm/runtime_impl.h"
#include "realm/transfer/ib_memory.h"
#include "realm/logging.h"

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"
#endif

#include "realm/ucx/ucp_internal.h"
#include "realm/ucx/ucp_utils.h"

#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace Realm {

  // defined in ucp_module.cc
  extern Logger log_ucp;

  Logger log_ucp_am("ucpam");
  Logger log_ucp_seg("ucpseg");
  Logger log_ucp_ep("ucpep");
  Logger log_ucp_ar("ucpar");

namespace UCP {

  static constexpr unsigned AM_ID             = 1;
  static constexpr unsigned AM_ID_RDMA        = 2;
  static constexpr unsigned AM_ID_REPLY       = 3;
  static constexpr size_t   AM_ALIGNMENT      = 8;
  static constexpr size_t   GET_ZCOPY_MAX     = (1 << 30); // 1G
  static constexpr size_t   IB_SEG_SIZE       = (8 << 10); // 8K
  static constexpr size_t   ZCOPY_THRESH_HOST = (2 << 10); // 2K
#ifdef REALM_USE_CUDA
  static constexpr size_t   ZCOPY_THRESH_DEV  = 0;
#endif

  namespace ThreadLocal {
    REALM_THREAD_LOCAL const TimeLimit *ucp_work_until = nullptr;
  };

  struct CompList {
    size_t bytes{0};

    static const size_t TOTAL_CAPACITY = 256;
    typedef char Storage_unaligned[TOTAL_CAPACITY];
    REALM_ALIGNED_TYPE_CONST(Storage_aligned, Storage_unaligned,
        Realm::CompletionCallbackBase::ALIGNMENT);
    Storage_aligned storage;
  };

  struct RemoteComp {
    enum {
      REMOTE_COMP_FLAG_FAILURE = 1ul << 0
    };
    CompList        *comp_list;
    atomic<size_t>  remote_pending;
    uint8_t         flags;
    RemoteComp(size_t _remote_pending)
      : comp_list(new CompList)
      , remote_pending(_remote_pending)
      , flags(0)
    {}

    ~RemoteComp()
    {
      delete comp_list;
    }
  };

  struct UCPRDMAInfo {
    uint64_t reg_base;
    int dev_index;
    char rkey[0];

    UCPRDMAInfo() = delete;
    ~UCPRDMAInfo() = delete;
  } __attribute__ ((packed)); // gcc-specific

  struct MCDesc {
    enum {
      REQUEST_AM_FLAG_FAILURE = 1ul << 0
    };
    uint8_t           flags;
    atomic<size_t>    local_pending; // number of targets pending local
                                     // completion (to support multicast)
    MCDesc(size_t _local_pending)
      : flags(0)
      , local_pending(_local_pending)
    {}
  };

  struct Request {
    // UCPContext::Request must be the first field because
    // the space preceding it is used internally by ucp
    UCPWorker::Request        ucp;
    UCPInternal               *internal;
    UCPWorker                 *worker;
    union {
      struct {
        PayloadBaseType       payload_base_type;
        CompList              *local_comp;
        MCDesc                *mc_desc;
      } am_send;

      struct {
        void                  *header;
        void                  *payload;
        size_t                header_size;
        size_t                payload_size;
        int                   payload_mode;
        // header buffer from am rndv should always be freed
      } am_rndv_recv;

      struct {
        UCPRDMAInfo           *rdma_info_buf;
      } rma;
    };

    // Should be allocated/freed only through UCPInternal because it must
    // always have UCP-request-size bytes of available space before itself.
    Request() = delete;
    ~Request() = delete;
  };

  struct RealmCallbackArgs {
    UCPInternal    *internal;
    UCPWorker      *rx_worker;
    RemoteComp     *remote_comp;
    void           *payload; // Payload buffer to release
    int            payload_mode;
    int            remote_dev_index;
  };

  static uint32_t compute_packet_crc(const void *header_base,
				     size_t header_size, size_t payload_size)
  {
    uint32_t accum = 0xFFFFFFFF;
    accum = crc32c_accumulate(accum, &header_size, sizeof(header_size));
    accum = crc32c_accumulate(accum, &payload_size, sizeof(payload_size));
    accum = crc32c_accumulate(accum, header_base, header_size);
    return ~accum;
  }

  static void insert_packet_crc(UCPMsgHdr *ucp_msg_hdr,
      size_t realm_hdr_size, size_t payload_size)
  {
    // the crc field itself should not be included in CRC checksum
    ucp_msg_hdr->crc = compute_packet_crc(&ucp_msg_hdr->src,
        sizeof(*ucp_msg_hdr) - sizeof(ucp_msg_hdr->crc) + realm_hdr_size,
		payload_size);
  }

  static void verify_packet_crc(const UCPMsgHdr *ucp_msg_hdr,
      size_t realm_hdr_size, size_t payload_size)
  {
    // the crc field itself should not be included in CRC checksum
    uint32_t exp_crc, act_crc;
    size_t header_size = sizeof(*ucp_msg_hdr)
      - sizeof(ucp_msg_hdr->crc) + realm_hdr_size;
    exp_crc = ucp_msg_hdr->crc;
    act_crc = compute_packet_crc(&ucp_msg_hdr->src, header_size, payload_size);
    if(exp_crc != act_crc) {
      log_ucp.fatal() << "CRC MISMATCH: "
		      << " header_size=" << header_size
		      << " payload_size=" << payload_size
		      << " src=" << ucp_msg_hdr->src
		      << " msgid=" << ucp_msg_hdr->msgid
		      << " rcomp=" << ucp_msg_hdr->remote_comp
              << " rdma_payload_addr=" << ucp_msg_hdr->rdma_payload_addr
              << " rdma_payload_size=" << ucp_msg_hdr->rdma_payload_size
		      << " exp=" << std::hex << exp_crc
		      << " act=" << act_crc << std::dec;
      abort();
    }
  }

  static bool string_to_val_units(const std::string &s, size_t *value)
  {
    if (s == "inf") {
      *value = SIZE_MAX;
      return true;
    }

    errno = 0;
    char *pos;
    *value = strtoull(s.c_str(), &pos, 10);
    if(errno != 0) return false;
    char unit = tolower(*pos ? *pos++ : 'b');
    switch(unit) {
      case 'k': *value *= 1024; break;
      case 'm': *value *= 1048576; break;
      case 'g': *value *= 1073741824; break;
      case 't': *value *= 1099511627776LL; break;
      case 0:
      case 'b': break;
      default: return false;
    }
    // allow a trailing 'b' so that things like 'kb' work
    if(*pos && ((unit == 'b') || (tolower(*pos) != 'b')))
      return false;

    return true;
  }

  static void read_env_var_update_map(const std::string &ev,
      const std::string &default_val, std::string *var,
      std::unordered_map<std::string, std::string> *ev_map)
  {
    char *ev_val;

    // does two things:
    // 1. updates var based on a corresponsing env var given by ev
    //  - if var is not empty, ignore ev's value
    //  - otherwise, if the ev is set, use its value for var
    //  - otherwise, use the given default value for var
    // 2. adds {ev, *var} to ev_map if the resulting var value is not empty
    if (var->empty()) {
      *var = (ev_val = getenv(ev.c_str())) ? ev_val : default_val;
    }

    if (!var->empty()) {
      ev_map->insert({ev, *var});
    }
  }

  static ucs_memory_type_t realm2ucs_memtype(
      const NetworkSegmentInfo::MemoryType &memtype)
  {
    switch(memtype) {
      case NetworkSegmentInfo::HostMem : return UCS_MEMORY_TYPE_HOST;
#ifdef REALM_USE_CUDA
      case NetworkSegmentInfo::CudaDeviceMem : return UCS_MEMORY_TYPE_CUDA;
#endif
      default: log_ucp.fatal() << "unsupported realm memtype " << memtype;
               return UCS_MEMORY_TYPE_UNKNOWN;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPPoller
  //
  UCPPoller::UCPPoller()
    : BackgroundWorkItem("ucp-poll")
    , shutdown_flag(false)
    , shutdown_cond(shutdown_mutex)
    , poll_notify_flag(false)
    , poll_notify_cond(poll_notify_mutex)
  {}

  void UCPPoller::begin_polling()
  {
    make_active();
  }

  void UCPPoller::end_polling()
  {
    AutoLock<> al(shutdown_mutex);

    assert(!shutdown_flag.load());
    shutdown_flag.store(true);
    shutdown_cond.wait();
  }

  void UCPPoller::wait_polling()
  {
    // wait until do_work is called again
    AutoLock<> al(poll_notify_mutex);

    poll_notify_flag.store(true);
    poll_notify_cond.wait();
  }

  bool UCPPoller::do_work(TimeLimit work_until)
  {
    ThreadLocal::ucp_work_until = &work_until;

    for (auto worker : workers) {
      (void) worker->progress();
    }

    ThreadLocal::ucp_work_until = nullptr;

    // if a poll notify has been requested, wake the waiter
    if(poll_notify_flag.load()) {
      AutoLock<> al(poll_notify_mutex);
      poll_notify_flag.store(false);
      poll_notify_cond.broadcast();
    }

    // if a shutdown has been requested, wake the waiter - if not, requeue
    if(shutdown_flag.load()) {
      AutoLock<> al(shutdown_mutex);
      shutdown_flag.store(false);
      shutdown_cond.broadcast();
      return false;
    }

    return true;
  }

  void UCPPoller::add_worker(UCPWorker *worker)
  {
    workers.push_back(worker);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPInternal
  //

  UCPInternal::UCPInternal(UCPModule *_module, RuntimeImpl *_runtime)
    : module(_module)
    , runtime(_runtime)
    , total_msg_sent(0)
    , total_msg_received(0)
    , total_rcomp_sent(0)
    , total_rcomp_received(0)
    , outstanding_reqs(0)
  {}

  UCPInternal::~UCPInternal()
  {
    assert(!initialized_ucp && !initialized_boot);
  }

  bool UCPInternal::create_eps(uint8_t priority)
  {
    // the eps are created from a local send worker to a
    // remote recv worker. So, we need to allgahter the
    // worker address of all the rx workers.

    int world_size           = boot_handle.pg_size;
    int self_rank            = boot_handle.pg_rank;
    size_t self_num_rx_workers = workers.size();
    std::vector<size_t> all_num_rx_workers(world_size);
    std::vector<size_t> all_max_addrlen(world_size);
    std::vector<char>   self_rx_workers_addr;
    std::vector<char>   all_rx_workers_addr;
    std::vector<int>    self_dev_indices;
    std::vector<int>    all_dev_indices;
    size_t self_max_addrlen = 0, max_addrlen = 0, max_num_rx_workers = 0;
    char *buf;
    ucs_status_t status;
    int rc;
    bool ret = true;

    struct WorkerInfo {
      UCPWorker     *worker;
      ucp_address_t *addr;
      size_t        addrlen;
      int           dev_index;
    };

    // get details of all rx workers of my own contexts
    std::vector<WorkerInfo> self_rx_workers_info;
    for (const auto &context : ucp_contexts) {
      WorkerInfo wi;
      wi.worker    = get_rx_worker(&context, priority);
      wi.dev_index = -1; // implying host context by default
      status = ucp_worker_get_address(wi.worker->get_ucp_worker(),
          &wi.addr, &wi.addrlen);
      CHKERR_JUMP(status != UCS_OK, "ucp_worker_get_address failed", log_ucp, err);
#ifdef REALM_USE_CUDA
      if (context.gpu) {
        wi.dev_index = context.gpu->info->index;
      }
#endif
      self_rx_workers_info.push_back(wi);
      self_max_addrlen = std::max(self_max_addrlen, wi.addrlen);
    }

    // to avoid allgatherv, we find and use the maximum values across all ranks

    // find the global max rx worker address length across all ranks
    rc = boot_handle.allgather(&self_max_addrlen, all_max_addrlen.data(),
        sizeof(self_max_addrlen), &boot_handle);
    CHKERR_JUMP(rc != 0, "failed to allgather max worker address length",
        log_ucp, err);

    max_addrlen = *std::max_element(all_max_addrlen.begin(),
        all_max_addrlen.end());

    // get the number of rx workers created by each rank
    rc = boot_handle.allgather(&self_num_rx_workers, all_num_rx_workers.data(),
        sizeof(self_num_rx_workers), &boot_handle);
    CHKERR_JUMP(rc != 0, "failed to allgather the number of workers",
        log_ucp, err);

    max_num_rx_workers = *std::max_element(all_num_rx_workers.begin(),
        all_num_rx_workers.end());

    // gather the device index of all rx workers across all ranks
    for (const auto &wi : self_rx_workers_info) {
      self_dev_indices.push_back(wi.dev_index);
    }
    self_dev_indices.resize(max_num_rx_workers);
    all_dev_indices.resize(world_size * max_num_rx_workers);
    rc = boot_handle.allgather(self_dev_indices.data(), all_dev_indices.data(),
        max_num_rx_workers * sizeof(self_dev_indices[0]), &boot_handle);
    CHKERR_JUMP(rc != 0, "failed to allgather worker device indices",
        log_ucp, err);

    // gather the addresses of all rx workers across all ranks.
    // first copy all self rx worker addresses into a contiguous buffer
    self_rx_workers_addr.resize(max_num_rx_workers * max_addrlen);
    buf = self_rx_workers_addr.data();
    for (const auto &wi : self_rx_workers_info) {
      std::memcpy(buf, wi.addr, wi.addrlen);
      buf += max_addrlen;
    }
    // do the allgather now
    all_rx_workers_addr.resize(world_size * max_num_rx_workers * max_addrlen);
    rc = boot_handle.allgather(self_rx_workers_addr.data(), all_rx_workers_addr.data(),
        max_num_rx_workers * max_addrlen, &boot_handle);
    CHKERR_JUMP(rc != 0, "failed to allgather worker addresses", log_ucp, err);

    // create an ep from each local tx worker to each remote rx worker
    for (int peer = 0; peer < world_size; peer++) {
      if (peer == self_rank) continue;

      for (size_t i = 0; i < all_num_rx_workers[peer]; i++) {
        ucp_address_t *peer_worker_addr = reinterpret_cast<ucp_address_t*>(
            &all_rx_workers_addr[((peer * max_num_rx_workers) + i) * max_addrlen]);
        int peer_dev_index = all_dev_indices[(peer * max_num_rx_workers) + i];

        for (const auto &context : ucp_contexts) {
          UCPWorker *worker = get_tx_worker(&context, priority);
          CHKERR_JUMP(!worker->ep_add(peer, peer_worker_addr, peer_dev_index),
              "failed to add ep. peer " << peer <<
              " peer_dev_index "        << peer_dev_index, log_ucp, err);

          log_ucp_ep.debug()
            << "added ep target "   << peer
            << " context "          << &context
            << " worker "           << worker
            << " remote dev_index " << peer_dev_index;
        }
      }
    }

out:
    for (auto &wi : self_rx_workers_info) {
      ucp_worker_release_address(wi.worker->get_ucp_worker(), wi.addr);
    }
    return ret;

err:
    ret = false;
    goto out;
  }

  bool UCPInternal::create_eps()
  {
    for (uint8_t priority = 0; priority < config.num_priorities; priority++) {
      if (!create_eps(priority)) return false;
    }
    return true;
  }

  bool UCPInternal::create_workers()
  {
    const UCPContext *context;
    UCPWorker *worker;

    std::vector<UCPWorker::Type> wts = {
      UCPWorker::WORKER_TX,
      UCPWorker::WORKER_RX
    };

    // create num_priorities workers per worker type (tx, rx) per context
    context = get_context_host();
    for (const auto & wt : wts) {
      for (uint8_t priority = 0; priority < config.num_priorities; priority++) {
        worker = new UCPWorker(context, wt,
            AM_ALIGNMENT, config.use_wakeup, config.prog_boff_max,
            config.prog_itr_max, config.rdesc_rel_max,
            (wt == UCPWorker::WORKER_TX) ? UCS_THREAD_MODE_MULTI : UCS_THREAD_MODE_SERIALIZED,
            sizeof(Request), alignof(Request),
            config.pbuf_max_size + AM_ALIGNMENT, config.pbuf_max_chunk_size,
            config.pbuf_max_count, config.pbuf_init_count,
            config.mmp_max_obj_size, config.mpool_leakcheck);

        CHKERR_JUMP(!worker->init(),
            "failed to init worker for host context " << context, log_ucp, err);

        if (wt == UCPWorker::WORKER_TX) {
          workers[context].tx_workers.push_back(worker);
        } else {
          workers[context].rx_workers.push_back(worker);
        }
      }
    }

#ifdef REALM_USE_CUDA
    for (auto &kv : dev_ctx_map) {
      context = kv.second;
      for (const auto & wt : wts) {
        for (uint8_t priority = 0; priority < config.num_priorities; priority++) {
          worker = new UCPWorker(context, wt,
              AM_ALIGNMENT, config.use_wakeup, config.prog_boff_max,
              config.prog_itr_max, config.rdesc_rel_max,
              (wt == UCPWorker::WORKER_TX) ? UCS_THREAD_MODE_MULTI : UCS_THREAD_MODE_SERIALIZED,
              sizeof(Request), alignof(Request),
              1, 1, 0, 0, /* we never get payload buffer from GPU contexts */
              config.mmp_max_obj_size, config.mpool_leakcheck);

          CHKERR_JUMP(!worker->init(),
              "failed to init worker for device context " << context, log_ucp, err);

          if (wt == UCPWorker::WORKER_TX) {
            workers[context].tx_workers.push_back(worker);
          } else {
            workers[context].rx_workers.push_back(worker);
          }
        }
      }
    }
#endif

    return true;

err:
    destroy_workers();
    return false;
  }

  void UCPInternal::destroy_workers()
  {
    for (auto &kv : workers) {
      for (UCPWorker *worker : kv.second.tx_workers) {
        worker->finalize();
        delete worker;
      }
      for (UCPWorker *worker : kv.second.rx_workers) {
        worker->finalize();
        delete worker;
      }
    }
    workers.clear();
  }

  size_t UCPInternal::get_num_workers()
  {
    size_t num_workers = 0;
    for (const auto &kv : workers) {
      assert(kv.second.tx_workers.size() == kv.second.rx_workers.size());
      num_workers += 2 * kv.second.tx_workers.size();
    }
    return num_workers;
  }

  bool UCPInternal::set_am_handlers()
  {
    struct AmHandler {
      unsigned id;
      ucp_am_recv_callback_t cb;
      std::string name;
    };

    static std::vector<AmHandler> am_handlers = {
      {AM_ID, &am_msg_recv_handler, "am"},
      {AM_ID_RDMA, &am_rdma_msg_recv_handler, "rdma"},
      {AM_ID_REPLY, &am_remote_comp_handler, "am reply"}
    };

    // the handlers should be installed on the rx workers only
    for (const auto &context : ucp_contexts) {
      for (UCPWorker *worker : get_rx_workers(&context)) {
        am_handlers_args.push_back({this, worker});
        for (const auto &ah: am_handlers) {
          if (!worker->set_am_handler(ah.id, ah.cb, &am_handlers_args.back())) {
            log_ucp.error() << "failed to set ucp "  << ah.name
                            << " handler for worker " << worker;
            return false;
          }
        }
      }
    }

    return true;
  }

  bool UCPInternal::create_pollers()
  {
    // distribute ucp workers among poller background work items
    size_t num_txrx_workers        = get_num_workers() / 2;
    size_t txrx_workers_per_poller = (num_txrx_workers / config.pollers_max) +
                                     (num_txrx_workers % config.pollers_max != 0);
    size_t idx                     = 0;

    for (const auto &kv : workers) {
      for (size_t i = 0; i < kv.second.tx_workers.size(); i++) {
        if (!idx) pollers.emplace_back();
        pollers.back().add_worker(kv.second.tx_workers[i]);
        pollers.back().add_worker(kv.second.rx_workers[i]);
        idx = (idx + 1) % txrx_workers_per_poller;
      }
    }

    log_ucp.info() << "created " << pollers.size() << " ucp poller items";

    for (auto &poller : pollers) {
      poller.add_to_manager(&runtime->bgwork);
      poller.begin_polling();
    }

    return true;
  }

#ifdef REALM_USE_CUDA
  bool UCPInternal::init_ucp_contexts(
      const std::unordered_set<Cuda::GPU*> &gpus)
  {
    size_t ep_nums_est = boot_handle.pg_size * (1 + gpus.size());
    std::unordered_map<int, std::string> gpu_nics;
#else
  bool UCPInternal::init_ucp_contexts()
  {
    size_t ep_nums_est = boot_handle.pg_size;
#endif
    std::unordered_map<std::string, std::string> ev_map;
    size_t total_num_eps = 0;

    assert(!initialized_ucp);

    // payload buffer mpool config
    if (config.pbuf_init_count > config.pbuf_max_count) {
      log_ucp.info() << "pbuf_init_count "    << config.pbuf_init_count
                     << " gt pbuf_max_count " << config.pbuf_max_count
                     << "; decreased it to "  << config.pbuf_max_count;
      config.pbuf_init_count = config.pbuf_max_count;
    }
    if (config.pbuf_max_size > config.pbuf_max_chunk_size) {
      log_ucp.info() << "pbuf_max_size "           << config.pbuf_max_size
                     << " gt pbuf_max_chunk_size " << config.pbuf_max_chunk_size
                     << "; decreased it to "       << config.pbuf_max_chunk_size;
      config.pbuf_max_size = config.pbuf_max_chunk_size;
    }
    if (config.mmp_max_obj_size < config.pbuf_mp_thresh) {
      log_ucp.warning() << "WARNING: mmp_max_obj_size "
                        << config.mmp_max_obj_size
                        << " lt pbuf_mp_thresh "
                        << config.pbuf_mp_thresh;
    }

    read_env_var_update_map("UCX_NET_DEVICES", "", &config.host_nics, &ev_map);
    read_env_var_update_map("UCX_TLS", "", &config.tls_host, &ev_map);
    read_env_var_update_map("UCX_IB_SEG_SIZE",
        std::to_string(IB_SEG_SIZE), &config.ib_seg_size, &ev_map);
    read_env_var_update_map("UCX_ZCOPY_THRESH",
        std::to_string(ZCOPY_THRESH_HOST), &config.zcopy_thresh_host, &ev_map);

    CHKERR_JUMP(!string_to_val_units(config.ib_seg_size, &ib_seg_size),
        "failed to read ib_seg_size value", log_ucp, err);

    // create the host context
    ucp_contexts.emplace_back(ep_nums_est);

    CHKERR_JUMP(!ucp_contexts.back().init(ev_map),
        "failed to initialize host ucp context", log_ucp, err_fin_contexts);

#ifdef REALM_USE_CUDA
    // create a separate ucp context for each gpu observed in attach

    // first, parse gpu_nics config if present
    // example arg format: -ucx:gpu_nics 0:mlx5_1:1,mlx5_2:1;1:mlx5_3:1
    // 1. tokenize by ';'
    // 2. for each token, find the first ':' (only the first and left-most colon
    //    to support colons in the NIC names such as mlx5_0:1)
    // 3. CUDA device index on the left, comma-separated list of NICs on the right
    if (!config.gpu_nics.empty()) {
      std::istringstream stream(config.gpu_nics);
      std::string token;
      while (std::getline(stream, token, ';')) {
        const size_t c = token.find(':');
        if ((c != std::string::npos) && (c > 0)) {
          const int cuda_device_index = std::stoi(token.substr(0, c));
          const std::string nics = token.substr(c + 1);
          gpu_nics[cuda_device_index] = nics;
          log_ucp.info() << "NICs for CUDA device " << cuda_device_index << ": " << nics;
        }
      }
    }

    ev_map.erase("UCX_TLS");
    ev_map.erase("UCX_ZCOPY_THRESH");
    read_env_var_update_map("UCX_TLS", "", &config.tls_dev, &ev_map);
    read_env_var_update_map("UCX_ZCOPY_THRESH",
        std::to_string(ZCOPY_THRESH_DEV), &config.zcopy_thresh_dev, &ev_map);

    for (Cuda::GPU *gpu : gpus) {
      ev_map.erase("UCX_NET_DEVICES");
      auto it = gpu_nics.find(gpu->info->index);
      if (it != gpu_nics.end()) {
        read_env_var_update_map("UCX_NET_DEVICES", "", &it->second, &ev_map);
      }
      ucp_contexts.emplace_back(ep_nums_est);

      ucp_contexts.back().gpu = gpu;
      dev_ctx_map[gpu->info->index] = &ucp_contexts.back();
      CHKERR_JUMP(!ucp_contexts.back().init(ev_map),
          "failed to initialize gpu ucp context", log_ucp, err_fin_contexts);
    }
#endif

    CHKERR_JUMP(!create_workers(),
        "failed to create workers", log_ucp, err_fin_contexts);

    CHKERR_JUMP(!set_am_handlers(),
        "failed to set ucp am handlers", log_ucp, err_destroy_workers);

    CHKERR_JUMP(!create_eps(),
        "failed to create ucp end points", log_ucp, err_destroy_workers);

    CHKERR_JUMP(!create_pollers(),
        "failed to create ucp pollers", log_ucp, err_destroy_workers);

    initialized_ucp = true;
    log_ucp.info() << "initialized " << ucp_contexts.size() << " ucp contexts";

    for (UCPContext &context : ucp_contexts) {
      total_num_eps += num_eps(context);
      log_ucp_ep.info() << "context " << &context
                        << " num_eps " << num_eps(context);
    }

    log_ucp.info() << "total num_eps " << total_num_eps;

    return true;

err_destroy_workers:
    destroy_workers();
err_fin_contexts:
    while (!ucp_contexts.empty()) {
      ucp_contexts.back().finalize();
      ucp_contexts.pop_back();
    }
err:
    return false;
  }

  bool UCPInternal::bootstrap()
  {
    assert(!initialized_boot && !initialized_ucp);

    BootstrapConfig boot_config;

    const char *bootstrap_mode_str = getenv("REALM_UCP_BOOTSTRAP_MODE");
    if (bootstrap_mode_str == NULL) {
      // use MPI as the default bootstrap
      boot_config.mode = Realm::UCP::BOOTSTRAP_MPI;
    } else if (strcmp(bootstrap_mode_str, "mpi") == 0) {
        boot_config.mode = Realm::UCP::BOOTSTRAP_MPI;
    } else if (strcmp(bootstrap_mode_str, "plugin") == 0) {
        boot_config.mode = Realm::UCP::BOOTSTRAP_PLUGIN;
    } else {
      log_ucp.fatal() << "invalid UCP bootstrap mode %s" << bootstrap_mode_str;
      goto err;
    }

    boot_config.plugin_name = getenv("REALM_UCP_BOOTSTRAP_PLUGIN");

    CHKERR_JUMP(bootstrap_init(&boot_config, &boot_handle),
        "failed to bootstrap ucp", log_ucp, err);

    Network::my_node_id  = boot_handle.pg_rank;
    Network::max_node_id = boot_handle.pg_size - 1;
    Network::all_peers.add_range(0, boot_handle.pg_size - 1);
    Network::all_peers.remove(boot_handle.pg_rank);

    for(int i = 0; i < boot_handle.num_shared_ranks; i++) {
      if (boot_handle.shared_ranks[i] != boot_handle.pg_rank) {
        Network::shared_peers.add(boot_handle.shared_ranks[i]);
      }
    }

    initialized_boot = true;
    log_ucp.info() << "bootstrapped UCP network module";

    return true;

err:
    return false;
  }

  bool UCPInternal::init(const Config& _config)
  {
    config = _config;

    // RealmCallbackArgs mpool
    rcba_mp = new MPool("rcba_mp", config.mpool_leakcheck,
        sizeof(RealmCallbackArgs), alignof(RealmCallbackArgs), 0);

    return true;
    // most of initialization happens after attach()
  }

  void UCPInternal::finalize()
  {
    assert(initialized_boot);

    delete rcba_mp;

    destroy_workers();

    if (initialized_ucp) {
      for (UCPContext &context : ucp_contexts) {
        context.finalize();
      }
      log_ucp.info() << "finalized ucp contexts";

      initialized_ucp = false;
    }

    int rc = bootstrap_finalize(&boot_handle);
    if (rc != 0) {
      log_ucp.error() << "failed to finalize ucp bootstrap";
    } else {
      initialized_boot = false;
      log_ucp.info() << "finalized ucp bootstrap";
    }
  }

  ucs_status_t UCPInternal::am_remote_comp_handler(void *arg,
      const void *header, size_t header_size,
      void *data, size_t data_size,
      const ucp_am_recv_param_t *param)
  {
    RemoteComp *remote_comp;
    size_t remote_pending;
    AmHandlersArgs *am_args = reinterpret_cast<AmHandlersArgs*>(arg);
    UCPInternal *internal   = am_args->internal;

    log_ucp_am.debug() << "am_remote_comp_handler invoked";

    // The header value is the pointer to the corresponding remote_comp
    assert(header_size == sizeof(CompList*));
    memcpy(&remote_comp, header, header_size);

    remote_pending = remote_comp->remote_pending.fetch_sub(1);
    assert(remote_pending != 0);

    if (--remote_pending == 0) {
      if (!(remote_comp->flags & RemoteComp::REMOTE_COMP_FLAG_FAILURE)) {
        CompletionCallbackBase::invoke_all(remote_comp->comp_list->storage,
            remote_comp->comp_list->bytes);
        CompletionCallbackBase::destroy_all(remote_comp->comp_list->storage,
            remote_comp->comp_list->bytes);
      }
      delete remote_comp;
    }

    (void) internal->total_rcomp_received.fetch_add(1);
    return UCS_OK;
  }

  void UCPInternal::am_send_reply_comp_handler(void *request,
      ucs_status_t status, void *user_data)
  {
    Request *req          = reinterpret_cast<Request*>(user_data);
    UCPInternal *internal = req->internal;

    log_ucp_am.debug() << "am_send_reply_comp_handler invoked for request " << req;

    CHKERR_JUMP(status != UCS_OK, "failed to complete am reply", log_ucp, err);

    (void) internal->total_rcomp_sent.fetch_add(1);
    internal->request_release(req);

    return;

err:
    // TODO: should invoke some higher-level error handler
    internal->request_release(req);
  }

  void UCPInternal::am_realm_comp_handler(NodeID sender,
      IncomingMessageManager::CallbackData cb_data1,
      IncomingMessageManager::CallbackData cb_data2)
  {
    // This comp hanlder is NOT called by the progress thread.
    // So, set the correct GPU context must be set where needed.

    RealmCallbackArgs *cb_args = reinterpret_cast<RealmCallbackArgs*>(cb_data1);
    UCPWorker *rx_worker       = cb_args->rx_worker;
    UCPInternal *internal      = cb_args->internal;
    int remote_dev_index       = cb_args->remote_dev_index;
    Request *req;

    log_ucp_am.debug() << "am_realm_comp_handler invoked. Sender " << sender;

    // cb_data2 not used
    (void) cb_data2;

    if (cb_args->payload_mode == PAYLOAD_KEEP) {
#ifdef REALM_USE_CUDA
      Cuda::AutoGPUContext agc(rx_worker->get_context()->gpu);
#endif
      rx_worker->return_am_rdesc(cb_args->payload);
    } else if (cb_args->payload_mode == PAYLOAD_FREE) {
      internal->pbuf_release(rx_worker, cb_args->payload);
    }

    if (cb_args->remote_comp != nullptr) {
      // Send an active message to the sender with remote_comp as the header
      UCPWorker *tx_worker = internal->get_tx_worker(rx_worker->get_context(), 0);
      req = internal->request_get(tx_worker);
      CHKERR_JUMP(req == nullptr, "failed to get request", log_ucp, err);

      CHKERR_JUMP(!tx_worker->ep_get(sender, remote_dev_index, &req->ucp.ep),
          "failed to get reply ep", log_ucp, err_rel_req);

      req->ucp.op_type        = UCPWorker::OpType::AM_SEND;
      req->ucp.flags          = UCP_AM_SEND_FLAG_EAGER;
      req->ucp.args           = req;
      req->ucp.payload        = nullptr;
      req->ucp.payload_size   = 0;
      req->ucp.memtype        = UCS_MEMORY_TYPE_HOST;
      req->ucp.cb             = &UCPInternal::am_send_reply_comp_handler;
      req->ucp.am.id          = AM_ID_REPLY;
      req->ucp.am.header      = &cb_args->remote_comp;
      req->ucp.am.header_size = sizeof(cb_args->remote_comp);

      CHKERR_JUMP(!tx_worker->submit_req(&req->ucp),
          "failed to send am reply", log_ucp, err_rel_req);
      // TODO: Should we notify the source somehow?

      log_ucp_am.debug() << "am reply sent to " << sender << ". Req " << req;
    }

    {
      AutoLock<SpinLock> al(internal->rcba_mp_spinlock);
      MPool::put(cb_args);
    }

    return;

err_rel_req:
    internal->request_release(req);
err:
    AutoLock<SpinLock> al(internal->rcba_mp_spinlock);
    MPool::put(cb_args);
  }

  bool UCPInternal::am_msg_recv_data_ready(UCPInternal *internal,
      UCPWorker *worker, const UCPMsgHdr *ucp_msg_hdr, size_t header_size,
      void *payload, size_t payload_size, int payload_mode)
  {
    const void *realm_hdr = reinterpret_cast<const void*>(ucp_msg_hdr->realm_hdr);
    size_t realm_hdr_size = header_size - sizeof(*ucp_msg_hdr);

    if (internal->config.crc_check) {
      verify_packet_crc(ucp_msg_hdr, realm_hdr_size, payload_size);
    }

    RealmCallbackArgs *cb_args;
    {
      AutoLock<SpinLock> al(rcba_mp_spinlock);
      cb_args = reinterpret_cast<RealmCallbackArgs*>(internal->rcba_mp->get());
      *cb_args = {};
    }
    assert(cb_args);

    log_ucp_am.debug() << "am_msg_recv_data_ready invoked. Sender " << ucp_msg_hdr->src;

    (void) internal->total_msg_received.fetch_add(1);

    cb_args->remote_comp      = ucp_msg_hdr->remote_comp;
#ifdef REALM_USE_CUDA
    cb_args->remote_dev_index = ucp_msg_hdr->src_dev_index;
#else
    cb_args->remote_dev_index = -1;
#endif
    cb_args->payload          = payload;
    cb_args->payload_mode     = payload_mode;
    cb_args->internal         = internal;
    cb_args->rx_worker        = worker;

    IncomingMessageManager::CallbackData cb_data1 =
      reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(cb_args));

    bool completed = runtime->message_manager->add_incoming_message(
        ucp_msg_hdr->src, ucp_msg_hdr->msgid,
        realm_hdr, realm_hdr_size, PAYLOAD_COPY, // header is always copied
        payload, payload_size,
        payload_mode == PAYLOAD_FREE ? PAYLOAD_KEEP : payload_mode,
        &am_realm_comp_handler, cb_data1, 0,
        (ThreadLocal::ucp_work_until != nullptr) ? *ThreadLocal::ucp_work_until
                                                 : TimeLimit::relative(0));

    if (completed) {
      am_realm_comp_handler(ucp_msg_hdr->src, cb_data1, 0);
      return true;
    }

    return false; // Realm has not completed message processing yet
  }

  void UCPInternal::am_rndv_recv_data_handler(void *request,
      ucs_status_t status, size_t length, void *user_data)
  {
    Request *req           = reinterpret_cast<Request*>(request);
    UCPMsgHdr *ucp_msg_hdr = reinterpret_cast<UCPMsgHdr*>(req->am_rndv_recv.header);
    UCPWorker *worker      = req->worker;
    UCPInternal *internal  = req->internal;

    assert(length == req->am_rndv_recv.payload_size);

    log_ucp_am.debug() << "am_rndv_recv_data_handler invoked for request " << req;

    if (status == UCS_OK) {
      (void) internal->am_msg_recv_data_ready(internal, worker,
          ucp_msg_hdr, req->am_rndv_recv.header_size,
          req->am_rndv_recv.payload, req->am_rndv_recv.payload_size,
          req->am_rndv_recv.payload_mode);
    } else {
      log_ucp.error() << "failed to receive am rndv data";
      if (req->am_rndv_recv.payload_mode == PAYLOAD_FREE) {
        internal->pbuf_release(worker, req->am_rndv_recv.payload);
      }
    }

    internal->hbuf_release(worker, req->am_rndv_recv.header);

    internal->request_release(req);
  }

  ucs_status_t UCPInternal::am_msg_recv_handler(
      void *arg, const void *header, size_t header_size,
      void *payload, size_t payload_size,
      const ucp_am_recv_param_t *param)
  {
    const UCPMsgHdr *ucp_msg_hdr = reinterpret_cast<const UCPMsgHdr*>(header);
    AmHandlersArgs *am_args      = reinterpret_cast<AmHandlersArgs*>(arg);
    UCPWorker *worker            = am_args->worker;
    UCPInternal *internal        = am_args->internal;

    assert((header != nullptr) && (header_size >= sizeof(UCPMsgHdr)));

    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) {
      // Should receive the payload first
      size_t recv_info_length;
      ucs_status_t status;
      ucs_status_ptr_t status_ptr;
      void *data_desc = payload;
      Request *req;

      log_ucp_am.debug() << "am received with UCP_AM_RECV_ATTR_FLAG_RNDV";

      req = internal->request_get(worker);
      if (req == nullptr) {
        log_ucp.error() << "failed to get request";
        return UCS_ERR_NO_MEMORY;
      }

      ucp_request_param_t param;
      param.op_attr_mask     = UCP_OP_ATTR_FIELD_CALLBACK  |
                               UCP_OP_ATTR_FIELD_REQUEST   |
                               UCP_OP_ATTR_FIELD_RECV_INFO |
                               UCP_OP_ATTR_FIELD_MEMORY_TYPE;
      param.cb.recv_am       = &am_rndv_recv_data_handler;
      param.request          = req;
      param.recv_info.length = &recv_info_length;
#ifdef REALM_USE_CUDA
      param.memory_type      = worker->get_context()->gpu ?
          UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
#else
      param.memory_type      = UCS_MEMORY_TYPE_HOST;
#endif

      if (ucp_msg_hdr->rdma_payload_addr != nullptr) {
        // am with remote address
        req->am_rndv_recv.payload      = ucp_msg_hdr->rdma_payload_addr;
        req->am_rndv_recv.payload_mode = PAYLOAD_KEEPREG;
      } else {
        // Allocate buffer to receive the payload
        req->am_rndv_recv.payload      = internal->pbuf_get(worker, payload_size);
        req->am_rndv_recv.payload_mode = PAYLOAD_FREE;
      }

      req->am_rndv_recv.header         = internal->hbuf_get(worker, header_size);
      req->am_rndv_recv.header_size    = header_size;
      req->am_rndv_recv.payload_size   = payload_size;

      memcpy(req->am_rndv_recv.header, header, header_size);

      status_ptr = ucp_am_recv_data_nbx(worker->get_ucp_worker(),
          data_desc, req->am_rndv_recv.payload,
          req->am_rndv_recv.payload_size, &param);

      if (UCS_PTR_IS_ERR(status_ptr)) {
        log_ucp.error() << "ucp_am_recv_data_nbx failed";
        internal->hbuf_release(worker, req->am_rndv_recv.header);
        if (req->am_rndv_recv.payload_mode == PAYLOAD_FREE) {
          internal->pbuf_release(worker, req->am_rndv_recv.payload);
        }
        internal->request_release(req);
        status = UCS_PTR_STATUS(status_ptr);
      } else if (status_ptr != NULL) {
        // Non-immediate completion
        status = UCS_INPROGRESS;
      } else {
        // immediate completion
        am_rndv_recv_data_handler(req, UCS_OK, recv_info_length, NULL);
        status = UCS_OK;
      }

      return status;
    }

    int payload_mode;
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
      // Persistent payload buffer
      payload_mode = PAYLOAD_KEEP; // UCP module keeps ownership
      log_ucp_am.debug() << "am received with UCP_AM_RECV_ATTR_FLAG_DATA";
    } else {
      // Non-persistent payload buffer
      payload_mode = PAYLOAD_COPY;
      log_ucp_am.debug() << "am received with no flags (non-persistent payload buffer)";
    }

    (void) internal->am_msg_recv_data_ready(internal, worker,
        ucp_msg_hdr, header_size, payload, payload_size, payload_mode);

    /* IMPORTANT: Do NOT return UCS_OK even when we
                  have FLAG_DATA+completed, because
                  the persistent data buffer is always
                  relesed in am_realm_comp_handler
    */
    return (payload_mode == PAYLOAD_KEEP) ? UCS_INPROGRESS : UCS_OK;
  }

  ucs_status_t UCPInternal::am_rdma_msg_recv_handler(
      void *arg, const void *header, size_t header_size,
      void *payload, size_t payload_size,
      const ucp_am_recv_param_t *param)
  {
    const UCPMsgHdr *ucp_msg_hdr = reinterpret_cast<const UCPMsgHdr*>(header);
    AmHandlersArgs *am_args      = reinterpret_cast<AmHandlersArgs*>(arg);
    UCPWorker *worker            = am_args->worker;
    UCPInternal *internal        = am_args->internal;

    assert((header != nullptr) && (header_size >= sizeof(UCPMsgHdr)));
    assert(payload_size == 0);

    log_ucp_am.debug() << "am rdma received";

    payload      = ucp_msg_hdr->rdma_payload_addr;
    payload_size = ucp_msg_hdr->rdma_payload_size;

    internal->am_msg_recv_data_ready(internal, worker,
        ucp_msg_hdr,header_size, payload, payload_size, PAYLOAD_KEEPREG);

    return UCS_OK;
  }

  const SegmentInfo* UCPInternal::find_segment(const void *srcptr) const
  {
    uintptr_t ptr = reinterpret_cast<uintptr_t>(srcptr);
    // binary search
    unsigned lo = 0;
    unsigned hi = segments_by_addr.size();
    while(lo < hi) {
      unsigned mid = (lo + hi) >> 1;
      if(ptr < segments_by_addr[mid].base) {
        hi = mid;
      } else if(ptr >= segments_by_addr[mid].limit) {
        lo = mid + 1;
      } else {
#if defined(REALM_USE_CUDA)
        Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU*>(segments_by_addr[mid].memextra);
        assert(gpu);
        log_ucp_seg.debug() << "found segment info for src ptr " << srcptr
                            << " device index " << gpu->info->index;
#endif
        return &segments_by_addr[mid];
      }
    }

    return nullptr;
  }

  struct UCPInternal::SegmentInfoSorter {
    bool operator()(const SegmentInfo& lhs, const SegmentInfo& rhs) const
    {
      return lhs.base < rhs.base;
    }
  };

  bool UCPInternal::add_rdma_info(NetworkSegment *segment,
      const UCPContext *context, ucp_mem_h mem_h)
  {
    UCPRDMAInfo *rdma_info;
    void *rkey_buf;
    size_t rkey_buf_size;
    ucs_status_t status;

    assert(segment->base != nullptr);

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    status = ucp_rkey_pack(context->get_ucp_context(),
        mem_h, &rkey_buf, &rkey_buf_size);
    CHKERR_JUMP(status != UCS_OK, "ucp_rkey_pack failed", log_ucp, err);

    rdma_info = reinterpret_cast<UCPRDMAInfo *>(
        malloc(sizeof(*rdma_info) + rkey_buf_size));
    CHKERR_JUMP(rdma_info == nullptr,
        "failed to malloc rdma info", log_ucp, err_rkey_rel);

    rdma_info->reg_base  = reinterpret_cast<uint64_t>(segment->base);
    rdma_info->dev_index = -1;

#if defined(REALM_USE_CUDA)
    if (segment->memtype == NetworkSegmentInfo::CudaDeviceMem) {
      Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU *>(segment->memextra);
      rdma_info->dev_index = gpu->info->index;
    }
#endif

    memcpy(rdma_info->rkey, rkey_buf, rkey_buf_size);
    segment->add_rdma_info(module, rdma_info,
        sizeof(*rdma_info) + rkey_buf_size);
    free(rdma_info);
    ucp_rkey_buffer_release(rkey_buf);

    return true;

err_rkey_rel:
    ucp_rkey_buffer_release(rkey_buf);
err:
    return false;
  }

  void UCPInternal::attach(std::vector<NetworkSegment *>& segments)
  {
    size_t total_alloc_size = 0;
    const UCPContext *context;
    ucp_mem_map_params_t mem_map_params;
    ucp_mem_attr_t mem_attr;
    ucp_mem_h alloc_mem_h, mem_h;
    ucs_status_t status;
    uintptr_t alloc_base, offset;
    ByteArray alloc_rdma_info;

#if defined(REALM_USE_CUDA)
    std::unordered_set<Cuda::GPU*> gpus;
    for (NetworkSegment *segment : segments) {
      if(segment->base == nullptr) continue;
      if (segment->memtype == NetworkSegmentInfo::CudaDeviceMem) {
        uintptr_t base_as_uint = reinterpret_cast<uintptr_t>(segment->base);
        segments_by_addr.push_back({
            base_as_uint,
            base_as_uint + segment->bytes,
            segment->memtype,
            segment->memextra});
        Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU *>(segment->memextra);
        gpus.insert(gpu);
        log_ucp_seg.debug()
          << "tracking pre-allocated gpu segment " << segment
          << " (base " << segment->base << " length " << segment->bytes
          << " with device index " << gpu->info->index;
      } else if (segment->memtype == NetworkSegmentInfo::CudaDeviceMem) {
        log_ucp.fatal() << "Cuda Managed Memory not supported";
        return;
      }
    }

    // sort the by_addr list
    std::sort(segments_by_addr.begin(), segments_by_addr.end(),
          SegmentInfoSorter());
    // sanity-check that there's no overlap
    for(size_t i = 1; i < segments_by_addr.size(); i++)
      assert(segments_by_addr[i-1].limit <= segments_by_addr[i].base);

    bool ok = init_ucp_contexts(gpus);
#else
    bool ok = init_ucp_contexts();
#endif
    if (!ok) {
      log_ucp.fatal() << "failed to initialized ucp contexts";
      return;
    }

    // Try to register allocation requests first
    // The bind_hostmem option does not apply to allocation requests
    for (NetworkSegment *segment : segments) {
      if (segment->bytes == 0) continue;
      if (segment->base != nullptr) continue;
      // Must be host memory
      // TODO: Have two separate allocations: one for host and one for device
      if (segment->memtype != NetworkSegmentInfo::HostMem) {
        log_ucp.info() << "non-host-memory allocation request not supported in attach";
        continue;
      }
      total_alloc_size += segment->bytes;
    }

    context = get_context_host();
    assert(context);

    mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                                UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_map_params.flags      = UCP_MEM_MAP_ALLOCATE;
    mem_map_params.length     = total_alloc_size;

    CHKERR_JUMP(!context->mem_map(&mem_map_params, &alloc_mem_h),
        "ucp_mem_map failed for allocation segments", log_ucp, err);
    attach_mem_hs[context].push_back(alloc_mem_h);

    // Now try to register non-allocation requests
    mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                                UCP_MEM_MAP_PARAM_FIELD_ADDRESS;

    for (NetworkSegment *segment : segments) {
      if(segment->base == nullptr) continue;
#if defined(REALM_USE_CUDA)
      if (segment->memtype == NetworkSegmentInfo::CudaDeviceMem) {
        if (!config.bind_cudamem) continue;
        Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU *>(segment->memextra);
        context = get_context_device(gpu->info->index);
        log_ucp_seg.info()
          << "attaching pre-allocated segment " << segment
          << " (base " << segment->base << " length " << segment->bytes
          << ") in gpu context " << context
          << " device index " << gpu->info->index;
      } else {
        context = get_context_host();
        log_ucp_seg.info()
          << "attaching pre-allocated segment " << segment
          << " (base " << segment->base << " length " << segment->bytes
          << ") in host context " << context;
      }
      assert(context);
#endif

      if (!config.bind_hostmem &&
          segment->memtype == NetworkSegmentInfo::HostMem) {
        continue;
      }

      mem_map_params.address = segment->base;
      mem_map_params.length  = segment->bytes;
      if (!context->mem_map(&mem_map_params, &mem_h)) {
        log_ucp.info() << "ucp_mem_map failed for pre-allocated segment "
                       << segment;
        continue;
      }
      attach_mem_hs[context].push_back(mem_h);

      if (!add_rdma_info(segment, context, mem_h)) {
        log_ucp.info() << "failed to add rdma info for pre-allocated segment "
                       << segment;
      }
    }

    // Set address and add RDMA info for allocated segments
    context = get_context_host();
    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(alloc_mem_h, &mem_attr);
    assert(status == UCS_OK);
    alloc_base = reinterpret_cast<uintptr_t>(mem_attr.address);
    offset     = 0;
    for (NetworkSegment *segment : segments) {
      if (segment->bytes == 0) continue;
      if (segment->base != nullptr) continue;
      // Must be host memory (for now at least)
      if (segment->memtype != NetworkSegmentInfo::HostMem) continue;

      segment->base = reinterpret_cast<void *>(alloc_base + offset);
      if (offset == 0) {
        if (!add_rdma_info(segment, context, alloc_mem_h)) {
          log_ucp.info() << "failed to add rdma info for allocation segment "
                         << segment;
        }
        alloc_rdma_info = *(segment->get_rdma_info(module));
      } else {
        // already have the rkey in alloc_rdma_info,
        // but should update the base pointer
        UCPRDMAInfo *rdma_info = reinterpret_cast<UCPRDMAInfo *>(
          alloc_rdma_info.base());
        rdma_info->reg_base = reinterpret_cast<uint64_t>(segment->base);
        segment->add_rdma_info(module,
            alloc_rdma_info.base(), alloc_rdma_info.size());
      }
      offset += segment->bytes;
    }

    log_ucp.info() << "attached segments";

err:
    return;
  }

  void UCPInternal::detach(std::vector<NetworkSegment *>& segments)
  {
    // This relies on the fact that Realm calls detach only once
    // during shutdown. Thus, we deregister all ucp memory handles
    // that were created during attach.

    log_ucp.info() << "detaching segments";

    for (auto &poller : pollers) {
      poller.end_polling();
#ifdef DEBUG_REALM
      poller.shutdown_work_item();
#endif
    }
    log_ucp.info() << "ended ucp pollers";

    for (UCPContext &context : ucp_contexts) {
      auto &mem_hs = attach_mem_hs[&context];
      while (!mem_hs.empty()) {
        if (!context.mem_unmap(mem_hs.back())) {
          log_ucp.info() << "failed to unmap segment mem_h " << mem_hs.back();
        }
        mem_hs.pop_back();
      }
    }
    log_ucp.info() << "unmapped attached segments";
  }

  void UCPInternal::barrier()
  {
    int rc;
    rc = boot_handle.barrier(&boot_handle);
    if (rc != 0) {
      log_ucp.error() << "UCP barrier failed";
    }
  }

  void UCPInternal::broadcast(NodeID root,
      const void *val_in, void *val_out, size_t bytes)
  {
    int rc;

    if (Network::my_node_id == root) {
      std::memcpy(val_out, val_in, bytes);
    }

    rc = boot_handle.bcast(val_out, bytes, root, &boot_handle);
    if (rc != 0) {
      log_ucp.error() << "UCP broadcast failed";
    }
  }

  void UCPInternal::gather(NodeID root,
      const void *val_in, void *vals_out, size_t bytes)
  {
    int rc;

    rc = boot_handle.gather(val_in, vals_out, bytes, root, &boot_handle);
    if (rc != 0) {
      log_ucp.error() << "UCP gather failed";
    }
  }

  size_t UCPInternal::sample_messages_received_count()
  {
    return total_msg_received.load();
  }

  bool UCPInternal::check_for_quiescence(size_t sampled_receive_count)
  {
    /* The check for outstanding_reqs must be collective. Otherwise,
     * the rank(s) with outstanding requests will return false too quickly,
     * which will consume quiescence-check iterations too quickly without
     * giving communications a chance to be progressed.
     */

    static constexpr int num_counters   = 6;
    uint64_t total_counts[num_counters] = { 0, 0, 0, 0, 0, 0 };
    uint64_t local_counts[num_counters] = {
      total_msg_sent.load(),
      total_msg_received.load(),
      sampled_receive_count,
      total_rcomp_sent.load(),
      total_rcomp_received.load(),
      outstanding_reqs.load(),
    };

    log_ucp.debug() << "local quiescence counters:"
                    << " total_msg_sent "        << local_counts[0]
                    << " total_msg_received "    << local_counts[1]
                    << " sampled_receive_count " << local_counts[2]
                    << " total_rcomp_sent "      << local_counts[3]
                    << " total_rcomp_received "  << local_counts[4]
                    << " outstanding_reqs "      << local_counts[5];


    int rc = boot_handle.allreduce_ull(local_counts, total_counts,
        num_counters, REDUCTION_SUM, &boot_handle);
    if (rc != 0) {
      log_ucp.error() << "allreduce failed in check_for_quiescence";
      return false;
    }

    log_ucp.debug() << "reduced quiescence counters:"
                    << " total_msg_sent "        << total_counts[0]
                    << " total_msg_received "    << total_counts[1]
                    << " sampled_receive_count " << total_counts[2]
                    << " total_rcomp_sent "      << total_counts[3]
                    << " total_rcomp_received "  << total_counts[4]
                    << " outstanding_reqs "      << total_counts[5];

    bool ret = ((total_counts[0] == total_counts[1]) && // msg_sent == msg_recv
                (total_counts[1] == total_counts[2]) && // msg_recv == sampled
                (total_counts[3] == total_counts[4]) && // rcomp_sent == rcomp_recv
                (total_counts[5] == 0));                // outstanding_reqs == 0

    if (!ret) {
      // wait until the poller threads get a chance to
      // progress communications before returning. Otherwise,
      // we might consume quiescence-check iterations too quickly.
      for (auto &poller : pollers) {
        poller.wait_polling();
      }
    }

    return ret;
  }

  bool UCPInternal::is_congested()
  {
    return (outstanding_reqs.load() > config.outstanding_reqs_limit);
  }

  size_t UCPInternal::recommended_max_payload(
      const RemoteAddress *dest_payload_addr,
      bool with_congestion,
      size_t header_size)
  {
    size_t ret;

    if (with_congestion && is_congested()) {
      return 0;
    }
    if (dest_payload_addr) {
      ret = GET_ZCOPY_MAX;
    } else {
      ret = std::min(ib_seg_size - header_size, config.pbuf_max_size);
    }

    return ret;
  }

  size_t UCPInternal::recommended_max_payload(NodeID target,
      const RemoteAddress *dest_payload_addr,
      bool with_congestion,
      size_t header_size)
  {
    // don't have a target-specific version yet
    return recommended_max_payload(dest_payload_addr,
        with_congestion, header_size);
  }

  size_t UCPInternal::recommended_max_payload(
      const RemoteAddress *dest_payload_addr,
      const void *data, size_t bytes_per_line,
      size_t lines, size_t line_stride,
      bool with_congestion,
      size_t header_size)
  {
    return recommended_max_payload(dest_payload_addr,
        with_congestion, header_size);
  }

  size_t UCPInternal::recommended_max_payload(NodeID target,
      const RemoteAddress *dest_payload_addr,
      const void *data, size_t bytes_per_line,
      size_t lines, size_t line_stride,
      bool with_congestion,
      size_t header_size)
  {
    // don't have a target-specific version yet
    return recommended_max_payload(dest_payload_addr,
        data, bytes_per_line, lines, line_stride,
        with_congestion, header_size);
  }

  const UCPContext *UCPInternal::get_context_host() const
  {
    return &ucp_contexts.front();
  }

#if defined(REALM_USE_CUDA)
  const UCPContext *UCPInternal::get_context_device(int dev_index) const
  {
    auto iter = dev_ctx_map.find(dev_index);
    assert(iter != dev_ctx_map.end());
    return iter->second;
  }
#endif

  const UCPContext *UCPInternal::get_context(const SegmentInfo *seg_info) const
  {
#if defined(REALM_USE_CUDA)
    // see if we should use the device ucp context
    if (seg_info) {
      Cuda::GPU *gpu = reinterpret_cast<Cuda::GPU*>(seg_info->memextra);
      assert(gpu);
      return get_context_device(gpu->info->index);
    }
#endif
    return get_context_host();
  }

  const std::vector<UCPWorker*>
    &UCPInternal::get_tx_workers(const UCPContext *context) const
  {
    WorkersMap::const_iterator itr = workers.find(context);
    assert(itr != workers.end());
    return itr->second.tx_workers;
  }

  UCPWorker *UCPInternal::get_tx_worker(const UCPContext *context, uint8_t priority) const
  {
    return get_tx_workers(context)[priority];
  }

  const std::vector<UCPWorker*>
    &UCPInternal::get_rx_workers(const UCPContext *context) const
  {
    WorkersMap::const_iterator itr = workers.find(context);
    assert(itr != workers.end());
    return itr->second.rx_workers;
  }

  UCPWorker *UCPInternal::get_rx_worker(const UCPContext *context, uint8_t priority) const
  {
    return get_rx_workers(context)[priority];
  }

  size_t UCPInternal::num_eps(const UCPContext &context) const
  {
    size_t n = 0;
    for (const UCPWorker *worker : get_tx_workers(&context)) {
      n += worker->num_eps();
    }
    return n;
  }

  Request *UCPInternal::request_get(UCPWorker *worker)
  {
    Request *req = reinterpret_cast<Request*>(worker->request_get());
    if (!req) return nullptr;
    memset(req, 0, sizeof(*req));

    req->internal = this;
    req->worker   = worker;
    (void) outstanding_reqs.fetch_add(1);
    log_ucp_ar.debug() << "acquired request " << req;

    return req;
  }

  void UCPInternal::request_release(Request *req)
  {
    req->worker->request_release(req);
    (void) outstanding_reqs.fetch_sub(1);
    log_ucp_ar.debug() << "released request " << req;
  }

  void *UCPInternal::hbuf_get(UCPWorker *worker, size_t size)
  {
    void *buf;

    if (config.hbuf_malloc) {
      buf = malloc(size);
    } else {
      buf = worker->mmp_get(size);
    }

    log_ucp_ar.debug() << "acquired header buffer " << buf << " size " << size;
    return buf;
  }

  void UCPInternal::hbuf_release(UCPWorker *worker, void *buf)
  {
    if (config.hbuf_malloc) {
      free(buf);
    } else {
      worker->mmp_release(buf);
    }
    log_ucp_ar.debug() << "released header buffer " << buf;
  }

  void *UCPInternal::pbuf_get(UCPWorker *worker, size_t size)
  {
    char *buf;
    assert(size <= ib_seg_size);
#ifdef REALM_USE_CUDA
    // we should never ask for a payload buffer from a GPU ucp context
    assert(!worker->get_context()->gpu);
#endif
    // use one extra byte at the beginning of the buffer
    // to determine whether the buffer is from context mpool or malloc.
    // But, for alignment's sake, allocate AM_ALIGNMENT extra bytes.
    size += AM_ALIGNMENT;

    if (size < config.pbuf_mp_thresh) {
      if (config.pbuf_malloc) {
        buf = reinterpret_cast<char*>(malloc(size));
      } else {
        buf = reinterpret_cast<char*>(worker->mmp_get(size));
      }
      CHKERR_JUMP(!buf, "", log_ucp, err);
      *buf = 0;
    } else {
      // TODO: put an upper bound on the mpool size
      //       context->pbuf_mp->has(size, false /* with_expand */)
      buf = reinterpret_cast<char*>(worker->pbuf_get(size));
      CHKERR_JUMP(!buf, "", log_ucp, err);
      *buf = 1;
    }

    log_ucp_ar.debug() << "acquired payload buffer " << buf + AM_ALIGNMENT
                       << " size " << size - AM_ALIGNMENT;
    return buf + AM_ALIGNMENT;

err:
    return nullptr;
  }

  void UCPInternal::pbuf_release(UCPWorker *worker, void *buf)
  {
    char *alloc = reinterpret_cast<char*>(buf) - AM_ALIGNMENT;
    if (*alloc == 0) {
      if (config.pbuf_malloc) {
        free(alloc);
      } else {
        worker->mmp_release(alloc);
      }
    } else {
      worker->pbuf_release(alloc);
    }

    log_ucp_ar.debug() << "released payload buffer " << buf;
  }

  void UCPInternal::notify_msg_sent(uint64_t count)
  {
    (void) total_msg_sent.fetch_add(count);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPMessageImpl
  //

  UCPMessageImpl::UCPMessageImpl(
      UCPInternal *_internal,
      NodeID _target,
      unsigned short _msgid,
      size_t _header_size,
      size_t _max_payload_size,
      const void *_src_payload_addr,
      size_t _src_payload_lines,
      size_t _src_payload_line_stride,
      size_t _storage_size)
    : internal(_internal)
    , target(_target)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
  {
    constructor_common(_msgid, _header_size, _max_payload_size, _storage_size);
    ucp_msg_hdr.rdma_payload_addr = nullptr;
  }

  UCPMessageImpl::UCPMessageImpl(
      UCPInternal *_internal,
      NodeID _target,
      unsigned short _msgid,
      size_t _header_size,
      size_t _max_payload_size,
      const void *_src_payload_addr,
      size_t _src_payload_lines,
      size_t _src_payload_line_stride,
      const RemoteAddress& _dest_payload_addr,
      size_t _storage_size)
    : internal(_internal)
    , target(_target)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
  {
    constructor_common(_msgid, _header_size, _max_payload_size, _storage_size);

    dest_payload_rdma_info = reinterpret_cast<UCPRDMAInfo*>(
        malloc(sizeof(_dest_payload_addr)));
    assert(dest_payload_rdma_info != nullptr);

    memcpy(dest_payload_rdma_info, _dest_payload_addr.raw_bytes,
        sizeof(_dest_payload_addr));

    ucp_msg_hdr.rdma_payload_addr = reinterpret_cast<void*>(
        dest_payload_rdma_info->reg_base);
  }

  UCPMessageImpl::UCPMessageImpl(
      UCPInternal *_internal,
      const NodeSet &_targets,
      unsigned short _msgid,
      size_t _header_size,
      size_t _max_payload_size,
      const void *_src_payload_addr,
      size_t _src_payload_lines,
      size_t _src_payload_line_stride,
      size_t _storage_size)
    : internal(_internal)
    , targets(_targets)
    , src_payload_addr(_src_payload_addr)
    , src_payload_lines(_src_payload_lines)
    , src_payload_line_stride(_src_payload_line_stride)
  {
    constructor_common(_msgid, _header_size, _max_payload_size, _storage_size);
    ucp_msg_hdr.rdma_payload_addr = nullptr;

    // treat single-target multicast messages as unicast
    if (targets.size() > 1) {
      is_multicast = true;
    } else {
      target = *targets.begin();
    }
  }

  UCPMessageImpl::~UCPMessageImpl()
  {}

  bool UCPMessageImpl::set_inline_payload_base()
  {
    // use the unsed space at the end of this object
    // for the payload_base if possible.
    constexpr size_t alignment = 8;
    uintptr_t limit        = reinterpret_cast<uintptr_t>(this) + sizeof(*this);
    uintptr_t addr         = reinterpret_cast<uintptr_t>(header_base) + header_size;
    uintptr_t addr_aligned = (addr + alignment - 1) & ~(alignment - 1);

    if (addr_aligned + payload_size > limit) {
      return false;
    }

    payload_base      = reinterpret_cast<void*>(addr_aligned);
    payload_base_type = PAYLOAD_BASE_INLINE;
    return true;
  }

  void UCPMessageImpl::constructor_common(
      unsigned short _msgid,
      size_t _header_size,
      size_t _max_payload_size,
      size_t _storage_size)
  {
    const SegmentInfo *segment_info = internal->find_segment(src_payload_addr);
    const UCPContext *context       = internal->get_context(segment_info);
    uint8_t priority                = (header_size + _max_payload_size <=
                                       internal->config.priority_size_max) ?
                                      internal->config.num_priorities - 1 : 0;

    worker  = internal->get_tx_worker(context, priority);
    memtype = segment_info ? realm2ucs_memtype(segment_info->memtype)
                           : UCS_MEMORY_TYPE_HOST;


    size_t max_header_size    = _storage_size - sizeof(*this);
    max_header_size           = std::min(max_header_size, worker->get_max_am_header());
    assert(_header_size      <= max_header_size);
    header_size               = _header_size;
    header_base               = reinterpret_cast<void*>(&ucp_msg_hdr.realm_hdr[0]);
    ucp_msg_hdr.src           = Network::my_node_id;
    ucp_msg_hdr.msgid         = _msgid;
#ifdef REALM_USE_CUDA
    ucp_msg_hdr.src_dev_index = context->gpu ? context->gpu->info->index : -1;
#endif
    payload_size              = _max_payload_size;

    payload_base_type = PAYLOAD_BASE_LAST;
    if (payload_size > 0) {
      if ((src_payload_addr != nullptr) && (src_payload_lines <= 1)) {
        // contiguous source data can be used directly
        payload_base = const_cast<void *>(src_payload_addr);
        payload_base_type = PAYLOAD_BASE_EXTERNAL;
      } else if (!set_inline_payload_base()) {
        payload_base = internal->pbuf_get(worker, payload_size);
        assert(payload_base != nullptr);
        payload_base_type = PAYLOAD_BASE_INTERNAL;
      }
    } else {
      payload_base = nullptr;  // no payload
    }
  }

  void *UCPMessageImpl::add_local_completion(size_t size)
  {
    if(local_comp == nullptr) {
      local_comp = new CompList;
    }
    size_t ofs = local_comp->bytes;
    local_comp->bytes += size;
    assert(local_comp->bytes <= CompList::TOTAL_CAPACITY);
    return (local_comp->storage + ofs);
  }

  void *UCPMessageImpl::add_remote_completion(size_t size)
  {
    if(remote_comp == nullptr) {
      size_t remote_pending = is_multicast ? targets.size() : 1;
      remote_comp = new RemoteComp(remote_pending);
    }
    size_t ofs = remote_comp->comp_list->bytes;
    remote_comp->comp_list->bytes += size;
    assert(remote_comp->comp_list->bytes <= CompList::TOTAL_CAPACITY);
    return (remote_comp->comp_list->storage + ofs);
  }

  void UCPMessageImpl::am_local_failure_handler(Request *req,
      UCPInternal *internal)
  {
    UCPMsgHdr *ucp_msg_hdr  = reinterpret_cast<UCPMsgHdr*>(req->ucp.am.header);
    RemoteComp *remote_comp = ucp_msg_hdr->remote_comp;
    size_t local_pending = 1;
    size_t remote_pending;

    log_ucp_am.debug()
      << "am_local_failure_handler invoked for request " << req;

    if (remote_comp != nullptr) {
      remote_comp->flags |= RemoteComp::REMOTE_COMP_FLAG_FAILURE;
      remote_pending      = remote_comp->remote_pending.fetch_sub_acqrel(1);
      assert(remote_pending != 0);
      if (remote_pending == 1) {
        delete remote_comp;
      }
    }

    if (req->am_send.mc_desc != nullptr) {
      req->am_send.mc_desc->flags |= MCDesc::REQUEST_AM_FLAG_FAILURE;
      local_pending = req->am_send.mc_desc->local_pending.fetch_sub_acqrel(1);
    }

    assert(local_pending != 0);

    if (local_pending == 1) {
      UCPMessageImpl::cleanup_request(req, internal);
    }

    internal->request_release(req);
  }

  void UCPMessageImpl::am_local_comp_handler(void *request,
      ucs_status_t status, void *user_data)
  {
    Request *req          = reinterpret_cast<Request*>(user_data);
    UCPInternal *internal = req->internal;
    CompList *local_comp  = req->am_send.local_comp;
    size_t local_pending  = 1;
    uint8_t comp_flags    = 0;

    log_ucp_am.debug() << "am_local_comp_handler invoked for request " << req;

    CHKERR_JUMP(status != UCS_OK, "failed to complete am locally", log_ucp, err);

    internal->notify_msg_sent(1);

    if (req->am_send.mc_desc != nullptr) {
      local_pending = req->am_send.mc_desc->local_pending.fetch_sub(1);
      comp_flags    = req->am_send.mc_desc->flags;
    }

    assert(local_pending != 0);

    if (local_pending == 1) {
      if (!(comp_flags & MCDesc::REQUEST_AM_FLAG_FAILURE)) {
        // all senders completed without failure
        if (local_comp != nullptr) {
          CompletionCallbackBase::invoke_all(local_comp->storage, local_comp->bytes);
          CompletionCallbackBase::destroy_all(local_comp->storage, local_comp->bytes);
        }
      } else {
        // TODO: should CompletionCallbackBase::destroy_all() be called here?
        // TODO: should invoke some higher-level error handler
      }
      UCPMessageImpl::cleanup_request(req, internal);
    }

    internal->request_release(req);

    return;

err:
    UCPMessageImpl::am_local_failure_handler(req, internal);
  }

  bool UCPMessageImpl::send_fast_path(ucp_ep_h ep, size_t act_payload_size)
  {
    bool ok = worker->am_send_fast_path(ep, AM_ID,
        &ucp_msg_hdr, sizeof(ucp_msg_hdr) + header_size,
        payload_base, act_payload_size, memtype);

    if (ok) {
      internal->notify_msg_sent(1);
      if (local_comp != nullptr) {
        CompletionCallbackBase::invoke_all(local_comp->storage, local_comp->bytes);
        CompletionCallbackBase::destroy_all(local_comp->storage, local_comp->bytes);
        delete local_comp;
      }
      if (payload_base_type == PAYLOAD_BASE_INTERNAL) {
        internal->pbuf_release(worker, payload_base);
      }
      log_ucp_am.info() << "successful send with enforced immediate completion";
      return true;
    }

    return false;
  }

  bool UCPMessageImpl::send_slow_path(ucp_ep_h ep,
      size_t act_payload_size, uint32_t flags)
  {
    Request *req = make_request(act_payload_size);
    CHKERR_JUMP(req == nullptr, "failed to make am request", log_ucp, err);

    req->ucp.ep     = ep;
    req->ucp.flags |= flags;

    CHKERR_JUMP(!UCPMessageImpl::send_request(req, AM_ID),
        "failed to send am request in slow path", log_ucp, err_rel_req);

    return true;

err_rel_req:
    UCPMessageImpl::am_local_failure_handler(req, internal);
err:
    return false;
  }

  bool UCPMessageImpl::send_request(Request *req, unsigned am_id)
  {
    if (req->am_send.payload_base_type == PAYLOAD_BASE_INTERNAL) {
      // should not have enforced rndv with an internal payload buffer
      assert(!(req->ucp.flags & UCP_AM_SEND_FLAG_RNDV));
      req->ucp.flags |= UCP_AM_SEND_FLAG_EAGER;
    }

    req->ucp.op_type = UCPWorker::OpType::AM_SEND;
    req->ucp.args    = req;
    req->ucp.cb      = &UCPMessageImpl::am_local_comp_handler;
    req->ucp.am.id   = am_id;

    return req->worker->submit_req(&req->ucp);
  }

  Request *UCPMessageImpl::make_request(size_t act_payload_size)
  {
    size_t ucp_msg_hdr_size = sizeof(ucp_msg_hdr) + header_size;
    Request *req            = internal->request_get(worker);
    CHKERR_JUMP(req == nullptr, "failed to get request", log_ucp, err);

    req->ucp.am.header      = &ucp_msg_hdr;
    req->ucp.am.header_size = ucp_msg_hdr_size;

    // Copy the payload if we used the inline header buffer for it
    if (payload_base_type == PAYLOAD_BASE_INLINE) {
      req->ucp.payload = internal->pbuf_get(worker, act_payload_size);
      CHKERR_JUMP(req->ucp.payload == nullptr,
          "failed to get payload buffer", log_ucp, err_rel_req);

      memcpy(req->ucp.payload, payload_base, act_payload_size);
      payload_base      = req->ucp.payload;
      payload_base_type = PAYLOAD_BASE_INTERNAL;
    }

    req->am_send.payload_base_type = payload_base_type;
    req->am_send.local_comp        = local_comp;
    req->ucp.payload               = payload_base;
    req->ucp.payload_size          = act_payload_size;
    req->ucp.memtype               = memtype;
    req->ucp.flags                 = 0;

    if (is_multicast) {
      req->am_send.mc_desc = new MCDesc(targets.size());
      CHKERR_JUMP(req->am_send.mc_desc == nullptr,
          "failed to new multicast desc", log_ucp, err_rel_payload);
    } else {
      req->am_send.mc_desc = nullptr;
    }

    return req;

err_rel_payload:
    if (payload_base_type == PAYLOAD_BASE_INTERNAL) {
      internal->pbuf_release(worker, payload_base);
    }
err_rel_req:
    internal->request_release(req);
err:
    return nullptr;
  }

  // static because it will be called from completion callbacks too
  void UCPMessageImpl::cleanup_request(Request *req, UCPInternal *internal)
  {
    if (req->am_send.payload_base_type == PAYLOAD_BASE_INTERNAL) {
      internal->pbuf_release(req->worker, req->ucp.payload);
    }
    delete req->am_send.local_comp;
    delete req->am_send.mc_desc;
  }

  void UCPMessageImpl::am_put_comp_handler(void *request,
      ucs_status_t status, void *user_data)
  {
    Request *req          = reinterpret_cast<Request*>(user_data);
    UCPInternal *internal = req->internal;

    log_ucp_am.debug() << "am_put_comp_handler invoked for request " << req;

    if (status != UCS_OK) {
      log_ucp.error() << "failed to complete put for am";
    }

    ucp_rkey_destroy(req->ucp.rma.rkey);
    free(req->rma.rdma_info_buf);
    internal->request_release(req);
  }

  void UCPMessageImpl::am_put_flush_comp_handler(void *request,
      ucs_status_t status, void *user_data)
  {
    Request *req          = reinterpret_cast<Request*>(user_data);
    UCPInternal *internal = req->internal;

    log_ucp_am.debug()
      << "am_put_flush_comp_handler invoked for request " << req;

    CHKERR_JUMP(status != UCS_OK,
        "failed to complete flush for am", log_ucp, err);

    req->ucp.flags |= UCP_AM_SEND_FLAG_EAGER;
    // am payload has already been put on the target.
    req->ucp.payload_size = 0;
    // we're sending header only; memtype will be host
    req->ucp.memtype = UCS_MEMORY_TYPE_HOST;
    CHKERR_JUMP(!UCPMessageImpl::send_request(req, AM_ID_RDMA),
        "failed to send am request in put flush callback", log_ucp, err);

    return;

err:
    UCPMessageImpl::am_local_failure_handler(req, internal);
    // TODO: should invoke some higher-level error handler
  }

  bool UCPMessageImpl::commit_with_rma(ucp_ep_h ep)
  {
    Request *req, *req_put;
    ucp_rkey_h rkey;
    ucs_status_t status;

    req = make_request(ucp_msg_hdr.rdma_payload_size);
    CHKERR_JUMP(req == nullptr, "failed to make am request", log_ucp, err);

    req_put = internal->request_get(worker);
    CHKERR_JUMP(req_put == nullptr, "failed to get request", log_ucp, err_rel_req);

    status = ucp_ep_rkey_unpack(ep, dest_payload_rdma_info->rkey, &rkey);
    CHKERR_JUMP(status != UCS_OK,
        "ucp_ep_rkey_unpack failed", log_ucp, err_rel_req_put);

    req_put->ucp.op_type         = UCPWorker::OpType::PUT;
    req_put->ucp.ep              = ep;
    req_put->ucp.payload         = req->ucp.payload;
    req_put->ucp.payload_size    = req->ucp.payload_size;
    req_put->ucp.memtype         = req->ucp.memtype;
    req_put->ucp.flags           = 0;
    req_put->ucp.args            = req_put;
    req_put->ucp.cb              = &UCPMessageImpl::am_put_comp_handler;
    req_put->ucp.rma.rkey        = rkey;
    req_put->ucp.rma.remote_addr = dest_payload_rdma_info->reg_base;

    CHKERR_JUMP(!worker->submit_req(&req_put->ucp),
        "failed to commit with rma", log_ucp, err_dest_rkey);

    // we reuse the same req for both flush and am send after it
    req->ucp.op_type = UCPWorker::OpType::EP_FLUSH;
    req->ucp.ep      = ep;
    req->ucp.args    = req;
    req->ucp.cb      = &UCPMessageImpl::am_put_flush_comp_handler;

    CHKERR_JUMP(!worker->submit_req(&req->ucp),
        "failed to commit with rma", log_ucp, err_dest_rkey);

    return true;

err_dest_rkey:
    ucp_rkey_destroy(rkey);
err_rel_req_put:
    internal->request_release(req_put);
err_rel_req:
    UCPMessageImpl::am_local_failure_handler(req, internal);
err:
    free(dest_payload_rdma_info);
    return false;
  }

  bool UCPMessageImpl::commit_multicast(size_t act_payload_size)
  {
    size_t to_submit     = targets.size();
    int remote_dev_index = dest_payload_rdma_info ?
                           dest_payload_rdma_info->dev_index : -1;
    Request *req_prim, *req;

    req_prim = make_request(act_payload_size);
    CHKERR_JUMP(req_prim == nullptr, "failed to make am request", log_ucp, err);

    for (const NodeID &target : targets) {
      // shallow-copy the primary request for each target
      // IMPORTANT: the primary requet must not be released
      //            until all target copies have been created.
      //            Otherwise, it may be released upon completion
      //            which will make furhter copies invalid.
      req = internal->request_get(worker);
      if (req == nullptr) {
        log_ucp.error() << "failed to get additional request for multicast";
        req = req_prim;
        goto err_update_pending;
      }
      *req = *req_prim;
      CHKERR_JUMP(!worker->ep_get(target, remote_dev_index, &req->ucp.ep),
          "failed to get ep", log_ucp, err);
      if (!UCPMessageImpl::send_request(req, AM_ID)) {
        log_ucp.error() << "failed to send multicast am request";
        goto err_update_pending;
      }

      to_submit--;
    }

    internal->request_release(req_prim);

    return true;

err_update_pending:
    req->am_send.mc_desc->local_pending.fetch_sub(to_submit);
    if (remote_comp != nullptr) {
      ucp_msg_hdr.remote_comp->remote_pending.fetch_sub(to_submit);
    }
    UCPMessageImpl::am_local_failure_handler(req, internal);
err:
    return false;
  }

  bool UCPMessageImpl::commit_unicast(size_t act_payload_size)
  {
    int remote_dev_index = dest_payload_rdma_info ?
                           dest_payload_rdma_info->dev_index : -1;
    bool ret;
    ucp_ep_h ep;

    CHKERR_JUMP(!worker->ep_get(target, remote_dev_index, &ep),
        "failed to get ep", log_ucp, err);

    if (dest_payload_rdma_info == nullptr) {
      if (header_size + act_payload_size <= internal->config.fp_max) {
        ret = send_fast_path(ep, act_payload_size) ||
              send_slow_path(ep, act_payload_size, 0);
      } else {
        ret = send_slow_path(ep, act_payload_size, 0);
      }
    } else if ((internal->config.am_wra_mode == AM_WITH_REMOTE_ADDR_MODE_AUTO ||
               internal->config.am_wra_mode == AM_WITH_REMOTE_ADDR_MODE_AM)) {
      // have remote buffer info and am mode is not rma
      // send with am rndv
      // TODO: use rma if source buffer is GPU, and
      //       use am rndv if source buffer is host.
      //       Because, ucp put does not support multi-rail,
      //       but multi-rail benefits are seen when sending
      //       from host buffer.
      log_ucp_am.debug() << "sending am with remote address using forced rndv";
      ret = send_slow_path(ep, act_payload_size, UCP_AM_SEND_FLAG_RNDV);
    } else {
      // send with ucp rma
      log_ucp_am.debug() << "sending am with remote address using rma";
      ret = commit_with_rma(ep);
    }

    return ret;

err:
    return false;
  }

  void UCPMessageImpl::commit(size_t act_payload_size)
  {
    bool status;
    const UCPContext *context = worker->get_context();

#ifdef REALM_USE_CUDA
    Cuda::AutoGPUContext agc(context->gpu);
#endif

    // For now, copy non-contiguous data. TODO: Use UCP IOV
    if ((src_payload_addr != nullptr) && (src_payload_lines > 1)) {
      assert(payload_base != nullptr);
      log_ucp_am.info() << "committing non-contiguous payload";
      size_t bytes_per_line = act_payload_size / src_payload_lines;
      for(size_t i = 0; i < src_payload_lines; i++) {
        memcpy(reinterpret_cast<char *>(payload_base) + (i * bytes_per_line),
            reinterpret_cast<const char *>(src_payload_addr) +
                                          (i * src_payload_line_stride),
            bytes_per_line);
      }
    }

    // payload_base must point to payload data unless payload size is 0
    assert((payload_base != nullptr) || (act_payload_size == 0));
    assert(payload_base_type != PAYLOAD_BASE_LAST || payload_size == 0);

    ucp_msg_hdr.remote_comp       = remote_comp;
    ucp_msg_hdr.rdma_payload_size = act_payload_size;

    if (internal->config.crc_check) {
      insert_packet_crc(&ucp_msg_hdr, header_size, act_payload_size);
    }

    if (is_multicast) {
      status = commit_multicast(act_payload_size);
    } else {
      status = commit_unicast(act_payload_size);
    }

    if (!status) {
      log_ucp.error() << "failed to commit am";
    }

    log_ucp_am.info()
       << "msg commit "     << (is_multicast ? "multicast" : "unicast")
       << "context "        << context
       << "worker "         << worker
       << " target "        << (is_multicast ? -1 : target)
       << " hsize "         << header_size
       << " psize "         << act_payload_size
       << " ptype "         << payload_base_type
#ifdef REALM_USE_CUDA
       << " src_mem_type "  << (context->gpu ? "cuda" : "host")
       << " dst_mem_type "  << (dest_payload_rdma_info ? (dest_payload_rdma_info->dev_index == -1 ? "host" : "cuda") : "host")
       << " src_dev_index " << (context->gpu ? context->gpu->info->index : -1)
       << " dst_dev_index " << (dest_payload_rdma_info ? dest_payload_rdma_info->dev_index : -1)
#endif
       << " remote_addr "   << (dest_payload_rdma_info ? dest_payload_rdma_info->reg_base : 0);
  }

  void UCPMessageImpl::cancel()
  {
    delete local_comp;
    delete remote_comp;
    free(dest_payload_rdma_info);
    if (payload_base_type == PAYLOAD_BASE_INTERNAL) {
      internal->pbuf_release(worker, payload_base);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPRemoteMemoryCommon
  //

  UCPRemoteMemoryCommon::UCPRemoteMemoryCommon(const ByteArray& _rdma_info_ba)
      : rdma_info_size(_rdma_info_ba.size())
  {
    rdma_info = reinterpret_cast<UCPRDMAInfo*>(malloc(_rdma_info_ba.size()));
    assert(rdma_info != nullptr);

    memcpy(rdma_info, _rdma_info_ba.base(), _rdma_info_ba.size());
  }

  UCPRemoteMemoryCommon::~UCPRemoteMemoryCommon()
  {
    free(rdma_info);
  }

  bool UCPRemoteMemoryCommon::get_remote_addr(off_t offset,
      RemoteAddress& remote_addr)
  {
    assert(rdma_info_size <= sizeof(remote_addr.raw_bytes));

    memcpy(remote_addr.raw_bytes, rdma_info, rdma_info_size);

    uint64_t addr = rdma_info->reg_base + offset;
    memcpy(remote_addr.raw_bytes, &addr, sizeof(addr));

    return true;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPRemoteMemory
  //

  UCPRemoteMemory::UCPRemoteMemory(Memory _me,
      size_t _size,
      Memory::Kind _kind,
      const ByteArray& _rdma_info_ba,
      UCPInternal *_internal)
      : UCPRemoteMemoryCommon(_rdma_info_ba)
      , RemoteMemory(_me, _size, _kind, MKIND_RDMA)
  {}

  void UCPRemoteMemory::get_bytes(off_t offset, void *dst, size_t size)
  {
    // rkey unpack requires an ep. With multiple ucp contexts, the ep
    // is not known until we have both send and recv buffer info in
    // addition to node id. So, unpacking here.
    //ucs_status_t status = ucp_ep_rkey_unpack(ep, rdma_info->rkey, &rkey);
    //assert(status == UCS_OK);
    //ucp_rkey_destroy(rkey);
    assert(0);
  }

  void UCPRemoteMemory::put_bytes(off_t offset, const void *src, size_t size)
  {
    assert(0);
  }

  bool UCPRemoteMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    return UCPRemoteMemoryCommon::get_remote_addr(offset, remote_addr);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPIBMemory
  //

  UCPIBMemory::UCPIBMemory(Memory _me,
      size_t _size,
      Memory::Kind _kind,
      const ByteArray& _rdma_info_ba,
      UCPInternal *_internal)
      : UCPRemoteMemoryCommon(_rdma_info_ba)
      , IBMemory(_me, _size, MKIND_REMOTE, _kind, 0, 0)
  {}

  bool UCPIBMemory::get_remote_addr(off_t offset, RemoteAddress& remote_addr)
  {
    return UCPRemoteMemoryCommon::get_remote_addr(offset, remote_addr);
  }

}; // namespace UCP

}; // namespace Realm
