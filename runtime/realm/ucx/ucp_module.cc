
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

// UCP network module implementation for Realm

#include "realm/network.h"
#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"

#include "realm/ucx/ucp_module.h"
#include "realm/ucx/ucp_internal.h"

#ifdef REALM_UCX_MODULE_DYNAMIC
REGISTER_REALM_NETWORK_MODULE_DYNAMIC(Realm::UCPModule);
#endif

namespace Realm {

  Logger log_ucp("ucp");

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPModule
  //

  UCPModule::UCPModule(Realm::RuntimeImpl *runtime)
    : NetworkModule("ucp")
  {
    internal = new Realm::UCP::UCPInternal(this, runtime);
    assert(internal);
  }

  UCPModule::~UCPModule()
  {
    delete internal;
  }

  /*static*/
  NetworkModule *UCPModule::create_network_module(
          RuntimeImpl *runtime,
          int *argc,
		  const char ***argv)
  {
    UCPModule *mod = new UCPModule(runtime);
    assert(mod);

    bool status = mod->internal->bootstrap();
    CHKERR_JUMP(!status, "failed to create UCP network module", log_ucp, err_del_mod);

    return mod;

err_del_mod:
    delete mod;
    return NULL;
  }

  void UCPModule::get_shared_peers(NodeSet &shared_peers)
  {
    internal->get_shared_peers(shared_peers);
  }

  void UCPModule::parse_command_line(RuntimeImpl *runtime,
                                     std::vector<std::string> &cmdline)
  {
    CommandLineParser cp;
    Realm::UCP::UCPInternal::Config config;
    // deferred_allocs realm test always passes -ll:gsize
    size_t deprecated_gsize = 0;
    cp.add_option_int_units("-ll:gsize", deprecated_gsize, 'm');

    std::string am_mode;
    cp.add_option_string("-ucx:am_mode", am_mode);

    cp.add_option_string("-ucx:host_nics", config.host_nics);

    cp.add_option_int("-ucx:bindhost", config.bind_hostmem);

    // maximum number of background work items to use for UCP polling
    cp.add_option_int("-ucx:pollers_max", config.pollers_max);

    // number of message priority levels to support
    cp.add_option_int("-ucx:num_priorities", config.num_priorities);

    // maximum message size that is sent with higher priority
    cp.add_option_int_units("-ucx:priority_size_max", config.priority_size_max);

    // check memory pools for leak
    cp.add_option_bool("-ucx:mpool_leakcheck", config.mpool_leakcheck);

    // CRC check
    cp.add_option_int("-ucx:crc_check", config.crc_check);

    cp.add_option_bool("-ucx:hbuf_malloc", config.hbuf_malloc);
    cp.add_option_bool("-ucx:pbuf_malloc", config.pbuf_malloc);

    // use UCX wakeup feature for progress
    cp.add_option_bool("-ucx:use_wakeup", config.use_wakeup);

    cp.add_option_int("-ucx:prog_boff_max", config.prog_boff_max);
    cp.add_option_int("-ucx:prog_itr_max", config.prog_itr_max);
    cp.add_option_int("-ucx:rdesc_rel_max", config.rdesc_rel_max);

    // pbuf mpool
    cp.add_option_int("-ucx:pb_init_count", config.pbuf_init_count);
    cp.add_option_int("-ucx:pb_max_count", config.pbuf_max_count);
    cp.add_option_int_units("-ucx:pb_max_size", config.pbuf_max_size);
    cp.add_option_int_units("-ucx:pb_max_chunk_size",
        config.pbuf_max_chunk_size);
    cp.add_option_int_units("-ucx:pb_mp_thresh", config.pbuf_mp_thresh);

    // malloc mpool
    cp.add_option_int_units("-ucx:mmp_max_obj_size", config.mmp_max_obj_size);

    // max size for which we try the fast path first
    cp.add_option_int_units("-ucx:fp_max", config.fp_max);

    cp.add_option_string("-ucx:ib_seg_size", config.ib_seg_size);

    cp.add_option_int("-ucx:oreqs_limit", config.outstanding_reqs_limit);

    cp.add_option_string("-ucx:zcopy_host", config.zcopy_thresh_host);
    cp.add_option_string("-ucx:tls_host", config.tls_host);

#ifdef REALM_USE_CUDA
    cp.add_option_int("-ucx:bindcuda", config.bind_cudamem);
    cp.add_option_string("-ucx:gpu_nics", config.gpu_nics);
    cp.add_option_string("-ucx:zcopy_dev", config.zcopy_thresh_dev);
    cp.add_option_string("-ucx:tls_dev", config.tls_dev);
#endif

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    if(deprecated_gsize > 0) {
      log_ucp.fatal() << "Realm UCX backend does not provide a 'global' memory."
                      << " '-ll:gsize' not permitted";
      abort();
    }

    //// set internal config ////

    // am-with-remote-address config
    if (!am_mode.empty()) {
      if (am_mode == "auto") {
        config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_AUTO;
      } else if (am_mode == "put") {
        config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_PUT;
      } else if (am_mode == "am") {
        config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_AM;
      } else {
        log_ucp.fatal() << "invalid mode for am with remote address " << am_mode
                        << ". Valid choices: auto, put, am";
        abort();
      }
    }

    if (!internal->init(config)) {
      log_ucp.fatal() << "internal init failed";
      abort();
    }
  }

  void UCPModule::attach(RuntimeImpl *runtime,
      std::vector<NetworkSegment *>& segments)
  {
    internal->attach(segments);
  }

  void UCPModule::detach(RuntimeImpl *runtime,
      std::vector<NetworkSegment *>& segments)
  {
    internal->detach(segments);
    // Call finalize here because it uses other realm
    // objects (e.g., gpu) which may be destroyed before
    // the network module's destructor is called.
    internal->finalize();
  }

  void UCPModule::create_memories(RuntimeImpl *runtime)
  {
  }

  void UCPModule::barrier(void)
  {
    internal->barrier();
  }

  void UCPModule::broadcast(NodeID root,
      const void *val_in, void *val_out, size_t bytes)
  {
    internal->broadcast(root, val_in, val_out, bytes);
  }

  void UCPModule::gather(NodeID root,
      const void *val_in, void *vals_out, size_t bytes)
  {
    internal->gather(root, val_in, vals_out, bytes);
  }

  void UCPModule::allgatherv(const char *val_in, size_t bytes,
                             std::vector<char> &vals_out, std::vector<size_t> &lengths)
  {
    internal->allgatherv(val_in, bytes, vals_out, lengths);
  }

  size_t UCPModule::sample_messages_received_count(void)
  {
    return internal->sample_messages_received_count();
  }

  bool UCPModule::check_for_quiescence(size_t sampled_receive_count)
  {
    return internal->check_for_quiescence(sampled_receive_count);
  }

  MemoryImpl* UCPModule::create_remote_memory(Memory me,
      size_t size,
      Memory::Kind kind,
      const ByteArray& rdma_info_ba)
  {
    return new Realm::UCP::UCPRemoteMemory(me, size, kind, rdma_info_ba, internal);
  }

  IBMemory* UCPModule::create_remote_ib_memory(Memory me,
      size_t size,
      Memory::Kind kind,
      const ByteArray& rdma_info_ba)
  {
    return new Realm::UCP::UCPIBMemory(me, size, kind, rdma_info_ba, internal);
  }

  ActiveMessageImpl* UCPModule::create_active_message_impl(NodeID target,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      void *storage_base,
      size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal,
        target,
        msgid,
        header_size,
        max_payload_size,
        src_payload_addr,
        src_payload_lines,
        src_payload_line_stride,
        storage_size);
  }

  ActiveMessageImpl* UCPModule::create_active_message_impl(NodeID target,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      const RemoteAddress& dest_payload_addr,
      void *storage_base,
      size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal,
        target,
        msgid,
        header_size,
        max_payload_size,
        src_payload_addr,
        src_payload_lines,
        src_payload_line_stride,
        dest_payload_addr,
        storage_size);
  }

  ActiveMessageImpl* UCPModule::create_active_message_impl(const NodeSet& targets,
      unsigned short msgid,
      size_t header_size,
      size_t max_payload_size,
      const void *src_payload_addr,
      size_t src_payload_lines,
      size_t src_payload_line_stride,
      void *storage_base,
      size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal,
        targets,
        msgid,
        header_size,
        max_payload_size,
        src_payload_addr,
        src_payload_lines,
        src_payload_line_stride,
        storage_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
      bool with_congestion,
      size_t header_size)
  {
    return internal->recommended_max_payload(target, nullptr,
        with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(const NodeSet& targets,
      bool with_congestion,
      size_t header_size)
  {
    if (targets.size() == 1) {
      // this is same as single-target case
      NodeID target = *(targets.begin());
      return internal->recommended_max_payload(target, nullptr,
          with_congestion, header_size);
    }

    // just use the no-target version
    return internal->recommended_max_payload(nullptr, with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
      const RemoteAddress& dest_payload_addr,
      bool with_congestion,
      size_t header_size)
  {
    const RemoteAddress *dp_addr = &dest_payload_addr;

    return internal->recommended_max_payload(target, dp_addr,
        with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
      const void *data, size_t bytes_per_line,
      size_t lines, size_t line_stride,
      bool with_congestion,
      size_t header_size)
  {
    return internal->recommended_max_payload(target, nullptr,
        data, bytes_per_line, lines, line_stride,
        with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(const NodeSet& targets,
      const void *data, size_t bytes_per_line,
      size_t lines, size_t line_stride,
      bool with_congestion,
      size_t header_size)
  {
    if (targets.size() == 1) {
      // this is same as single-target case
      NodeID target = *(targets.begin());
      return internal->recommended_max_payload(target, nullptr,
          data, bytes_per_line, lines, line_stride,
          with_congestion, header_size);
    }

    // just use the no-target version
    return internal->recommended_max_payload(nullptr,
        data, bytes_per_line, lines, line_stride,
        with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
      const void *data, size_t bytes_per_line,
      size_t lines, size_t line_stride,
      const RemoteAddress& dest_payload_addr,
      bool with_congestion,
      size_t header_size)
  {
    const RemoteAddress *dp_addr = &dest_payload_addr;

    return internal->recommended_max_payload(target, dp_addr,
        data, bytes_per_line, lines, line_stride,
        with_congestion, header_size);
  }

}; // namespace Realm
