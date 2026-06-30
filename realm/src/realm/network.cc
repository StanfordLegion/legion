/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

// Realm inter-node networking abstractions

#include "realm/network.h"
#include "realm/cmdline.h"
#include "realm/logging.h"
#include "realm/activemsg.h"
#include "realm/timers.h"

#ifdef REALM_USE_DLFCN
#include <dlfcn.h>
#endif

static void *aligned_malloc(size_t bytes, size_t alignment)
{
#ifdef REALM_ON_WINDOWS
  return _aligned_malloc(bytes, alignment);
#else
  void *ptr = 0;
  int ret = posix_memalign(&ptr, alignment, bytes);
  return ((ret == 0) ? ptr : 0);
#endif
}

static void aligned_free(void *ptr)
{
#ifdef REALM_ON_WINDOWS
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

namespace Realm {

  Logger log_quiesce("quiesce");

  namespace Network {
    REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeID my_node_id = 0;
    REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeID max_node_id = 0;
    REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeSet all_peers;
    REALM_INTERNAL_API_EXTERNAL_LINKAGE NodeSet shared_peers;
    NetworkModule *single_network = 0;

    // Persistent state across calls to check_for_quiescence.  The function
    //  is only invoked from RuntimeImpl::wait_for_shutdown, single-threaded,
    //  so a translation-unit-local static is sufficient.  Reset in
    //  reset_quiescence_state() below if the runtime ever needs to start over.
    namespace QuiescenceCheck {
      static NetworkModule::QuiescenceState prev_totals = {};
      static bool have_prev = false;
      // wall-clock timestamp (ns) of the last round in which any counter
      //  changed.  When counters stay frozen for longer than WARN_INTERVAL_NS,
      //  we emit a warning - but never abort.  If the user's network is
      //  genuinely slow, we just keep iterating; if it's hung, we just keep
      //  warning so the user knows where the process is stuck.
      static long long last_change_time_ns = 0;
      // wall-clock timestamp (ns) at which the next "no progress" warning
      //  should fire.  Bumped by WARN_INTERVAL_NS every time a warning is
      //  emitted, so warnings repeat at a steady cadence rather than
      //  spamming.
      static long long next_warn_time_ns = 0;
      // emit a warning if the counters have been frozen this long.  60 s is
      //  long enough to outlast any reasonable network round-trip but short
      //  enough that a CI hang surfaces as actionable log output.
      static const long long WARN_INTERVAL_NS = 60LL * 1000 * 1000 * 1000;
    } // namespace QuiescenceCheck

    void reset_quiescence_state(void)
    {
      QuiescenceCheck::prev_totals = NetworkModule::QuiescenceState{};
      QuiescenceCheck::have_prev = false;
      QuiescenceCheck::last_change_time_ns = 0;
      QuiescenceCheck::next_warn_time_ns = 0;
    }

    QuiescenceStatus check_for_quiescence(IncomingMessageManager *message_manager)
    {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      if(REALM_UNLIKELY(single_network == 0)) {
        return QuiescenceStatus::DONE;
      }
#endif

      QuiescenceStatus custom_status;
      if(single_network->custom_quiescence_check(message_manager, custom_status))
        return custom_status;

      // Drain the incoming-message queue first, so that any messages already
      //  delivered by the network layer have been dispatched to their
      //  handlers (which may queue more local work, also captured by
      //  queued_items).  The drain target is messages_to_drain - a strict
      //  subset of packets_received that excludes wire packets that don't
      //  pass through IMM (UCX remote-completion replies in particular).
      //  Using packets_received here would hang on UCX since IMM's
      //  total_messages_handled can never include rcomp arrivals.
      NetworkModule::QuiescenceState predrain;
      single_network->sample_quiescence_state(predrain);
      message_manager->drain_incoming_messages(predrain.messages_to_drain);

      // Now sample the actual local state we'll feed into the allreduce.
      //  Sampling AFTER the drain matters: drain may have caused handlers to
      //  fire, which may have changed queued_items and pending counts.
      NetworkModule::QuiescenceState local;
      single_network->sample_quiescence_state(local);

      // Allreduce the five fields.  All use SUM:
      //   queued_items: each rank contributes its current queue-item count;
      //     sum > 0 means at least one rank has work queued.  Using a count
      //     rather than a 0/1 boolean lets a draining queue register as
      //     "progressing" (sum decreasing) instead of looking unchanged
      //     across rounds while still non-zero
      //   events_added: monotonic count of items ever added to any queue;
      //     required to detect activity in cases where queued_items happens
      //     to be the same value across rounds while contents flowed through
      //   packets_reserved/received: cumulative counts; sum across ranks
      //   pending_completions: count per rank; sum > 0 means at least one
      //     rank is waiting on a remote completion
      uint64_t local_arr[5] = {local.queued_items, local.events_added,
                               local.packets_reserved, local.packets_received,
                               local.pending_completions};
      uint64_t total_arr[5] = {0, 0, 0, 0, 0};
      single_network->quiescence_allreduce_sum(local_arr, total_arr, 5);

      NetworkModule::QuiescenceState totals = {total_arr[0], total_arr[1], total_arr[2],
                                               total_arr[3], total_arr[4]};

      // Quiet means: no rank has anything queued, no rank is waiting on a
      //  remote completion, and the cumulative sent/received counts balance
      //  globally (no messages in flight).  These are the conditions Mattern's
      //  algorithm requires for termination of one round.
      bool quiet_now = (totals.queued_items == 0) && (totals.pending_completions == 0) &&
                       (totals.packets_reserved == totals.packets_received);

      // The Mattern's stability check: termination is confirmed only when
      //  two CONSECUTIVE rounds both observe a quiet state AND the same
      //  cumulative counts.  If anything changed between rounds - a new send
      //  was initiated, a queued event was processed and the count dropped,
      //  etc. - the counters will differ, and we restart.  Two-round
      //  agreement rules out the ghost-message scenario where a work item
      //  runs between rounds and produces a send that happens to be in
      //  flight at exactly the wrong moment.  Using a count for queued_items
      //  also keeps a slow-but-progressing drain visible (the sum
      //  decreases monotonically), so the loop keeps iterating
      //  PROGRESSING rather than appearing stable at a non-zero value.
      //
      // ALIASING SAFETY: queued_items and pending_completions are
      //  snapshot fields (non-monotonic).  Each is checked alongside its
      //  monotonic mate so that activity which leaves the snapshot field
      //  unchanged still surfaces:
      //    queued_items       paired with events_added      (queue activity)
      //    pending_completions paired with packets_reserved (comp alloc/recycle)
      //  Joint stability of each pair rules out balanced add+remove
      //  activity.  See QuiescenceState in network.h for the per-field
      //  pairing argument.
      bool stable =
          QuiescenceCheck::have_prev &&
          (totals.packets_reserved == QuiescenceCheck::prev_totals.packets_reserved) &&
          (totals.packets_received == QuiescenceCheck::prev_totals.packets_received) &&
          (totals.pending_completions ==
           QuiescenceCheck::prev_totals.pending_completions) &&
          (totals.queued_items == QuiescenceCheck::prev_totals.queued_items) &&
          (totals.events_added == QuiescenceCheck::prev_totals.events_added);

      if(quiet_now && stable) {
        // both rounds agreed on a quiet state - termination confirmed
        reset_quiescence_state();
        return QuiescenceStatus::DONE;
      }

      // Track wall-clock time since counters last moved.  We never abort -
      //  without a way to introspect the network we cannot distinguish a
      //  slow-but-progressing in-flight message from a lost one, so any
      //  timeout-based abort would be a guess.  Instead, we periodically
      //  emit a warning when counters have been frozen for a while so the
      //  user knows shutdown is making no observable progress.
      long long now_ns = Clock::current_time_in_nanoseconds();
      if(!stable) {
        // counters changed (or this is the first call) - reset the timer
        QuiescenceCheck::last_change_time_ns = now_ns;
        QuiescenceCheck::next_warn_time_ns = now_ns + QuiescenceCheck::WARN_INTERVAL_NS;
      } else if(now_ns >= QuiescenceCheck::next_warn_time_ns) {
        // counters frozen for at least WARN_INTERVAL_NS, and we haven't
        //  warned recently - emit a warning and schedule the next one.  Only
        //  rank 0 prints to avoid every rank logging the same thing.
        if(my_node_id == 0) {
          long long stalled_ns = now_ns - QuiescenceCheck::last_change_time_ns;
          log_quiesce.warning()
              << "no progress on quiescence for " << (stalled_ns / 1000000000)
              << " s; queued=" << totals.queued_items
              << " events_added=" << totals.events_added
              << " reserved=" << totals.packets_reserved
              << " received=" << totals.packets_received
              << " pending_comps=" << totals.pending_completions
              << " (this is a warning, not a fatal error - shutdown will keep retrying)";
        }
        QuiescenceCheck::next_warn_time_ns += QuiescenceCheck::WARN_INTERVAL_NS;
      }

      // record state for the next round
      QuiescenceCheck::prev_totals = totals;
      QuiescenceCheck::have_prev = true;
      return QuiescenceStatus::PROGRESSING;
    }
  } // namespace Network

  ////////////////////////////////////////////////////////////////////////
  //
  // class NetworkModule
  //

  NetworkModule::NetworkModule(const std::string &_name)
    : Module(_name)
  {}

  void NetworkModule::parse_command_line(RuntimeImpl *runtime,
                                         std::vector<std::string> &cmdline)
  {}

  bool NetworkModule::custom_quiescence_check(IncomingMessageManager *message_manager,
                                              Network::QuiescenceStatus &status)
  {
    (void)message_manager;
    (void)status;
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class NetworkSegment
  //

  NetworkSegment::NetworkSegment()
    : base(0)
    , bytes(0)
    , alignment(0)
    , memtype(NetworkSegmentInfo::Unknown)
    , memextra(0)
    , single_network(0)
    , single_network_data(0)
  {}

#if 0
  // normally a request will just be for a particular size
  inline NetworkSegment::NetworkSegment(size_t _bytes, size_t _alignment)
    : base(0), bytes(_bytes), alignment(_alignment)
    , single_network(0), single_network_data(0)
  {}

  // but it can also be for a pre-allocated chunk of memory with a fixed address
  inline NetworkSegment::NetworkSegment(void *_base, size_t _bytes)
    : base(_base), bytes(_bytes), alignment(0)
    , single_network(0), single_network_data(0)
  {}
#endif

  void NetworkSegment::request(NetworkSegmentInfo::MemoryType _memtype, size_t _bytes,
                               size_t _alignment,
                               NetworkSegmentInfo::MemoryTypeExtraData _memextra /*= 0*/,
                               NetworkSegmentInfo::FlagsType _flags /*= 0*/)
  {
    memtype = _memtype;
    bytes = _bytes;
    alignment = _alignment;
    memextra = _memextra;
    flags = _flags;
  }

  void NetworkSegment::assign(NetworkSegmentInfo::MemoryType _memtype, void *_base,
                              size_t _bytes,
                              NetworkSegmentInfo::MemoryTypeExtraData _memextra /*= 0*/,
                              NetworkSegmentInfo::FlagsType _flags /*= 0*/)
  {
    memtype = _memtype;
    base = _base;
    bytes = _bytes;
    memextra = _memextra;
    flags = _flags;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class LoopbackNetworkModule
  //

  // used when there are no other networks

  class LoopbackNetworkModule : public NetworkModule {
  protected:
    LoopbackNetworkModule();

  public:
    static NetworkModule *create_network_module(RuntimeImpl *runtime, int *argc,
                                                const char ***argv);

    // Enumerates all the peers that the current node could potentially share memory with
    virtual void get_shared_peers(NodeSet &shared_peers);

    // actual parsing of the command line should wait until here if at all
    //  possible
    virtual void parse_command_line(RuntimeImpl *runtime,
                                    std::vector<std::string> &cmdline);

    // "attaches" to the network, if that is meaningful - attempts to
    //  bind/register/(pick your network-specific verb) the requested memory
    //  segments with the network
    virtual void attach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments);

    // detaches from the network
    virtual void detach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments);

    // collective communication within this network
    virtual void barrier(void);
    virtual void broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes);
    virtual void gather(NodeID root, const void *val_in, void *vals_out, size_t bytes);
    virtual void allgatherv(const char *val_in, size_t bytes, std::vector<char> &vals_out,
                            std::vector<size_t> &lengths);

    virtual void sample_quiescence_state(QuiescenceState &state);
    virtual void quiescence_allreduce_sum(const uint64_t *local_counts,
                                          uint64_t *total_counts, size_t count);

    // used to create a remote proxy for a memory
    virtual MemoryImpl *create_remote_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                             Memory::Kind kind,
                                             const ByteArray &rdma_info);
    virtual IBMemory *create_remote_ib_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                              Memory::Kind kind,
                                              const ByteArray &rdma_info);

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size);

    virtual ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size);

    virtual ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const RemoteAddress &dest_payload_addr, void *storage_base, size_t storage_size);

    virtual ActiveMessageImpl *create_active_message_impl(
        const NodeSet &targets, unsigned short msgid, size_t header_size,
        size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, void *storage_base, size_t storage_size);

    virtual size_t recommended_max_payload(NodeID target, bool with_congestion,
                                           size_t header_size);
    virtual size_t recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                           size_t header_size);
    virtual size_t recommended_max_payload(NodeID target,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size);
    virtual size_t recommended_max_payload(NodeID target, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size);
    virtual size_t recommended_max_payload(const NodeSet &targets, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size);
    virtual size_t recommended_max_payload(NodeID target,
                                           const LocalAddress &src_payload_addr,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size);

    virtual size_t max_payload_size(size_t header_size, const void *src_payload_addr);
  };

  LoopbackNetworkModule::LoopbackNetworkModule()
    : NetworkModule("loopback")
  {}

  /*static*/ NetworkModule *
  LoopbackNetworkModule::create_network_module(RuntimeImpl *runtime, int *argc,
                                               const char ***argv)
  {
    return new LoopbackNetworkModule;
  }

  void LoopbackNetworkModule::get_shared_peers(NodeSet &shared_peers) {}

  // actual parsing of the command line should wait until here if at all
  //  possible
  void LoopbackNetworkModule::parse_command_line(RuntimeImpl *runtime,
                                                 std::vector<std::string> &cmdline)
  {
    NetworkModule::parse_command_line(runtime, cmdline);

    size_t global_size = 0;
    CommandLineParser cp;
    cp.add_option_int_units("-ll:gsize", global_size, 'm');
    bool ok = cp.parse_command_line(cmdline);
    assert(ok);
    assert((global_size == 0) && "no global mem support in dummy network yet");
  }

  // "attaches" to the network, if that is meaningful - attempts to
  //  bind/register/(pick your network-specific verb) the requested memory
  //  segments with the network
  void LoopbackNetworkModule::attach(RuntimeImpl *runtime,
                                     std::vector<NetworkSegment *> &segments)
  {
    // service any still-unbound request by doing a malloc
    for(std::vector<NetworkSegment *>::iterator it = segments.begin();
        it != segments.end(); ++it) {
      if(((*it)->bytes > 0) && ((*it)->base == 0)) {
        void *memptr =
            aligned_malloc((*it)->bytes, std::max((*it)->alignment, sizeof(void *)));
        assert(memptr != 0);
        (*it)->base = memptr;
        (*it)->add_rdma_info(this, &memptr, sizeof(void *));
      }
    }
  }

  // detaches from the network
  void LoopbackNetworkModule::detach(RuntimeImpl *runtime,
                                     std::vector<NetworkSegment *> &segments)
  {
    // free any segment memory we allocated
    for(std::vector<NetworkSegment *>::iterator it = segments.begin();
        it != segments.end(); ++it) {
      const ByteArray *rdma_info = (*it)->get_rdma_info(this);
      if(rdma_info) {
        aligned_free((*it)->base);
        (*it)->base = 0;
      }
    }
  }

  // collective communication within this network
  void LoopbackNetworkModule::barrier(void)
  {
    // nothing to do
  }

  void LoopbackNetworkModule::broadcast(NodeID root, const void *val_in, void *val_out,
                                        size_t bytes)
  {
    memcpy(val_out, val_in, bytes);
  }

  void LoopbackNetworkModule::gather(NodeID root, const void *val_in, void *vals_out,
                                     size_t bytes)
  {
    memcpy(vals_out, val_in, bytes);
  }

  void LoopbackNetworkModule::allgatherv(const char *val_in, size_t bytes,
                                         std::vector<char> &vals_out,
                                         std::vector<size_t> &lengths)
  {
    vals_out.resize(bytes);
    lengths[0] = bytes;
    memcpy(vals_out.data(), val_in, bytes);
  }

  void LoopbackNetworkModule::sample_quiescence_state(QuiescenceState &state)
  {
    // single-rank loopback: nothing to send to anyone, nothing in flight.
    //  All counters are zero, no queues, no pending.
    state = QuiescenceState{0, 0, 0, 0, 0, 0};
  }

  void LoopbackNetworkModule::quiescence_allreduce_sum(const uint64_t *local_counts,
                                                       uint64_t *total_counts,
                                                       size_t count)
  {
    // single-rank: the "all-reduced sum" is just the local value
    for(size_t i = 0; i < count; i++)
      total_counts[i] = local_counts[i];
  }

  // used to create a remote proxy for a memory
  MemoryImpl *LoopbackNetworkModule::create_remote_memory(RuntimeImpl *runtime, Memory m,
                                                          size_t size, Memory::Kind kind,
                                                          const ByteArray &rdma_info)
  {
    // should never be called
    abort();
  }

  IBMemory *LoopbackNetworkModule::create_remote_ib_memory(RuntimeImpl *runtime, Memory m,
                                                           size_t size, Memory::Kind kind,
                                                           const ByteArray &rdma_info)
  {
    // should never be called
    abort();
  }

  ActiveMessageImpl *LoopbackNetworkModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const void *src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, void *storage_base, size_t storage_size)
  {
    // should never be called
    abort();
  }

  ActiveMessageImpl *LoopbackNetworkModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const LocalAddress &src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
      void *storage_base, size_t storage_size)
  {
    // should never be called
    abort();
  }

  ActiveMessageImpl *LoopbackNetworkModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const RemoteAddress &dest_payload_addr, void *storage_base, size_t storage_size)
  {
    // should never be called
    abort();
  }

  ActiveMessageImpl *LoopbackNetworkModule::create_active_message_impl(
      const NodeSet &targets, unsigned short msgid, size_t header_size,
      size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, void *storage_base, size_t storage_size)
  {
    // should never be called
    abort();
  }

  size_t LoopbackNetworkModule::recommended_max_payload(NodeID target,
                                                        bool with_congestion,
                                                        size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t LoopbackNetworkModule::recommended_max_payload(const NodeSet &targets,
                                                        bool with_congestion,
                                                        size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t
  LoopbackNetworkModule::recommended_max_payload(NodeID target,
                                                 const RemoteAddress &dest_payload_addr,
                                                 bool with_congestion, size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t LoopbackNetworkModule::recommended_max_payload(NodeID target, const void *data,
                                                        size_t bytes_per_line,
                                                        size_t lines, size_t line_stride,
                                                        bool with_congestion,
                                                        size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t LoopbackNetworkModule::recommended_max_payload(
      const NodeSet &targets, const void *data, size_t bytes_per_line, size_t lines,
      size_t line_stride, bool with_congestion, size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t LoopbackNetworkModule::recommended_max_payload(
      NodeID target, const LocalAddress &src_payload_addr, size_t bytes_per_line,
      size_t lines, size_t line_stride, const RemoteAddress &dest_payload_addr,
      bool with_congestion, size_t header_size)
  {
    // should never be called
    abort();
    return 0;
  }

  size_t LoopbackNetworkModule::max_payload_size(size_t header_size,
                                                 const void *src_payload_addr)
  {
    // loopback has no real limit
    (void)src_payload_addr;
    return std::numeric_limits<size_t>::max();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class NetworkSegment
  //

  void NetworkSegment::add_rdma_info(NetworkModule *network, const void *data, size_t len)
  {
    ByteArray &ba = networks[network];
    ba.set(data, len);
#ifdef REALM_USE_MULTIPLE_NETWORKS
#else
    assert(single_network == 0);
    single_network = network;
    single_network_data = &ba;
#endif
  }

  const ByteArray *NetworkSegment::get_rdma_info(NetworkModule *network) const
  {
    if(single_network == network) {
      return single_network_data;
    } else {
#ifdef REALM_USE_MULTIPLE_NETWORKS
      std::map<NetworkModule *, ByteArray>::iterator it = networks.find(network);
      if(it != networks.end())
        return &(it->second);
#endif
      return 0;
    }
  }

  bool NetworkSegment::is_registered() const
  {
    // first part - need rdma info
    if(single_network && single_network_data)
      return true;
#ifdef REALM_USE_MULTIPLE_NETWORKS
    // TODO: how do we know if a network is missing?
    return false;
#endif

    return false;
  }

  bool NetworkSegment::is_registered(NetworkModule *network) const
  {
    if(single_network) {
      return ((single_network == network) && single_network_data);
    }
#ifdef REALM_USE_MULTIPLE_NETWORKS
    if(networks.find(network) != networks.end())
      return true;
#endif

    return false;
  }

  bool NetworkSegment::in_segment(uintptr_t range_base, size_t range_bytes) const
  {
    uintptr_t reg_lo = reinterpret_cast<uintptr_t>(base);

    if(reg_lo == 0)
      return true;
    if(range_base < reg_lo)
      return false;

    uintptr_t reg_hi = reg_lo + (bytes - 1);
    if((bytes > 0) && ((range_base + range_bytes - 1) > reg_hi))
      return false;

    return true;
  }

  bool NetworkSegment::in_segment(const void *range_base, size_t range_bytes) const
  {
    return in_segment(reinterpret_cast<uintptr_t>(range_base), range_bytes);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ModuleRegistrar
  //

  namespace {
    typedef std::map<std::string, ModuleRegistrar::NetworkRegistrationBase *>
        NetworkModuleRegistrationMap;
    typedef std::vector<std::pair<size_t, ModuleRegistrar::NetworkRegistrationBase *>>
        NetworkModuleOrderedRegistrationList;
    NetworkModuleRegistrationMap *registered_network_modules = NULL;
    NetworkModuleOrderedRegistrationList *registered_ordered_network_modules = NULL;
  }; // namespace

#ifdef REALM_USE_DLFCN
  // accepts a colon-separated list of so files to try to load
  static int load_network_module_list(const char *sonames, RuntimeImpl *runtime,
                                      int *argc, const char ***argv,
                                      std::vector<void *> &handles,
                                      std::vector<NetworkModule *> &modules)
  {
    // null/empty strings are nops
    if(!sonames || !*sonames)
      return 0;

    int count = 0;
    const char *p1 = sonames;
    while(true) {
      // skip leading colons
      while(*p1 == ':')
        p1++;
      if(!*p1)
        break;

      const char *p2 = p1 + 1;
      while(*p2 && (*p2 != ':'))
        p2++;

      char filename[1024];
      strncpy(filename, p1, p2 - p1);
      filename[p2 - p1] = 0;

      // skip the color after the filename (if it exists)
      p1 = p2 + (*p2 ? 1 : 0);

      // no leftover errors from anybody else please...
      assert(dlerror() == 0);

      // open so file, resolving all symbols but not polluting global namespace
      void *handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
      if(handle == 0) {
        std::cerr << "ERROR: could not load " << filename << ": " << dlerror() << "\n";
        continue;
      }

      {
        // this file should have a "realm_module_version" symbol
        void *sym = dlsym(handle, "realm_module_version");
        if(!sym) {
          std::cerr << "ERROR: symbol 'realm_module_version' not found in '" << filename
                    << "'\n";
          dlclose(handle);
          continue;
        }
        const char *module_version = static_cast<const char *>(sym);

        // a module version mismatch can lead to crashes/hangs/etc.
        if(strcmp(REALM_VERSION, module_version)) {
          const char *e = getenv("REALM_PERMIT_MODULE_VERSION_MISMATCH");
          if(e && (atoi(e) > 0)) {
            std::cerr << "WARNING: module version mismatch in '" << filename
                      << "': realm='" << REALM_VERSION << "' module='" << module_version
                      << "'\n";
          } else {
            std::cerr << "ERROR: module version mismatch in '" << filename << "': realm='"
                      << REALM_VERSION << "' module='" << module_version
                      << "' - set REALM_PERMIT_MODULE_VERSION_MISMATCH to load anyway\n";
            dlclose(handle);
            continue;
          }
        }
      }

      // this file should also have a "create_realm_network_module" symbol
      void *sym = dlsym(handle, "create_realm_network_module");
      if(!sym) {
        std::cerr << "ERROR: symbol 'create_realm_network_module' not found in '"
                  << filename << "'\n";
        dlclose(handle);
        continue;
      }

      // TODO: hold onto the handle even if it doesn't create a module?
      handles.push_back(handle);

      NetworkModule *m = ((NetworkModule * (*)(RuntimeImpl *, int *, const char ***))
                              sym)(runtime, argc, argv);
      if(m) {
        modules.push_back(m);
        Network::single_network = m;
        count++;
#ifndef REALM_USE_MULTIPLE_NETWORKS
        break; // Found a network module that works, no need to load the rest
#endif
      }
    }

    return count;
  }
#endif

  // called by the runtime during init - these may change the command line!
  void ModuleRegistrar::create_network_modules(std::vector<NetworkModule *> &modules,
                                               int *argc, const char ***argv)
  {
    // iterate over the network module list, trying to create each module
    // if need_loopback == false, it means a network module has been created
    bool need_loopback = true;

    // this is for -ll:networks none, and we do not enable any network, but use
    // LoopbackNetworkModule
    bool disable_network = false;

    // TODO: Check for the argument, if it exists, tokenize it and load each module
    // else, load each module and pick the first one that works
    if(registered_network_modules != NULL) {
      CommandLineParser cp;
      std::vector<std::string> network_list;
      cp.add_option_stringlist("-ll:networks", network_list);
      if(!cp.parse_command_line(*argc, *argv)) {
        std::cerr << "Unable to parse network command line" << std::endl;
        abort();
      }
      for(const std::string &name : network_list) {

        // if -ll:networks is none, do not enable any networks
        if(name == "none") {
          // make sure none is not passed with other values
          if(network_list.size() != 1) {
            std::cerr << "Cannot specify both 'none' and another value in -ll:networks"
                      << std::endl;
            abort();
          }
          disable_network = true;
          break;
        }
        NetworkModuleRegistrationMap::const_iterator it =
            registered_network_modules->find(name);
        if(it == registered_network_modules->end()) {
          std::cerr << "Unable to find specified registered network module '" << name
                    << '\'' << std::endl;
          abort();
        }
        NetworkModule *m = it->second->create_network_module(runtime, argc, argv);
        if(m == NULL) {
          std::cerr << "Unable to create specified registered network module '" << name
                    << '\'' << std::endl;
          abort();
        }
        modules.push_back(m);
        Network::single_network = m;
        need_loopback = false;
#ifndef REALM_USE_MULTIPLE_NETWORKS
        break; // Found one network backend that works, no need to create the rest
#endif
      }
    }

    if(need_loopback && !disable_network) {
      const char *e = getenv("REALM_DYNAMIC_NETWORK_MODULES");
      if(e) {
#ifdef REALM_USE_DLFCN
        if(!check_symbol_visibility()) {
          // no loggers yet - use stderr
          std::cerr << "FATAL: symbols for Realm internal API are not visible - dynamic "
                       "modules will not work";
          abort();
        }

        int count = load_network_module_list(e, runtime, argc, argv,
                                             network_sofile_handles, modules);
        if(count > 0)
          need_loopback = false;
#else
        // no loggers yet - use stderr
        std::cerr << "FATAL: loading of dynamic Realm modules requested, but "
                     "REALM_USE_DLFCN=0!";
        abort();
#endif
      }
    }

    // networks were not specified on command-line,
    // so fallback to loading all the modules and
    // picking the first one
    if(need_loopback && !disable_network &&
       (registered_ordered_network_modules != NULL)) {
      for(const NetworkModuleOrderedRegistrationList::value_type &v :
          *registered_ordered_network_modules) {
        NetworkModule *m = v.second->create_network_module(runtime, argc, argv);
        if(m) {
          modules.push_back(m);
          Network::single_network = m;
          need_loopback = false;
#ifndef REALM_USE_MULTIPLE_NETWORKS
          break; // Found one network backend that works, no need to create the rest
#endif
        }
      }
    }

    if(need_loopback) {
      NetworkModule *m =
          LoopbackNetworkModule::create_network_module(runtime, argc, argv);
      assert(m != 0);
      modules.push_back(m);
      assert(Network::single_network == 0);
      Network::single_network = m;
    }
  }

  /*static*/ void ModuleRegistrar::add_network_registration(NetworkRegistrationBase *reg,
                                                            const std::string &name,
                                                            size_t order /*= 9999*/)
  {
    // Enforce a constructor order for the global network registration
    // map by defining it static here and capture a global pointer to it.
    static NetworkModuleRegistrationMap registered_network_module_map;
    static NetworkModuleOrderedRegistrationList ordered_network_modules;
    registered_network_modules = &registered_network_module_map;
    registered_ordered_network_modules = &ordered_network_modules;

    if(!registered_network_modules->insert(std::make_pair(name, reg)).second) {
      std::cerr << "Failed to register network module " << name << std::endl;
      abort();
    }
    // Insert in order based on the order given
    NetworkModuleOrderedRegistrationList::value_type p = std::make_pair(order, reg);
    ordered_network_modules.insert(std::upper_bound(ordered_network_modules.begin(),
                                                    ordered_network_modules.end(), p),
                                   p);
  }
}; // namespace Realm
