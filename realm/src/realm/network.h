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

#ifndef REALM_NETWORK_H
#define REALM_NETWORK_H

#include "realm/realm_config.h"
#include "realm/module.h"
#include "realm/nodeset.h"
#include "realm/memory.h"
#include "realm/bytearray.h"

#include <map>

namespace Realm {

  // NodeID defined in nodeset.h

  class NetworkModule;
  class MemoryImpl;
  class IBMemory;
  class ByteArray;
  class ActiveMessageImpl;
  class IncomingMessageManager;
  class NetworkSegment;

  // a RemoteAddress is used to name the target of an RDMA operation - in some
  //  cases it's as simple as a pointer, but in others additional info is needed
  //  (hopefully we won't need more than 16B anywhere though)
  struct RemoteAddress {
    union {
      struct {
        uintptr_t ptr;
        uintptr_t extra;
      };
      unsigned char raw_bytes[384];
    };
  };

  // a LocalAddress is used to name the local source of an RDMA write or target
  //  of an RDMA read
  struct LocalAddress {
    const NetworkSegment *segment;
    uintptr_t offset;
  };

  namespace Network {
    // a few globals for efficiency
    extern NodeID my_node_id;
    extern NodeID max_node_id;
    extern NodeSet all_peers;
    // all peers that can access shared memory from this node
    // NOTE: This is an over-estimation.  Users should be robust to the fact that this may
    //       include peers that are not able to access shared memory.
    extern NodeSet shared_peers;

    // in most cases, there will be a single network module - if so, we set
    //  this so we don't have to do a per-node lookup
    extern NetworkModule *single_network;

    // gets the network for a given node
    NetworkModule *get_network(NodeID node);

    // and a few "global" operations that abstract over any/all networks
    void barrier(void);

    // result of a single quiescence-check round.  The runtime's shutdown loop
    //  calls Network::check_for_quiescence repeatedly and stops when it sees
    //  DONE.  We deliberately do not have a "stuck" status: without
    //  introspection into the network layer there is no way to tell a slow
    //  in-flight message from a lost one, so any timeout-based abort would
    //  produce spurious failures.  If the system is genuinely hung, the
    //  shutdown loop hangs (which is debuggable - attach gdb).  A periodic
    //  warning is emitted while counters are frozen, but the function never
    //  fails the run.
    enum class QuiescenceStatus
    {
      // confirmed quiescent: two consecutive rounds agreed on the global state
      //  AND the state is "no in-flight messages, no queued work, no pending
      //  completions".  The runtime can safely proceed with detach.
      DONE,
      // not yet quiescent - counters may or may not have changed this round,
      //  but the system is not in a confirmed-quiet state.  Caller should call
      //  again.
      PROGRESSING,
    };

    // a quiescence check across all nodes - returns DONE when the global system
    //  is confirmed quiescent, otherwise PROGRESSING.  The implementation is a
    //  Mattern's-shaped two-round stability check; correctness requires that
    //  the application has already promised it will not initiate new work
    //  (e.g., post-precondition during shutdown).
    QuiescenceStatus check_for_quiescence(IncomingMessageManager *message_manager);

    // collective communication across all nodes (TODO: subcommunicators?)
    template <typename T>
    T broadcast(NodeID root, T val);

    template <typename T>
    void gather(NodeID root, T val, std::vector<T> &result);
    template <typename T>
    void gather(NodeID root, T val); // for non-root participants

    // untyped versions
    void broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes);
    void gather(NodeID root, const void *val_in, void *vals_out, size_t bytes);

    // for sending active messages
    ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const RemoteAddress &dest_payload_addr, void *storage_base, size_t storage_size);

    ActiveMessageImpl *create_active_message_impl(
        const NodeSet &targets, unsigned short msgid, size_t header_size,
        size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, void *storage_base, size_t storage_size);

    size_t recommended_max_payload(NodeID target, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(NodeID target, const RemoteAddress &dest_payload_addr,
                                   bool with_congestion, size_t header_size);
    size_t recommended_max_payload(NodeID target, const void *data, size_t bytes_per_line,
                                   size_t lines, size_t line_stride, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(const NodeSet &targets, const void *data,
                                   size_t bytes_per_line, size_t lines,
                                   size_t line_stride, bool with_congestion,
                                   size_t header_size);
    size_t recommended_max_payload(NodeID target, const LocalAddress &src_payload_addr,
                                   size_t bytes_per_line, size_t lines,
                                   size_t line_stride,
                                   const RemoteAddress &dest_payload_addr,
                                   bool with_congestion, size_t header_size);

    // returns the strict upper bound on payload size for a single active
    //  message - payloads larger than this CANNOT be sent without
    //  fragmentation
    // if src_payload_addr is non-null, the limit may be higher because the
    //  backend can use the caller's buffer directly (e.g. UCX rendezvous);
    //  if null, the network must allocate the buffer internally
    // network backends that handle fragmentation internally (e.g. UCX with
    //  caller-provided buffers) may return SIZE_MAX to indicate no
    //  practical limit
    // NOTE: sending payloads up to this limit is safe but may not be
    //  optimal - use recommended_max_payload() to query the size that
    //  avoids performance penalties such as protocol downgrades, extra
    //  copies, or increased latency
    size_t max_payload_size(size_t header_size, const void *src_payload_addr);
  }; // namespace Network

  // a network module provides additional functionality on top of a normal Realm
  //  module
  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NetworkModule : public Module {
  protected:
    NetworkModule(const std::string &_name);

  public:
    // all subclasses should define this (static) method - its responsibilities
    // are:
    // 1) determine if the network module should even be loaded
    // 2) fix the command line if the spawning system hijacked it
    // static NetworkModule *create_network_module(RuntimeImpl *runtime,
    //                                            int *argc, const char ***argv);

    // Enumerates all the peers that the current node could potentially share memory with
    virtual void get_shared_peers(NodeSet &shared_peers) = 0;

    // actual parsing of the command line should wait until here if at all
    //  possible
    virtual void parse_command_line(RuntimeImpl *runtime,
                                    std::vector<std::string> &cmdline);

    // "attaches" to the network, if that is meaningful - attempts to
    //  bind/register/(pick your network-specific verb) the requested memory
    //  segments with the network
    virtual void attach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) = 0;

    // detaches from the network
    virtual void detach(RuntimeImpl *runtime,
                        std::vector<NetworkSegment *> &segments) = 0;

    // collective communication within this network
    virtual void barrier(void) = 0;
    virtual void broadcast(NodeID root, const void *val_in, void *val_out,
                           size_t bytes) = 0;
    virtual void gather(NodeID root, const void *val_in, void *vals_out,
                        size_t bytes) = 0;
    virtual void allgatherv(const char *val_in, size_t bytes, std::vector<char> &vals_out,
                            std::vector<size_t> &lengths) = 0;

    // Per-rank state used by the quiescence-detection algorithm in
    //  Network::check_for_quiescence.  Each backend exposes its current local
    //  state, the algorithm sums across ranks, and termination is declared
    //  when two consecutive rounds agree on a quiet state.
    //
    // Correctness depends on every source of "future messages from this rank"
    //  being captured by either:
    //   - queued_items > 0 (anything queued that could spontaneously
    //     produce a send when it runs), or
    //   - packets_reserved exceeding packets_received globally (anything
    //     in flight on the wire), or
    //   - pending_completions > 0 (any local-completion bookkeeping that
    //     could fire and produce comp replies)
    //
    // MONOTONICITY INVARIANT: the two-round stability check requires that
    //  any real activity between rounds shows up as a counter change.  Two
    //  of the five reduced fields below are snapshot counters (rises AND
    //  falls): queued_items and pending_completions.  Snapshot counters
    //  can alias - the same value across rounds while items flowed
    //  through.  Each one is therefore paired with a monotonic field in
    //  the same allreduce array, so the joint stability of the pair rules
    //  out aliased activity:
    //    queued_items       <- paired with -> events_added (monotonic)
    //    pending_completions <- paired with -> packets_reserved (monotonic)
    //  The monotonic fields packets_reserved, packets_received, and
    //  events_added are themselves never decremented (paths that would
    //  decrement them - e.g., cancel_pbuf in GASNet-EX - are stubbed to
    //  abort).  See per-field comments for the pairing argument.
    struct QuiescenceState {
      // total number of items currently queued on this rank that could
      //  produce a future network operation: work-item queues (injector,
      //  completer, poller, rgetter), pending in-flight requests, etc.
      //  This is a SNAPSHOT (rises and falls); for stability detection it
      //  must be combined with events_added (monotonic) so that a queue
      //  whose count happens to stay the same across rounds while items
      //  flow through still registers as activity.  Reduced via SUM.
      uint64_t queued_items;
      // Monotonic count of items ever added to any queue on this rank;
      //  the monotonic mate of queued_items.  REQUIRED for stability
      //  detection: queued_items alone can be the same value across two
      //  consecutive rounds while the queue contents change (e.g., a
      //  put-completion event firing pops 1 from pending_events and adds
      //  1 to ready_xpairs - net change in queued_items is zero, but
      //  real work happened).  events_added goes up whenever any queue
      //  gains an entry, so concurrent add+remove activity that nets to
      //  zero in queued_items still shows as a change in events_added.
      //  Together with queued_items (snapshot) this fully captures
      //  queue-level activity: pure pops show up as queued_items
      //  decreasing, pure adds show up in both, and balanced add+pop
      //  shows up in events_added.  Reduced via SUM.
      uint64_t events_added;
      // Monotonic cumulative count of network messages this rank has
      //  originated.  At quiescence, sum across ranks must equal sum of
      //  packets_received.  Also serves as the monotonic mate of
      //  pending_completions (see below): every alloc of a
      //  PendingCompletion is co-located with a packets_reserved++ at the
      //  same send site, so balanced alloc+recycle activity that leaves
      //  pending_completions unchanged still shows up here.
      uint64_t packets_reserved;
      // Monotonic cumulative count of network messages this rank has
      //  received and counted - sampled post-drain so the count reflects
      //  every received message that has been dispatched to handlers as
      //  of the call.
      uint64_t packets_received;
      // Snapshot count of pending remote completions on this rank waiting
      //  for replies to arrive; rises on PendingCompletion alloc, falls
      //  on recycle (comp_reply receipt or local-completion firing).
      //  Reduced via SUM.  Although non-monotonic, the stability check is
      //  aliasing-safe because every alloc is paired with packets_reserved++
      //  at the same send site: between two rounds, if N allocs and M
      //  recycles happened then Δpending = N-M and Δreserved = N.  Joint
      //  stability of pending_completions AND packets_reserved forces
      //  N = M = 0, ruling out balanced alloc+recycle activity that
      //  pending_completions alone would miss.
      uint64_t pending_completions;
      // local-only (NOT reduced) count of messages this rank has received
      //  that pass through IncomingMessageManager and need to be drained
      //  before the post-drain sample is taken.  Must be a subset of
      //  packets_received: any wire packet that bypasses IMM (e.g., UCX
      //  remote-completion replies handled directly, GASNet-EX
      //  comp_reply/rget control AMs that are processed inline) MUST be
      //  excluded, otherwise drain_incoming_messages waits forever for
      //  total_messages_handled to reach a target it cannot.  For backends
      //  where every received packet goes through IMM (GASNet-EX), this
      //  equals packets_received.  MPI has a separate subset counter because
      //  completion replies bypass IMM.
      uint64_t messages_to_drain;
    };

    // Sample the current quiescence state of this network.  Called after the
    //  IncomingMessageManager has been drained, so 'packets_received' reflects
    //  every received message that has been dispatched as of the call.
    virtual void sample_quiescence_state(QuiescenceState &state) = 0;

    // Optional escape hatch for backends with an existing quiescence protocol
    //  that does not naturally expose the sampled counters above.  Return true
    //  after setting 'status' if the backend handled the check itself.
    virtual bool custom_quiescence_check(IncomingMessageManager *message_manager,
                                         Network::QuiescenceStatus &status);

    // Sum-reduce a small array of uint64_t across all ranks.  Used by the
    //  Mattern's-shaped Network::check_for_quiescence loop to combine
    //  per-rank QuiescenceState samples into a global tally.
    virtual void quiescence_allreduce_sum(const uint64_t *local_counts,
                                          uint64_t *total_counts, size_t count) = 0;

    // used to create a remote proxy for a memory
    virtual MemoryImpl *create_remote_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                             Memory::Kind kind,
                                             const ByteArray &rdma_info) = 0;
    virtual IBMemory *create_remote_ib_memory(RuntimeImpl *runtime, Memory m, size_t size,
                                              Memory::Kind kind,
                                              const ByteArray &rdma_info) = 0;

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size, const void *src_payload_addr,
                               size_t src_payload_lines, size_t src_payload_line_stride,
                               void *storage_base, size_t storage_size) = 0;

    virtual ActiveMessageImpl *create_active_message_impl(
        NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
        const LocalAddress &src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
        void *storage_base, size_t storage_size) = 0;

    virtual ActiveMessageImpl *
    create_active_message_impl(NodeID target, unsigned short msgid, size_t header_size,
                               size_t max_payload_size,
                               const RemoteAddress &dest_payload_addr, void *storage_base,
                               size_t storage_size) = 0;

    virtual ActiveMessageImpl *create_active_message_impl(
        const NodeSet &targets, unsigned short msgid, size_t header_size,
        size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
        size_t src_payload_line_stride, void *storage_base, size_t storage_size) = 0;

    virtual size_t recommended_max_payload(NodeID target, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(const NodeSet &targets, const void *data,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride, bool with_congestion,
                                           size_t header_size) = 0;
    virtual size_t recommended_max_payload(NodeID target,
                                           const LocalAddress &src_payload_addr,
                                           size_t bytes_per_line, size_t lines,
                                           size_t line_stride,
                                           const RemoteAddress &dest_payload_addr,
                                           bool with_congestion, size_t header_size) = 0;

    // returns the hard upper bound on payload size - see Network::max_payload_size
    //  for full documentation; callers wanting optimal performance should use
    //  recommended_max_payload() instead
    virtual size_t max_payload_size(size_t header_size, const void *src_payload_addr) = 0;
  };

  namespace NetworkSegmentInfo {
    // "enum" (using a namespace so that they can be extended in other
    //  headers) describing the different kind of memories that a network
    //  segment can live in
    typedef unsigned MemoryType;

    // each memory type gets to define what the extra data means for itself
    typedef uintptr_t MemoryTypeExtraData;

    static const MemoryType Unknown = 0;

    // generic memory that is read/write-able by the host CPUs
    static const MemoryType HostMem = 1;

    // optional flags for a network segment
    typedef unsigned FlagsType;
    struct OptionFlags {
      // registration should be performed on-demand rather than eagerly
      static const FlagsType OnDemandRegistration = 1U << 0;
    };
  }; // namespace NetworkSegmentInfo

  class REALM_INTERNAL_API_EXTERNAL_LINKAGE NetworkSegment {
  public:
    NetworkSegment();

    // normally a request will just be for a particular size
    void request(NetworkSegmentInfo::MemoryType _memtype, size_t _bytes,
                 size_t _alignment, NetworkSegmentInfo::MemoryTypeExtraData _memextra = 0,
                 NetworkSegmentInfo::FlagsType _flags = 0);

    // but it can also be for a pre-allocated chunk of memory with a fixed address
    void assign(NetworkSegmentInfo::MemoryType _memtype, void *_base, size_t _bytes,
                NetworkSegmentInfo::MemoryTypeExtraData _memextra = 0,
                NetworkSegmentInfo::FlagsType _flags = 0);

    void *base; // once this is non-null, it cannot be changed
    size_t bytes, alignment;
    NetworkSegmentInfo::MemoryType memtype;
    NetworkSegmentInfo::MemoryTypeExtraData memextra;
    NetworkSegmentInfo::FlagsType flags;

    // again, a single network puts itself here in addition to adding to the map
    NetworkModule *single_network;
    ByteArray *single_network_data;

    // a map from each of the networks that successfully bound the segment to
    //  whatever data (if any) that network needs to track the binding
    std::map<NetworkModule *, ByteArray> networks;

    void add_rdma_info(NetworkModule *network, const void *data, size_t len);
    const ByteArray *get_rdma_info(NetworkModule *network) const;

    // returns whether the segment is registered for all networks,
    //  or for a specific network
    bool is_registered() const;
    bool is_registered(NetworkModule *network) const;

    // tests whether an address range is in segment
    bool in_segment(const void *range_base, size_t range_bytes) const;
    bool in_segment(uintptr_t range_base, size_t range_bytes) const;
  };

}; // namespace Realm

#include "realm/network.inl"

#endif
