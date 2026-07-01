/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_INTEROP_IMPL_H__
#define __LEGION_INTEROP_IMPL_H__

#include "legion/api/interop.h"
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    class HandshakeImpl : public Collectable,
                          public Heapify<HandshakeImpl, LONG_LIFETIME> {
    public:
      HandshakeImpl(bool init_in_ext);
      HandshakeImpl(const HandshakeImpl& rhs) = delete;
      ~HandshakeImpl(void);
    public:
      HandshakeImpl& operator=(const HandshakeImpl& rhs) = delete;
    public:
      void initialize(void);
    public:
      void ext_handoff_to_legion(void);
      void ext_wait_on_legion(void);
    public:
      void legion_handoff_to_ext(void);
      void legion_wait_on_ext(void);
    public:
      PhaseBarrier get_legion_wait_phase_barrier(void);
      PhaseBarrier get_legion_arrive_phase_barrier(void);
      void advance_legion_handshake(void);
      void record_external_handshake(Provenance* provenance);
    private:
      const bool init_in_ext;
    private:
      // Whether the legion side is in split mode execution or not
      bool split;
      ApBarrier ext_wait_barrier;
      ApBarrier ext_arrive_barrier;
      ApBarrier legion_wait_barrier;
      ApBarrier legion_next_barrier;  // one gen ahead of wait
      ApBarrier legion_arrive_barrier;
    private:
      // For profiling
      std::optional<long long> previous_external_time;
      static std::atomic<Provenance*> external_wait;
      static std::atomic<Provenance*> external_handoff;
      static constexpr std::string_view EXTERNAL_WAIT =
          "External Legion Handshake Wait on Legion";
      static constexpr std::string_view EXTERNAL_HANDOFF =
          "External Legion Handshake Handoff to Legion";
    };

    class MPIRankTable {
    public:
      MPIRankTable(
          int radix, AddressSpaceID address_space, size_t total_address_spaces);
      MPIRankTable(const MPIRankTable& rhs) = delete;
      ~MPIRankTable(void);
    public:
      MPIRankTable& operator=(const MPIRankTable& rhs) = delete;
    public:
      void perform_rank_exchange(void);
      void handle_mpi_rank_exchange(Deserializer& derez);
    protected:
      bool initiate_exchange(void);
      void send_remainder_stage(void);
      bool send_ready_stages(const int start_stage = 1);
      void unpack_exchange(int stage, Deserializer& derez);
      void complete_exchange(void);
    public:
      bool participating;
    public:
      std::map<int, AddressSpace> forward_mapping;
      std::map<AddressSpace, int> reverse_mapping;
    protected:
      mutable LocalLock reservation;
      RtUserEvent done_event;
      std::vector<int> stage_notifications;
      std::vector<bool> sent_stages;
    protected:
      int collective_radix;
      int collective_log_radix;
      int collective_stages;
      int collective_participating_spaces;
      int collective_last_radix;
      // Handle a small race on deciding who gets to
      // trigger the done event
      bool done_triggered;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INTEROP_IMPL_H__
