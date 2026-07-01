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

#include "legion/kernel/runtime.h"
#include "legion/api/interop_impl.h"
#include "legion/api/sync.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // LegionHandshake
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  LegionHandshake::LegionHandshake(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  LegionHandshake::LegionHandshake(const LegionHandshake& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  LegionHandshake::~LegionHandshake(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  LegionHandshake::LegionHandshake(Internal::LegionHandshakeImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  LegionHandshake& LegionHandshake::operator=(const LegionHandshake& rhs)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
    }
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  void LegionHandshake::ext_handoff_to_legion(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->ext_handoff_to_legion();
  }

  //--------------------------------------------------------------------------
  void LegionHandshake::ext_wait_on_legion(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->ext_wait_on_legion();
  }

  //--------------------------------------------------------------------------
  void LegionHandshake::legion_handoff_to_ext(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->legion_handoff_to_ext();
  }

  //--------------------------------------------------------------------------
  void LegionHandshake::legion_wait_on_ext(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->legion_wait_on_ext();
  }

  //--------------------------------------------------------------------------
  PhaseBarrier LegionHandshake::get_legion_wait_phase_barrier(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->get_legion_wait_phase_barrier();
  }

  //--------------------------------------------------------------------------
  PhaseBarrier LegionHandshake::get_legion_arrive_phase_barrier(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->get_legion_arrive_phase_barrier();
  }

  //--------------------------------------------------------------------------
  void LegionHandshake::advance_legion_handshake(void) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->advance_legion_handshake();
  }

  /////////////////////////////////////////////////////////////
  // MPILegionHandshake
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  MPILegionHandshake::MPILegionHandshake(void) : LegionHandshake()
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  MPILegionHandshake::MPILegionHandshake(const MPILegionHandshake& rhs)
    : LegionHandshake(rhs)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  MPILegionHandshake::~MPILegionHandshake(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  MPILegionHandshake::MPILegionHandshake(Internal::LegionHandshakeImpl* i)
    : LegionHandshake(i)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  MPILegionHandshake& MPILegionHandshake::operator=(
      const MPILegionHandshake& rhs)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      if (impl->remove_reference())
        delete impl;
    }
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Handshake Impl
    /////////////////////////////////////////////////////////////

    /*static*/ std::atomic<Provenance*> LegionHandshakeImpl::external_wait =
        nullptr;
    /*static*/ std::atomic<Provenance*> LegionHandshakeImpl::external_handoff =
        nullptr;

    //--------------------------------------------------------------------------
    HandshakeImpl::HandshakeImpl(bool init_ext)
      : init_in_ext(init_ext), split(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    HandshakeImpl::~HandshakeImpl(void)
    //--------------------------------------------------------------------------
    {
      ext_wait_barrier.destroy_barrier();
      legion_next_barrier.destroy_barrier();
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::initialize(void)
    //--------------------------------------------------------------------------
    {
      ext_wait_barrier = runtime->create_ap_barrier(1);
      legion_wait_barrier = runtime->create_ap_barrier(1);
      ext_arrive_barrier = legion_wait_barrier;
      legion_arrive_barrier = ext_wait_barrier;
      // Legion runs split-phase on its side so we need to advance its
      // wait barrier so that we can always refer to the previous phase
      legion_next_barrier = legion_wait_barrier;
      Runtime::advance_barrier(legion_next_barrier);
      // If control is starting on the Legion side then make it seems like
      // the previous phase of the legion_wait barrier has already triggered
      if (!init_in_ext)
      {
        // Trigger the first generation of the ext arrival so that
        // the legion_wait_barrier always points to a valid generation
        // This makes it look like we always start the cycle from the
        // external side which is how we know that the barriers from
        // the external to the legion side will be the first ones to
        // exhaust their generations and we know we need to generate
        // new barriers for both sides.
        // Same trick as below for the profiler to tell it this is an
        // external handshake
        const LgEvent previous_fevent = implicit_fevent;
        implicit_fevent = ext_arrive_barrier;
        runtime->phase_barrier_arrive(ext_arrive_barrier, 1);
        implicit_fevent = previous_fevent;
        Runtime::advance_barrier(ext_arrive_barrier);
      }
      if (runtime->profiler != nullptr)
      {
        if (external_wait.load() == nullptr)
          external_wait.store(runtime->find_or_create_provenance(
              EXTERNAL_WAIT.data(), EXTERNAL_WAIT.size()));
        if (external_handoff.load() == nullptr)
          external_handoff.store(runtime->find_or_create_provenance(
              EXTERNAL_HANDOFF.data(), EXTERNAL_HANDOFF.size()));
      }
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::ext_handoff_to_legion(void)
    //--------------------------------------------------------------------------
    {
      if (Processor::get_executing_processor().exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to perform an external handshake hand-off "
              << "to Legion while on a Realm processor. All external calls "
              << "must be done from external threads.";
        error.raise();
      }
      if (implicit_context != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Detected an illegal handshake calling "
              << "'ext_handoff_to_legion' from inside of a Legion task.";
        error.raise();
      }
      if (runtime->profiler != nullptr)
      {
        if (implicit_profiler == nullptr)
          runtime->profiler->instantiate_profiling_instance();
        if (!previous_external_time)
          previous_external_time = Realm::Clock::current_time_in_nanoseconds();
      }
      // We need to detect the case where we are about to trigger the last
      // external barrier generation and update the legion side with new
      // barriers before we do that
      ApBarrier to_arrive = ext_arrive_barrier;
      Runtime::advance_barrier(ext_arrive_barrier);
      if (!ext_arrive_barrier.exists())
      {
        legion_assert(!ext_wait_barrier.exists());
        legion_assert(!legion_next_barrier.exists());
        legion_assert(!legion_arrive_barrier.exists());
        ext_wait_barrier = runtime->create_ap_barrier(1);
        legion_next_barrier = runtime->create_ap_barrier(1);
        ext_arrive_barrier = legion_next_barrier;
        legion_arrive_barrier = ext_wait_barrier;
      }
      runtime->phase_barrier_arrive(to_arrive, 1);
      if (implicit_profiler != nullptr)
        record_external_handshake(external_handoff.load());
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::ext_wait_on_legion(void)
    //--------------------------------------------------------------------------
    {
      if (Processor::get_executing_processor().exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to perform an external handshake wait on Legion "
              << "while on a Realm processor. All external calls must be done "
                 "from "
              << "external threads.";
        error.raise();
      }
      if (implicit_context != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Detected an illegal handshake calling 'ext_wait_on_legion' "
              << "from inside of a Legion task.";
        error.raise();
      }
      // Wait for ext to be ready to run
      // Note we use the external wait to be sure
      // we don't get drafted by the Realm runtime
      ext_wait_barrier.external_wait();
      // Now we can advance our wait barrier
      Runtime::advance_barrier(ext_wait_barrier);
      if (implicit_profiler != nullptr)
        record_external_handshake(external_wait.load());
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::record_external_handshake(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const long long next_external_time =
          Realm::Clock::current_time_in_nanoseconds();
      implicit_profiler->record_application_range(
          provenance->pid, *previous_external_time, next_external_time);
      previous_external_time = next_external_time;
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::legion_handoff_to_ext(void)
    //--------------------------------------------------------------------------
    {
      if (!implicit_fevent.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Detected an illegal handshake calling "
              << "'legion_handoff_to_ext' hile not inside of a Legion task.";
        error.raise();
      }
      if (split)
      {
        Runtime::advance_barrier(legion_arrive_barrier);
        split = false;
      }
      // Always advance this barrier before doing the arrival to avoid a
      // race when we run out of barrier generations
      ApBarrier to_arrive = legion_arrive_barrier;
      Runtime::advance_barrier(legion_arrive_barrier);
      runtime->phase_barrier_arrive(to_arrive, 1);
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::legion_wait_on_ext(void)
    //--------------------------------------------------------------------------
    {
      if (!implicit_fevent.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Detected an illegal handshake calling 'legion_wait_on_ext' "
              << "while not inside of a Legion task.";
        error.raise();
      }
      // Wait for Legion to be ready to run
      // No need to avoid being drafted by the
      // Realm runtime here
      legion_wait_barrier.wait_faultignorant();
      // Now we can advance our wait barrier
      legion_wait_barrier = legion_next_barrier;
      Runtime::advance_barrier(legion_next_barrier);
      // Check to see if we're out of generations and need to wait for the
      // external side to catch up and give us a new barrier
      if (!legion_next_barrier.exists())
        legion_wait_barrier.wait_faultignorant();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier HandshakeImpl::get_legion_wait_phase_barrier(void)
    //--------------------------------------------------------------------------
    {
      // A bit non-intuitive but return the next barrier because this is
      // going to be passed into the launcher's phase barrier which will
      // do the work of getting the previous phase for us
      return legion_next_barrier;
    }

    //--------------------------------------------------------------------------
    PhaseBarrier HandshakeImpl::get_legion_arrive_phase_barrier(void)
    //--------------------------------------------------------------------------
    {
      return legion_arrive_barrier;
    }

    //--------------------------------------------------------------------------
    void HandshakeImpl::advance_legion_handshake(void)
    //--------------------------------------------------------------------------
    {
      if (!implicit_fevent.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Detected an illegal handshake calling "
            << "'advance_legion_handshake ' while not inside of a Legion task.";
        error.raise();
      }
      legion_wait_barrier = legion_next_barrier;
      Runtime::advance_barrier(legion_next_barrier);
      if (split)  // already in split mode execution
        Runtime::advance_barrier(legion_arrive_barrier);
      else  // not in split mode execution yet
        split = true;
      // Check to see if we're out of generations and need to wait for the
      // external side to catch up and give us a new barrier
      if (!legion_next_barrier.exists())
        legion_wait_barrier.wait_faultignorant();
    }

    /////////////////////////////////////////////////////////////
    // MPI Rank Table
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPIRankTable::MPIRankTable(
        int radix, AddressSpaceID address_space, size_t total_address_spaces)
      : collective_radix(radix), done_triggered(false)
    //--------------------------------------------------------------------------
    {
      if (total_address_spaces > 1)
      {
        configure_collective_settings(
            total_address_spaces, address_space, collective_radix,
            collective_log_radix, collective_stages,
            collective_participating_spaces, collective_last_radix);
        participating = (int(address_space) < collective_participating_spaces);
        // We already have our contributions for each stage so
        // we can set the inditial participants to 1
        if (participating)
        {
          sent_stages.resize(collective_stages, false);
          legion_assert(collective_stages > 0);
          stage_notifications.resize(collective_stages, 1);
          // Stage 0 always starts with 0 notifications since we'll
          // explictcly arrive on it
          stage_notifications[0] = 0;
        }
        done_event = Runtime::create_rt_user_event();
      }
      // Add ourselves to the set before any exchanges start
      legion_assert(Runtime::mpi_rank >= 0);
      forward_mapping[Runtime::mpi_rank] = address_space;
    }

    //--------------------------------------------------------------------------
    MPIRankTable::~MPIRankTable(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MPIRankTable::perform_rank_exchange(void)
    //--------------------------------------------------------------------------
    {
      // We can skip this part if there are not multiple nodes
      if (runtime->total_address_spaces > 1)
      {
        // See if we are participating node or not
        if (participating)
        {
          // We are a participating node
          // See if we are waiting for an initial notification
          // if not we can just send our message now
          if ((int(runtime->total_address_spaces) ==
               collective_participating_spaces) ||
              (runtime->address_space >= (runtime->total_address_spaces -
                                          collective_participating_spaces)))
          {
            const bool all_stages_done = initiate_exchange();
            if (all_stages_done)
              complete_exchange();
          }
        }
        else
        {
          // We are not a participating node
          // so we just have to send notification to one node
          send_remainder_stage();
        }
        // Wait for our done event to be ready
        done_event.wait();
      }
      legion_assert(forward_mapping.size() == runtime->total_address_spaces);
      // Reverse the mapping
      for (const std::pair<const int, AddressSpace>& mapping_pair :
           forward_mapping)
        reverse_mapping[mapping_pair.second] = mapping_pair.first;
    }

    //--------------------------------------------------------------------------
    bool MPIRankTable::initiate_exchange(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          participating);  // should only get this for participating shards
      {
        AutoLock r_lock(reservation);
        legion_assert(!sent_stages.empty());
        legion_assert(!sent_stages[0]);  // stage 0 shouldn't be sent yet
        legion_assert(!stage_notifications.empty());
        if (collective_stages == 1)
          legion_assert(stage_notifications[0] < collective_last_radix);
        else
          legion_assert(stage_notifications[0] < collective_radix);
        stage_notifications[0]++;
      }
      return send_ready_stages(0 /*start stage*/);
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::send_remainder_stage(void)
    //--------------------------------------------------------------------------
    {
      MPIRankExchange rez;
      {
        RezCheck z(rez);
        rez.serialize(-1);
        AutoLock r_lock(reservation, false /*exclusive*/);
        rez.serialize<size_t>(forward_mapping.size());
        for (const std::pair<const int, AddressSpace>& mapping_pair :
             forward_mapping)
        {
          rez.serialize(mapping_pair.first);
          rez.serialize(mapping_pair.second);
        }
      }
      if (participating)
      {
        // Send back to the nodes that are not participating
        AddressSpaceID target =
            runtime->address_space + collective_participating_spaces;
        legion_assert(target < runtime->total_address_spaces);
        rez.dispatch(target);
      }
      else
      {
        // Sent to a node that is participating
        AddressSpaceID target =
            runtime->address_space % collective_participating_spaces;
        rez.dispatch(target);
      }
    }

    //--------------------------------------------------------------------------
    bool MPIRankTable::send_ready_stages(const int start_stage)
    //--------------------------------------------------------------------------
    {
      legion_assert(participating);
      // Iterate through the stages and send any that are ready
      // Remember that stages have to be done in order
      for (int stage = start_stage; stage < collective_stages; stage++)
      {
        MPIRankExchange rez;
        {
          RezCheck z(rez);
          rez.serialize(stage);
          AutoLock r_lock(reservation);
          // If this stage has already been sent then we can keep going
          if (sent_stages[stage])
            continue;
          // Check to see if we're sending this stage
          // We need all the notifications from the previous stage before
          // we can send this stage
          if ((stage > 0) &&
              (stage_notifications[stage - 1] < collective_radix))
            return false;
          // If we get here then we can send the stage
          sent_stages[stage] = true;
#ifdef LEGION_DEBUG
          {
            size_t expected_size = 1;
            for (int idx = 0; idx < stage; idx++)
              expected_size *= collective_radix;
            legion_assert(expected_size <= forward_mapping.size());
          }
#endif
          rez.serialize<size_t>(forward_mapping.size());
          for (const std::pair<const int, AddressSpace>& mapping_pair :
               forward_mapping)
          {
            rez.serialize(mapping_pair.first);
            rez.serialize(mapping_pair.second);
          }
        }
        // Now we can do the send
        if (stage == (collective_stages - 1))
        {
          for (int r = 1; r < collective_last_radix; r++)
          {
            AddressSpaceID target =
                runtime->address_space ^ (r << (stage * collective_log_radix));
            legion_assert(int(target) < collective_participating_spaces);
            rez.dispatch(target);
          }
        }
        else
        {
          for (int r = 1; r < collective_radix; r++)
          {
            AddressSpaceID target =
                runtime->address_space ^ (r << (stage * collective_log_radix));
            legion_assert(int(target) < collective_participating_spaces);
            rez.dispatch(target);
          }
        }
      }
      // If we make it here, then we sent the last stage, check to see
      // if we've seen all the notifications for it
      AutoLock r_lock(reservation);
      if ((stage_notifications.back() == collective_last_radix) &&
          !done_triggered)
      {
        done_triggered = true;
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::handle_mpi_rank_exchange(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      int stage;
      derez.deserialize(stage);
      legion_assert(participating || (stage == -1));
      unpack_exchange(stage, derez);
      bool all_stages_done = false;
      if (stage == -1)
      {
        if (!participating)
          all_stages_done = true;
        else  // we can now send our stage 0
          all_stages_done = initiate_exchange();
      }
      else
        all_stages_done = send_ready_stages();
      if (all_stages_done)
        complete_exchange();
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::unpack_exchange(int stage, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_entries;
      derez.deserialize(num_entries);
      AutoLock r_lock(reservation);
      for (unsigned idx = 0; idx < num_entries; idx++)
      {
        int rank;
        derez.deserialize(rank);
        unsigned space;
        derez.deserialize(space);
        // Duplicates are possible because later messages aren't "held", but
        // they should be exact matches
        legion_assert(
            (forward_mapping.count(rank) == 0) ||
            (forward_mapping[rank] == space));
        forward_mapping[rank] = space;
      }
      if (stage >= 0)
      {
        legion_assert(stage < int(stage_notifications.size()));
        if (stage < (collective_stages - 1))
          legion_assert(stage_notifications[stage] < collective_radix);
        else
          legion_assert(stage_notifications[stage] < collective_last_radix);
        stage_notifications[stage]++;
      }
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(forward_mapping.size() == runtime->total_address_spaces);
      // See if we have to send a message back to a
      // non-participating node
      if ((int(runtime->total_address_spaces) >
           collective_participating_spaces) &&
          (int(runtime->address_space) < int(runtime->total_address_spaces -
                                             collective_participating_spaces)))
        send_remainder_stage();
      // We are done
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MPIRankExchange::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      legion_assert(runtime->mpi_rank_table != nullptr);
      runtime->mpi_rank_table->handle_mpi_rank_exchange(derez);
    }

  }  // namespace Internal
}  // namespace Legion
