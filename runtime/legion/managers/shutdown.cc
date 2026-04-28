/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/managers/shutdown.h"
#include "legion/kernel/runtime.h"
#include "legion/api/future_impl.h"
#include "legion/managers/memory.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Shutdown Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(
        ShutdownPhase p, AddressSpaceID s, unsigned r, ShutdownManager* own,
        uint64_t expected)
      : phase(p), source(s), radix(r), owner(own), expected_messages(expected),
        needed_responses(0), total_sent(0), total_received(0),
        return_code(runtime->return_code), result(true)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShutdownManager::~ShutdownManager(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool ShutdownManager::attempt_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // Do the broadcast tree to the other nodes
      // Figure out who we have to send messages to
      std::vector<AddressSpaceID> targets;
      const AddressSpaceID local_space = runtime->address_space;
      const AddressSpaceID start = local_space * radix + 1;
      for (unsigned idx = 0; idx < radix; idx++)
      {
        AddressSpaceID next = start + idx;
        if (next < runtime->total_address_spaces)
          targets.emplace_back(next);
        else
          break;
      }
      if (!targets.empty())
      {
        // Set the number of needed_responses
        needed_responses = targets.size();
        ShutdownNotification rez;
        rez.serialize(this);
        rez.serialize(phase);
        for (const AddressSpaceID& target : targets) rez.dispatch(target);
        return false;
      }
      else  // no messages means we can finalize right now
      {
        finalize();
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::handle_response(
        int code, bool success, uint64_t sent, uint64_t received, RtEvent wait)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock s_lock(shutdown_lock);
        if ((return_code == 0) && (code != 0))
          return_code = code;
        if (result && !success)
          result = false;
        total_sent += sent;
        total_received += received;
        if (wait.exists())
          wait_for.insert(wait);
        legion_assert(needed_responses > 0);
        needed_responses--;
        done = (needed_responses == 0);
      }
      if (done)
      {
        finalize();
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::finalize(void)
    //--------------------------------------------------------------------------
    {
      // Do our local check
      runtime->confirm_runtime_shutdown(this);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      if (!result)
      {
        LG_TASK_DESCRIPTIONS(task_descs);
        // Only need to see tasks less than this
        for (unsigned idx = 0; idx < LG_BEGIN_SHUTDOWN_TASK_IDS; idx++)
        {
          if (runtime->outstanding_counts[idx].load() == 0)
            continue;
          log_shutdown.info(
              "Meta-Task %s: %d outstanding", task_descs[idx],
              runtime->outstanding_counts[idx].load());
        }
      }
#endif
      if (runtime->address_space != source)
      {
        legion_assert(owner != nullptr);
        // Send the message back to the owner
        ShutdownResponse rez;
        rez.serialize(owner);
        rez.serialize(return_code);
        rez.serialize<bool>(result);
        rez.serialize(Runtime::merge_events(wait_for));
        rez.serialize(total_sent);
        rez.serialize(total_received);
        rez.dispatch(source);
      }
      else if (
          result && (total_sent == total_received) &&
          // If we're on a "CHECK" phase we just need to know that the
          // message counts are the same, if we're on a "CONFIRM" phase
          // then we need to verify that the count hasn't changed
          // otherwise it's fine to just establish that the counts are
          // the same on a "CHECK" phase
          (((phase % 2) == 1) || (total_sent == expected_messages)))
      {
        log_shutdown.info("SHUTDOWN PHASE %d SUCCESS!", phase);
        if (phase != CONFIRM_SHUTDOWN)
        {
          if (phase == CONFIRM_TERMINATION)
            runtime->prepare_runtime_shutdown();
          // Do the next phase
          runtime->initiate_runtime_shutdown(
              source, (ShutdownPhase)(phase + 1), nullptr, total_sent);
        }
        else
        {
          log_shutdown.info("SHUTDOWN SUCCEEDED!");
          std::vector<RtEvent> shutdown_events;
          Realm::ProfilingRequestSet empty_requests;
          const Processor utility_group = runtime->find_utility_group();
          shutdown_events.emplace_back(RtEvent(utility_group.spawn(
              LG_SHUTDOWN_TASK_ID, nullptr, 0, empty_requests)));
          // One last really crazy precondition on shutdown, we actually need to
          // make sure that this task itself is done executing before trying to
          // shutdown so add our own completion event as a precondition
          shutdown_events.emplace_back(
              RtEvent(Processor::get_current_finish_event()));
          // Then tell Realm to shutdown when they are all done
          RealmRuntime realm = RealmRuntime::get_runtime();
          realm.shutdown(Runtime::merge_events(shutdown_events), return_code);
        }
      }
      else
      {
        if (!result)
          log_shutdown.info()
              << "FAILED SHUTDOWN PHASE " << phase
              << " because of outstanding tasks! Trying again...";
        else if (total_sent != total_received)
          log_shutdown.info()
              << "FAILED SHUTDOWN PHASE " << phase
              << " because of mismatched message counts (sent=" << total_sent
              << ",received=" << total_received << ")! Trying again...";
        else
          log_shutdown.info() << "FAILED SHUTDOWN PHASE " << phase
                              << " because total message count " << total_sent
                              << " does not equal the expected message count "
                              << expected_messages << "! Trying again...";
        RtEvent precondition;
        if (!wait_for.empty())
          precondition = Runtime::merge_events(wait_for);
        // If we failed an even phase we go back to the one before it
        RetryShutdownArgs args(
            ((phase % 2) == 0) ? (ShutdownPhase)(phase - 1) : phase);
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::RetryShutdownArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      runtime->initiate_runtime_shutdown(runtime->address_space, phase);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownNotification::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShutdownManager* owner;
      derez.deserialize(owner);
      ShutdownManager::ShutdownPhase phase;
      derez.deserialize(phase);
      runtime->initiate_runtime_shutdown(source, phase, owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      ShutdownManager* shutdown_manager;
      derez.deserialize(shutdown_manager);
      int return_code;
      derez.deserialize(return_code);
      bool success;
      derez.deserialize(success);
      RtEvent wait;
      derez.deserialize(wait);
      uint64_t sent, received;
      derez.deserialize(sent);
      derez.deserialize(received);
      if (shutdown_manager->handle_response(
              return_code, success, sent, received, wait))
        delete shutdown_manager;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      log_shutdown.info("Outstanding tasks on node %d", runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_message_counts(
        uint64_t sent, uint64_t received)
    //--------------------------------------------------------------------------
    {
      // No need for a lock here since we're sequentially polling message
      // managers and having them respond back to us
      total_sent += sent;
      total_received += received;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_pending_message(RtEvent pending_event)
    //--------------------------------------------------------------------------
    {
      wait_for.insert(pending_event);
      log_shutdown.info("Pending message on node %d", runtime->address_space);
    }

  }  // namespace Internal
}  // namespace Legion
