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

#include "legion/managers/message.h"
#include "legion/kernel/runtime.h"
#include "legion/managers/shard.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    MessageHeader::MessageHeader(
        MessageKind k, VirtualChannelKind vc, bool escape_ctx, bool escape_op)
      : LgTaskArgs<MessageHeader>(escape_ctx, escape_op), kind(k), channel(vc),
        sender(runtime->address_space)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ void MessageHeader::handle(const void* data, size_t size)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(data, size);
      MessageHeader header;
      derez.deserialize(header);
      legion_assert(header.lg_task_id == TASK_ID);
#ifdef LEGION_DEBUG_CALLERS
      implicit_task_caller = header.lg_call_id;
#endif
      implicit_operation = nullptr;
      implicit_enclosing_context = header.enclosing_context;
      implicit_provenance = header.provenance;
      implicit_unique_op_id = header.unique_op_id;
      // Need to do all this stuff before invoking the handler because if
      // this is a handler for a shutdown message then we could race with
      // the runtime shutdown
      MessageManager* manager = runtime->find_messenger(header.sender);
      VirtualChannel& channel = manager->find_channel(header.channel);
      // See if there is any profiling work to do
      if (channel.profile_outgoing_messages)
      {
        const LgEvent original_fevent = runtime->profiler->find_message_fevent(
            implicit_fevent, false /*remove*/);
        if (channel.ordered_channel &&
            (header.channel != PROFILING_VIRTUAL_CHANNEL))
          implicit_profiler->record_event_trigger(
              original_fevent, implicit_fevent);
      }
      // Find the handler for this message
      legion_assert(header.kind < LAST_SEND_KIND);
      void (*handler)(Deserializer&, AddressSpaceID) =
          MessageManager::message_handler_table[header.kind];
      legion_assert(handler != nullptr);
      (*handler)(derez, header.sender);
      // If this is not a shutdown message then record that we've
      // finished receiving the message
      if ((header.kind != SEND_SHUTDOWN_NOTIFICATION) &&
          (header.kind != SEND_SHUTDOWN_RESPONSE))
        channel.record_received();
    }

    /////////////////////////////////////////////////////////////
    // Virtual Channel
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(
        VirtualChannelKind kind, AddressSpaceID local_address_space,
        size_t max_message_size, bool profile_outgoing)
      : ordered_channel(
            (kind != DEFAULT_VIRTUAL_CHANNEL) &&
            (kind != THROUGHPUT_VIRTUAL_CHANNEL)),
        profile_outgoing_messages(profile_outgoing),
        request_priority(
            (kind == THROUGHPUT_VIRTUAL_CHANNEL) ?
                LG_THROUGHPUT_MESSAGE_PRIORITY :
            (kind == UPDATE_VIRTUAL_CHANNEL) ? LG_LATENCY_DEFERRED_PRIORITY :
                                               LG_LATENCY_MESSAGE_PRIORITY),
        response_priority(
            (kind == THROUGHPUT_VIRTUAL_CHANNEL) ?
                LG_THROUGHPUT_RESPONSE_PRIORITY :
            (kind == UPDATE_VIRTUAL_CHANNEL) ? LG_LATENCY_MESSAGE_PRIORITY :
                                               LG_LATENCY_RESPONSE_PRIORITY)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    VirtualChannel::~VirtualChannel(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void VirtualChannel::send_message(
        MessageKind kind, const Serializer& rez, RtEvent send_precondition,
        Processor target, bool response)
    //--------------------------------------------------------------------------
    {
      // Need an exclusive lock to update internal data structures
      AutoLock c_lock(channel_lock);
      // If this is not a shutdown message record that we're sending it
      if ((kind != SEND_SHUTDOWN_NOTIFICATION) &&
          (kind != SEND_SHUTDOWN_RESPONSE))
        sent_messages++;
      const RtEvent precondition =
          ordered_channel ? (send_precondition.exists() ?
                                 Runtime::merge_events(
                                     send_precondition, last_message_event) :
                                 last_message_event) :
                            send_precondition;
      Realm::ProfilingRequestSet requests;
      if (profile_outgoing_messages)
        LegionProfiler::add_message_request(
            requests, kind, target, precondition);
      // Send the message directly there, don't go through the
      // runtime interface to avoid being counted, still include
      // a profiling request though if necessary in order to
      // see waits on message handlers
      const RtEvent message_done(target.spawn(
#ifdef LEGION_SEPARATE_META_TASKS
          LG_TASK_ID + LG_MESSAGE_ID + kind,
#else
          LG_TASK_ID,
#endif
          rez.get_buffer(), rez.get_used_bytes(), requests, precondition,
          response ? response_priority : request_priority));
      if (!ordered_channel)
      {
        while (!unordered_events.empty())
        {
          if (!unordered_events.front().has_triggered())
            break;
          unordered_events.pop_front();
        }
        unordered_events.push_back(message_done);
      }
      else
        last_message_event = message_done;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::record_received(void)
    //--------------------------------------------------------------------------
    {
      // Atomic!
      received_messages++;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::confirm_shutdown(ShutdownManager* shutdown_manager)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(channel_lock);
      // Record our total message counts and any events we might need
      // to end up waiting for to ensure messages are done
      shutdown_manager->record_message_counts(
          sent_messages, received_messages.load());
      if (!ordered_channel)
      {
        while (!unordered_events.empty())
        {
          if (!unordered_events.front().has_triggered())
            break;
          unordered_events.pop_front();
        }
        for (RtEvent pending : unordered_events)
          shutdown_manager->record_pending_message(pending);
      }
      else
        shutdown_manager->record_pending_message(last_message_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ MessageManager* MessageManager::find_manager(
        AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      return runtime->find_messenger(target);
    }

    /////////////////////////////////////////////////////////////
    // Message Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(
        AddressSpaceID remote, size_t max_message_size,
        const Processor remote_util_group)
      : channels((VirtualChannel*)malloc(
            MAX_NUM_VIRTUAL_CHANNELS * sizeof(VirtualChannel))),
        remote_address_space(remote), target(remote_util_group)
    //--------------------------------------------------------------------------
    {
      legion_assert(remote != runtime->address_space);
      const bool has_profiler = (runtime->profiler != nullptr);
      // Initialize our virtual channels
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        new (channels + idx) VirtualChannel(
            (VirtualChannelKind)idx, runtime->address_space, max_message_size,
            has_profiler);
      }
    }

    //--------------------------------------------------------------------------
    MessageManager::~MessageManager(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].~VirtualChannel();
      free(channels);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_message(
        MessageKind kind, VirtualChannelKind channel, const Serializer& rez,
        bool response, RtEvent flush_precondition)
    //--------------------------------------------------------------------------
    {
      channels[channel].send_message(
          kind, rez, flush_precondition, target, response);
    }

    //--------------------------------------------------------------------------
    VirtualChannel& MessageManager::find_channel(VirtualChannelKind vc)
    //--------------------------------------------------------------------------
    {
      legion_assert(vc < MAX_NUM_VIRTUAL_CHANNELS);
      return channels[vc];
    }

    //--------------------------------------------------------------------------
    void MessageManager::confirm_shutdown(ShutdownManager* shutdown_manager)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].confirm_shutdown(shutdown_manager);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MessageManager::register_handlers(void)
    //--------------------------------------------------------------------------
    {
#define REGISTER_HANDLERS(kind, type, name, response, escape_ctx, escape_op) \
  legion_assert(message_handler_table[kind] == nullptr);                     \
  message_handler_table[kind] = type::handle;
      LEGION_ACTIVE_MESSAGES(REGISTER_HANDLERS)
#undef REGISTER_HANDLERS
#define REGISTER_SHARD_COLLECTIVES(kind, name)           \
  legion_assert(message_handler_table[kind] == nullptr); \
  message_handler_table[kind] = ShardCollectiveMessage::handle;
      LEGION_SHARD_COLLECTIVE_ACTIVE_MESSAGES(REGISTER_SHARD_COLLECTIVES)
#undef REGISTER_SHARD_COLLECTIVES
    }

  }  // namespace Internal
}  // namespace Legion
