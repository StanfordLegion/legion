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

#include "legion/nodes/field.h"
#include "legion/instances/physical.h"
#include "legion/nodes/region.h"
#include "legion/operations/attach.h"
#include "legion/utilities/instance_set.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Field Space Node
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(
        FieldSpace sp, RtEvent init, CollectiveMapping* map, Provenance* prov)
      : DistributedCollectable(sp.did, false /*register with runtime*/, map),
        handle(sp), provenance(prov), initialized(init),
        allocation_state(
            (map != nullptr) ? FIELD_ALLOC_COLLECTIVE :
            is_owner()       ? FIELD_ALLOC_READ_ONLY :
                               FIELD_ALLOC_INVALID),
        outstanding_allocators(0), outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        local_index_infos.resize(
            runtime->max_local_fields, std::pair<size_t, CustomSerdezID>(0, 0));
        if (collective_mapping != nullptr)
        {
          const CollectiveMapping& mapping = *collective_mapping;
          for (unsigned idx = 0; idx < mapping.size(); idx++)
          {
            const AddressSpaceID space = mapping[idx];
            if (space != local_space)
              remote_field_infos.insert(mapping[idx]);
          }
          // We can have control replication inside of just a single node
          if (remote_field_infos.empty())
            allocation_state = FIELD_ALLOC_READ_ONLY;
        }
      }
      else if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        unallocated_indexes = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Field Space %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.did);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldSpaceNode(
        FieldSpace sp, RtEvent init, CollectiveMapping* map, Provenance* prov,
        Deserializer& derez)
      : DistributedCollectable(sp.did, false /*register with runtime*/, map),
        handle(sp), provenance(prov), initialized(init),
        allocation_state(FIELD_ALLOC_INVALID), outstanding_allocators(0),
        outstanding_invalidations(0)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      size_t num_fields;
      derez.deserialize(num_fields);
      if (num_fields > 0)
      {
        allocation_state = FIELD_ALLOC_READ_ONLY;
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
      }
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Field Space %lld %d %lld",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, handle.did);
#endif
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::~FieldSpaceNode(void)
    //--------------------------------------------------------------------------
    {
      // Next we can delete our layouts
      for (const std::pair<
               const LEGION_FIELD_MASK_FIELD_TYPE,
               lng::list<LayoutDescription*>>& it : layouts)
      {
        for (LayoutDescription* layout : it.second)
        {
          if (layout->remove_reference())
            delete layout;
        }
      }
      layouts.clear();
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      // Unregister ourselves from the context
      if (registered_with_runtime)
        runtime->remove_node(handle);
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(void)
      : field_size(0), idx(0), serdez_id(0), provenance(nullptr),
        collective(false), local(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(
        size_t size, unsigned id, CustomSerdezID sid, Provenance* prov,
        bool loc, bool col)
      : field_size(size), idx(id), serdez_id(sid), provenance(prov),
        collective(col), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(
        ApEvent ready, unsigned id, CustomSerdezID sid, Provenance* prov,
        bool loc, bool col)
      : field_size(0), size_ready(ready), idx(id), serdez_id(sid),
        provenance(prov), collective(col), local(loc)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(const FieldInfo& rhs)
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance),
        collective(rhs.collective), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::FieldInfo(FieldInfo&& rhs) noexcept
      : field_size(rhs.field_size), size_ready(rhs.size_ready), idx(rhs.idx),
        serdez_id(rhs.serdez_id), provenance(rhs.provenance),
        collective(rhs.collective), local(rhs.local)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo::~FieldInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo& FieldSpaceNode::FieldInfo::operator=(
        const FieldInfo& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      field_size = rhs.field_size;
      size_ready = rhs.size_ready;
      idx = rhs.idx;
      serdez_id = rhs.serdez_id;
      provenance = rhs.provenance;
      collective = rhs.collective;
      local = rhs.local;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode::FieldInfo& FieldSpaceNode::FieldInfo::operator=(
        FieldInfo&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      field_size = rhs.field_size;
      size_ready = rhs.size_ready;
      idx = rhs.idx;
      serdez_id = rhs.serdez_id;
      provenance = rhs.provenance;
      collective = rhs.collective;
      local = rhs.local;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FieldInfo::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(field_size);
      rez.serialize(size_ready);
      rez.serialize(idx);
      rez.serialize<bool>(collective);
      rez.serialize<bool>(local);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FieldInfo::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(field_size);
      derez.deserialize(size_ready);
      derez.deserialize(idx);
      derez.deserialize<bool>(collective);
      derez.deserialize<bool>(local);
      // Reference passed back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID FieldSpaceNode::get_owner_space(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner_space(handle);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID FieldSpaceNode::get_owner_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      return (handle.get_id() % runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::attach_semantic_information(
        SemanticTag tag, AddressSpaceID source, const void* buffer, size_t size,
        bool is_mutable, bool local_only)
    //--------------------------------------------------------------------------
    {
      bool added = true;
      {
        AutoLock n_lock(node_lock);
        // See if it already exists
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.buffer.get_size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Inconsistent Semantic Tag value for tag " << tag
                      << " with different sizes of " << size << " and "
                      << finder->second.buffer.get_size()
                      << " for index tree node";
                error.raise();
              }
              // Otherwise do a bitwise comparison
              {
                const char* orig =
                    (const char*)finder->second.buffer.get_buffer();
                const char* next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    Error error(LEGION_INTERFACE_EXCEPTION);
                    error << "Inconsistent Semantic Tag value for tag " << tag
                          << " with different values at byte " << idx
                          << " for index tree node, " << std::hex
                          << (int)orig[idx] << " != " << (int)next[idx];
                    error.raise();
                  }
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can overwrite
              finder->second.buffer.save_buffer(buffer, size);
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer.save_buffer(buffer, size);
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_info[tag] = SemanticInfo(buffer, size, is_mutable);
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message
        // didn't come from the owner, then send it
        if ((owner_space != runtime->address_space) &&
            (source != owner_space) && !local_only)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          send_semantic_info(owner_space, tag, buffer, size, is_mutable, done);
          if (!done.has_triggered())
            done.wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::attach_semantic_information(
        FieldID fid, SemanticTag tag, AddressSpaceID source, const void* buffer,
        size_t size, bool is_mutable, bool local_only)
    //--------------------------------------------------------------------------
    {
      bool added = true;
      {
        AutoLock n_lock(node_lock);
        // See if it already exists
        lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>::iterator
            finder = semantic_field_info.find(
                std::pair<FieldID, SemanticTag>(fid, tag));
        if (finder != semantic_field_info.end())
        {
          // First check to see if it is valid
          if (finder->second.is_valid())
          {
            if (!finder->second.is_mutable)
            {
              // Check to make sure that the bits are the same
              if (size != finder->second.buffer.get_size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Inconsistent Semantic Tag value for tag " << tag
                      << " with different sizes of " << size << " and "
                      << finder->second.buffer.get_size()
                      << " for index tree node";
                error.raise();
              }
              // Otherwise do a bitwise comparison
              {
                const char* orig =
                    (const char*)finder->second.buffer.get_buffer();
                const char* next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    Error error(LEGION_INTERFACE_EXCEPTION);
                    error << "Inconsistent Semantic Tag value for tag " << tag
                          << " with different values at byte " << idx
                          << " for index tree node, " << std::hex
                          << (int)orig[idx] << " != " << (int)next[idx];
                    error.raise();
                  }
                }
              }
              added = false;
            }
            else
            {
              // Mutable so we can overwrite
              finder->second.buffer.save_buffer(buffer, size);
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer.save_buffer(buffer, size);
            // Trigger will happen by caller
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
        {
          semantic_field_info[std::pair<FieldID, SemanticTag>(fid, tag)] =
              SemanticInfo(buffer, size, is_mutable);
        }
      }
      if (added)
      {
        AddressSpaceID owner_space = get_owner_space();
        // If we are not the owner and the message
        // didn't come from the owner, then send it
        if ((owner_space != runtime->address_space) &&
            (source != owner_space) && !local_only)
          send_semantic_field_info(
              owner_space, fid, tag, buffer, size, is_mutable);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::retrieve_semantic_information(
        SemanticTag tag, const void*& result, size_t& size, bool can_fail,
        bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else  // can use the canonical event
              wait_on = finder->second.ready_event;
          }
          else if (wait_until)  // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        // Nothing to wait on so we have to do something
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "invalid semantic tag " << tag << " for field space "
              << handle.get_id();
        error.raise();
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
        {
          FieldSpaceSemanticInfoRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(tag);
            rez.serialize(can_fail);
            rez.serialize(wait_until);
            rez.serialize(wait_on);
          }
          rez.dispatch(owner_space);
        }
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock, false /*exclusive*/);
      lng::map<SemanticTag, SemanticInfo>::const_iterator finder =
          semantic_info.find(tag);
      if (finder == semantic_info.end())
      {
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "invalid semantic tag " << tag << " for field space "
              << handle.get_id();
        error.raise();
      }
      result = finder->second.buffer.get_buffer();
      size = finder->second.buffer.get_size();
      return true;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::retrieve_semantic_information(
        FieldID fid, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != runtime->address_space);
      {
        AutoLock n_lock(node_lock);
        lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>::const_iterator
            finder = semantic_field_info.find(
                std::pair<FieldID, SemanticTag>(fid, tag));
        if (finder != semantic_field_info.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else  // can use the canonical event
              wait_on = finder->second.ready_event;
          }
          else if (wait_until)  // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_info[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "invalid semantic tag " << tag << " for field " << fid
              << " of field space " << handle.get_id();
        error.raise();
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
        {
          FieldSemanticInfoRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(fid);
            rez.serialize(tag);
            rez.serialize(can_fail);
            rez.serialize(wait_until);
            rez.serialize(wait_on);
          }
          rez.dispatch(owner_space);
        }
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock n_lock(node_lock, false /*exclusive*/);
      lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>::const_iterator
          finder = semantic_field_info.find(
              std::pair<FieldID, SemanticTag>(fid, tag));
      if (finder == semantic_field_info.end())
      {
        if (can_fail)
          return false;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "invalid semantic tag " << tag << " for field " << fid
              << " of field space " << handle.get_id();
        error.raise();
      }
      result = finder->second.buffer.get_buffer();
      size = finder->second.buffer.get_size();
      return true;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* result, size_t size,
        bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      FieldSpaceSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(result, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_semantic_field_info(
        AddressSpaceID target, FieldID fid, SemanticTag tag, const void* result,
        size_t size, bool is_mutable, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      FieldSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize(fid);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(result, size);
        rez.serialize(is_mutable);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_semantic_request(
        SemanticTag tag, AddressSpaceID source, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(get_owner_space() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        lng::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_info.find(tag);
        if (finder != semantic_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_info[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(source, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_semantic_field_request(
        FieldID fid, SemanticTag tag, AddressSpaceID source, bool can_fail,
        bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(get_owner_space() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock n_lock(node_lock);
        // See if we already have the data
        std::pair<FieldID, SemanticTag> key(fid, tag);
        lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>::iterator
            finder = semantic_field_info.find(key);
        if (finder != semantic_field_info.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_field_info[key] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticFieldRequestArgs args(this, fid, tag, source);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_field_info(
            source, fid, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::SemanticFieldRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_field_request(
          fid, tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      FieldSpaceNode* node = runtime->get_node(handle);
      node->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      FieldSpaceNode* node = runtime->get_node(handle);
      node->process_semantic_field_request(
          fid, tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*local only*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent ready;
      derez.deserialize(ready);
      runtime->get_node(handle)->attach_semantic_information(
          fid, tag, source, buffer, size, is_mutable, false /*local lonly*/);
      if (ready.exists())
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::FindTargetsFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      targets.emplace_back(target);
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::create_allocator(
        AddressSpaceID source, RtUserEvent ready_event,
        bool sharded_owner_context, bool owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      if (sharded_owner_context)
      {
        // If we were the sharded collective context that made this
        // field space and we're still in collective allocation mode
        // then we are trivially done
        if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        {
          legion_assert(outstanding_allocators == 0);
          outstanding_allocators = 1;
          return RtEvent::NO_RT_EVENT;
        }
        // Otherwise if we're not the owner shard then we're also done since
        // the owner shard is the only one doing the allocation
        if (!owner_shard)
          return RtEvent::NO_RT_EVENT;
      }
      if (is_owner())
      {
        switch (allocation_state)
        {
          case FIELD_ALLOC_INVALID:
            {
              legion_assert(outstanding_allocators == 0);
              legion_assert(remote_field_infos.size() == 1);
              const AddressSpaceID remote_owner = *(remote_field_infos.begin());
              remote_field_infos.clear();
              legion_assert(remote_owner != local_space);
              // Should never get the ships in the night case either
              legion_assert(remote_owner != source);
              if (!ready_event.exists())
                ready_event = Runtime::create_rt_user_event();
              outstanding_invalidations++;
              // Add a reference that will be remove when the flush returns
              add_base_gc_ref(FIELD_ALLOCATOR_REF);
              // Send the invalidation and make ourselves the new
              // pending exclusive allocator value
              FieldSpaceAllocatorInvalidation rez;
              {
                RezCheck z(rez);
                rez.serialize(handle);
                rez.serialize(ready_event);
                rez.serialize<bool>(true);   // flush allocation
                rez.serialize<bool>(false);  // need merge
              }
              rez.dispatch(remote_owner);
              outstanding_allocators = 1;
              pending_field_allocation = ready_event;
              allocation_state = FIELD_ALLOC_PENDING;
              break;
            }
          case FIELD_ALLOC_COLLECTIVE:
            {
              // This is the case when we're still in collective mode
              // and we need to switch to exclusive mode on just one node
              // because someone else asked for an allocator
              if (outstanding_allocators > 0)
              {
                legion_assert(!remote_field_infos.empty());
                legion_assert(outstanding_invalidations == 0);
                std::set<RtEvent> preconditions;
                for (const AddressSpaceID& it : remote_field_infos)
                {
                  const RtUserEvent done = Runtime::create_rt_user_event();
                  outstanding_invalidations++;
                  FieldSpaceAllocatorInvalidation rez;
                  {
                    RezCheck z(rez);
                    rez.serialize(handle);
                    rez.serialize(done);
                    rez.serialize<bool>(true);  // flush allocation
                    rez.serialize<bool>(true);  // need merge
                  }
                  rez.dispatch(it);
                  preconditions.insert(done);
                }
                remote_field_infos.clear();
                pending_field_allocation = Runtime::merge_events(preconditions);
                allocation_state = FIELD_ALLOC_PENDING;
                break;
              }
              // otherwise we fall through to the identical read-only case
            }
          case FIELD_ALLOC_READ_ONLY:
            {
              legion_assert(outstanding_allocators == 0);
              // Send any invalidations to anyone not the source
              bool full_update = true;
              RtEvent invalidations_done;
              if (!remote_field_infos.empty())
              {
                legion_assert(outstanding_invalidations == 0);
                std::set<RtEvent> preconditions;
                for (const AddressSpaceID& it : remote_field_infos)
                {
                  if (it == source)
                  {
                    full_update = false;
                    continue;
                  }
                  const RtUserEvent done = Runtime::create_rt_user_event();
                  outstanding_invalidations++;
                  // Add a reference that will be remove when the flush returns
                  add_base_gc_ref(FIELD_ALLOCATOR_REF);
                  FieldSpaceAllocatorInvalidation rez;
                  {
                    RezCheck z(rez);
                    rez.serialize(handle);
                    rez.serialize(done);
                    rez.serialize<bool>(false);  // flush allocation
                    rez.serialize<bool>(false);  // need merge
                  }
                  rez.dispatch(it);
                  preconditions.insert(done);
                }
                remote_field_infos.clear();
                if (!preconditions.empty())
                  invalidations_done = Runtime::merge_events(preconditions);
              }
              if (source != local_space)
              {
                legion_assert(ready_event.exists());
                // Send the response back to the source and mark that
                // we are now invalid
                FieldSpaceAllocatorResponse rez;
                {
                  RezCheck z(rez);
                  rez.serialize(handle);
                  rez.serialize(invalidations_done);
                  if (full_update)
                  {
                    rez.serialize(field_infos.size());
                    for (std::map<FieldID, FieldInfo>::iterator it =
                             field_infos.begin();
                         it != field_infos.end();
                         /*nothing*/)
                    {
                      rez.serialize(it->first);
                      it->second.serialize(rez);
                      if (!it->second.local)
                      {
                        std::map<FieldID, FieldInfo>::iterator to_delete = it++;
                        field_infos.erase(to_delete);
                      }
                      else
                        it++;  // skip deleting local fields
                    }
                  }
                  if (full_update ||
                      (allocation_state != FIELD_ALLOC_COLLECTIVE))
                  {
                    rez.serialize(unallocated_indexes);
                    rez.serialize<size_t>(available_indexes.size());
                    for (const std::pair<unsigned, RtEvent>& it :
                         available_indexes)
                    {
                      rez.serialize(it.first);
                      rez.serialize(it.second);
                    }
                  }
                  unallocated_indexes.clear();
                  available_indexes.clear();
                  rez.serialize(ready_event);
                }
                // Add a reference to this node to keep it alive until we
                // get the corresponding free operation from the remote node
                add_base_gc_ref(FIELD_ALLOCATOR_REF);
                rez.dispatch(source);
                remote_field_infos.insert(source);
                allocation_state = FIELD_ALLOC_INVALID;
              }
              else
              {
                // We are now the exclusive allocation owner
                if (outstanding_invalidations > 0)
                {
                  pending_field_allocation = invalidations_done;
                  allocation_state = FIELD_ALLOC_PENDING;
                }
                else  // we're ready now
                  allocation_state = FIELD_ALLOC_EXCLUSIVE;
                outstanding_allocators = 1;
                if (ready_event.exists())
                  Runtime::trigger_event(ready_event, invalidations_done);
                return invalidations_done;
              }
              break;
            }
          case FIELD_ALLOC_PENDING:
            {
              outstanding_allocators++;
              if (ready_event.exists())
                Runtime::trigger_event(ready_event, pending_field_allocation);
              return pending_field_allocation;
            }
          case FIELD_ALLOC_EXCLUSIVE:
            {
              outstanding_allocators++;
              if (ready_event.exists())
                Runtime::trigger_event(ready_event, pending_field_allocation);
              break;
            }
          default:
            std::abort();
        }
      }
      else
      {
        legion_assert(!ready_event.exists());
        legion_assert(source == local_space);
        // Order remote allocation requests to prevent ships-in-the-night
        while (pending_field_allocation.exists())
        {
          const RtEvent wait_on = pending_field_allocation;
          if (!wait_on.has_triggered())
          {
            n_lock.release();
            wait_on.wait();
            n_lock.reacquire();
          }
          else
            break;
        }
        // See if we already have allocation privileges
        if (allocation_state != FIELD_ALLOC_EXCLUSIVE)
        {
          ready_event = Runtime::create_rt_user_event();
          FieldSpaceAllocatorRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(ready_event);
          }
          rez.dispatch(owner_space);
          pending_field_allocation = ready_event;
        }
        else  // Have privileges, increment our allocator count
          outstanding_allocators++;
      }
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::destroy_allocator(
        AddressSpaceID source, bool sharded_owner_context, bool owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      legion_assert(
          (allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
          (allocation_state == FIELD_ALLOC_COLLECTIVE) ||
          (allocation_state == FIELD_ALLOC_INVALID));
      if (sharded_owner_context)
      {
        // If we were the sharded collective context that made this
        // field space and we're still in collective allocation mode
        // then we are trivially done
        if (allocation_state == FIELD_ALLOC_COLLECTIVE)
        {
          legion_assert(outstanding_allocators == 1);
          outstanding_allocators = 0;
          return RtEvent::NO_RT_EVENT;
        }
        // Otherwise if we're not the owner shard then we're also done since
        // the owner shard is the only one doing the allocation
        if (!owner_shard)
          return RtEvent::NO_RT_EVENT;
      }
      if (allocation_state == FIELD_ALLOC_INVALID)
      {
        legion_assert(!is_owner());
        return RtEvent::NO_RT_EVENT;
      }
      else
      {
        legion_assert(outstanding_allocators > 0);
        if (--outstanding_allocators == 0)
        {
          // Now we go back to read-only mode
          allocation_state = FIELD_ALLOC_READ_ONLY;
          if (!is_owner())
          {
            const RtUserEvent done_event = Runtime::create_rt_user_event();
            // Send the allocation data back to the owner node
            FieldSpaceAllocatorFree rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize<bool>(true);  // return allocation
              rez.serialize(field_infos.size());
              for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
              {
                rez.serialize(it.first);
                it.second.serialize(rez);
              }
              rez.serialize(unallocated_indexes);
              unallocated_indexes.clear();
              rez.serialize<size_t>(available_indexes.size());
              while (!available_indexes.empty())
              {
                std::pair<unsigned, RtEvent>& next = available_indexes.front();
                rez.serialize(next.first);
                rez.serialize(next.second);
                available_indexes.pop_front();
              }
              rez.serialize(done_event);
            }
            rez.dispatch(owner_space);
            return done_event;
          }
        }
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::initialize_fields(
        const std::vector<size_t>& sizes, const std::vector<FieldID>& fids,
        CustomSerdezID serdez_id, Provenance* prov, bool collective)
    //--------------------------------------------------------------------------
    {
      legion_assert(!fids.empty());
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal duplicate field ID " << fid << " used by the "
                << "application in field space " << handle.get_id();
          error.raise();
        }
        // Find an index in which to allocate this field
        RtEvent dummy_event;
        int result = allocate_index(dummy_event, true /*initializing*/);
        if (result < 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Exceeded maximum number of allocated fields for "
                << "field space " << handle.get_id()
                << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
                << " and related macros at the top of legion_config.h "
                << "and recompile.";
          error.raise();
        }
        legion_assert(!dummy_event.exists());
        const unsigned index = result;
        field_infos[fid] = FieldInfo(
            sizes[idx], index, serdez_id, prov, false /*local*/, collective);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::initialize_fields(
        ApEvent sizes_ready, const std::vector<FieldID>& fids,
        CustomSerdezID serdez_id, Provenance* prov, bool collective)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        FieldID fid = fids[idx];
        if (field_infos.find(fid) != field_infos.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal duplicate field ID " << fid << " used by the "
                << "application in field space " << handle.get_id() << ".";
          error.raise();
        }
        // Find an index in which to allocate this field
        RtEvent dummy_event;
        int result = allocate_index(dummy_event, true /*initializing*/);
        if (result < 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Exceeded maximum number of allocated fields for "
                << "field space " << handle.get_id()
                << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
                << " and related macros at the top of legion_config.h "
                << "and recompile.";
          error.raise();
        }
        legion_assert(!dummy_event.exists());
        const unsigned index = result;
        field_infos[fid] = FieldInfo(
            sizes_ready, index, serdez_id, prov, false /*local*/, collective);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(
        FieldID fid, size_t size, CustomSerdezID serdez_id, Provenance* prov,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        legion_assert(!is_owner());
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        FieldAllocationRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(ApEvent::NO_AP_EVENT);
          if (prov != nullptr)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(1);  // only allocating one field
          rez.serialize(fid);
          rez.serialize(size);
        }
        rez.dispatch(owner_space);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
      if (finder != field_infos.end())
      {
        // Handle the case of deduplicating fields that were allocated
        // in a collective mode but are now merged together
        if (finder->second.collective)
          return RtEvent::NO_RT_EVENT;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal duplicate field ID " << fid << " used by the "
              << "application in field space " << handle.get_id() << ".";
        error.raise();
      }
      // Find an index in which to allocate this field
      RtEvent ready_event;
      int result = allocate_index(ready_event);
      if (result < 0)
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Exceeded maximum number of allocated fields for "
              << "field space " << handle.get_id()
              << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
              << " and related macros at the top of legion_config.h "
              << "and recompile.";
        error.raise();
      }
      const unsigned index = result;
      field_infos[fid] = FieldInfo(
          size, index, serdez_id, prov, false /*local*/,
          (allocation_state == FIELD_ALLOC_COLLECTIVE));
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_field(
        FieldID fid, ApEvent size_ready, CustomSerdezID serdez_id,
        Provenance* prov, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        legion_assert(!is_owner());
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        FieldAllocationRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(size_ready);  // size ready
          if (prov != nullptr)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(1);  // only allocating one field
          rez.serialize(fid);
        }
        rez.dispatch(owner_space);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
      if (finder != field_infos.end())
      {
        // Handle the case of deduplicating fields that were allocated
        // in a collective mode but are now merged together
        if (finder->second.collective)
          return RtEvent::NO_RT_EVENT;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal duplicate field ID " << fid << " used by the "
              << "application in field space " << handle.get_id() << ".";
        error.raise();
      }
      // Find an index in which to allocate this field
      RtEvent ready_event;
      int result = allocate_index(ready_event);
      if (result < 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Exceeded maximum number of allocated fields for "
              << "field space " << handle.get_id()
              << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
              << " and related macros at the top of legion_config.h "
              << "and recompile.";
        error.raise();
      }
      const unsigned index = result;
      field_infos[fid] = FieldInfo(
          size_ready, index, serdez_id, prov, false /*local*/,
          (allocation_state == FIELD_ALLOC_COLLECTIVE));
      return ready_event;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(
        const std::vector<size_t>& sizes, const std::vector<FieldID>& fids,
        CustomSerdezID serdez_id, Provenance* prov, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      legion_assert(!fids.empty());
      legion_assert(sizes.size() == fids.size());
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        FieldAllocationRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(ApEvent::NO_AP_EVENT);
          if (prov != nullptr)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
          {
            rez.serialize(fids[idx]);
            rez.serialize(sizes[idx]);
          }
        }
        rez.dispatch(owner_space);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        const FieldID fid = fids[idx];
        std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
        if (finder != field_infos.end())
        {
          // Handle the case of deduplicating fields that were allocated
          // in a collective mode but are now merged together
          if (finder->second.collective)
            continue;
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal duplicate field ID " << fid << " used by the "
                << "application in field space " << handle.get_id() << ".";
          error.raise();
        }
        // Find an index in which to allocate this field
        RtEvent ready_event;
        int result = allocate_index(ready_event);
        if (result < 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Exceeded maximum number of allocated fields for "
                << "field space " << handle.get_id()
                << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
                << " and related macros at the top of legion_config.h "
                << "and recompile.";
          error.raise();
        }
        if (ready_event.exists())
          allocated_events.insert(ready_event);
        const unsigned index = result;
        field_infos[fid] = FieldInfo(
            sizes[idx], index, serdez_id, prov, false /*local*/,
            (allocation_state == FIELD_ALLOC_COLLECTIVE));
      }
      if (!allocated_events.empty())
        return Runtime::merge_events(allocated_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::allocate_fields(
        ApEvent sizes_ready, const std::vector<FieldID>& fids,
        CustomSerdezID serdez_id, Provenance* prov, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return RtEvent::NO_RT_EVENT;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      // Check to see if we can do the allocation
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        const RtUserEvent allocated_event = Runtime::create_rt_user_event();
        FieldAllocationRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          rez.serialize(sizes_ready);
          if (prov != nullptr)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
            rez.serialize(fids[idx]);
        }
        rez.dispatch(owner_space);
        return allocated_event;
      }
      // We're the owner so do the field allocation
      std::set<RtEvent> allocated_events;
      for (unsigned idx = 0; idx < fids.size(); idx++)
      {
        const FieldID fid = fids[idx];
        std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
        if (finder != field_infos.end())
        {
          // Handle the case of deduplicating fields that were allocated
          // in a collective mode but are now merged together
          if (finder->second.collective)
            continue;
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal duplicate field ID " << fid << " used by the "
                << "application in field space " << handle.get_id() << ".";
          error.raise();
        }
        // Find an index in which to allocate this field
        RtEvent ready_event;
        int result = allocate_index(ready_event);
        if (result < 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Exceeded maximum number of allocated fields for "
                << "field space " << handle.get_id()
                << ". Change LEGION_MAX_FIELDS from " << LEGION_MAX_FIELDS
                << " and related macros at the top of legion_config.h "
                << "and recompile.";
          error.raise();
        }
        if (ready_event.exists())
          allocated_events.insert(ready_event);
        const unsigned index = result;
        field_infos[fid] = FieldInfo(
            sizes_ready, index, serdez_id, prov, false /*local*/,
            (allocation_state == FIELD_ALLOC_COLLECTIVE));
      }
      if (!allocated_events.empty())
        return Runtime::merge_events(allocated_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::update_field_size(
        FieldID fid, size_t field_size, std::set<RtEvent>& update_events,
        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
      if (finder != field_infos.end())
      {
        legion_assert(finder->second.field_size == 0);
        legion_assert(finder->second.size_ready.exists());
        finder->second.field_size = field_size;
        finder->second.size_ready = ApEvent::NO_AP_EVENT;
      }
      // Now figure out where the updates need to go
      if (is_owner())
      {
        // If we're not the exclusive allocator then broadcast
        // this out to all the other nodes so that they see updates
        if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
            (allocation_state != FIELD_ALLOC_COLLECTIVE))
        {
          // Send messages to all the read-only field infos
          for (const AddressSpaceID& it : remote_field_infos)
          {
            if (it == source)
              continue;
            const RtUserEvent done_event = Runtime::create_rt_user_event();
            FieldSizeUpdate rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(done_event);
              rez.serialize(fid);
              rez.serialize(field_size);
            }
            pack_global_ref();
            rez.dispatch(it);
            update_events.insert(done_event);
          }
        }
      }
      else
      {
        // If the source is not the owner and we're not in a collective
        // mode then we have to send the message to the owner
        if ((source != owner_space) &&
            (allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
            (allocation_state != FIELD_ALLOC_COLLECTIVE))
        {
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          FieldSizeUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(done_event);
            rez.serialize(fid);
            rez.serialize(field_size);
          }
          pack_global_ref();
          rez.dispatch(owner_space);
          update_events.insert(done_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field(
        FieldID fid, AddressSpaceID source, std::set<RtEvent>& applied,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        legion_assert(!is_owner());
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        FieldFreeMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(1);
          rez.serialize(fid);
          rez.serialize(done_event);
        }
        rez.dispatch(owner_space);
        applied.insert(done_event);
        return;
      }
      std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
      legion_assert(finder != field_infos.end());
      // Remove it from the field map
      field_infos.erase(finder);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_fields(
        const std::vector<FieldID>& to_free, AddressSpaceID source,
        std::set<RtEvent>& applied, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        legion_assert(!is_owner());
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        FieldFreeMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (const FieldID& field_id : to_free) rez.serialize(field_id);
          rez.serialize(done_event);
        }
        rez.dispatch(owner_space);
        applied.insert(done_event);
        return;
      }
      for (const FieldID& fid : to_free)
      {
        std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
        legion_assert(finder != field_infos.end());
        // Remove it from the fields map
        field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_field_indexes(
        const std::vector<FieldID>& to_free, RtEvent freed_event,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      // For control replication see if we've been invalidated and do not need
      // to do anything because we are not the owner any longer
      if (sharded_non_owner && (allocation_state != FIELD_ALLOC_COLLECTIVE))
        return;
      while (allocation_state == FIELD_ALLOC_PENDING)
      {
        legion_assert(is_owner());
        const RtEvent wait_on = pending_field_allocation;
        n_lock.release();
        if (!wait_on.has_triggered())
          wait_on.wait();
        n_lock.reacquire();
      }
      if ((allocation_state != FIELD_ALLOC_EXCLUSIVE) &&
          (allocation_state != FIELD_ALLOC_COLLECTIVE))
      {
        legion_assert(!is_owner());
        FieldFreeIndexes rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize<size_t>(to_free.size());
          for (const FieldID& field_id : to_free) rez.serialize(field_id);
          rez.serialize(freed_event);
        }
        rez.dispatch(owner_space);
        return;
      }
      for (const FieldID& fid : to_free)
      {
        std::map<FieldID, FieldInfo>::iterator finder = field_infos.find(fid);
        legion_assert(finder != field_infos.end());
        // Skip freeing any local field indexes here
        if (!finder->second.local)
          free_index(finder->second.idx, freed_event);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_fields(
        const std::vector<FieldID>& fids, const std::vector<size_t>& sizes,
        CustomSerdezID serdez_id, const std::set<unsigned>& indexes,
        std::vector<unsigned>& new_indexes, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      legion_assert(fids.size() == sizes.size());
      legion_assert(new_indexes.empty());
      if (!is_owner())
      {
        // If we're not the owner, send a message to the owner
        // to do the local field allocation
        RtUserEvent allocated_event = Runtime::create_rt_user_event();
        LocalFieldAllocRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(allocated_event);
          rez.serialize(serdez_id);
          if (prov != nullptr)
            prov->serialize(rez);
          else
            Provenance::serialize_null(rez);
          rez.serialize<size_t>(fids.size());
          for (unsigned idx = 0; idx < fids.size(); idx++)
          {
            rez.serialize(fids[idx]);
            rez.serialize(sizes[idx]);
          }
          rez.serialize<size_t>(indexes.size());
          for (const unsigned& index : indexes) rez.serialize(index);
          rez.serialize(&new_indexes);
        }
        rez.dispatch(owner_space);
        // Wait for the result
        allocated_event.wait();
        if (new_indexes.empty())
          return false;
        // When we wake up then fill in the field information
        AutoLock n_lock(node_lock);
        legion_assert(!fids.empty());
        legion_assert(new_indexes.size() == fids.size());
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          field_infos[fid] = FieldInfo(
              sizes[idx], new_indexes[idx], serdez_id, prov, true /*local*/);
        }
      }
      else
      {
        // We're the owner so do the field allocation
        AutoLock n_lock(node_lock);
        if (!allocate_local_indexes(serdez_id, sizes, indexes, new_indexes))
          return false;
        legion_assert(!fids.empty());
        for (unsigned idx = 0; idx < fids.size(); idx++)
        {
          FieldID fid = fids[idx];
          if (field_infos.find(fid) != field_infos.end())
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Illegal duplicate field ID " << fid << " used by the "
                  << "application in field space " << handle.get_id() << ".";
            error.raise();
          }
          field_infos[fid] = FieldInfo(
              sizes[idx], new_indexes[idx], serdez_id, prov, true /*local*/);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_local_fields(
        const std::vector<FieldID>& to_free,
        const std::vector<unsigned>& indexes, const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(to_free.size() == indexes.size());
      if (mapping != nullptr)
      {
        if (mapping->contains(owner_space))
        {
          if (local_space != owner_space)
            return;
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == local_space)
          {
            LocalFieldFreeMessage rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize<size_t>(to_free.size());
              for (unsigned idx = 0; idx < to_free.size(); idx++)
              {
                rez.serialize(to_free[idx]);
                rez.serialize(indexes[idx]);
              }
            }
            rez.dispatch(owner_space);
          }
          return;
        }
      }
      else
      {
        if (!is_owner())
        {
          // Send a message to the owner to do the free of the fields
          LocalFieldFreeMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize<size_t>(to_free.size());
            for (unsigned idx = 0; idx < to_free.size(); idx++)
            {
              rez.serialize(to_free[idx]);
              rez.serialize(indexes[idx]);
            }
          }
          rez.dispatch(owner_space);
          return;
        }
      }
      legion_assert(is_owner());
      // Do the local free
      AutoLock n_lock(node_lock);
      for (const FieldID& field_id : to_free)
      {
        std::map<FieldID, FieldInfo>::iterator finder =
            field_infos.find(field_id);
        legion_assert(finder != field_infos.end());
        field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::update_local_fields(
        const std::vector<FieldID>& fids, const std::vector<size_t>& sizes,
        const std::vector<CustomSerdezID>& serdez_ids,
        const std::vector<unsigned>& indexes, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(fids.size() == sizes.size());
      legion_assert(fids.size() == serdez_ids.size());
      legion_assert(fids.size() == indexes.size());
      AutoLock n_lock(node_lock);
      for (unsigned idx = 0; idx < fids.size(); idx++)
        field_infos[fids[idx]] = FieldInfo(
            sizes[idx], indexes[idx], serdez_ids[idx], provenance,
            true /*local*/);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::remove_local_fields(
        const std::vector<FieldID>& to_remove)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      for (const FieldID& field_id : to_remove)
      {
        std::map<FieldID, FieldInfo>::iterator finder =
            field_infos.find(field_id);
        if (finder != field_infos.end())
          field_infos.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::has_field(FieldID fid)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        // Check to see if we have a valid copy of the field infos
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID, FieldInfo>::const_iterator finder =
              field_infos.find(fid);
          if (finder == field_infos.end())
            return false;
          else
            return true;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      std::map<FieldID, FieldInfo>::const_iterator finder =
          local_infos.find(fid);
      if (finder == local_infos.end())
        return false;
      else
        return true;
    }

    //--------------------------------------------------------------------------
    size_t FieldSpaceNode::get_field_size(FieldID fid)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_for;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID, FieldInfo>::const_iterator finder =
              field_infos.find(fid);
          legion_assert(finder != field_infos.end());
          // See if this field has been allocated or not yet
          if (!finder->second.size_ready.exists())
            return finder->second.field_size;
          wait_for = Runtime::protect_event(finder->second.size_ready);
        }
      }
      if (!wait_for.exists())
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        std::map<FieldID, FieldInfo>::const_iterator finder =
            local_infos.find(fid);
        legion_assert(finder != local_infos.end());
        // See if this field has been allocated or not yet
        if (!finder->second.size_ready.exists())
          return finder->second.field_size;
        wait_for = Runtime::protect_event(finder->second.size_ready);
      }
      if (!wait_for.has_triggered())
        wait_for.wait();
      return get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID FieldSpaceNode::get_field_serdez(FieldID fid)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_for;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID, FieldInfo>::const_iterator finder =
              field_infos.find(fid);
          legion_assert(finder != field_infos.end());
          // See if this field has been allocated or not yet
          if (!finder->second.size_ready.exists())
            return finder->second.serdez_id;
          wait_for = Runtime::protect_event(finder->second.size_ready);
        }
      }
      if (!wait_for.exists())
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        std::map<FieldID, FieldInfo>::const_iterator finder =
            local_infos.find(fid);
        legion_assert(finder != local_infos.end());
        // See if this field has been allocated or not yet
        if (!finder->second.size_ready.exists())
          return finder->second.serdez_id;
        wait_for = Runtime::protect_event(finder->second.size_ready);
      }
      if (!wait_for.has_triggered())
        wait_for.wait();
      return get_field_serdez(fid);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_all_fields(std::vector<FieldID>& to_set)
    //--------------------------------------------------------------------------
    {
      to_set.clear();
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          to_set.reserve(field_infos.size());
          for (const std::pair<const FieldID, FieldInfo>& info : field_infos)
            to_set.emplace_back(info.first);
          return;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      to_set.reserve(local_infos.size());
      for (const std::pair<const FieldID, FieldInfo>& info : local_infos)
        to_set.emplace_back(info.first);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(
        const FieldMask& mask, TaskContext* ctx,
        std::set<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      std::set<unsigned> local_indexes;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
          {
            if (mask.is_set(it.second.idx))
            {
              if (it.second.local)
                local_indexes.insert(it.second.idx);
              else
                to_set.insert(it.first);
            }
          }
          if (local_indexes.empty())
            return;
        }
      }
      if (local_indexes.empty())
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (const std::pair<const FieldID, FieldInfo>& it : local_infos)
        {
          if (mask.is_set(it.second.idx))
          {
            if (it.second.local)
              local_indexes.insert(it.second.idx);
            else
              to_set.insert(it.first);
          }
        }
      }
      if (!local_indexes.empty())
        ctx->get_local_field_set(handle, local_indexes, to_set);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(
        const FieldMask& mask, TaskContext* ctx,
        std::vector<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      std::set<unsigned> local_indexes;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
          {
            if (mask.is_set(it.second.idx))
            {
              if (it.second.local)
                local_indexes.insert(it.second.idx);
              else
                to_set.emplace_back(it.first);
            }
          }
          if (local_indexes.empty())
            return;
        }
      }
      if (local_indexes.empty())
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (const std::pair<const FieldID, FieldInfo>& it : local_infos)
        {
          if (mask.is_set(it.second.idx))
          {
            if (it.second.local)
              local_indexes.insert(it.second.idx);
            else
              to_set.emplace_back(it.first);
          }
        }
      }
      if (!local_indexes.empty())
        ctx->get_local_field_set(handle, local_indexes, to_set);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_set(
        const FieldMask& mask, const std::set<FieldID>& basis,
        std::set<FieldID>& to_set) const
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          // Only iterate over the basis fields here
          for (const FieldID& fid : basis)
          {
            std::map<FieldID, FieldInfo>::const_iterator finder =
                field_infos.find(fid);
            legion_assert(finder != field_infos.end());
            if (mask.is_set(finder->second.idx))
              to_set.insert(finder->first);
          }
          return;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Only iterate over the basis fields here
      for (const FieldID& fid : basis)
      {
        std::map<FieldID, FieldInfo>::const_iterator finder =
            local_infos.find(fid);
        legion_assert(finder != local_infos.end());
        if (mask.is_set(finder->second.idx))
          to_set.insert(finder->first);
      }
    }

    //--------------------------------------------------------------------------
    FieldMask FieldSpaceNode::get_field_mask(
        const std::set<FieldID>& privilege_fields) const
    //--------------------------------------------------------------------------
    {
      FieldMask result;
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (const FieldID& fid : privilege_fields)
          {
            std::map<FieldID, FieldInfo>::const_iterator finder =
                field_infos.find(fid);
            legion_assert(finder != field_infos.end());
            result.set_bit(finder->second.idx);
          }
          return result;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      for (const FieldID& fid : privilege_fields)
      {
        std::map<FieldID, FieldInfo>::const_iterator finder =
            local_infos.find(fid);
        legion_assert(finder != local_infos.end());
        result.set_bit(finder->second.idx);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned FieldSpaceNode::get_field_index(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          std::map<FieldID, FieldInfo>::const_iterator finder =
              field_infos.find(fid);
          legion_assert(finder != field_infos.end());
          return finder->second.idx;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      std::map<FieldID, FieldInfo>::const_iterator finder =
          local_infos.find(fid);
      legion_assert(finder != local_infos.end());
      return finder->second.idx;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::get_field_indexes(
        const std::vector<FieldID>& needed,
        std::vector<unsigned>& indexes) const
    //--------------------------------------------------------------------------
    {
      legion_assert(needed.size() == indexes.size());
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (unsigned idx = 0; idx < needed.size(); idx++)
          {
            std::map<FieldID, FieldInfo>::const_iterator finder =
                field_infos.find(needed[idx]);
            legion_assert(finder != field_infos.end());
            indexes[idx] = finder->second.idx;
          }
          return;
        }
      }
      std::map<FieldID, FieldInfo> local_infos;
      const RtEvent ready = request_field_infos_copy(&local_infos, local_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      for (unsigned idx = 0; idx < needed.size(); idx++)
      {
        std::map<FieldID, FieldInfo>::const_iterator finder =
            local_infos.find(needed[idx]);
        legion_assert(finder != local_infos.end());
        indexes[idx] = finder->second.idx;
      }
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::compute_field_layout(
        const std::vector<FieldID>& create_fields,
        std::vector<size_t>& field_sizes, std::vector<unsigned>& mask_index_map,
        std::vector<CustomSerdezID>& serdez, FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(field_sizes.size() == create_fields.size());
      legion_assert(mask_index_map.size() == create_fields.size());
      legion_assert(serdez.size() == create_fields.size());
      bool invalid = false;
      std::set<ApEvent> defer_events;
      std::map<unsigned /*mask index*/, unsigned /*layout index*/> index_map;
      {
        // Need to hold the lock when accessing field infos
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (unsigned idx = 0; idx < create_fields.size(); idx++)
          {
            const FieldID fid = create_fields[idx];
            std::map<FieldID, FieldInfo>::const_iterator finder =
                field_infos.find(fid);
            // Catch unknown fields here for now
            if (finder == field_infos.end())
            {
              Fatal fatal;
              fatal << "unknown field ID " << fid
                    << " requested during instance creation";
              fatal.raise();
            }
            if (finder->second.size_ready.exists())
              defer_events.insert(finder->second.size_ready);
            else if (defer_events.empty())
            {
              field_sizes[idx] = finder->second.field_size;
              index_map[finder->second.idx] = idx;
              serdez[idx] = finder->second.serdez_id;
              mask.set_bit(finder->second.idx);
            }
          }
        }
        else
          invalid = true;
      }
      if (invalid)
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (unsigned idx = 0; idx < create_fields.size(); idx++)
        {
          const FieldID fid = create_fields[idx];
          std::map<FieldID, FieldInfo>::const_iterator finder =
              local_infos.find(fid);
          // Catch unknown fields here for now
          if (finder == local_infos.end())
          {
            Fatal fatal;
            fatal << "unknown field ID " << fid
                  << " requested during instance creation";
            fatal.raise();
          }
          if (finder->second.size_ready.exists())
            defer_events.insert(finder->second.size_ready);
          else if (defer_events.empty())
          {
            field_sizes[idx] = finder->second.field_size;
            index_map[finder->second.idx] = idx;
            serdez[idx] = finder->second.serdez_id;
            mask.set_bit(finder->second.idx);
          }
        }
      }
      if (!defer_events.empty())
      {
        const RtEvent wait_on = Runtime::protect_merge_events(defer_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
        compute_field_layout(
            create_fields, field_sizes, mask_index_map, serdez, mask);
        return;
      }
      // Now we can linearize the index map without holding the lock
      unsigned idx = 0;
      for (const std::pair<const unsigned, unsigned>& it : index_map)
        mask_index_map[idx++] = it.second;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldAllocationRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      CustomSerdezID serdez_id;
      derez.deserialize(serdez_id);
      ApEvent sizes_ready;
      derez.deserialize(sizes_ready);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fids(num_fields);
      RtEvent ready;
      if (!sizes_ready.exists())
      {
        std::vector<size_t> sizes(num_fields);
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          derez.deserialize(fids[idx]);
          derez.deserialize(sizes[idx]);
        }
        FieldSpaceNode* node = runtime->get_node(handle);
        ready = node->allocate_fields(sizes, fids, serdez_id, provenance);
      }
      else
      {
        for (unsigned idx = 0; idx < num_fields; idx++)
          derez.deserialize(fids[idx]);
        FieldSpaceNode* node = runtime->get_node(handle);
        ready = node->allocate_fields(sizes_ready, fids, serdez_id, provenance);
      }
      Runtime::trigger_event(done, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldFreeMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
        derez.deserialize(fields[idx]);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      FieldSpaceNode* node = runtime->get_node(handle);
      std::set<RtEvent> applied;
      node->free_fields(fields, source, applied);
      if (done_event.exists())
      {
        if (!applied.empty())
          Runtime::trigger_event(done_event, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldFreeIndexes::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
        derez.deserialize(fields[idx]);
      RtEvent freed_event;
      derez.deserialize(freed_event);
      FieldSpaceNode* node = runtime->get_node(handle);
      node->free_field_indexes(fields, freed_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceLayoutInvalidation::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      FieldSpaceNode* node = runtime->get_node(handle);
      std::set<RtEvent> applied;
      node->invalidate_layouts(index, applied, source);
      node->unpack_global_ref();
      if (!applied.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LocalFieldAllocRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      CustomSerdezID serdez_id;
      derez.deserialize(serdez_id);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      std::vector<size_t> sizes(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fields[idx]);
        derez.deserialize(sizes[idx]);
      }
      size_t num_indexes;
      derez.deserialize(num_indexes);
      std::set<unsigned> current_indexes;
      for (unsigned idx = 0; idx < num_indexes; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        current_indexes.insert(index);
      }
      std::vector<unsigned>* destination;
      derez.deserialize(destination);

      FieldSpaceNode* node = runtime->get_node(handle);
      std::vector<unsigned> new_indexes;
      if (node->allocate_local_fields(
              fields, sizes, serdez_id, current_indexes, new_indexes,
              provenance))
      {
        LocalFieldAllocResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(destination);
          rez.serialize<size_t>(new_indexes.size());
          for (unsigned idx = 0; idx < new_indexes.size(); idx++)
            rez.serialize(new_indexes[idx]);
          rez.serialize(done_event);
        }
        rez.dispatch(source);
      }
      else  // if we failed we can just trigger the event
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LocalFieldAllocResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<unsigned>* destination;
      derez.deserialize(destination);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      destination->resize(num_indexes);
      for (unsigned idx = 0; idx < num_indexes; idx++)
        derez.deserialize((*destination)[idx]);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LocalFieldFreeMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      std::vector<unsigned> indexes(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fields[idx]);
        derez.deserialize(indexes[idx]);
      }

      FieldSpaceNode* node = runtime->get_node(handle);
      node->free_local_fields(fields, indexes, nullptr /*no collective*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSizeUpdate::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      FieldID fid;
      derez.deserialize(fid);
      size_t field_size;
      derez.deserialize(field_size);

      FieldSpaceNode* node = runtime->get_node(handle);
      std::set<RtEvent> done_events;
      node->update_field_size(fid, field_size, done_events, source);
      node->unpack_global_ref();
      if (!done_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::DeferRequestFieldInfoArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->request_field_infos_copy(copy, source, to_trigger);
    }

    //--------------------------------------------------------------------------
    InstanceRef FieldSpaceNode::create_external_instance(
        const std::set<FieldID>& priv_fields,
        const std::vector<FieldID>& field_set, RegionNode* node,
        AttachOp* attach_op)
    //--------------------------------------------------------------------------
    {
      legion_assert(node->column_source == this);
      std::vector<size_t> field_sizes(field_set.size());
      std::vector<unsigned> mask_index_map(field_set.size());
      std::vector<CustomSerdezID> serdez(field_set.size());
      FieldMask external_mask;
      compute_field_layout(
          field_set, field_sizes, mask_index_map, serdez, external_mask);
      FieldMask privilege_mask = (priv_fields.size() == field_set.size()) ?
                                     external_mask :
                                     get_field_mask(priv_fields);
      // Now make the instance, this should always succeed
      PhysicalManager* manager = attach_op->create_manager(
          node, field_set, field_sizes, mask_index_map, serdez, external_mask);
      return InstanceRef(manager, privilege_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalCreateRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldSpaceNode* fs = runtime->get_node(handle);
      PhysicalInstance inst;
      derez.deserialize(inst);
      ApEvent ready_event;
      derez.deserialize(ready_event);
      LgEvent unique_event;
      derez.deserialize(unique_event);
      size_t footprint;
      derez.deserialize(footprint);
      LayoutConstraintSet constraints;
      constraints.deserialize(derez);
      FieldMask file_mask;
      derez.deserialize(file_mask);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> field_set(num_fields);
      std::vector<size_t> field_sizes(num_fields);
      std::vector<unsigned> mask_index_map(num_fields);
      std::vector<CustomSerdezID> serdez(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(field_set[idx]);
        derez.deserialize(field_sizes[idx]);
        derez.deserialize(mask_index_map[idx]);
        derez.deserialize(serdez[idx]);
      }
      LogicalRegion region_handle;
      derez.deserialize(region_handle);
      RegionNode* region_node = runtime->get_node(region_handle);
      size_t collective_mapping_size;
      derez.deserialize(collective_mapping_size);
      CollectiveMapping* collective_mapping = nullptr;
      if (collective_mapping_size > 0)
      {
        collective_mapping =
            new CollectiveMapping(derez, collective_mapping_size);
        collective_mapping->add_reference();
      }
      std::atomic<DistributedID>* did_ptr;
      derez.deserialize(did_ptr);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      PhysicalManager* manager = fs->create_external_manager(
          inst, ready_event, footprint, constraints, field_set, field_sizes,
          file_mask, mask_index_map, unique_event, region_node, serdez,
          runtime->get_available_distributed_id(), collective_mapping);

      ExternalCreateRequest rez;
      {
        RezCheck z2(rez);
        rez.serialize(did_ptr);
        rez.serialize(manager->did);
        rez.serialize(done_event);
      }
      rez.dispatch(source);

      if ((collective_mapping != nullptr) &&
          collective_mapping->remove_reference())
        delete collective_mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalCreateResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<DistributedID>* did_ptr;
      derez.deserialize(did_ptr);
      DistributedID did;
      derez.deserialize(did);
      did_ptr->store(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* FieldSpaceNode::create_external_manager(
        PhysicalInstance inst, ApEvent ready_event, size_t instance_footprint,
        LayoutConstraintSet& constraints, const std::vector<FieldID>& field_set,
        const std::vector<size_t>& field_sizes, const FieldMask& external_mask,
        const std::vector<unsigned>& mask_index_map, LgEvent unique_event,
        RegionNode* node, const std::vector<CustomSerdezID>& serdez,
        DistributedID did, CollectiveMapping* collective_mapping)
    //--------------------------------------------------------------------------
    {
      // Pull out the pointer constraint so that we can use it separately
      // and not have it included in the layout constraints
      constraints.pointer_constraint = PointerConstraint();
      const unsigned total_dims = node->row_source->get_num_dims();
      // Get the layout
      LayoutDescription* layout =
          find_layout_description(external_mask, total_dims, constraints);
      if (layout == nullptr)
      {
        LayoutConstraints* layout_constraints =
            runtime->register_layout(handle, constraints, true /*internal*/);
        layout = create_layout_description(
            external_mask, total_dims, layout_constraints, mask_index_map,
            field_set, field_sizes, serdez);
      }
      legion_assert(layout != nullptr);
      MemoryManager* memory = runtime->find_memory_manager(inst.get_location());
      PhysicalManager* result = new PhysicalManager(
          did, memory, inst, node->row_source, nullptr /*piece list*/,
          0 /*piece list size*/, node->column_source,
          node->handle.get_tree_id(), layout, 0 /*redop*/,
          true /*register now*/, instance_footprint, ready_event, unique_event,
          PhysicalManager::EXTERNAL_ATTACHED_INSTANCE_KIND, nullptr /*redop*/,
          collective_mapping);
      // Remove the reference that was returned to us from either finding
      // or creating the layout
      if (layout->remove_reference())
        delete layout;
      legion_assert(result != nullptr);
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::find_layout_description(
        const FieldMask& mask, unsigned num_dims,
        const LayoutConstraintSet& constraints)
    //--------------------------------------------------------------------------
    {
      std::deque<LayoutDescription*> candidates;
      {
        uint64_t hash_key = mask.get_hash_key();
        AutoLock n_lock(node_lock, false /*exclusive*/);
        std::map<LEGION_FIELD_MASK_FIELD_TYPE, lng::list<LayoutDescription*>>::
            const_iterator finder = layouts.find(hash_key);
        if (finder == layouts.end())
          return nullptr;
        // Get the ones with a matching mask
        for (LayoutDescription* layout : finder->second)
        {
          if (layout->total_dims != num_dims)
            continue;
          if (layout->allocated_fields == mask)
            candidates.emplace_back(layout);
        }
      }
      if (candidates.empty())
        return nullptr;
      // First go through the existing descriptions and see if we find
      // one that matches the existing layout
      for (LayoutDescription* layout : candidates)
      {
        if (layout->match_layout(constraints, num_dims))
        {
          layout->add_reference();
          return layout;
        }
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::find_layout_description(
        const FieldMask& mask, LayoutConstraints* constraints)
    //--------------------------------------------------------------------------
    {
      // This one better work
      uint64_t hash_key = mask.get_hash_key();
      AutoLock n_lock(node_lock, false /*exclusive*/);
      std::map<LEGION_FIELD_MASK_FIELD_TYPE, lng::list<LayoutDescription*>>::
          const_iterator finder = layouts.find(hash_key);
      legion_assert(finder != layouts.end());
      for (LayoutDescription* layout : finder->second)
      {
        if (layout->constraints != constraints)
          continue;
        if (layout->allocated_fields != mask)
          continue;
        layout->add_reference();
        return layout;
      }
      std::abort();
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::create_layout_description(
        const FieldMask& layout_mask, const unsigned total_dims,
        LayoutConstraints* constraints,
        const std::vector<unsigned>& mask_index_map,
        const std::vector<FieldID>& fids,
        const std::vector<size_t>& field_sizes,
        const std::vector<CustomSerdezID>& serdez)
    //--------------------------------------------------------------------------
    {
      // Make the new field description and then register it
      LayoutDescription* result = new LayoutDescription(
          this, layout_mask, total_dims, constraints, mask_index_map, fids,
          field_sizes, serdez);
      result->add_reference();
      return register_layout_description(result);
    }

    //--------------------------------------------------------------------------
    LayoutDescription* FieldSpaceNode::register_layout_description(
        LayoutDescription* layout)
    //--------------------------------------------------------------------------
    {
      uint64_t hash_key = layout->allocated_fields.get_hash_key();
      AutoLock n_lock(node_lock);
      lng::list<LayoutDescription*>& descs = layouts[hash_key];
      if (!descs.empty())
      {
        for (lng::list<LayoutDescription*>::const_iterator it = descs.begin();
             it != descs.end(); it++)
        {
          if (layout->match_layout(*it, layout->total_dims))
          {
            // Delete the layout we are trying to register
            // and return the matching one
            if (layout->remove_reference())
              delete layout;
            (*it)->add_reference();
            return (*it);
          }
        }
      }
      // Otherwise we successfully registered it
      descs.emplace_back(layout);
      // Add the reference here for our local data structure
      layout->add_reference();
      return layout;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::send_node(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Only send it if we're the owner without a collective mapping
      // or the target is not in the collective mapping and we're the
      // closest node in the collective mapping to the target
      legion_assert(
          (is_owner() && (collective_mapping == nullptr)) ||
          ((collective_mapping != nullptr) &&
           !collective_mapping->contains(target) &&
           collective_mapping->contains(local_space) &&
           (local_space == collective_mapping->find_nearest(target))));
      // See if this is in our creation set, if not, send it and all the fields
      AutoLock n_lock(node_lock);
      if (!has_remote_instance(target))
      {
        // First send the node info and then send all the fields
        FieldSpaceNodeMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(handle);
          rez.serialize(initialized);
          if (provenance != nullptr)
            provenance->serialize(rez);
          else
            Provenance::serialize_null(rez);
          if (collective_mapping != nullptr)
            collective_mapping->pack(rez);
          else
            CollectiveMapping::pack_null(rez);
          // Pack the field infos
          if (allocation_state == FIELD_ALLOC_READ_ONLY)
          {
            size_t num_fields = field_infos.size();
            rez.serialize<size_t>(num_fields);
            for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
            {
              rez.serialize(it.first);
              it.second.serialize(rez);
            }
            remote_field_infos.insert(target);
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize<size_t>(semantic_info.size());
          for (lng::map<SemanticTag, SemanticInfo>::iterator it =
                   semantic_info.begin();
               it != semantic_info.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize<size_t>(it->second.buffer.get_size());
            rez.serialize(
                it->second.buffer.get_buffer(), it->second.buffer.get_size());
            rez.serialize(it->second.is_mutable);
          }
          rez.serialize<size_t>(semantic_field_info.size());
          for (lng::map<std::pair<FieldID, SemanticTag>, SemanticInfo>::iterator
                   it = semantic_field_info.begin();
               it != semantic_field_info.end(); it++)
          {
            rez.serialize(it->first.first);
            rez.serialize(it->first.second);
            rez.serialize(it->second.buffer.get_size());
            rez.serialize(
                it->second.buffer.get_buffer(), it->second.buffer.get_size());
            rez.serialize(it->second.is_mutable);
          }
        }
        rez.dispatch(target);
        // Finally add it to the creation set
        update_remote_instances(target);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceNodeMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent initialized;
      derez.deserialize(initialized);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
        mapping = new CollectiveMapping(derez, num_spaces);
      FieldSpaceNode* node =
          runtime->create_node(handle, initialized, provenance, mapping, derez);
      legion_assert(node != nullptr);
      size_t num_semantic;
      derez.deserialize(num_semantic);
      for (unsigned idx = 0; idx < num_semantic; idx++)
      {
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void* buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(
            tag, source, buffer, buffer_size, is_mutable, false /*local only*/);
      }
      size_t num_field_semantic;
      derez.deserialize(num_field_semantic);
      for (unsigned idx = 0; idx < num_field_semantic; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        SemanticTag tag;
        derez.deserialize(tag);
        size_t buffer_size;
        derez.deserialize(buffer_size);
        const void* buffer = derez.get_current_pointer();
        derez.advance_pointer(buffer_size);
        bool is_mutable;
        derez.deserialize(is_mutable);
        node->attach_semantic_information(
            fid, tag, source, buffer, buffer_size, is_mutable,
            false /*local only*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      AddressSpaceID source;
      derez.deserialize(source);
      FieldSpaceNode* target =
          runtime->get_node(handle, nullptr, true /*can fail*/);
      if (target == nullptr)
      {
        Runtime::trigger_event(to_trigger);
        return;
      }
      // If there is a collective mapping, check to see if we're on the
      // right node and if not forward it on to the right node
      if (target->collective_mapping != nullptr)
      {
        legion_assert(!target->collective_mapping->contains(source));
        legion_assert(
            target->collective_mapping->contains(target->local_space));
        if (target->is_owner())
        {
          const AddressSpaceID nearest =
              target->collective_mapping->find_nearest(source);
          // If we're not the nearest then forward it on to the
          // proper node to handle the request
          if (nearest != target->local_space)
          {
            FieldSpaceRequest rez;
            rez.serialize(handle);
            rez.serialize(to_trigger);
            rez.serialize(source);
            rez.dispatch(nearest);
            return;
          }
        }
        else
        {
          legion_assert(
              target->local_space ==
              target->collective_mapping->find_nearest(source));
        }
      }
      target->send_node(source);
      FieldSpaceReturn rez;
      rez.serialize(to_trigger);
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceReturn::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceAllocatorRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);

      FieldSpaceNode* node = runtime->get_node(handle);

      legion_assert(node->is_owner());
      node->create_allocator(source, ready_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceAllocatorResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent invalidations_done;
      derez.deserialize(invalidations_done);

      FieldSpaceNode* node = runtime->get_node(handle);
      // wait for the invalidations to be done before handling ourselves
      if (invalidations_done.exists() && !invalidations_done.has_triggered())
        invalidations_done.wait();
      node->process_allocator_response(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceAllocatorInvalidation::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool flush_allocation;
      derez.deserialize(flush_allocation);
      bool merge;
      derez.deserialize(merge);

      FieldSpaceNode* node = runtime->get_node(handle);
      node->process_allocator_invalidation(done_event, flush_allocation, merge);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceAllocatorFlush::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);

      FieldSpaceNode* node = runtime->get_node(handle);
      const bool remove_free_reference = node->process_allocator_flush(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      if (node->remove_base_gc_ref(
              FIELD_ALLOCATOR_REF, (remove_free_reference ? 2 : 1)))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceAllocatorFree::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);

      FieldSpaceNode* node = runtime->get_node(handle);
      node->process_allocator_free(derez, source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      // Remove the reference that we added when we originally got the request
      if (node->remove_base_gc_ref(FIELD_ALLOCATOR_REF))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceInfosRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      std::map<FieldID, FieldSpaceNode::FieldInfo>* target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      legion_assert(to_trigger.exists());
      FieldSpaceNode* node = runtime->get_node(handle);
      node->request_field_infos_copy(target, source, to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceInfosResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::map<FieldID, FieldSpaceNode::FieldInfo>* target;
      derez.deserialize(target);
      size_t num_infos;
      derez.deserialize(num_infos);
      for (unsigned idx = 0; idx < num_infos; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        (*target)[fid].deserialize(derez);
      }
      FieldSpace handle;
      derez.deserialize(handle);
      if (handle.exists())
      {
        FieldSpaceNode* node = runtime->get_node(handle);
        node->record_read_only_infos(*target);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      legion_assert(to_trigger.exists());
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    char* FieldSpaceNode::to_string(
        const FieldMask& mask, TaskContext* ctx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!!mask);
      std::string result;
      std::set<unsigned> local_indexes;
      bool invalid = false;
      size_t count = 0;  // used to skip leading comma
      {
        AutoLock n_lock(node_lock, false /*exclusive*/);
        while (allocation_state == FIELD_ALLOC_PENDING)
        {
          legion_assert(is_owner());
          const RtEvent wait_on = pending_field_allocation;
          n_lock.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          n_lock.reacquire();
        }
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
          {
            if (mask.is_set(it.second.idx))
            {
              if (!it.second.local)
              {
                if (count++)
                  result += ',';
                char temp[32];
                snprintf(temp, 32, "%d", it.first);
                result += temp;
              }
              else
                local_indexes.insert(it.second.idx);
            }
          }
        }
        else
          invalid = true;
      }
      if (invalid)
      {
        std::map<FieldID, FieldInfo> local_infos;
        const RtEvent ready =
            request_field_infos_copy(&local_infos, local_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        for (const std::pair<const FieldID, FieldInfo>& it : local_infos)
        {
          if (mask.is_set(it.second.idx))
          {
            if (!it.second.local)
            {
              if (count++)
                result += ',';
              char temp[32];
              snprintf(temp, 32, "%d", it.first);
              result += temp;
            }
            else
              local_indexes.insert(it.second.idx);
          }
        }
      }
      if (!local_indexes.empty())
      {
        std::vector<FieldID> local_fields;
        ctx->get_local_field_set(handle, local_indexes, local_fields);
        for (const FieldID& it : local_fields)
        {
          if (count++)
            result += ',';
          char temp[32];
          snprintf(temp, 32, "%d", it);
          result += temp;
        }
      }
      return strdup(result.c_str());
    }

    //--------------------------------------------------------------------------
    int FieldSpaceNode::allocate_index(RtEvent& ready_event, bool initializing)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
          (allocation_state == FIELD_ALLOC_COLLECTIVE) || initializing);
      // Check to see if we still have spots
      int result = unallocated_indexes.find_first_set();
      if ((result >= 0) &&
          (result < int(LEGION_MAX_FIELDS - runtime->max_local_fields)))
      {
        // We still have unallocated indexes so use those first
        unallocated_indexes.unset_bit(result);
        return result;
      }
      // If there are no available indexes then we are done
      if (available_indexes.empty())
        return -1;
      std::list<std::pair<unsigned, RtEvent>>::iterator backup =
          available_indexes.end();
      for (std::list<std::pair<unsigned, RtEvent>>::iterator it =
               available_indexes.begin();
           it != available_indexes.end(); it++)
      {
        if (!it->second.exists() || it->second.has_triggered())
        {
          // Found one without an event precondition so use it
          result = it->first;
          available_indexes.erase(it);
          return result;
        }
        else if (backup == available_indexes.end())
        {
          // If we haven't recorded a back-up then this is the
          // first once we've found so record it
          backup = it;
        }
      }
      // We didn't find one without a precondition, see if we got a backup
      if (backup != available_indexes.end())
      {
        result = backup->first;
        available_indexes.erase(backup);
        return result;
      }
      // Didn't find anything
      return -1;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::free_index(unsigned index, RtEvent ready_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
          (allocation_state == FIELD_ALLOC_COLLECTIVE));
      // Perform the invalidations across all nodes too
      std::set<RtEvent> invalidation_events;
      invalidate_layouts(
          index, invalidation_events, runtime->address_space,
          false /*need lock*/);
      if (!invalidation_events.empty())
      {
        if (ready_event.exists())
          invalidation_events.insert(ready_event);
        ready_event = Runtime::merge_events(invalidation_events);
      }
      // Record this as an available index
      available_indexes.emplace_back(
          std::pair<unsigned, RtEvent>(index, ready_event));
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::invalidate_layouts(
        unsigned index, std::set<RtEvent>& applied, AddressSpaceID source,
        bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock n_lock(node_lock);
        invalidate_layouts(index, applied, source, false /*need lock*/);
        return;
      }
      // Send messages to any remote nodes to perform the invalidation
      // We're already holding the lock
      if (has_remote_instances())
      {
        std::deque<AddressSpaceID> targets;
        FindTargetsFunctor functor(targets);
        map_over_remote_instances(functor);
        std::set<RtEvent> remote_ready;
        for (const AddressSpaceID& target : targets)
        {
          if (target == source)
            continue;
          RtUserEvent remote_done = Runtime::create_rt_user_event();
          FieldSpaceLayoutInvalidation rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(index);
            rez.serialize(remote_done);
          }
          pack_global_ref();
          rez.dispatch(target);
          applied.insert(remote_done);
        }
      }
      std::vector<LEGION_FIELD_MASK_FIELD_TYPE> to_delete;
      for (std::pair<
               const LEGION_FIELD_MASK_FIELD_TYPE,
               lng::list<LayoutDescription*>>& lit : layouts)
      {
        // If the bit is set, remove the layout descriptions
        if (lit.first & (1ULL << index))
        {
          lng::list<LayoutDescription*>& descs = lit.second;
          bool perform_delete = true;
          for (lng::list<LayoutDescription*>::iterator it = descs.begin();
               it != descs.end();
               /*nothing*/)
          {
            if ((*it)->allocated_fields.is_set(index))
            {
              if ((*it)->remove_reference())
                delete (*it);
              it = descs.erase(it);
            }
            else
            {
              it++;
              perform_delete = false;
            }
          }
          if (perform_delete)
            to_delete.emplace_back(lit.first);
        }
      }
      for (const LEGION_FIELD_MASK_FIELD_TYPE& it : to_delete)
        layouts.erase(it);
    }

    //--------------------------------------------------------------------------
    RtEvent FieldSpaceNode::request_field_infos_copy(
        std::map<FieldID, FieldInfo>* copy, AddressSpaceID source,
        RtUserEvent to_trigger) const
    //--------------------------------------------------------------------------
    {
      legion_assert(copy != nullptr);
      if (is_owner())
      {
        RtEvent wait_on;
        // May need to iterate this in the case of allocation pending
        while (true)
        {
          if (wait_on.exists() && !wait_on.has_triggered())
          {
            if (source != local_space)
            {
              // Need to defer this to avoid blocking the virtual channel
              legion_assert(to_trigger.exists());
              DeferRequestFieldInfoArgs args(this, copy, source, to_trigger);
              runtime->issue_runtime_meta_task(
                  args, LG_LATENCY_DEFERRED_PRIORITY, wait_on);
              return to_trigger;
            }
            else
              wait_on.wait();
          }
          AutoLock n_lock(node_lock);
          if (allocation_state == FIELD_ALLOC_INVALID)
          {
            // If we're invalid, that means there should be exactly
            // one remote copy which is where the allocation privileges are
            legion_assert(remote_field_infos.size() == 1);
            // forward this message onto the node with the privileges
            const AddressSpaceID target = *(remote_field_infos.begin());
            if (!to_trigger.exists())
              to_trigger = Runtime::create_rt_user_event();
            FieldSpaceInfosRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(handle);
              rez.serialize(copy);
              rez.serialize(source);
              rez.serialize(to_trigger);
            }
            rez.dispatch(target);
          }
          else if (allocation_state == FIELD_ALLOC_READ_ONLY)
          {
            // We can send back a response, make them a reader if they
            // are not one already
            if (source != local_space)
            {
              legion_assert(to_trigger.exists());
              FieldSpaceInfosResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(copy);
                rez.serialize<size_t>(field_infos.size());
                for (const std::pair<const FieldID, FieldInfo>& it :
                     field_infos)
                {
                  rez.serialize(it.first);
                  it.second.serialize(rez);
                }
                std::set<AddressSpaceID>::const_iterator finder =
                    remote_field_infos.find(source);
                if (finder == remote_field_infos.end())
                {
                  rez.serialize(handle);
                  remote_field_infos.insert(source);
                }
                else
                  rez.serialize(FieldSpace::NO_SPACE);
                rez.serialize(to_trigger);
              }
              rez.dispatch(source);
            }
            else
            {
              *copy = field_infos;
              if (to_trigger.exists())
                Runtime::trigger_event(to_trigger);
            }
          }
          else if (allocation_state == FIELD_ALLOC_PENDING)
          {
            wait_on = pending_field_allocation;
            continue;
          }
          else
          {
            // If we have allocation privileges we can send the response
            // but we can't make them a read-only copy
            if (source != local_space)
            {
              legion_assert(to_trigger.exists());
              FieldSpaceInfosResponse rez;
              {
                RezCheck z(rez);
                rez.serialize(copy);
                rez.serialize<size_t>(field_infos.size());
                for (const std::pair<const FieldID, FieldInfo>& it :
                     field_infos)
                {
                  rez.serialize(it.first);
                  it.second.serialize(rez);
                }
                rez.serialize(FieldSpace::NO_SPACE);
                rez.serialize(to_trigger);
              }
              rez.dispatch(source);
            }
            else
            {
              *copy = field_infos;
              if (to_trigger.exists())
                Runtime::trigger_event(to_trigger);
            }
          }
          // Always break out if we make it here
          break;
        }
      }
      else
      {
        // Not the owner
        AutoLock n_lock(node_lock, false /*exclusive*/);
        // check to see if we lost the race
        if (allocation_state != FIELD_ALLOC_INVALID)
        {
          if (source != local_space)
          {
            legion_assert(to_trigger.exists());
            // Send the response back to the source
            FieldSpaceInfosResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(copy);
              rez.serialize<size_t>(field_infos.size());
              for (const std::pair<const FieldID, FieldInfo>& it : field_infos)
              {
                rez.serialize(it.first);
                it.second.serialize(rez);
              }
              // We can't give them read-only privileges
              rez.serialize(FieldSpace::NO_SPACE);
              rez.serialize(to_trigger);
            }
            rez.dispatch(source);
          }
          else
          {
            *copy = field_infos;
            if (to_trigger.exists())
              Runtime::trigger_event(to_trigger);
          }
        }
        else
        {
          // Did not lose the race, send the request back to the owner
          if (!to_trigger.exists())
            to_trigger = Runtime::create_rt_user_event();
          FieldSpaceInfosRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(handle);
            rez.serialize(copy);
            rez.serialize(source);
            rez.serialize(to_trigger);
          }
          rez.dispatch(owner_space);
        }
      }
      return to_trigger;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::record_read_only_infos(
        const std::map<FieldID, FieldInfo>& infos)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      AutoLock n_lock(node_lock);
      legion_assert(allocation_state == FIELD_ALLOC_INVALID);
      field_infos.insert(infos.begin(), infos.end());
      allocation_state = FIELD_ALLOC_READ_ONLY;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      legion_assert(!is_owner());
      legion_assert(
          (allocation_state == FIELD_ALLOC_INVALID) ||
          (allocation_state == FIELD_ALLOC_READ_ONLY) ||
          (allocation_state == FIELD_ALLOC_COLLECTIVE));
      legion_assert(outstanding_allocators == 0);
      if (allocation_state == FIELD_ALLOC_INVALID)
      {
        size_t num_infos;
        derez.deserialize(num_infos);
        for (unsigned idx = 0; idx < num_infos; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
      }
      if (allocation_state != FIELD_ALLOC_COLLECTIVE)
      {
        legion_assert(!unallocated_indexes);
        legion_assert(available_indexes.empty());
        derez.deserialize(unallocated_indexes);
        size_t num_indexes;
        derez.deserialize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          std::pair<unsigned, RtEvent> index;
          derez.deserialize(index.first);
          derez.deserialize(index.second);
          available_indexes.emplace_back(index);
        }
      }
      // Make that we now have this in exclusive mode
      outstanding_allocators = 1;
      allocation_state = FIELD_ALLOC_EXCLUSIVE;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_invalidation(
        RtUserEvent done_event, bool flush_allocation, bool need_merge)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      legion_assert(!is_owner());
      legion_assert(
          (allocation_state == FIELD_ALLOC_EXCLUSIVE) ||
          (allocation_state == FIELD_ALLOC_COLLECTIVE) ||
          (allocation_state == FIELD_ALLOC_READ_ONLY));
      FieldSpaceAllocatorFlush rez;
      // It's possible to be in the read-only state even with a flush because
      // of ships passing in the night. We get sent an invalidation, but we
      // already released our allocator and sent it back to the owner so we are
      // in the read-only state and the messages pass like ships in the night
      if (flush_allocation && (allocation_state != FIELD_ALLOC_READ_ONLY))
      {
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize<bool>(true);  // allocation meta data
        rez.serialize<bool>(need_merge);
        rez.serialize(field_infos.size());
        for (std::map<FieldID, FieldInfo>::iterator it = field_infos.begin();
             it != field_infos.end();
             /*nothing*/)
        {
          rez.serialize(it->first);
          it->second.serialize(rez);
          if (!it->second.local)
          {
            std::map<FieldID, FieldInfo>::iterator to_delete = it++;
            field_infos.erase(to_delete);
          }
          else
            it++;
        }
        rez.serialize(unallocated_indexes);
        unallocated_indexes.clear();
        rez.serialize(available_indexes.size());
        while (!available_indexes.empty())
        {
          std::pair<unsigned, RtEvent>& front = available_indexes.front();
          rez.serialize(front.first);
          rez.serialize(front.second);
          available_indexes.pop_front();
        }
        rez.serialize(outstanding_allocators);
        outstanding_allocators = 0;
        rez.serialize(done_event);
      }
      else
      {
        legion_assert(
            (allocation_state == FIELD_ALLOC_READ_ONLY) ||
            (allocation_state == FIELD_ALLOC_COLLECTIVE));
        RezCheck z(rez);
        rez.serialize(handle);
        rez.serialize<bool>(false);  // allocation meta data
        // Invalidate our field infos
        for (std::map<FieldID, FieldInfo>::iterator it = field_infos.begin();
             it != field_infos.end();
             /*nothing*/)
        {
          if (!it->second.local)
          {
            std::map<FieldID, FieldInfo>::iterator to_delete = it++;
            field_infos.erase(to_delete);
          }
          else
            it++;
        }
        rez.serialize(done_event);
      }
      rez.dispatch(owner_space);
      // back to the invalid state
      allocation_state = FIELD_ALLOC_INVALID;
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::process_allocator_flush(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      bool allocator_meta_data;
      derez.deserialize(allocator_meta_data);
      AutoLock n_lock(node_lock);
      if (allocator_meta_data)
      {
        bool need_merge;
        derez.deserialize(need_merge);
        if (need_merge)
        {
          size_t num_infos;
          derez.deserialize(num_infos);
          for (unsigned idx = 0; idx < num_infos; idx++)
          {
            FieldID fid;
            derez.deserialize(fid);
            field_infos[fid].deserialize(derez);
          }
          FieldMask unallocated;
          derez.deserialize(unallocated);
          unallocated_indexes |= unallocated;
          size_t num_available;
          derez.deserialize(num_available);
          for (unsigned idx = 0; idx < num_available; idx++)
          {
            std::pair<unsigned, RtEvent> next;
            derez.deserialize(next.first);
            derez.deserialize(next.second);
            bool found = false;
            for (const std::pair<unsigned, RtEvent>& it : available_indexes)
            {
              if (it.first != next.first)
                continue;
              found = true;
              break;
            }
            if (!found)
              available_indexes.emplace_back(next);
          }
          derez.advance_pointer(sizeof(outstanding_allocators));
        }
        else
        {
          size_t num_infos;
          derez.deserialize(num_infos);
          for (unsigned idx = 0; idx < num_infos; idx++)
          {
            FieldID fid;
            derez.deserialize(fid);
            field_infos[fid].deserialize(derez);
          }
          legion_assert(!unallocated_indexes);
          legion_assert(available_indexes.empty());
          derez.deserialize(unallocated_indexes);
          size_t num_available;
          derez.deserialize(num_available);
          for (unsigned idx = 0; idx < num_available; idx++)
          {
            std::pair<unsigned, RtEvent> next;
            derez.deserialize(next.first);
            derez.deserialize(next.second);
            available_indexes.emplace_back(next);
          }
          unsigned remote_allocators;
          derez.deserialize(remote_allocators);
          outstanding_allocators += remote_allocators;
        }
      }
      legion_assert(outstanding_invalidations > 0);
      legion_assert(
          (allocation_state == FIELD_ALLOC_PENDING) ||
          (allocation_state == FIELD_ALLOC_INVALID));
      if ((--outstanding_invalidations == 0) &&
          (allocation_state == FIELD_ALLOC_PENDING))
        allocation_state = FIELD_ALLOC_EXCLUSIVE;
      return allocator_meta_data;
    }

    //--------------------------------------------------------------------------
    void FieldSpaceNode::process_allocator_free(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      bool return_allocation;
      derez.deserialize(return_allocation);
      if (return_allocation)
      {
        AutoLock n_lock(node_lock);
        legion_assert(
            (allocation_state == FIELD_ALLOC_INVALID) ||
            (allocation_state == FIELD_ALLOC_PENDING));
        if (allocation_state == FIELD_ALLOC_INVALID)
        {
          legion_assert(remote_field_infos.size() == 1);
          legion_assert(
              remote_field_infos.find(source) != remote_field_infos.end());
          legion_assert(outstanding_allocators == 0);
        }
        legion_assert(!unallocated_indexes);
        legion_assert(available_indexes.empty());
        size_t num_infos;
        derez.deserialize(num_infos);
        for (unsigned idx = 0; idx < num_infos; idx++)
        {
          FieldID fid;
          derez.deserialize(fid);
          field_infos[fid].deserialize(derez);
        }
        derez.deserialize(unallocated_indexes);
        size_t num_indexes;
        derez.deserialize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          std::pair<unsigned, RtEvent> next;
          derez.deserialize(next.first);
          derez.deserialize(next.second);
          available_indexes.emplace_back(next);
        }
        if (allocation_state == FIELD_ALLOC_INVALID)
          allocation_state = FIELD_ALLOC_READ_ONLY;
      }
      else
        destroy_allocator(source);
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceNode::allocate_local_indexes(
        CustomSerdezID serdez, const std::vector<size_t>& sizes,
        const std::set<unsigned>& current_indexes,
        std::vector<unsigned>& new_indexes)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      new_indexes.resize(sizes.size());
      // Iterate over the different fields to allocate and try to find
      // an index for them in our list of local fields
      for (unsigned fidx = 0; fidx < sizes.size(); fidx++)
      {
        const size_t field_size = sizes[fidx];
        int chosen_index = -1;
        unsigned global_idx = LEGION_MAX_FIELDS - runtime->max_local_fields;
        for (unsigned local_idx = 0; local_idx < local_index_infos.size();
             local_idx++, global_idx++)
        {
          // If it's already been allocated in this context then
          // we can't use it
          if (current_indexes.find(global_idx) != current_indexes.end())
            continue;
          // Check if the current local field index is used
          if (local_index_infos[local_idx].first > 0)
          {
            // Already in use, check to see if the field sizes are the same
            if ((local_index_infos[local_idx].first == field_size) &&
                (local_index_infos[local_idx].second == serdez))
            {
              // Same size so we can use it
              chosen_index = global_idx;
              break;
            }
            // Else different field size means we can't reuse it
          }
          else
          {
            // Not in use, so we can assign the size and make
            // ourselves the first user
            local_index_infos[local_idx] =
                std::pair<size_t, CustomSerdezID>(field_size, serdez);
            chosen_index = global_idx;
            break;
          }
        }
        // If we didn't pick a valid index then we failed
        if (chosen_index < 0)
          return false;
        // Save the result
        new_indexes[fidx] = chosen_index;
      }
      return true;
    }

  }  // namespace Internal
}  // namespace Legion
