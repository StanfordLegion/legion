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

#include "legion/api/exception.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/managers/message.h"
#include "legion/operations/allreduce.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Future
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  Future::Future(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Future::Future(const Future& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_base_gc_ref(Internal::APPLICATION_REF);
  }

  //--------------------------------------------------------------------------
  Future::Future(Future&& rhs) noexcept : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    rhs.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  Future::~Future(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) &&
        impl->remove_base_gc_ref(Internal::APPLICATION_REF))
      delete impl;
  }

  //--------------------------------------------------------------------------
  Future::Future(Internal::FutureImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_base_gc_ref(Internal::APPLICATION_REF);
  }

  //--------------------------------------------------------------------------
  Future& Future::operator=(const Future& rhs)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) &&
        impl->remove_base_gc_ref(Internal::APPLICATION_REF))
      delete impl;
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_base_gc_ref(Internal::APPLICATION_REF);
    return *this;
  }

  //--------------------------------------------------------------------------
  Future& Future::operator=(Future&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) &&
        impl->remove_base_gc_ref(Internal::APPLICATION_REF))
      delete impl;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //--------------------------------------------------------------------------
  std::size_t Future::hash(void) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return std::hash<DistributedID>{}(impl->did);
    else
      return std::hash<DistributedID>{}(0);
  }

  //--------------------------------------------------------------------------
  void Future::get_void_result(
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->wait(silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  bool Future::is_empty(
      bool block /*= true*/, bool silence_warnings /*=false*/,
      const char* warning_string /*=nullptr*/) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return impl->is_empty(block, silence_warnings, warning_string);
    return true;
  }

  //--------------------------------------------------------------------------
  bool Future::is_ready(bool subscribe) const
  //--------------------------------------------------------------------------
  {
    if ((impl == nullptr) || (Internal::implicit_context != impl->context))
      return true;
    return impl->is_ready(subscribe);
  }

  //--------------------------------------------------------------------------
  const void* Future::get_buffer(
      Memory::Kind memory, size_t* extent_in_bytes, bool check_size,
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal invocation of Future::get_buffer on an null future";
      error.raise();
    }
    if (Internal::implicit_context == nullptr)
      return impl->get_buffer(
          Processor::NO_PROC, memory, extent_in_bytes, check_size,
          silence_warnings, warning_string);
    else
      return impl->get_buffer(
          Internal::implicit_context->get_executing_processor(), memory,
          extent_in_bytes, check_size, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  void Future::get_memories(
      std::set<Memory>& memories, bool silence_warnings,
      const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal invocation of Future::get_memories on an null future";
      error.raise();
    }
    impl->get_memories(memories, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  size_t Future::get_untyped_size(void) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error
          << "Illegal invocation of Future::get_untyped_size on an null future";
      error.raise();
    }
    return impl->get_untyped_size();
  }

  //--------------------------------------------------------------------------
  const void* Future::get_metadata(size_t* size) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal invocation of Future::get_metadata on an null future";
      error.raise();
    }
    return impl->get_metadata(size);
  }

  //--------------------------------------------------------------------------
  Realm::RegionInstance Future::get_instance(
      Memory::Kind memkind, size_t field_size, bool check_field_size,
      const char* warning_string, bool silence_warnings) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Illegal invocation of Future::get_instance on an null future";
      error.raise();
    }
    return impl->get_instance(
        memkind, field_size, check_field_size, silence_warnings,
        warning_string);
  }

  //--------------------------------------------------------------------------
  void Future::report_incompatible_accessor(
      const char* accessor_kind, Realm::RegionInstance instance) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->report_incompatible_accessor(accessor_kind, instance);
  }

  //--------------------------------------------------------------------------
  /*static*/ void SparsityReferenceHelper::deletion_function(
      const Realm::ExternalInstanceResource& r)
  //--------------------------------------------------------------------------
  {
    const Realm::ExternalMemoryResource* resource =
        static_cast<const Realm::ExternalMemoryResource*>(&r);
    legion_assert(resource->size_in_bytes == sizeof(Domain));
    Domain* domain = reinterpret_cast<Domain*>(resource->base);
    domain->destroy();
    delete domain;
  }

  //--------------------------------------------------------------------------
  /*static*/ Future Future::from_domain(
      const Domain& d, bool take_ownership, const char* provenance,
      bool shard_local)
  //--------------------------------------------------------------------------
  {
    if (!d.dense())
    {
      if (!take_ownership)
      {
        SparsityReferenceHelper functor(d);
        Internal::NT_TemplateHelper::demux<SparsityReferenceHelper>(
            d.is_type, &functor);
      }
      Domain* domain = new Domain(d);
      Realm::ExternalMemoryResource resource(domain, sizeof(d));
      return Future::from_value(
          domain, sizeof(d), true /*owned*/, resource,
          SparsityReferenceHelper::deletion_function, provenance, shard_local);
    }
    else
      return from_untyped_pointer(
          &d, sizeof(d), false /*take ownership*/, provenance, shard_local);
  }

  //--------------------------------------------------------------------------
  /*static*/ Future Future::from_untyped_pointer(
      Runtime* rt, const void* value, size_t value_size, bool owned)
  //--------------------------------------------------------------------------
  {
    if (Internal::implicit_context == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Creating a Future from a buffer is only permitted to be "
            << "performed inside of Legion tasks.";
      error.raise();
    }
    return Internal::implicit_context->from_value(
        value, value_size, owned, nullptr /*provenance*/,
        false /*shard local*/);
  }

  //--------------------------------------------------------------------------
  /*static*/ Future Future::from_untyped_pointer(
      const void* value, size_t value_size, bool owned, const char* prov,
      bool shard_local)
  //--------------------------------------------------------------------------
  {
    if (Internal::implicit_context == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Creating a Future from a buffer is only permitted to be "
            << "performed inside of Legion tasks.";
      error.raise();
    }
    Internal::AutoProvenance provenance(prov);
    return Internal::implicit_context->from_value(
        value, value_size, owned, provenance, shard_local);
  }

  //--------------------------------------------------------------------------
  /*static*/ Future Future::from_value(
      const void* buffer, size_t size, bool owned,
      const Realm::ExternalInstanceResource& resource,
      void (*freefunc)(const Realm::ExternalInstanceResource&),
      const char* prov, bool shard_local)
  //--------------------------------------------------------------------------
  {
    if (Internal::implicit_context == nullptr)
    {
      Error error(LEGION_INTERFACE_EXCEPTION);
      error << "Creating a Future from a buffer is only permitted to be "
            << "performed inside of Legion tasks.";
      error.raise();
    }
    Internal::AutoProvenance provenance(prov);
    return Internal::implicit_context->from_value(
        buffer, size, owned, resource, freefunc, provenance, shard_local);
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Future Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(
        TaskContext* ctx, bool register_now, DistributedID did,
        Provenance* prov, Operation* o /*= nullptr*/)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_DC), register_now),
        context(ctx), producer_op(o),
        op_gen((o == nullptr) ? 0 : o->get_generation()),
        producer_depth((o == nullptr) ? -1 : o->get_context()->get_depth()),
        producer_uid((o == nullptr) ? 0 : o->get_unique_op_id()),
        coordinate(
            (o == nullptr) ?
                ContextCoordinate(InnerContext::NO_BLOCKING_INDEX) :
                ContextCoordinate(o->get_context()->get_next_blocking_index())),
        provenance(prov), local_visible_memory(Memory::NO_MEMORY),
        metadata(nullptr), metasize(0), future_size(0),
        upper_bound_size(SIZE_MAX), callback_functor(nullptr),
        own_callback_functor(false), future_size_set(false)
    //--------------------------------------------------------------------------
    {
      empty.store(true);
      sampled.store(false);
      if (producer_op != nullptr)
        producer_op->add_mapping_reference(op_gen);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Future %lld %d", LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(
        TaskContext* ctx, bool register_now, DistributedID did, Operation* o,
        GenerationID gen, const ContextCoordinate& coord, UniqueID uid,
        int depth, Provenance* prov, CollectiveMapping* map)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_DC), register_now, map),
        context(ctx), producer_op(o), op_gen(gen), producer_depth(depth),
        producer_uid(uid), coordinate(coord), provenance(prov),
        local_visible_memory(Memory::NO_MEMORY), metadata(nullptr), metasize(0),
        future_size(0), upper_bound_size(SIZE_MAX), callback_functor(nullptr),
        own_callback_functor(false), future_size_set(false)
    //--------------------------------------------------------------------------
    {
      empty.store(true);
      sampled.store(false);
      if (producer_op != nullptr)
        producer_op->add_mapping_reference(op_gen);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Future %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!subscription_event.exists());
      for (std::pair<const Memory, FutureInstanceTracker>& it : instances)
      {
        if (it.second.remote_postcondition.exists())
        {
          // This is a remote instance that we unpacked and nobody
          // used it so we can just clean it up since we don't own it
          legion_assert(it.second.read_events.empty());
          Runtime::trigger_event_untraced(
              it.second.remote_postcondition, it.second.ready_event);
          delete it.second.instance;
        }
        else
        {
          // Merge together all the events for destroying this future instance
          ApEvent precondition = it.second.ready_event;
          if (!it.second.read_events.empty())
          {
            if (precondition.exists())
              it.second.read_events.emplace_back(precondition);
            precondition =
                Runtime::merge_events(nullptr, it.second.read_events);
          }
          if (!it.second.instance->defer_deletion(precondition))
            delete it.second.instance;
        }
      }
      if (producer_op != nullptr)
        producer_op->remove_mapping_reference(op_gen);
      if (callback_functor != nullptr)
      {
        // Dispatch the deletion functionon the callback processor
        CallbackReleaseArgs args(callback_functor, own_callback_functor);
        runtime->issue_application_processor_task(
            args, LG_LATENCY_WORK_PRIORITY, callback_proc);
      }
      if (metadata != nullptr)
        free(metadata);
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // This future is leaking, so just force delete all our instances
      AutoLock f_lock(future_lock);
      for (std::pair<const Memory, FutureInstanceTracker>& it : instances)
      {
        // Merge together all the events for destroying this future instance
        ApEvent precondition = it.second.ready_event;
        if (!it.second.read_events.empty())
        {
          if (precondition.exists())
            it.second.read_events.emplace_back(precondition);
          precondition = Runtime::merge_events(nullptr, it.second.read_events);
        }
        if (!it.second.instance->defer_deletion(precondition))
          delete it.second.instance;
      }
      instances.clear();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::is_ready(bool do_subscribe)
    //--------------------------------------------------------------------------
    {
      if (do_subscribe)
        subscribe();
      if (producer_op == nullptr)
        return true;
      const int context_depth = (implicit_context == nullptr) ?
                                    producer_depth :
                                    implicit_context->get_depth();
      legion_assert(producer_depth <= context_depth);
      if (producer_depth < context_depth)
        return true;
      // This is not fully accurate since we might still need to wait across
      // the shards for control replication but it is close enough
      return producer_op->get_commit_event(op_gen).has_triggered();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::wait(bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      if (runtime->runtime_warnings && !silence_warnings &&
          (implicit_context != nullptr))
      {
        if (!implicit_context->is_leaf_context())
        {
          Warning warning;
          warning
              << "Waiting on a future in non-leaf task "
              << *implicit_context->owner_task << " is a violation of "
              << "Legion's deferred execution model best practices. You may "
              << "notice a severe performance degradation.";
          if (warning_string != nullptr)
            warning << " Warning string: " << warning_string;
          warning.raise();
        }
      }
      mark_sampled();
      // We only need to wait for the commit opertion for this future if we're
      // outside of the task tree or at the same level as the depth of the
      // context that produced the future (note this handles control
      // replication correctly as well since all the shards will appear to
      // be at the same depth). It should be impossible to have a depth above
      // the level where the future was created because futures cannot
      // escape back up the task tree
      if (producer_op == nullptr)
        return;
      if (implicit_context != nullptr)
      {
        const int context_depth = implicit_context->get_depth();
        legion_assert(producer_depth <= context_depth);
        if (producer_depth < context_depth)
          return;
        context->record_blocking_call(coordinate.context_index);
        context->wait_on_future(this, producer_op->get_commit_event(op_gen));
      }
      else
        get_complete_event().external_wait();
    }

    //--------------------------------------------------------------------------
    const void* FutureImpl::get_buffer(
        Processor proc, Memory::Kind memkind, size_t* extent_in_bytes,
        bool check_extent, bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          proc.exists() || (memkind == runtime->runtime_system_memory.kind()));
      // Figure out which memory we are looking for
      // If the user passed in a NO_PROC, then use the local system memory
      Memory memory = proc.exists() ?
                          runtime->find_local_memory(proc, memkind) :
                          runtime->runtime_system_memory;
      if (!memory.exists())
      {
        if (memkind != Memory::SYSTEM_MEM)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Unable to find a " << memkind << " memory associated with "
                << proc.kind() << " processor " << proc
                << " in which to create a future buffer.";
          error.raise();
        }
        else
          memory = runtime->runtime_system_memory;
      }
      return get_buffer(
          memory, extent_in_bytes, check_extent, silence_warnings,
          warning_string);
    }

    //--------------------------------------------------------------------------
    const void* FutureImpl::get_buffer(
        Memory memory, size_t* extent_in_bytes, bool check_extent,
        bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      // Make sure that we've subscribed
      const RtEvent subscribed = subscribe();
      // Wait to make sure that the future is complete first
      wait(silence_warnings, warning_string);
      // Do this wait after everything is ready for pipelining of communication
      subscribed.wait();
      ApEvent inst_ready;
      FutureInstance* instance = find_or_create_instance(
          memory, inst_ready, silence_warnings, warning_string);
      if (extent_in_bytes != nullptr)
      {
        if (check_extent)
        {
          if (empty.load())
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "Accessing empty " << *this << ".";
            error.raise();
          }
          else if (instance == nullptr)
          {
            if ((*extent_in_bytes) != 0)
            {
              Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
              error << "Future size mismatch! Expected type of 0 bytes but "
                    << "requested type is " << *extent_in_bytes << " bytes "
                    << "from " << *this << ".";
              error.raise();
            }
          }
          else if (future_size != *extent_in_bytes)
          {
            Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
            error << "Future size mismatch! Expected type of " << future_size
                  << " bytes but requested type is " << *extent_in_bytes
                  << " bytes from " << *this << ".";
            error.raise();
          }
        }
        else
          (*extent_in_bytes) = future_size;
      }
      if (instance == nullptr)
        return nullptr;
      bool poisoned = false;
      if (!inst_ready.has_triggered_faultaware(poisoned))
        inst_ready.wait_faultaware(poisoned);
      if (poisoned && (implicit_context != nullptr))
        implicit_context->raise_poison_exception();
      return instance->get_data();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::get_memories(
        std::set<Memory>& memories, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      // Wait for the future to be ready
      memories.clear();
      wait(silence_warnings, warning_string);
      AutoLock f_lock(future_lock, false /*exclusive*/);
      for (const std::pair<const Memory, FutureInstanceTracker>& instance :
           instances)
        memories.insert(instance.first);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance FutureImpl::get_instance(
        Memory::Kind memkind, size_t extent_in_bytes, bool check_extent,
        bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      // Make sure that we've subscribed
      const RtEvent subscribed = subscribe();
      Processor proc = implicit_context->get_executing_processor();
      // A heuristic to help out applications that are unsure of themselves
      if (memkind == Memory::NO_MEMKIND)
        memkind = (proc.kind() == Processor::TOC_PROC) ? Memory::GPU_FB_MEM :
                                                         Memory::SYSTEM_MEM;
      Memory memory = runtime->find_local_memory(proc, memkind);
      if (!memory.exists())
      {
        if (memkind != Memory::SYSTEM_MEM)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Unable to find a " << memkind << " memory associated with "
                << proc.kind() << " processor " << proc
                << " in which to create a future buffer for " << *this << ".";
          error.raise();
        }
        else
          memory = runtime->runtime_system_memory;
      }
      // Wait to make sure that the future is complete first
      wait(silence_warnings, warning_string);
      // Do this wait after everything is ready for pipelining of communication
      subscribed.wait();
      ApEvent inst_ready;
      FutureInstance* instance = find_or_create_instance(
          memory, inst_ready, silence_warnings, warning_string);
      if (empty.load())
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Accessing empty " << *this << " when making an accessor.";
        error.raise();
      }
      else if ((instance == nullptr) || (instance->size == 0))
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Future size mismatch! Expected non-empty future for "
              << "making an accessor for " << *this << " but future has a "
              << "payload of 0 bytes.";
        error.raise();
      }
      if (check_extent && (future_size != extent_in_bytes))
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Future size mismatch! Expected type of " << future_size
              << " bytes for " << *this << " but requested type is "
              << extent_in_bytes << " bytes.";
        error.raise();
      }
      PhysicalInstance result;
      {
        bool dummy_owner = true;
        // Need to hold the lock when creating the instance since
        // the future instance object is not thread safe
        AutoLock f_lock(future_lock);
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        LgEvent dummy_event;
        result =
            instance->get_instance(instance->size, dummy_event, dummy_owner);
#else
        result = instance->get_instance(instance->size, dummy_owner);
#endif
        // Should never be set to true here
        legion_assert(!dummy_owner);
      }
      bool poisoned = false;
      if (!inst_ready.has_triggered_faultaware(poisoned))
        inst_ready.wait_faultaware(poisoned);
      if (poisoned && (implicit_context != nullptr))
        implicit_context->raise_poison_exception();
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::report_incompatible_accessor(
        const char* accessor_kind, PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
      error << "Unable to create Realm " << accessor_kind
            << " for future instance " << instance << " of " << *this << " in "
            << *implicit_context->owner_task << ".";
      error.raise();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::request_application_instance(
        Memory target, SingleTask* task, RtEvent* safe_for_unbounded_pools,
        bool can_fail, size_t known_upper_bound_size)
    //--------------------------------------------------------------------------
    {
      legion_assert(task != nullptr);
      legion_assert(target.exists());
      TaskTreeCoordinates coordinates;
      task->compute_task_tree_coordinates(coordinates);
      const UniqueID task_uid = task->get_unique_id();
      // Check to see if this target is local, if not we need to send it
      // to the node where it needs to be made and wait for the response
      const AddressSpaceID target_space = target.address_space();
      if (target_space != runtime->address_space)
      {
        // Send the request off to the target node
        bool result = true;
        const RtUserEvent wait_on = Runtime::create_rt_user_event();
        FutureCreateInstanceRequest rez;
        {
          RezCheck z(rez);
          pack_future(rez, target_space);
          rez.serialize(target);
          rez.serialize(known_upper_bound_size);
          rez.serialize(task_uid);
          coordinates.serialize(rez);
          rez.serialize(safe_for_unbounded_pools);
          rez.serialize(wait_on);
          rez.serialize(&result);
        }
        rez.dispatch(target_space);
        wait_on.wait();
        if (result)
          return true;
      }
      else if (find_or_create_application_instance(
                   target, known_upper_bound_size, task_uid, coordinates,
                   safe_for_unbounded_pools))
        return true;
      if (!can_fail && ((safe_for_unbounded_pools == nullptr) ||
                        !safe_for_unbounded_pools->exists()))
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Failed to allocate instance of " << *this << " for " << *task
              << " because " << target.kind() << " memory " << target
              << " is full.";
        error.raise();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::find_or_create_application_instance(
        Memory target, size_t known_upper_bound_size, UniqueID task_uid,
        const TaskTreeCoordinates& coordinates,
        RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      legion_assert(target.address_space() == runtime->address_space);
      // Check to see if we already have an instance that will be ready
      {
        AutoLock f_lock(future_lock);
        if (instances.find(target) != instances.end())
          return true;
        if (pending_instances.find(target) != pending_instances.end())
          return true;
        // // Do the subscription if necessary
        if (!subscription_event.exists())
          subscribe(false /*need lock*/);
        // Check to see if we know the size of the future yet
        if (known_upper_bound_size == SIZE_MAX)
        {
          if (future_size_set)
            known_upper_bound_size = future_size;
          else if (upper_bound_size < SIZE_MAX)
            known_upper_bound_size = upper_bound_size;
          else
          {
            // We don't have any bound on the size of this future so we
            // need to wait for it so we can try to make the instance
            if (!future_size_ready.exists())
              future_size_ready = Runtime::create_rt_user_event();
            const RtEvent wait_on = future_size_ready;
            f_lock.release();
            wait_on.wait();
            f_lock.reacquire();
            legion_assert(future_size_set);
            // Check to see if we lost the race
            if (instances.find(target) != instances.end())
              return true;
            if (pending_instances.find(target) != pending_instances.end())
              return true;
            known_upper_bound_size = future_size;
          }
        }
      }
      if (known_upper_bound_size == 0)
        return true;
      legion_assert(known_upper_bound_size < SIZE_MAX);
      // Create the future instance of the given size
      MemoryManager* manager = runtime->find_memory_manager(target);
      FutureInstance* instance = manager->create_future_instance(
          task_uid, coordinates, known_upper_bound_size,
          safe_for_unbounded_pools);
      if (instance == nullptr)
        return false;
      // Retake the lock and register the instance in the right place
      // to be updated once the future is set
      AutoLock f_lock(future_lock);
      if (instances.find(target) != instances.end())
      {
        // Somebody else already made the instance
        if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
          delete instance;
        return true;
      }
      std::map<Memory, PendingInstance>::iterator finder =
          pending_instances.find(target);
      if (finder != pending_instances.end())
      {
        // See if we tightened the size of the instance or not
        if (instance->size < finder->second.instance->size)
        {
          if (!finder->second.instance->defer_deletion(ApEvent::NO_AP_EVENT))
            delete finder->second.instance;
          finder->second.instance = instance;
        }
        else if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
          delete instance;
        return true;
      }
      // See if the future is set yet or not
      if (empty.load())
        pending_instances.emplace(
            std::make_pair(target, PendingInstance(instance, task_uid)));
      else if (future_size > 0)
        record_instance(instance, task_uid);
      // Future is zero-sized so don't need the instance
      else if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
        delete instance;
      return true;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::request_runtime_instance(Operation* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(op != nullptr);
      // Check to see if we have an internal buffer to use already
      // If not record that we need one and do the subscription
      size_t known_upper_bound_size = SIZE_MAX;
      {
        AutoLock f_lock(future_lock);
        if (empty.load())
        {
          for (const std::pair<const Memory, PendingInstance>& pending :
               pending_instances)
            if (FutureInstance::check_meta_visible(pending.first))
              return;
        }
        else if (local_visible_memory.exists())
          return;
        // Do the subscription if necessary
        if (!subscription_event.exists())
          subscribe(false /*need lock*/);
        // Don't have a local instance yet so we need to make one
        // See if we know the upper bound size of the future
        if (future_size_set)
          known_upper_bound_size = future_size;
        else if (upper_bound_size < SIZE_MAX)
          known_upper_bound_size = upper_bound_size;
        else
        {
          // We don't have any bound on the size of this future so we
          // need to wait for it so we can try to make the instance
          if (!future_size_ready.exists())
            future_size_ready = Runtime::create_rt_user_event();
          const RtEvent wait_on = future_size_ready;
          f_lock.release();
          wait_on.wait();
          f_lock.reacquire();
          legion_assert(future_size_set);
          // Check to see if we lost the race
          if (empty.load())
          {
            for (const std::pair<const Memory, PendingInstance>& pending :
                 pending_instances)
              if (FutureInstance::check_meta_visible(pending.first))
                return;
          }
          else if (local_visible_memory.exists())
            return;
          known_upper_bound_size = future_size;
        }
      }
      legion_assert(known_upper_bound_size < SIZE_MAX);
      // Create the future instance of the given size
      TaskTreeCoordinates coordinates;
      op->compute_task_tree_coordinates(coordinates);
      MemoryManager* manager =
          runtime->find_memory_manager(runtime->runtime_system_memory);
      // Safe to block here indefinitely waiting for unbounded pools
      FutureInstance* instance = manager->create_future_instance(
          op->get_unique_op_id(), coordinates, known_upper_bound_size,
          nullptr /*safe_for_unbounded_pools*/);
      if (instance == nullptr)
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        error << "Failed to allocate instance of " << *this << " for " << *op
              << " because " << manager->get_name() << " memory "
              << runtime->runtime_system_memory << " is full.";
        error.raise();
      }
      // Retake the lock and see if we lost the race making the instance
      AutoLock f_lock(future_lock);
      if (empty.load())
      {
        for (const std::pair<const Memory, PendingInstance>& pending :
             pending_instances)
        {
          if (!FutureInstance::check_meta_visible(pending.first))
            continue;
          // Somebody else already made the instance
          if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
            delete instance;
          return;
        }
      }
      else if (local_visible_memory.exists())
      {
        // Somebody else already made the instance
        if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
          delete instance;
        return;
      }
      for (std::pair<const Memory, PendingInstance>& it : pending_instances)
      {
        if (!FutureInstance::check_meta_visible(it.first))
          continue;
        // See if we tightened the size of the instance or not
        if (instance->size < it.second.instance->size)
        {
          if (!it.second.instance->defer_deletion(ApEvent::NO_AP_EVENT))
            delete it.second.instance;
          it.second.instance = instance;
        }
        else if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
          delete instance;
        return;
      }
      // See if the future is set yet or not
      if (empty.load())
        pending_instances.emplace(std::make_pair(
            runtime->runtime_system_memory,
            PendingInstance(instance, op->get_unique_op_id())));
      else if (future_size > 0)
        record_instance(instance, op->get_unique_op_id());
      // Future is zero-sized so don't need the instance
      else if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
        delete instance;
    }

    //--------------------------------------------------------------------------
    ApEvent FutureImpl::find_application_instance_ready(
        Memory target, SingleTask* task)
    //--------------------------------------------------------------------------
    {
      legion_assert(target.address_space() == runtime->address_space);
      // Check to see if we have it
      AutoLock f_lock(future_lock);
      // Handle the case where we have a future with no payload
      if (future_size_set && (future_size == 0))
      {
        if (empty.load() && !future_complete.exists())
        {
          future_complete = Runtime::create_ap_user_event(nullptr);
          if (!subscription_event.exists())
            subscribe(false /*need lock*/);
        }
        return future_complete;
      }
      std::map<Memory, FutureInstanceTracker>::const_iterator finder =
          instances.find(target);
      if (finder != instances.end())
        return finder->second.ready_event;
      std::map<Memory, PendingInstance>::iterator pending_finder =
          pending_instances.find(target);
      legion_assert(pending_finder != pending_instances.end());
      if (!pending_finder->second.inst_ready.exists())
        pending_finder->second.inst_ready =
            Runtime::create_ap_user_event(nullptr);
      return pending_finder->second.inst_ready;
    }

    //--------------------------------------------------------------------------
    RtEvent FutureImpl::find_runtime_instance_ready(void)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      if (future_size_set && (future_size == 0))
        return RtEvent::NO_RT_EVENT;
      if (local_visible_memory.exists())
      {
        std::map<Memory, FutureInstanceTracker>::iterator finder =
            instances.find(local_visible_memory);
        legion_assert(finder != instances.end());
        if (!finder->second.ready_event.exists())
          return RtEvent::NO_RT_EVENT;
        if (!finder->second.safe_ready_event.exists())
          finder->second.safe_ready_event =
              Runtime::protect_event(finder->second.ready_event);
        return finder->second.safe_ready_event;
      }
      else
      {
        for (std::pair<const Memory, PendingInstance>& it : pending_instances)
        {
          if (!FutureInstance::check_meta_visible(it.first))
            continue;
          if (!it.second.inst_ready.exists())
            it.second.inst_ready = Runtime::create_ap_user_event(nullptr);
          if (!it.second.safe_inst_ready.exists())
            it.second.safe_inst_ready =
                Runtime::protect_event(it.second.inst_ready);
          return it.second.safe_inst_ready;
        }
      }
      // Should never get here because we should have called
      // request_runtime_instance first
      std::abort();
    }

    //--------------------------------------------------------------------------
    const void* FutureImpl::find_runtime_buffer(TaskContext* ctx, size_t& size)
    //--------------------------------------------------------------------------
    {
      RtEvent ready_event;
      FutureInstance* instance = nullptr;
      {
        AutoLock f_lock(future_lock);
        legion_assert(future_size_set);
        if (future_size == 0)
        {
          size = 0;
          return nullptr;
        }
        legion_assert(!empty.load());
        size = future_size;
        if (local_visible_memory.exists())
        {
          std::map<Memory, FutureInstanceTracker>::iterator finder =
              instances.find(local_visible_memory);
          legion_assert(finder != instances.end());
          instance = finder->second.instance;
          if (finder->second.ready_event.exists() &&
              !finder->second.safe_ready_event.exists())
            finder->second.safe_ready_event =
                Runtime::protect_event(finder->second.ready_event);
          ready_event = finder->second.safe_ready_event;
        }
        else
        {
          for (std::pair<const Memory, PendingInstance>& it : pending_instances)
          {
            if (it.second.instance == nullptr)
              continue;
            if (!it.second.instance->is_meta_visible)
              continue;
            instance = it.second.instance;
            if (!it.second.inst_ready.exists())
              it.second.inst_ready = Runtime::create_ap_user_event(nullptr);
            if (!it.second.safe_inst_ready.exists())
              it.second.safe_inst_ready =
                  Runtime::protect_event(it.second.inst_ready);
            ready_event = it.second.safe_inst_ready;
            break;
          }
        }
      }
      legion_assert(instance != nullptr);
      // Make sure the instance is safe to use
      ready_event.wait();
      return instance->get_data();
    }

    //--------------------------------------------------------------------------
    ApEvent FutureImpl::copy_to(
        FutureInstance* target, Operation* op, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      Memory source = find_best_source(target->memory);
      FutureInstanceTracker& tracker = instances[source];
      legion_assert(tracker.instance != nullptr);
      if (tracker.ready_event.exists())
      {
        if (precondition.exists())
          precondition =
              Runtime::merge_events(nullptr, precondition, tracker.ready_event);
        else
          precondition = tracker.ready_event;
      }
      const ApEvent ready_event =
          target->copy_from(tracker.instance, op, precondition);
      if (ready_event.exists())
        tracker.read_events.emplace_back(ready_event);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    ApEvent FutureImpl::reduce_to(
        FutureInstance* target, AllReduceOp* op, const ReductionOpID redop_id,
        const ReductionOp* redop, bool exclusive, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      Memory source = find_best_source(target->memory);
      FutureInstanceTracker& tracker = instances[source];
      legion_assert(tracker.instance != nullptr);
      if (tracker.ready_event.exists())
      {
        if (precondition.exists())
          precondition =
              Runtime::merge_events(nullptr, precondition, tracker.ready_event);
        else
          precondition = tracker.ready_event;
      }
      const ApEvent ready_event = target->reduce_from(
          tracker.instance, op, redop_id, redop, exclusive, precondition);
      if (ready_event.exists())
        tracker.read_events.emplace_back(ready_event);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    size_t FutureImpl::get_untyped_size(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we are subscribed
      AutoLock f_lock(future_lock);
      if (!future_size_set)
      {
        if (!future_size_ready.exists())
        {
          future_size_ready = Runtime::create_rt_user_event();
          // Do the subscription if we have need to
          if (!subscription_event.exists())
            subscribe(false /*need lock*/);
        }
        const RtEvent wait_on = future_size_ready;
        f_lock.release();
        wait_on.wait();
        f_lock.reacquire();
        legion_assert(future_size_set);
      }
      return future_size;
    }

    //--------------------------------------------------------------------------
    const void* FutureImpl::get_metadata(size_t* size)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready_event = subscribe();
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      if (size != nullptr)
        *size = metasize;
      return metadata;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::is_empty(
        bool block, bool silence_warnings, const char* warning_string,
        bool internal)
    //--------------------------------------------------------------------------
    {
      if (!internal)
      {
        if (runtime->runtime_warnings && !silence_warnings &&
            (producer_op != nullptr))
        {
          TaskContext* context = producer_op->get_context();
          if (!context->is_leaf_context())
          {
            Warning warning;
            warning
                << "Performing a blocking is_empty test on a future "
                << "in non-leaf " << *context << " is a violation of Legion's "
                << "deferred execution model best practices. You may notice a "
                << "severe performance degradation.";
            if (warning_string != nullptr)
              warning << " Warning string: " << warning_string;
            warning.raise();
          }
        }
        if (block && (context != nullptr) && (implicit_context == context))
          context->record_blocking_call(coordinate.context_index);
      }
      if (block)
      {
        const RtEvent ready_event = subscribe();
        mark_sampled();
        if (ready_event.exists() && !ready_event.has_triggered())
          ready_event.wait();
      }
      return empty.load();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::save_metadata(const void* meta, size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(metadata == nullptr);
      metasize = size;
      metadata = malloc(metasize);
      memcpy(metadata, meta, metasize);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(
        ApEvent complete, FutureInstance* instance, const void* meta,
        size_t size)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      if (!empty.load() || (callback_functor != nullptr))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Duplicate future set! This can be either a runtime bug or a "
            << "user error. If you have a must epoch launch in this program "
            << "please check that all of the point tasks that it creates have "
            << "unique index points. If your program has no must epoch "
               "launches "
            << "then this is likely a runtime bug.";
        error.raise();
      }
      legion_assert(instances.empty());
      legion_assert(metadata == nullptr);
      if (instance != nullptr)
      {
        instances.emplace(std::make_pair(
            instance->memory, FutureInstanceTracker(instance, complete)));
        if (instance->is_meta_visible)
          local_visible_memory = instance->memory;
      }
      if (size > 0)
        save_metadata(meta, size);
      finish_set_future(complete);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_results(
        ApEvent complete, const std::vector<FutureInstance*>& insts,
        const void* meta, size_t size)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      if (!empty.load() || (callback_functor != nullptr))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Duplicate future set! This can be either a runtime bug or a "
            << "user error. If you have a must epoch launch in this program "
            << "please check that all of the point tasks that it creates have "
            << "unique index points. If your program has no must epoch "
               "launches "
            << "then this is likely a runtime bug.";
        error.raise();
      }
      legion_assert(instances.empty());
      legion_assert(!insts.empty());
      legion_assert(metadata == nullptr);
      for (FutureInstance* const & inst : insts)
      {
        legion_assert(instances.find(inst->memory) == instances.end());
        instances.emplace(std::make_pair(
            inst->memory, FutureInstanceTracker(inst, complete)));
        if (!local_visible_memory.exists() && inst->is_meta_visible)
          local_visible_memory = inst->memory;
      }
      if (size > 0)
        save_metadata(meta, size);
      finish_set_future(complete);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(
        ApEvent complete, FutureFunctor* functor, bool own, Processor proc)
    //--------------------------------------------------------------------------
    {
      legion_assert(proc.kind() != Processor::UTIL_PROC);
      AutoLock f_lock(future_lock);
      if (!empty.load() || (callback_functor != nullptr))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Duplicate future set! This can be either a runtime bug or a "
            << "user error. If you have a must epoch launch in this program "
            << "please check that all of the point tasks that it creates have "
            << "unique index points. If your program has no must epoch "
               "launches "
            << "then this is likely a runtime bug.";
        error.raise();
      }
      callback_functor = functor;
      own_callback_functor = own;
      callback_proc = proc;
      finish_set_future(complete);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(
        Operation* op, FutureImpl* previous, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      const RtEvent subscribed = previous->subscribe();
      if (subscribed.exists() && !subscribed.has_triggered())
        subscribed.wait();
      const size_t size = previous->get_untyped_size();
      ApEvent complete = previous->get_complete_event();
      ApEvent ready;
      FutureInstance* instance = nullptr;
      if (size > 0)
      {
        TaskTreeCoordinates coordinates;
        op->compute_task_tree_coordinates(coordinates);
        instance = create_instance(
            op, coordinates, runtime->runtime_system_memory, size,
            safe_for_unbounded_pools);
        ready = previous->copy_to(instance, op, ApEvent::NO_AP_EVENT);
      }
      AutoLock f_lock(future_lock);
      if (!empty.load() || (callback_functor != nullptr))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Duplicate future set! This can be either a runtime bug or a "
            << "user error. If you have a must epoch launch in this program "
            << "please check that all of the point tasks that it creates have "
            << "unique index points. If your program has no must epoch "
               "launches "
            << "then this is likely a runtime bug.";
        error.raise();
      }
      legion_assert(instances.empty());
      legion_assert(metadata == nullptr);
      future_size = size;
      if (instance != nullptr)
      {
        instances.emplace(std::make_pair(
            instance->memory, FutureInstanceTracker(instance, ready)));
        local_visible_memory = instance->memory;
      }
      const void* meta = previous->get_metadata(&metasize);
      if (metasize > 0)
        save_metadata(meta, metasize);
      finish_set_future(complete);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(TaskContext* ctx, FutureImpl* previous)
    //--------------------------------------------------------------------------
    {
      const RtEvent subscribed = previous->subscribe();
      if (subscribed.exists() && !subscribed.has_triggered())
        subscribed.wait();
      const size_t size = previous->get_untyped_size();
      ApEvent complete = previous->get_complete_event();
      AutoLock f_lock(future_lock);
      if (!empty.load() || (callback_functor != nullptr))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Duplicate future set! This can be either a runtime bug or a "
            << "user error. If you have a must epoch launch in this program "
            << "please check that all of the point tasks that it creates have "
            << "unique index points. If your program has no must epoch "
               "launches "
            << "then this is likely a runtime bug.";
        error.raise();
      }
      legion_assert(instances.empty());
      legion_assert(metadata == nullptr);
      if (size > 0)
      {
        FutureInstance* instance =
            ctx->create_task_local_future(runtime->runtime_system_memory, size);
        ApEvent ready =
            previous->copy_to(instance, ctx->owner_task, ApEvent::NO_AP_EVENT);
        instances.emplace(std::make_pair(
            instance->memory, FutureInstanceTracker(instance, ready)));
        local_visible_memory = instance->memory;
      }
      const void* meta = previous->get_metadata(&metasize);
      if (metasize > 0)
        save_metadata(meta, metasize);
      finish_set_future(complete);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_local(const void* value, size_t size, bool own)
    //--------------------------------------------------------------------------
    {
      FutureInstance* instance =
          (size == 0) ? nullptr :
                        FutureInstance::create_local(value, size, own);
      set_result(ApEvent::NO_AP_EVENT, instance);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_future_result_size(size_t size, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      if (!empty.load() || future_size_set)
        return;
      upper_bound_size = size;
      future_size = size;
      future_size_set = true;
      if (!is_owner() && (source == local_space))
      {
        // Send the message back to the owner so it can broadcast it out
        // to any subscribers
        FutureSizeMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(size);
          pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
      }
      if (!subscribers.empty())
      {
        for (const AddressSpaceID& subscriber : subscribers)
        {
          if ((subscriber == source) || (subscriber == local_space))
            continue;
          FutureSizeMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(size);
            pack_global_ref(rez);
          }
          rez.dispatch(subscriber);
        }
      }
      // If the future size is zero then clean out any pending instances
      // that were made with the upper bound size
      if (future_size == 0)
      {
        for (std::pair<const Memory, PendingInstance>& it : pending_instances)
        {
          if (!it.second.instance->defer_deletion(ApEvent::NO_AP_EVENT))
            delete it.second.instance;
          if (it.second.inst_ready.exists())
            Runtime::trigger_event_untraced(it.second.inst_ready);
        }
        pending_instances.clear();
      }
      // If we have a future size ready then trigger it now
      if (future_size_ready.exists())
      {
        Runtime::trigger_event(future_size_ready);
        future_size_ready = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::finish_set_future(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          !future_size_set ||
          ((instances.empty() ? 0 : instances.begin()->second.instance->size) <=
           future_size));
      // must be called while we are already holding the lock
      future_size =
          instances.empty() ? 0 : instances.begin()->second.instance->size;
      future_size_set = true;
      if (future_size_ready.exists())
      {
        Runtime::trigger_event(future_size_ready);
        future_size_ready = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (future_complete.exists())
      {
        // If there's already a complete here then we know that it is a
        // user event that we need to trigger
        ApUserEvent to_trigger;
        to_trigger.id = future_complete.id;
        Runtime::trigger_event_untraced(to_trigger, complete);
      }
      else
        future_complete = complete;
      result_set_space = local_space;
      if (!pending_instances.empty())
        create_pending_instances();
      empty.store(false);
      if (!is_owner())
        // The owner always needs to be told of the result
        subscribers.insert(owner_space);
      if (!subscribers.empty())
        broadcast_result();
      if (subscription_event.exists())
      {
        Runtime::trigger_event(subscription_event);
        subscription_event = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::create_pending_instances(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_size_set);
      legion_assert(!pending_instances.empty());
      for (std::pair<const Memory, PendingInstance>& it : pending_instances)
      {
        // Check to see if we already have an instance for this memory
        // or if the future doesn't have any size to begin with
        std::map<Memory, FutureInstanceTracker>::iterator finder =
            instances.find(it.first);
        if ((finder != instances.end()) || (future_size == 0))
        {
          // If we do then we just trigger any events that we need to
          if (it.second.inst_ready.exists())
            Runtime::trigger_event_untraced(
                it.second.inst_ready, finder->second.ready_event);
          // Delete the instance we had made since it can't have escaped
          // yet and therefore we can replace it with the new instance
          if (!it.second.instance->defer_deletion(ApEvent::NO_AP_EVENT))
            delete it.second.instance;
          continue;
        }
        ApEvent inst_ready =
            record_instance(it.second.instance, it.second.creator_uid);
        if (it.second.inst_ready.exists())
          Runtime::trigger_event_untraced(it.second.inst_ready, inst_ready);
      }
      pending_instances.clear();
    }

    //--------------------------------------------------------------------------
    FutureInstance* FutureImpl::find_or_create_instance(
        Memory target, ApEvent& inst_ready, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      // No need to track subscribe since that was done in the caller
      size_t known_upper_bound_size = SIZE_MAX;
      {
        AutoLock f_lock(future_lock);
        legion_assert(future_size_set);
        legion_assert((future_size == 0) == instances.empty());
        // This is safe for control replication because all shards will
        // see the same size for the future and know if they need to
        // sync or not
        if (instances.empty())
          return nullptr;
        std::map<Memory, FutureInstanceTracker>::const_iterator finder =
            instances.find(target);
        if (finder != instances.end())
        {
          inst_ready = finder->second.ready_event;
          return finder->second.instance;
        }
        known_upper_bound_size = future_size;
      }
      legion_assert(0 < known_upper_bound_size);
      legion_assert(known_upper_bound_size < SIZE_MAX);
      // Make the instance and then try to record it
      FutureInstance* instance = nullptr;
      if (implicit_context == nullptr)
      {
        // This only happens with external requests for futures, this happens
        // outside the Legion mapping pipeline so there's no hope of ever
        // being able to do this in a sane way, we need to make an external
        // instance for this using malloc
        if (target != runtime->runtime_system_memory)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Request for Legion to allocate a future instance in "
                << target.kind() << " memory " << target
                << " by external code. "
                << "All requests for future instances by external code are "
                << "required to ask for the data in the system memory.";
          error.raise();
        }
        instance = new FutureInstance(
            malloc(known_upper_bound_size), known_upper_bound_size,
            true /*external*/, true /*own allocation*/);
      }
      else
        instance = implicit_context->create_task_local_future(
            target, known_upper_bound_size, silence_warnings, warning_string);
      if (instance == nullptr)
        return instance;
      // Retake the lock and see if we lost the race to register it
      AutoLock f_lock(future_lock);
      std::map<Memory, FutureInstanceTracker>::const_iterator finder =
          instances.find(target);
      if (finder != instances.end())
      {
        FutureInstance* result = finder->second.instance;
        inst_ready = finder->second.ready_event;
        f_lock.release();
        if (!instance->defer_deletion(ApEvent::NO_AP_EVENT))
          delete instance;
        return result;
      }
      else
        inst_ready = record_instance(
            instance, (implicit_context == nullptr) ?
                          0 :
                          implicit_context->get_unique_id());
      return instance;
    }

    //--------------------------------------------------------------------------
    FutureInstance* FutureImpl::create_instance(
        Operation* op, const TaskTreeCoordinates& coordinates, Memory memory,
        size_t size, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(instances.find(memory) == instances.end());
      legion_assert(memory.address_space() == runtime->address_space);
      MemoryManager* manager = runtime->find_memory_manager(memory);
      FutureInstance* instance = manager->create_future_instance(
          (op == nullptr) ? context->get_unique_id() : op->get_unique_op_id(),
          coordinates, size, safe_for_unbounded_pools);
      if (instance == nullptr)
      {
        Error error(LEGION_RESOURCE_EXCEPTION);
        if (op == nullptr)
          error << "Failed to allocate instance of " << *this << " of " << size
                << " bytes in " << *context << " because "
                << manager->get_name() << " memory " << memory << " is full.";
        else
          error << "Failed to allocate instance of " << *this << " of " << size
                << " bytes for " << *op << " because " << manager->get_name()
                << " memory " << memory << " is full.";
        error.raise();
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    ApEvent FutureImpl::record_instance(FutureInstance* instance, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_size > 0);
      legion_assert(!instances.empty());
      legion_assert(instances.find(instance->memory) == instances.end());
      legion_assert(instance->memory.address_space() == runtime->address_space);
      // Initialize the instance from one of the existing instances
      const Memory source = find_best_source(instance->memory);
      FutureInstanceTracker& tracker = instances[source];
      legion_assert(tracker.instance != nullptr);
      const ApEvent ready_event =
          instance->copy_from(tracker.instance, uid, tracker.ready_event);
      if (tracker.remote_postcondition.exists())
      {
        // This is a remote instance that we don't own so once we're
        // done with the copy we can clean it up
        Runtime::trigger_event_untraced(
            tracker.remote_postcondition, ready_event);
        delete tracker.instance;
        instances.erase(source);
      }
      else if (ready_event.exists())
        tracker.read_events.emplace_back(ready_event);
      instances.emplace(std::make_pair(
          instance->memory, FutureInstanceTracker(instance, ready_event)));
      if (!local_visible_memory.exists() && instance->is_meta_visible)
        local_visible_memory = instance->memory;
      return ready_event;
    }

    //--------------------------------------------------------------------------
    Memory FutureImpl::find_best_source(Memory target) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!instances.empty());
      // Check check to see if there is already an instance in the memory
      // since that will always be fastest if it is already there
      if (instances.find(target) != instances.end())
        return target;
      if (LEGION_MAX_RETURN_SIZE < future_size)
      {
        // Big future so search through to find source with the best affinity
        size_t best_bandwidth = 0;
        Memory best = Memory::NO_MEMORY;
        std::vector<Machine::MemoryMemoryAffinity> affinity;
        for (const std::pair<const Memory, FutureInstanceTracker>& instance :
             instances)
        {
          affinity.clear();
          runtime->machine.get_mem_mem_affinity(
              affinity, target, instance.first);
          if (affinity.empty())
            continue;
          if (!best.exists() || (best_bandwidth < affinity.front().bandwidth))
          {
            best = instance.first;
            best_bandwidth = affinity.front().bandwidth;
          }
        }
        if (best.exists())
          return best;
      }
      // Nothing left to do here other than to potentially pick the local
      // visible memory since we know it is ready to go, otherwise we
      // just randomly pick the first instance in the set
      if (local_visible_memory.exists())
        return local_visible_memory;
      return instances.begin()->first;
    }

    //--------------------------------------------------------------------------
    RtEvent FutureImpl::invoke_callback(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(callback_functor != nullptr);
      if (!subscription_event.exists())
      {
        subscription_event = Runtime::create_rt_user_event();
        FutureCallbackArgs args(this);
        runtime->issue_application_processor_task(
            args, LG_LATENCY_WORK_PRIORITY, callback_proc);
      }
      return subscription_event;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::perform_callback(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(callback_functor != nullptr);
      legion_assert(subscription_event.exists());
      size_t result_size = 0;
      bool owned = false;
      const Realm::ExternalInstanceResource* resource = nullptr;
      void (*freefunc)(const Realm::ExternalInstanceResource&) = nullptr;
      const void* metaptr = nullptr;
      const void* result = callback_functor->callback_get_future(
          result_size, owned, resource, freefunc, metaptr, metasize);
      FutureInstance* instance;
      if (resource == nullptr)
      {
        const Realm::ExternalMemoryResource local(
            reinterpret_cast<uintptr_t>(result), result_size,
            true /*read only*/);
        instance = new FutureInstance(
            result, result_size, owned, local.clone(),
            FutureInstance::free_host_memory, callback_proc);
      }
      else
        instance = new FutureInstance(
            result, result_size, owned, resource->clone(), freefunc,
            callback_proc);
      // If we have any metadata, copy that now
      if (metasize > 0)
      {
        legion_assert(metadata == nullptr);
        metadata = malloc(metasize);
        memcpy(metadata, metaptr, metasize);
      }
      if (owned)
      {
        // if we took ownership, we can release the functor now
        callback_functor->callback_release_future();
        if (own_callback_functor)
          delete callback_functor;
        callback_functor = nullptr;
      }
      // Retake the lock and remove the guards
      AutoLock f_lock(future_lock);
      instances.emplace(std::make_pair(
          instance->memory,
          FutureInstanceTracker(instance, ApEvent::NO_AP_EVENT)));
      if (!local_visible_memory.exists() && instance->is_meta_visible)
        local_visible_memory = instance->memory;
      if (!pending_instances.empty())
        create_pending_instances();
      Runtime::trigger_event(subscription_event);
      subscription_event = RtUserEvent::NO_RT_USER_EVENT;
      // Check for any subscribers that we need to tell about the result
      if (!subscribers.empty())
        broadcast_result();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::perform_broadcast(void)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      broadcast_result();
    }

    //--------------------------------------------------------------------------
    FutureImpl::FutureCallbackArgs::FutureCallbackArgs(FutureImpl* i)
      : LgTaskArgs<FutureCallbackArgs>(false, false), impl(i)
    //--------------------------------------------------------------------------
    {
      impl->add_base_gc_ref(DEFERRED_TASK_REF);
    }

    //--------------------------------------------------------------------------
    FutureImpl::CallbackReleaseArgs::CallbackReleaseArgs(
        FutureFunctor* f, bool own)
      : LgTaskArgs<CallbackReleaseArgs>(true, true), functor(f),
        own_functor(own)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FutureImpl::FutureBroadcastArgs::FutureBroadcastArgs(FutureImpl* i)
      : LgTaskArgs<FutureBroadcastArgs>(true, true), impl(i)
    //--------------------------------------------------------------------------
    {
      impl->add_base_gc_ref(DEFERRED_TASK_REF);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::FutureCallbackArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      impl->perform_callback();
      if (impl->remove_base_gc_ref(DEFERRED_TASK_REF))
        delete impl;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::CallbackReleaseArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      functor->callback_release_future();
      if (own_functor)
        delete functor;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::FutureBroadcastArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      impl->perform_broadcast();
      if (impl->remove_base_gc_ref(DEFERRED_TASK_REF))
        delete impl;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::unpack_future_result(Deserializer& derez)
    //-------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AutoLock f_lock(future_lock);
      legion_assert(empty.load());
      legion_assert(subscription_event.exists() || is_owner());
      legion_assert(metadata == nullptr);
      derez.deserialize(future_size);
      future_size_set = true;
      derez.deserialize(result_set_space);
      if (future_size > 0)
      {
        bool has_local = false;
        size_t num_instances;
        derez.deserialize(num_instances);
        for (unsigned idx1 = 0; idx1 < num_instances; idx1++)
        {
          FutureInstance* instance = FutureInstance::unpack_instance(derez);
          if (instance->memory.address_space() == runtime->address_space)
          {
            has_local = true;
            if (!local_visible_memory.exists() && instance->is_meta_visible)
              local_visible_memory = instance->memory;
          }
          legion_assert(instances.find(instance->memory) == instances.end());
          ApEvent ready_event;
          derez.deserialize(ready_event);
          FutureInstanceTracker& tracker =
              instances
                  .emplace(std::make_pair(
                      instance->memory,
                      FutureInstanceTracker(instance, ready_event)))
                  .first->second;
          size_t num_read_events;
          derez.deserialize(num_read_events);
          tracker.read_events.resize(num_read_events);
          for (unsigned idx2 = 0; idx2 < num_read_events; idx2++)
            derez.deserialize(tracker.read_events[idx2]);
        }
        if (!has_local)
        {
          // If we didn't get a local instance then we'll also get a copy
          // of the future instance from the source
          FutureInstance* instance = FutureInstance::unpack_instance(derez);
          if (instance->is_meta_visible)
          {
            // We unpacked by value if it is meta-visible here
            legion_assert(!local_visible_memory.exists());
            legion_assert(instances.find(instance->memory) == instances.end());
            instances.emplace(std::make_pair(
                instance->memory,
                FutureInstanceTracker(instance, ApEvent::NO_AP_EVENT)));
            local_visible_memory = instance->memory;
          }
          else
          {
            // The instance is remote from us, we need to copy to one of the
            // pending instances so we can fill in the rest of the results
            ApEvent precondition;
            derez.deserialize(precondition);
            ApUserEvent postcondition;
            derez.deserialize(postcondition);
            if (pending_instances.empty())
            {
              // Save this in the list of instances to consume once
              // someone tries to read from it
              instances.emplace(std::make_pair(
                  instance->memory,
                  FutureInstanceTracker(
                      instance, precondition, postcondition)));
            }
            else
            {
              std::map<Memory, PendingInstance>::iterator pending =
                  pending_instances.begin();
              // Issue the copy to the pending instance
              ApEvent ready = pending->second.instance->copy_from(
                  instance, pending->second.creator_uid, precondition);
              Runtime::trigger_event_untraced(postcondition, ready);
              if (pending->second.inst_ready.exists())
                Runtime::trigger_event_untraced(
                    pending->second.inst_ready, ready);
              instances.emplace(std::make_pair(
                  pending->second.instance->memory,
                  FutureInstanceTracker(pending->second.instance, ready)));
              legion_assert(!local_visible_memory.exists());
              if (pending->second.instance->is_meta_visible)
                local_visible_memory = pending->second.instance->memory;
              pending_instances.erase(pending);
              delete instance;
            }
          }
        }
      }
      if (future_complete.exists())
      {
        ApUserEvent to_trigger;
        to_trigger.id = future_complete.id;
        ApEvent precondition;
        derez.deserialize(precondition);
        Runtime::trigger_event_untraced(to_trigger, precondition);
      }
      else
        derez.deserialize(future_complete);
      derez.deserialize(metasize);
      if (metasize > 0)
      {
        save_metadata(derez.get_current_pointer(), metasize);
        derez.advance_pointer(metasize);
      }
      if (!pending_instances.empty())
        create_pending_instances();
      empty.store(false);
      // If we have a future size ready then trigger it now
      if (future_size_ready.exists())
      {
        Runtime::trigger_event(future_size_ready);
        future_size_ready = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (subscription_event.exists())
      {
        Runtime::trigger_event(subscription_event);
        subscription_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (!subscribers.empty())
        broadcast_result();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::reset_future(void)
    //--------------------------------------------------------------------------
    {
      // TODO: update this for resilience
      std::abort();
      bool was_sampled = sampled.load();
      sampled.store(false);
      return was_sampled;
    }

    //--------------------------------------------------------------------------
    ApEvent FutureImpl::get_complete_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      if (empty.load())
      {
        if (!future_complete.exists())
          future_complete = Runtime::create_ap_user_event(nullptr);
        if (!subscription_event.exists())
          subscribe(false /*need lock*/);
      }
      return future_complete;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::get_boolean_value(TaskContext* ctx)
    //--------------------------------------------------------------------------
    {
      size_t size = 0;
      const void* result = find_runtime_buffer(ctx, size);
      legion_assert(sizeof(bool) == size);
      return *((const bool*)result);
    }

    //--------------------------------------------------------------------------
    RtEvent FutureImpl::subscribe(bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (!empty.load() && (callback_functor == nullptr))
        return RtEvent::NO_RT_EVENT;
      if (need_lock)
      {
        AutoLock f_lock(future_lock);
        return subscribe(false /*need lock*/);
      }
      // See if we lost the race
      if (empty.load())
      {
        if (!subscription_event.exists())
        {
          subscription_event = Runtime::create_rt_user_event();
          if (!is_owner())
          {
            // Send a request to the owner node to subscribe
            FutureSubscription rez;
            rez.serialize(did);
            pack_global_ref(rez);
            if ((collective_mapping != nullptr) &&
                collective_mapping->contains(local_space))
              rez.dispatch(
                  collective_mapping->get_parent(owner_space, local_space));
            else
              rez.dispatch(owner_space);
          }
          else
            record_subscription(local_space, false /*need lock*/);
        }
        return subscription_event;
      }
      else
      {
        if (callback_functor != nullptr)
          return invoke_callback();
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    size_t FutureImpl::get_upper_bound_size(void)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock f_lock(future_lock, false /*exclusive*/);
        if (!empty.load() || future_size_set)
          return upper_bound_size;
      }
      const RtEvent subscribed = subscribe();
      if (subscribed.exists() && !subscribed.has_triggered())
        subscribed.wait();
      AutoLock f_lock(future_lock, false /*exclusive*/);
      legion_assert(!empty.load() || future_size_set);
      return upper_bound_size;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::get_context_coordinate(
        const TaskContext* ctx, ContextCoordinate& coord) const
    //--------------------------------------------------------------------------
    {
      // If contexts don't match then we don't return the coordinate
      if (ctx != context)
        return false;
      // No coordinates if we are an application-generated future
      if (coordinate.context_index == InnerContext::NO_BLOCKING_INDEX)
        return false;
      coord = coordinate;
      return true;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::pack_future(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize<DistributedID>(did);
      if ((collective_mapping != nullptr) &&
          collective_mapping->contains(target))
      {
        rez.serialize<bool>(true);  // collective
        pack_global_ref(rez);
        return;
      }
      else
        rez.serialize<bool>(false);  // collective
      rez.serialize(context->did);
      coordinate.serialize(rez);
      if (collective_mapping != nullptr)
        collective_mapping->pack(rez);
      else
        rez.serialize<size_t>(0);  // no collective mapping
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      pack_global_ref(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ Future FutureImpl::unpack_future(
        Deserializer& derez, Operation* op, GenerationID op_gen,
        UniqueID op_uid, int op_depth)
    //--------------------------------------------------------------------------
    {
      DistributedID future_did, ctx_did;
      derez.deserialize(future_did);
      if (future_did == 0)
        return Future();
      bool collective;
      derez.deserialize(collective);
      if (collective)
      {
        // Wait until we find it here
        Future result(static_cast<FutureImpl*>(
            runtime->find_distributed_collectable(future_did)));
        result.impl->unpack_global_ref(derez);
        return result;
      }
      derez.deserialize(ctx_did);
      ContextCoordinate coordinate;
      coordinate.deserialize(derez);
      size_t collective_spaces;
      derez.deserialize(collective_spaces);
      CollectiveMapping* collective_mapping =
          (collective_spaces == 0) ?
              nullptr :
              new CollectiveMapping(derez, collective_spaces);
      if (collective_mapping != nullptr)
        collective_mapping->add_reference();
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      Future result(runtime->find_or_create_future(
          future_did, ctx_did, coordinate, provenance, op, op_gen, op_uid,
          op_depth, collective_mapping));
      result.impl->unpack_global_ref(derez);
      if ((collective_mapping != nullptr) &&
          collective_mapping->remove_reference())
        delete collective_mapping;
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::notify_local(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void FutureImpl::register_dependence(Operation* consumer_op)
    //--------------------------------------------------------------------------
    {
      // Only record dependences on things from the same context
      // We know futures can never flow up the task tree so the
      // only way they have the same depth is if they are from
      // the same parent context
      TaskContext* consumer_context = consumer_op->get_context();
      if (consumer_context != context)
      {
        // Check that the consumer is contained within the task
        // sub-tree of the producer task
        TaskTreeCoordinates prod_coords, con_coords;
        context->compute_task_tree_coordinates(prod_coords);
        consumer_context->compute_task_tree_coordinates(con_coords);
        bool contained = (prod_coords.size() <= con_coords.size());
        if (contained)
        {
          for (unsigned idx = 0; idx < prod_coords.size(); idx++)
          {
            if (prod_coords[idx] == con_coords[idx])
              continue;
            contained = false;
            break;
          }
        }
        if (!contained)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal use of " << *this << " produced in context "
                << *context << " but consumed in context " << *consumer_context
                << " by operation " << *consumer_op
                << ". Futures are only permitted to be used in the task "
                << "sub-tree rooted by the context that produced the future.";
          error.raise();
        }
      }
      else if (producer_op != nullptr)
      {
        consumer_op->register_dependence(producer_op, op_gen);
        LegionSpy::log_future_dependence(
            context->get_unique_id(), producer_uid,
            consumer_op->get_unique_op_id());
      }
      else
        legion_assert(
            !empty.load());  // better not be empty if it doesn't have an op
    }

    //--------------------------------------------------------------------------
    void FutureImpl::mark_sampled(void)
    //--------------------------------------------------------------------------
    {
      sampled.store(true);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::broadcast_result(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!empty.load());
      if (callback_functor != nullptr)
      {
        // Handle the special case where the only subscriber is the local
        // node so we can lazily defer this until later and the user
        // actually asks us for the result
        if (!subscribers.empty() &&
            ((subscribers.size() > 1) ||
             (subscribers.find(local_space) == subscribers.end())))
        {
          // If we still have a callback to perform do
          // that now to get it in flight, it will send
          // out any updates to subscribers
          invoke_callback();
        }
        return;
      }
      // If this is a small future, we need to have a local visible
      // instance that we can use for packing this by value
      if ((0 < future_size) && (future_size <= LEGION_MAX_RETURN_SIZE))
      {
        ApEvent inst_ready;
        if (!local_visible_memory.exists())
        {
          // Check to see if we have targets for all the subscribers
          // if so then we don't need to make a local copy
          std::set<AddressSpaceID> subscribers_without_instances = subscribers;
          for (const std::pair<const Memory, FutureInstanceTracker>& instance :
               instances)
          {
            subscribers_without_instances.erase(instance.first.address_space());
            if (subscribers_without_instances.empty())
              break;
          }
          if (!subscribers_without_instances.empty())
          {
            // This looks scary because we're not actually in the mapping
            // stage of the pipeline for some operation, but it's a "small"
            // future and it's going to the host memory so it will hit the
            // path that just calls malloc and not perform an allocation
            TaskTreeCoordinates coordinates;
            context->compute_task_tree_coordinates(coordinates);
            FutureInstance* instance = create_instance(
                context->owner_task, coordinates,
                runtime->runtime_system_memory, future_size,
                nullptr /*safe_for_unbounded_pools*/);
            inst_ready = record_instance(instance, context->get_unique_id());
            legion_assert(local_visible_memory.exists());
          }
        }
        else
          inst_ready = instances[local_visible_memory].ready_event;
        if (inst_ready.exists() && !inst_ready.has_triggered_faultignorant())
        {
          // Defer until the instance can be packed by value
          const RtEvent precondition = Runtime::protect_event(inst_ready);
          FutureBroadcastArgs args(this);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
          return;
        }
      }
      for (const AddressSpaceID& subscriber : subscribers)
      {
        if ((subscriber == local_space) || (subscriber == result_set_space))
          continue;
        // Need to pack each of these separately in case we need to make
        // events for each future instance being packed
        FutureResultMessage rez;
        pack_future_result(rez, subscriber);
        pack_global_ref(rez);
        rez.dispatch(subscriber);
      }
      subscribers.clear();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::record_subscription(
        AddressSpaceID subscriber, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock f_lock(future_lock);
        record_subscription(subscriber, false /*need lock*/);
        return;
      }
      if (empty.load())
      {
        // Send the future size back to the subscriber so they
        // can have it to be able to create instances
        if (future_size_set && (subscriber != local_space))
        {
          FutureSizeMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(future_size);
            pack_global_ref(rez);
          }
          rez.dispatch(subscriber);
        }
        legion_assert(subscribers.find(subscriber) == subscribers.end());
        subscribers.insert(subscriber);
        if (!is_owner())
        {
          // Not the owner, we should be in a collective future
          legion_assert(collective_mapping != nullptr);
          legion_assert(collective_mapping->contains(local_space));
          subscribe(false /*needs lock*/);
        }
      }
      else
      {
        // Handle the race where the subscription from the node that
        // ultimately set the future result arrives after the future
        // result makes it here
        if ((subscriber == result_set_space) && (subscriber != local_space))
          return;
        if (callback_functor != nullptr)
        {
          invoke_callback();
          // If we still have a callback to be done, make sure
          // it is in flight and that the subscriber is there
          // for it to be messaged when the callback is done
          subscribers.insert(subscriber);
        }
        else if (!instances.empty() && (future_size <= LEGION_MAX_RETURN_SIZE))
        {
          // Check to see if we have any instances to send directly
          bool has_local_target = false;
          for (const std::pair<const Memory, FutureInstanceTracker>& instance :
               instances)
          {
            if (instance.first.address_space() != subscriber)
              continue;
            has_local_target = true;
            break;
          }
          ApEvent local_visible_ready;
          if (!has_local_target)
          {
            // If we don't see if we need to make a local visible
            if (!local_visible_memory.exists())
            {
              // This looks scary because we're not actually in the mapping
              // stage of the pipeline for some operation, but it's a "small"
              // future and it's going to the host memory so it will hit the
              // path that just calls malloc and not perform an allocation
              TaskTreeCoordinates coordinates;
              context->compute_task_tree_coordinates(coordinates);
              FutureInstance* instance = create_instance(
                  context->owner_task, coordinates,
                  runtime->runtime_system_memory, future_size,
                  nullptr /*safe_for_unbounded_pools*/);
              local_visible_ready =
                  record_instance(instance, context->get_unique_id());
              legion_assert(local_visible_memory.exists());
            }
            else
              local_visible_ready = instances[local_visible_memory].ready_event;
          }
          if (local_visible_ready.exists() &&
              !local_visible_ready.has_triggered_faultignorant())
          {
            // Save the subscriber and launch a broadcast task
            // if one is not already in flight to broadcast the
            // result to the subscriptions once it is ready
            if (subscribers.empty())
            {
              // First one so launch the broadcast task
              FutureBroadcastArgs args(this);
              runtime->issue_runtime_meta_task(
                  args, LG_LATENCY_WORK_PRIORITY,
                  Runtime::protect_event(local_visible_ready));
            }
            subscribers.insert(subscriber);
          }
          else
          {
            // We can send the result right now
            FutureResultMessage rez;
            pack_future_result(rez, subscriber);
            pack_global_ref(rez);
            rez.dispatch(subscriber);
          }
        }
        else
        {
          // Send the result back to the subscriber since right away
          FutureResultMessage rez;
          pack_future_result(rez, subscriber);
          pack_global_ref(rez);
          rez.dispatch(subscriber);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::pack_future_result(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert((future_size == 0) == instances.empty());
      rez.serialize(did);
      RezCheck z(rez);
      rez.serialize(future_size);
      rez.serialize(result_set_space);
      if (!instances.empty())
      {
        // Find any target instances to pack up to send to the subscriber
        bool has_exact_target = false;
        std::vector<Memory> target_memories;
        for (const std::pair<const Memory, FutureInstanceTracker>& instance :
             instances)
        {
          AddressSpaceID inst_space = instance.first.address_space();
          if (inst_space == target)
          {
            has_exact_target = true;
            target_memories.emplace_back(instance.first);
          }
          else if (
              (collective_mapping != nullptr) &&
              collective_mapping->contains(inst_space) &&
              (subscribers.find(inst_space) == subscribers.end()))
          {
            // If there is a collective mapping we need to check whether
            // we should send the instance to the target because the
            // actual subscriber will come through the collective mapping
            // subscriber tree to the target.
            while ((inst_space != owner_space) && (inst_space != target) &&
                   (inst_space != local_space) &&
                   (subscribers.find(inst_space) == subscribers.end()))
              inst_space =
                  collective_mapping->get_parent(owner_space, inst_space);
            if (inst_space == target)
              target_memories.emplace_back(instance.first);
            else if ((inst_space == owner_space) && (target == owner_space))
              target_memories.emplace_back(instance.first);
          }
        }
        if (!target_memories.empty())
        {
          // Check to see if we're packing all our instances to send away.
          // If we are we still need to keep one of them around to be able
          // to copy from it if we need to
          const Memory keep =
              (target_memories.size() < instances.size()) ?
                  Memory::NO_MEMORY :
                  find_best_source(runtime->runtime_system_memory);
          // Send the instances to the future impl that should own them
          rez.serialize<size_t>(target_memories.size());
          for (const Memory& target_memory : target_memories)
          {
            std::map<Memory, FutureInstanceTracker>::iterator finder =
                instances.find(target_memory);
            legion_assert(finder != instances.end());
            // Don't allow this to be packed by value
            finder->second.instance->pack_instance(
                rez, ApEvent::NO_AP_EVENT, true /*move ownership*/,
                false /*allow by value*/);
            rez.serialize(finder->second.ready_event);
            if (target_memory == keep)
            {
              finder->second.remote_postcondition =
                  Runtime::create_ap_user_event(nullptr);
              finder->second.read_events.emplace_back(
                  finder->second.remote_postcondition);
            }
            rez.serialize<size_t>(finder->second.read_events.size());
            for (const ApEvent& read_event : finder->second.read_events)
              rez.serialize(read_event);
            if (target_memory != keep)
            {
              // Now we can delete the instance remove it from the entry
              delete finder->second.instance;
              instances.erase(finder);
            }
            else
              finder->second.read_events.clear();
          }
        }
        else
          rez.serialize<size_t>(0);
        if (!has_exact_target)
        {
          // Pack our local visible copy by value so that the subscriber
          // will have it's own local copy of the data
          std::map<Memory, FutureInstanceTracker>::iterator finder =
              local_visible_memory.exists() ?
                  instances.find(local_visible_memory) :
                  instances.begin();
          legion_assert(finder != instances.end());
          if (!finder->second.instance->pack_instance(
                  rez, finder->second.ready_event, false /*move ownership*/))
          {
            // Couldn't pack this by value so we need to pack up events
            rez.serialize(finder->second.ready_event);
            const ApUserEvent read_done =
                Runtime::create_ap_user_event(nullptr);
            rez.serialize(read_done);
            finder->second.read_events.emplace_back(read_done);
          }
        }
      }
      rez.serialize(future_complete);
      rez.serialize(metasize);
      if (metasize > 0)
        rez.serialize(metadata, metasize);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::record_future_registered(void)
    //--------------------------------------------------------------------------
    {
      // Similar to DistributedCollectable::register_with_runtime but
      // we don't actually need to do the registration since we know
      // it has already been done
      legion_assert(!registered_with_runtime);
      registered_with_runtime = true;
      if (!is_owner())
        send_remote_registration();
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureResultMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureImpl* future = legion_safe_cast<FutureImpl*>(dc);
#ifdef LEGION_DEBUG
      // A little bit strange, but if we go to do the broadcast when
      // unpacking the result, we might need to pack other global references
      // and the check in global references wants to see that we have at
      // least one global reference on this node. Technically we have one
      // since we haven't unpacked our global reference yet, but that check
      // can't see it, so instead we do a global acquire and make sure that
      // works so we have at least one concrete reference on this node in
      // case we need to pack any global references.
      legion_no_skip_assert(future->check_global_and_increment(RUNTIME_REF));
      future->unpack_future_result(derez);
      future->unpack_global_ref(derez);
      if (future->remove_base_gc_ref(RUNTIME_REF))
        delete future;
#else
      future->unpack_future_result(derez);
      future->unpack_global_ref(derez);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureSizeMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      // The future might be a collective future so wait for it if
      // it hasn't been registered yet
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureImpl* future = legion_safe_cast<FutureImpl*>(dc);
      size_t future_size;
      derez.deserialize(future_size);
#ifdef LEGION_DEBUG
      // Same case here as above to avoid overzealous assertions
      legion_no_skip_assert(future->check_global_and_increment(RUNTIME_REF));
      future->unpack_global_ref(derez);
      future->set_future_result_size(future_size, source);
      if (future->remove_base_gc_ref(RUNTIME_REF))
        delete future;
#else
      future->set_future_result_size(future_size, source);
      future->unpack_global_ref(derez);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureSubscription::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureImpl* future = legion_safe_cast<FutureImpl*>(dc);
#ifdef LEGION_DEBUG
      // Same case here as above to avoid overzealous assertions
      legion_no_skip_assert(future->check_global_and_increment(RUNTIME_REF));
      future->unpack_global_ref(derez);
      future->record_subscription(source, true /*need lock*/);
      if (future->remove_base_gc_ref(RUNTIME_REF))
        delete future;
#else
      future->record_subscription(source, true /*need lock*/);
      future->unpack_global_ref(derez);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureCreateInstanceRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Future f = FutureImpl::unpack_future(derez);
      Memory target;
      derez.deserialize(target);
      size_t known_upper_bound_size;
      derez.deserialize(known_upper_bound_size);
      UniqueID creator_uid;
      derez.deserialize(creator_uid);
      TaskTreeCoordinates coordinates;
      coordinates.deserialize(derez);
      RtEvent* remote_safe_for_unbounded_pools;
      derez.deserialize(remote_safe_for_unbounded_pools);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool* result;
      derez.deserialize(result);

      RtEvent safe_for_unbounded_pools;
      if (!f.impl->find_or_create_application_instance(
              target, known_upper_bound_size, creator_uid, coordinates,
              (remote_safe_for_unbounded_pools == nullptr) ?
                  nullptr :
                  &safe_for_unbounded_pools))
      {
        FutureCreateInstanceResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(result);
          rez.serialize(remote_safe_for_unbounded_pools);
          if (remote_safe_for_unbounded_pools != nullptr)
            rez.serialize(safe_for_unbounded_pools);
          rez.serialize(done_event);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureCreateInstanceResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      bool* result;
      derez.deserialize(result);
      *result = false;
      RtEvent* safe_for_unbounded_pools;
      derez.deserialize(safe_for_unbounded_pools);
      if (safe_for_unbounded_pools != nullptr)
        derez.deserialize(*safe_for_unbounded_pools);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::contribute_to_collective(
        const DynamicCollective& dc, unsigned count)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready = subscribe();
      if (ready.exists() && !ready.has_triggered())
      {
        // If we're not done then defer the operation until we are triggerd
        // First add a garbage collection reference so we don't get
        // collected while we are waiting for the contribution task to run
        add_base_gc_ref(PENDING_COLLECTIVE_REF);
        ContributeCollectiveArgs args(this, dc, count);
        // Spawn the task dependent on the future being ready
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, ready);
      }
      else  // If we've already triggered, then we can do the arrival now
      {
        size_t result_size = 0;
        const void* result = get_buffer(
            runtime->runtime_system_memory, &result_size, false /*check size*/,
            true /*silence warnings*/);
        runtime->phase_barrier_arrive(
            dc, count, ApEvent::NO_AP_EVENT, result, result_size);
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::ContributeCollectiveArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      impl->contribute_to_collective(dc, count);
      // Now remote the garbage collection reference and see if we can
      // reclaim the future
      if (impl->remove_base_gc_ref(PENDING_COLLECTIVE_REF))
        delete impl;
    }

    /////////////////////////////////////////////////////////////
    // Future Instance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureInstance::FutureInstance(
        const void* d, size_t s, bool external, bool own, LgEvent unique,
        PhysicalInstance inst, Processor p, RtEvent use)
      : size(s), memory(
                     inst.exists() ? inst.get_location() :
                                     runtime->runtime_system_memory),
        resource(
            inst.exists() ?
                nullptr :
                new Realm::ExternalMemoryResource(
                    reinterpret_cast<uintptr_t>(d), s, false /*read only*/)),
        freefunc(inst.exists() || !p.exists() ? nullptr : free_host_memory),
        freeproc(p), external_allocation(external),
        is_meta_visible(check_meta_visible(memory)), own_allocation(own),
        data(d), instance(inst), use_event(use), unique_event(unique),
        own_instance(own && inst.exists())
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(memory.exists());
      legion_assert((freefunc == nullptr) || freeproc.exists());
      legion_assert((freefunc == nullptr) || external_allocation);
      legion_assert(
          !freeproc.exists() || (freeproc.kind() != Processor::UTIL_PROC));
      legion_assert(instance.exists() || external_allocation);
      legion_assert((data != nullptr) || instance.exists());
      legion_assert(
          unique_event.exists() || !instance.exists() ||
          (runtime->profiler == nullptr));
    }

    //--------------------------------------------------------------------------
    FutureInstance::FutureInstance(
        const void* d, size_t s, bool own,
        const Realm::ExternalInstanceResource* allocation,
        void (*func)(const Realm::ExternalInstanceResource&), Processor proc,
        LgEvent unique, PhysicalInstance inst, RtEvent use)
      : size(s), memory(
                     inst.exists() ? inst.get_location() :
                                     allocation->suggested_memory()),
        resource(allocation), freefunc(func), freeproc(proc),
        external_allocation(true), is_meta_visible(check_meta_visible(memory)),
        own_allocation(own), data(d), instance(inst), use_event(use),
        unique_event(unique), own_instance(own && inst.exists())
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(memory.exists());
      legion_assert((freefunc == nullptr) || freeproc.exists());
      legion_assert((freefunc == nullptr) || external_allocation);
      legion_assert(
          !freeproc.exists() || (freeproc.kind() != Processor::UTIL_PROC));
      legion_assert(instance.exists() || external_allocation);
      legion_assert((data != nullptr) || instance.exists());
      legion_assert((resource != nullptr) || inst.exists());
      legion_assert(
          unique_event.exists() || !instance.exists() ||
          (runtime->profiler == nullptr));
    }

    //--------------------------------------------------------------------------
    FutureInstance::~FutureInstance(void)
    //--------------------------------------------------------------------------
    {
      // Make sure our instance is valid before we try to delete it
      if (instance.exists() && use_event.exists() && !use_event.has_triggered())
        use_event.wait();
      bool free_resource = true;
      // Only need to free resources if we own the allocation
      if (own_allocation)
      {
        if (external_allocation)
        {
          const AddressSpaceID target_space = memory.address_space();
          if (target_space != runtime->address_space)
          {
            legion_assert(
                !freeproc.exists() || freeproc.address_space() == target_space);
            // Send the message to the remote node to do the free
            FreeExternalAllocation rez;
            {
              RezCheck z(rez);
              rez.serialize(freeproc);
              rez.serialize(freefunc);
              rez.serialize(instance);
              rez.serialize(unique_event);
            }
            rez.dispatch(target_space);
          }
          else
          {
            // We're already local, see if we need to launch a task
            if (freeproc.exists())
            {
              FreeExternalArgs args(
                  resource, (freefunc != nullptr) ? freefunc : free_host_memory,
                  instance);
              if (freeproc.exists())
                runtime->issue_application_processor_task(
                    args, LG_THROUGHPUT_WORK_PRIORITY, freeproc);
              else
                runtime->issue_runtime_meta_task(
                    args, LG_THROUGHPUT_WORK_PRIORITY);
              // No longer safe to free the resource since that is going
              // to be done by the free external args task
              free_resource = false;
            }
            else
            {
              // We can do the free now
              free(const_cast<void*>(data.load()));
              if (instance.exists())
                instance.destroy();
            }
          }
        }
        else
        {
          legion_assert(instance.exists());
          legion_assert(own_instance);
          // Free the future instance through the memory manager
          MemoryManager* manager = runtime->find_memory_manager(memory);
          manager->free_future_instance(instance, size, RtEvent::NO_RT_EVENT);
        }
      }
      else if (own_instance)
      {
        legion_assert(instance.exists());
        instance.destroy();
      }
      if ((resource != nullptr) && free_resource)
        delete resource;
    }

    //--------------------------------------------------------------------------
    ApEvent FutureInstance::initialize(
        const ReductionOp* redop, Operation* op, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      // Check to see if this is visible or not
      if (!is_meta_visible || (precondition.exists() &&
                               !precondition.has_triggered_faultignorant()))
      {
        Realm::CopySrcDstField src, dst;
        src.set_fill(redop->identity, redop->sizeof_rhs);
        bool own_inst = false;
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        LgEvent inst_event;
        PhysicalInstance dst_inst =
            get_instance(redop->sizeof_rhs, inst_event, own_inst);
#else
        PhysicalInstance dst_inst = get_instance(redop->sizeof_rhs, own_inst);
#endif
        // Should only be writing to instances that this future instance owns
        legion_assert(own_instance);
        dst.set_field(dst_inst, 0 /*field id*/, size);
        std::vector<Realm::CopySrcDstField> srcs(1, src);
        std::vector<Realm::CopySrcDstField> dsts(1, dst);
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
        {
          SmallNameClosure<1>* closure = new SmallNameClosure<1>(nullptr);
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
          closure->record_instance_name(dst_inst, inst_event, 0 /*full*/);
#else
          closure->record_instance_name(dst_inst, unique_event, 0 /*full*/);
#endif
          runtime->profiler->add_fill_request(
              requests, closure, op, precondition);
        }
        const Point<1, coord_t> zero(0);
        const Rect<1, coord_t> rect(zero, zero);
        const ApEvent result(rect.copy(srcs, dsts, requests, precondition));
        if (own_inst)
        {
          if (result.exists())
            dst_inst.destroy(Runtime::protect_event(result));
          else
            dst_inst.destroy();
        }
        return result;
      }
      else
      {
        memcpy(
            const_cast<void*>(get_data()), redop->identity, redop->sizeof_rhs);
        return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent FutureInstance::copy_from(
        FutureInstance* source, Operation* op, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      return copy_from(source, op->get_unique_op_id(), precondition);
    }

    //--------------------------------------------------------------------------
    ApEvent FutureInstance::copy_from(
        FutureInstance* source, UniqueID uid, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      // Only copying the minimum size between the two, this is not very
      // safe, but it's how we deal with upper bound instances so we're
      // just trusing that the caller code is correct
      const size_t copy_size = std::min(size, source->size);
      if (!is_meta_visible || !source->is_meta_visible ||
          (precondition.exists() &&
           !precondition.has_triggered_faultignorant()))
      {
        // We need to offload this to realm
        bool own_src = false, own_dst = false;
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        LgEvent src_event, dst_event;
        PhysicalInstance src_inst =
            source->get_instance(copy_size, src_event, own_src);
        PhysicalInstance dst_inst = get_instance(copy_size, dst_event, own_dst);
#else
        PhysicalInstance src_inst = source->get_instance(copy_size, own_src);
        PhysicalInstance dst_inst = get_instance(copy_size, own_dst);
#endif
        // Should only be writing to instances that this future instance owns
        // Might also happen if we have an "external" (not really external but
        // made-using-malloc instance) that is bigger than the copy and we
        // make an intermediate instance to handle that.
        legion_assert(
            own_instance ||
            (own_dst && external_allocation && (copy_size < size)));
        std::vector<Realm::CopySrcDstField> srcs(1);
        std::vector<Realm::CopySrcDstField> dsts(1);
        srcs.back().set_field(src_inst, 0 /*field id*/, copy_size);
        dsts.back().set_field(dst_inst, 0 /*field id*/, copy_size);
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
        {
          SmallNameClosure<2>* closure = new SmallNameClosure<2>(nullptr);
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
          closure->record_instance_name(src_inst, src_event, 0 /*full*/);
          closure->record_instance_name(dst_inst, dst_event, 0 /*full*/);
#else
          closure->record_instance_name(
              src_inst, source->unique_event, 0 /*full*/);
          closure->record_instance_name(dst_inst, unique_event, 0 /*full*/);
#endif
          runtime->profiler->add_copy_request(
              requests, closure, uid, precondition);
        }
        const Point<1, coord_t> zero(0);
        const Rect<1, coord_t> rect(zero, zero);
        const ApEvent result(rect.copy(srcs, dsts, requests, precondition));
        RtEvent protect;
        if (own_src)
        {
          if (result.exists())
          {
            protect = Runtime::protect_event(result);
            src_inst.destroy(protect);
          }
          else
            src_inst.destroy();
        }
        if (own_dst)
        {
          if (result.exists())
          {
            if (!protect.exists())
              protect = Runtime::protect_event(result);
            dst_inst.destroy(protect);
          }
          else
            dst_inst.destroy();
        }
        return result;
      }
      else
      {
        // We can do this as a straight memcpy, no need to offload to realm
        memcpy(const_cast<void*>(get_data()), source->get_data(), copy_size);
        return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent FutureInstance::reduce_from(
        FutureInstance* source, Operation* op, const ReductionOpID redop_id,
        const ReductionOp* redop, bool exclusive, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (!is_meta_visible || !source->is_meta_visible ||
          (precondition.exists() &&
           !precondition.has_triggered_faultignorant()))
      {
        // We need to offload this to realm
        bool own_src = false, own_dst = false;
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        LgEvent src_event, dst_event;
        PhysicalInstance src_inst =
            source->get_instance(redop->sizeof_rhs, src_event, own_src);
        PhysicalInstance dst_inst =
            get_instance(redop->sizeof_rhs, dst_event, own_dst);
#else
        PhysicalInstance src_inst =
            source->get_instance(redop->sizeof_rhs, own_src);
        PhysicalInstance dst_inst = get_instance(redop->sizeof_rhs, own_dst);
#endif
        // Should only be reducing to instances that this future instance owns
        // Might also happen if we have an "external" (not really external but
        // made-using-malloc instance) that is bigger than the copy and we
        // make an intermediate instance to handle that.
        legion_assert(
            own_instance ||
            (own_dst && external_allocation && (redop->sizeof_rhs < size)));
        std::vector<Realm::CopySrcDstField> srcs(1);
        std::vector<Realm::CopySrcDstField> dsts(1);
        srcs.back().set_field(src_inst, 0 /*field id*/, size);
        dsts.back().set_field(dst_inst, 0 /*field id*/, size);
        dsts.back().set_redop(redop_id, true /*fold*/, exclusive);
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
        {
          SmallNameClosure<2>* closure =
              new SmallNameClosure<2>(nullptr, redop_id);
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
          closure->record_instance_name(src_inst, src_event, 0 /*full*/);
          closure->record_instance_name(dst_inst, dst_event, 0 /*full*/);
#else
          closure->record_instance_name(
              src_inst, source->unique_event, 0 /*full*/);
          closure->record_instance_name(dst_inst, unique_event, 0 /*full*/);
#endif
          runtime->profiler->add_copy_request(
              requests, closure, op, precondition);
        }
        const Point<1, coord_t> zero(0);
        const Rect<1, coord_t> rect(zero, zero);
        const ApEvent result(rect.copy(srcs, dsts, requests, precondition));
        RtEvent protect;
        if (own_src)
        {
          if (result.exists())
          {
            protect = Runtime::protect_event(result);
            src_inst.destroy(protect);
          }
          else
            src_inst.destroy();
        }
        if (own_dst)
        {
          if (result.exists())
          {
            if (!protect.exists())
              protect = Runtime::protect_event(result);
            dst_inst.destroy(protect);
          }
          else
            dst_inst.destroy();
        }
        return result;
      }
      else
      {
        // We can do this as a straight fold, no need to offload to realm
        if (exclusive)
        {
          legion_assert(redop->cpu_fold_excl_fn);
          (redop->cpu_fold_excl_fn)(
              const_cast<void*>(get_data()), 0 /*stride*/, source->get_data(),
              0 /*stride*/, 1 /*count*/, redop->userdata);
        }
        else
        {
          legion_assert(redop->cpu_fold_nonexcl_fn);
          (redop->cpu_fold_nonexcl_fn)(
              const_cast<void*>(get_data()), 0 /*stride*/, source->get_data(),
              0 /*stride*/, 1 /*count*/, redop->userdata);
        }
        return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    const void* FutureInstance::get_data(void)
    //--------------------------------------------------------------------------
    {
      if (size == 0)
        return nullptr;
      const void* result = data.load();
      if (result != nullptr)
        return result;
      if (use_event.exists() && !use_event.has_triggered())
      {
        use_event.wait();
        use_event = RtEvent::NO_RT_EVENT;
      }
      legion_assert(instance.exists());
      result = instance.pointer_untyped(0, size);
      data.store(result);
      return result;
    }

    //--------------------------------------------------------------------------
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
    PhysicalInstance FutureInstance::get_instance(
        size_t needed, LgEvent& inst_event, bool& own_inst)
#else
    PhysicalInstance FutureInstance::get_instance(size_t needed, bool& own_inst)
#endif
    //--------------------------------------------------------------------------
    {
      if (needed != size)
      {
        // The unusual case where we need to make a new instance to reflect
        // a different size than the original
        legion_assert(needed < size);
        const Point<1, coord_t> zero(0);
        const Realm::IndexSpace<1, coord_t> rect_space(
            Realm::Rect<1, coord_t>(zero, zero));
        const Realm::ExternalInstanceResource* alt_resource = resource;
        // Check to see if we already have a resource or not
        if (alt_resource == nullptr)
        {
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
          const PhysicalInstance inst =
              get_instance(size, inst_event, own_inst);
#else
          const PhysicalInstance inst = get_instance(size, own_inst);
#endif
          alt_resource = inst.generate_resource_info(
              rect_space, 0 /*fid*/, false /*read only*/);
          // Note that if you hit this then that likely means that Realm
          // doesn't support 'generate_resource_info' yet for that kind of
          // memory and it probably just needs to be implemented
          legion_assert(alt_resource != nullptr);
        }
        const std::vector<Realm::FieldID> fids(1, 0 /*field id*/);
        const std::vector<size_t> sizes(1, needed);
        const int dim_order[1] = {0};
        const Realm::InstanceLayoutConstraints constraints(fids, sizes, 1);
        Realm::InstanceLayoutGeneric* ilg =
            Realm::InstanceLayoutGeneric::choose_instance_layout<1, coord_t>(
                rect_space, constraints, dim_order);
        const uintptr_t base = reinterpret_cast<uintptr_t>(get_data());
        ilg->alignment_reqd = (base & -base);  // maximum alignment
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        const Realm::UserEvent temp_unique_event =
            Realm::UserEvent::create_user_event();
        temp_unique_event.trigger();
#endif
        // If it is not an external allocation then ignore suggested_memory
        // because we know we're making this on top of an existing instance
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_inst_request(
              requests,
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
              implicit_provenance, LgEvent(temp_unique_event));
#else
              implicit_provenance, unique_event);
#endif
        PhysicalInstance result;
        const RtEvent inst_ready(PhysicalInstance::create_external_instance(
            result, alt_resource->suggested_memory(), *ilg, *alt_resource,
            requests));
        delete ilg;
        if (inst_ready.exists() && (implicit_profiler != nullptr))
          implicit_profiler->record_instance_ready(inst_ready, unique_event);
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        inst_event = LgEvent(temp_unique_event);
#endif
        own_inst = true;
        if (resource == nullptr)
          delete alt_resource;
        if (inst_ready.exists() && !inst_ready.has_triggered())
          inst_ready.wait();
        return result;
      }
      else if (!instance.exists())
      {
        legion_assert(!own_instance);
        legion_assert(external_allocation);
        legion_assert(resource != nullptr);
        legion_assert(!unique_event.exists());
        RtEvent wait_on;
        // Make our instance and see if we lost the race
        const std::vector<Realm::FieldID> fids(1, 0 /*field id*/);
        const std::vector<size_t> sizes(1, size);
        const int dim_order[1] = {0};
        const Realm::InstanceLayoutConstraints constraints(fids, sizes, 1);
        const Point<1, coord_t> zero(0);
        const Realm::IndexSpace<1, coord_t> rect_space(
            Realm::Rect<1, coord_t>(zero, zero));
        Realm::InstanceLayoutGeneric* ilg =
            Realm::InstanceLayoutGeneric::choose_instance_layout<1, coord_t>(
                rect_space, constraints, dim_order);
        const uintptr_t base = reinterpret_cast<uintptr_t>(get_data());
        ilg->alignment_reqd = (base & -base);  // alignment reqd
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
        {
          // Need to try to make a unique event
          Runtime::rename_event(unique_event);
          runtime->profiler->add_inst_request(
              requests, implicit_provenance, unique_event);
        }
        // If it is not an external allocation then ignore suggested_memory
        // because we know we're making this on top of an existing instance
        use_event = RtEvent(PhysicalInstance::create_external_instance(
            instance, resource->suggested_memory(), *ilg, *resource, requests));
        delete ilg;
        own_instance = true;
        if (use_event.exists() && (implicit_profiler != nullptr))
          implicit_profiler->record_instance_ready(use_event, unique_event);
      }
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
      inst_event = unique_event;
#endif
      own_inst = false;
      if (use_event.exists() && !use_event.has_triggered())
      {
        use_event.wait();
        use_event = RtEvent::NO_RT_EVENT;
      }
      return instance;
    }

    //--------------------------------------------------------------------------
    bool FutureInstance::defer_deletion(ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (own_allocation)
      {
        if (precondition.exists() &&
            !precondition.has_triggered_faultignorant())
        {
          DeferDeleteFutureInstanceArgs args(this);
          runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_WORK_PRIORITY,
              Runtime::protect_event(precondition));
          return true;
        }
      }
      else if (own_instance)
      {
        legion_assert(instance.exists());
        if (precondition.exists() &&
            !precondition.has_triggered_faultignorant())
        {
          instance.destroy(Runtime::protect_event(precondition));
          own_instance = false;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool FutureInstance::is_immediate(void) const
    //--------------------------------------------------------------------------
    {
      return !use_event.exists() || use_event.has_triggered();
    }

    //--------------------------------------------------------------------------
    bool FutureInstance::can_pack_by_value(void) const
    //--------------------------------------------------------------------------
    {
      return (is_meta_visible && (size <= LEGION_MAX_RETURN_SIZE));
    }

    //--------------------------------------------------------------------------
    bool FutureInstance::pack_instance(
        Serializer& rez, ApEvent ready_event, bool pack_ownership,
        bool allow_value)
    //--------------------------------------------------------------------------
    {
      rez.serialize(size);
      // Check to see if we can just pass this future instance by value
      if (allow_value && is_meta_visible && (size <= LEGION_MAX_RETURN_SIZE) &&
          (!ready_event.exists() || ready_event.has_triggered_faultignorant()))
      {
        // We can just pass this future by value because we can
        // see it here, it's tiny, and it's ready to be read
        rez.serialize<bool>(true);  // by value
        rez.serialize(data.load(), size);
        // We packed this by value so return true
        return true;
      }
      else
      {
        rez.serialize<bool>(false);  // by value
        rez.serialize(data.load());
        bool dummy_owner = true;
#ifndef LEGION_UNDO_FUTURE_INSTANCE_HACK
        LgEvent dummy_event;
        rez.serialize(get_instance(size, dummy_event, dummy_owner));
#else
        rez.serialize(get_instance(size, dummy_owner));
#endif
        rez.serialize(unique_event);
        // should never end up owning this instance
        legion_assert(!dummy_owner);
        if (pack_ownership)
        {
          legion_assert(own_instance);
          legion_assert(own_allocation);
          rez.serialize<bool>(true);  // own the allocation on the destination
          own_allocation = false;
          // we no longer own this instance either
          own_instance = false;
        }
        else
          rez.serialize<bool>(false);  // do not own allocation on destination
        if (external_allocation)
        {
          rez.serialize<bool>(true);  // external allocation
          rez.serialize(freefunc);
          rez.serialize(freeproc);
        }
        else
          rez.serialize<bool>(false);  // external allocation
        // Not packed by value
        return false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureInstance::pack_null(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    /*static*/ FutureInstance* FutureInstance::unpack_instance(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t size;
      derez.deserialize(size);
      if (size == 0)
        return nullptr;
      bool pass_by_value;
      derez.deserialize<bool>(pass_by_value);
      if (pass_by_value)
      {
        void* data = malloc(size);
        derez.deserialize(data, size);
        return new FutureInstance(data, size, true /*external*/);
      }
      void* data;
      derez.deserialize(data);
      PhysicalInstance instance;
      derez.deserialize(instance);
      LgEvent unique_event;
      derez.deserialize(unique_event);
      RtEvent use_event;
      if (instance.exists())
      {
        use_event = RtEvent(
            instance.fetch_metadata(Processor::get_executing_processor()));
        if (implicit_profiler != nullptr)
          implicit_profiler->record_fetch_metadata(use_event, unique_event);
      }
      bool own_allocation, external_allocation;
      derez.deserialize<bool>(own_allocation);
      derez.deserialize<bool>(external_allocation);
      if (external_allocation)
      {
        void (*freefunc)(const Realm::ExternalInstanceResource&);
        derez.deserialize(freefunc);
        Processor proc;
        derez.deserialize(proc);
        return new FutureInstance(
            data, size, own_allocation, nullptr /*resource*/, freefunc, proc,
            unique_event, instance, use_event);
      }
      else
        return new FutureInstance(
            data, size, false /*external*/, own_allocation, unique_event,
            instance, Processor::NO_PROC, use_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ bool FutureInstance::check_meta_visible(Memory memory)
    //--------------------------------------------------------------------------
    {
      // Common case, if it is the local system memory, we can see it
      if (runtime->runtime_system_memory == memory)
        return true;
      // If it's not in the local process, we definitely can't see it
      if (memory.address_space() != runtime->address_space)
        return false;
      // switch on the memory kind and see if there are any we support
      switch (memory.kind())
      {
        case Memory::GLOBAL_MEM:
        case Memory::SYSTEM_MEM:
        case Memory::REGDMA_MEM:
        case Memory::SOCKET_MEM:
        case Memory::Z_COPY_MEM:
          return true;
        default:
          break;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ FutureInstance* FutureInstance::create_local(
        const void* value, size_t size, bool own)
    //--------------------------------------------------------------------------
    {
      // Copy the data into a buffer we own if we don't already
      if (!own)
      {
        void* buffer = malloc(size);
        memcpy(buffer, value, size);
        value = buffer;
        own = true;
      }
      return new FutureInstance(value, size, true /*external*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FreeExternalAllocation::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor freeproc;
      derez.deserialize(freeproc);
      void (*freefunc)(const Realm::ExternalInstanceResource&);
      derez.deserialize(freefunc);
      if (freefunc == nullptr)
        freefunc = FutureInstance::free_host_memory;
      PhysicalInstance instance;
      derez.deserialize(instance);
      LgEvent unique_event;
      derez.deserialize(unique_event);
      const RtEvent use_event(instance.fetch_metadata(freeproc));
      if (implicit_profiler != nullptr)
        implicit_profiler->record_fetch_metadata(use_event, unique_event);
      FutureInstance::FreeExternalArgs args(
          nullptr /*no resource*/, freefunc, instance);
      if (freeproc.exists())
        runtime->issue_application_processor_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, freeproc, use_event);
      else
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_WORK_PRIORITY, use_event);
    }

    //--------------------------------------------------------------------------
    FutureInstance::FreeExternalArgs::FreeExternalArgs(
        const Realm::ExternalInstanceResource* r,
        void (*func)(const Realm::ExternalInstanceResource&),
        PhysicalInstance inst)
      : LgTaskArgs<FreeExternalArgs>(true, true),
        resource((r == nullptr) ? r : r->clone()), freefunc(func),
        instance(inst)
    //--------------------------------------------------------------------------
    {
      legion_assert((resource != nullptr) || instance.exists());
    }

    //--------------------------------------------------------------------------
    void FutureInstance::FreeExternalArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      const Realm::ExternalInstanceResource* res = resource;
      if (resource == nullptr)
      {
        const Point<1, coord_t> zero(0);
        const Realm::IndexSpace<1, coord_t> rect_space(
            Realm::Rect<1, coord_t>(zero, zero));
        res = instance.generate_resource_info(
            rect_space, 0 /*fid*/, true /*read only*/);
      }
      (*freefunc)(*res);
      if (instance.exists())
        instance.destroy();
      delete res;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureInstance::free_host_memory(
        const Realm::ExternalInstanceResource& resource)
    //--------------------------------------------------------------------------
    {
      const Realm::ExternalMemoryResource& allocation =
          static_cast<const Realm::ExternalMemoryResource&>(resource);
      free(reinterpret_cast<void*>(allocation.base));
    }

    //--------------------------------------------------------------------------
    void FutureInstance::DeferDeleteFutureInstanceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      delete instance;
    }

  }  // namespace Internal
}  // namespace Legion
