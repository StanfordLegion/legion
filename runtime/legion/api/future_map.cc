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

#include "legion/contexts/inner.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/api/future_map_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/operations/operation.h"
#include "legion/utilities/provenance.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Future Map
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  FutureMap::FutureMap(void) : impl(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  FutureMap::FutureMap(const FutureMap& map) : impl(map.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_base_gc_ref(Internal::APPLICATION_REF);
  }

  //--------------------------------------------------------------------------
  FutureMap::FutureMap(FutureMap&& map) noexcept : impl(map.impl)
  //--------------------------------------------------------------------------
  {
    map.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  FutureMap::FutureMap(Internal::FutureMapImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_base_gc_ref(Internal::APPLICATION_REF);
  }

  //--------------------------------------------------------------------------
  FutureMap::~FutureMap(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) &&
        impl->remove_base_gc_ref(Internal::APPLICATION_REF))
      delete impl;
  }

  //--------------------------------------------------------------------------
  FutureMap& FutureMap::operator=(const FutureMap& rhs)
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
  FutureMap& FutureMap::operator=(FutureMap&& rhs) noexcept
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
  std::size_t FutureMap::hash(void) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      return std::hash<DistributedID>{}(impl->did);
    else
      return std::hash<DistributedID>{}(0);
  }

  //--------------------------------------------------------------------------
  Future FutureMap::get_future(const DomainPoint& point) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->get_future(point, false /*internal*/);
  }

  //--------------------------------------------------------------------------
  void FutureMap::get_void_result(
      const DomainPoint& point, bool silence_warnings,
      const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->get_void_result(point, silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  void FutureMap::wait_all_results(
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->wait_all_results(silence_warnings, warning_string);
  }

  //--------------------------------------------------------------------------
  Domain FutureMap::get_future_map_domain(void) const
  //--------------------------------------------------------------------------
  {
    if (impl == nullptr)
      return Domain::NO_DOMAIN;
    else
      return impl->get_domain();
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Future Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(
        TaskContext* ctx, Operation* o, IndexSpaceNode* domain,
        DistributedID did, Provenance* prov, bool register_now,
        CollectiveMapping* mapping)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_MAP_DC), register_now,
            mapping),
        context(ctx), op(o), op_gen(o->get_generation()),
        op_depth(o->get_context()->get_depth()), op_uid(o->get_unique_op_id()),
        blocking_index(o->get_context()->get_next_blocking_index()),
        provenance(prov), future_map_domain(domain)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain != nullptr);
      future_map_domain->add_nested_valid_ref(did);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Future Map %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(
        TaskContext* ctx, IndexSpaceNode* d, DistributedID did,
        uint64_t blocking, const std::optional<uint64_t>& index,
        Provenance* prov, bool register_now, CollectiveMapping* mapping)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_MAP_DC), register_now,
            mapping),
        context(ctx), op(nullptr), op_gen(0), op_depth(0), op_uid(0),
        blocking_index(blocking), provenance(prov), future_map_domain(d),
        remote_context_index(index)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain != nullptr);
      future_map_domain->add_nested_valid_ref(did);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Future Map %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(
        TaskContext* ctx, Operation* o, uint64_t index, GenerationID gen,
        int depth, UniqueID uid, IndexSpaceNode* domain, DistributedID did,
        Provenance* prov)
      : DistributedCollectable(
            LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_MAP_DC)),
        context(ctx), op(o), op_gen(gen), op_depth(depth), op_uid(uid),
        blocking_index(index), provenance(prov), future_map_domain(domain)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain != nullptr);
      future_map_domain->add_nested_valid_ref(did);
      if (provenance != nullptr)
        provenance->add_reference();
#ifdef LEGION_GC
      log_garbage.info(
          "GC Future Map %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::~FutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
        if (future.second->remove_nested_resource_ref(did))
          delete future.second;
      futures.clear();
      if (future_map_domain->remove_nested_valid_ref(did))
        delete future_map_domain;
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::notify_local(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
        future.second->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    Domain FutureMapImpl::get_domain(void) const
    //--------------------------------------------------------------------------
    {
      return future_map_domain->get_tight_domain();
    }

    //--------------------------------------------------------------------------
    std::optional<uint64_t> FutureMapImpl::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        return remote_context_index;
      if (op != nullptr)
        return op->get_context_index(op_gen);
      else
        return std::optional<uint64_t>();
    }

    //--------------------------------------------------------------------------
    Future FutureMapImpl::get_future(
        const DomainPoint& point, bool internal, RtEvent* wait_on)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
      {
        if (!future_map_domain->contains_point(point))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Invalid request for a point " << point
                << " which is not contained in the domain of a " << *this
                << " in " << *context << ".";
          error.raise();
        }
      }
      if (!is_owner())
      {
        // See if we already have it
        {
          AutoLock fm_lock(future_map_lock, false /*exclusive*/);
          std::map<DomainPoint, FutureImpl*>::const_iterator finder =
              futures.find(point);
          if (finder != futures.end())
            return Future(finder->second);
        }
        // Make an event for when we have the answer
        RtUserEvent future_ready_event = Runtime::create_rt_user_event();
        // If not send a message to get it
        FutureMapFutureRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(point);
          rez.serialize(future_ready_event);
          rez.serialize<bool>(internal);
        }
        rez.dispatch(owner_space);
        if (wait_on != nullptr)
        {
          *wait_on = future_ready_event;
          return Future();
        }
        future_ready_event.wait();
        // When we wake up it should be here
        AutoLock fm_lock(future_map_lock, false /*exclusive*/);
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            futures.find(point);
        legion_assert(finder != futures.end());
        return Future(finder->second);
      }
      else
      {
        AutoLock fm_lock(future_map_lock);
        // Check to see if we already have a future for the point
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            futures.find(point);
        if (finder != futures.end())
          return Future(finder->second);
        // Otherwise we need a future from the context to use for
        // the point that we will fill in later
        FutureImpl* result = new FutureImpl(
            context, true /*register*/, runtime->get_available_distributed_id(),
            op, op_gen, ContextCoordinate(blocking_index, point), op_uid,
            op_depth, provenance);
        result->add_nested_gc_ref(did);
        result->add_nested_resource_ref(did);
        futures[point] = result;
        LegionSpy::log_future_creation(op_uid, result->did, point);
        return Future(result);
      }
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_future(const DomainPoint& point, FutureImpl* impl)
    //--------------------------------------------------------------------------
    {
      // Add the reference first and then set the future
      impl->add_nested_gc_ref(did);
      impl->add_nested_resource_ref(did);
      AutoLock fm_lock(future_map_lock);
      legion_assert(futures.find(point) == futures.end());
      futures[point] = impl;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::get_void_result(
        const DomainPoint& point, bool silence_warnings,
        const char* warning_string)
    //--------------------------------------------------------------------------
    {
      Future f = get_future(point, false /*internal*/);
      f.get_void_result(silence_warnings, warning_string);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::wait_all_results(
        bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      if (implicit_context != context)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Invalid request to wait on all results of a future map "
              << *this << " outside of " << *context << " task context where "
              << "it was created. Future maps can only be used in the task "
              << "where they were made.";
        error.raise();
      }
      if (runtime->runtime_warnings && !silence_warnings &&
          (context != nullptr) && !context->is_leaf_context())
      {
        Warning warning;
        warning << "Waiting for all futures of " << *this << " in non-leaf "
                << *context << " is a violation of Legion's "
                << "deferred execution model best practices. You may notice a "
                << "severe performance degredation.";
        if (warning_string != nullptr)
          warning << " Warning string: " << warning_string;
        warning.raise();
      }
      context->record_blocking_call(blocking_index);
      if (op != nullptr)
        context->wait_on_future_map(this, op->get_commit_event(op_gen));
    }

    //--------------------------------------------------------------------------
    bool FutureMapImpl::reset_all_futures(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      // TODO: send messages to all the remote copies of this
      std::abort();
      bool result = false;
      AutoLock fm_lock(future_map_lock);
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
        if (future.second->reset_future())
          result = true;
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::pack_future_map(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);
      if ((target != owner_space) && ((collective_mapping == nullptr) ||
                                      !collective_mapping->contains(target)))
      {
        rez.serialize<bool>(true);  // can create
        rez.serialize(future_map_domain->handle);
        rez.serialize(blocking_index);
        rez.serialize(get_context_index());
        if (provenance != nullptr)
          provenance->serialize(rez);
        else
          Provenance::serialize_null(rez);
      }
      else
        rez.serialize<bool>(false);  // cannot make it, need to wait
      pack_global_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ FutureMap FutureMapImpl::unpack_future_map(
        Deserializer& derez, AddressSpaceID source, TaskContext* ctx)
    //--------------------------------------------------------------------------
    {
      DistributedID future_map_did;
      derez.deserialize(future_map_did);
      if (future_map_did == 0)
        return FutureMap();
      bool can_create;
      derez.deserialize<bool>(can_create);
      if (!can_create)
      {
        // Have to wait to find this one since it is created collectively
        FutureMap result(static_cast<FutureMapImpl*>(
            runtime->find_distributed_collectable(future_map_did)));
        // Make sure we know about all the remote instances before
        // decrementing to avoid races with registration/unpack
        if (result.impl->is_owner() &&
            ((result.impl->collective_mapping == nullptr) ||
             (!result.impl->collective_mapping->contains(source))))
          result.impl->update_remote_instances(source);
        result.impl->unpack_global_ref();
        return result;
      }
      IndexSpace future_map_domain;
      derez.deserialize(future_map_domain);
      uint64_t coordinate;
      derez.deserialize(coordinate);
      std::optional<uint64_t> index;
      derez.deserialize(index);
      AutoProvenance provenance(
          Provenance::deserialize(derez), true /*has ref*/);
      FutureMap result(runtime->find_or_create_future_map(
          future_map_did, ctx, coordinate, future_map_domain, provenance,
          index));
      result.impl->unpack_global_ref();
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::get_all_futures(
        std::map<DomainPoint, FutureImpl*>& others)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      Domain domain = future_map_domain->get_tight_domain();
      const size_t needed = domain.get_volume();
      AutoLock fm_lock(future_map_lock);
      legion_assert(futures.size() <= needed);
      if (futures.size() < needed)
      {
        fm_lock.release();
        std::vector<RtEvent> ready_events;
        for (Domain::DomainPointIterator itr(domain); itr; itr++)
        {
          RtEvent ready;
          get_future(itr.p, true /*internal only*/, &ready);
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
        if (!ready_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
        fm_lock.reacquire();
      }
      others = futures;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_all_futures(
        const std::map<DomainPoint, Future>& others)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we're initializing
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
      {
        future.second->remove_nested_gc_ref(did);
        if (future.second->remove_nested_resource_ref(did))
          delete future.second;
      }
      futures.clear();
      for (const std::pair<const DomainPoint, Future>& future_pair : others)
      {
        FutureImpl* impl = future_pair.second.impl;
        impl->add_nested_resource_ref(did);
        impl->add_nested_gc_ref(did);
        futures[future_pair.first] = impl;
      }
    }

    //--------------------------------------------------------------------------
    FutureImpl* FutureMapImpl::find_local_future(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain->contains_point(point));
      Future result = get_future(point, true /*internal only*/);
      return result.impl;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::get_shard_local_futures(
        ShardID shard, std::map<DomainPoint, FutureImpl*>& others)
    //--------------------------------------------------------------------------
    {
      // This is only called on this kind of future map when we know we
      // already have all the futures so there's no need to wait or lock
      others = futures;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::register_dependence(Operation* consumer_op)
    //--------------------------------------------------------------------------
    {
      if (op == nullptr)
        return;
      // Only record dependences on things from the same context
      // We know futures can never flow up the task tree so the
      // only way they have the same depth is if they are from
      // the same parent context
      InnerContext* context = consumer_op->get_context();
      const int consumer_depth = context->get_depth();
      legion_assert(consumer_depth >= op_depth);
      if (consumer_depth == op_depth)
      {
        consumer_op->register_dependence(op, op_gen);
        LegionSpy::log_future_dependence(
            context->get_unique_id(), op_uid, consumer_op->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FutureMapImpl::find_pointwise_dependence(
        const DomainPoint& point, int context_depth, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      legion_assert(context_depth >= op_depth);
      const std::optional<uint64_t> context_index = get_context_index();
      if (!context_index || (context_depth != op_depth))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      if (!is_owner())
      {
        // Make an event and send it back to the owner node
        if (!to_trigger.exists())
          to_trigger = Runtime::create_rt_user_event();
        FutureMapPointwise rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(point);
          rez.serialize(context_depth);
          rez.serialize(to_trigger);
        }
        rez.dispatch(owner_space);
        return to_trigger;
      }
      else
        return context->find_pointwise_dependence(
            *context_index, point, 0 /*shard*/, false /*intra space*/,
            to_trigger);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::record_future_map_registered(void)
    //--------------------------------------------------------------------------
    {
      // Similar to DistributedCollectable::register_with_runtime but
      // we don't actually need to do the registration since we know
      // it has already been done
      legion_assert(!registered_with_runtime);
      registered_with_runtime = true;
      // We always have a global unpack reference from
      // FutureMapImpl::unpack_future_map and that ensures that we can
      // send the registration method without blocking since the
      // distributed collectable cannot collect itself until it finds
      // the unpacked global reference
      if (!is_owner())
        send_remote_registration(true /*has global reference*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureMapFutureRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DomainPoint point;
      derez.deserialize(point);
      RtUserEvent done;
      derez.deserialize(done);
      bool internal;
      derez.deserialize(internal);

      // Should always find it since this is the owner node except in the
      // replicated case in which case a shard on this node might not have
      // actually made it yet, so wait in that case
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureMapImpl* impl = legion_safe_cast<FutureMapImpl*>(dc);
      Future f = impl->get_future(point, internal);
      FutureMapFutureResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(point);
        rez.serialize(f.impl->did);
        f.impl->pack_global_ref();
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::process_future_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ContextCoordinate coordinate(blocking_index);
      derez.deserialize(coordinate.index_point);
      DistributedID future_did;
      derez.deserialize(future_did);
      RtEvent dummy;
      FutureImpl* impl = runtime->find_or_create_future(
          future_did, context->did, coordinate, provenance,
          true /*has global ref*/, dummy, op, op_gen, op_uid, op_depth);
      set_future(coordinate.index_point, impl);
      impl->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureMapFutureResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      // Should always find it since this is the source node
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureMapImpl* impl = legion_safe_cast<FutureMapImpl*>(dc);
      // Add it to the map
      impl->process_future_response(derez);
      // Trigger the done event
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureMapPointwise::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      // Should always find it since this is the source node
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      FutureMapImpl* impl = legion_safe_cast<FutureMapImpl*>(dc);
      DomainPoint point;
      derez.deserialize(point);
      int context_depth;
      derez.deserialize(context_depth);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      impl->find_pointwise_dependence(point, context_depth, to_trigger);
    }

    /////////////////////////////////////////////////////////////
    // Transform Future Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TransformFutureMapImpl::TransformFutureMapImpl(
        FutureMapImpl* prev, IndexSpaceNode* domain,
        PointTransformFunctor* func, bool own_func, Provenance* prov)
      : FutureMapImpl(
            prev->context, prev->op, prev->blocking_index, prev->op_gen,
            prev->op_depth, prev->op_uid, domain,
            runtime->get_available_distributed_id(), prov),
        previous(prev), functor(func), own_functor(own_func)
    //--------------------------------------------------------------------------
    {
      prev->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    TransformFutureMapImpl::~TransformFutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      if (previous->remove_nested_gc_ref(did))
        delete previous;
      if (own_functor)
        delete functor;
    }

    //--------------------------------------------------------------------------
    bool TransformFutureMapImpl::is_replicate_future_map(void) const
    //--------------------------------------------------------------------------
    {
      return previous->is_replicate_future_map();
    }

    //--------------------------------------------------------------------------
    Future TransformFutureMapImpl::get_future(
        const DomainPoint& point, bool internal_only, RtEvent* wait_on)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain->contains_point(point));
      Domain domain = future_map_domain->get_tight_domain();
      Domain range = previous->future_map_domain->get_tight_domain();
      const DomainPoint transformed =
          functor->transform_point(point, domain, range);
      legion_assert(previous->future_map_domain->contains_point(transformed));
      return previous->get_future(transformed, internal_only, wait_on);
    }

    //--------------------------------------------------------------------------
    void TransformFutureMapImpl::get_all_futures(
        std::map<DomainPoint, FutureImpl*>& futures)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint, FutureImpl*> previous_futures;
      previous->get_all_futures(previous_futures);
      Domain domain = future_map_domain->get_tight_domain();
      Domain range = previous->future_map_domain->get_tight_domain();
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        const DomainPoint transformed =
            functor->transform_point(itr.p, domain, range);
        legion_assert(previous->future_map_domain->contains_point(transformed));
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            previous_futures.find(transformed);
        legion_assert(finder != previous_futures.end());
        futures[itr.p] = finder->second;
      }
    }

    //--------------------------------------------------------------------------
    void TransformFutureMapImpl::wait_all_results(
        bool silence_warnings, const char* warning_string)
    //--------------------------------------------------------------------------
    {
      previous->wait_all_results(silence_warnings, warning_string);
    }

    //--------------------------------------------------------------------------
    FutureImpl* TransformFutureMapImpl::find_local_future(
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain->contains_point(point));
      Domain domain = future_map_domain->get_tight_domain();
      Domain range = previous->future_map_domain->get_tight_domain();
      const DomainPoint transformed =
          functor->transform_point(point, domain, range);
      legion_assert(previous->future_map_domain->contains_point(transformed));
      return previous->find_local_future(transformed);
    }

    //--------------------------------------------------------------------------
    void TransformFutureMapImpl::get_shard_local_futures(
        ShardID shard, std::map<DomainPoint, FutureImpl*>& futures)
    //--------------------------------------------------------------------------
    {
      std::map<DomainPoint, FutureImpl*> previous_futures;
      previous->get_shard_local_futures(shard, previous_futures);
      Domain domain = future_map_domain->get_tight_domain();
      Domain range = previous->future_map_domain->get_tight_domain();
      if (functor->is_invertible())
      {
        for (const std::pair<const DomainPoint, FutureImpl*>& future :
             previous_futures)
        {
          const DomainPoint inverted =
              functor->invert_point(future.first, domain, range);
          legion_assert(future_map_domain->contains_point(inverted));
          futures[inverted] = future.second;
        }
      }
      else
      {
        // Not invertible so do it the hard way by enumerating all
        // the points and seeing which ones we find
        for (Domain::DomainPointIterator itr(domain); itr; itr++)
        {
          const DomainPoint transformed =
              functor->transform_point(itr.p, domain, range);
          legion_assert(
              previous->future_map_domain->contains_point(transformed));
          std::map<DomainPoint, FutureImpl*>::const_iterator finder =
              previous_futures.find(transformed);
          if (finder != previous_futures.end())
            futures[itr.p] = finder->second;
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent TransformFutureMapImpl::find_pointwise_dependence(
        const DomainPoint& point, int context_depth, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map_domain->contains_point(point));
      const Domain domain = future_map_domain->get_tight_domain();
      const Domain range = previous->future_map_domain->get_tight_domain();
      const DomainPoint transformed =
          functor->transform_point(point, domain, range);
      legion_assert(previous->future_map_domain->contains_point(transformed));
      return previous->find_pointwise_dependence(
          transformed, context_depth, to_trigger);
    }

    /////////////////////////////////////////////////////////////
    // Repl Future Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplFutureMapImpl::ReplFutureMapImpl(
        TaskContext* ctx, ShardManager* man, Operation* op,
        IndexSpaceNode* domain, IndexSpaceNode* shard_dom, DistributedID did,
        Provenance* prov, CollectiveMapping* mapping)
      : FutureMapImpl(
            ctx, op, domain, did, prov, false /*register now*/, mapping),
        shard_manager(man), shard_domain(shard_dom), op_depth(ctx->get_depth()),
        sharding_function(nullptr), own_sharding_function(false),
        collective_performed(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_domain != nullptr);
      shard_domain->add_nested_valid_ref(did);
      shard_manager->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    ReplFutureMapImpl::ReplFutureMapImpl(
        TaskContext* ctx, ShardManager* man, IndexSpaceNode* domain,
        IndexSpaceNode* shard_dom, DistributedID did, uint64_t coord,
        std::optional<uint64_t> ctx_index, Provenance* prov,
        CollectiveMapping* mapping)
      : FutureMapImpl(
            ctx, domain, did, coord, ctx_index, prov, false /*register now*/,
            mapping),
        shard_manager(man), shard_domain(shard_dom), op_depth(ctx->get_depth()),
        sharding_function(nullptr), own_sharding_function(false),
        collective_performed(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_domain != nullptr);
      shard_domain->add_nested_valid_ref(did);
      shard_manager->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    ReplFutureMapImpl::~ReplFutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      if (shard_domain->remove_nested_valid_ref(did))
        delete shard_domain;
      if (shard_manager->remove_nested_gc_ref(did))
        delete shard_manager;
      if (own_sharding_function)
        delete sharding_function.load();
    }

    //--------------------------------------------------------------------------
    Future ReplFutureMapImpl::get_future(
        const DomainPoint& point, bool internal, RtEvent* wait_on)
    //--------------------------------------------------------------------------
    {
      // Do a quick check to see if we've already got it
      {
        AutoLock f_lock(future_map_lock, false /*exclusive*/);
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            futures.find(point);
        if (finder != futures.end())
          return Future(finder->second);
      }
      // Now we need to figure out which shard we're on, see if we know
      // the sharding function yet, if not we have to wait
      if (sharding_function == nullptr)
      {
        RtEvent wait_on = get_sharding_function_ready();
        if (wait_on.exists() && !wait_on.has_triggered())
        {
          // This is a bit unfortunate but should be a pretty rare
          // situation so we just need to make it correct right now
          // Index tasks and other operations going through the pipeline
          // don't pick their sharding functor until they start going
          // through the pipeline, so we need to tell auto-tracing to
          // not buffer the producer of this future map to avoid a hang.
          // This doesn't apply to invalidating traces though since we're
          // not actually blocking waiting on a result computed by the
          // trace so we set the flag saying this doesn't invalidate
          // the trace replay.
          if (!internal)
            context->record_blocking_call(
                blocking_index, false /*invalidate trace*/);
          wait_on.wait();
        }
      }
      Domain domain = shard_domain->get_tight_domain();
      const ShardID owner_shard =
          sharding_function.load()->find_owner(point, domain);
      // Figure out which node has this future
      const AddressSpaceID space = shard_manager->get_shard_space(owner_shard);
      if (space != runtime->address_space)
      {
        // Make an event for when we have the answer
        RtUserEvent future_ready_event = Runtime::create_rt_user_event();
        // If not send a message to get it
        FutureMapFutureRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(point);
          rez.serialize(future_ready_event);
          rez.serialize<bool>(internal);
        }
        rez.dispatch(space);
        if (wait_on != nullptr)
        {
          *wait_on = future_ready_event;
          return Future();
        }
        future_ready_event.wait();
        // Now we can wake up see if we found it
        AutoLock f_lock(future_map_lock, false /*exclusive*/);
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            futures.find(point);
        legion_assert(finder != futures.end());
        return Future(finder->second);
      }
      else  // If we're the owner shard we can just do the normal thing
      {
        AutoLock fm_lock(future_map_lock);
        // Check to see if we already have a future for the point
        std::map<DomainPoint, FutureImpl*>::const_iterator finder =
            futures.find(point);
        if (finder != futures.end())
          return Future(finder->second);
        // Otherwise we need a future from the context to use for
        // the point that we will fill in later
        FutureImpl* result = new FutureImpl(
            context, true /*register*/, runtime->get_available_distributed_id(),
            op, op_gen, ContextCoordinate(blocking_index, point), op_uid,
            op_depth, provenance);
        result->add_nested_gc_ref(did);
        result->add_nested_resource_ref(did);
        futures[point] = result;
        LegionSpy::log_future_creation(op_uid, result->did, point);
        return Future(result);
      }
    }

    //--------------------------------------------------------------------------
    void ReplFutureMapImpl::get_all_futures(
        std::map<DomainPoint, FutureImpl*>& others)
    //--------------------------------------------------------------------------
    {
      // We know this call only comes from the application so we don't
      // need to worry about thread safety
      if (!collective_performed)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(implicit_context);
        for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
        {
          ReplicateContext::HashVerifier hasher(
              repl_ctx, runtime->safe_control_replication > 1, i > 0);
          hasher.hash(
              ReplicateContext::REPLICATE_FUTURE_MAP_GET_ALL_FUTURES, __func__);
          repl_ctx->hash_future_map(hasher, FutureMap(this), "future map");
          if (hasher.verify(__func__))
            break;
        }
        std::map<DomainPoint, FutureImpl*> local_futures;
        get_shard_local_futures(repl_ctx->owner_shard->shard_id, local_futures);
        FutureNameExchange collective(repl_ctx, COLLECTIVE_LOC_32);
        collective.exchange_future_names(local_futures);
        AutoLock f_lock(future_map_lock);
        for (const std::pair<const DomainPoint, FutureImpl*>& future :
             local_futures)
        {
          if (futures.insert(future).second)
          {
            future.second->add_nested_resource_ref(did);
            future.second->add_nested_gc_ref(did);
          }
        }
        collective_performed = true;
      }
      // No need for the lock now that we know that we have all of them
      others = futures;
    }

    //--------------------------------------------------------------------------
    void ReplFutureMapImpl::get_shard_local_futures(
        ShardID local_shard, std::map<DomainPoint, FutureImpl*>& others)
    //--------------------------------------------------------------------------
    {
      Domain sharding_domain = shard_domain->get_tight_domain();
      if (sharding_function == nullptr)
      {
        RtEvent wait_on = get_sharding_function_ready();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      ShardingFunction* function = sharding_function.load();
      IndexSpace local_space = function->find_shard_space(
          local_shard, future_map_domain, shard_domain->handle, provenance);
      // Handle the case where there are no points for the local shard
      if (!local_space.exists())
        return;
      IndexSpaceNode* local_points = runtime->get_node(local_space);
      Domain domain = local_points->get_tight_domain();
      std::vector<RtEvent> ready_events;
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        const ShardID shard = function->find_owner(itr.p, sharding_domain);
        if (shard == local_shard)
        {
          RtEvent ready;
          others[itr.p] = get_future(itr.p, true /*internal*/, &ready).impl;
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
      }
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    bool ReplFutureMapImpl::set_sharding_function(
        ShardingFunction* function, bool own_function)
    //--------------------------------------------------------------------------
    {
      // Deduplicate sharding function sets across multiple shards
      RtUserEvent to_trigger;
      {
        AutoLock fm_lock(future_map_lock);
        if (sharding_function == nullptr)
        {
          sharding_function = function;
          own_sharding_function = own_function;
          to_trigger = sharding_function_ready;
        }
        else
          return false;
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplFutureMapImpl::get_sharding_function_ready(void)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_map_lock);
      if (sharding_function == nullptr)
      {
        if (!sharding_function_ready.exists())
          sharding_function_ready = Runtime::create_rt_user_event();
        return sharding_function_ready;
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplFutureMapImpl::find_pointwise_dependence(
        const DomainPoint& point, int context_depth, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      legion_assert(context_depth >= op_depth);
      const std::optional<uint64_t> context_index = get_context_index();
      if (!context_index || (context_depth != op_depth))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      const Domain sharding_domain = shard_domain->get_tight_domain();
      if (sharding_function.load() == nullptr)
      {
        RtEvent wait_on = get_sharding_function_ready();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      const ShardID owner_shard =
          sharding_function.load()->find_owner(point, sharding_domain);
      return context->find_pointwise_dependence(
          *context_index, point, owner_shard, false /*intra space*/,
          to_trigger);
    }

    /////////////////////////////////////////////////////////////
    // Future Name Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureNameExchange::FutureNameExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FutureNameExchange::~FutureNameExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FutureNameExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      const AddressSpaceID target_space = manager->get_mapping()[target];
      for (const std::pair<const DomainPoint, Future>& result_pair : results)
      {
        rez.serialize(result_pair.first);
        if (result_pair.second.impl != nullptr)
          result_pair.second.impl->pack_future(rez, target_space);
        else
          rez.serialize<DistributedID>(0);
      }
    }

    //--------------------------------------------------------------------------
    void FutureNameExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_futures;
      derez.deserialize(num_futures);
      for (unsigned idx = 0; idx < num_futures; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        results[point] = FutureImpl::unpack_future(derez);
      }
    }

    //--------------------------------------------------------------------------
    void FutureNameExchange::exchange_future_names(
        std::map<DomainPoint, FutureImpl*>& futures)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
        results[future.first] = Future(future.second);
      perform_collective_sync();
      for (const std::pair<const DomainPoint, Future>& result_pair : results)
        futures.insert(
            std::make_pair(result_pair.first, result_pair.second.impl));
    }

  }  // namespace Internal
}  // namespace Legion
