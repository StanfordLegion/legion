/* Copyright 2016 Stanford University, NVIDIA Corporation
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


#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "legion_trace.h"
#include "legion_utilities.h"
#include "region_tree.h"
#include "default_mapper.h"
#include "legion_spy.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"
#include "garbage_collection.h"
#ifdef HANG_TRACE
#include <signal.h>
#include <execinfo.h>
#endif
#include <unistd.h> // sleep for warnings

namespace LegionRuntime {

#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
  namespace Accessor {
    namespace AccessorType {
#ifdef PRIVILEGE_CHECKS
      const char* find_privilege_task_name(void *impl)
      {
        // Have to bounce this off the Runtime because C++ is stupid
        return HighLevel::Internal::find_privilege_task_name(impl); 
      }
#endif
#ifdef BOUNDS_CHECKS
      void check_bounds(void *impl, ptr_t ptr)
      {
        // Have to bounce this off the Runtime because C++ is stupid 
        HighLevel::Internal::check_bounds(impl, ptr);
      }
      void check_bounds(void *impl, const Realm::DomainPoint &dp)
      {
        // Have to bounce this off the Runtime because C++ is stupid
        HighLevel::Internal::check_bounds(impl, dp);
      }
#endif
    };
  };
#endif
 
  namespace HighLevel {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Argument Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::Impl::Impl(void)
      : Collectable(), next(NULL), 
        store(legion_new<ArgumentMapStore>()), frozen(false)
    //--------------------------------------------------------------------------
    {
      // This is the first impl in the chain so we make the store
      // then we add a reference to the store so it isn't collected
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl::Impl(ArgumentMapStore *st)
      : Collectable(), next(NULL), store(st), frozen(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl::Impl(ArgumentMapStore *st,
                            const std::map<DomainPoint,TaskArgument> &args)
      : Collectable(), arguments(args), next(NULL), store(st), frozen(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl::Impl(const Impl &impl)
      : Collectable(), next(NULL), store(NULL), frozen(false)
    //--------------------------------------------------------------------------
    {
      // This should never ever be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    ArgumentMap::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      if (next != NULL)
      {
        // Remove our reference to the next thing in the list
        // and garbage collect it if necessary
        if (next->remove_reference())
        {
          legion_delete(next);
        }
      }
      else
      {
        // We're the last one in the chain being deleted,
        // so we have to delete the store as well
        legion_delete(store);
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl& ArgumentMap::Impl::operator=(const Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // This should never ever be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::Impl::has_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      // Go to the end of the list
      if (next == NULL)
      {
        return (arguments.find(point) != arguments.end());
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen);
#endif
        return next->has_point(point);
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::Impl::set_point(const DomainPoint &point, 
                                      const TaskArgument &arg,
                                      bool replace)
    //--------------------------------------------------------------------------
    {
      // Go to the end of the list
      if (next == NULL)
      {
        // Check to see if we're frozen or not, note we don't really need the 
        // lock here since there is only one thread that is traversing the list.
        // The only multi-threaded part is with the references and we clearly 
        // have reference if we're traversing this list.
        if (frozen)
        {
          next = clone();
          next->set_point(point, arg, replace);
        }
        else // Not frozen so just do the update
        {
          // If we're trying to replace, check to see if
          // we can find the old point
          if (replace)
          {
            std::map<DomainPoint,TaskArgument>::iterator finder = 
              arguments.find(point);
            if (finder != arguments.end())
            {
              finder->second = store->add_arg(arg);
              return;
            }
          }
          arguments[point] = store->add_arg(arg);
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen); // this should be frozen if there is a next
#endif
        next->set_point(point, arg, replace);
      }
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::Impl::remove_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (next == NULL)
      {
        if (frozen)
        {
          next = clone();
          return next->remove_point(point);
        }
        else
        {
          std::map<DomainPoint,TaskArgument>::iterator finder = 
                                            arguments.find(point);
          if (finder != arguments.end())
          {
            arguments.erase(finder);
            return true;
          }
          return false;
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen); // this should be frozen if there is a next
#endif
        return next->remove_point(point);
      }
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMap::Impl::get_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
      if (next == NULL)
      {
        std::map<DomainPoint,TaskArgument>::const_iterator finder = 
                                                arguments.find(point);
        if (finder != arguments.end())
          return finder->second;
        // Couldn't find it so return an empty argument
        return TaskArgument();
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(frozen); // this should be frozen if there is a next
#endif
        return next->get_point(point);
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::Impl::pack_arguments(Serializer &rez, const Domain &dom)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      // Count how many points in the domain
      size_t num_points = 0;
      for (Domain::DomainPointIterator itr(dom); itr; itr++)
      {
        if (has_point(itr.p))
          num_points++;
      }
      rez.serialize(num_points);
      for (Domain::DomainPointIterator itr(dom); itr; itr++)
      {
        if (has_point(itr.p))
        {
          rez.serialize(itr.p);
          TaskArgument arg = get_point(itr.p);
          rez.serialize(arg.get_size());
          rez.serialize(arg.get_ptr(), arg.get_size());
        }
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::Impl::unpack_arguments(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint p;
        derez.deserialize(p);
        size_t arg_size;
        derez.deserialize(arg_size);
        // We know that adding an argument will make a deep copy
        // so we can make the copy directly out of the buffer
        TaskArgument arg(derez.get_current_pointer(), arg_size);
        set_point(p, arg, true/*replace*/);
        // Now advance the buffer since we ready the argument
        derez.advance_pointer(arg_size);
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl* ArgumentMap::Impl::freeze(void)
    //--------------------------------------------------------------------------
    {
      if (next == NULL)
      {
        frozen = true;
        return this;
      }
      else
        return next->freeze();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::Impl* ArgumentMap::Impl::clone(void)
    //--------------------------------------------------------------------------
    {
      // Make sure everyone in the chain shares the same store
      Impl *new_impl = legion_new<Impl>(store, arguments); 
      // Add a reference so it doesn't get collected
      new_impl->add_reference();
      return new_impl;
    }

    /////////////////////////////////////////////////////////////
    // Argument Map Store 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMapStore::ArgumentMapStore(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMapStore::ArgumentMapStore(const ArgumentMapStore &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ArgumentMapStore::~ArgumentMapStore(void)
    //--------------------------------------------------------------------------
    {
      // Free up all the values that we had stored
      for (std::set<TaskArgument>::const_iterator it = values.begin();
            it != values.end(); it++)
      {
        legion_free(STORE_ARGUMENT_ALLOC, it->get_ptr(), it->get_size());
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMapStore& ArgumentMapStore::operator=(const ArgumentMapStore &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMapStore::add_arg(const TaskArgument &arg)
    //--------------------------------------------------------------------------
    {
      void *buffer = legion_malloc(STORE_ARGUMENT_ALLOC, arg.get_size());
      memcpy(buffer, arg.get_ptr(), arg.get_size());
      TaskArgument new_arg(buffer,arg.get_size());
      values.insert(new_arg);
      return new_arg;
    }

    /////////////////////////////////////////////////////////////
    // Future Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Future::Impl::Impl(Internal *rt, bool register_future, DistributedID did, 
                       AddressSpaceID own_space, AddressSpaceID loc_space,
                       Operation *o /*= NULL*/)
      : DistributedCollectable(rt, did, own_space, loc_space),
        producer_op(o), op_gen((o == NULL) ? 0 : o->get_generation()),
#ifdef LEGION_SPY
        producer_uid((o == NULL) ? 0 : o->get_unique_op_id()),
#endif
        ready_event(UserEvent::create_user_event()), result(NULL),
        result_size(0), empty(true), sampled(false)
    //--------------------------------------------------------------------------
    {
      if (register_future)
        runtime->register_future(did, this);
      if (producer_op != NULL)
        producer_op->add_mapping_reference(op_gen);
      // If we're not the owner, add a resource reference on ourself that
      // will be removed when the owner cleans up
      if (!is_owner())
        add_base_resource_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Future %ld", did);
#endif
    }

    //--------------------------------------------------------------------------
    Future::Impl::Impl(const Future::Impl &rhs)
      : DistributedCollectable(NULL, 0, 0, 0), producer_op(NULL), op_gen(0)
#ifdef LEGION_SPY
        , producer_uid(0)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Future::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      // If we are the owner, remove our resource references on all
      // the instances on remote nodes
      if (is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
      runtime->unregister_future(did);
      // don't want to leak events
      if (!ready_event.has_triggered())
        ready_event.trigger();
      if (result != NULL)
      {
        free(result);
        result = NULL;
        result_size = 0;
      }
      if (producer_op != NULL)
        producer_op->remove_mapping_reference(op_gen);
    }

    //--------------------------------------------------------------------------
    Future::Impl& Future::Impl::operator=(const Future::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }
    
    //--------------------------------------------------------------------------
    void Future::Impl::get_void_result(void)
    //--------------------------------------------------------------------------
    {
      if (!ready_event.has_triggered())
      {
        Processor exec_proc = Processor::get_executing_processor();
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
      }
      if (empty)
      {
        if (producer_op != NULL)
          log_run.error("Accessing empty future! (UID %lld)",
                              producer_op->get_unique_op_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ACCESSING_EMPTY_FUTURE);
      }
      mark_sampled();
    }

    //--------------------------------------------------------------------------
    void* Future::Impl::get_untyped_result(void)
    //--------------------------------------------------------------------------
    {
      if (!ready_event.has_triggered())
      {
        Processor exec_proc = Processor::get_executing_processor();
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
      }
      if (empty)
      {
        if (producer_op != NULL)
          log_run.error("Accessing empty future! (UID %lld)",
                              producer_op->get_unique_op_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ACCESSING_EMPTY_FUTURE);
      }
      mark_sampled();
      return result;
    }

    //--------------------------------------------------------------------------
    size_t Future::Impl::get_untyped_size(void)
    //--------------------------------------------------------------------------
    {
      // Call this first to make sure the future is ready
      get_void_result();
      return result_size;
    }

    //--------------------------------------------------------------------------
    bool Future::Impl::is_empty(bool block)
    //--------------------------------------------------------------------------
    {
      if (block && !ready_event.has_triggered())
      {
        Processor exec_proc = Processor::get_executing_processor();
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
      }
      if (block)
        mark_sampled();
      return empty;
    }

    //--------------------------------------------------------------------------
    void Future::Impl::set_result(const void *args, size_t arglen, bool own)
    //--------------------------------------------------------------------------
    {
      // Should only happen on the owner
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
#endif
      // Clean out any previous results we've save
      if (result != NULL)
        free(result);
      if (own)
      {
        result = const_cast<void*>(args);
        result_size = arglen;
      }
      else
      {
        result_size = arglen;
        result = malloc(result_size);
        memcpy(result,args,result_size);
      }
      empty = false; 
    }

    //--------------------------------------------------------------------------
    void Future::Impl::unpack_future(Deserializer &derez)
    //-------------------------------------------------------------------------
    {
      // Should only happen on the owner
      // Clean out any previous results we've save
      DerezCheck z(derez);
      derez.deserialize(result_size);
      // Handle the case where we get a double send of the
      // result once from another remote node and once
      // from the original owner
      if (result == NULL)
        result = malloc(result_size);
      if (!ready_event.has_triggered())
      {
        derez.deserialize(result,result_size);
        empty = false;
      }
      else
        derez.advance_pointer(result_size);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::complete_future(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!ready_event.has_triggered());
#endif
      ready_event.trigger();
      // If we're the owner send our result to any remote spaces
      if (is_owner())
        broadcast_result();
    }

    //--------------------------------------------------------------------------
    bool Future::Impl::reset_future(void)
    //--------------------------------------------------------------------------
    {
      if (ready_event.has_triggered())
        ready_event = UserEvent::create_user_event();
      bool was_sampled = sampled;
      sampled = false;
      return was_sampled;
    }

    //--------------------------------------------------------------------------
    bool Future::Impl::get_boolean_value(bool &valid)
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        valid = ready_event.has_triggered();
        return *((const bool*)result); 
      }
      valid = false;
      return false; 
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_active(void)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, send a gc reference back to the owner
      if (!is_owner())
        send_remote_gc_update(owner_space, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, remove our gc reference
      if (!is_owner())
        send_remote_gc_update(owner_space, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::register_dependence(Operation *consumer_op)
    //--------------------------------------------------------------------------
    {
      if (producer_op != NULL)
        consumer_op->register_dependence(producer_op, op_gen);
#ifdef DEBUG_HIGH_LEVEL
      else
        assert(!empty); // better not be empty if it doesn't have an op
#endif
    }

    //--------------------------------------------------------------------------
    void Future::Impl::mark_sampled(void)
    //--------------------------------------------------------------------------
    {
      sampled = true;
    }

    //--------------------------------------------------------------------------
    void Future::Impl::broadcast_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
#endif
      Serializer rez;
      {
        rez.serialize(did);
        RezCheck z(rez);
        rez.serialize(result_size);
        rez.serialize(result,result_size);
      }
      // Need to hold the lock when reading the set of remote spaces
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      for (std::set<AddressSpaceID>::const_iterator it = 
            registered_waiters.begin(); it != registered_waiters.end(); it++)
      {
        runtime->send_future_result(*it, rez); 
      }
    }

    //--------------------------------------------------------------------------
    void Future::Impl::send_future(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(sid != local_space);
#endif
      // Skip any requests to send ourselves to the owner
      if (sid == owner_space)
        return;
      // Two phase approach, check first to see if we need to do the send
      bool need_send;
      {
        AutoLock gc(gc_lock,1,false/*exclusive*/);
        if (remote_instances.contains(sid))
          need_send = false;
        else
          need_send = true;
      }
      // Need to send this first to avoid race
      bool perform_registration = false;
      if (need_send)
      {
        // Retake the lock and make sure we didn't lose the race
        AutoLock gc(gc_lock);
        if (!remote_instances.contains(sid))
        {
          Serializer rez;
          bool send_result = ready_event.has_triggered();
          {
            rez.serialize(did);
            rez.serialize(owner_space);
            rez.serialize(send_result);
            if (send_result)
            {
              RezCheck z(rez);
              rez.serialize(result_size);
              rez.serialize(result,result_size);
            }
          }
          // Actually do the send and then mark that we
          // have already sent an instance there
          runtime->send_future(sid, rez);
          remote_instances.add(sid);
          if (!send_result)
            perform_registration = true;
        }
      }
      if (perform_registration)
        register_waiter(sid);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::register_waiter(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        bool send_result;
        {
          AutoLock gc(gc_lock);
          if (registered_waiters.find(sid) == registered_waiters.end())
          {
            send_result = ready_event.has_triggered();
            if (!send_result)
              registered_waiters.insert(sid);
          }
          else
            send_result = false;
        }
        if (send_result)
        {
          Serializer rez;
          {
            rez.serialize(did);
            RezCheck z(rez);
            rez.serialize(result_size);
            rez.serialize(result,result_size);
          }
          runtime->send_future_result(sid, rez);
        }
      }
      else
      {
        // not the owner so send a message to the owner
        Serializer rez;
        rez.serialize(did);
        rez.serialize(sid);
        runtime->send_future_subscription(owner_space, rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Future::Impl::handle_future_send(Deserializer &derez,
                                       Internal *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID own_space;
      derez.deserialize(own_space);
      bool is_complete;
      derez.deserialize(is_complete);
      // Check to see if the runtime already has this future
      // if not then we need to make one
      Future::Impl *future = runtime->find_or_create_future(did, own_space);
      future->update_remote_instances(source); 
      if (is_complete)
      {
        future->unpack_future(derez);
        future->complete_future();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Future::Impl::handle_future_result(Deserializer &derez,
                                                       Internal *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      Future::Impl *future = runtime->find_future(did);
      future->unpack_future(derez);
      future->complete_future();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Future::Impl::handle_future_subscription(
                                         Deserializer &derez, Internal *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID subscriber;
      derez.deserialize(subscriber);
      Future::Impl *future = runtime->find_future(did);
      future->register_waiter(subscriber); 
    }

    //--------------------------------------------------------------------------
    void Future::Impl::contribute_to_collective(Barrier bar, unsigned count)
    //--------------------------------------------------------------------------
    {
      if (!ready_event.has_triggered())
      {
        // If we're not done then defer the operation until we are triggerd
        // First add a garbage collection reference so we don't get
        // collected while we are waiting for the contribution task to run
        add_base_gc_ref(PENDING_COLLECTIVE_REF);
        ContributeCollectiveArgs args;
        args.hlr_id = HLR_CONTRIBUTE_COLLECTIVE_ID;
        args.impl = this;
        args.barrier = bar;
        args.count = count;
        // Spawn the task dependent on the future being ready
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_CONTRIBUTE_COLLECTIVE_ID,
                                         NULL, ready_event);
      }
      else // If we've already triggered, then we can do the arrival now
        bar.arrive(count, Event::NO_EVENT, result, result_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Future::Impl::handle_contribute_to_collective(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const ContributeCollectiveArgs *cargs = (ContributeCollectiveArgs*)args;
      cargs->impl->contribute_to_collective(cargs->barrier, cargs->count);
      // Now remote the garbage collection reference and see if we can 
      // reclaim the future
      if (cargs->impl->remove_base_gc_ref(PENDING_COLLECTIVE_REF))
        delete cargs->impl;
    }
      
    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, TaskOp *t, Internal *rt)
      : Collectable(), context(ctx), task(t), task_gen(t->get_generation()),
        valid(true), runtime(rt), ready_event(t->get_completion_event()),
        lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, Event comp_event, Internal *rt)
      : Collectable(), context(ctx), task(NULL), task_gen(0),
        valid(true), runtime(rt), ready_event(comp_event),
        lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, Internal *rt)
      : Collectable(), context(ctx), task(NULL), task_gen(0),
        valid(false), runtime(rt), ready_event(Event::NO_EVENT), 
        lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(const FutureMap::Impl &rhs)
      : Collectable(), context(NULL), task(NULL), task_gen(0), 
        valid(false), runtime(NULL) 
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      futures.clear();
      if (lock.exists())
      {
        lock.destroy_reservation();
        lock = Reservation::NO_RESERVATION;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl& FutureMap::Impl::operator=(const FutureMap::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Future FutureMap::Impl::get_future(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
#ifndef NDEBUG
      // Check to make sure we are asking for something in the domain
      if (valid_points.find(point) == valid_points.end())
      {
        bool is_valid_point = false;
        for (std::vector<Domain>::const_iterator it = 
              valid_domains.begin(); it != valid_domains.end(); it++)
        {
          if (it->contains(point))
          {
            is_valid_point = true;
            break;
          }
        }
        assert(is_valid_point);
      }
#endif
#endif
      if (valid)
      {
        Event lock_event = lock.acquire(0, true/*exclusive*/);
        lock_event.wait();
        // Check to see if we already have a future for the point
        std::map<DomainPoint,Future>::const_iterator finder = 
                                              futures.find(point);
        if (finder != futures.end())
        {
          Future result = finder->second;
          lock.release();
          return result;
        }
        // Otherwise we need a future from the context to use for
        // the point that we will fill in later
        Future result = runtime->help_create_future(task);
        futures[point] = result;
        lock.release();
        return result;
      }
      else
        return runtime->help_create_future();
    }

    //--------------------------------------------------------------------------
    void FutureMap::Impl::get_void_result(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      Future f = get_future(point);
      f.get_void_result();
    }

    //--------------------------------------------------------------------------
    void FutureMap::Impl::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      if (valid)
      {
        // Wait on the event that indicates the entire task has finished
        if (!ready_event.has_triggered())
        {
          Processor proc = context->get_executing_processor();
          runtime->pre_wait(proc);
          ready_event.wait();
          runtime->post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FutureMap::Impl::complete_all_futures(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      AutoLock l_lock(lock);
      for (std::map<DomainPoint,Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        runtime->help_complete_future(it->second);
      }
    }

    //--------------------------------------------------------------------------
    bool FutureMap::Impl::reset_all_futures(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(valid);
#endif
      bool result = false;
      AutoLock l_lock(lock);
      for (std::map<DomainPoint,Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        bool restart = runtime->help_reset_future(it->second);
        if (restart)
          result = true;
      }
      return result;
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void FutureMap::Impl::add_valid_domain(const Domain &d)
    //--------------------------------------------------------------------------
    {
      valid_domains.push_back(d);
    }

    //--------------------------------------------------------------------------
    void FutureMap::Impl::add_valid_point(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      valid_points.insert(dp);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Physical Region Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::Impl::Impl(const RegionRequirement &r, Event ready, bool m, 
                               SingleTask *ctx, MapperID mid, MappingTagID t,
                               bool leaf, Internal *rt)
      : Collectable(), runtime(rt), context(ctx), map_id(mid), tag(t),
        leaf_region(leaf), ready_event(ready), req(r), mapped(m), 
        valid(false), trigger_on_unmap(false)
    //--------------------------------------------------------------------------
    {
#ifdef BOUNDS_CHECKS
      bounds = runtime->get_index_space_domain(req.region.get_index_space());
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::Impl::Impl(const PhysicalRegion::Impl &rhs)
      : Collectable(), runtime(NULL), context(NULL), map_id(0), tag(0),
        leaf_region(false), ready_event(Event::NO_EVENT), mapped(false),
        valid(false), trigger_on_unmap(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      // If we still have a trigger on unmap, do that before
      // deleting ourselves to avoid leaking events
      if (trigger_on_unmap)
      {
        trigger_on_unmap = false;
        termination_event.trigger();
      }
      if (reference.has_ref())
        reference.remove_valid_reference(PHYSICAL_REGION_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::Impl& PhysicalRegion::Impl::operator=(
                                                const PhysicalRegion::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapped); // should only be waiting on mapped regions
#endif
      // If we've already gone through this process we're good
      if (valid)
        return;
      if (!ready_event.has_triggered())
      {
        // Need to tell the runtime that we're about to
        // wait on this value which will pre-empt the
        // executing tasks
        Processor proc = context->get_executing_processor();
        runtime->pre_wait(proc);
        ready_event.wait();
        runtime->post_wait(proc);
      }
      // Now wait for the reference to be ready
      Event ref_ready = reference.get_ready_event();
      if (!ref_ready.has_triggered())
      {
        Processor proc = context->get_executing_processor();
        runtime->pre_wait(proc);
        // If we need a lock for this instance taken it
        // once the reference event is ready, we can also issue
        // the unlock operations contingent upon the termination 
        // event having triggered
        if (reference.has_required_locks())
        {
          std::map<Reservation,bool> required_locks;
          reference.update_atomic_locks(required_locks, true/*exclusive*/);
          Event locked_event = ref_ready;
          for (std::map<Reservation,bool>::const_iterator it = 
                required_locks.begin(); it != required_locks.end(); it++)
          {
            locked_event = it->first.acquire(0, it->second, locked_event);
            it->first.release(termination_event);
          }
          locked_event.wait();
        }
        else
          ref_ready.wait();
        runtime->post_wait(proc);
      }
      valid = true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::Impl::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (valid)
        return true;
      return (mapped && ready_event.has_triggered() && 
              reference.get_ready_event().has_triggered());
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::Impl::is_mapped(void) const
    //--------------------------------------------------------------------------
    {
      return mapped;
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::Impl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      return req.region;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::Impl::get_accessor(void)
    //--------------------------------------------------------------------------
    {
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        runtime->remap_region(context, PhysicalRegion(this));
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_HIGH_LEVEL
        assert(mapped);
#endif
      }
      // Wait until we are valid before returning the accessor
      wait_until_valid();
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        result = reference.get_accessor();
      result.set_region_untyped(this);
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(req.get_accessor_privilege()); 
#endif
      return result;
#else // privilege or bounds checks
      return reference.get_accessor();
#endif
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::Impl::get_field_accessor(FieldID fid)
    //--------------------------------------------------------------------------
    {
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        runtime->remap_region(context, PhysicalRegion(this));
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_HIGH_LEVEL
        assert(mapped);
#endif 
      }
      // Wait until we are valid before returning the accessor
      wait_until_valid();
#ifdef DEBUG_HIGH_LEVEL
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
      {
        log_inst.error("Requested field accessor for field %d "
            "without privileges!", fid);
        assert(false);
        exit(ERROR_INVALID_FIELD_PRIVILEGES);
      }
#endif
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
      Accessor::RegionAccessor<Accessor::AccessorType::Generic>
        result = reference.get_field_accessor(fid);
      result.set_region_untyped(this);
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(req.get_accessor_privilege());
#endif
      return result;
#else // privilege or bounds checks
      return reference.get_field_accessor(fid);
#endif
    } 

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::unmap_region(void)
    //--------------------------------------------------------------------------
    {
      if (!mapped)
        return;
      if (trigger_on_unmap)
      {
        trigger_on_unmap = false;
        // Can only do the trigger when we have actually ready
        Event ref_ready = reference.get_ready_event();
        termination_event.trigger(Event::merge_events(ready_event,ref_ready));
      }
      valid = false;
      mapped = false;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::remap_region(Event new_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!mapped);
#endif
      ready_event = new_ready;
      mapped = true;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& PhysicalRegion::Impl::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return req;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::set_reference(const InstanceRef &ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!reference.has_ref());
#endif
      reference = ref;
      if (reference.has_ref())
        reference.add_valid_reference(PHYSICAL_REGION_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::reset_reference(const InstanceRef &ref,
                                               UserEvent term_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapped);
#endif
      if (reference.has_ref())
        reference.remove_valid_reference(PHYSICAL_REGION_REF);
      reference = ref;
      if (reference.has_ref())
        reference.add_valid_reference(PHYSICAL_REGION_REF);
      termination_event = term_event;
      trigger_on_unmap = true;
    }

    //--------------------------------------------------------------------------
    Event PhysicalRegion::Impl::get_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      return ready_event;
    }

    //--------------------------------------------------------------------------
    const InstanceRef& PhysicalRegion::Impl::get_reference(void) const
    //--------------------------------------------------------------------------
    {
      return reference;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::get_memories(std::set<Memory>& memories) const
    //--------------------------------------------------------------------------
    {
      memories.insert(reference.get_memory());
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::Impl::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      const PhysicalManager* manager = reference.get_manager();
      if (manager->is_reduction_manager())
        // if the instance is a reduction instance,
        // just extract the list of fields from the region requirement
        fields.insert(fields.end(), req.instance_fields.begin(),
            req.instance_fields.end());
      else
        // otherwise, read the list from the layout description
        manager->as_instance_manager()->layout->get_fields(fields);
    }


#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    //--------------------------------------------------------------------------
    const char* PhysicalRegion::Impl::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      return context->variants->name;
    }
#endif

#ifdef BOUNDS_CHECKS 
    //--------------------------------------------------------------------------
    bool PhysicalRegion::Impl::contains_ptr(ptr_t ptr) const 
    //--------------------------------------------------------------------------
    {
      DomainPoint dp(ptr.value);
      return bounds.contains(dp);
    }
    
    //--------------------------------------------------------------------------
    bool PhysicalRegion::Impl::contains_point(const DomainPoint &dp) const
    //--------------------------------------------------------------------------
    {
      return bounds.contains(dp);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Grant Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Grant::Impl::Impl(void)
      : acquired(false), grant_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Grant::Impl::Impl(const std::vector<ReservationRequest> &reqs)
      : requests(reqs), acquired(false), 
        grant_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Grant::Impl::Impl(const Grant::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Grant::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      // clean up our reservation
      grant_lock.destroy_reservation();
      grant_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    Grant::Impl& Grant::Impl::operator=(const Grant::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void Grant::Impl::register_operation(Event completion_event)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      completion_events.insert(completion_event);
    }

    //--------------------------------------------------------------------------
    Event Grant::Impl::acquire_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      if (!acquired)
      {
        grant_event = Event::NO_EVENT;
        for (std::vector<ReservationRequest>::const_iterator it = 
              requests.begin(); it != requests.end(); it++)
        {
          grant_event = it->reservation.acquire(it->mode, 
                                                it->exclusive, grant_event);
        }
        acquired = true;
      }
      return grant_event;
    }

    //--------------------------------------------------------------------------
    void Grant::Impl::release_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      Event deferred_release = Event::merge_events(completion_events);
      for (std::vector<ReservationRequest>::const_iterator it = 
            requests.begin(); it != requests.end(); it++)
      {
        it->reservation.release(deferred_release);
      }
    }

    //--------------------------------------------------------------------------
    void Grant::Impl::pack_grant(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      Event pack_event = acquire_grant();
      rez.serialize(pack_event);
    }

    //--------------------------------------------------------------------------
    void Grant::Impl::unpack_grant(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Event unpack_event;
      derez.deserialize(unpack_event);
      AutoLock g_lock(grant_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(!acquired);
#endif
      grant_event = unpack_event;
      acquired = true;
    }

    /////////////////////////////////////////////////////////////
    // MPI Legion Handshake Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPILegionHandshake::Impl::Impl(bool in_mpi, int mpi_parts, int legion_parts)
      : mpi_participants(mpi_parts), legion_participants(legion_parts),
        state(in_mpi ? IN_MPI : IN_LEGION), mpi_count(0), legion_count(0),
        mpi_ready(UserEvent::create_user_event()),
        legion_ready(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::Impl::Impl(const MPILegionHandshake::Impl &rhs)
      : mpi_participants(-1), legion_participants(-1)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
      // Trigger any leftover events to give them 
      // back to the low-level runtime
      if (!mpi_ready.has_triggered())
        mpi_ready.trigger();
      if (!legion_ready.has_triggered())
        legion_ready.trigger();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshake::Impl& MPILegionHandshake::Impl::operator=(
                                            const MPILegionHandshake::Impl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::Impl::mpi_handoff_to_legion(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(state == IN_MPI);
#endif
      const int count = __sync_add_and_fetch(&mpi_count, 1);
      if (count == mpi_participants)
      {
        // Create a new waiter for the mpi_ready
        UserEvent mpi_trigger = legion_ready;
        legion_ready = UserEvent::create_user_event();
        // Reset the count for the next iteration
        mpi_count = 0;
        // Switch the state to being in Legion
        state = IN_LEGION;
        // Now tell all the Legion threads that we are ready
        mpi_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::Impl::mpi_wait_on_legion(void)
    //--------------------------------------------------------------------------
    {
      // Wait for mpi to be ready to run
      // Note we use the external wait to be sure 
      // we don't get drafted by the low-level runtime
      mpi_ready.external_wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(state == IN_MPI);
#endif
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::Impl::legion_handoff_to_mpi(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(state == IN_LEGION);
#endif
      const int count = __sync_add_and_fetch(&legion_count, 1);
      if (count == legion_participants)
      {
        UserEvent legion_trigger = mpi_ready;
        mpi_ready = UserEvent::create_user_event();
        // Reset the count for the next iteration
        legion_count = 0;
        // Switch the state to being in MPI
        state = IN_MPI;
        // Now tell all the MPI threads that we are ready
        legion_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshake::Impl::legion_wait_on_mpi(void)
    //--------------------------------------------------------------------------
    {
      // Wait for Legion to be ready to run
      // No need to avoid being drafted by the
      // low-level runtime here
      legion_ready.wait();
#ifdef DEBUG_HIGH_LEVEL
      assert(state == IN_LEGION);
#endif
    }

    /////////////////////////////////////////////////////////////
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind,
                                       Internal *rt, unsigned width, 
                                       unsigned def_mappers, bool no_steal, 
                                       unsigned max_steals)
      : runtime(rt), local_proc(proc), proc_kind(kind), 
        superscalar_width(width), 
        stealing_disabled(no_steal), max_outstanding_steals(max_steals),
        next_local_index(0),
        task_scheduler_enabled(false), total_active_contexts(0),
        ready_queues(std::vector<std::list<TaskOp*> >(def_mappers)),
        mapper_objects(std::vector<Mapper*>(def_mappers,NULL)),
        mapper_locks(
            std::vector<Reservation>(def_mappers,Reservation::NO_RESERVATION)),
        mapper_messages(std::vector<std::vector<MapperMessage> >(def_mappers)),
        inside_mapper_call(std::vector<bool>(def_mappers,false)),
        defer_mapper_event(std::vector<Event>(def_mappers,Event::NO_EVENT))
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < def_mappers; idx++)
      {
        ready_queues[idx].clear();
        outstanding_steal_requests[idx] = std::set<Processor>();
      }
      this->local_queue_lock = Reservation::create_reservation();
      this->queue_lock = Reservation::create_reservation();
      this->message_lock = Reservation::create_reservation();
      this->stealing_lock = Reservation::create_reservation();
      this->thieving_lock = Reservation::create_reservation();
      context_states.resize(DEFAULT_CONTEXTS);
      local_scheduler_preconditions.resize(superscalar_width, Event::NO_EVENT);
    }

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(const ProcessorManager &rhs)
      : runtime(NULL), local_proc(Processor::NO_PROC),
        proc_kind(Processor::LOC_PROC),
        superscalar_width(0), stealing_disabled(false), 
        max_outstanding_steals(0), next_local_index(0),
        task_scheduler_enabled(false), total_active_contexts(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ProcessorManager::~ProcessorManager(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < mapper_objects.size(); idx++)
      {
        if (mapper_objects[idx] != NULL)
        {
          delete mapper_objects[idx];
          mapper_objects[idx] = NULL;
#ifdef DEBUG_HIGH_LEVEL
          assert(mapper_locks[idx].exists());
#endif
          mapper_locks[idx].destroy_reservation();
          mapper_locks[idx] = Reservation::NO_RESERVATION;
        }
      }
      mapper_objects.clear();
      mapper_locks.clear();
      ready_queues.clear();
      local_queue_lock.destroy_reservation();
      local_queue_lock = Reservation::NO_RESERVATION;
      queue_lock.destroy_reservation();
      queue_lock = Reservation::NO_RESERVATION;
      message_lock.destroy_reservation();
      message_lock = Reservation::NO_RESERVATION;
      stealing_lock.destroy_reservation();
      stealing_lock = Reservation::NO_RESERVATION;
      thieving_lock.destroy_reservation();
      thieving_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    ProcessorManager& ProcessorManager::operator=(const ProcessorManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_mapper(MapperID mid, Mapper *m, bool check)
    //--------------------------------------------------------------------------
    {
      log_run.spew("Adding mapper %d on processor " IDFMT "", 
                          mid, local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      if (check && (mid == 0))
      {
        log_run.error("Invalid mapping ID.  ID 0 is reserved.");
        assert(false);
        exit(ERROR_RESERVED_MAPPING_ID);
      } 
#endif
      if (mid >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(mid+1);
        mapper_locks.resize(mid+1);
        mapper_messages.resize(mid+1);
        inside_mapper_call.resize(mid+1);
        defer_mapper_event.resize(mid+1);
        ready_queues.resize(mid+1);
        for (unsigned int i=old_size; i<(mid+1); i++)
        {
          mapper_objects[i] = NULL;
          mapper_locks[i].destroy_reservation();
          mapper_locks[i] = Reservation::NO_RESERVATION;
          inside_mapper_call[i] = false;
          defer_mapper_event[i] = Event::NO_EVENT;
          ready_queues[i].clear();
          outstanding_steal_requests[i] = std::set<Processor>();
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(mid < mapper_objects.size());
      assert(mapper_objects[mid] == NULL);
      assert(!mapper_locks[mid].exists());
#endif
      mapper_locks[mid] = Reservation::create_reservation();
      mapper_objects[mid] = m;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::replace_default_mapper(Mapper *m)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects[0] != NULL);
#endif
      delete mapper_objects[0];
      mapper_objects[0] = m;
    }

    //--------------------------------------------------------------------------
    Mapper* ProcessorManager::find_mapper(MapperID mid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mid < mapper_objects.size());
#endif
      return mapper_objects[mid];
    }

    //--------------------------------------------------------------------------
    template<typename MAPPER_CONTINUATION>
    void ProcessorManager::invoke_mapper(MapperID map_id, 
                   MAPPER_CONTINUATION &continuation, bool block, bool has_lock)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      Reservation map_lock = mapper_locks[map_id];
      // If we don't have the lock try to get it
      if (!has_lock)
      {
        Event acquire_event = map_lock.acquire();
        if (!acquire_event.has_triggered())
        {
          // Failed to get the lock, defer the mapping
          continuation.defer(runtime, acquire_event);
          return;
        }
      }
      // If we make it here we have the lock, go ahead and invoke the mapper
      inside_mapper_call[map_id] = true;
      continuation.invoke_mapper(mapper);
      inside_mapper_call[map_id] = false;
      // See if we have a deferal event
      Event defer_event = defer_mapper_event[map_id];
      // Clear it if necessary
      if (defer_event.exists())
        defer_mapper_event[map_id] = Event::NO_EVENT;
      // Now we can release our lock
      map_lock.release();
      // If we have to defer mapping, do that now, otherwise mark complete
      if (defer_event.exists())
      {
        if (block)
        {
          // Only reacquire lock after the event is ready
          Event acquire_event = 
            map_lock.acquire(0, true/*exclusive*/, defer_event);
          continuation.defer(runtime, acquire_event);
        }
        else
        {
          log_run.warning("Ignoring mapper deferral request in "
                          "non-blocking mapper call!");
          continuation.complete_mapping();
        }
      }
      else
        continuation.complete_mapping();
      // Now send any messages
      check_mapper_messages(map_id, true/*need lock*/);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::check_mapper_messages(MapperID map_id,bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        Event acquire_event = message_lock.acquire(); 
        // If we didn't get it we have to defer this operation
        if (!acquire_event.has_triggered())
        {
          CheckMessagesContinuation continuation(this, map_id);
          Event done_event = continuation.defer(runtime, acquire_event);
          done_event.wait();
          return;
        }
      }
      // Once we get here we have the message lock
      std::vector<MapperMessage> messages;
      std::vector<MapperMessage> &current = mapper_messages[map_id]; 
      if (!current.empty())
      {
        messages = current;
        current.clear();
      }
      // Release the lock
      message_lock.release();
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_set_task_options(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<Task*,&Mapper::select_task_options,true/*block*/>
                                         continuation(this, task->map_id, task);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_pre_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1Ret<bool,Task*,&Mapper::pre_map_task,true/*block*/>
                          continuation(this, task->map_id, false/*init*/, task);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_select_variant(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<Task*,&Mapper::select_task_variant,true/*block*/>
                                         continuation(this, task->map_id, task);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      // This one is special since it both maps a task and picks its variant
      // This will be fixed with the new mapper interface so keep it for now
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      // actually set on all possible paths below, but compiler can't tell
      bool result = false;  
      std::vector<MapperMessage> messages;
      Event wait_on = Event::NO_EVENT;
      do
      {
        if (wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(task->map_id, messages);
          wait_on.wait();
        }
        AutoLock m_lock(mapper_locks[task->map_id]);
        inside_mapper_call[task->map_id] = true;
        // First select the variant
        mapper_objects[task->map_id]->select_task_variant(task);
        inside_mapper_call[task->map_id] = false;
        if (defer_mapper_event[task->map_id].exists())
        {
          wait_on = defer_mapper_event[task->map_id];
          defer_mapper_event[task->map_id] = Event::NO_EVENT;
          AutoLock g_lock(message_lock);
          if (!mapper_messages[task->map_id].empty())
          {
            messages = mapper_messages[task->map_id];
            mapper_messages[task->map_id].clear();
          }
          continue;
        }
        // Then perform the mapping
        inside_mapper_call[task->map_id] = true;
        result = mapper_objects[task->map_id]->map_task(task);
        inside_mapper_call[task->map_id] = false;
        if (defer_mapper_event[task->map_id].exists())
        {
          wait_on = defer_mapper_event[task->map_id];
          defer_mapper_event[task->map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[task->map_id].empty())
        {
          messages = mapper_messages[task->map_id];
          mapper_messages[task->map_id].clear();
        }
      } while (wait_on.exists());
      if (!messages.empty())
        send_mapper_messages(task->map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_post_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<Task*,&Mapper::post_map_task,true/*block*/>
                                      continuation(this, task->map_id, task);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_failed_mapping(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<const Mappable*,&Mapper::notify_mapping_failed,
                  true/*block*/> continuation(this, mappable->map_id, mappable);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_notify_result(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<const Mappable*,&Mapper::notify_mapping_result,
                  true/*block*/> continuation(this, mappable->map_id, mappable);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_slice_domain(TaskOp *task,
                                      std::vector<Mapper::DomainSplit> &splits)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3NoRet<const Task*,const Domain&,
                               std::vector<Mapper::DomainSplit>&,
                               &Mapper::slice_domain,true/*block*/>
             continuation(this, task->map_id, task, task->index_domain, splits);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_inline(Inline *op)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1Ret<bool,Inline*,&Mapper::map_inline,true/*block*/>
                              continuation(this, op->map_id, false/*init*/, op);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_copy(Copy *op)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1Ret<bool,Copy*,&Mapper::map_copy,true/*block*/>
                              continuation(this, op->map_id, false/*init*/, op);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_speculate(Mappable *op, bool &value)
    //--------------------------------------------------------------------------
    {
      MapperContinuation2Ret<bool,const Mappable*,bool&,
                             &Mapper::speculate_on_predicate, true/*block*/>
                     continuation(this, op->map_id, false/*init*/, op, value);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_configure_context(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<Task*,&Mapper::configure_context, true/*block*/>
                                  continuation(this, task->map_id, task);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_rank_copy_targets(Mappable *mappable,
                                           LogicalRegion handle, 
                                           const std::set<Memory> &memories,
                                           bool complete,
                                           size_t max_blocking_factor,
                                           std::set<Memory> &to_reuse,
                                           std::vector<Memory> &to_create,
                                           bool &create_one,
                                           size_t &blocking_factor)
    //--------------------------------------------------------------------------
    {
      // Don't both updating this one, it is going away soon
#ifdef DEBUG_HIGH_LEVEL
      assert(mappable->map_id < mapper_objects.size());
      assert(mapper_objects[mappable->map_id] != NULL);
#endif
      bool result;
      std::vector<MapperMessage> messages;
      Event wait_on = Event::NO_EVENT;
      do
      {
        if (wait_on.exists())
        {
          if (!messages.empty())
            send_mapper_messages(mappable->map_id, messages);
          wait_on.wait();
        }
        AutoLock m_lock(mapper_locks[mappable->map_id]);
        inside_mapper_call[mappable->map_id] = true;
        result = mapper_objects[mappable->map_id]->rank_copy_targets(mappable,
            handle, memories, complete, max_blocking_factor, to_reuse, 
            to_create, create_one, blocking_factor);
        inside_mapper_call[mappable->map_id] = false;
        if (defer_mapper_event[mappable->map_id].exists())
        {
          wait_on = defer_mapper_event[mappable->map_id];
          defer_mapper_event[mappable->map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[mappable->map_id].empty())
        {
          messages = mapper_messages[mappable->map_id];
          mapper_messages[mappable->map_id].clear();
        }
      } while (wait_on.exists());
      if (!messages.empty())
        send_mapper_messages(mappable->map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_rank_copy_sources(Mappable *mappable,
                                           const std::set<Memory> &memories,
                                           Memory destination,
                                           std::vector<Memory> &order)
    //--------------------------------------------------------------------------
    {
      MapperContinuation4NoRet<const Mappable*, const std::set<Memory>&,
                               Memory,std::vector<Memory>&,
                               &Mapper::rank_copy_sources,true/*block*/>
                                continuation(this, mappable->map_id, mappable, 
                                             memories, destination, order);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_notify_profiling(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<const Task*,&Mapper::notify_profiling_info,
                         true/*block*/> continuation(this, task->map_id, task);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_must_epoch(
        const std::vector<Task*> &tasks,
        const std::vector<Mapper::MappingConstraint> &constraints,
        MapperID map_id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3Ret<bool,const std::vector<Task*>&,
                             const std::vector<Mapper::MappingConstraint>&,
                             MappingTagID,&Mapper::map_must_epoch,true/*block*/>
                               continuation(this, map_id, false/*init*/, 
                                            tasks, constraints, tag);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    int ProcessorManager::invoke_mapper_get_tunable_value(TaskOp *task,
                                                          TunableID tid,
                                                          MapperID map_id,
                                                          MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3Ret<int,const Task*,TunableID,MappingTagID,
                             &Mapper::get_tunable_value,true/*block*/>
                         continuation(this, map_id, 0/*init*/, task, tid, tag);
      return continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_handle_message(MapperID map_id,
                                                        Processor source,
                                                        const void *message,
                                                        size_t length)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3NoRet<Processor,const void*,size_t,
                               &Mapper::handle_message,true/*block*/>
                         continuation(this, map_id, source, message, length);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_task_result(MapperID map_id,
                                                     Event event,
                                                     const void *result,
                                                     size_t result_size)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3NoRet<MapperEvent,const void*,size_t,
                               &Mapper::handle_mapper_task_result,true/*block*/>
                         continuation(this, map_id, event, result, result_size);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_permit_task_steal(MapperID map_id,
                Processor thief, const std::vector<const Task*> &stealable,
                std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------
    {
      MapperContinuation3NoRet<Processor,const std::vector<const Task*>&,
            std::set<const Task*>&,&Mapper::permit_task_steal,true/*block*/>
                      continuation(this, map_id, thief, stealable, to_steal);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_target_task_steal(MapperID map_id, 
                                         const std::set<Processor> &blacklist,
                                         std::set<Processor> &steal_targets)
    //--------------------------------------------------------------------------
    {
      MapperContinuation2NoRet<const std::set<Processor>&,std::set<Processor>&,
                               &Mapper::target_task_steal,false/*block*/>
                           continuation(this, map_id, blacklist, steal_targets);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_select_tasks_to_schedule(
                          MapperID map_id, const std::list<Task*> &ready_tasks)
    //--------------------------------------------------------------------------
    {
      MapperContinuation1NoRet<const std::list<Task*>&,
                              &Mapper::select_tasks_to_schedule, false/*block*/>
                               continuation(this, map_id, ready_tasks);
      continuation.perform_mapping();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::defer_mapper_message(Processor target, 
                            MapperID map_id, const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_messages.size());
#endif
      // Take the message lock
      AutoLock g_lock(message_lock);
      // Check to see if we are inside of a mapper call
      if (inside_mapper_call[map_id])
      {
        // Need to make a copy here of the message
        void *copy = malloc(length);
        memcpy(copy, message, length);
        // save the message
        mapper_messages[map_id].push_back(MapperMessage(target, copy, length));
      }
      else
      {
        // Otherwise the application has explicitly invoked one
        // of its mapper calls, so we can safely send the message now
        // without needing to worry about deadlock 
        runtime->invoke_mapper_handle_message(target, map_id, local_proc,
                                              message, length);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::defer_mapper_broadcast(MapperID map_id, 
                                                  const void *message,
                                                  size_t length, int radix)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_messages.size());
#endif
      // Handle bad cases here
      if (radix < 2)
        radix = 2;
      // Take the message lock
      AutoLock g_lock(message_lock);
      // Check to see if we are inside of a mapper call
      if (inside_mapper_call[map_id])
      {
        // Need to make a copy of the message here
        void *copy = malloc(length);
        memcpy(copy, message, length);
        mapper_messages[map_id].push_back(MapperMessage(copy, length, radix));
      }
      else
      {
        // Otherwise the application has explicitly invoked one of its
        // mapper calls, so we can safely send the message now without
        // needing to worry about deadlock.
        runtime->invoke_mapper_broadcast(map_id, local_proc, 
                                         message, length, radix, 1/*index*/);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::defer_mapper_call(MapperID map_id, Event wait_on)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_messages.size());
      assert(inside_mapper_call[map_id]);
#endif
      defer_mapper_event[map_id] = wait_on;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::send_mapper_messages(MapperID map_id, 
                                           std::vector<MapperMessage> &messages)
    //--------------------------------------------------------------------------
    {
      for (std::vector<MapperMessage>::iterator it = messages.begin();
            it != messages.end(); it++)
      {
        // Check to see if this a specific message or a broadcast
        if (it->target.exists())
          runtime->invoke_mapper_handle_message(it->target, map_id, local_proc,
                                                it->message, it->length);
        else
          runtime->invoke_mapper_broadcast(map_id, local_proc, 
                                           it->message, it->length, 
                                           it->radix, 1/*index*/);
        // After we are done sending the message, we can free the memory
        free(it->message);
      }
      messages.clear();
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------
    {
      perform_mapping_operations(); 
      // Now re-take the lock and re-check the condition to see 
      // if the next scheduling task should be launched
      AutoLock q_lock(queue_lock);
      if (total_active_contexts > 0)
      {
        task_scheduler_enabled = true;
        launch_task_scheduler();
      }
      else
        task_scheduler_enabled = false; 
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::launch_task_scheduler(void)
    //--------------------------------------------------------------------------
    {
      SchedulerArgs sched_args;
      sched_args.hlr_id = HLR_SCHEDULER_ID;
      sched_args.proc = local_proc;
      runtime->issue_runtime_meta_task(&sched_args, sizeof(sched_args),
                                       HLR_SCHEDULER_ID);
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::activate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_context_id();
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_HIGH_LEVEL
      assert(!state.active);
#endif
      state.active = true;
      if (state.owned_tasks > 0)
        increment_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::deactivate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_context_id();
      // We can do this without holding the lock because we know
      // the size of this vector is fixed
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_HIGH_LEVEL
      assert(state.active);
#endif
      state.active = false;
      if (state.owned_tasks > 0)
        decrement_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::update_max_context_count(unsigned max_contexts)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      context_states.resize(max_contexts);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if ((total_active_contexts == 0) && !task_scheduler_enabled)
      {
        task_scheduler_enabled = true;
        launch_task_scheduler();
      }
      total_active_contexts++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
#ifdef DEBUG_HIGH_LEVEL
      assert(total_active_contexts > 0);
#endif
      total_active_contexts--;
      if (total_active_contexts == 0)
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_steal_request(Processor thief,
                                           const std::vector<MapperID> &thieves)
    //--------------------------------------------------------------------------
    {
      log_run.spew("handling a steal request on processor " IDFMT " "
                         "from processor " IDFMT "", local_proc.id,thief.id);
      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal the task
      std::set<TaskOp*> stolen;
      for (std::vector<MapperID>::const_iterator steal_it = thieves.begin();
            steal_it != thieves.end(); steal_it++)
      {
        MapperID stealer = *steal_it;
        // Handle a race condition here where some processors can 
        // issue steal requests to another processor before the mappers 
        // have been initialized on that processor.  There's no 
        // correctness problem for ignoring a steal request so just do that.
        if ((mapper_objects.size() <= stealer) ||
            (mapper_objects[stealer] == NULL))
          continue;
        
        // Construct a vector of tasks eligible for stealing
        std::vector<TaskOp*> mapper_tasks;
        {
          AutoLock q_lock(queue_lock,1,false/*exclusive*/);
          for (std::list<TaskOp*>::const_iterator it = 
                ready_queues[stealer].begin(); it != 
                ready_queues[stealer].end(); it++)
          {
            if ((*it)->is_stealable() && !(*it)->is_locally_mapped())
              mapper_tasks.push_back(*it);
          }
        }
        std::set<TaskOp*> to_steal;
        // Ask the mapper what it wants to allow be stolen
        if (!mapper_tasks.empty())
        {
          // Time to stomp on the C++ type system again 
          std::set<const Task*> &to_steal_prime = 
            *((std::set<const Task*>*)(&to_steal));
          std::vector<const Task*> &stealable = 
            *((std::vector<const Task*>*)(&mapper_tasks));
          invoke_mapper_permit_task_steal(stealer, thief, stealable,
                                          to_steal_prime);
        }
        std::deque<TaskOp*> temp_stolen;
        if (!to_steal.empty())
        {
          // See if we can still get it out of the queue
          AutoLock q_lock(queue_lock);
          for (unsigned idx = 0; idx < to_steal.size(); idx++)
          for (std::set<TaskOp*>::const_iterator steal_it = to_steal.begin();
                steal_it != to_steal.end(); steal_it++)
          {
            TaskOp *target = *steal_it;
            bool found = false;
            for (std::list<TaskOp*>::iterator it = 
                  ready_queues[stealer].begin(); it !=
                  ready_queues[stealer].end(); it++)
            {
              if ((*it) == target)
              {
                ready_queues[stealer].erase(it);
                found = true;
                break;
              }
            }
            if (found)
            {
              temp_stolen.push_back(target);
              // Wait until we are no longer holding the lock
              // to mark that this is no longer an outstanding task
              ContextID ctx_id = target->get_parent()->get_context_id();
              ContextState &state = context_states[ctx_id];
#ifdef DEBUG_HIGH_LEVEL
              assert(state.owned_tasks > 0);
#endif
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
            }
          }
        }
        // Now see if we can actually steal the task, if not
        // then we have to put it back on the queue
        bool successful_steal = false;
        for (unsigned idx = 0; idx < temp_stolen.size(); idx++)
        {
          if (temp_stolen[idx]->prepare_steal())
          {
            // Mark this as stolen and update the target processor
            temp_stolen[idx]->mark_stolen(thief);
            stolen.insert(temp_stolen[idx]);
            successful_steal = true;
            temp_stolen[idx]->deactivate_outstanding_task();
          }
          else
          {
            // Always set this before putting anything on
            // the ready queue
            temp_stolen[idx]->schedule = false;
            ContextID ctx_id = temp_stolen[idx]->get_parent()->get_context_id();
            AutoLock q_lock(queue_lock);
            ContextState &state = context_states[ctx_id];
            ready_queues[stealer].push_front(temp_stolen[idx]);
            if (state.active && (state.owned_tasks == 0))
              increment_active_contexts();
            state.owned_tasks++;
          }
        }

        if (!successful_steal) 
        {
          AutoLock thief_lock(thieving_lock);
          failed_thiefs.insert(std::pair<MapperID,Processor>(stealer,thief));
        }
      }
      if (!stolen.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        for (std::set<TaskOp*>::const_iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          log_task.debug("task %s (ID %lld) stolen from processor " IDFMT
                               " by processor " IDFMT "", (*it)->variants->name,
                               (*it)->get_unique_task_id(), local_proc.id,
                               thief.id);
        }
#endif
        runtime->send_tasks(thief, stolen);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_advertisement(Processor advertiser,
                                                 MapperID mid)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock steal_lock(stealing_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(outstanding_steal_requests.find(mid) !=
                outstanding_steal_requests.end());
#endif
        outstanding_steal_requests[mid].erase(advertiser);
      }
      // Do a one time enabling of the scheduler so we can try
      // asking any of the mappers if they would like to try stealing again
      AutoLock q_lock(queue_lock);
      if (!task_scheduler_enabled)
      {
        task_scheduler_enabled = true;
        launch_task_scheduler();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_ready_queue(TaskOp *task, bool prev_failure)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task != NULL);
      assert(task->map_id <= ready_queues.size());
#endif
      // always set this before putting something on the ready queue
      task->schedule = false; 
      // have to do this when we are not holding the lock
      task->activate_outstanding_task();
      // We can do this without holding the lock because the
      // vector is of a fixed size
      ContextID ctx_id = task->get_parent()->get_context_id();
      AutoLock q_lock(queue_lock);
      ContextState &state = context_states[ctx_id];
      if (prev_failure)
        ready_queues[task->map_id].push_front(task);
      else
        ready_queues[task->map_id].push_back(task);
      if (state.active && (state.owned_tasks == 0))
        increment_active_contexts();
      state.owned_tasks++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_local_ready_queue(Operation *op, 
                                                    bool prev_failure)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op != NULL);
#endif
      TriggerOpArgs args;
      args.hlr_id = HLR_TRIGGER_OP_ID;
      args.manager = this;
      args.op = op;
      Event precondition = op->invoke_state_analysis();
      // Only do executing throttling if we don't have an even to wait on
      if (precondition.has_triggered())
      {
        if (!prev_failure)
        {
          AutoLock l_lock(local_queue_lock); 
          Event next = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                                        HLR_TRIGGER_OP_ID, op,
                             local_scheduler_preconditions[next_local_index]);
          local_scheduler_preconditions[next_local_index++] = next;
          if (next_local_index == superscalar_width)
            next_local_index = 0;
        }
        else
          runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                           HLR_TRIGGER_OP_ID, op);
      }
      else
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_TRIGGER_OP_ID, op, precondition);
    }

    //--------------------------------------------------------------------------
    unsigned ProcessorManager::sample_unmapped_tasks(MapperID map_id)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock, 1, false/*exclusive*/);
      return ready_queues[map_id].size();
    }

#ifdef HANG_TRACE
    //--------------------------------------------------------------------------
    void ProcessorManager::dump_state(FILE *target)
    //--------------------------------------------------------------------------
    {
      fprintf(target,"State of Processor: " IDFMT " (kind=%d)\n", 
              local_proc.id, proc_kind);
      fprintf(target,"  Current Pending Task: %d\n", current_pending);
      fprintf(target,"  Has Executing Task: %d\n", current_executing);
      fprintf(target,"  Idle Task Enabled: %d\n", idle_task_enabled);
      fprintf(target,"  Dependence Queue Depth: %ld\n", 
              dependence_queues.size());
      for (unsigned idx = 0; idx < dependence_queues.size(); idx++)
        fprintf(target,"    Queue at depth %d has %ld elements\n",
                idx, dependence_queues[idx].queue.size());
      fprintf(target,"  Ready Queue Count: %ld\n", ready_queues.size());
      for (unsigned idx = 0; idx < ready_queues.size(); idx++)
        fprintf(target,"    Ready queue %d has %ld elements\n",
                idx, ready_queues[idx].size());
      fprintf(target,"  Local Queue has %ld elements\n", 
              local_ready_queue.size());
      fflush(target);
    }
#endif

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_mapping_operations(void)
    //--------------------------------------------------------------------------
    {
      std::multimap<Processor,MapperID> stealing_targets;
      std::vector<MapperID> mappers_with_work;
      for (unsigned map_id = 0; map_id < ready_queues.size(); map_id++)
      {
        if (mapper_objects[map_id] == NULL)
          continue;
        std::list<TaskOp*> visible_tasks;
        // We also need to capture the generations here
        std::vector<GenerationID> visible_generations;
        // Pull out the current tasks for this mapping operation
        {
          AutoLock q_lock(queue_lock,1,false/*exclusive*/);
          visible_tasks.insert(visible_tasks.begin(),
               ready_queues[map_id].begin(), ready_queues[map_id].end());
          visible_generations.resize(visible_tasks.size());
          unsigned idx = 0;
          for (std::list<TaskOp*>::const_iterator it = visible_tasks.begin();
                it != visible_tasks.end(); it++, idx++)
          {
            visible_generations[idx] = (*it)->get_generation();
          }
        }
        // Ask the mapper which tasks it would like to schedule
        if (!visible_tasks.empty())
        {
          // Watch me stomp all over the C++ type system here
          const std::list<Task*> &ready_tasks = 
                                *((std::list<Task*>*)(&(visible_tasks)));
          invoke_mapper_select_tasks_to_schedule(map_id, ready_tasks); 
        }
        if (!stealing_disabled)
        {
          std::set<Processor> black_copy;
          // Make a local copy of our blacklist
          {
            AutoLock steal_lock(stealing_lock,1,false/*exclusive*/);
            black_copy = outstanding_steal_requests[map_id];
          }
          std::set<Processor> steal_targets;
          invoke_mapper_target_task_steal(map_id, black_copy, steal_targets);
          AutoLock steal_lock(stealing_lock);
          std::set<Processor> &blacklist = outstanding_steal_requests[map_id];
          for (std::set<Processor>::const_iterator it = 
                steal_targets.begin(); it != steal_targets.end(); it++)
          {
            if (it->exists() && ((*it) != local_proc) &&
                (blacklist.find(*it) == blacklist.end()))
            {
              stealing_targets.insert(std::pair<Processor,MapperID>(
                                                        *it,map_id));
              blacklist.insert(*it);
            }
          }
        }
        // Process the results first remove the operations that were
        // selected to be mapped from the queue.  Note its possible
        // that we can't actually find the task because it has been
        // stolen from the queue while we were deciding what to
        // map.  It's also possible the task is no longer in the same
        // place if the queue was prepended to.
        {
          std::list<TaskOp*> &rqueue = ready_queues[map_id];
          AutoLock q_lock(queue_lock);
          unsigned gen_idx = 0;
          for (std::list<TaskOp*>::iterator vis_it = visible_tasks.begin(); 
                vis_it != visible_tasks.end(); gen_idx++)
          {
            if ((*vis_it)->schedule || 
                ((*vis_it)->target_proc != local_proc))
            {
              bool found = false;
              for (std::list<TaskOp*>::iterator it = rqueue.begin();
                    it != rqueue.end(); it++)
              {
                // In order to be the same task, they need to have the
                // same pointer and have the same generation
                if (((*it) == (*vis_it)) &&
                    (visible_generations[gen_idx] == (*it)->get_generation()))
                {
                  rqueue.erase(it);
                  found = true;
                  break;
                }
              }
              if (!found) // stolen
              {
                // Remove it from our list
                vis_it = visible_tasks.erase(vis_it);
              }
              else
              {
                // Wait until we are not holding the queue lock
                // to mark that this task is no longer outstanding
                ContextID ctx_id = (*vis_it)->get_parent()->get_context_id(); 
                ContextState &state = context_states[ctx_id];
#ifdef DEBUG_HIGH_LEVEL
                assert(state.owned_tasks > 0);
#endif
                state.owned_tasks--;
                if (state.active && (state.owned_tasks == 0))
                  decrement_active_contexts();
                vis_it++;
              }
            }
            else
            {
              // Decided not to do anything so remove it
              // from the list of things to do
              vis_it = visible_tasks.erase(vis_it);
            }
          }
          if (!rqueue.empty())
            mappers_with_work.push_back(map_id);
        }
        // Now that we've removed them from the queue, issue the
        // mapping analysis calls
        TriggerTaskArgs args;
        args.hlr_id = HLR_TRIGGER_TASK_ID;
        args.manager = this;
        for (std::list<TaskOp*>::iterator vis_it = visible_tasks.begin();
              vis_it != visible_tasks.end(); vis_it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert((*vis_it)->schedule || 
                 ((*vis_it)->target_proc != local_proc));
#endif
          (*vis_it)->deactivate_outstanding_task();
          Event wait_on = (*vis_it)->invoke_state_analysis();
          // We give a slight priority to triggering the execution
          // of tasks relative to other runtime operations because
          // they actually have a feedback mechanism controlling
          // how far they get ahead.  We give a slight edge in priority
          // to tasks being sent remotely to get them in flight.
          // Give priority to things which are getting sent remotely
          args.op = *vis_it;
          int priority = ((*vis_it)->target_proc != local_proc) ? 2 : 1;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_TRIGGER_TASK_ID, *vis_it,
                                           wait_on, priority);
        }
      }

      // Advertise any work that we have
      if (!stealing_disabled && !mappers_with_work.empty())
      {
        for (std::vector<MapperID>::const_iterator it = 
              mappers_with_work.begin(); it != mappers_with_work.end(); it++)
        {
          issue_advertisements(*it);
        }
      }

      // Finally issue any steal requeusts
      if (!stealing_disabled && !stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::issue_advertisements(MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Create a clone of the processors we want to advertise so that
      // we don't call into the high level runtime holding a lock
      std::set<Processor> failed_waiters;
      // Check to see if we have any failed thieves with the mapper id
      {
        AutoLock theif_lock(thieving_lock);
        if (failed_thiefs.lower_bound(map_id) != 
            failed_thiefs.upper_bound(map_id))
        {
          for (std::multimap<MapperID,Processor>::iterator it = 
                failed_thiefs.lower_bound(map_id); it != 
                failed_thiefs.upper_bound(map_id); it++)
          {
            failed_waiters.insert(it->second);
          } 
          // Erase all the failed theives
          failed_thiefs.erase(failed_thiefs.lower_bound(map_id),
                              failed_thiefs.upper_bound(map_id));
        }
      }
      if (!failed_waiters.empty())
        runtime->send_advertisements(failed_waiters, map_id, local_proc);
    }

    /////////////////////////////////////////////////////////////
    // Memory Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MemoryManager::MemoryManager(Memory m, Internal *rt)
      : memory(m), capacity(m.capacity()),
        remaining_capacity(capacity), runtime(rt), 
        manager_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MemoryManager::MemoryManager(const MemoryManager &rhs)
      : memory(Memory::NO_MEMORY), capacity(0), runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);   
    }

    //--------------------------------------------------------------------------
    MemoryManager::~MemoryManager(void)
    //--------------------------------------------------------------------------
    {
      manager_lock.destroy_reservation();
      manager_lock = Reservation::NO_RESERVATION;
      physical_instances.clear();
      reduction_instances.clear();
    }

    //--------------------------------------------------------------------------
    MemoryManager& MemoryManager::operator=(const MemoryManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::allocate_physical_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      const size_t inst_size = manager->get_instance_size();  
      AutoLock m_lock(manager_lock);
      remaining_capacity -= inst_size;
      if (manager->is_reduction_manager())
      {
        ReductionManager *reduc = manager->as_reduction_manager();
        reduction_instances[reduc] = inst_size;
      }
      else
      {
        InstanceManager *inst = manager->as_instance_manager();
        physical_instances[inst] = inst_size;
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::free_physical_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      if (manager->is_reduction_manager())
      {
        std::map<ReductionManager*,size_t>::iterator finder =
          reduction_instances.find(manager->as_reduction_manager());
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != reduction_instances.end());
#endif
        remaining_capacity += finder->second;
        reduction_instances.erase(finder);
      }
      else
      {
        std::map<InstanceManager*,size_t>::iterator finder = 
          physical_instances.find(manager->as_instance_manager());
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != physical_instances.end());
#endif
        remaining_capacity += finder->second;
        physical_instances.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    size_t MemoryManager::sample_allocated_space(void)
    //--------------------------------------------------------------------------
    {
      return (capacity - remaining_capacity); 
    }

    //--------------------------------------------------------------------------
    size_t MemoryManager::sample_free_space(void)
    //--------------------------------------------------------------------------
    {
      return remaining_capacity;
    }

    //--------------------------------------------------------------------------
    unsigned MemoryManager::sample_allocated_instances(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
      return (physical_instances.size() + reduction_instances.size());
    }

    /////////////////////////////////////////////////////////////
    // Virtual Channel 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(VirtualChannelKind kind, 
        AddressSpaceID local_address_space, 
        size_t max_message_size, bool profile)
      : sending_buffer((char*)malloc(max_message_size)), 
        sending_buffer_size(max_message_size), 
        observed_recent(false), profile_messages(profile)
    //--------------------------------------------------------------------------
    {
      send_lock = Reservation::create_reservation();
      receiving_buffer_size = max_message_size;
      receiving_buffer = (char*)legion_malloc(MESSAGE_BUFFER_ALLOC,
                                              receiving_buffer_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(sending_buffer != NULL);
      assert(receiving_buffer != NULL);
#endif
      // Set up the buffer for sending the first batch of messages
      // Only need to write the processor once
      *((HLRTaskID*)sending_buffer) = HLR_MESSAGE_ID;
      sending_index = sizeof(HLRTaskID);
      *((AddressSpaceID*)
          (((char*)sending_buffer)+sending_index)) = local_address_space;
      sending_index += sizeof(local_address_space);
      *((VirtualChannelKind*)
          (((char*)sending_buffer)+sending_index)) = kind;
      sending_index += sizeof(kind);
      header = FULL_MESSAGE;
      sending_index += sizeof(header);
      packaged_messages = 0;
      sending_index += sizeof(packaged_messages);
      last_message_event = Event::NO_EVENT;
      partial = false;
      // Set up the receiving buffer
      received_messages = 0;
      receiving_index = 0;
    }

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(const VirtualChannel &rhs)
      : sending_buffer(NULL), sending_buffer_size(0), profile_messages(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualChannel::~VirtualChannel(void)
    //--------------------------------------------------------------------------
    {
      send_lock.destroy_reservation();
      send_lock = Reservation::NO_RESERVATION;
      free(sending_buffer);
      free(receiving_buffer);
      receiving_buffer = NULL;
      receiving_buffer_size = 0;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::package_message(Serializer &rez, MessageKind k,
                                bool flush, Internal *runtime, Processor target)
    //--------------------------------------------------------------------------
    {
      // First check to see if the message fits in the current buffer    
      // including the overhead for the message: kind and size
      size_t buffer_size = rez.get_used_bytes();
      const char *buffer = (const char*)rez.get_buffer();
      // Need to hold the lock when manipulating the buffer
      AutoLock s_lock(send_lock);
      if ((sending_index+buffer_size+sizeof(k)+sizeof(buffer_size)) > 
          sending_buffer_size)
      {
        // Make sure we can at least get the meta-data into the buffer
        // Since there is no partial data we can fake the flush
        if ((sending_buffer_size - sending_index) <= 
            (sizeof(k)+sizeof(buffer_size)))
          send_message(true/*complete*/, runtime, target);
        // Now can package up the meta data
        packaged_messages++;
        *((MessageKind*)(sending_buffer+sending_index)) = k;
        sending_index += sizeof(k);
        *((size_t*)(sending_buffer+sending_index)) = buffer_size;
        sending_index += sizeof(buffer_size);
        while (buffer_size > 0)
        {
          unsigned remaining = sending_buffer_size - sending_index;
          if (remaining == 0)
            send_message(false/*complete*/, runtime, target);
          remaining = sending_buffer_size - sending_index;
#ifdef DEBUG_HIGH_LEVEL
          assert(remaining > 0); // should be space after the send
#endif
          // Figure out how much to copy into the buffer
          unsigned to_copy = (remaining < buffer_size) ? 
                                            remaining : buffer_size;
          memcpy(sending_buffer+sending_index,buffer,to_copy);
          buffer_size -= to_copy;
          buffer += to_copy;
          sending_index += to_copy;
        } 
      }
      else
      {
        packaged_messages++;
        // Package up the kind and the size first
        *((MessageKind*)(sending_buffer+sending_index)) = k;
        sending_index += sizeof(k);
        *((size_t*)(sending_buffer+sending_index)) = buffer_size;
        sending_index += sizeof(buffer_size);
        // Then copy over the buffer
        memcpy(sending_buffer+sending_index,buffer,buffer_size); 
        sending_index += buffer_size;
      }
      if (flush)
        send_message(true/*complete*/, runtime, target);
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::send_message(bool complete, Internal *runtime,
                                      Processor target)
    //--------------------------------------------------------------------------
    {
      // See if we need to switch the header file
      // and update the state of partial
      if (!complete)
      {
        header = PARTIAL_MESSAGE;
        partial = true;
      }
      else if (partial)
      {
        header = FINAL_MESSAGE;
        partial = false;
      }
      // Save the header and the number of messages into the buffer
      const size_t base_size = sizeof(HLRTaskID) + sizeof(AddressSpaceID) 
                                + sizeof(VirtualChannelKind);
      *((MessageHeader*)(sending_buffer + base_size)) = header;
      *((unsigned*)(sending_buffer + base_size + sizeof(header))) = 
                                                            packaged_messages;
      // Send the message
      Event next_event = runtime->issue_runtime_meta_task(sending_buffer, 
                                      sending_index, HLR_MESSAGE_ID, NULL,
                                      last_message_event, 0/*priority*/,
                                      false/*hold reservation*/, target);
      // Update the event
      last_message_event = next_event;
      // Reset the state of the buffer
      sending_index = base_size + sizeof(header) + sizeof(unsigned);
      if (partial)
        header = PARTIAL_MESSAGE;
      else
        header = FULL_MESSAGE;
      packaged_messages = 0;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::process_message(const void *args, size_t arglen,
                         Internal *runtime, AddressSpaceID remote_address_space)
    //--------------------------------------------------------------------------
    {
      // Strip off our header and the number of messages, the 
      // processor part was already stipped off by the Legion runtime
      const char *buffer = (const char*)args;
      MessageHeader head = *((const MessageHeader*)buffer);
      buffer += sizeof(head);
      arglen -= sizeof(head);
      unsigned num_messages = *((const unsigned*)buffer);
      buffer += sizeof(num_messages);
      arglen -= sizeof(num_messages);
      switch (head)
      {
        case FULL_MESSAGE:
          {
            // Can handle these messages directly
            handle_messages(num_messages, runtime, 
                            remote_address_space, buffer, arglen);
            break;
          }
        case PARTIAL_MESSAGE:
          {
            // Save these messages onto the receiving buffer
            // but do not handle them
            buffer_messages(num_messages, buffer, arglen);
            break;
          }
        case FINAL_MESSAGE:
          {
            // Save the remaining messages onto the receiving
            // buffer, then handle them and reset the state.
            buffer_messages(num_messages, buffer, arglen);
            handle_messages(received_messages, runtime,
                            remote_address_space,
                            receiving_buffer, receiving_index);
            receiving_index = 0;
            received_messages = 0;
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::handle_messages(unsigned num_messages,
                                         Internal *runtime,
                                         AddressSpaceID remote_address_space,
                                         const char *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // For profiling if we are doing it
      unsigned long long start = 0, stop = 0;
      for (unsigned idx = 0; idx < num_messages; idx++)
      {
        // Pull off the message kind and the size of the message
#ifdef DEBUG_HIGH_LEVEL
        assert(arglen >= (sizeof(MessageKind)+sizeof(size_t)));
#endif
        MessageKind kind = *((const MessageKind*)args);
        // Any message that is not a shutdown message needs to be recorded
        if (!observed_recent && (kind != SEND_SHUTDOWN_NOTIFICATION) &&
            (kind != SEND_SHUTDOWN_RESPONSE))
          observed_recent = true;
        args += sizeof(kind);
        arglen -= sizeof(kind);
        size_t message_size = *((const size_t*)args);
        args += sizeof(message_size);
        arglen -= sizeof(message_size);
#ifdef DEBUG_HIGH_LEVEL
        if (idx == (num_messages-1))
          assert(message_size == arglen);
#endif
        if (profile_messages)
          start = Realm::Clock::current_time_in_microseconds();
        // Build the deserializer
        Deserializer derez(args,message_size);
        switch (kind)
        {
          case TASK_MESSAGE:
            {
              runtime->handle_task(derez);
              break;
            }
          case STEAL_MESSAGE:
            {
              runtime->handle_steal(derez);
              break;
            }
          case ADVERTISEMENT_MESSAGE:
            {
              runtime->handle_advertisement(derez);
              break;
            }
          case SEND_INDEX_SPACE_NODE:
            {
              runtime->handle_index_space_node(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_REQUEST:
            {
              runtime->handle_index_space_request(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_RETURN:
            {
              runtime->handle_index_space_return(derez);
              break;
            }
          case SEND_INDEX_SPACE_CHILD_REQUEST:
            {
              runtime->handle_index_space_child_request(derez);
              break;
            }
          case SEND_INDEX_PARTITION_NODE:
            {
              runtime->handle_index_partition_node(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_REQUEST:
            {
              runtime->handle_index_partition_request(derez, 
                                                      remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_RETURN:
            {
              runtime->handle_index_partition_return(derez);
              break;
            }
          case SEND_INDEX_PARTITION_CHILD_REQUEST:
            {
              runtime->handle_index_partition_child_request(derez);
              break;
            }
          case SEND_FIELD_SPACE_NODE:
            {
              runtime->handle_field_space_node(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_REQUEST:
            {
              runtime->handle_field_space_request(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_RETURN:
            {
              runtime->handle_field_space_return(derez);
              break;
            }
          case SEND_FIELD_ALLOC_REQUEST:
            {
              runtime->handle_field_alloc_request(derez);
              break;
            }
          case SEND_FIELD_ALLOC_NOTIFICATION:
            {
              runtime->handle_field_alloc_notification(derez);
              break;
            }
          case SEND_FIELD_SPACE_TOP_ALLOC:
            {
              runtime->handle_field_space_top_alloc(derez,remote_address_space);
              break;
            }
          case SEND_FIELD_FREE:
            {
              runtime->handle_field_free(derez, remote_address_space);
              break;
            }
          case SEND_TOP_LEVEL_REGION_REQUEST:
            {
              runtime->handle_top_level_region_request(derez, 
                                                       remote_address_space);
              break;
            }
          case SEND_TOP_LEVEL_REGION_RETURN:
            {
              runtime->handle_top_level_region_return(derez);
              break;
            }
          case SEND_LOGICAL_REGION_NODE:
            {
              runtime->handle_logical_region_node(derez, remote_address_space);
              break;
            }
          case INDEX_SPACE_DESTRUCTION_MESSAGE:
            {
              runtime->handle_index_space_destruction(derez, 
                                                      remote_address_space);
              break;
            }
          case INDEX_PARTITION_DESTRUCTION_MESSAGE:
            {
              runtime->handle_index_partition_destruction(derez, 
                                                          remote_address_space);
              break;
            }
          case FIELD_SPACE_DESTRUCTION_MESSAGE:
            {
              runtime->handle_field_space_destruction(derez, 
                                                      remote_address_space);
              break;
            }
          case LOGICAL_REGION_DESTRUCTION_MESSAGE:
            {
              runtime->handle_logical_region_destruction(derez, 
                                                         remote_address_space);
              break;
            }
          case LOGICAL_PARTITION_DESTRUCTION_MESSAGE:
            {
              runtime->handle_logical_partition_destruction(derez, 
                                                          remote_address_space);
              break;
            }
          case INDIVIDUAL_REMOTE_MAPPED:
            {
              runtime->handle_individual_remote_mapped(derez); 
              break;
            }
          case INDIVIDUAL_REMOTE_COMPLETE:
            {
              runtime->handle_individual_remote_complete(derez);
              break;
            }
          case INDIVIDUAL_REMOTE_COMMIT:
            {
              runtime->handle_individual_remote_commit(derez);
              break;
            }
          case SLICE_REMOTE_MAPPED:
            {
              runtime->handle_slice_remote_mapped(derez, remote_address_space);
              break;
            }
          case SLICE_REMOTE_COMPLETE:
            {
              runtime->handle_slice_remote_complete(derez);
              break;
            }
          case SLICE_REMOTE_COMMIT:
            {
              runtime->handle_slice_remote_commit(derez);
              break;
            }
          case DISTRIBUTED_REMOTE_REGISTRATION:
            {
              runtime->handle_did_remote_registration(derez, 
                                                      remote_address_space);
              break;
            }
          case DISTRIBUTED_VALID_UPDATE:
            {
              runtime->handle_did_remote_valid_update(derez);
              break;
            }
          case DISTRIBUTED_GC_UPDATE:
            {
              runtime->handle_did_remote_gc_update(derez); 
              break;
            }
          case DISTRIBUTED_RESOURCE_UPDATE:
            {
              runtime->handle_did_remote_resource_update(derez);
              break;
            }
          case DISTRIBUTED_CREATE_ADD:
            {
              runtime->handle_did_create_add(derez);
              break;
            }
          case DISTRIBUTED_CREATE_REMOVE:
            {
              runtime->handle_did_create_remove(derez);
              break;
            }
          case VIEW_REMOTE_REGISTRATION:
            {
              runtime->handle_view_remote_registration(derez, 
                                                       remote_address_space);
              break;
            }
          case SEND_BACK_ATOMIC:
            {
              runtime->handle_send_back_atomic(derez, remote_address_space);
              break;
            }
          case SEND_MATERIALIZED_VIEW:
            {
              runtime->handle_send_materialized_view(derez, 
                                                     remote_address_space);
              break;
            }
          case SEND_MATERIALIZED_UPDATE:
            {
              runtime->handle_send_materialized_update(derez,
                                                       remote_address_space);
              break;
            }
          case SEND_COMPOSITE_VIEW:
            {
              runtime->handle_send_composite_view(derez, remote_address_space);
              break;
            }
          case SEND_FILL_VIEW:
            {
              runtime->handle_send_fill_view(derez, remote_address_space);
              break;
            }
          case SEND_DEFERRED_UPDATE:
            {
              runtime->handle_send_deferred_update(derez, remote_address_space);
              break;
            }
          case SEND_REDUCTION_VIEW:
            {
              runtime->handle_send_reduction_view(derez, remote_address_space);
              break;
            }
          case SEND_REDUCTION_UPDATE:
            {
              runtime->handle_send_reduction_update(derez, 
                                                    remote_address_space);
              break;
            }
          case SEND_INSTANCE_MANAGER:
            {
              runtime->handle_send_instance_manager(derez, 
                                                    remote_address_space);
              break;
            }
          case SEND_REDUCTION_MANAGER:
            {
              runtime->handle_send_reduction_manager(derez,
                                                     remote_address_space);
              break;
            }
          case SEND_FUTURE:
            {
              runtime->handle_future_send(derez, remote_address_space);
              break;
            }
          case SEND_FUTURE_RESULT:
            {
              runtime->handle_future_result(derez);
              break;
            }
          case SEND_FUTURE_SUBSCRIPTION:
            {
              runtime->handle_future_subscription(derez);
              break;
            }
          case SEND_MAKE_PERSISTENT:
            {
              runtime->handle_make_persistent(derez, remote_address_space);
              break;
            }
          case SEND_UNMAKE_PERSISTENT:
            {
              runtime->handle_unmake_persistent(derez, remote_address_space);
              break;
            }
          case SEND_MAPPER_MESSAGE:
            {
              runtime->handle_mapper_message(derez);
              break;
            }
          case SEND_MAPPER_BROADCAST:
            {
              runtime->handle_mapper_broadcast(derez);
              break;
            }
          case SEND_TASK_IMPL_SEMANTIC_REQ:
            {
              runtime->handle_task_impl_semantic_request(derez, 
                                                        remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_SEMANTIC_REQ:
            {
              runtime->handle_index_space_semantic_request(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_SEMANTIC_REQ:
            {
              runtime->handle_index_partition_semantic_request(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_SEMANTIC_REQ:
            {
              runtime->handle_field_space_semantic_request(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_FIELD_SEMANTIC_REQ:
            {
              runtime->handle_field_semantic_request(derez, 
                                                     remote_address_space);
              break;
            }
          case SEND_LOGICAL_REGION_SEMANTIC_REQ:
            {
              runtime->handle_logical_region_semantic_request(derez,
                                                          remote_address_space);
              break;
            }
          case SEND_LOGICAL_PARTITION_SEMANTIC_REQ:
            {
              runtime->handle_logical_partition_semantic_request(derez,
                                                          remote_address_space);
              break;
            }
          case SEND_TASK_IMPL_SEMANTIC_INFO:
            {
              runtime->handle_task_impl_semantic_info(derez,
                                                      remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_index_space_semantic_info(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_index_partition_semantic_info(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_field_space_semantic_info(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_FIELD_SEMANTIC_INFO:
            {
              runtime->handle_field_semantic_info(derez, remote_address_space);
              break;
            }
          case SEND_LOGICAL_REGION_SEMANTIC_INFO:
            {
              runtime->handle_logical_region_semantic_info(derez,
                                                          remote_address_space);
              break;
            }
          case SEND_LOGICAL_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_logical_partition_semantic_info(derez,
                                                          remote_address_space);
              break;
            }
          case SEND_SUBSCRIBE_REMOTE_CONTEXT:
            {
              runtime->handle_subscribe_remote_context(derez, 
                                                       remote_address_space);
              break;
            }
          case SEND_FREE_REMOTE_CONTEXT:
            {
              runtime->handle_free_remote_context(derez);
              break;
            }
          case SEND_VERSION_STATE_PATH:
            {
              runtime->handle_version_state_path_only(derez, 
                                                      remote_address_space);
              break;
            }
          case SEND_VERSION_STATE_INIT:
            {
              runtime->handle_version_state_initialization(derez, 
                                                          remote_address_space);
              break;
            }
          case SEND_VERSION_STATE_REQUEST:
            {
              runtime->handle_version_state_request(derez);
              break;
            }
          case SEND_VERSION_STATE_RESPONSE:
            {
              runtime->handle_version_state_response(derez,
                                                     remote_address_space);
              break;
            }
          case SEND_INSTANCE_CREATION:
            {
              runtime->handle_remote_instance_creation(derez, 
                                                       remote_address_space);
              break;
            }
          case SEND_REDUCTION_CREATION:
            {
              runtime->handle_remote_reduction_creation(derez,
                                                        remote_address_space);
              break;
            }
          case SEND_CREATION_RESPONSE:
            {
              runtime->handle_remote_creation_response(derez);
              break;
            }
          case SEND_BACK_LOGICAL_STATE:
            {
              runtime->handle_logical_state_return(derez, remote_address_space);
              break;
            }
          case SEND_VARIANT_REQUEST:
            {
              runtime->handle_variant_request(derez, remote_address_space);
              break;
            }
          case SEND_VARIANT_RESPONSE:
            {
              runtime->handle_variant_response(derez);
              break;
            }
          case SEND_CONSTRAINT_REQUEST:
            {
              runtime->handle_constraint_request(derez, remote_address_space);
              break;
            }
          case SEND_CONSTRAINT_RESPONSE:
            {
              runtime->handle_constraint_response(derez);
              break;
            }
          case SEND_CONSTRAINT_RELEASE:
            {
              runtime->handle_constraint_release(derez, remote_address_space);
              break;
            }
          case SEND_SHUTDOWN_NOTIFICATION:
            {
              runtime->handle_shutdown_notification(remote_address_space);
              break;
            }
          case SEND_SHUTDOWN_RESPONSE:
            {
              runtime->handle_shutdown_response(derez, remote_address_space);
              break;
            }
          default:
            assert(false); // should never get here
        }
        if (profile_messages)
        {
          stop = Realm::Clock::current_time_in_microseconds();
#ifdef DEBUG_HIGH_LEVEL
          assert(runtime->profiler != NULL);
#endif
          runtime->profiler->record_message(kind, start, stop);
        }
        // Update the args and arglen
        args += message_size;
        arglen -= message_size;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(arglen == 0); // make sure we processed everything
#endif
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::buffer_messages(unsigned num_messages,
                                         const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      received_messages += num_messages;
      // Check to see if it fits
      if (receiving_buffer_size < (receiving_index+arglen))
      {
        // Figure out what the new size should be
        // Keep doubling until it's larger
        size_t new_buffer_size = receiving_buffer_size;
        while (new_buffer_size < (receiving_index+arglen))
          new_buffer_size *= 2;
#ifdef DEBUG_HIGH_LEVEL
        assert(new_buffer_size != 0); // would cause deallocation
#endif
        // Now realloc the memory
        void *new_ptr = legion_realloc(MESSAGE_BUFFER_ALLOC, receiving_buffer,
                                       receiving_buffer_size, new_buffer_size);
        receiving_buffer_size = new_buffer_size;
#ifdef DEBUG_HIGH_LEVEL
        assert(new_ptr != NULL);
#endif
        receiving_buffer = (char*)new_ptr;
      }
      // Copy the data in
      memcpy(receiving_buffer+receiving_index,args,arglen);
      receiving_index += arglen;
    }

    //--------------------------------------------------------------------------
    Event VirtualChannel::notify_pending_shutdown(void)
    //--------------------------------------------------------------------------
    {
      return last_message_event;
    }

    //--------------------------------------------------------------------------
    bool VirtualChannel::has_recent_messages(void) const
    //--------------------------------------------------------------------------
    {
      return observed_recent;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::clear_recent_messages(void)
    //--------------------------------------------------------------------------
    {
      observed_recent = false;
    }

    /////////////////////////////////////////////////////////////
    // Message Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(AddressSpaceID remote,
                                   Internal *rt, size_t max_message_size,
                                   const std::set<Processor> &remote_util_procs)
      : remote_address_space(remote), runtime(rt), channels((VirtualChannel*)
                      malloc(MAX_NUM_VIRTUAL_CHANNELS*sizeof(VirtualChannel))) 
    //--------------------------------------------------------------------------
    {
      // Figure out which processor to send to based on our address
      // space ID.  If there is an explicit utility processor for one
      // of the processors in our set then we use that.  Otherwise we
      // round-robin senders onto different target processors on the
      // remote node to avoid over-burdening any one of them with messages.
      {
        unsigned idx = 0;
        const unsigned target_idx = rt->address_space % 
                                    remote_util_procs.size();
        // Iterate over all the processors and either choose a 
        // utility processor to be our target or get the target processor
        target = Processor::NO_PROC;
        for (std::set<Processor>::const_iterator it = 
              remote_util_procs.begin(); it != 
              remote_util_procs.end(); it++,idx++)
        {
          if (idx == target_idx)
            target = (*it);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(target.exists());
#endif
      }
      // Initialize our virtual channels 
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        new (channels+idx) VirtualChannel((VirtualChannelKind)idx,
            rt->address_space, max_message_size, (runtime->profiler != NULL));
      }
    }

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(const MessageManager &rhs)
      : remote_address_space(0), runtime(NULL), channels(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MessageManager::~MessageManager(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        channels[idx].~VirtualChannel();
      }
      free(channels);
    }

    //--------------------------------------------------------------------------
    MessageManager& MessageManager::operator=(const MessageManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Event MessageManager::notify_pending_shutdown(void)
    //--------------------------------------------------------------------------
    {
      std::set<Event> wait_events;
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        wait_events.insert(channels[idx].notify_pending_shutdown());
      }
      return Event::merge_events(wait_events);
    }

    //--------------------------------------------------------------------------
    bool MessageManager::has_recent_messages(void) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        if (channels[idx].has_recent_messages())
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void MessageManager::clear_recent_messages(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].clear_recent_messages();
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_message(Serializer &rez, MessageKind kind,
                                      VirtualChannelKind channel, bool flush)
    //--------------------------------------------------------------------------
    {
      channels[channel].package_message(rez, kind, flush, runtime, target);
    }

    //--------------------------------------------------------------------------
    void MessageManager::receive_message(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Pull the channel off to do the receiving
      const char *buffer = (const char*)args;
      VirtualChannelKind channel = *((const VirtualChannelKind*)buffer);
      buffer += sizeof(channel);
      arglen -= sizeof(channel);
      channels[channel].process_message(buffer, arglen, runtime, 
                                        remote_address_space);
    }

    /////////////////////////////////////////////////////////////
    // Shutdown Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(Internal *rt, AddressSpaceID s,
                                     MessageManager *m)
      : runtime(rt), source(s), source_manager(m),
        shutdown_lock(Reservation::create_reservation()), 
        observed_responses(0), result(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(const ShutdownManager &rhs)
      : runtime(NULL), source(0), source_manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShutdownManager::~ShutdownManager(void)
    //--------------------------------------------------------------------------
    {
      shutdown_lock.destroy_reservation();
      shutdown_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    ShutdownManager& ShutdownManager::operator=(const ShutdownManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::has_managers(void) const
    //--------------------------------------------------------------------------
    {
      // No need for the lock
      return !managers.empty();
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::add_manager(AddressSpaceID target, 
                                      MessageManager *manager)
    //--------------------------------------------------------------------------
    {
      // No need for the lock
#ifdef DEBUG_HIGH_LEVEL
      assert(target != source);
      assert(managers.find(target) == managers.end());
#endif
      managers[target] = manager;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::send_notifications(void)
    //--------------------------------------------------------------------------
    {
      NotificationArgs args;
      args.hlr_id = HLR_SHUTDOWN_NOTIFICATION_TASK_ID;
      // Clean out our managers and then send the messages
      for (std::map<AddressSpaceID,MessageManager*>::const_iterator it = 
            managers.begin(); it != managers.end(); it++)
      {
        log_shutdown.info("Sending notification from node %d to %d",
                          runtime->address_space, it->first);
        it->second->clear_recent_messages();
        Event precondition = it->second->notify_pending_shutdown();
        args.manager = it->second;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_SHUTDOWN_NOTIFICATION_TASK_ID,
                                         NULL, precondition);
      }
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::send_response(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(source_manager != NULL);
#endif
      log_shutdown.info("Sending response from node %d to %d",
                        runtime->address_space, source);
      ResponseArgs args;
      args.hlr_id = HLR_SHUTDOWN_RESPONSE_TASK_ID;
      args.target = source_manager;
      args.result = result;
      Event precondition = source_manager->notify_pending_shutdown();
      runtime->issue_runtime_meta_task(&args, sizeof(args),
                                       HLR_SHUTDOWN_RESPONSE_TASK_ID, NULL,
                                       precondition);
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::handle_response(AddressSpaceID sender, bool res)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(managers.find(sender) != managers.end());
#endif
      log_shutdown.info("Received response on node %d from %d",
                        runtime->address_space, sender);
      AutoLock shut(shutdown_lock);
      if (!res)
        result = false;
      observed_responses++;
      return (observed_responses == managers.size());
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_outstanding_profiling_requests(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::finalize(void)
    //--------------------------------------------------------------------------
    {
      if (result && !managers.empty())
      {
        // Check our managers for messages
        for (std::map<AddressSpaceID,MessageManager*>::const_iterator it = 
              managers.begin(); it != managers.end(); it++)
        {
          if (it->second->has_recent_messages())
          {
            result = false;
            break;
          }
        }
      }
      // No need for the lock here
      if (source != runtime->address_space)
      {
        send_response();
      }
      else if (result)
      {
        // We succeeded so shutdown the runtime
        runtime->finalize_runtime_shutdown();
      }
      else
      {
#ifdef DEBUG_SHUTDOWN_HANG
        HLR_TASK_DESCRIPTIONS(task_descs);
        // Only need to see tasks less than this 
        for (unsigned idx = 0; idx < HLR_MESSAGE_ID; idx++)
        {
          if (runtime->outstanding_counts[idx] == 0)
            continue;
          log_shutdown.info("Meta-Task %s: %d outstanding",
                task_descs[idx], runtime->outstanding_counts[idx]);
        }
#endif
        // We failed, so try again
        log_shutdown.info("FAILED SHUTDOWN!  Trying again...");
        runtime->issue_runtime_shutdown_attempt();
      }
    }

    /////////////////////////////////////////////////////////////
    // Garbage Collection Epoch 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GarbageCollectionEpoch::GarbageCollectionEpoch(Internal *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    GarbageCollectionEpoch::GarbageCollectionEpoch(
                                              const GarbageCollectionEpoch &rhs)
      : runtime(rhs.runtime)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    GarbageCollectionEpoch::~GarbageCollectionEpoch(void)
    //--------------------------------------------------------------------------
    {
      runtime->complete_gc_epoch(this);
    }

    //--------------------------------------------------------------------------
    GarbageCollectionEpoch& GarbageCollectionEpoch::operator=(
                                              const GarbageCollectionEpoch &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void GarbageCollectionEpoch::add_collection(LogicalView *view, Event term)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalView*,std::set<Event> >::iterator finder = 
        collections.find(view);
      if (finder == collections.end())
      {
        // Add a garbage collection reference to the view, it will
        // be removed in LogicalView::handle_deferred_collect
        view->add_base_gc_ref(PENDING_GC_REF);
        collections[view].insert(term);
      }
      else
        finder->second.insert(term);
    }

    //--------------------------------------------------------------------------
    Event GarbageCollectionEpoch::launch(int priority)
    //--------------------------------------------------------------------------
    {
      // Set remaining to the total number of collections
      remaining = collections.size();
      GarbageCollectionArgs args;
      args.hlr_id = HLR_DEFERRED_COLLECT_ID;
      args.epoch = this;
      std::set<Event> events;
      for (std::map<LogicalView*,std::set<Event> >::const_iterator it =
            collections.begin(); it != collections.end(); /*nothing*/)
      {
        args.view = it->first;
        Event precondition = Event::merge_events(it->second);
        // Avoid the deletion race by testing the condition 
        // before launching the task
        it++;
        bool done = (it == collections.end());
        Event e = runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                         HLR_DEFERRED_COLLECT_ID, NULL,
                                         precondition, priority);
        events.insert(e);
        if (done)
          break;
      }
      return Event::merge_events(events);
    }

    //--------------------------------------------------------------------------
    bool GarbageCollectionEpoch::handle_collection(
                                              const GarbageCollectionArgs *args)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalView*,std::set<Event> >::iterator finder = 
        collections.find(args->view);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != collections.end());
#endif
      LogicalView::handle_deferred_collect(args->view, finder->second);
      // See if we are done
      return (__sync_add_and_fetch(&remaining, -1) == 0);
    }

    /////////////////////////////////////////////////////////////
    // Pending Registrations 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingVariantRegistration::PendingVariantRegistration(VariantID v,
                                  bool has_ret, const TaskVariantRegistrar &reg,
                                  const void *udata, size_t udata_size,
                                  CodeDescriptor *realm,
                                  const char *task_name)
      : vid(v), has_return(has_ret), registrar(reg), 
        realm_desc(realm), logical_task_name(NULL)
    //--------------------------------------------------------------------------
    {
      // If we're doing a pending registration, this is a static
      // registration so we don't have to register it globally
      registrar.global_registration = false;
      // Make sure we own the task variant name
      if (reg.task_variant_name != NULL)
        registrar.task_variant_name = strdup(reg.task_variant_name);
      // We need to own the user data too
      if (udata != NULL)
      {
        user_data_size = udata_size;
        user_data = malloc(user_data_size);
        memcpy(user_data,udata,user_data_size);
      }
      else
      {
        user_data_size = 0;
        user_data = NULL;
      }
      if (task_name != NULL)
        logical_task_name = strdup(task_name);
    }

    //--------------------------------------------------------------------------
    PendingVariantRegistration::PendingVariantRegistration(
                                          const PendingVariantRegistration &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PendingVariantRegistration::~PendingVariantRegistration(void)
    //--------------------------------------------------------------------------
    {
      if (registrar.task_variant_name != NULL)
        free(const_cast<char*>(registrar.task_variant_name));
      if (user_data != NULL)
        free(user_data);
      if (logical_task_name != NULL)
        free(logical_task_name);
    }

    //--------------------------------------------------------------------------
    PendingVariantRegistration& PendingVariantRegistration::operator=(
                                          const PendingVariantRegistration &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PendingVariantRegistration::perform_registration(Internal *runtime)
    //--------------------------------------------------------------------------
    {
      runtime->register_variant(registrar, user_data, user_data_size,
                      realm_desc, has_return, vid, false/*check task*/);
      // If we have a logical task name, attach the name info
      if (logical_task_name != NULL)
        runtime->attach_semantic_information(registrar.task_id, 
                          NAME_SEMANTIC_TAG, logical_task_name, 
                          strlen(logical_task_name)+1, 
                          false/*mutable*/, false/*send to owner*/);
    }

    /////////////////////////////////////////////////////////////
    // Task Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskImpl::TaskImpl(TaskID tid, Internal *rt, const char *name/*=NULL*/)
      : task_id(tid), runtime(rt), task_lock(Reservation::create_reservation()),
        has_return_type(false), all_idempotent(false)
    //--------------------------------------------------------------------------
    {
      // Always fill in semantic info 0 with a name for the task
      if (name == NULL)
      {
        const size_t name_size = 64 * sizeof(char);
        char *noname = (char*)legion_malloc(SEMANTIC_INFO_ALLOC, name_size);
        snprintf(noname,name_size,"unnamed_task_%d", task_id);
        semantic_infos[NAME_SEMANTIC_TAG] = 
          SemanticInfo(noname, name_size, true/*mutable*/);
      }
      else
      {
        const size_t name_size = strlen(name) + 1; // for \0
        char *name_copy = (char*)legion_malloc(SEMANTIC_INFO_ALLOC, name_size);
        memcpy(name_copy, name, name_size);
        semantic_infos[NAME_SEMANTIC_TAG] = 
          SemanticInfo(name_copy, name_size, false/*mutable*/);
      }
      // Register this task with the profiler if necessary
      if (runtime->profiler != NULL)
      {
        const SemanticInfo &info = semantic_infos[NAME_SEMANTIC_TAG]; 
        const char *name = (const char*)info.buffer;
        runtime->profiler->register_task_kind(task_id, name);
      }
      collection = new TaskVariantCollection(task_id, get_name(), 
                                             false/*idempotent*/, 0);
    }

    //--------------------------------------------------------------------------
    TaskImpl::TaskImpl(const TaskImpl &rhs)
      : task_id(rhs.task_id), runtime(rhs.runtime)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskImpl::~TaskImpl(void)
    //-------------------------------------------------------------------------
    {
      task_lock.destroy_reservation();
      task_lock = Reservation::NO_RESERVATION;
      for (std::map<SemanticTag,SemanticInfo>::const_iterator it = 
            semantic_infos.begin(); it != semantic_infos.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer,
                    it->second.size);
      }
      semantic_infos.clear();
      delete collection;
    }

    //--------------------------------------------------------------------------
    TaskImpl& TaskImpl::operator=(const TaskImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::add_variant(VariantImpl *impl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl->owner == this);
#endif
      AutoLock t_lock(task_lock);
      if (!variants.empty())
      {
        // Make sure that all the variants agree whether there is 
        // a return type or not
        if (has_return_type != impl->returns_value())
        {
          log_run.error("Variants of task %s (ID %d) disagree on whether "
                        "there is a return type or not. All variants "
                        "of a task must agree on whether there is a "
                        "return type.", get_name(false/*need lock*/),
                        task_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RETURN_SIZE_MISMATCH);
        }
        if (all_idempotent != impl->is_idempotent())
        {
          log_run.error("Variants of task %s (ID %d) have different idempotent "
                        "options.  All variants of the same task must "
                        "all be either idempotent or non-idempotent.",
                        get_name(false/*need lock*/), task_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_IDEMPOTENT_MISMATCH);
        }
      }
      else
      {
        has_return_type = impl->returns_value();
        all_idempotent  = impl->is_idempotent();
      }
      variants[impl->vid] = impl;
      collection->add_variant(impl->vid, impl->get_processor_kind(false),
                              true/*single*/, true/*index*/,
                              impl->is_inner(), impl->is_leaf(), impl->vid); 
    }

    //--------------------------------------------------------------------------
    VariantImpl* TaskImpl::find_variant_impl(VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      // See if we already have the variant
      {
        AutoLock t_lock(task_lock,1,false/*exclusive*/);
        std::map<VariantID,VariantImpl*>::const_iterator finder = 
          variants.find(variant_id);
        if (finder != variants.end())
          return finder->second;
      }
      // If we don't have it, see if we can go get it
      AddressSpaceID owner_space = 
        VariantImpl::get_owner_space(variant_id, runtime); 
      if (owner_space == runtime->address_space)
      {
        log_run.error("Unable to find variant %ld of task %s!",
                      variant_id, get_name(false));
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNREGISTERED_VARIANT);
      }
      // Send a request to the owner node for the variant
      UserEvent wait_on = UserEvent::create_user_event(); 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(variant_id);
        rez.serialize(wait_on);
        runtime->send_variant_request(owner_space, rez);
      }
      // Wait for the results
      wait_on.wait();
      // Now we can re-take the lock and find our variant
      AutoLock t_lock(task_lock,1,false/*exclusive*/);
      std::map<VariantID,VariantImpl*>::const_iterator finder = 
        variants.find(variant_id);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != variants.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    TaskVariantCollection* TaskImpl::get_collection(void)
    //--------------------------------------------------------------------------
    {
      return collection;
    }

    //--------------------------------------------------------------------------
    const char* TaskImpl::get_name(bool needs_lock /*= true*/) const
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock t_lock(task_lock,1,false/*exclusive*/);
        std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_infos.find(NAME_SEMANTIC_TAG);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != semantic_infos.end());
#endif
        return reinterpret_cast<const char*>(finder->second.buffer);
      }
      else
      {
        std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_infos.find(NAME_SEMANTIC_TAG);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != semantic_infos.end());
#endif
        return reinterpret_cast<const char*>(finder->second.buffer);
      }
    }

    //--------------------------------------------------------------------------
    void TaskImpl::attach_semantic_information(SemanticTag tag,
                                               AddressSpaceID source,
                                               const void *buffer, size_t size,
                                            bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      if ((tag == NAME_SEMANTIC_TAG) && (runtime->profiler != NULL))
        runtime->profiler->register_task_kind(task_id, (const char*)buffer);

      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      if (tag == NAME_SEMANTIC_TAG)
        collection->name = (const char*)local;
      bool added = true;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock t_lock(task_lock);
        std::map<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          // Check to see if it is valid
          if (finder->second.is_valid())
          {
            // See if it is mutable or not
            if (!finder->second.is_mutable)
            {
              // Note mutable so check to make sure that the bits are the same
              if (size != finder->second.size)
              {
                log_run.error("ERROR: Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %ld"
                              " and %ld for task impl", 
                              tag, size, finder->second.size);
#ifdef DEBUG_HIGH_LEVEL
                assert(false);
#endif
                exit(ERROR_INCONSISTENT_SEMANTIC_TAG);
              }
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    log_run.error("ERROR: Inconsistent Semantic Tag value "
                                  "for tag %ld with different values at"
                                  "byte %d for task impl, %x != %x",
                                  tag, idx, orig[idx], next[idx]);
#ifdef DEBUG_HIGH_LEVEL
                    assert(false);
#endif
                    exit(ERROR_INCONSISTENT_SEMANTIC_TAG);
                  }
                }
              }
              added = false;
            }
            else
            {
              // It is mutable so just overwrite it
              legion_free(SEMANTIC_INFO_ALLOC, 
                          finder->second.buffer, finder->second.size);
              finder->second.buffer = local;
              finder->second.size = size;
              finder->second.ready_event = UserEvent::NO_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            to_trigger = finder->second.ready_event;
            finder->second.ready_event = UserEvent::NO_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_infos[tag] = SemanticInfo(local, size, is_mutable);
      }
      if (to_trigger.exists())
        to_trigger.trigger();
      if (added)
      {
        if (send_to_owner)
        {
          AddressSpaceID owner_space = get_owner_space();
          // if we are not the owner and the message didn't come
          // from the owner, then send it
          if ((owner_space != runtime->address_space) && 
              (source != owner_space))
            send_semantic_info(owner_space, tag, buffer, size, is_mutable);
        }
      }
      else
        legion_free(SEMANTIC_INFO_ALLOC, local, size);
    }

    //--------------------------------------------------------------------------
    bool TaskImpl::retrieve_semantic_information(SemanticTag tag,
              const void *&result, size_t &size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      UserEvent wait_on = UserEvent::NO_USER_EVENT;
      {
        AutoLock t_lock(task_lock);
        std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            return true;
          }
          else if (!can_fail && wait_until)
            wait_on = finder->second.ready_event;
          else // we can fail, so make our own user event
            wait_on = UserEvent::create_user_event();
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make the ready event and record it
            wait_on = UserEvent::create_user_event();
            semantic_infos[tag] = SemanticInfo(wait_on);
          }
          else
            wait_on = UserEvent::create_user_event();
        }
      }
      // If we are not the owner, send a request otherwise we are
      // the owner and the information will get sent here
      AddressSpaceID owner_space = get_owner_space();
      if (owner_space != runtime->address_space)
        send_semantic_request(owner_space, tag, can_fail, wait_until, wait_on);
      else if (!wait_until)
      {
        if (can_fail)
          return false;
        log_run.error("Invalid semantic tag %ld for task implementation", tag);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_SEMANTIC_TAG);
      }
      wait_on.wait();
      // When we wake up, we should be able to find everything
      AutoLock t_lock(task_lock,1,false/*exclusive*/);
      std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
        semantic_infos.find(tag);
      if (finder == semantic_infos.end())
      {
        if (can_fail)
          return false;
        log_run.error("ERROR: invalid semantic tag %ld for "
                            "task implementation", tag);   
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_SEMANTIC_TAG);
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_info(AddressSpaceID target, SemanticTag tag,
                               const void *buffer, size_t size, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
      }
      runtime->send_task_impl_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_request(AddressSpaceID target, 
               SemanticTag tag, bool can_fail, bool wait_until, UserEvent ready)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      runtime->send_task_impl_semantic_request(target, rez);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::process_semantic_request(SemanticTag tag, 
         AddressSpaceID target, bool can_fail, bool wait_until, UserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(get_owner_space() == runtime->address_space);
#endif
      Event precondition = Event::NO_EVENT;
      void *result = NULL;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock t_lock(task_lock);
        // See if we already have the data
        std::map<SemanticTag,SemanticInfo>::iterator finder = 
          semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer;
            size = finder->second.size;
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          UserEvent ready_event = UserEvent::create_user_event();
          precondition = ready_event;
          semantic_infos[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        if (can_fail || !wait_until)
          ready.trigger(); // this will cause a failure on the original node 
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args;
          args.hlr_id = HLR_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID;
          args.proxy_this = this;
          args.tag = tag;
          args.source = target;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                        HLR_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID,
                        NULL/*op*/, precondition);
        }
      }
      else
        send_semantic_info(target, tag, result, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_semantic_request(Internal *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      UserEvent ready;
      derez.deserialize(ready);
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      impl->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_semantic_info(Internal *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      impl->attach_semantic_information(tag, source, buffer, size, 
                                        is_mutable, false/*send to owner*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID TaskImpl::get_owner_space(TaskID task_id,
                                                        Internal *runtime)
    //--------------------------------------------------------------------------
    {
      return (task_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_variant_request(Internal *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      TaskID task_id;
      derez.deserialize(task_id);
      VariantID variant_id;
      derez.deserialize(variant_id);
      UserEvent done_event;
      derez.deserialize(done_event);
      TaskImpl *task_impl = runtime->find_task_impl(task_id);
      VariantImpl *var_impl = task_impl->find_variant_impl(variant_id);
      var_impl->send_variant_response(source, done_event);
    }

    /////////////////////////////////////////////////////////////
    // Variant Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VariantImpl::VariantImpl(Internal *rt, VariantID v, TaskImpl *own, 
                           const TaskVariantRegistrar &registrar, bool ret,
                           CodeDescriptor *realm,
                           const void *udata /*=NULL*/, size_t udata_size/*=0*/)
      : vid(v), owner(own), runtime(rt), global(registrar.global_registration),
        has_return_value(ret), realm_descriptor(realm),
        execution_constraints(registrar.execution_constraints),
        layout_constraints(registrar.layout_constraints),
        user_data_size(udata_size), leaf_variant(registrar.leaf_variant), 
        inner_variant(registrar.inner_variant),
        idempotent_variant(registrar.idempotent_variant)
    //--------------------------------------------------------------------------
    { 
      if (udata != NULL)
      {
        user_data = malloc(user_data_size);
        memcpy(user_data, udata, user_data_size);
      }
      else
        user_data = NULL;
      // Perform the registration, the normal case is not to have separate
      // runtime instances, but if we do have them, we only register on
      // the local processor
      if (!Internal::separate_runtime_instances)
      {
        Realm::ProfilingRequestSet profiling_requests;
        ready_event = Processor::register_task_by_kind(
            get_processor_kind(true), false/*global*/, vid, *realm_descriptor,
            profiling_requests, user_data, user_data_size);
      }
      else
      {
        // This is a debug case for when we have one runtime instance
        // for each processor
        Processor proc = Processor::get_executing_processor();
        Realm::ProfilingRequestSet profiling_requests;
        ready_event = proc.register_task(vid, *realm_descriptor,
            profiling_requests, user_data, user_data_size);
      }
      // If we have a variant name, then record it
      if (registrar.task_variant_name == NULL)
      {
        variant_name = (char*)malloc(64*sizeof(char));
        snprintf(variant_name,64,"unnamed_variant_%ld", vid);
      }
      else
        variant_name = strdup(registrar.task_variant_name);
      // register this with the runtime profiler if we have to
      if (runtime->profiler != NULL)
        runtime->profiler->register_task_variant(own->task_id, vid,
            variant_name);
      // Check that global registration has portable implementations
      if (global && (!realm_descriptor->has_portable_implementations()))
      {
        log_run.error("Variant %s requested global registration without "
                      "a portable implementation.", variant_name);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_GLOBAL_VARIANT_REGISTRATION);
      }
      if (leaf_variant && inner_variant)
      {
        log_run.error("Task variant %s (ID %ld) of task %s (ID %d) is not "
                      "permitted to be both inner and leaf tasks "
                      "simultaneously.", variant_name, vid,
                      owner->get_name(), owner->task_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INNER_LEAF_MISMATCH);
      }
      if (Internal::record_registration)
        log_run.print("Task variant %s of task %s (ID %d) has Realm ID %ld",
                      variant_name, owner->get_name(), owner->task_id, vid);
    }

    //--------------------------------------------------------------------------
    VariantImpl::VariantImpl(const VariantImpl &rhs) 
      : vid(rhs.vid), owner(rhs.owner), runtime(rhs.runtime), 
        global(rhs.global), has_return_value(rhs.has_return_value),
        realm_descriptor(rhs.realm_descriptor) 
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VariantImpl::~VariantImpl(void)
    //--------------------------------------------------------------------------
    {
      delete realm_descriptor;
      if (user_data != NULL)
        free(user_data);
      if (variant_name != NULL)
        free(variant_name);
    }

    //--------------------------------------------------------------------------
    VariantImpl& VariantImpl::operator=(const VariantImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Event VariantImpl::dispatch_task(Processor target, SingleTask *task, 
                                     Event precondition, int priority, 
                                     Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
      // Add any profiling requests
      if (runtime->profiler != NULL)
        runtime->profiler->add_task_request(requests, vid, task);
      // Increment the number of outstanding tasks
      runtime->increment_total_outstanding_tasks();
      // If our ready event hasn't triggered, include it in the precondition
      if (!ready_event.has_triggered())
        return target.spawn(vid, &task, sizeof(task), requests,
                  Event::merge_events(precondition, ready_event), priority);
      return target.spawn(vid, &task, sizeof(task), requests, 
                          precondition, priority);
    }

    //--------------------------------------------------------------------------
    void VariantImpl::dispatch_inline(Processor current, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(realm_descriptor != NULL);
      assert(task->is_inline_task());
#endif
      const Realm::FunctionPointerImplementation *fp_impl = 
        realm_descriptor->find_impl<Realm::FunctionPointerImplementation>();
#ifdef DEBUG_HIGH_LEVEL
      assert(fp_impl != NULL);
#endif
      RealmFnptr inline_ptr = fp_impl->get_impl<RealmFnptr>();
      (*inline_ptr)(&task, sizeof(task), user_data, user_data_size, current);
    }

    //--------------------------------------------------------------------------
    Processor::Kind VariantImpl::get_processor_kind(bool warn) const
    //--------------------------------------------------------------------------
    {
      const ProcessorConstraint &constraint = 
                                  execution_constraints.processor_constraint;
      if (constraint.is_valid())
        return constraint.get_kind();
      if (warn)
        log_run.warning("WARNING: NO PROCESSOR CONSTRAINT SPECIFIED FOR VARIANT"
                        " %s (ID %ld) OF TASK %s (ID %d)! ASSUMING LOC_PROC!",
                      variant_name, vid, owner->get_name(false),owner->task_id);
      return Processor::LOC_PROC;
    }

    //--------------------------------------------------------------------------
    void VariantImpl::send_variant_response(AddressSpaceID target, Event done)
    //--------------------------------------------------------------------------
    {
      if (!global)
      {
        log_run.error("Illegal remote use of variant %s of task %s "
                      "which was not globally registered.",
                      variant_name, owner->get_name());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_USE_OF_NON_GLOBAL_VARIANT);
      }
      // Package up this variant and send it over to the target 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(owner->task_id);
        rez.serialize(vid);
        rez.serialize(has_return_value);
        // pack the code descriptors 
        Realm::Serialization::ByteCountSerializer counter;
        realm_descriptor->serialize(counter, true/*portable*/);
        const size_t impl_size = counter.bytes_used();
        rez.serialize(impl_size);
        {
          Realm::Serialization::FixedBufferSerializer 
            serializer(rez.reserve_bytes(impl_size), impl_size);
          realm_descriptor->serialize(serializer, true/*portable*/);
        }
        rez.serialize(user_data_size);
        if (user_data_size > 0)
          rez.serialize(user_data, user_data_size);
        rez.serialize(leaf_variant);
        rez.serialize(inner_variant);
        rez.serialize(idempotent_variant);
        size_t name_size = strlen(variant_name)+1;
        rez.serialize(variant_name, name_size);
        // Pack the constraints
        execution_constraints.serialize(rez);
        layout_constraints.serialize(rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID VariantImpl::get_owner_space(VariantID vid,
                                                           Internal *runtime)
    //--------------------------------------------------------------------------
    {
      return (vid % runtime->address_space);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VariantImpl::handle_variant_response(Internal *runtime,
                                                         Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      TaskVariantRegistrar registrar(task_id);
      VariantID variant_id;
      derez.deserialize(variant_id);
      bool has_return;
      derez.deserialize(has_return);
      size_t impl_size;
      derez.deserialize(impl_size);
      Realm::Serialization::FixedBufferDeserializer
        deserializer(derez.get_current_pointer(), impl_size);
      derez.advance_pointer(impl_size);
      CodeDescriptor *realm_desc = new CodeDescriptor();
      realm_desc->deserialize(deserializer);
      size_t user_data_size;
      derez.deserialize(user_data_size);
      const void *user_data = derez.get_current_pointer();
      derez.advance_pointer(user_data_size);
      derez.deserialize(registrar.leaf_variant);
      derez.deserialize(registrar.inner_variant);
      derez.deserialize(registrar.idempotent_variant);
      // The last thing will be the name
      registrar.task_variant_name = (const char*)derez.get_current_pointer();
      size_t name_size = strlen(registrar.task_variant_name)+1;
      derez.advance_pointer(name_size);
      // Unpack the constraints
      registrar.execution_constraints.deserialize(derez);
      registrar.layout_constraints.deserialize(derez);
      // Ask the runtime to perform the registration 
      runtime->register_variant(registrar, user_data, user_data_size,
                    realm_desc, has_return, variant_id, false/*check task*/);
    }

    /////////////////////////////////////////////////////////////
    // Layout Constraints 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(LayoutConstraintID id, Internal *rt,
                                     const LayoutConstraintRegistrar &registrar)
      : LayoutConstraintSet(registrar.layout_constraints), layout_id(id), 
        runtime(rt), handle(registrar.handle), ready_event(Event::NO_EVENT),
        layout_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      if (registrar.layout_name == NULL)
      {
        constraints_name = (char*)malloc(64*sizeof(char));
        snprintf(constraints_name,64,"layout constraints %ld", layout_id);
      }
      else
        constraints_name = strdup(registrar.layout_name);
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(LayoutConstraintID id, Internal *rt)
      : layout_id(id), runtime(rt), constraints_name(NULL),
        layout_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      // Make a user event for the result and send the request
      UserEvent done_event = UserEvent::create_user_event();
      // Save the done event as the ready event
      ready_event = done_event;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(layout_id);
        rez.serialize(done_event);
      }
      AddressSpaceID target = get_owner();
      runtime->send_constraint_request(target, rez);
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(const LayoutConstraints &rhs)
      : layout_id(rhs.layout_id), runtime(rhs.runtime)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::~LayoutConstraints(void)
    //--------------------------------------------------------------------------
    {
      if (constraints_name != NULL)
        free(constraints_name);
      layout_lock.destroy_reservation();
      layout_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    LayoutConstraints& LayoutConstraints::operator=(const LayoutConstraints &rh)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::ReleaseFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target != to_skip)
        runtime->send_constraint_release(target, rez);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::release_constraints(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner = get_owner();
      // See if this is the owner node or not 
      if (owner == runtime->address_space)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(layout_id);
        }
        ReleaseFunctor functor(rez, source, runtime);
        remote_instances.map(functor);
      }
      else if (source != owner)
      {
        // Send notification to the owner to do the release
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(layout_id);
        }
        runtime->send_constraint_release(owner, rez);
      }
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::send_constraints(AddressSpaceID target,
                                             UserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // Hold our lock when updating our se of remote instances
      {
        AutoLock l_lock(layout_lock);
        remote_instances.add(target);
      }
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(layout_id);
        rez.serialize(handle);
        size_t name_len = strlen(constraints_name)+1;
        rez.serialize(name_len);
        rez.serialize(constraints_name, name_len);
        // pack the constraints
        serialize(rez);   
        // pack the done events
        rez.serialize(done_event);
      }
      runtime->send_constraint_response(target, rez);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::update_constraints(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Unpack the constraints
      derez.deserialize(handle);
#ifdef DEBUG_HIGH_LEVEL
      assert(constraints_name == NULL);
#endif
      size_t name_len;
      derez.deserialize(name_len);
      constraints_name = (char*)malloc(name_len);
      derez.deserialize(constraints_name, name_len);
      // unpack the constraints
      deserialize(derez); 
      // Then trigger the event
      UserEvent done_event;
      derez.deserialize(done_event);
      done_event.trigger();
    }

    //--------------------------------------------------------------------------
    AddressSpaceID LayoutConstraints::get_owner(void) const
    //--------------------------------------------------------------------------
    {
      return get_owner(layout_id, runtime);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID LayoutConstraints::get_owner(LayoutConstraintID layout_id,
                                                Internal *runtime)
    //--------------------------------------------------------------------------
    {
      return (layout_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LayoutConstraints::process_request(Internal *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      UserEvent done_event;
      derez.deserialize(done_event);
      LayoutConstraints *constraints = runtime->find_layout_constraints(lay_id);
      constraints->send_constraints(source, done_event); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void LayoutConstraints::process_response(Internal *runtime,
                                                        Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      LayoutConstraints *constraints = runtime->find_layout_constraints(lay_id);
      constraints->update_constraints(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LayoutConstraints::process_release(Internal *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      runtime->release_layout(lay_id, source);
    }

    /////////////////////////////////////////////////////////////
    // Legion Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Internal::Internal(Machine m, AddressSpaceID unique,
                     const std::set<Processor> &locals,
                     const std::set<Processor> &local_utilities,
                     const std::set<AddressSpaceID> &address_spaces,
                     const std::map<Processor,AddressSpaceID> &processor_spaces)
      : high_level(new Runtime(this)),machine(m),address_space(unique),
        runtime_stride(address_spaces.size()), profiler(NULL),
        forest(new RegionTreeForest(this)), 
        has_explicit_utility_procs(!local_utilities.empty()), 
        total_outstanding_tasks(0), outstanding_top_level_tasks(0), 
        shutdown_manager(NULL),
        shutdown_lock(Reservation::create_reservation()),
        local_procs(locals), local_utils(local_utilities),
        memory_manager_lock(Reservation::create_reservation()),
        message_manager_lock(Reservation::create_reservation()),
        proc_spaces(processor_spaces),
        task_variant_lock(Reservation::create_reservation()),
        layout_constraints_lock(Reservation::create_reservation()),
        mapper_info_lock(Reservation::create_reservation()),
        unique_index_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_partition_id((unique == 0) ? runtime_stride : unique), 
        unique_field_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_tree_id((unique == 0) ? runtime_stride : unique),
        unique_region_tree_id((unique == 0) ? runtime_stride : unique),
        unique_operation_id((unique == 0) ? runtime_stride : unique),
        unique_field_id((unique == 0) ? runtime_stride : unique),
        unique_variant_id((unique == 0) ? runtime_stride : unique),
        unique_constraint_id((unique == 0) ? runtime_stride : unique),
        unique_task_id(get_current_static_task_id()+unique),
        unique_mapper_id(get_current_static_mapper_id()+unique),
        available_lock(Reservation::create_reservation()), total_contexts(0),
        group_lock(Reservation::create_reservation()),
        distributed_id_lock(Reservation::create_reservation()),
        unique_distributed_id((unique == 0) ? runtime_stride : unique),
        distributed_collectable_lock(Reservation::create_reservation()),
        gc_epoch_lock(Reservation::create_reservation()), gc_epoch_counter(0),
        future_lock(Reservation::create_reservation()),
        remote_lock(Reservation::create_reservation()),
        random_lock(Reservation::create_reservation()),
        individual_task_lock(Reservation::create_reservation()), 
        point_task_lock(Reservation::create_reservation()),
        index_task_lock(Reservation::create_reservation()), 
        slice_task_lock(Reservation::create_reservation()),
        remote_task_lock(Reservation::create_reservation()),
        inline_task_lock(Reservation::create_reservation()),
        map_op_lock(Reservation::create_reservation()), 
        copy_op_lock(Reservation::create_reservation()), 
        fence_op_lock(Reservation::create_reservation()),
        frame_op_lock(Reservation::create_reservation()),
        deletion_op_lock(Reservation::create_reservation()), 
        inter_close_op_lock(Reservation::create_reservation()), 
        read_close_op_lock(Reservation::create_reservation()),
        post_close_op_lock(Reservation::create_reservation()),
        virtual_close_op_lock(Reservation::create_reservation()),
        dynamic_collective_op_lock(Reservation::create_reservation()),
        future_pred_op_lock(Reservation::create_reservation()), 
        not_pred_op_lock(Reservation::create_reservation()),
        and_pred_op_lock(Reservation::create_reservation()),
        or_pred_op_lock(Reservation::create_reservation()),
        acquire_op_lock(Reservation::create_reservation()),
        release_op_lock(Reservation::create_reservation()),
        capture_op_lock(Reservation::create_reservation()),
        trace_op_lock(Reservation::create_reservation()),
        epoch_op_lock(Reservation::create_reservation()),
        pending_partition_op_lock(Reservation::create_reservation()),
        dependent_partition_op_lock(Reservation::create_reservation()),
        fill_op_lock(Reservation::create_reservation()),
        attach_op_lock(Reservation::create_reservation()),
        detach_op_lock(Reservation::create_reservation()),
        timing_op_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      log_run.debug("Initializing high-level runtime in address space %x",
                            address_space);
      // Construct a local utility processor group
      if (local_utils.empty())
      {
        // make the utility group the set of all the local processors
#ifdef DEBUG_HIGH_LEVEL
        assert(!locals.empty());
#endif
        if (locals.size() == 1)
          utility_group = *(locals.begin());
        else
        {
          std::vector<Processor> util_group(locals.begin(), locals.end());
          utility_group = Processor::create_group(util_group);
        }
      }
      else if (local_utils.size() == 1)
        utility_group = *(local_utils.begin());
      else
      {
        std::vector<Processor> util_g(local_utils.begin(), local_utils.end());
        utility_group = Processor::create_group(util_g);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(utility_group.exists());
#endif
 
      // For each of the processors in our local set, construct a manager
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it).kind() != Processor::UTIL_PROC);
#endif
	std::set<Processor> all_procs;
	machine.get_all_processors(all_procs);
        ProcessorManager *manager = new ProcessorManager(*it,
				    (*it).kind(), this,
                                    superscalar_width,
                                    DEFAULT_MAPPER_SLOTS, 
                                    stealing_disabled,
				    all_procs.size()-1);
        proc_managers[*it] = manager;
        manager->add_mapper(0, new DefaultMapper(machine, high_level, *it),
                            false/*needs check*/);
      }
      // Initialize the message manager array so that we can construct
      // message managers lazily as they are needed
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
        message_managers[idx] = NULL;
      
      // Make the default number of contexts
      // No need to hold the lock yet because nothing is running
      for (total_contexts = 0; total_contexts < DEFAULT_CONTEXTS; 
            total_contexts++)
      {
        available_contexts.push_back(RegionTreeContext(total_contexts)); 
      }
      // Create our first GC epoch
      current_gc_epoch = new GarbageCollectionEpoch(this);
      pending_gc_epochs.insert(current_gc_epoch);
      // Initialize our random number generator state
      random_state[0] = address_space & 0xFFFF; // low-order bits of node ID 
      random_state[1] = (address_space >> 16) & 0xFFFF; // high-order bits
      random_state[2] = LEGION_INIT_SEED;
      // Do some mixing
      for (int i = 0; i < 256; i++)
        nrand48(random_state);

#ifdef DEBUG_HIGH_LEVEL
      if (logging_region_tree_state)
      {
	tree_state_logger = new TreeStateLogger(address_space, 
                                                verbose_logging,
                                                logical_logging_only,
                                                physical_logging_only);
	assert(tree_state_logger != NULL);
      } else {
	tree_state_logger = NULL;
      }
#endif
#ifdef TRACE_ALLOCATION
      allocation_lock = Reservation::create_reservation(); 
      allocation_tracing_count = 0;
      // Instantiate all the kinds of allocations
      for (unsigned idx = ARGUMENT_MAP_ALLOC; idx < LAST_ALLOC; idx++)
        allocation_manager[((AllocationType)idx)] = AllocationTracker();
#endif
      // Set up the profiler if it was requested
      // If it is less than zero, all nodes enabled by default, otherwise
      // we have to be less than the maximum number to enable profiling
      if (address_space < Internal::num_profiling_nodes)
      {
        HLR_TASK_DESCRIPTIONS(hlr_task_descriptions);
        profiler = new LegionProfiler((local_utils.empty() ? 
                                      Processor::NO_PROC : utility_group), 
                                      machine, HLR_LAST_TASK_ID,
                                      hlr_task_descriptions, 
                                      Operation::LAST_OP_KIND, 
                                      Operation::op_names); 
        HLR_MESSAGE_DESCRIPTIONS(hlr_message_descriptions);
        profiler->record_message_kinds(hlr_message_descriptions,LAST_SEND_KIND);
      }
#ifdef LEGION_GC
      {
        REFERENCE_NAMES_ARRAY(reference_names);
        for (unsigned idx = 0; idx < LAST_SOURCE_REF; idx++)
        {
          log_garbage.info("GC Source Kind %d %s", idx, reference_names[idx]);
        }
      }
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      outstanding_counts.resize(HLR_LAST_TASK_ID, 0);
#endif
      // Check to see which operations we buffered before the 
      // runtime started that we now need to do
      std::deque<PendingVariantRegistration*> &pending_variants = 
        get_pending_variant_table();
      const size_t num_static_variants = 
        TASK_ID_AVAILABLE + pending_variants.size();
      if (!pending_variants.empty())
      {
        for (std::deque<PendingVariantRegistration*>::const_iterator it =
              pending_variants.begin(); it != pending_variants.end(); it++)
        {
          (*it)->perform_registration(this);
          // avoid races on seaparte runtime instances
          if (!Internal::separate_runtime_instances)
            delete *it;
        }
        // avoid races on separate runtime instances
        if (!Internal::separate_runtime_instances)
          pending_variants.clear();
      }
      // All the runtime instances registered the static variants
      // starting at 1 and counting by 1, so just increment our
      // unique_variant_id until it is greater than the
      // number of static variants, no need to use atomics
      // here since we are still initializing the runtime
      while (unique_variant_id <= num_static_variants)
        unique_variant_id += runtime_stride;
      // Register any pending constraint sets
      std::map<LayoutConstraintID,LayoutConstraintRegistrar> 
        &pending_constraints = get_pending_constraint_table();
      if (!pending_constraints.empty())
      {
        // Update the next available constraint
        while (pending_constraints.find(unique_constraint_id) !=
                pending_constraints.end())
          unique_constraint_id += runtime_stride;
        // Now do the registrations
        for (std::map<LayoutConstraintID,LayoutConstraintRegistrar>::
              const_iterator it = pending_constraints.begin(); 
              it != pending_constraints.end(); it++)
        {
          register_layout(it->second, it->first);
        }
        // avoid races if we are doing separate runtime creation
        if (!Internal::separate_runtime_instances)
          pending_constraints.clear();
      } 

      // Before launching the top level task, see if the user requested
      // a callback to be performed before starting the application
      if (Internal::registration_callback != NULL)
        (*Internal::registration_callback)(machine, high_level, 
                                                local_procs);
    }

    //--------------------------------------------------------------------------
    Internal::Internal(const Internal &rhs)
      : high_level(NULL), machine(rhs.machine), address_space(0), 
        runtime_stride(0), profiler(NULL), forest(NULL),
        has_explicit_utility_procs(false),
        local_procs(rhs.local_procs), proc_spaces(rhs.proc_spaces)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Internal::~Internal(void)
    //--------------------------------------------------------------------------
    {
      if (profiler != NULL)
      {
        profiler->finalize();
        delete profiler;
        profiler = NULL;
      }
      delete high_level;
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        delete it->second;
      }
      proc_managers.clear();
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
      {
        if (message_managers[idx] != NULL)
          delete message_managers[idx];
      }
      for (std::map<ProjectionID,ProjectionFunctor*>::const_iterator it = 
            projection_functors.begin(); it != projection_functors.end(); it++)
      {
        delete it->second;
      }
      memory_manager_lock.destroy_reservation();
      memory_manager_lock = Reservation::NO_RESERVATION;
      message_manager_lock.destroy_reservation();
      message_manager_lock = Reservation::NO_RESERVATION;
      memory_managers.clear();
      projection_functors.clear();
      mapper_info_lock.destroy_reservation();
      mapper_info_lock = Reservation::NO_RESERVATION;
      available_lock.destroy_reservation();
      available_lock = Reservation::NO_RESERVATION;
      group_lock.destroy_reservation();
      group_lock = Reservation::NO_RESERVATION;
      distributed_id_lock.destroy_reservation();
      distributed_id_lock = Reservation::NO_RESERVATION;
      distributed_collectable_lock.destroy_reservation();
      distributed_collectable_lock = Reservation::NO_RESERVATION;
      gc_epoch_lock.destroy_reservation();
      gc_epoch_lock = Reservation::NO_RESERVATION;
      future_lock.destroy_reservation();
      future_lock = Reservation::NO_RESERVATION;
      remote_lock.destroy_reservation();
      remote_lock = Reservation::NO_RESERVATION;
      shutdown_lock.destroy_reservation();
      shutdown_lock = Reservation::NO_RESERVATION;
      for (std::deque<IndividualTask*>::const_iterator it = 
            available_individual_tasks.begin(); 
            it != available_individual_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_individual_tasks.clear();
      individual_task_lock.destroy_reservation();
      individual_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<PointTask*>::const_iterator it = 
            available_point_tasks.begin(); it != 
            available_point_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_point_tasks.clear();
      point_task_lock.destroy_reservation();
      point_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<IndexTask*>::const_iterator it = 
            available_index_tasks.begin(); it != 
            available_index_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_index_tasks.clear();
      index_task_lock.destroy_reservation();
      index_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<SliceTask*>::const_iterator it = 
            available_slice_tasks.begin(); it != 
            available_slice_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_slice_tasks.clear();
      slice_task_lock.destroy_reservation();
      slice_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<RemoteTask*>::const_iterator it = 
            available_remote_tasks.begin(); it != 
            available_remote_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_remote_tasks.clear();
      remote_task_lock.destroy_reservation();
      remote_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<InlineTask*>::const_iterator it = 
            available_inline_tasks.begin(); it !=
            available_inline_tasks.end(); it++)
      {
        legion_delete(*it);
      }
      available_inline_tasks.clear();
      inline_task_lock.destroy_reservation();
      inline_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<MapOp*>::const_iterator it = 
            available_map_ops.begin(); it != 
            available_map_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_map_ops.clear();
      map_op_lock.destroy_reservation();
      map_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<CopyOp*>::const_iterator it = 
            available_copy_ops.begin(); it != 
            available_copy_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_copy_ops.clear();
      copy_op_lock.destroy_reservation();
      copy_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FenceOp*>::const_iterator it = 
            available_fence_ops.begin(); it != 
            available_fence_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_fence_ops.clear();
      fence_op_lock.destroy_reservation();
      fence_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FrameOp*>::const_iterator it = 
            available_frame_ops.begin(); it !=
            available_frame_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_frame_ops.clear();
      frame_op_lock.destroy_reservation();
      frame_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<DeletionOp*>::const_iterator it = 
            available_deletion_ops.begin(); it != 
            available_deletion_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_deletion_ops.clear();
      deletion_op_lock.destroy_reservation();
      deletion_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<InterCloseOp*>::const_iterator it = 
            available_inter_close_ops.begin(); it !=
            available_inter_close_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_inter_close_ops.clear();
      inter_close_op_lock.destroy_reservation();
      inter_close_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<ReadCloseOp*>::const_iterator it = 
            available_read_close_ops.begin(); it != 
            available_read_close_ops.end(); it++)
      {
        legion_delete(*it);
      }
      read_close_op_lock.destroy_reservation();
      read_close_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<PostCloseOp*>::const_iterator it = 
            available_post_close_ops.begin(); it !=
            available_post_close_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_post_close_ops.clear();
      post_close_op_lock.destroy_reservation();
      post_close_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<VirtualCloseOp*>::const_iterator it = 
            available_virtual_close_ops.begin(); it !=
            available_virtual_close_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_virtual_close_ops.clear();
      virtual_close_op_lock.destroy_reservation();
      virtual_close_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<DynamicCollectiveOp*>::const_iterator it = 
            available_dynamic_collective_ops.begin(); it !=
            available_dynamic_collective_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_dynamic_collective_ops.end();
      dynamic_collective_op_lock.destroy_reservation();
      dynamic_collective_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FuturePredOp*>::const_iterator it = 
            available_future_pred_ops.begin(); it !=
            available_future_pred_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_future_pred_ops.clear();
      future_pred_op_lock.destroy_reservation();
      future_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<NotPredOp*>::const_iterator it = 
            available_not_pred_ops.begin(); it !=
            available_not_pred_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_not_pred_ops.clear();
      not_pred_op_lock.destroy_reservation();
      not_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<AndPredOp*>::const_iterator it = 
            available_and_pred_ops.begin(); it !=
            available_and_pred_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_and_pred_ops.clear();
      and_pred_op_lock.destroy_reservation();
      and_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<OrPredOp*>::const_iterator it = 
            available_or_pred_ops.begin(); it !=
            available_or_pred_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_or_pred_ops.clear();
      or_pred_op_lock.destroy_reservation();
      or_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<AcquireOp*>::const_iterator it = 
            available_acquire_ops.begin(); it !=
            available_acquire_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_acquire_ops.clear();
      acquire_op_lock.destroy_reservation();
      acquire_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<ReleaseOp*>::const_iterator it = 
            available_release_ops.begin(); it !=
            available_release_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_release_ops.clear();
      release_op_lock.destroy_reservation();
      release_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<TraceCaptureOp*>::const_iterator it = 
            available_capture_ops.begin(); it !=
            available_capture_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_capture_ops.clear();
      capture_op_lock.destroy_reservation();
      capture_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<TraceCompleteOp*>::const_iterator it = 
            available_trace_ops.begin(); it !=
            available_trace_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_trace_ops.clear();
      trace_op_lock.destroy_reservation();
      trace_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<MustEpochOp*>::const_iterator it = 
            available_epoch_ops.begin(); it !=
            available_epoch_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_epoch_ops.clear();
      epoch_op_lock.destroy_reservation();
      epoch_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<PendingPartitionOp*>::const_iterator it = 
            available_pending_partition_ops.begin(); it !=
            available_pending_partition_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_pending_partition_ops.clear();
      pending_partition_op_lock.destroy_reservation();
      pending_partition_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<DependentPartitionOp*>::const_iterator it = 
            available_dependent_partition_ops.begin(); it !=
            available_dependent_partition_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_dependent_partition_ops.clear();
      dependent_partition_op_lock.destroy_reservation();
      dependent_partition_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FillOp*>::const_iterator it = 
            available_fill_ops.begin(); it !=
            available_fill_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_fill_ops.clear();
      fill_op_lock.destroy_reservation();
      fill_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<AttachOp*>::const_iterator it = 
            available_attach_ops.begin(); it !=
            available_attach_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_attach_ops.clear();
      attach_op_lock.destroy_reservation();
      attach_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<DetachOp*>::const_iterator it = 
            available_detach_ops.begin(); it !=
            available_detach_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_detach_ops.clear();
      detach_op_lock.destroy_reservation();
      detach_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<TimingOp*>::const_iterator it = 
            available_timing_ops.begin(); it != 
            available_timing_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_timing_ops.clear();
      timing_op_lock.destroy_reservation();
      timing_op_lock = Reservation::NO_RESERVATION;
      for (std::map<TaskID,TaskImpl*>::const_iterator it = 
            task_table.begin(); it != task_table.end(); it++)
      {
        legion_delete(it->second);
      }
      task_table.clear();
      // Skip this if we are in separate runtime mode
      if (!Internal::separate_runtime_instances)
      {
        for (std::deque<VariantImpl*>::const_iterator it = 
              variant_table.begin(); it != variant_table.end(); it++)
        {
          legion_delete(*it);
        }
      }
      variant_table.clear();
      task_variant_lock.destroy_reservation();
      task_variant_lock = Reservation::NO_RESERVATION;
      // Skip this if we are in separate runtime mode
      if (!Internal::separate_runtime_instances)
      {
        for (std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
              it = layout_constraints_table.begin(); it != 
              layout_constraints_table.end(); it++)
        {
          legion_delete(it->second);
        }
      }
      layout_constraints_lock.destroy_reservation();
      layout_constraints_lock = Reservation::NO_RESERVATION;

      delete forest;

#ifdef DEBUG_HIGH_LEVEL
      if (logging_region_tree_state)
	delete tree_state_logger;
#endif
    }

    //--------------------------------------------------------------------------
    Internal& Internal::operator=(const Internal &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void Internal::construct_mpi_rank_tables(Processor proc, int rank)
    //--------------------------------------------------------------------------
    {
      // Initialize our mpi rank event
      Internal::mpi_rank_event = UserEvent::create_user_event();
      // Now broadcast our address space and rank to all the other nodes
      MPIRankArgs args;
      args.hlr_id = HLR_MPI_RANK_ID;
      args.mpi_rank = rank;
      args.source_space = address_space;
      std::set<AddressSpace> sent_targets;
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        AddressSpace target_space = it->address_space();
        if (target_space == address_space)
          continue;
        if (sent_targets.find(target_space) != sent_targets.end())
          continue;
        Processor::Kind kind = it->kind();
        if (kind != Processor::LOC_PROC)
          continue;
        issue_runtime_meta_task(&args, sizeof(args), HLR_MPI_RANK_ID, NULL,
                                Event::NO_EVENT, 0/*priority*/, 
                                false/*holds reservation*/, *it);
        sent_targets.insert(target_space);
      }
      // Now set our own value, update the count, and see if we're done
      Internal::mpi_rank_table[rank] = address_space;
      unsigned count = 
        __sync_add_and_fetch(&Internal::remaining_mpi_notifications, 1);
      const size_t total_ranks = machine.get_address_space_count();
      if (count == total_ranks)
        Internal::mpi_rank_event.trigger();
      // Wait on the event
      mpi_rank_event.wait();
      // Once we've triggered, then we can build the maps
      for (unsigned local_rank = 0; local_rank < count; local_rank++)
      {
        AddressSpace local_space = Internal::mpi_rank_table[local_rank];
#ifdef DEBUG_HIGH_LEVEL
        assert(reverse_mpi_mapping.find(local_space) == 
               reverse_mpi_mapping.end());
#endif
        forward_mpi_mapping[local_rank] = local_space;
        reverse_mpi_mapping[local_space] = local_rank;
      }
    }

    //--------------------------------------------------------------------------
    void Internal::launch_top_level_task(Processor target)
    //--------------------------------------------------------------------------
    {
      // Get an individual task to be the top-level task
      IndividualTask *top_task = get_available_individual_task(false);
      // Get a remote task to serve as the top of the top-level task
      RemoteTask *top_context = get_available_remote_task(false);
      top_context->initialize_remote(0, NULL, true/*top*/);
      // Set the executing processor
      top_context->set_executing_processor(target);
      TaskLauncher launcher(Internal::legion_main_id, TaskArgument());
      // Mark that this task is the top-level task
      top_task->set_top_level();
      top_task->initialize_task(top_context, launcher, 
                                false/*check priv*/, false/*track parent*/);
      // Set up the input arguments
      top_task->arglen = sizeof(InputArgs);
      top_task->args = malloc(top_task->arglen);
      top_task->depth = 0;
      memcpy(top_task->args,&Internal::get_input_args(),top_task->arglen);
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(target) != proc_managers.end());
#endif
      proc_managers[target]->invoke_mapper_set_task_options(top_task);
      invoke_mapper_configure_context(target, top_task);
      if (legion_spy_enabled)
      {
        Internal::log_machine(machine);
        LegionSpy::log_top_level_task(Internal::legion_main_id,
                                      top_task->get_unique_task_id(),
                                      top_task->variants->name);
      }
      increment_outstanding_top_level_tasks();
      // Put the task in the ready queue
      add_to_ready_queue(target, top_task, false/*prev failure*/);
    }

    //--------------------------------------------------------------------------
    Event Internal::launch_mapper_task(Mapper *mapper, Processor proc, 
                                      Processor::TaskFuncID tid,
                                      const TaskArgument &arg, MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Get an individual task to be the top-level task
      IndividualTask *mapper_task = get_available_individual_task(false);
      // Get a remote task to serve as the top of the top-level task
      RemoteTask *map_context = get_available_remote_task(false);
      map_context->initialize_remote(0, NULL, true/*top*/);
      map_context->set_executing_processor(proc);
      TaskLauncher launcher(tid, arg, Predicate::TRUE_PRED, map_id);
      Future f = mapper_task->initialize_task(map_context, launcher, 
                                   false/*check priv*/, false/*track parent*/);
      mapper_task->depth = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_set_task_options(mapper_task);
      invoke_mapper_configure_context(proc, mapper_task);
      // Create a temporary event to name the result since we 
      // have to pack it in the task that runs, but it also depends
      // on the task being reported back to the mapper
      UserEvent result = UserEvent::create_user_event();
      // Add a reference to the future impl to prevent it being collected
      f.impl->add_base_gc_ref(FUTURE_HANDLE_REF);
      // Create a meta-task to return the results to the mapper
      MapperTaskArgs args;
      args.hlr_id = HLR_MAPPER_TASK_ID;
      args.future = f.impl;
      args.map_id = map_id;
      args.proc = proc;
      args.event = result;
      args.context = map_context;
      Event pre = f.impl->get_ready_event();
      Event post = issue_runtime_meta_task(&args, sizeof(args), 
                                           HLR_MAPPER_TASK_ID, NULL, pre);
      // Chain the events properly
      result.trigger(post);
      // Mark that we have another outstanding top level task
      increment_outstanding_top_level_tasks();
      // Now we can put it on the queue
      add_to_ready_queue(proc, mapper_task, false/*prev failure*/);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space(Context ctx, size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(get_unique_index_space_id(),get_unique_index_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space %x in task %s "
                            "(ID %lld) with %ld maximum elements", handle.id, 
                            ctx->variants->name, ctx->get_unique_task_id(), 
                            max_num_elmts); 
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index space creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      Realm::IndexSpace space = 
                      Realm::IndexSpace::create_index_space(max_num_elmts);
      forest->create_index_space(handle, Domain(space), 
                                 UNSTRUCTURED_KIND, MUTABLE);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(domain.exists());
#endif
      IndexSpace handle(get_unique_index_space_id(),get_unique_index_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating dummy index space %x in task %s "
                            "(ID %lld) for domain", 
                            handle.id, ctx->variants->name,
                            ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index space creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      forest->create_index_space(handle, domain, DENSE_ARRAY_KIND, NO_MEMORY);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space(Context ctx, 
                                           const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(get_unique_index_space_id(),get_unique_index_tree_id());
      // First compute the convex hull of all the domains
      Domain hull = *(domains.begin());
#ifdef DEBUG_HIGH_LEVEL
      assert(!domains.empty());
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index space creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (hull.get_dim() == 0)
      {
        log_index.error("Create index space with multiple domains "
                              "must be created with domains for non-zero "
                              "dimension in task %s (ID %lld)",
                              ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_DOMAIN_DIM_MISMATCH);
      }
      for (std::set<Domain>::const_iterator it = domains.begin();
            it != domains.end(); it++)
      {
        assert(it->exists());
        if (hull.get_dim() != it->get_dim())
        {
          log_index.error("A set of domains passed to create_index_space "
                                "must all have the same dimensions in task "
                                "%s (ID %lld)",
                                ctx->variants->name, ctx->get_unique_task_id());
          assert(false);
          exit(ERROR_DOMAIN_DIM_MISMATCH);
        }
      }
#endif
      switch (hull.get_dim())
      {
        case 1:
          {
            Rect<1> base = hull.get_rect<1>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              Rect<1> next = it->get_rect<1>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<1>(base);
            break;
          }
        case 2:
          {
            Rect<2> base = hull.get_rect<2>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              Rect<2> next = it->get_rect<2>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<2>(base);
            break;
          }
        case 3:
          {
            Rect<3> base = hull.get_rect<3>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              Rect<3> next = it->get_rect<3>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<3>(base);
            break;
          }
        default:
          assert(false);
      }
#ifdef DEBUG_HIGH_LEVEL
      log_index.debug("Creating dummy index space %x in task %s "
                            "(ID %lld) for domain", 
                            handle.id, ctx->variants->name,
                            ctx->get_unique_task_id());
#endif
      if (legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      forest->create_index_space(handle, hull, domains,
                                 DENSE_ARRAY_KIND, NO_MEMORY);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Destroying index space %x in task %s "
                             "(ID %lld)", 
                      handle.id, ctx->variants->name, 
                      ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index space deletion performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }

#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_index_space_deletion(ctx, handle);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool Internal::finalize_index_space_destroy(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->destroy_index_space(handle, address_space);
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Domain &color_space,
                                          const PointColoring &coloring,
                                          PartitionKind part_kind,
                                          int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context finalize index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      std::map<DomainPoint,Domain> new_index_spaces; 
      Domain parent_dom = forest->get_index_space_domain(parent);
      const size_t num_elmts = 
        parent_dom.get_index_space().get_valid_mask().get_num_elmts();
      const int first_element =
        parent_dom.get_index_space().get_valid_mask().get_first_element();
      for (std::map<DomainPoint,ColoredPoints<ptr_t> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        Realm::ElementMask child_mask(num_elmts, first_element);
        const ColoredPoints<ptr_t> &pcoloring = it->second;
        for (std::set<ptr_t>::const_iterator pit = pcoloring.points.begin();
              pit != pcoloring.points.end(); pit++)
        {
          child_mask.enable(pit->value,1);
        }
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator pit = 
              pcoloring.ranges.begin(); pit != pcoloring.ranges.end(); pit++)
        {
          if (pit->second.value >= pit->first.value)
            child_mask.enable(pit->first.value,
                (size_t)(pit->second.value - pit->first.value) + 1);
        }
        Realm::IndexSpace child_space = 
          Realm::IndexSpace::create_index_space(
                          parent_dom.get_index_space(), child_mask, allocable);
        new_index_spaces[it->first] = Domain(child_space);
      }
#ifdef DEBUG_HIGH_LEVEL
      if ((part_kind == DISJOINT_KIND) && verify_disjointness)
        validate_unstructured_disjointness(pid, new_index_spaces);
#endif
      ColorPoint partition_color;
      // If we have a valid color, set it now
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_index_spaces, color_space, part_kind, 
                                     allocable ? MUTABLE : NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Coloring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context finalize index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no "
                            "colors in task %s (ID %lld). Index partitions "
                            "must have at least one color.",
                            ctx->variants->name, ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_EMPTY_INDEX_PARTITION);
      }
      Point<1> lower_bound(coloring.begin()->first);
      Point<1> upper_bound(coloring.rbegin()->first);
      Rect<1> color_range(lower_bound,upper_bound);
      Domain color_space = Domain::from_rect<1>(color_range);
      // Perform the coloring by iterating over all the colors in the
      // range.  For unspecified colors there is nothing wrong with
      // making empty index spaces.  We do this so we can save the
      // color space as a dense 1D domain.
      std::map<DomainPoint,Domain> new_index_spaces; 
      Domain parent_dom = forest->get_index_space_domain(parent);
      const size_t num_elmts = 
        parent_dom.get_index_space().get_valid_mask().get_num_elmts();
      const int first_element =
        parent_dom.get_index_space().get_valid_mask().get_first_element();
      for (GenericPointInRectIterator<1> pir(color_range); pir; pir++)
      {
        Realm::ElementMask child_mask(num_elmts, first_element);
        Color c = pir.p;
        std::map<Color,ColoredPoints<ptr_t> >::const_iterator finder = 
          coloring.find(c);
        // If we had a coloring provided, then fill in all the elements
        if (finder != coloring.end())
        {
          const ColoredPoints<ptr_t> &pcoloring = finder->second;
          for (std::set<ptr_t>::const_iterator it = pcoloring.points.begin();
                it != pcoloring.points.end(); it++)
          {
            child_mask.enable(it->value,1);
          }
          for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
                pcoloring.ranges.begin(); it != pcoloring.ranges.end(); it++)
          {
            if (it->second.value >= it->first.value)
              child_mask.enable(it->first.value,
                  (size_t)(it->second.value - it->first.value) + 1);
          }
        }
        else
          continue;
        // Now make the index space and save the information
#ifdef ASSUME_UNALLOCABLE
        Realm::IndexSpace child_space = 
          Realm::IndexSpace::create_index_space(
              parent_dom.get_index_space(), child_mask, false/*allocable*/);
#else
        Realm::IndexSpace child_space = 
          Realm::IndexSpace::create_index_space(
                          parent_dom.get_index_space(), child_mask);
#endif
        new_index_spaces[DomainPoint::from_point<1>(
            Arrays::Point<1>(finder->first))] = Domain(child_space);
      }
#if 0
      // Now check for completeness
      bool complete = true;
      {
        IndexIterator iterator(parent);
        while (iterator.has_next())
        {
          ptr_t ptr = iterator.next();
          bool found = false;
          for (std::map<Color,ColoredPoints<ptr_t> >::const_iterator cit =
                coloring.begin(); (cit != coloring.end()) && !found; cit++)
          {
            const ColoredPoints<ptr_t> &pcoloring = cit->second; 
            if (pcoloring.points.find(ptr) != pcoloring.points.end())
            {
              found = true;
              break;
            }
            for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
                  pcoloring.ranges.begin(); it != pcoloring.ranges.end(); it++)
            {
              if ((it->first.value <= ptr.value) && 
                  (ptr.value <= it->second.value))
              {
                found = true;
                break;
              }
            }
          }
          if (!found)
          {
            complete = false;
            break;
          }
        }
      }
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (disjoint && verify_disjointness)
        validate_unstructured_disjointness(pid, new_index_spaces);
#endif 
      ColorPoint partition_color;
      // If we have a valid color, set it now
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_index_spaces, color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     MUTABLE);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Domain &color_space,
                                          const DomainPointColoring &coloring,
                                          PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if ((part_kind == DISJOINT_KIND) && verify_disjointness)
        validate_structured_disjointness(pid, coloring);
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     coloring, color_space, 
                                     part_kind, NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const DomainColoring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no "
                            "colors in task %s (ID %lld). Index partitions "
                            "must have at least one color.",
                            ctx->variants->name, ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_EMPTY_INDEX_PARTITION);
      }
      ColorPoint partition_color;
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      std::map<DomainPoint,Domain> new_subspaces;
      for (std::map<Color,Domain>::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        new_subspaces[DomainPoint::from_point<1>(
            Arrays::Point<1>(it->first))] = it->second;
      }
#ifdef DEBUG_HIGH_LEVEL
      if (disjoint && verify_disjointness)
        validate_structured_disjointness(pid, new_subspaces);
#endif
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_subspaces, color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Domain &color_space,
                                       const MultiDomainPointColoring &coloring,
                                          PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // Build all the convex hulls
      std::map<DomainPoint,Domain> convex_hulls;
      for (std::map<DomainPoint,std::set<Domain> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        Domain hull = construct_convex_hull(it->second);
        convex_hulls[it->first] = hull;
      }
#ifdef DEBUG_HIGH_LEVEL
      if ((part_kind == DISJOINT_KIND) && verify_disjointness)
        validate_multi_structured_disjointness(pid, coloring);
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     convex_hulls, coloring,
                                     color_space, part_kind, NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const MultiDomainColoring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no "
                            "colors in task %s (ID %lld). Index partitions "
                            "must have at least one color.",
                            ctx->variants->name, ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_EMPTY_INDEX_PARTITION);
      }
      // TODO: Construct the validity of all the domains in the set
      // Build all the convex hulls
      std::map<DomainPoint,Domain> convex_hulls;
      std::map<DomainPoint,std::set<Domain> > color_sets;
      for (std::map<Color,std::set<Domain> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        Domain hull = construct_convex_hull(it->second);
        DomainPoint color = DomainPoint::from_point<1>(Point<1>(it->first));
        convex_hulls[color] = hull;
        color_sets[color] = it->second; 
      }
#ifdef DEBUG_HIGH_LEVEL
      if (disjoint && verify_disjointness)
        validate_multi_structured_disjointness(pid, color_sets);
#endif
      ColorPoint partition_color;
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     convex_hulls, color_sets,
                                     color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_index_partition(
                                          Context ctx, IndexSpace parent,
       Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // Perform the coloring
      std::map<DomainPoint,Domain> new_index_spaces;
      Domain color_space;
      // Iterate over the parent index space and make the sub-index spaces
      // for each of the different points in the space
      Accessor::RegionAccessor<Accessor::AccessorType::Generic,int> 
        fa_coloring = field_accessor.typeify<int>();
      {
        std::map<Color,Realm::ElementMask> child_masks;
        Domain parent_dom = forest->get_index_space_domain(parent);
        size_t parent_elmts = 
          parent_dom.get_index_space().get_valid_mask().get_num_elmts();
	const int parent_first_element =
	  parent_dom.get_index_space().get_valid_mask().get_first_element();
        for (Domain::DomainPointIterator itr(parent_dom); itr; itr++)
        {
          ptr_t cur_ptr = itr.p.get_index();
          int c = fa_coloring.read(cur_ptr);
          // Ignore all colors less than zero
          if (c >= 0)
          {
            Color color = (Color)c; 
            std::map<Color,Realm::ElementMask>::iterator finder = 
              child_masks.find(color);
            // Haven't made an index space for this color yet
            if (finder == child_masks.end())
            {
              child_masks[color] = Realm::ElementMask(parent_elmts,
						      parent_first_element);
              finder = child_masks.find(color);
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != child_masks.end());
#endif
            finder->second.enable(cur_ptr.value);
          }
        }
        // Now make the index spaces and their domains
        Point<1> lower_bound(child_masks.begin()->first);
        Point<1> upper_bound(child_masks.rbegin()->first);
        Rect<1> color_range(lower_bound,upper_bound);
        color_space = Domain::from_rect<1>(color_range);
        // Iterate over all the colors in the range from the lower
        // bound to upper bound so we can store the color space as
        // a dense array of colors.
        for (GenericPointInRectIterator<1> pir(color_range); pir; pir++)
        {
          Color c = pir.p;
          std::map<Color,Realm::ElementMask>::const_iterator finder = 
            child_masks.find(c);
          Realm::IndexSpace child_space;
          if (finder != child_masks.end())
          {
#ifdef ASSUME_UNALLOCABLE
            child_space = 
              Realm::IndexSpace::create_index_space(
                parent_dom.get_index_space(), finder->second, false);
#else
            child_space = 
              Realm::IndexSpace::create_index_space(
                    parent_dom.get_index_space(), finder->second);
#endif
          }
          else
          {
            Realm::ElementMask empty_mask;
#ifdef ASSUME_UNALLOCABLE
            child_space = 
              Realm::IndexSpace::create_index_space(
                    parent_dom.get_index_space(), empty_mask, false);
#else
            child_space = 
              Realm::IndexSpace::create_index_space(
                    parent_dom.get_index_space(), empty_mask);
#endif
          }
          new_index_spaces[DomainPoint::from_point<1>(
              Arrays::Point<1>(c))] = Domain(child_space);
        }
      }
      ColorPoint partition_color;
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color,
                                     new_index_spaces, color_space,
                                     DISJOINT_KIND, MUTABLE);
      return pid;
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_index_partition(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy index partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Destroying index partition %x in task %s "
                             "(ID %lld)", 
                    handle.id, ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal index partition deletion performed in "
                             "leaf task %s (ID %lld)",
                              ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_index_part_deletion(ctx, handle);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_index_partition_destroy(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_index_partition(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Internal::validate_unstructured_disjointness(IndexPartition pid,
                                    const std::map<DomainPoint,Domain> &domains)
    //--------------------------------------------------------------------------
    {
      std::set<DomainPoint> current_colors;
      for (std::map<DomainPoint,Domain>::const_iterator it1 = 
            domains.begin(); it1 != domains.end(); it1++)
      {
        current_colors.insert(it1->first);
        for (std::map<DomainPoint,Domain>::const_iterator it2 = 
              domains.begin(); it2 != domains.end(); it2++)
        {
          // Skip pairs that we already checked
          if (current_colors.find(it2->first) != current_colors.end())
            continue;
          // Otherwise perform the check
          const Realm::ElementMask &em1 = 
            it1->second.get_index_space().get_valid_mask();
          const Realm::ElementMask &em2 = 
            it2->second.get_index_space().get_valid_mask();
          Realm::ElementMask::OverlapResult result = 
            em1.overlaps_with(em2, 1/*effort level*/);
          if (result == Realm::ElementMask::OVERLAP_YES)
          {
            log_run.error("ERROR: colors %d and %d of partition %d "
                          "are not disjoint when they were claimed to be!",
                          (int)it1->first.get_index(),
                          (int)it2->first.get_index(), pid.id);
            assert(false);
            exit(ERROR_DISJOINTNESS_TEST_FAILURE);
          }
          else if (result == Realm::ElementMask::OVERLAP_MAYBE)
          {
            log_run.warning("WARNING: colors %d and %d of partition "
                          "%d may not be disjoint when they were claimed to be!"
                           "(At least according to the low-level runtime.  You "
                            "might also try telling the the low-level runtime "
                            "to stop being lazy and try harder.)",
                            (int)it1->first.get_index(),
                            (int)it2->first.get_index(), pid.id);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::validate_structured_disjointness(IndexPartition pid,
                                    const std::map<DomainPoint,Domain> &domains)
    //--------------------------------------------------------------------------
    {
      std::set<DomainPoint> current_colors;
      for (std::map<DomainPoint,Domain>::const_iterator it1 = 
            domains.begin(); it1 != domains.end(); it1++)
      {
        current_colors.insert(it1->first);
        for (std::map<DomainPoint,Domain>::const_iterator it2 = 
              domains.begin(); it2 != domains.end(); it2++)
        {
          if (current_colors.find(it2->first) != current_colors.end())
            continue;
          assert(it1->second.get_dim() == it2->second.get_dim());
          switch (it1->second.get_dim())
          {
            case 1:
              {
                Rect<1> d1 = it1->second.get_rect<1>();
                Rect<1> d2 = it2->second.get_rect<1>();
                if (d1.overlaps(d2))
                {
                  log_run.error("ERROR: colors %d and %d of "
                                       "partition %d are not disjoint "
                                       "when they are claimed to be!",
                                (int)it1->first[0], (int)it2->first[0], pid.id);
                  assert(false);
                  exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                }
                break;
              }
            case 2:
              {
                Rect<2> d1 = it1->second.get_rect<2>();
                Rect<2> d2 = it2->second.get_rect<2>();
                if (d1.overlaps(d2))
                {
                  log_run.error("ERROR: colors (%d,%d) and "
                                      "(%d,%d) of partition %d are "
                                      "not disjoint when they are "
                                      "claimed to be!",
                            (int)it1->first[0], (int)it1->first[1],
                            (int)it2->first[0], (int)it2->first[1], pid.id);
                  assert(false);
                  exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                }
                break;
              }
            case 3:
              {
                Rect<3> d1 = it1->second.get_rect<3>();
                Rect<3> d2 = it2->second.get_rect<3>();
                if (d1.overlaps(d2))
                {
                  log_run.error("ERROR: colors (%d,%d,%d) and "
                                       "(%d,%d,%d) of partition %d are "
                                       "not disjoint when they are "
                                       "claimed to be!",
                                       (int)it1->first[0],
                                       (int)it1->first[1],
                                       (int)it1->first[2],
                                       (int)it2->first[0],
                                       (int)it2->first[1],
                                       (int)it2->first[2], pid.id);
                  assert(false);
                  exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                }
                break;
              }
            default:
              assert(false); // should never get here
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::validate_multi_structured_disjointness(IndexPartition pid,
                         const std::map<DomainPoint,std::set<Domain> > &domains)
    //--------------------------------------------------------------------------
    {
      std::set<DomainPoint> current_colors;
      for (std::map<DomainPoint,std::set<Domain> >::const_iterator it1 = 
            domains.begin(); it1 != domains.end(); it1++)
      {
        current_colors.insert(it1->first);
        for (std::map<DomainPoint,std::set<Domain> >::const_iterator it2 = 
              domains.begin(); it2 != domains.end(); it2++)
        {
          if (current_colors.find(it2->first) != current_colors.end())
            continue;
          for (std::set<Domain>::const_iterator it3 = it1->second.begin();
                it3 != it1->second.end(); it3++)
          {
            for (std::set<Domain>::const_iterator it4 = it2->second.begin();
                  it4 != it2->second.end(); it4++)
            {
              assert(it3->get_dim() == it4->get_dim());
              switch (it3->get_dim())
              {
                case 1:
                  {
                    Rect<1> d1 = it3->get_rect<1>();
                    Rect<1> d2 = it4->get_rect<1>();
                    if (d1.overlaps(d2))
                    {
                      log_run.error("ERROR: colors %d and %d of "
                                           "multi-domain partition %d are "
                                           "not disjoint when they are "
                                           "claimed to be!", 
                                           (int)it1->first[0],
                                           (int)it2->first[0], pid.id);
                      assert(false);
                      exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                    }
                    break;
                  }
                case 2:
                  {
                    Rect<2> d1 = it3->get_rect<2>();
                    Rect<2> d2 = it4->get_rect<2>();
                    if (d1.overlaps(d2))
                    {
                      log_run.error("ERROR: colors (%d,%d) and (%d,%d) "
                                           "of multi-domain partition %d are "
                                           "not disjoint when they are "
                                           "claimed to be!", 
                                           (int)it1->first[0],
                                           (int)it1->first[1],
                                           (int)it2->first[0],
                                           (int)it2->first[1], pid.id);
                      assert(false);
                      exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                    }
                    break;
                  }
                case 3:
                  {
                    Rect<3> d1 = it3->get_rect<3>();
                    Rect<3> d2 = it4->get_rect<3>();
                    if (d1.overlaps(d2))
                    {
                      log_run.error("ERROR: colors (%d,%d,%d) and "
                                           "(%d,%d,%d) of multi-domain "
                                           "partition %d are not disjoint "
                                           "when they are claimed to be!", 
                                           (int)it1->first[0],
                                           (int)it1->first[1],
                                           (int)it1->first[2],
                                           (int)it2->first[0],
                                           (int)it2->first[1],
                                           (int)it2->first[2], pid.id);
                      assert(false);
                      exit(ERROR_DISJOINTNESS_TEST_FAILURE);
                    }
                    break;
                  }
                default:
                  assert(false);
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    Domain Internal::construct_convex_hull(const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      Domain hull = *(domains.begin());
      switch (hull.get_dim())
      {
        case 1:
          {
            Rect<1> base = hull.get_rect<1>();
            for (std::set<Domain>::const_iterator dom_it =
                  domains.begin(); dom_it != domains.end(); dom_it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(dom_it->get_dim() == 1);
#endif
              Rect<1> next = dom_it->get_rect<1>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<1>(base);
            break;
          }
        case 2:
          {
            Rect<2> base = hull.get_rect<2>();
            for (std::set<Domain>::const_iterator dom_it =
                  domains.begin(); dom_it != domains.end(); dom_it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(dom_it->get_dim() == 2);
#endif
              Rect<2> next = dom_it->get_rect<2>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<2>(base);
            break;
          }
        case 3:
          {
            Rect<3> base = hull.get_rect<3>();
            for (std::set<Domain>::const_iterator dom_it =
                  domains.begin(); dom_it != domains.end(); dom_it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(dom_it->get_dim() == 3);
#endif
              Rect<3> next = dom_it->get_rect<3>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<3>(base);
            break;
          }
        default:
          assert(false);
      }
      return hull;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_equal_partition(Context ctx, 
                                                   IndexSpace parent,
                                                   const Domain &color_space,
                                                   size_t granularity,
                                                   int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create equal partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating equal partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal equal partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_equal_partition(ctx, pid, granularity);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space,
                                       partition_color, DISJOINT_KIND,
                                       allocable, handle_ready, term_event);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid; 
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_weighted_partition(Context ctx, 
                                                      IndexSpace parent,
                                                      const Domain &color_space,
                                       const std::map<DomainPoint,int> &weights,
                                                      size_t granularity,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create weighted partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating weighted partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal weighted partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_weighted_partition(ctx, pid, granularity, weights);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, DISJOINT_KIND,
                                       allocable, handle_ready, term_event);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_union(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      PartitionKind kind,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create partition by union!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating union partition %d with parent index "
                            "space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal union partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by union!",
                              handle1.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by union!",
                              handle2.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          Realm::IndexSpace::ISO_UNION);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_union_partition(ctx, pid, handle1, handle2);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_intersection(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      PartitionKind kind,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create partition "
                            "by intersection!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating intersection partition %d with parent "
                            "index space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal intersection partition creation "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by intersection!",
                              handle1.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by intersection!",
                              handle2.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          Realm::IndexSpace::ISO_INTERSECT);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_intersection_partition(ctx, pid, handle1, handle2);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_difference(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      PartitionKind kind,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create difference "
                            "partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating difference partition %d with parent "
                            "index space %x in task %s (ID %lld)", 
                            pid.id, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal difference partition creation "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by difference!",
                              handle1.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by difference!",
                              handle2.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          Realm::IndexSpace::ISO_SUBTRACT);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_difference_partition(ctx, pid, handle1, handle2);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    void Internal::create_cross_product_partition(Context ctx,
                                                 IndexPartition handle1,
                                                 IndexPartition handle2,
                                  std::map<DomainPoint,IndexPartition> &handles,
                                                 PartitionKind kind,
                                                 int color, bool allocable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create cross product "
                            "partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating cross product partitions "
                            "in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create cross product partitions "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (handle1.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                              "index tree as IndexPartition %d in create "
                              "cross product partitions!",
                              handle1.id, handle2.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      Event handle_ready = part_op->get_handle_ready();
      Event term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      std::map<DomainPoint,IndexPartition> local;
      forest->create_pending_cross_product(handle1, handle2, local, handles,
                                           kind, partition_color, allocable,
                                           handle_ready, term_event);
      part_op->initialize_cross_product(ctx, handle1, handle2, local);
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_field(Context ctx,
                                                      LogicalRegion handle,
                                                      LogicalRegion parent_priv,
                                                      FieldID fid,
                                                      const Domain &color_space,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexSpace parent = handle.get_index_space();
      IndexPartition pid(get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context partition by field!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating partition by field "
                            "in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal partition by field "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        get_available_dependent_partition_op(true);
      part_op->initialize_by_field(ctx, pid, handle, 
                                   parent_priv, color_space, fid);
      Event term_event = part_op->get_completion_event();
      Event handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition 
      forest->create_pending_partition(pid, parent, color_space, part_color,
                                       DISJOINT_KIND, allocable, 
                                       handle_ready, term_event); 
      Processor proc = ctx->get_executing_processor();
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      add_to_dependence_queue(proc, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_image(Context ctx,
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), handle.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context partition by image!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating partition by image "
                            "in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal partition by image "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        get_available_dependent_partition_op(true);
      part_op->initialize_by_image(ctx, pid, projection,
                                   parent, fid, color_space);
      Event term_event = part_op->get_completion_event();
      Event handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, handle, color_space, part_color,
                                       part_kind, allocable, 
                                       handle_ready, term_event); 
      Processor proc = ctx->get_executing_processor();
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      add_to_dependence_queue(proc, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_partition_by_preimage(Context ctx,
                                                    IndexPartition projection,
                                                    LogicalRegion handle,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context partition by preimage!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating partition by preimage "
                            "in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal partition by preimage "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        get_available_dependent_partition_op(true);
      part_op->initialize_by_preimage(ctx, pid, projection, handle,
                                      parent, fid, color_space);
      Event term_event = part_op->get_completion_event();
      Event handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, handle.get_index_space(), 
                                       color_space, part_color, part_kind,
                                       allocable, handle_ready, term_event);
      Processor proc = ctx->get_executing_processor();
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      add_to_dependence_queue(proc, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::create_pending_partition(Context ctx, 
                                                     IndexSpace parent, 
                                                     const Domain &color_space,
                                                     PartitionKind part_kind,
                                                     int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid(get_unique_index_partition_id(), parent.get_tree_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create pending partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating pending partition in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create pending partition "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      forest->create_pending_partition(pid, parent, color_space, part_color,
                                       part_kind, allocable, Event::NO_EVENT,
                                       Event::NO_EVENT, true/*separate*/);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space_union(Context ctx, 
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index space union!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index space union "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      UserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_index_space_union(ctx, result, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(part_op->get_handle_ready(), 
                                      handle_ready);
      LegionSpy::log_event_dependence(part_op->get_completion_event(), 
                                      domain_ready);
#endif
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
#ifdef INORDER_EXECUTION
      Event term_event = part_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space_union(Context ctx,
                                                 IndexPartition parent,
                                                 const DomainPoint &color,
                                                 IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index space union!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index space union "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      UserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_index_space_union(ctx, result, handle);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(part_op->get_handle_ready(), 
                                      handle_ready);
      LegionSpy::log_event_dependence(part_op->get_completion_event(), 
                                      domain_ready);
#endif
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
#ifdef INORDER_EXECUTION
      Event term_event = part_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space_intersection(Context ctx,
                                                        IndexPartition parent,
                                                       const DomainPoint &color,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index "
                            "space intersection!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space intersection in task %s "
                            "(ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index space intersection"
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      UserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_index_space_intersection(ctx, result, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(part_op->get_handle_ready(), 
                                      handle_ready);
      LegionSpy::log_event_dependence(part_op->get_completion_event(), 
                                      domain_ready);
#endif
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
#ifdef INORDER_EXECUTION
      Event term_event = part_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space_intersection(Context ctx,
                                                        IndexPartition parent,
                                                       const DomainPoint &color,
                                                        IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index "
                            "space intersection!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space intersection in task %s "
                            "(ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index space intersection "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      UserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_index_space_intersection(ctx, result, handle);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(part_op->get_handle_ready(), 
                                      handle_ready);
      LegionSpy::log_event_dependence(part_op->get_completion_event(), 
                                      domain_ready);
#endif
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
#ifdef INORDER_EXECUTION
      Event term_event = part_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::create_index_space_difference(Context ctx,
                                                      IndexPartition parent,
                                                      const DomainPoint &color,
                                                      IndexSpace initial,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index "
                            "space difference!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_index.debug("Creating index space difference in task %s "
                            "(ID %lld)", 
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index space difference "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      UserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = get_available_pending_partition_op(true);
      part_op->initialize_index_space_difference(ctx, result, initial, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(part_op->get_handle_ready(), 
                                      handle_ready);
      LegionSpy::log_event_dependence(part_op->get_completion_event(), 
                                      domain_ready);
#endif
      // Now we can add the operation to the queue
      Processor proc = ctx->get_executing_processor();
#ifdef INORDER_EXECUTION
      Event term_event = part_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, part_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return get_index_partition(parent, color);
      
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = forest->get_index_partition(parent, 
                                                          ColorPoint(color));
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid color %d for get index partitions", 
                                color);
        assert(false);
        exit(ERROR_INVALID_INDEX_SPACE_COLOR);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_index_partition(Context ctx,
                                    IndexSpace parent, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return get_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_index_partition(IndexSpace parent,
                                                const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = forest->get_index_partition(parent, 
                                                          ColorPoint(color));
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        switch (color.get_dim())
        {
          case 0:
          case 1:
            log_index.error("Invalid color %d for get index partitions", 
                            (int)color.point_data[0]);
            break;
          case 2:
            log_index.error("Invalid color (%d,%d) for get index partitions", 
                            (int)color.point_data[0], (int)color.point_data[1]);
            break;
          case 3:
            log_index.error("Invalid color (%d,%d,%d) for get index "
                            "partitions", (int)color.point_data[0], 
                            (int)color.point_data[1], (int)color.point_data[2]);
            break;
        }
        assert(false);
        exit(ERROR_INVALID_INDEX_SPACE_COLOR);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    bool Internal::has_index_partition(Context ctx, IndexSpace parent,
                                      const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return has_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_index_partition(IndexSpace parent,
                                      const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = forest->get_index_partition(parent, 
                                                          ColorPoint(color));
      return result.exists();
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      return get_index_subspace(p, color); 
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_index_subspace(IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpace result = forest->get_index_subspace(p, ColorPoint(color));
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid color %d for get index subspace", color);
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_COLOR); 
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_index_subspace(Context ctx, IndexPartition p, 
                                           const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return get_index_subspace(p, color); 
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_index_subspace(IndexPartition p, 
                                           const DomainPoint &color) 
    //--------------------------------------------------------------------------
    {
      IndexSpace result = forest->get_index_subspace(p, ColorPoint(color));
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        switch (color.get_dim())
        {
          case 0:
          case 1:
            log_index.error("Invalid color %d for get index subspace",
                            (int)color.point_data[0]);
            break;
          case 2:
            log_index.error("Invalid color (%d,%d) for get index subspace",
                            (int)color.point_data[0], (int)color.point_data[1]);
            break;
          case 3:
            log_index.error("Invalid color (%d,%d,%d) for get index subspace",
                            (int)color.point_data[0],
                            (int)color.point_data[1],
                            (int)color.point_data[2]);
            break;
        }
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_COLOR); 
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    bool Internal::has_index_subspace(Context ctx, IndexPartition p,
                                     const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return has_index_subspace(p, color); 
    }

    //--------------------------------------------------------------------------
    bool Internal::has_index_subspace(IndexPartition p, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      IndexSpace result = forest->get_index_subspace(p, ColorPoint(color));
      return result.exists();
    }

    //--------------------------------------------------------------------------
    bool Internal::has_multiple_domains(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_multiple_domains(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain Internal::get_index_space_domain(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    Domain Internal::get_index_space_domain(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      Domain result = forest->get_index_space_domain(handle);
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid handle %x for get index space "
                               "domain", 
                                handle.id);
        assert(false);
        exit(ERROR_INVALID_INDEX_DOMAIN);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_domains(Context ctx, IndexSpace handle,
                                          std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_domains(IndexSpace handle,
                                          std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Internal::get_index_partition_color_space(Context ctx, 
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return get_index_partition_color_space(p); 
    }

    //--------------------------------------------------------------------------
    Domain Internal::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      Domain result = forest->get_index_partition_color_space(p);
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid partition handle %d for get index "
                               "partition color space", p.id);
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_DOMAIN);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      return get_index_space_partition_colors(sp, colors);
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_partition_colors(IndexSpace handle,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      std::set<ColorPoint> color_points;
      forest->get_index_space_partition_colors(handle, color_points);
      for (std::set<ColorPoint>::const_iterator it = color_points.begin();
            it != color_points.end(); it++)
      {
        colors.insert(it->get_index());
      }
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                  std::set<DomainPoint> &colors)
    //--------------------------------------------------------------------------
    {
      get_index_space_partition_colors(sp, colors);
    }

    //--------------------------------------------------------------------------
    void Internal::get_index_space_partition_colors(IndexSpace sp,
                                                  std::set<DomainPoint> &colors)
    //--------------------------------------------------------------------------
    {
      std::set<ColorPoint> color_points;
      forest->get_index_space_partition_colors(sp, color_points);
      for (std::set<ColorPoint>::const_iterator it = color_points.begin();
            it != color_points.end(); it++)
      {
        colors.insert(it->get_point());
      }
    }

    //--------------------------------------------------------------------------
    bool Internal::is_index_partition_disjoint(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool Internal::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool Internal::is_index_partition_complete(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_complete(p);
    }

    bool Internal::is_index_partition_complete(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_complete(p);
    }

    //--------------------------------------------------------------------------
    Color Internal::get_index_space_color(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_index_space_color(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_index_space_color_point(Context ctx, 
                                                     IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_index_space_color_point(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_index_partition_color(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_index_partition_color_point(Context ctx,
                                                         IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_index_partition_color_point(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_parent_index_space(Context ctx,   
                                               IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Internal::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_parent_index_partition(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_parent_index_partition(Context ctx,
                                                       IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Internal::get_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    ptr_t Internal::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (pointer.is_null())
        return pointer;
      return ctx->perform_safe_cast(region.get_index_space(), pointer);
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (point.is_null())
        return point;
      return ctx->perform_safe_cast(region.get_index_space(), point);
    }

    //--------------------------------------------------------------------------
    FieldSpace Internal::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      FieldSpace space(get_unique_field_space_id());
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create field space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_field.debug("Creating field space %x in task %s (ID %lld)", 
                      space.id, ctx->variants->name,ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create field space performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (legion_spy_enabled)
        LegionSpy::log_field_space(space.id);

      forest->create_field_space(space);
      ctx->register_field_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_field_space(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy field space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_field.debug("Destroying field space %x in task %s (ID %lld)", 
                    handle.id, ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal destroy field space performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_field_space_deletion(ctx, handle);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    size_t Internal::get_field_size(Context ctx, FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return forest->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    size_t Internal::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return forest->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Internal::get_field_space_fields(Context ctx, FieldSpace handle,
                                          std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      forest->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    void Internal::get_field_space_fields(FieldSpace handle, 
                                          std::set<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      forest->get_field_space_fields(handle, fields);
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_field_space_destroy(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_field_space(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_field_destroy(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      forest->free_field(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_field_destroy(FieldSpace handle, 
                                               const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> dense(to_free.begin(), to_free.end());
      forest->free_fields(handle, dense);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::create_logical_region(Context ctx, 
                                IndexSpace index_space, FieldSpace field_space)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tid = get_unique_region_tree_id();
      LogicalRegion region(tid, index_space, field_space);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create logical region!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_region.debug("Creating logical region in task %s (ID %lld) "
                              "with index space %x and field space %x "
                              "in new tree %d",
                              ctx->variants->name,ctx->get_unique_task_id(), 
                              index_space.id, field_space.id, tid);
      if (ctx->is_leaf())
      {
        log_task.error("Illegal region creation performed in leaf task "
                             "%s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (legion_spy_enabled)
        LegionSpy::log_top_region(index_space.id, field_space.id, tid);

      forest->create_logical_region(region);
      // Register the creation of a top-level region with the context
      ctx->register_region_creation(region);
      return region;
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_logical_region(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy logical region!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_region.debug("Deleting logical region (%x,%x) in "
                              "task %s (ID %lld)",
                              handle.index_space.id, handle.field_space.id, 
                              ctx->variants->name,ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal region destruction performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_logical_region_deletion(ctx, handle);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_logical_partition(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy logical partition!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_region.debug("Deleting logical partition (%x,%x) in task %s "
                              "(ID %lld)",
                              handle.index_partition.id, handle.field_space.id,
                              ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal partition destruction performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_logical_partition_deletion(ctx, handle);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool Internal::finalize_logical_region_destroy(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->destroy_logical_region(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_logical_partition_destroy(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_logical_partition(handle, address_space);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition(Context ctx, 
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition(LogicalRegion parent,
                                                    IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_color(
                        Context ctx, LogicalRegion parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_color(LogicalRegion par,
                                                             Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_color(LogicalRegion par,
                                                           const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    bool Internal::has_logical_partition_by_color(Context ctx, 
                                 LogicalRegion parent, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_partition_by_color(parent, ColorPoint(color));
    }

    //--------------------------------------------------------------------------
    bool Internal::has_logical_partition_by_color(LogicalRegion parent, 
                                                 const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_partition_by_color(parent, ColorPoint(color));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_logical_partition_by_tree(
                                                            IndexPartition part,
                                                            FieldSpace fspace,
                                                            RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion(LogicalPartition parent,
                                                 IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_color(Context ctx,
                                  LogicalPartition parent, const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_color(LogicalPartition par,
                                                          Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_color(LogicalPartition par,
                                                          const DomainPoint &c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    bool Internal::has_logical_subregion_by_color(Context ctx,
                              LogicalPartition parent, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_subregion_by_color(parent, ColorPoint(color));
    }

    //--------------------------------------------------------------------------
    bool Internal::has_logical_subregion_by_color(LogicalPartition parent, 
                                                 const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_subregion_by_color(parent, ColorPoint(color));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_logical_subregion_by_tree(IndexSpace handle,
                                                         FieldSpace fspace,
                                                         RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Internal::get_logical_region_color(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_logical_region_color(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_logical_region_color_point(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_logical_region_color_point(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_logical_partition_color(Context ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Internal::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_logical_partition_color_point(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    DomainPoint Internal::get_logical_partition_color_point(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle).get_point();
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_parent_logical_region(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Internal::get_parent_logical_region(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_parent_logical_partition(Context ctx, 
                                               LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_parent_logical_partition(Context ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Internal::get_parent_logical_partition(
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexAllocator Internal::create_index_allocator(Context ctx, 
                                                            IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create index allocator!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create index allocation requested in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return IndexAllocator(handle, forest->get_index_space_allocator(handle));
    }

    //--------------------------------------------------------------------------
    FieldAllocator Internal::create_field_allocator(Context ctx, 
                                                            FieldSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create field allocator!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal create field allocation requested in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return FieldAllocator(handle, ctx, high_level);
    }

    //--------------------------------------------------------------------------
    ArgumentMap Internal::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------
    {
      ArgumentMap::Impl *impl = legion_new<ArgumentMap::Impl>(
                                    legion_new<ArgumentMapStore>());
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return ArgumentMap(impl);
    }

    //--------------------------------------------------------------------------
    Future Internal::execute_task(Context ctx, 
                                          const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute task!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
        {
#ifdef INORDER_EXECUTION
          launcher.predicate_false_future.get_void_result();
#endif
          return launcher.predicate_false_future;
        }
        // Otherwise check to see if we have a value
        Future::Impl *result = legion_new<Future::Impl>(this, true/*register*/,
          get_available_distributed_id(true), address_space, address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskImpl *impl = find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
          {
            log_run.error("Predicated task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "TaskLauncher struct.",
                          impl->get_name(), ctx->get_task_name(),
                          ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      IndividualTask *task = get_available_individual_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute task call performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, launcher, check_privileges);
      log_task.debug("Registering new single task with unique id %lld "
                      "and task %s (ID %lld) with high level runtime in "
                      "addresss space %d",
                      task->get_unique_task_id(), task->variants->name, 
                      task->get_unique_task_id(), address_space);
#else
      Future result = task->initialize_task(ctx, launcher,
                                            false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.get_void_result();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap Internal::execute_index_space(Context ctx, 
                                                  const IndexLauncher &launcher)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        return execute_must_epoch(ctx, epoch_launcher);
      }
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        FutureMap::Impl *result = legion_new<FutureMap::Impl>(ctx, this);
        if (launcher.predicate_false_future.impl != NULL)
        {
#ifdef INORDER_EXECUTION
          // Wait for the result if we need things to happen in order
          launcher.predicate_false_future.get_void_result();
#endif
          Event ready_event = 
            launcher.predicate_false_future.impl->get_ready_event(); 
          if (ready_event.has_triggered())
          {
            const void *f_result = 
              launcher.predicate_false_future.impl->get_untyped_result();
            size_t f_result_size = 
              launcher.predicate_false_future.impl->get_untyped_size();
            for (Domain::DomainPointIterator itr(launcher.launch_domain); 
                  itr; itr++)
            {
              Future f = result->get_future(itr.p);
              f.impl->set_result(f_result, f_result_size, false/*own*/);
            }
            result->complete_all_futures();
          }
          else
          {
            // Otherwise launch a task to complete the future map,
            // add the necessary references to prevent premature
            // garbage collection by the runtime
            result->add_reference();
            launcher.predicate_false_future.impl->add_base_gc_ref(
                                                    FUTURE_HANDLE_REF);
            DeferredFutureMapSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_MAP_SET_ID;
            args.future_map = result;
            args.result = launcher.predicate_false_future.impl;
            args.domain = launcher.launch_domain;
            issue_runtime_meta_task(&args, sizeof(args), 
                                    HLR_DEFERRED_FUTURE_MAP_SET_ID, 
                                    NULL, ready_event); 
          }
          return FutureMap(result);
        }
        if (launcher.predicate_false_result.get_size() == 0)
        {
          // Check to make sure the task actually does expect to
          // have a void return type
          TaskImpl *impl = find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
          {
            log_run.error("Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "IndexLauncher struct.",
                          impl->get_name(), ctx->get_task_name(),
                          ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
          // Just initialize all the futures
          for (Domain::DomainPointIterator itr(launcher.launch_domain); 
                itr; itr++)
            result->get_future(itr.p);
        }
        else
        {
          const void *ptr = launcher.predicate_false_result.get_ptr();
          size_t ptr_size = launcher.predicate_false_result.get_size();
          for (Domain::DomainPointIterator itr(launcher.launch_domain); 
                itr; itr++)
          {
            Future f = result->get_future(itr.p);
            f.impl->set_result(ptr, ptr_size, false/*own*/);
          }
        }
        result->complete_all_futures();
        return FutureMap(result);
      }
      IndexTask *task = get_available_index_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      FutureMap result = task->initialize_task(ctx, launcher, check_privileges);
      log_task.debug("Registering new index space task with unique id "
                  "%lld and task %s (ID %lld) with high level runtime in "
                  "address space %d",
                  task->get_unique_task_id(), task->variants->name, 
                  task->get_unique_task_id(), address_space);
#else
      FutureMap result = 
                    task->initialize_task(ctx, launcher,
                                          false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.wait_all_results();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future Internal::execute_index_space(Context ctx, 
                            const IndexLauncher &launcher, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
          return launcher.predicate_false_future;
        // Otherwise check to see if we have a value
        Future::Impl *result = legion_new<Future::Impl>(this, true/*register*/, 
          get_available_distributed_id(true), address_space, address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskImpl *impl = find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
          {
            log_run.error("Predicated index task launch for task %s "
                                "in parent task %s (UID %lld) has non-void "
                                "return type but no default value for its "
                                "future if the task predicate evaluates to "
                                "false.  Please set either the "
                                "'predicate_false_result' or "
                                "'predicate_false_future' fields of the "
                                "IndexLauncher struct.",
                                impl->get_name(), ctx->get_task_name(),
                                ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      IndexTask *task = get_available_index_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, launcher, redop, 
                                            check_privileges);
      log_task.debug("Registering new index space task with unique id "
                  "%lld and task %s (ID %lld) with high level runtime in "
                  "address space %d",
                  task->get_unique_task_id(), task->variants->name, 
                  task->get_unique_task_id(), address_space);
#else
      Future result = 
            task->initialize_task(ctx, launcher, redop, 
                                  false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.get_void_result();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future Internal::execute_task(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &arg, 
                        const Predicate &predicate,
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      // Quick out for predicate false
      if (predicate == Predicate::FALSE_PRED)
        return Future(legion_new<Future::Impl>(this, true/*register*/,
          get_available_distributed_id(true), address_space, address_space));
      IndividualTask *task = get_available_individual_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute task!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute task call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, task_id, indexes, regions, arg,
                            predicate, id, tag, check_privileges);
      log_task.debug("Registering new single task with unique id %lld "
                      "and task %s (ID %lld) with high level runtime in "
                      "address space %d",
                      task->get_unique_task_id(), task->variants->name, 
                      task->get_unique_task_id(), address_space);
#else
      Future result = task->initialize_task(ctx, task_id, indexes, regions, arg,
                            predicate, id, tag, false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.get_void_result();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap Internal::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      // Quick out for predicate false
      if (predicate == Predicate::FALSE_PRED)
        return FutureMap(legion_new<FutureMap::Impl>(ctx,this));
      IndexTask *task = get_available_index_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      FutureMap result = task->initialize_task(ctx, task_id, domain, indexes,
                                regions, global_arg, arg_map, predicate,
                                must_parallelism, id, tag, check_privileges);
      log_task.debug("Registering new index space task with unique id "
                  "%lld and task %s (ID %lld) with high level runtime in "
                  "address space %d",
                  task->get_unique_task_id(), task->variants->name, 
                  task->get_unique_task_id(), address_space);
#else
      FutureMap result = task->initialize_task(ctx, task_id, domain, indexes,
                                regions, global_arg, arg_map, predicate,
                          must_parallelism, id, tag, false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.wait_all_results();
      }
#endif
      return result;
    }


    //--------------------------------------------------------------------------
    Future Internal::execute_index_space(Context ctx, 
                        Processor::TaskFuncID task_id,
                        const Domain domain,
                        const std::vector<IndexSpaceRequirement> &indexes,
                        const std::vector<FieldSpaceRequirement> &fields,
                        const std::vector<RegionRequirement> &regions,
                        const TaskArgument &global_arg, 
                        const ArgumentMap &arg_map,
                        ReductionOpID reduction, 
                        const TaskArgument &initial_value,
                        const Predicate &predicate,
                        bool must_parallelism, 
                        MapperID id, 
                        MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      // Quick out for predicate false
      if (predicate == Predicate::FALSE_PRED)
        return Future(legion_new<Future::Impl>(this, true/*register*/,
          get_available_distributed_id(true), address_space, address_space));
      IndexTask *task = get_available_index_task(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context execute index space!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, task_id, domain, indexes,
                            regions, global_arg, arg_map, reduction, 
                            initial_value, predicate, must_parallelism,
                            id, tag, check_privileges);
      log_task.debug("Registering new index space task with unique id "
                  "%lld and task %s (ID %lld) with high level runtime in "
                  "address space %d",
                  task->get_unique_task_id(), task->variants->name, 
                  task->get_unique_task_id(), address_space);
#else
      Future result = task->initialize_task(ctx, task_id, domain, indexes,
                            regions, global_arg, arg_map, reduction, 
                            initial_value, predicate, must_parallelism,
                            id, tag, false/*check privileges*/);
#endif
      execute_task_launch(ctx, task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.get_void_result();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Internal::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      MapOp *map_op = get_available_map_op(true);
#ifdef DEBUG_HIGH_LEVEL
      PhysicalRegion result = map_op->initialize(ctx, launcher, 
                                                 check_privileges);
      log_run.debug("Registering a map operation for region "
                           "(%x,%x,%x) in task %s (ID %lld)",
                           launcher.requirement.region.index_space.id, 
                           launcher.requirement.region.field_space.id, 
                           launcher.requirement.region.tree_id, 
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      PhysicalRegion result = map_op->initialize(ctx, launcher, 
                                                 false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;  
      int index = ctx->has_conflicting_regions(map_op, parent_conflict, 
                                               inline_conflict);
      if (parent_conflict)
      {
        log_run.error("Attempted an inline mapping of region "
                            "(%x,%x,%x) that conflicts with mapped region " 
                            "(%x,%x,%x) at index %d of parent task %s "
                            "(ID %lld) that would ultimately result in "
                            "deadlock. Instead you receive this error "
                            "message.",
                            launcher.requirement.region.index_space.id,
                            launcher.requirement.region.field_space.id,
                            launcher.requirement.region.tree_id,
                            ctx->regions[index].region.index_space.id,
                            ctx->regions[index].region.field_space.id,
                            ctx->regions[index].region.tree_id,
                            index, ctx->variants->name, 
                            ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK);
      }
      if (inline_conflict)
      {
        log_run.error("Attempted an inline mapping of region " 
                            "(%x,%x,%x) "
                            "that conflicts with previous inline mapping in "
                            "task %s (ID %lld) that would "
                            "ultimately result in deadlock.  Instead you "
                            "receive this error message.",
                            launcher.requirement.region.index_space.id,
                            launcher.requirement.region.field_space.id,
                            launcher.requirement.region.tree_id,
                            ctx->variants->name, 
                            ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      ctx->register_inline_mapped_region(result);
      add_to_dependence_queue(ctx->get_executing_processor(), map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.wait_until_valid();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Internal::map_region(Context ctx, 
                    const RegionRequirement &req, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      MapOp *map_op = get_available_map_op(true);
#ifdef DEBUG_HIGH_LEVEL
      PhysicalRegion result = map_op->initialize(ctx, req, id, tag, 
                                                 check_privileges);
      log_run.debug("Registering a map operation for region " 
                           "(%x,%x,%x) "
                           "in task %s (ID %lld)",
                           req.region.index_space.id, req.region.field_space.id,
                           req.region.tree_id, ctx->variants->name, 
                           ctx->get_unique_task_id());
#else
      PhysicalRegion result = map_op->initialize(ctx, req, id, tag, 
                                                 false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;
      int index = ctx->has_conflicting_regions(map_op, parent_conflict,
                                               inline_conflict);
      if (parent_conflict)
      {
        log_run.error("Attempted an inline mapping of region " 
                            "(%x,%x,%x) "
                            "that conflicts with mapped region " 
		            "(%x,%x,%x) at "
                            "index %d of parent task %s (ID %lld) that would "
                            "ultimately result in deadlock.  Instead you "
                            "receive this error message.",
                            req.region.index_space.id,
                            req.region.field_space.id,
                            req.region.tree_id,
                            ctx->regions[index].region.index_space.id,
                            ctx->regions[index].region.field_space.id,
                            ctx->regions[index].region.tree_id,
                            index, ctx->variants->name, 
                            ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK);
      }
      if (inline_conflict)
      {
        log_run.error("Attempted an inline mapping of region " 
                            "(%x,%x,%x) "
                            "that conflicts with previous inline mapping in "
                            "task %s (ID %lld) that would "
                            "ultimately result in deadlock.  Instead you "
                            "receive this error message.",
                            req.region.index_space.id,
                            req.region.field_space.id,
                            req.region.tree_id,
                            ctx->variants->name, 
                            ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      ctx->register_inline_mapped_region(result);
      add_to_dependence_queue(ctx->get_executing_processor(), map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.wait_until_valid();
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Internal::map_region(Context ctx, unsigned idx, 
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      PhysicalRegion result = ctx->get_physical_region(idx);
      // Check to see if we are already mapped, if not, then remap it
      if (!result.impl->is_mapped())
        remap_region(ctx, result);
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::remap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.impl->is_mapped())
        return;
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context remap region!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal remap operation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      MapOp *map_op = get_available_map_op(true);
      map_op->initialize(ctx, region);
      ctx->register_inline_mapped_region(region);
      add_to_dependence_queue(ctx->get_executing_processor(), map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        region.wait_until_valid();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context unmap region!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal unmap operation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ctx->unregister_inline_mapped_region(region);
      if (region.impl->is_mapped())
        region.impl->unmap_region();
    }

    //--------------------------------------------------------------------------
    void Internal::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      ctx->unmap_all_mapped_regions();
    }

    //--------------------------------------------------------------------------
    void Internal::fill_field(Context ctx, LogicalRegion handle,
                             LogicalRegion parent, FieldID fid,
                             const void *value, size_t value_size,
                             const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context fill operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal fill operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      fill_op->initialize(ctx, handle, parent, fid, 
                          value, value_size, pred, check_privileges);
      log_run.debug("Registering a fill operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      fill_op->initialize(ctx, handle, parent, fid,
                          value, value_size, pred, false/*check privileges*/);
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
#ifdef INORDER_EXECUTION
      Event term_event = fill_op->get_completion_event();
#endif
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::fill_field(Context ctx, LogicalRegion handle,
                             LogicalRegion parent, FieldID fid,
                             Future f, const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context fill operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal fill operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      fill_op->initialize(ctx, handle, parent, fid, f,
                          pred, check_privileges);
      log_run.debug("Registering a fill operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      fill_op->initialize(ctx, handle, parent, fid, f,
                          pred, false/*check privileges*/);
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
#ifdef INORDER_EXECUTION
      Event term_event = fill_op->get_completion_event();
#endif
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::fill_fields(Context ctx, LogicalRegion handle,
                              LogicalRegion parent,
                              const std::set<FieldID> &fields,
                              const void *value, size_t value_size,
                              const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context fill operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal fill operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      fill_op->initialize(ctx, handle, parent, fields, 
                          value, value_size, pred, check_privileges);
      log_run.debug("Registering a fill operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      fill_op->initialize(ctx, handle, parent, fields,
                          value, value_size, pred, false/*check privileges*/);
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
#ifdef INORDER_EXECUTION
      Event term_event = fill_op->get_completion_event();
#endif
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::fill_fields(Context ctx, LogicalRegion handle,
                              LogicalRegion parent,
                              const std::set<FieldID> &fields,
                              Future f, const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context fill operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal fill operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      fill_op->initialize(ctx, handle, parent, fields, f, 
                          pred, check_privileges);
      log_run.debug("Registering a fill operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      fill_op->initialize(ctx, handle, parent, fields, f,
                          pred, false/*check privileges*/);
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
#ifdef INORDER_EXECUTION
      Event term_event = fill_op->get_completion_event();
#endif
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::fill_fields(Context ctx, const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context fill operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal fill operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      fill_op->initialize(ctx, launcher, check_privileges);
      log_run.debug("Registering a fill operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      fill_op->initialize(ctx, launcher, false/*check privileges*/);
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
#ifdef INORDER_EXECUTION
      Event term_event = fill_op->get_completion_event();
#endif
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Internal::attach_hdf5(Context ctx, const char *file_name,
                                        LogicalRegion handle, 
                                        LogicalRegion parent,
                                  const std::map<FieldID,const char*> field_map,
                                        LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      AttachOp *attach_op = get_available_attach_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context attach hdf5 file!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal attach hdf5 file operation performed in "
                       "leaf task %s (ID %lld)",
                       ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      PhysicalRegion result = attach_op->initialize_hdf5(ctx, file_name,
                       handle, parent, field_map, mode, check_privileges);
#else
      PhysicalRegion result = attach_op->initialize_hdf5(ctx, file_name,
               handle, parent, field_map, mode, false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;
      int index = ctx->has_conflicting_regions(attach_op, parent_conflict,
                                               inline_conflict);
      if (parent_conflict)
      {
        log_run.error("Attempted an attach hdf5 file operation on region " 
                      "(%x,%x,%x) that conflicts with mapped region " 
                      "(%x,%x,%x) at index %d of parent task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_hdf5 on file %s",
                      handle.index_space.id, handle.field_space.id, 
                      handle.tree_id, ctx->regions[index].region.index_space.id,
                      ctx->regions[index].region.field_space.id,
                      ctx->regions[index].region.tree_id, index, 
                      ctx->variants->name, ctx->get_unique_task_id(), 
                      file_name);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK);
      }
      if (inline_conflict)
      {
        log_run.error("Attempted an attach hdf5 file operation on region " 
                      "(%x,%x,%x) that conflicts with previous inline "
                      "mapping in task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_hdf5 on file %s",
                      handle.index_space.id, handle.field_space.id, 
                      handle.tree_id, ctx->variants->name, 
                      ctx->get_unique_task_id(), file_name);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      add_to_dependence_queue(ctx->get_executing_processor(), attach_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
        result.wait_until_valid();
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::detach_hdf5(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context detach hdf5 file!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal detach hdf5 file operation performed in "
                       "leaf task %s (ID %lld)",
                       ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      
      // Then issue the detach operation
      Processor proc = ctx->get_executing_processor();
      DetachOp *detach_op = get_available_detach_op(true);
      detach_op->initialize_detach(ctx, region);
#ifdef INORDER_EXECUTION
      Event term_event = detach_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, detach_op);
      // If the region is still mapped, then unmap it
      if (region.impl->is_mapped())
      {
        ctx->unregister_inline_mapped_region(region);
        region.impl->unmap_region();
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Internal::attach_file(Context ctx, const char *file_name,
                                        LogicalRegion handle,
                                        LogicalRegion parent,
                                  const std::vector<FieldID> field_vec,
                                        LegionFileMode mode)
    //--------------------------------------------------------------------------
    {
      AttachOp *attach_op = get_available_attach_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context attach normal file!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal attach normal file operation performed in "
                       "leaf task %s (ID %lld)",
                       ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      PhysicalRegion result = attach_op->initialize_file(ctx, file_name,
                       handle, parent, field_vec, mode, check_privileges);
#else
      PhysicalRegion result = attach_op->initialize_file(ctx, file_name,
               handle, parent, field_vec, mode, false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;
      int index = ctx->has_conflicting_regions(attach_op, parent_conflict,
                                               inline_conflict);
      if (parent_conflict)
      {
        log_run.error("Attempted an attach file operation on region "
                      "(%x,%x,%x) that conflicts with mapped region "
                      "(%x,%x,%x) at index %d of parent task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_file on file %s",
                      handle.index_space.id, handle.field_space.id,
                      handle.tree_id, ctx->regions[index].region.index_space.id,
                      ctx->regions[index].region.field_space.id,
                      ctx->regions[index].region.tree_id, index,
                      ctx->variants->name, ctx->get_unique_task_id(),
                      file_name);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK);
      }
      if (inline_conflict)
      {
        log_run.error("Attempted an attach file operation on region "
                      "(%x,%x,%x) that conflicts with previous inline "
                      "mapping in task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_file on file %s",
                      handle.index_space.id, handle.field_space.id,
                      handle.tree_id, ctx->variants->name,
                      ctx->get_unique_task_id(), file_name);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      add_to_dependence_queue(ctx->get_executing_processor(), attach_op);
#ifdef INORDER_EXECUTION
      if (program_order_executiong)
        result.wait_until_valid();
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::detach_file(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context detach normal file!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal detach normal file operation performed in "
                       "leaf task %s (ID %lld)",
                       ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif

      // Then issue the detach operation
      Processor proc = ctx->get_executing_processor();
      DetachOp *detach_op = get_available_detach_op(true);
      detach_op->initialize_detach(ctx, region);
#ifdef INORDER_EXECUTION
      Event term_event = detach_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, detach_op);
      // If the region is still mapped, then unmap it
      if (region.impl->is_mapped())
      {
        ctx->unregister_inline_mapped_region(region);
	// Defer the unmap itself until DetachOp::trigger_execution to avoid
	// blocking the application task
	//   region.impl->unmap_region();
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::issue_copy_operation(Context ctx, 
                                       const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      CopyOp *copy_op = get_available_copy_op(true);  
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue copy operation!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal copy operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      copy_op->initialize(ctx, launcher, check_privileges);
      log_run.debug("Registering a copy operation in task %s "
                           "(ID %lld)",
                           ctx->variants->name, ctx->get_unique_task_id());
#else
      copy_op->initialize(ctx, launcher, false/*check privileges*/);
#endif
#ifdef INORDER_EXECUTION
      Event term_event = copy_op->get_completion_event();
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
      // Issue the copy operation
      add_to_dependence_queue(proc, copy_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
    }

    //--------------------------------------------------------------------------
    Predicate Internal::create_predicate(Context ctx, const Future &f) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create predicate!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      if (f.impl == NULL)
      {
        log_run.error("Illegal predicate creation performed on "
                            "empty future inside of task %s (ID %lld).",
                            ctx->variants->name, ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_PREDICATE_FUTURE);
      }
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal predicate creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      FuturePredOp *pred_op = get_available_future_pred_op(true);
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(ctx, f);
#ifdef INORDER_EXECUTION
      Event term_event = pred_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, pred_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate Internal::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create predicate not!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal NOT predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      NotPredOp *pred_op = get_available_not_pred_op(true);
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(ctx, p);
#ifdef INORDER_EXECUTION
      Event term_event = pred_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, pred_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate Internal::predicate_and(Context ctx, const Predicate &p1, 
                                                  const Predicate &p2) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create predicate and!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal AND predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      AndPredOp *pred_op = get_available_and_pred_op(true);
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(ctx, p1, p2);
#ifdef INORDER_EXECUTION
      Event term_event = pred_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, pred_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate Internal::predicate_or(Context ctx, const Predicate &p1, 
                                                 const Predicate &p2)  
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create predicate or!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
#endif
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal OR predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      OrPredOp *pred_op = get_available_or_pred_op(true);
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(ctx, p1, p2);
#ifdef INORDER_EXECUTION
      Event term_event = pred_op->get_completion_event();
#endif
      add_to_dependence_queue(proc, pred_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Lock Internal::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      return Lock(Reservation::create_reservation());
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      ctx->destroy_user_lock(l.reservation_lock);
    }

    //--------------------------------------------------------------------------
    Grant Internal::acquire_grant(Context ctx, 
                                 const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      // Kind of annoying, but we need to unpack and repack the
      // Lock type here to build new requests because the C++
      // type system is dumb with nested classes.
      std::vector<Grant::Impl::ReservationRequest> 
        unpack_requests(requests.size());
      for (unsigned idx = 0; idx < requests.size(); idx++)
      {
        unpack_requests[idx] = 
          Grant::Impl::ReservationRequest(requests[idx].lock.reservation_lock,
                                          requests[idx].mode,
                                          requests[idx].exclusive);
      }
      return Grant(legion_new<Grant::Impl>(unpack_requests));
    }

    //--------------------------------------------------------------------------
    void Internal::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      grant.impl->release_grant();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Internal::create_phase_barrier(Context ctx, unsigned arrivals) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create phase barrier!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Creating phase barrier in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Barrier result = Barrier::create_barrier(arrivals);
#ifdef LEGION_SPY
      LegionSpy::log_phase_barrier(result);
#endif
      return PhaseBarrier(result);
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy phase barrier!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Destroying phase barrier in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      ctx->destroy_user_barrier(pb.phase_barrier);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Internal::advance_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context advance phase barrier!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Advancing phase barrier in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Barrier bar = pb.phase_barrier;
      Barrier new_bar = bar.advance_barrier();
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(bar, new_bar);
#endif
      return PhaseBarrier(new_bar);
    }

    //--------------------------------------------------------------------------
    DynamicCollective Internal::create_dynamic_collective(Context ctx,
                                                         unsigned arrivals,
                                                         ReductionOpID redop,
                                                         const void *init_value,
                                                         size_t init_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context create dynamic collective!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Creating dynamic collective in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Barrier result = Barrier::create_barrier(arrivals, redop, 
                                               init_value, init_size);
#ifdef LEGION_SPY
      LegionSpy::log_phase_barrier(result);
#endif
      return DynamicCollective(result, redop);
    }

    //--------------------------------------------------------------------------
    void Internal::destroy_dynamic_collective(Context ctx, DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context destroy "
                            "dynamic collective!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Destroying dynamic collective in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      ctx->destroy_user_barrier(dc.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void Internal::arrive_dynamic_collective(Context ctx, DynamicCollective dc,
                                            const void *buffer, size_t size,
                                            unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context arrive dynamic collective!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Arrive dynamic collective in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      dc.phase_barrier.arrive(count, Event::NO_EVENT, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Internal::defer_dynamic_collective_arrival(Context ctx, 
                                                   DynamicCollective dc,
                                                   Future f, unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context defer dynamic "
                            "collective arrival!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Defer dynamic collective arrival in "
                          "task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      f.impl->contribute_to_collective(dc.phase_barrier, count);
    }

    //--------------------------------------------------------------------------
    Future Internal::get_dynamic_collective_result(Context ctx, 
                                                  DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context get dynamic "
                            "collective result!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Get dynamic collective result in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      DynamicCollectiveOp *collective = 
        get_available_dynamic_collective_op(true);
      Future result = collective->initialize(ctx, dc);
#ifdef INORDER_EXECUTION
      Event term_event = collective->get_completion_event();
#endif
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(proc, collective);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective Internal::advance_dynamic_collective(Context ctx,
                                                          DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context advance dynamic "
                            "collective!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Advancing dynamic collective in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Barrier bar = dc.phase_barrier;
      Barrier new_bar = bar.advance_barrier();
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(bar, new_bar);
#endif
      return DynamicCollective(new_bar, dc.redop);
    }

    //--------------------------------------------------------------------------
    void Internal::issue_acquire(Context ctx, const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AcquireOp *acquire_op = get_available_acquire_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue acquire!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Issuing an acquire operation in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal acquire operation performed in leaf task"
                              "%s (ID %lld)",
                              ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      acquire_op->initialize(ctx, launcher, check_privileges);
#else
      acquire_op->initialize(ctx, launcher, false/*check privileges*/);
#endif
#ifdef INORDER_EXECUTION
      Event term_event = acquire_op->get_completion_event();
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this acquire operation.
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(acquire_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
      // Issue the acquire operation
      add_to_dependence_queue(ctx->get_executing_processor(), acquire_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::issue_release(Context ctx, const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      ReleaseOp *release_op = get_available_release_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue release!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Issuing a release operation in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal release operation performed in leaf task"
                             "%s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      release_op->initialize(ctx, launcher, check_privileges);
#else
      release_op->initialize(ctx, launcher, false/*check privileges*/);
#endif
#ifdef INORDER_EXECUTION
      Event term_event = release_op->get_completion_event();
#endif
      Processor proc = ctx->get_executing_processor();
      // Check to see if we need to do any unmappings and remappings
      // before we can issue the release operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        ctx->find_conflicting_regions(release_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
      // Issue the release operation
      add_to_dependence_queue(ctx->get_executing_processor(), release_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      FenceOp *fence_op = get_available_fence_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue mapping fence!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Issuing a mapping fence in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal legion mapping fence call in leaf task "
                             "%s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      fence_op->initialize(ctx, FenceOp::MAPPING_FENCE);
#ifdef INORDER_EXECUTION
      Event term_event = fence_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), fence_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      FenceOp *fence_op = get_available_fence_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue execution fence!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Issuing an execution fence in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal Legion execution fence call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      fence_op->initialize(ctx, FenceOp::EXECUTION_FENCE);
#ifdef INORDER_EXECUTION
      Event term_event = fence_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), fence_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::begin_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context begin trace!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Beginning a trace in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal Legion begin trace call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // Mark that we are starting a trace
      ctx->begin_trace(tid);
    }

    //--------------------------------------------------------------------------
    void Internal::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context end trace!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Ending a trace in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal Legion end trace call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // Mark that we are done with the trace
      ctx->end_trace(tid); 
    }

    //--------------------------------------------------------------------------
    void Internal::complete_frame(Context ctx)
    //--------------------------------------------------------------------------
    {
      FrameOp *frame_op = get_available_frame_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue frame!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Issuing a frame in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal Legion complete frame call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      frame_op->initialize(ctx);
#ifdef INORDER_EXECUTION
      Event term_event = frame_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), frame_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    FutureMap Internal::execute_must_epoch(Context ctx, 
                                          const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      MustEpochOp *epoch_op = get_available_epoch_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context issue must epoch!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Executing a must epoch in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task.error("Illegal Legion execute must epoch call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      FutureMap result = epoch_op->initialize(ctx, launcher, check_privileges);
#else
      FutureMap result = epoch_op->initialize(ctx, launcher, 
                                              false/*check privileges*/);
#endif
      // Do all the stuff we normally have to do for a single task launch
      // except now for many task launches.
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      ProcessorManager *manager = proc_managers[proc];
      // Set the task options for all the constituent tasks
      epoch_op->set_task_options(manager);
      // Now find all the parent task regions we need to invalidate
      std::vector<PhysicalRegion> unmapped_regions;
      if (!unsafe_launch)
        epoch_op->find_conflicted_regions(unmapped_regions);
      if (!unmapped_regions.empty())
      {
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          unmapped_regions[idx].impl->unmap_region();
        }
      }
      // Now we can issue the must epoch
      add_to_dependence_queue(proc, epoch_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op(true);
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
        }
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        result.wait_all_results(); 
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    int Internal::get_tunable_value(Context ctx, TunableID tid,
                                   MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_run.debug("Getting a value for tunable variable %d in "
                          "task %s (ID %lld)", tid, ctx->variants->name,
                          ctx->get_unique_task_id());
#endif
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      ProcessorManager *manager = proc_managers[proc];
      return manager->invoke_mapper_get_tunable_value(ctx, tid, mid, tag);
    }

    //--------------------------------------------------------------------------
    Future Internal::get_current_time(Context ctx, const Future &precondition)
    //--------------------------------------------------------------------------
    {
      TimingOp *timing_op = get_available_timing_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context get current time!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Getting current time in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Future result = timing_op->initialize(ctx, precondition);
#ifdef INORDER_EXECUTION
      Event term_event = timing_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), timing_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future Internal::get_current_time_in_microseconds(Context ctx, 
                                                      const Future &pre)
    //--------------------------------------------------------------------------
    {
      TimingOp *timing_op = get_available_timing_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context get current "
                      "time in microseconds!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Getting current time in microseconds in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Future result = timing_op->initialize_microseconds(ctx, pre);
#ifdef INORDER_EXECUTION
      Event term_event = timing_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), timing_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future Internal::get_current_time_in_nanoseconds(Context ctx, 
                                                     const Future &pre)
    //--------------------------------------------------------------------------
    {
      TimingOp *timing_op = get_available_timing_op(true);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context get current time in nanoseconds!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      log_run.debug("Getting current time in nanoseconds in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
#endif
      Future result = timing_op->initialize_nanoseconds(ctx, pre);
#ifdef INORDER_EXECUTION
      Event term_event = timing_op->get_completion_event();
#endif
      add_to_dependence_queue(ctx->get_executing_processor(), timing_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Mapper* Internal::get_mapper(Context ctx, MapperID id, Processor target)
    //--------------------------------------------------------------------------
    {
      if (!target.exists())
      {
        Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
        assert(proc_managers.find(proc) != proc_managers.end());
#endif
        return proc_managers[proc]->find_mapper(id);
      }
      else
      {
        std::map<Processor,ProcessorManager*>::const_iterator finder = 
          proc_managers.find(target);
        if (finder == proc_managers.end())
        {
          log_run.error("Invalid processor " IDFMT " passed to "
                              "get mapper call.", target.id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_PROCESSOR_NAME);
        }
        return finder->second->find_mapper(id);
      }
    }

    //--------------------------------------------------------------------------
    Processor Internal::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      return ctx->get_executing_processor();
    }

    //--------------------------------------------------------------------------
    void Internal::raise_region_exception(Context ctx, 
                                               PhysicalRegion region, 
                                               bool nuclear)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& Internal::find_forward_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (forward_mpi_mapping.empty())
      {
        log_run.error("Forward MPI mapping call not supported with "
                            "calling configure_MPI_interoperability during "
                            "start up");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED);
      }
      return forward_mpi_mapping;
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>& Internal::find_reverse_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (reverse_mpi_mapping.empty())
      {
        log_run.error("Reverse MPI mapping call not supported with "
                            "calling configure_MPI_interoperability during "
                            "start up");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED);
      }
      return reverse_mpi_mapping;
    }

    //--------------------------------------------------------------------------
    MapperID Internal::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      MapperID result = __sync_fetch_and_add(&unique_mapper_id, runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // Check for overflow
      assert(result <= unique_mapper_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID& Internal::get_current_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      static MapperID current_mapper_id = MAX_APPLICATION_MAPPER_ID;
      return current_mapper_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID Internal::generate_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      MapperID &next_mapper = get_current_static_mapper_id(); 
      if (runtime_started)
      {
        log_run.error("Illegal call to 'generate_static_mapper_id' after "
                      "the runtime has been started!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_STATIC_CALL_POST_RUNTIME_START);
      }
      return next_mapper++;
    }

    //--------------------------------------------------------------------------
    void Internal::add_mapper(MapperID map_id, Mapper *mapper, 
                                      Processor proc)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out a way to put this check back in while
      // still supporting dynamic ID generation
#if 0
      if (map_id >= get_current_static_mapper_id())
      {
        log_run.error("Error registering mapper with ID %d. Exceeds the "
                      "statically set bounds on application mapper IDs of %d. "
                      "See %s in legion_config.h.", map_id, 
                      MAX_APPLICATION_MAPPER_ID, 
                      LEGION_MACRO_TO_STRING(MAX_APPLICATION_TASK_ID));
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MAX_APPLICATION_MAPPER_ID_EXCEEDED);
      }
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->add_mapper(map_id, mapper, true/*check*/);
      AutoLock m_lock(mapper_info_lock);
      mapper_infos[mapper] = MapperInfo(proc, map_id);
    }

    //--------------------------------------------------------------------------
    void Internal::replace_default_mapper(Mapper *mapper, 
                                                  Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->replace_default_mapper(mapper);
      AutoLock m_lock(mapper_info_lock);
      mapper_infos[mapper] = MapperInfo(proc, 0/*mapper id*/);
    }

    //--------------------------------------------------------------------------
    void Internal::register_projection_functor(ProjectionID pid,
                                              ProjectionFunctor *functor)
    //--------------------------------------------------------------------------
    {
      if (pid == 0)
      {
        log_run.error("ERROR: ProjectionID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      // No need for a lock because these all need to be reserved at
      // registration time before the runtime starts up
      std::map<ProjectionID,ProjectionFunctor*>::const_iterator finder = 
        projection_functors.find(pid);
      if (finder != projection_functors.end())
      {
        log_run.error("ERROR: ProjectionID %d has already been used in "
                                    "the region projection table\n", pid);
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_DUPLICATE_PROJECTION_ID);
      }
      projection_functors[pid] = functor;
    }

    //--------------------------------------------------------------------------
    ProjectionFunctor* Internal::find_projection_functor(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      std::map<ProjectionID,ProjectionFunctor*>::const_iterator finder = 
        projection_functors.find(pid);
      if (finder == projection_functors.end())
      {
        log_run.warning("Unable to find registered region projection "
                              "ID %d. Please upgrade to using projection "
                              "functors!", pid);
        // Uncomment this once we deprecate the old projection functions
#ifdef DEBUG_HIGH_LEVEL
        //assert(false);
#endif
        //exit(ERROR_INVALID_PROJECTION_ID);
        return NULL;
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(TaskID task_id, SemanticTag tag,
           const void *buffer, size_t size, bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = find_or_create_task_impl(task_id);
      impl->attach_semantic_information(tag, address_space, buffer, size, 
                                        is_mutable, send_to_owner);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(IndexSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(IndexPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(FieldSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(FieldSpace handle, FieldID fid,
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, fid, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(LogicalRegion handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Internal::attach_semantic_information(LogicalPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(TaskID task_id,SemanticTag tag,
              const void *&result, size_t &size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = find_or_create_task_impl(task_id);
      return impl->retrieve_semantic_information(tag, result, size, 
                                                 can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(IndexSpace handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, tag, result, size,
                                                   can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(IndexPartition handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, tag, result, size, 
                                                   can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(FieldSpace handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, tag, result, size,
                                                   can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, fid, tag, result, 
                                                   size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(LogicalRegion handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, tag, result, size,
                                                   can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Internal::retrieve_semantic_information(LogicalPartition handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size, bool can_fail,
                                                bool wait_until)
    //--------------------------------------------------------------------------
    {
      return forest->retrieve_semantic_information(handle, tag, result, size,
                                                   can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    FieldID Internal::allocate_field(Context ctx, FieldSpace space,
                                          size_t field_size, FieldID fid,
                                          bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context allocate field!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf() && !local)
      {
        log_task.error("Illegal non-local field allocation performed "
                             "in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (fid == AUTO_GENERATE_ID)
        fid = get_unique_field_id();

      if (legion_spy_enabled)
        LegionSpy::log_field_creation(space.id, fid);

      if (local)
        ctx->add_local_field(space, fid, field_size, serdez_id);
      else
      {
        forest->allocate_field(space, field_size, fid, serdez_id);
        ctx->register_field_creation(space, fid);
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void Internal::free_field(Context ctx, FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context free field!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal field destruction performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_field_deletion(ctx, space, fid);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::allocate_fields(Context ctx, FieldSpace space,
                                        const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context allocate fields!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf() && !local)
      {
        log_task.error("Illegal non-local field allocation performed "
                             "in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == AUTO_GENERATE_ID)
          resulting_fields[idx] = get_unique_field_id();

        if (legion_spy_enabled)
          LegionSpy::log_field_creation(space.id, resulting_fields[idx]);
      }
      if (local)
        ctx->add_local_fields(space, resulting_fields, sizes, serdez_id);
      else
      {
        forest->allocate_fields(space, sizes, resulting_fields, serdez_id); 
        ctx->register_field_creations(space, resulting_fields);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::free_fields(Context ctx, FieldSpace space,
                                    const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx == DUMMY_CONTEXT)
      {
        log_run.error("Illegal dummy context free fields!");
        assert(false);
        exit(ERROR_DUMMY_CONTEXT_OPERATION);
      }
      if (ctx->is_leaf())
      {
        log_task.error("Illegal field destruction performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Processor proc = ctx->get_executing_processor();
      DeletionOp *op = get_available_deletion_op(true);
      op->initialize_field_deletions(ctx, space, to_free);
#ifdef INORDER_EXECUTION
      Event term_event = op->get_completion_event();
#endif
      add_to_dependence_queue(proc, op);
#ifdef INORDER_EXECUTION
      if (program_order_execution && !term_event.has_triggered())
      {
        pre_wait(proc);
        term_event.wait();
        post_wait(proc);
      }
#endif
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& Internal::begin_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      if (!task->is_inline_task())
      {
#ifdef DEBUG_HIGH_LEVEL
        SingleTask *ctx = dynamic_cast<SingleTask*>(task);
        assert(ctx != NULL);
#else
        SingleTask *ctx = static_cast<SingleTask*>(task);
#endif
        return ctx->begin_task();
      }
      else
        return task->begin_inline_task();
    }

    //--------------------------------------------------------------------------
    void Internal::end_task(TaskOp *task, const void *result, 
                            size_t result_size, bool owned)
    //--------------------------------------------------------------------------
    {
      if (!task->is_inline_task())
      {
#ifdef DEBUG_HIGH_LEVEL
        SingleTask *ctx = dynamic_cast<SingleTask*>(task);
        assert(ctx != NULL);
#else
        SingleTask *ctx = static_cast<SingleTask*>(task);
#endif
        ctx->end_task(result, result_size, owned);
        decrement_total_outstanding_tasks();
      }
      else
        task->end_inline_task(result, result_size, owned);
    }

    //--------------------------------------------------------------------------
    TaskID Internal::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      TaskID result = __sync_fetch_and_add(&unique_task_id, runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // Check for overflow
      assert(result <= unique_task_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID Internal::register_variant(const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  CodeDescriptor *realm,
                                  bool ret,VariantID vid /*= AUTO_GENERATE_ID*/,
                                  bool check_task_id /*= true*/)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out a way to make this check safe with dynamic generation
#if 0
      if (check_task_id && (registrar.task_id >= MAX_APPLICATION_TASK_ID))
      {
        log_run.error("Error registering task with ID %d. Exceeds the "
                      "statically set bounds on application task IDs of %d. "
                      "See %s in legion_config.h.", 
                      registrar.task_id, MAX_APPLICATION_TASK_ID, 
                      LEGION_MACRO_TO_STRING(MAX_APPLICATION_TASK_ID));
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED);
      }
#endif
      // See if we need to make a new variant ID
      if (vid == AUTO_GENERATE_ID) // Make a variant ID to use
        vid = get_unique_variant_id();
      // First find the task implementation
      TaskImpl *task_impl = find_or_create_task_impl(registrar.task_id);
      // Make our variant and add it to the set of variants
      VariantImpl *impl = legion_new<VariantImpl>(this, vid, task_impl, 
                                                  registrar, ret, realm,
                                                  user_data, user_data_size);
      // Add this variant to the owner
      task_impl->add_variant(impl);
      AutoLock tv_lock(task_variant_lock);
      variant_table.push_back(impl);
      return vid;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Internal::find_or_create_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tv_lock(task_variant_lock,1,false/*exclusive*/);
        std::map<TaskID,TaskImpl*>::const_iterator finder = 
          task_table.find(task_id);
        if (finder != task_table.end())
          return finder->second;
      }
      TaskImpl *result = legion_new<TaskImpl>(task_id, this);
      AutoLock tv_lock(task_variant_lock);
      task_table[task_id] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Internal::find_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      AutoLock tv_lock(task_variant_lock,1,false/*exclusive*/);
      std::map<TaskID,TaskImpl*>::const_iterator finder = 
        task_table.find(task_id);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != task_table.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    VariantImpl* Internal::find_variant_impl(TaskID task_id, 
                                             VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      TaskImpl *owner = find_or_create_task_impl(task_id);
      return owner->find_variant_impl(variant_id);
    }

    //--------------------------------------------------------------------------
    TaskVariantCollection* Internal::get_collection(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = find_task_impl(task_id);
      return impl->get_collection();
    }

    //--------------------------------------------------------------------------
    MemoryManager* Internal::find_memory(Memory mem)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(memory_manager_lock);
      std::map<Memory,MemoryManager*>::const_iterator finder = 
        memory_managers.find(mem);
      if (finder != memory_managers.end())
        return finder->second;
      // Otherwise, if we haven't made it yet, make it now
      MemoryManager *result = new MemoryManager(mem, this);
      // Put it in the map
      memory_managers[mem] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::allocate_physical_instance(PhysicalManager *instance)
    //--------------------------------------------------------------------------
    {
      find_memory(instance->memory)->allocate_physical_instance(instance);
    }

    //--------------------------------------------------------------------------
    void Internal::free_physical_instance(PhysicalManager *instance)
    //--------------------------------------------------------------------------
    {
      find_memory(instance->memory)->free_physical_instance(instance);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Internal::find_address_space(Memory handle) const
    //--------------------------------------------------------------------------
    {
      // Just use the standard translation for now
      AddressSpaceID result = handle.address_space();
      return result;
    }

    //--------------------------------------------------------------------------
    size_t Internal::sample_allocated_space(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_allocated_space();
    }

    //--------------------------------------------------------------------------
    size_t Internal::sample_free_space(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_free_space();
    }

    //--------------------------------------------------------------------------
    unsigned Internal::sample_allocated_instances(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_allocated_instances();
    }

    //--------------------------------------------------------------------------
    unsigned Internal::sample_unmapped_tasks(Processor proc, Mapper *mapper)
    //--------------------------------------------------------------------------
    {
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(proc);
      if (finder != proc_managers.end())
      {
        MapperID map_id;
        // Find the ID of the mapper requesting the information
        {
          AutoLock m_lock(mapper_info_lock,1,false/*exclusive*/);
          std::map<Mapper*,MapperInfo>::const_iterator finder = 
            mapper_infos.find(mapper);
#ifdef DEBUG_HIGH_LEVEL
          assert(finder != mapper_infos.end());
#endif
          map_id = finder->second.map_id;
        }
        return finder->second->sample_unmapped_tasks(map_id);
      }
      else
        return 0;
    }

    //--------------------------------------------------------------------------
    MessageManager* Internal::find_messenger(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(sid < MAX_NUM_NODES);
      assert(sid != address_space); // shouldn't be sending messages to ourself
#endif
      MessageManager *result = message_managers[sid];
      if (result != NULL)
        return result;
      // If we made it here, then we don't have a message manager yet
      // re-take the lock and re-check to see if we don't have a manager
      // If we still don't then we need to make one
      AutoLock m_lock(message_manager_lock);
      // Re-check to see if we lost the race, force the compiler
      // to re-load the value here
      result = *(((MessageManager**volatile)message_managers)+sid);
      // If we're still NULL then we need to make the message manager
      if (result == NULL)
      {
        // Compute the set of processors in the remote address space
        std::set<Processor> remote_procs;
        std::set<Processor> remote_util_procs;
        for (std::map<Processor,AddressSpaceID>::const_iterator it = 
              proc_spaces.begin(); it != proc_spaces.end(); it++)
        {
          if (it->second != sid)
            continue;
          Processor::Kind k = it->first.kind();
          if (k == Processor::UTIL_PROC)
            remote_util_procs.insert(it->first);
          else
            remote_procs.insert(it->first);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(!remote_procs.empty() || !remote_util_procs.empty());
#endif
        result = new MessageManager(sid, this, max_message_size,
            (remote_util_procs.empty() ? remote_procs : remote_util_procs));
        // Save the result
        message_managers[sid] = result;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    MessageManager* Internal::find_messenger(Processor target)
    //--------------------------------------------------------------------------
    {
      return find_messenger(find_address_space(target));
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Internal::find_address_space(Processor target) const
    //--------------------------------------------------------------------------
    {
      std::map<Processor,AddressSpaceID>::const_iterator finder = 
        proc_spaces.find(target);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != proc_spaces.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void Internal::send_task(Processor target, TaskOp *task)
    //--------------------------------------------------------------------------
    {
      if (!target.exists())
      {
        log_run.error("Mapper requested invalid NO_PROC as target proc!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_TARGET_PROC);
      }
      // Check to see if the target processor is still local 
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(target);
      if (finder != proc_managers.end())
      {
        // Update the current processor
        task->current_proc = target;
        finder->second->add_to_ready_queue(task,false/*previous failure*/);
      }
      else
      {
        MessageManager *manager = find_messenger(target);
        Serializer rez;
        bool deactivate_task;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(task->get_task_kind());
          deactivate_task = task->pack_task(rez, target);
        }
        // Send tasks on the physical state virtual channel in case
        // they moved any state when they were sent
        manager->send_message(rez, TASK_MESSAGE, 
                              DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
        if (deactivate_task)
          task->deactivate();
      }
    }

    //--------------------------------------------------------------------------
    void Internal::send_tasks(Processor target, 
                                   const std::set<TaskOp*> &tasks)
    //--------------------------------------------------------------------------
    {
      if (!target.exists())
      {
        log_run.error("Mapper requested invalid NO_PROC as target proc!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_TARGET_PROC);
      }
      // Check to see if the target processor is still local 
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(target);
      if (finder != proc_managers.end())
      {
        // Still local
        for (std::set<TaskOp*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++)
        {
          // Update the current processor
          (*it)->current_proc = target;
          finder->second->add_to_ready_queue(*it,false/*previous failure*/);
        }
      }
      else
      {
        // Otherwise we need to send it remotely
        MessageManager *manager = find_messenger(target);
        unsigned idx = 1;
        for (std::set<TaskOp*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++,idx++)
        {
          Serializer rez;
          bool deactivate_task;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize((*it)->get_task_kind());
            deactivate_task = (*it)->pack_task(rez, target);
          }
          // Put it in the queue, flush the last task
          // Send tasks on the physical state virtual channel in case
          // they moved any state when they were sent
          manager->send_message(rez, TASK_MESSAGE,
                                DEFAULT_VIRTUAL_CHANNEL, (idx == tasks.size()));
          // Deactivate the task if it is remote
          if (deactivate_task)
            (*it)->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::send_steal_request(
              const std::multimap<Processor,MapperID> &targets, Processor thief)
    //--------------------------------------------------------------------------
    {
      for (std::multimap<Processor,MapperID>::const_iterator it = 
            targets.begin(); it != targets.end(); it++)
      {
        Processor target = it->first;
        std::map<Processor,ProcessorManager*>::const_iterator finder = 
          proc_managers.find(target);
        if (finder == proc_managers.end())
        {
          // Need to send remotely
          MessageManager *manager = find_messenger(target);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize(thief);
            int num_mappers = targets.count(target);
            rez.serialize(num_mappers);
            for ( ; it != targets.upper_bound(target); it++)
              rez.serialize(it->second);
          }
          manager->send_message(rez, STEAL_MESSAGE,
                                MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
        }
        else
        {
          // Still local, so notify the processor manager
          std::vector<MapperID> thieves;
          for ( ; it != targets.upper_bound(target); it++)
            thieves.push_back(it->second);
          finder->second->process_steal_request(thief, thieves);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::send_advertisements(const std::set<Processor> &targets,
                                            MapperID map_id, Processor source)
    //--------------------------------------------------------------------------
    {
      std::set<MessageManager*> already_sent;
      for (std::set<Processor>::const_iterator it = targets.begin();
            it != targets.end(); it++)
      {
        std::map<Processor,ProcessorManager*>::const_iterator finder = 
          proc_managers.find(*it);
        if (finder != proc_managers.end())
        {
          // still local
          finder->second->process_advertisement(source, map_id);
        }
        else
        {
          // otherwise remote, check to see if we already sent it
          MessageManager *messenger = find_messenger(*it);
          if (already_sent.find(messenger) != already_sent.end())
            continue;
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(source);
            rez.serialize(map_id);
          }
          messenger->send_message(rez, ADVERTISEMENT_MESSAGE, 
                                  MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
          already_sent.insert(messenger);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_NODE,
                               INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_request(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_REQUEST, 
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_return(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_RETURN,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_child_request(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_CHILD_REQUEST,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_node(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_NODE,
                               INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_request(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_REQUEST,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_return(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_RETURN,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_child_request(AddressSpaceID target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
                                SEND_INDEX_PARTITION_CHILD_REQUEST,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_NODE,
                               INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_REQUEST,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_return(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_RETURN,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_alloc_request(AddressSpaceID target,
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_ALLOC_REQUEST,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_alloc_notification(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_ALLOC_NOTIFICATION,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_top_alloc(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_TOP_ALLOC,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_free(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_FREE,
                                INDEX_AND_FIELD_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_top_level_region_request(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TOP_LEVEL_REGION_REQUEST,
                                LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_top_level_region_return(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TOP_LEVEL_REGION_RETURN,
                                  LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_region_node(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOGICAL_REGION_NODE,
                                  LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_destruction(IndexSpace handle, 
                                               AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_message(rez, INDEX_SPACE_DESTRUCTION_MESSAGE,
                               INDEX_AND_FIELD_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_destruction(IndexPartition handle, 
                                                   AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_message(rez, 
        INDEX_PARTITION_DESTRUCTION_MESSAGE, INDEX_AND_FIELD_VIRTUAL_CHANNEL,
                                                             false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_destruction(FieldSpace handle, 
                                               AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_message(rez, 
          FIELD_SPACE_DESTRUCTION_MESSAGE, INDEX_AND_FIELD_VIRTUAL_CHANNEL,
                                                              false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_region_destruction(LogicalRegion handle, 
                                                  AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_message(rez, 
          LOGICAL_REGION_DESTRUCTION_MESSAGE, LOGICAL_TREE_VIRTUAL_CHANNEL,
                                                              false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_partition_destruction(
                              LogicalPartition handle, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_message(rez, 
          LOGICAL_PARTITION_DESTRUCTION_MESSAGE, LOGICAL_TREE_VIRTUAL_CHANNEL,
                                                                false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_individual_remote_mapped(Processor target,
                                        Serializer &rez, bool flush /*= true*/)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_MAPPED,
                                           DEFAULT_VIRTUAL_CHANNEL, flush);
    }

    //--------------------------------------------------------------------------
    void Internal::send_individual_remote_complete(Processor target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_COMPLETE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_individual_remote_commit(Processor target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_COMMIT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_slice_remote_mapped(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_MAPPED,
                                       DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_slice_remote_complete(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_COMPLETE,
                                       DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_slice_remote_commit(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_COMMIT,
                                       DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_remote_registration(AddressSpaceID target, 
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_REMOTE_REGISTRATION,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_remote_valid_update(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_VALID_UPDATE,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_remote_gc_update(AddressSpaceID target,
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_GC_UPDATE,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_remote_resource_update(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_RESOURCE_UPDATE,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_add_create_reference(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_CREATE_ADD,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_did_remove_create_reference(AddressSpaceID target,
                                                    Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_CREATE_REMOVE,
                                           DISTRIBUTED_VIRTUAL_CHANNEL, flush);
    }

    //--------------------------------------------------------------------------
    void Internal::send_view_remote_registration(AddressSpaceID target, 
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, VIEW_REMOTE_REGISTRATION,
                                    DISTRIBUTED_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_back_atomic(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_BACK_ATOMIC,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_materialized_view(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MATERIALIZED_VIEW,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_materialized_update(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MATERIALIZED_UPDATE,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_composite_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_COMPOSITE_VIEW,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    } 

    //--------------------------------------------------------------------------
    void Internal::send_fill_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FILL_VIEW,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_deferred_update(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_DEFERRED_UPDATE,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_reduction_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_VIEW,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_reduction_update(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_UPDATE,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_instance_manager(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INSTANCE_MANAGER,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_reduction_manager(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_MANAGER,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_future(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_future_result(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_RESULT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_future_subscription(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_SUBSCRIPTION,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_make_persistent(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MAKE_PERSISTENT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_unmake_persistent(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_UNMAKE_PERSISTENT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_mapper_message(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MAPPER_MESSAGE,
                                        MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_mapper_broadcast(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MAPPER_BROADCAST,
                                         MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_task_impl_semantic_request(AddressSpaceID target,
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TASK_IMPL_SEMANTIC_REQ,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_semantic_request(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_SEMANTIC_REQ,
                                 SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_semantic_request(AddressSpaceID target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_INDEX_PARTITION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_semantic_request(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_SEMANTIC_REQ,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_semantic_request(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SEMANTIC_REQ,
                                SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_region_semantic_request(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
              SEND_LOGICAL_REGION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_partition_semantic_request(
                                         AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
            SEND_LOGICAL_PARTITION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_task_impl_semantic_info(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TASK_IMPL_SEMANTIC_INFO,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_SEMANTIC_INFO,
                                 SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_index_partition_semantic_info(AddressSpaceID target,
                                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_INDEX_PARTITION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_SEMANTIC_INFO,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_field_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SEMANTIC_INFO,
                                SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_region_semantic_info(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
              SEND_LOGICAL_REGION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_logical_partition_semantic_info(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
            SEND_LOGICAL_PARTITION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_subscribe_remote_context(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_SUBSCRIBE_REMOTE_CONTEXT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_free_remote_context(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FREE_REMOTE_CONTEXT,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_version_state_path_only(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_STATE_PATH,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_version_state_initialization(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_STATE_INIT,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_version_state_request(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_STATE_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_version_state_response(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_STATE_RESPONSE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_remote_instance_creation_request(AddressSpaceID target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INSTANCE_CREATION,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_remote_reduction_creation_request(AddressSpaceID target,
                                                         Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_CREATION,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_remote_creation_response(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CREATION_RESPONSE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }
    
    //--------------------------------------------------------------------------
    void Internal::send_back_logical_state(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_BACK_LOGICAL_STATE,
                                       DEFAULT_VIRTUAL_CHANNEL, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_variant_request(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VARIANT_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_variant_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VARIANT_RESPONSE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_constraint_request(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_constraint_response(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_RESPONSE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::send_constraint_release(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_RELEASE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      TaskOp::process_unpack_task(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_steal(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor target;
      derez.deserialize(target);
      Processor thief;
      derez.deserialize(thief);
      int num_mappers;
      derez.deserialize(num_mappers);
      std::vector<MapperID> thieves(num_mappers);
      for (int idx = 0; idx < num_mappers; idx++)
        derez.deserialize(thieves[idx]);
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(target) != proc_managers.end());
#endif
      proc_managers[target]->process_steal_request(thief, thieves);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_advertisement(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor source;
      derez.deserialize(source);
      MapperID map_id;
      derez.deserialize(map_id);
      // Just advertise it to all the managers
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->process_advertisement(source, map_id);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_node(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_return(derez); 
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_child_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_child_request(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_node(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_request(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_return(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_child_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_child_request(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_node(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_return(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_alloc_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_alloc_request(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_alloc_notification(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_alloc_notification(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_top_alloc(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_top_alloc(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_free(Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_free(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_top_level_region_request(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_top_level_request(forest, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Internal::handle_top_level_region_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_top_level_return(derez);   
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_region_node(Deserializer &derez, 
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_destruction(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      forest->destroy_index_space(handle, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_destruction(Deserializer &derez,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      forest->destroy_index_partition(handle, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_destruction(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      forest->destroy_field_space(handle, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_region_destruction(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      forest->destroy_logical_region(handle, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_partition_destruction(Deserializer &derez,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      forest->destroy_logical_partition(handle, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_individual_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_mapped(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_individual_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_individual_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_commit(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_slice_remote_mapped(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_slice_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_slice_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_remote_registration(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_registration(this,derez,source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_remote_valid_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_valid_update(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_remote_gc_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_gc_update(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_remote_resource_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_resource_update(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_create_add(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_add_create(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_did_create_remove(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remove_create(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_view_remote_registration(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalView::handle_view_remote_registration(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_back_atomic(Deserializer &derez,
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_back_atomic(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_materialized_view(Deserializer &derez, 
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_materialized_view(this, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_materialized_update(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_update(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_composite_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      CompositeView::handle_send_composite_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_fill_view(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FillView::handle_send_fill_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_deferred_update(Deserializer &derez, 
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DeferredView::handle_deferred_update(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_reduction_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_reduction_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_reduction_update(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_update(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_instance_manager(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceManager::handle_send_manager(this, source, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_send_reduction_manager(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionManager::handle_send_manager(this, source, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_future_send(Deserializer &derez,AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_send(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_future_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_result(derez, this);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_future_subscription(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_subscription(derez, this);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_make_persistent(Deserializer &derez,
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_make_persistent(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_unmake_persistent(Deserializer &derez,
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_unmake_persistent(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_mapper_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor target;
      derez.deserialize(target);
      MapperID map_id;
      derez.deserialize(map_id);
      Processor source;
      derez.deserialize(source);
      size_t length;
      derez.deserialize(length);
      const void *message = derez.get_current_pointer();
      derez.advance_pointer(length);
      invoke_mapper_handle_message(target, map_id, source, message, length);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_mapper_broadcast(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      MapperID map_id;
      derez.deserialize(map_id);
      Processor source;
      derez.deserialize(source);
      int radix, offset;
      derez.deserialize(radix);
      derez.deserialize(offset);
      size_t length;
      derez.deserialize(length);
      const void *message = derez.get_current_pointer();
      derez.advance_pointer(length);
      invoke_mapper_broadcast(map_id, source, message, length, radix, offset);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_task_impl_semantic_request(Deserializer &derez,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_semantic_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_semantic_request(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_semantic_request(Deserializer &derez, 
                                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_semantic_request(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_semantic_request(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_region_semantic_request(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_partition_semantic_request(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_task_impl_semantic_info(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_semantic_info(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_space_semantic_info(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_index_partition_semantic_info(Deserializer &derez, 
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_space_semantic_info(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_field_semantic_info(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_region_semantic_info(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_partition_semantic_info(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_subscribe_remote_context(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SingleTask *receiver;
      derez.deserialize(receiver);
      RemoteTask *remote_ctx;
      derez.deserialize(remote_ctx);
      receiver->record_remote_instance(source, remote_ctx);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_free_remote_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID remote_owner_uid;
      derez.deserialize(remote_owner_uid);
      release_remote_context(remote_owner_uid);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_version_state_path_only(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_path_only(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_version_state_initialization(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_initialization(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_version_state_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_request(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_version_state_response(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_response(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_remote_instance_creation(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_remote_instance_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_remote_reduction_creation(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_remote_reduction_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_remote_creation_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      UserEvent done_event;
      derez.deserialize(done_event);
      done_event.trigger();
    }

    //--------------------------------------------------------------------------
    void Internal::handle_logical_state_return(Deserializer &derez, 
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode::handle_logical_state_return(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_variant_request(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_variant_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_variant_response(Deserializer &derez) 
    //--------------------------------------------------------------------------
    {
      VariantImpl::handle_variant_response(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_constraint_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints::process_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_constraint_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints::process_response(this, derez);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_constraint_release(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints::process_release(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_top_level_task_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(address_space == 0); // should only happen on node 0
#endif
      UserEvent to_trigger;
      derez.deserialize(to_trigger);
      increment_outstanding_top_level_tasks();
      to_trigger.trigger();
    }

    //--------------------------------------------------------------------------
    void Internal::handle_top_level_task_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(address_space == 0); // should only happen on node 0
#endif
      decrement_outstanding_top_level_tasks();
    }

    //--------------------------------------------------------------------------
    void Internal::handle_shutdown_notification(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      initiate_runtime_shutdown(source);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_shutdown_response(Deserializer &derez, 
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      bool result;
      derez.deserialize(result);
#ifdef DEBUG_HIGH_LEVEL
      assert(shutdown_manager != NULL);
#endif
      if (shutdown_manager->handle_response(source, result))
      {
        ShutdownManager *local = shutdown_manager;
        shutdown_manager = NULL;
        local->finalize();
        delete local;
      }
    }

#ifdef HANG_TRACE
    //--------------------------------------------------------------------------
    void Internal::dump_processor_states(FILE *target)
    //--------------------------------------------------------------------------
    {
      // Don't need to hold the lock since we are hung when this is called  
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->dump_state(target);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void Internal::process_schedule_request(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(local_procs.find(proc) != local_procs.end());
#endif
      log_run.debug("Running scheduler on processor " IDFMT "", proc.id);
      ProcessorManager *manager = proc_managers[proc];
      manager->perform_scheduling();
#ifdef TRACE_ALLOCATION
      unsigned long long trace_count = 
        __sync_fetch_and_add(&allocation_tracing_count,1); 
      if ((trace_count % TRACE_ALLOCATION_FREQUENCY) == 0)
        dump_allocation_info();
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::process_profiling_task(Processor p, 
                                         const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(profiler != NULL);
#endif
      profiler->process_results(p, args, arglen);
    }

    //--------------------------------------------------------------------------
    void Internal::process_message_task(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      const char *buffer = (const char*)args;
      AddressSpaceID sender = *((const AddressSpaceID*)buffer);
      buffer += sizeof(sender);
      arglen -= sizeof(sender);
      find_messenger(sender)->receive_message(buffer, arglen);
    }

    //--------------------------------------------------------------------------
    void Internal::activate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it =
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->activate_context(context);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::deactivate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->deactivate_context(context);
      }
    }
 
    //--------------------------------------------------------------------------
    void Internal::execute_task_launch(Context ctx, TaskOp *task)
    //--------------------------------------------------------------------------
    {
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      ProcessorManager *manager = proc_managers[proc];
      // First ask the mapper to set the options for the task
      manager->invoke_mapper_set_task_options(task);
      // Now check to see if we're inling the task or just performing
      // a normal asynchronous task launch
      if (task->inline_task)
      {
        ctx->inline_child_task(task);
        // After we're done we can deactivate it since we
        // know that it will never be used again
        task->deactivate();
      }
      else
      {
        // Normal task launch, iterate over the context task's
        // regions and see if we need to unmap any of them
        std::vector<PhysicalRegion> unmapped_regions;
        if (!unsafe_launch)
          ctx->find_conflicting_regions(task, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          {
            unmapped_regions[idx].impl->unmap_region();
          }
        }
        // Issue the task call
        add_to_dependence_queue(proc, task);
        // Remap any unmapped regions
        if (!unmapped_regions.empty())
        {
          std::set<Event> mapped_events;
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          {
            MapOp *op = get_available_map_op(true);
            op->initialize(ctx, unmapped_regions[idx]);
            mapped_events.insert(op->get_completion_event());
            add_to_dependence_queue(proc, op);
          }
          // Wait for all the re-mapping operations to complete
          Event mapped_event = Event::merge_events(mapped_events);
          if (!mapped_event.has_triggered())
          {
            pre_wait(proc);
            mapped_event.wait();
            post_wait(proc);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Internal::add_to_dependence_queue(Processor p, Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.kind() != Processor::UTIL_PROC);
#endif
      SingleTask *parent = op->get_parent();
      parent->add_to_dependence_queue(op, false/*has lock*/);
    }
    
    //--------------------------------------------------------------------------
    void Internal::add_to_ready_queue(Processor p, 
                                           TaskOp *op, bool prev_fail)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.kind() != Processor::UTIL_PROC);
      assert(proc_managers.find(p) != proc_managers.end());
#endif
      proc_managers[p]->add_to_ready_queue(op, prev_fail);
    }

    //--------------------------------------------------------------------------
    void Internal::add_to_local_queue(Processor p, 
                                           Operation *op, bool prev_fail)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.kind() != Processor::UTIL_PROC);
      assert(proc_managers.find(p) != proc_managers.end());
#endif
      proc_managers[p]->add_to_local_ready_queue(op, prev_fail);
    }

    //--------------------------------------------------------------------------
    void Internal::pre_wait(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      // Don't do anything for now
    }

    //--------------------------------------------------------------------------
    void Internal::post_wait(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      // Don't do anything for now
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_pre_map_task(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_pre_map_task(task);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_select_variant(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_select_variant(task);
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_map_task(Processor proc, SingleTask *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_task(task);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_post_map_task(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_post_map_task(task);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_failed_mapping(Processor proc,
                                               Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_failed_mapping(mappable);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_notify_result(Processor proc,
                                              Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_notify_result(mappable);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_slice_domain(Processor proc, 
                      MultiTask *task, std::vector<Mapper::DomainSplit> &splits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_slice_domain(task, splits);
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_map_inline(Processor proc, Inline *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_inline(op);
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_map_copy(Processor proc, Copy *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_copy(op);
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_speculate(Processor proc, 
                                          Mappable *mappable, bool &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_speculate(mappable, value);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_configure_context(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif     
      // Values are initialized in SingleTask::activate_single
      proc_managers[proc]->invoke_mapper_configure_context(task);
      // Should only be using one of these at a time
      // Frames are the less common case so if we see them then use them
      if ((task->max_outstanding_frames > 0) && 
          (task->max_window_size > 0))
        task->max_window_size = -1;
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_rank_copy_targets(Processor proc,
                                                  Mappable *mappable,
                                                  LogicalRegion handle,
                                              const std::set<Memory> &memories,
                                                  bool complete,
                                                  size_t max_blocking_factor,
                                                  std::set<Memory> &to_reuse,
                                                std::vector<Memory> &to_create,
                                                  bool &create_one,
                                                  size_t &blocking_factor)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_rank_copy_targets(mappable,
          handle, memories, complete, max_blocking_factor, to_reuse, 
          to_create, create_one, blocking_factor);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_rank_copy_sources(Processor proc, 
                                           Mappable *mappable,
                                           const std::set<Memory> &memories,
                                           Memory destination,
                                           std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_rank_copy_sources(mappable,
                                memories, destination, chosen_order);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_notify_profiling(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_notify_profiling(task);
    }

    //--------------------------------------------------------------------------
    bool Internal::invoke_mapper_map_must_epoch(Processor proc,
        const std::vector<Task*> &tasks,
        const std::vector<Mapper::MappingConstraint> &constraints,
        MapperID map_id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_must_epoch(tasks, 
                                                      constraints, map_id, tag);
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_handle_message(Processor target,
          MapperID map_id, Processor source, const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      if (is_local(target))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(proc_managers.find(target) != proc_managers.end());
#endif
        proc_managers[target]->invoke_mapper_handle_message(map_id, source, 
                                                            message, length);
      }
      else
      {
        // Package up the message
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(length);
          rez.serialize(message,length);
        }
        send_mapper_message(find_address_space(target), rez);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_broadcast(MapperID map_id, Processor source,
                                          const void *message, size_t length, 
                                          int radix, int index)
    //--------------------------------------------------------------------------
    {
      // First send the message on to any other remote nodes
      int base = index * radix;
      int init = source.address_space();
      // The runtime stride is the same as the total number of address spaces
      const int total_address_spaces = runtime_stride;
      for (int r = 0; r < radix; r++)
      {
        int offset = base + r; 
        // If we've handled all of our address spaces then we are done
        if (offset > total_address_spaces)
          break;
        AddressSpaceID target = (init + offset - 1) % total_address_spaces; 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(radix);
          rez.serialize(offset);
          rez.serialize(length);
          rez.serialize(message,length);
        }
        send_mapper_broadcast(target, rez);
      }
      // Then send it to all of our local processors
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->invoke_mapper_handle_message(map_id, source, 
                                                 message, length);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::invoke_mapper_task_result(MapperID map_id, Processor proc,
                                            Event event, const void *result,
                                            size_t result_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_task_result(map_id, event, 
                                                     result, result_size);
    }

    //--------------------------------------------------------------------------
    Processor Internal::locate_mapper_info(Mapper *mapper, MapperID &map_id)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_info_lock,1,false/*exclusive*/);
      std::map<Mapper*,MapperInfo>::const_iterator finder = 
        mapper_infos.find(mapper);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != mapper_infos.end());
#endif
      map_id = finder->second.map_id;
      return finder->second.proc;
    }

    //--------------------------------------------------------------------------
    void Internal::handle_mapper_send_message(Mapper *mapper, Processor target,
                                             const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      MapperID map_id;
      // Find the source processor and the corresponding mapper ID
      Processor source = locate_mapper_info(mapper, map_id);
      // Note we can't actually send the message without risking deadlock
      // so instead lookup the processor manager for the mapper and 
      // tell it that it has a pending message to send when the mapper
      // call actually completes.
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(source) != proc_managers.end());
#endif
      proc_managers[source]->defer_mapper_message(target, map_id, 
                                                  message, length);
    }

    //--------------------------------------------------------------------------
    void Internal::handle_mapper_broadcast(Mapper *mapper, const void *message,
                                          size_t length, int radix)
    //--------------------------------------------------------------------------
    {
      MapperID map_id;
      // Find the source processor and the corresponding mapper ID
      Processor source = locate_mapper_info(mapper, map_id);
      // Note that we can't actually send the broadcast without risking 
      // deadlock so instead we're going to defer it until the mapper
      // call finishes running.
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(source) != proc_managers.end());
#endif
      proc_managers[source]->defer_mapper_broadcast(map_id, message, 
                                                    length, radix);
    }

    //--------------------------------------------------------------------------
    Event Internal::launch_mapper_task(Mapper *mapper, 
                             Processor::TaskFuncID tid, const TaskArgument &arg)
    //--------------------------------------------------------------------------
    {
      MapperID map_id;
      Processor source = locate_mapper_info(mapper, map_id);
      return launch_mapper_task(mapper, source, tid, arg, map_id);
    }

    //--------------------------------------------------------------------------
    void Internal::defer_mapper_call(Mapper *mapper, Event wait_on)
    //--------------------------------------------------------------------------
    {
      MapperID map_id;
      Processor source = locate_mapper_info(mapper, map_id);
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(source) != proc_managers.end());
#endif
      proc_managers[source]->defer_mapper_call(map_id, wait_on);
    }

    //--------------------------------------------------------------------------
    Processor Internal::find_processor_group(const std::set<Processor> &procs)
    //--------------------------------------------------------------------------
    {
      // Compute a hash of all the processor ids to avoid testing all sets 
      // Only need to worry about local IDs since all processors are
      // in this address space.
      ProcessorMask local_mask;
      for (std::set<Processor>::const_iterator it = procs.begin(); 
            it != procs.end(); it++)
      {
        uint64_t local_id = it->local_id();
#ifdef DEBUG_HIGH_LEVEL
        assert(local_id < MAX_NUM_PROCS);
#endif
        local_mask.set_bit(local_id);
      }
      uint64_t hash = local_mask.get_hash_key();
      AutoLock g_lock(group_lock);
      std::map<uint64_t,LegionDeque<ProcessorGroupInfo>::aligned >::iterator 
        finder = processor_groups.find(hash);
      if (finder != processor_groups.end())
      {
        for (LegionDeque<ProcessorGroupInfo>::aligned::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if (local_mask == it->processor_mask)
            return it->processor_group;
        }
      }
      // If we make it here create a new processor group and add it
      std::vector<Processor> input_procs(procs.begin(), procs.end());
      Processor group = Processor::create_group(input_procs);
      if (finder != processor_groups.end())
        finder->second.push_back(ProcessorGroupInfo(group, local_mask));
      else
        processor_groups[hash].push_back(ProcessorGroupInfo(group, local_mask));
      return group;
    }

    //--------------------------------------------------------------------------
    Event Internal::issue_runtime_meta_task(const void *args, size_t arglen,
                                           HLRTaskID tid, Operation *op,
                                           Event precondition, int priority,
                                           bool holds_reservation,
                                           Processor target)
    //--------------------------------------------------------------------------
    {
      // If this is not a task directly related to shutdown or is a message, 
      // to a remote node then increment the number of outstanding tasks
      if (tid < HLR_MESSAGE_ID)
        increment_total_outstanding_tasks();
#ifdef DEBUG_SHUTDOWN_HANG
      __sync_fetch_and_add(&outstanding_counts[tid],1);
#endif
      if (!target.exists())
      {
        // If we don't have a processor to explicitly target, figure
        // out which of our utility processors to use
        target = utility_group;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(target.exists());
#endif
      // If we hold a reservation, bump the priority by a lot to avoid
      // priority inversion when running meta task
      if (holds_reservation)
        priority += 1024;
      if (profiler != NULL && tid < HLR_MESSAGE_ID)
      {
        Realm::ProfilingRequestSet requests;
        profiler->add_meta_request(requests, tid, op);
        return target.spawn(HLR_TASK_ID, args, arglen,
                            requests, precondition, priority);
      }
      else
        return target.spawn(HLR_TASK_ID, args, arglen, precondition, priority);
    }

    //--------------------------------------------------------------------------
    void Internal::allocate_context(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      // Try getting something off the list of available contexts
      AutoLock avail_lock(available_lock);
      if (!available_contexts.empty())
      {
        task->assign_context(available_contexts.front());
        available_contexts.pop_front();
        return;
      }
      // If we failed to get a context, double the number of total 
      // contexts and then update the forest nodes to have the right
      // number of contexts available
      task->assign_context(RegionTreeContext(total_contexts));
      for (unsigned idx = 1; idx < total_contexts; idx++)
      {
        available_contexts.push_back(RegionTreeContext(total_contexts+idx));
      }
      // Mark that we doubled the total number of contexts
      // Very important that we do this before calling the
      // RegionTreeForest's resize method!
      total_contexts *= 2;
#if 0
      if (total_contexts > MAX_CONTEXTS)
      {
        log_run.error("ERROR: Maximum number of allowed contexts %d "
                            "exceeded when initializing task %s (UID %lld). "
                            "Please change 'MAX_CONTEXTS' at top "
                            "of legion_config.h and recompile. It is also "
                            "possible to reduce context usage by annotating "
                            "task variants as leaf tasks since leaf tasks do "
                            "not require context allocation.",
                            MAX_CONTEXTS, task->variants->name,
                            task->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_EXCEEDED_MAX_CONTEXTS);
      }
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(!available_contexts.empty());
#endif
      // Tell all the processor managers about the additional contexts
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->update_max_context_count(total_contexts); 
      }
    }

    //--------------------------------------------------------------------------
    void Internal::free_context(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext context = task->release_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
      forest->check_context_state(context);
#endif
      AutoLock avail_lock(available_lock);
      available_contexts.push_back(context);
    }

    //--------------------------------------------------------------------------
    DistributedID Internal::get_available_distributed_id(bool need_cont,
                                                        bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<DistributedID,
                     &Internal::get_available_distributed_id> 
                       continuation(this, distributed_id_lock);
        return continuation.get_result();
      }
      else if (!has_lock)
      {
        AutoLock d_lock(distributed_id_lock);
        return get_available_distributed_id(false,true);
      }
      if (!available_distributed_ids.empty())
      {
        DistributedID result = available_distributed_ids.front();
        available_distributed_ids.pop_front();
        return result;
      }
      DistributedID result = unique_distributed_id;
      unique_distributed_id += runtime_stride;
      return result;
    }

    //--------------------------------------------------------------------------
    void Internal::free_distributed_id(DistributedID did)
    //--------------------------------------------------------------------------
    {
      // Don't recycle distributed IDs if we're doing LegionSpy or LegionGC
#ifndef LEGION_GC
#ifndef LEGION_SPY
      AutoLock d_lock(distributed_id_lock);
      available_distributed_ids.push_back(did);
#endif
#endif
#ifdef DEBUG_HIGH_LEVEL
      AutoLock dist_lock(distributed_collectable_lock,1,false/*exclusive*/);
      assert(dist_collectables.find(did) == dist_collectables.end());
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::recycle_distributed_id(DistributedID did,Event recycle_event)
    //--------------------------------------------------------------------------
    {
      if (!recycle_event.has_triggered())
      {
        DeferredRecycleArgs deferred_recycle_args;
        deferred_recycle_args.hlr_id = HLR_DEFERRED_RECYCLE_ID;
        deferred_recycle_args.did = did;
        issue_runtime_meta_task(&deferred_recycle_args,
                                sizeof(deferred_recycle_args),
                                HLR_DEFERRED_RECYCLE_ID, NULL, recycle_event,
                                0/*priority*/, true/*holds reservation*/);
      }
      else
        free_distributed_id(did);
    }

    //--------------------------------------------------------------------------
    void Internal::register_distributed_collectable(DistributedID did,
                                                   DistributedCollectable *dc,
                                                   bool needs_lock)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        Event acquire_event = distributed_collectable_lock.acquire();
        if (!acquire_event.has_triggered())
        {
          RegisterDistributedContinuation continuation(did, dc, this);
          Event done_event = continuation.defer(this, acquire_event);
          done_event.wait();
          return;
        }
      }
      // If we make it here then we have the lock
#ifdef DEBUG_HIGH_LEVEL
      assert(dist_collectables.find(did) == dist_collectables.end());
#endif
      dist_collectables[did] = dc;
      distributed_collectable_lock.release();
    }
    
    //--------------------------------------------------------------------------
    void Internal::unregister_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(dist_collectables.find(did) != dist_collectables.end());
#endif
      dist_collectables.erase(did);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      return (dist_collectables.find(did) != dist_collectables.end());
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Internal::find_distributed_collectable(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,DistributedCollectable*>::const_iterator finder = 
        dist_collectables.find(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != dist_collectables.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Internal::weak_find_distributed_collectable(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,DistributedCollectable*>::const_iterator finder = 
        dist_collectables.find(did);
      if (finder == dist_collectables.end())
        return NULL;
      return finder->second;
    }
    
    //--------------------------------------------------------------------------
    void Internal::register_future(DistributedID did, Future::Impl *impl)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_futures.find(did) == local_futures.end());
#endif
      local_futures[did] = impl;
    }

    //--------------------------------------------------------------------------
    void Internal::unregister_future(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_futures.find(did) != local_futures.end());
#endif
      local_futures.erase(did);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_future(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock,1,false/*exclusive*/);
      return (local_futures.find(did) != local_futures.end());
    }

    //--------------------------------------------------------------------------
    Future::Impl* Internal::find_future(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock,1,false/*exclusive*/); 
      std::map<DistributedID,Future::Impl*>::const_iterator finder = 
        local_futures.find(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != local_futures.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    Future::Impl* Internal::find_or_create_future(DistributedID did,
                                                 AddressSpaceID owner_space)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
      std::map<DistributedID,Future::Impl*>::const_iterator finder = 
        local_futures.find(did);
      if (finder == local_futures.end())
      {
        Future::Impl *result = legion_new<Future::Impl>(this, false/*register*/,
                                                did, owner_space, 
                                                address_space);
        local_futures[did] = result;
        return result;
      }
      else
        return finder->second;
    }

    //--------------------------------------------------------------------------
    void Internal::defer_collect_user(LogicalView *view, Event term_event)
    //--------------------------------------------------------------------------
    {
      GarbageCollectionEpoch *to_trigger = NULL;
      {
        AutoLock gc(gc_epoch_lock);
        current_gc_epoch->add_collection(view, term_event);
        gc_epoch_counter++;
        if (gc_epoch_counter == Internal::gc_epoch_size)
        {
          to_trigger = current_gc_epoch;
          current_gc_epoch = new GarbageCollectionEpoch(this);
          pending_gc_epochs.insert(current_gc_epoch);
          gc_epoch_counter = 0;
        }
      }
      if (to_trigger != NULL)
        to_trigger->launch(0/*priority*/);
    }

    //--------------------------------------------------------------------------
    void Internal::complete_gc_epoch(GarbageCollectionEpoch *epoch)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_epoch_lock);
#ifdef DEBUG_HIGH_LEVEL
      std::set<GarbageCollectionEpoch*>::iterator finder = 
        pending_gc_epochs.find(epoch);
      assert(finder != pending_gc_epochs.end());
      pending_gc_epochs.erase(finder);
#else
      pending_gc_epochs.erase(epoch);
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::increment_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are on node 0 or not
      if (address_space != 0)
      {
        // Send a message to node 0 requesting permission to 
        // lauch a new top-level task and wait on an event
        // to signal that permission has been granted
        UserEvent grant_event = UserEvent::create_user_event();
        Serializer rez;
        rez.serialize(grant_event);
        find_messenger(0)->send_message(rez, SEND_TOP_LEVEL_TASK_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
        grant_event.wait();
      }
      else
      {
        __sync_fetch_and_add(&outstanding_top_level_tasks,1);
      }
    }

    //--------------------------------------------------------------------------
    void Internal::decrement_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are on node 0 or not
      if (address_space != 0)
      {
        // Send a message to node 0 indicating that we finished
        // executing a top-level task
        Serializer rez;
        find_messenger(0)->send_message(rez, SEND_TOP_LEVEL_TASK_COMPLETE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
      }
      else
      {
        unsigned prev = __sync_fetch_and_sub(&outstanding_top_level_tasks,1);
#ifdef DEBUG_HIGH_LEVEL
        assert(prev > 0);
#endif
        // Check to see if we have no more outstanding top-level tasks
        // If we don't launch a task to handle the try to shutdown the runtime 
        if (prev == 1)
          issue_runtime_shutdown_attempt();
      }
    }

    //--------------------------------------------------------------------------
    void Internal::issue_runtime_shutdown_attempt(void)
    //--------------------------------------------------------------------------
    {
      HLRTaskID hlr_id = HLR_SHUTDOWN_ATTEMPT_TASK_ID; 
      // Issue this with a low priority so that other meta-tasks
      // have an opportunity to run
      issue_runtime_meta_task(&hlr_id, sizeof(hlr_id), hlr_id, NULL,
                              Event::NO_EVENT, INT_MIN);
    }

    //--------------------------------------------------------------------------
    void Internal::attempt_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(address_space == 0); // should only happen on node 0
#endif
      // As long as we still don't have any top-level tasks, 
      // keep trying to shutdown the runtime
      if (__sync_fetch_and_add(&outstanding_top_level_tasks,0) == 0)
        initiate_runtime_shutdown(address_space);
    }

    //--------------------------------------------------------------------------
    void Internal::initiate_runtime_shutdown(AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      log_shutdown.info("Received notification on node %d from node %d",
                        address_space, source);
      // Launch our last garbage collection epoch and wait for it to
      // finish so we can try to have no outstanding tasks
      Event gc_done = Event::NO_EVENT;
      {
        AutoLock gc(gc_epoch_lock);
        if (current_gc_epoch != NULL)
        {
          gc_done = current_gc_epoch->launch(0/*priority*/);
          current_gc_epoch = NULL;
        }
      }
      if (!gc_done.has_triggered())
        gc_done.wait();
      ShutdownManager *local_manager = 
        new ShutdownManager(this, source, message_managers[source]);
      // IMPORTANT: always send messages to remote nodes
      // so that they can begin their shutdown process which
      // may be a pre-requisite for some of our oustanding
      // tasks being able to run.
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
      {
        if (idx == source)
          continue;
        if (message_managers[idx] != NULL)
          local_manager->add_manager(idx, message_managers[idx]);
      }

      // Record if we have any outstanding tasks
      if (has_outstanding_tasks())
        local_manager->record_outstanding_tasks();

      // Record if we have any outstanding profiling requests
      if (profiler != NULL && profiler->has_outstanding_requests())
        local_manager->record_outstanding_profiling_requests();

      // Check to see if we have any remote nodes
      if (local_manager->has_managers())
      {
        // Make a shutdown manager and see if we can register it
        bool already_marked = false;
        // Don't need to make a shutdown manager if we only have
        // one incoming direction from the source
        {
          AutoLock shut(shutdown_lock);
          if (shutdown_manager == NULL)
            shutdown_manager = local_manager;
          else
            already_marked = true;
        }
        // If we were already marked we can send the response now
        if (already_marked)
        {
          local_manager->send_response();
          delete local_manager;
        }
        else
        {
          local_manager->send_notifications();
        }
      }
      else
      {
        local_manager->finalize();
        delete local_manager;
      }
    }

    //--------------------------------------------------------------------------
    void Internal::finalize_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
      std::set<Event> shutdown_events;
      // Launch tasks to shutdown all the runtime instances
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      Realm::ProfilingRequestSet empty_requests;
      // TODO: Make this work with Sam's internal collective framework
      if (Internal::separate_runtime_instances)
      {
        // If we are doing separate runtime instances, run it once on every
        // processor since we have separate runtimes for every processor
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          shutdown_events.insert(
              it->spawn(SHUTDOWN_TASK_ID, NULL, 0, empty_requests));
                                  
        }
      }
      else
      {
        // In the normal case we just have to run this once on every node
        std::set<AddressSpace> shutdown_spaces;
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          AddressSpace space = it->address_space();
          if (shutdown_spaces.find(space) == shutdown_spaces.end())
          {
            shutdown_events.insert(
                it->spawn(SHUTDOWN_TASK_ID, NULL, 0, empty_requests));
            shutdown_spaces.insert(space);
          }
        }
      }
      // Then tell Realm to shutdown when they are all done
      Event shutdown_precondition = Event::merge_events(shutdown_events);
      RealmRuntime realm = RealmRuntime::get_runtime();
      realm.shutdown(shutdown_precondition);
    }

    //--------------------------------------------------------------------------
    bool Internal::has_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
      return (__sync_fetch_and_add(&total_outstanding_tasks,0) != 0);
    }

    //--------------------------------------------------------------------------
    IndividualTask* Internal::get_available_individual_task(bool need_cont,
                                                           bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<IndividualTask*,
                     &Internal::get_available_individual_task> 
                       continuation(this, individual_task_lock);
        return continuation.get_result();
      }
      IndividualTask *result = get_available(individual_task_lock, 
                                         available_individual_tasks, has_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      if (!has_lock)
      {
        AutoLock i_lock(individual_task_lock);
        out_individual_tasks.insert(result);
      }
      else
        out_individual_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PointTask* Internal::get_available_point_task(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<PointTask*,
                     &Internal::get_available_point_task> 
                       continuation(this, point_task_lock);
        return continuation.get_result();
      }
      PointTask *result = get_available(point_task_lock, 
                                        available_point_tasks, has_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      if (!has_lock)
      {
        AutoLock p_lock(point_task_lock);
        out_point_tasks.insert(result);
      }
      else
        out_point_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexTask* Internal::get_available_index_task(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<IndexTask*,
                     &Internal::get_available_index_task> 
                       continuation(this, index_task_lock);
        return continuation.get_result();
      }
      IndexTask *result = get_available(index_task_lock, 
                                       available_index_tasks, has_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      if (!has_lock)
      {
        AutoLock i_lock(index_task_lock);
        out_index_tasks.insert(result);
      }
      else
        out_index_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    SliceTask* Internal::get_available_slice_task(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<SliceTask*,
                     &Internal::get_available_slice_task> 
                       continuation(this, slice_task_lock);
        return continuation.get_result();
      }
      SliceTask *result = get_available(slice_task_lock, 
                                       available_slice_tasks, has_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      if (!has_lock)
      {
        AutoLock s_lock(slice_task_lock);
        out_slice_tasks.insert(result);
      }
      else
        out_slice_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    RemoteTask* Internal::get_available_remote_task(bool need_cont, 
                                                   bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<RemoteTask*,
                     &Internal::get_available_remote_task> 
                       continuation(this, remote_task_lock);
        return continuation.get_result();
      }
      return get_available(remote_task_lock, available_remote_tasks, has_lock);
    }

    //--------------------------------------------------------------------------
    InlineTask* Internal::get_available_inline_task(bool need_cont,
                                                   bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<InlineTask*,
                     &Internal::get_available_inline_task> 
                       continuation(this, inline_task_lock);
        return continuation.get_result();
      }
      return get_available(inline_task_lock, available_inline_tasks, has_lock);
    }

    //--------------------------------------------------------------------------
    MapOp* Internal::get_available_map_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<MapOp*,
                     &Internal::get_available_map_op> 
                       continuation(this, map_op_lock);
        return continuation.get_result();
      }
      return get_available(map_op_lock, available_map_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    CopyOp* Internal::get_available_copy_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<CopyOp*,
                     &Internal::get_available_copy_op> 
                       continuation(this, copy_op_lock);
        return continuation.get_result();
      }
      return get_available(copy_op_lock, available_copy_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    FenceOp* Internal::get_available_fence_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<FenceOp*,
                     &Internal::get_available_fence_op> 
                       continuation(this, fence_op_lock);
        return continuation.get_result();
      }
      return get_available(fence_op_lock, available_fence_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    FrameOp* Internal::get_available_frame_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<FrameOp*,
                     &Internal::get_available_frame_op> 
                       continuation(this, frame_op_lock);
        return continuation.get_result();
      }
      return get_available(frame_op_lock, available_frame_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    DeletionOp* Internal::get_available_deletion_op(bool need_cont, 
                                                   bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<DeletionOp*,
                     &Internal::get_available_deletion_op> 
                       continuation(this, deletion_op_lock);
        return continuation.get_result();
      }
      return get_available(deletion_op_lock, available_deletion_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    InterCloseOp* Internal::get_available_inter_close_op(bool need_cont,
                                                        bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<InterCloseOp*,
                     &Internal::get_available_inter_close_op> 
                       continuation(this, inter_close_op_lock);
        return continuation.get_result();
      }
      return get_available(inter_close_op_lock, 
                           available_inter_close_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    ReadCloseOp* Internal::get_available_read_close_op(bool need_cont,
                                                       bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<ReadCloseOp*,
                     &Internal::get_available_read_close_op> 
                       continuation(this, read_close_op_lock);
        return continuation.get_result();
      }
      return get_available(read_close_op_lock, 
                           available_read_close_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    PostCloseOp* Internal::get_available_post_close_op(bool need_cont,
                                                      bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<PostCloseOp*,
                     &Internal::get_available_post_close_op> 
                       continuation(this, post_close_op_lock);
        return continuation.get_result();
      }
      return get_available(post_close_op_lock, 
                           available_post_close_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    VirtualCloseOp* Internal::get_available_virtual_close_op(bool need_cont,
                                                            bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<VirtualCloseOp*,
                     &Internal::get_available_virtual_close_op> 
                       continuation(this, virtual_close_op_lock);
        return continuation.get_result();
      }
      return get_available(virtual_close_op_lock,
                           available_virtual_close_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp* Internal::get_available_dynamic_collective_op(
                                                  bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<DynamicCollectiveOp*,
                     &Internal::get_available_dynamic_collective_op> 
                       continuation(this, dynamic_collective_op_lock);
        return continuation.get_result();
      }
      return get_available(dynamic_collective_op_lock, 
                           available_dynamic_collective_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    FuturePredOp* Internal::get_available_future_pred_op(bool need_cont,
                                                        bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<FuturePredOp*,
                     &Internal::get_available_future_pred_op> 
                       continuation(this, future_pred_op_lock);
        return continuation.get_result();
      }
      return get_available(future_pred_op_lock, 
                           available_future_pred_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    NotPredOp* Internal::get_available_not_pred_op(bool need_cont,bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<NotPredOp*,
                     &Internal::get_available_not_pred_op> 
                       continuation(this, not_pred_op_lock);
        return continuation.get_result();
      }
      return get_available(not_pred_op_lock, available_not_pred_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    AndPredOp* Internal::get_available_and_pred_op(bool need_cont,bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<AndPredOp*,
                     &Internal::get_available_and_pred_op> 
                       continuation(this, and_pred_op_lock);
        return continuation.get_result();
      }
      return get_available(and_pred_op_lock, available_and_pred_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    OrPredOp* Internal::get_available_or_pred_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<OrPredOp*,
                     &Internal::get_available_or_pred_op> 
                       continuation(this, or_pred_op_lock);
        return continuation.get_result();
      }
      return get_available(or_pred_op_lock, available_or_pred_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    AcquireOp* Internal::get_available_acquire_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<AcquireOp*,
                     &Internal::get_available_acquire_op> 
                       continuation(this, acquire_op_lock);
        return continuation.get_result();
      }
      return get_available(acquire_op_lock, available_acquire_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    ReleaseOp* Internal::get_available_release_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<ReleaseOp*,
                     &Internal::get_available_release_op> 
                       continuation(this, release_op_lock);
        return continuation.get_result();
      }
      return get_available(release_op_lock, available_release_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp* Internal::get_available_capture_op(bool need_cont, 
                                                      bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<TraceCaptureOp*,
                     &Internal::get_available_capture_op> 
                       continuation(this, capture_op_lock);
        return continuation.get_result();
      }
      return get_available(capture_op_lock, available_capture_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp* Internal::get_available_trace_op(bool need_cont, 
                                                     bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<TraceCompleteOp*,
                     &Internal::get_available_trace_op> 
                       continuation(this, trace_op_lock);
        return continuation.get_result();
      }
      return get_available(trace_op_lock, available_trace_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    MustEpochOp* Internal::get_available_epoch_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<MustEpochOp*,
                     &Internal::get_available_epoch_op> 
                       continuation(this, epoch_op_lock);
        return continuation.get_result();
      }
      MustEpochOp *result = 
        get_available(epoch_op_lock, available_epoch_ops, has_lock);
#ifdef DEBUG_HIGH_LEVEL
      if (!has_lock)
      {
        AutoLock e_lock(epoch_op_lock);
        out_must_epoch.insert(result);
      }
      else
        out_must_epoch.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp* Internal::get_available_pending_partition_op(
                                                  bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<PendingPartitionOp*,
                     &Internal::get_available_pending_partition_op> 
                       continuation(this, pending_partition_op_lock);
        return continuation.get_result();
      }
      return get_available(pending_partition_op_lock, 
                           available_pending_partition_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp* Internal::get_available_dependent_partition_op(
                                                  bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<DependentPartitionOp*,
                     &Internal::get_available_dependent_partition_op> 
                       continuation(this, dependent_partition_op_lock);
        return continuation.get_result();
      }
      return get_available(dependent_partition_op_lock, 
                           available_dependent_partition_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    FillOp* Internal::get_available_fill_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<FillOp*,
                     &Internal::get_available_fill_op> 
                       continuation(this, fill_op_lock);
        return continuation.get_result();
      }
      return get_available(fill_op_lock, available_fill_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    AttachOp* Internal::get_available_attach_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<AttachOp*,
                     &Internal::get_available_attach_op> 
                       continuation(this, attach_op_lock);
        return continuation.get_result();
      }
      return get_available(attach_op_lock, available_attach_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    DetachOp* Internal::get_available_detach_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<DetachOp*,
                     &Internal::get_available_detach_op> 
                       continuation(this, detach_op_lock);
        return continuation.get_result();
      }
      return get_available(detach_op_lock, available_detach_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    TimingOp* Internal::get_available_timing_op(bool need_cont, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (need_cont)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!has_lock);
#endif
        GetAvailableContinuation<TimingOp*,
                     &Internal::get_available_timing_op> 
                       continuation(this, timing_op_lock);
        return continuation.get_result();
      }
      return get_available(timing_op_lock, available_timing_ops, has_lock);
    }

    //--------------------------------------------------------------------------
    void Internal::free_individual_task(IndividualTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(individual_task_lock);
      available_individual_tasks.push_front(task);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_individual_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::free_point_task(PointTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(point_task_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_point_tasks.erase(task);
#endif
      // Note that we can safely delete point tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      if (available_point_tasks.size() == LEGION_MAX_RECYCLABLE_OBJECTS)
        legion_delete(task);
      else
        available_point_tasks.push_front(task);
    }

    //--------------------------------------------------------------------------
    void Internal::free_index_task(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(index_task_lock);
      available_index_tasks.push_front(task);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_index_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Internal::free_slice_task(SliceTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(slice_task_lock);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_slice_tasks.erase(task);
#endif
      // Note that we can safely delete slice tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      if (available_slice_tasks.size() == LEGION_MAX_RECYCLABLE_OBJECTS)
        legion_delete(task);
      else
        available_slice_tasks.push_front(task);
    }

    //--------------------------------------------------------------------------
    void Internal::free_remote_task(RemoteTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(remote_task_lock);
      // Note that we can safely delete remote tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      if (available_remote_tasks.size() == LEGION_MAX_RECYCLABLE_OBJECTS)
        legion_delete(task);
      else
        available_remote_tasks.push_front(task);
    }

    //--------------------------------------------------------------------------
    void Internal::free_inline_task(InlineTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inline_task_lock);
      // Note that we can safely delete inline tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      if (available_inline_tasks.size() == LEGION_MAX_RECYCLABLE_OBJECTS)
        legion_delete(task);
      else
        available_inline_tasks.push_front(task);
    }

    //--------------------------------------------------------------------------
    void Internal::free_map_op(MapOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(map_op_lock);
      available_map_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_copy_op(CopyOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(copy_op_lock);
      available_copy_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_fence_op(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fence_op_lock);
      available_fence_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_frame_op(FrameOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(frame_op_lock);
      available_frame_ops.push_back(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_deletion_op(DeletionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(deletion_op_lock);
      available_deletion_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_inter_close_op(InterCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inter_close_op_lock);
      available_inter_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_read_close_op(ReadCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(read_close_op_lock);
      available_read_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_post_close_op(PostCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(post_close_op_lock);
      available_post_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_virtual_close_op(VirtualCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(virtual_close_op_lock);
      available_virtual_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_dynamic_collective_op(DynamicCollectiveOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock dc_lock(dynamic_collective_op_lock);
      available_dynamic_collective_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_future_predicate_op(FuturePredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_pred_op_lock);
      available_future_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_not_predicate_op(NotPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(not_pred_op_lock);
      available_not_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_and_predicate_op(AndPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(and_pred_op_lock);
      available_and_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_or_predicate_op(OrPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(or_pred_op_lock);
      available_or_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_acquire_op(AcquireOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(acquire_op_lock);
      available_acquire_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_release_op(ReleaseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(release_op_lock);
      available_release_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_capture_op(TraceCaptureOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(capture_op_lock);
      available_capture_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_trace_op(TraceCompleteOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(trace_op_lock);
      available_trace_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_epoch_op(MustEpochOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(epoch_op_lock);
      available_epoch_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_pending_partition_op(PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(pending_partition_op_lock);
      available_pending_partition_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_dependent_partition_op(DependentPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(dependent_partition_op_lock);
      available_dependent_partition_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_fill_op(FillOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fill_op_lock);
      available_fill_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_attach_op(AttachOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(attach_op_lock);
      available_attach_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_detach_op(DetachOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(detach_op_lock);
      available_detach_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Internal::free_timing_op(TimingOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(timing_op_lock);
      available_timing_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    RemoteTask* Internal::find_or_init_remote_context(UniqueID uid,
                                                     Processor orig_proc,
                                                     SingleTask *remote_parent)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock rem_lock(remote_lock,1,false/*exclusive*/);
        std::map<UniqueID,RemoteTask*>::const_iterator finder = 
          remote_contexts.find(uid);
        if (finder != remote_contexts.end())
          return finder->second;
      }
      // Otherwise we need to make one
      RemoteTask *remote_ctx = get_available_remote_task(false);
      RemoteTask *result = remote_ctx;
      bool lost_race = false;
      {
        AutoLock rem_lock(remote_lock);
        std::map<UniqueID,RemoteTask*>::const_iterator finder = 
          remote_contexts.find(uid);
        if (finder != remote_contexts.end())
        {
          lost_race = true;
          result = finder->second;
        }
        else // Put it in the map
          remote_contexts[uid] = remote_ctx;
      }
      if (lost_race)
        free_remote_task(remote_ctx);
      else
      {
        remote_ctx->initialize_remote(uid, remote_parent, false/*top*/);
        AddressSpaceID target = find_address_space(orig_proc);
        // In some cases we might be asked to temporarily make a
        // remote task for a task that has been sent back to its
        // owner node, in which case we don't have to send the
        // subscription message.
        if (target != address_space)
        {
          // Send back the subscription message
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(remote_parent);
            rez.serialize(remote_ctx);
          }
          send_subscribe_remote_context(target, rez);
        }
        else // We're back to being local so we can just notify the parent
          remote_parent->record_remote_instance(target, result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    SingleTask* Internal::find_remote_context(UniqueID uid, 
                                             SingleTask *remote_parent_ctx)
    //--------------------------------------------------------------------------
    {
      // See if we can find it in the set of remote contexts, if
      // not then we must be local so return the remote parent ctx
      AutoLock rem_lock(remote_lock,1,false/*exclusive*/);
      std::map<UniqueID,RemoteTask*>::const_iterator finder = 
        remote_contexts.find(uid);
      if (finder != remote_contexts.end())
        return finder->second;
      return remote_parent_ctx;
    }

    //--------------------------------------------------------------------------
    void Internal::release_remote_context(UniqueID remote_owner_uid)
    //--------------------------------------------------------------------------
    {
      RemoteTask *remote_task;
      {
        AutoLock rem_lock(remote_lock);
        std::map<UniqueID,RemoteTask*>::iterator finder = 
          remote_contexts.find(remote_owner_uid);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != remote_contexts.end());
#endif
        remote_task = finder->second;
        remote_contexts.erase(finder);
      }
      // Now we can deactivate it
      remote_task->deactivate();
    }

    //--------------------------------------------------------------------------
    bool Internal::is_local(Processor proc) const
    //--------------------------------------------------------------------------
    {
      return (local_procs.find(proc) != local_procs.end());
    }

    //--------------------------------------------------------------------------
    IndexSpaceID Internal::get_unique_index_space_id(void)
    //--------------------------------------------------------------------------
    {
      IndexSpaceID result = __sync_fetch_and_add(&unique_index_space_id,
                                                 runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of partitions created
      // then we are really in a bad place.
      assert(result <= unique_index_space_id); 
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartitionID Internal::get_unique_index_partition_id(void)
    //--------------------------------------------------------------------------
    {
      IndexPartitionID result = __sync_fetch_and_add(&unique_index_partition_id,
                                                     runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of partitions created
      // then we are really in a bad place.
      assert(result <= unique_index_partition_id); 
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceID Internal::get_unique_field_space_id(void)
    //--------------------------------------------------------------------------
    {
      FieldSpaceID result = __sync_fetch_and_add(&unique_field_space_id,
                                                 runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of field spaces
      // created then we are really in a bad place.
      assert(result <= unique_field_space_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexTreeID Internal::get_unique_index_tree_id(void)
    //--------------------------------------------------------------------------
    {
      IndexTreeID result = __sync_fetch_and_add(&unique_index_tree_id,
                                                runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of region trees
      // created then we are really in a bad place.
      assert(result <= unique_index_tree_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeID Internal::get_unique_region_tree_id(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeID result = __sync_fetch_and_add(&unique_region_tree_id,
                                                 runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of region trees
      // created then we are really in a bad place.
      assert(result <= unique_region_tree_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    UniqueID Internal::get_unique_operation_id(void)
    //--------------------------------------------------------------------------
    {
      UniqueID result = __sync_fetch_and_add(&unique_operation_id,
                                             runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      assert(result <= unique_operation_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    FieldID Internal::get_unique_field_id(void)
    //--------------------------------------------------------------------------
    {
      FieldID result = __sync_fetch_and_add(&unique_field_id,
                                            runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      assert(result <= unique_field_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID Internal::get_unique_variant_id(void)
    //--------------------------------------------------------------------------
    {
      VariantID result = __sync_fetch_and_add(&unique_variant_id,
                                              runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      assert(result <= unique_variant_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID Internal::get_unique_constraint_id(void)
    //--------------------------------------------------------------------------
    {
      LayoutConstraintID result = __sync_fetch_and_add(&unique_constraint_id,
                                                       runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      assert(result <= unique_constraint_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    LegionErrorType Internal::verify_requirement(
                               const RegionRequirement &req, FieldID &bad_field)
    //--------------------------------------------------------------------------
    {
      FieldSpace sp = (req.handle_type == SINGULAR) 
                      || (req.handle_type == REG_PROJECTION)
                        ? req.region.field_space : req.partition.field_space;
      // First make sure that all the privilege fields are valid for
      // the given field space of the region or partition
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        if (!forest->has_field(sp, *it))
        {
          bad_field = *it;
          return ERROR_FIELD_SPACE_FIELD_MISMATCH;
        }
      }
      // Make sure that the requested node is a valid request
      if ((req.handle_type == SINGULAR) || (req.handle_type == REG_PROJECTION))
      {
        forest->get_node(req.region);
        if (!forest->has_node(req.region))
          return ERROR_INVALID_REGION_HANDLE;
      }
      else
      {
        forest->get_node(req.partition);
        if (!forest->has_node(req.partition))
          return ERROR_INVALID_PARTITION_HANDLE;
      }

      // Then check that any instance fields are included in the privilege 
      // fields.  Make sure that there are no duplicates in the instance fields
      std::set<FieldID> inst_duplicates;
      for (std::vector<FieldID>::const_iterator it = 
            req.instance_fields.begin(); it != 
            req.instance_fields.end(); it++)
      {
        if (req.privilege_fields.find(*it) == req.privilege_fields.end())
        {
          bad_field = *it;
          return ERROR_INVALID_INSTANCE_FIELD;
        }
        if (inst_duplicates.find(*it) != inst_duplicates.end())
        {
          bad_field = *it;
          return ERROR_DUPLICATE_INSTANCE_FIELD;
        }
        inst_duplicates.insert(*it);
      }

      // If this is a projection requirement and the child region selected will 
      // need to be in exclusive mode then the partition must be disjoint
      if ((req.handle_type == PART_PROJECTION) && 
          (IS_WRITE(req)))
      {
        if (!forest->is_disjoint(req.partition))
          return ERROR_NON_DISJOINT_PARTITION;
      }

      // Made it here, then there is no error
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    Future Internal::help_create_future(Operation *op /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return Future(legion_new<Future::Impl>(this, true/*register*/,
                                     get_available_distributed_id(true),
                                     address_space, address_space, op));
    }

    //--------------------------------------------------------------------------
    void Internal::help_complete_future(const Future &f)
    //--------------------------------------------------------------------------
    {
      f.impl->complete_future();
    }

    //--------------------------------------------------------------------------
    bool Internal::help_reset_future(const Future &f)
    //--------------------------------------------------------------------------
    {
      return f.impl->reset_future();
    }

    //--------------------------------------------------------------------------
    unsigned Internal::generate_random_integer(void)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(random_lock);
      unsigned result = nrand48(random_state);
      return result;
    }

#ifdef TRACE_ALLOCATION 
    //--------------------------------------------------------------------------
    void Internal::trace_allocation(AllocationType type, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(allocation_lock);
      std::map<AllocationType,AllocationTracker>::iterator finder = 
        allocation_manager.find(type);
      size_t alloc_size = size * elems;
      finder->second.total_allocations += elems;
      finder->second.total_bytes += alloc_size;
      finder->second.diff_allocations += elems;
      finder->second.diff_bytes += alloc_size;
    }

    //--------------------------------------------------------------------------
    void Internal::trace_free(AllocationType type, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(allocation_lock);
      std::map<AllocationType,AllocationTracker>::iterator finder = 
        allocation_manager.find(type);
      size_t free_size = size * elems;
      finder->second.total_allocations -= elems;
      finder->second.total_bytes -= free_size;
      finder->second.diff_allocations -= elems;
      finder->second.diff_bytes -= free_size;
    }

    //--------------------------------------------------------------------------
    void Internal::dump_allocation_info(void)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(allocation_lock);
      for (std::map<AllocationType,AllocationTracker>::iterator it = 
            allocation_manager.begin(); it != allocation_manager.end(); it++)
      {
        // Skip anything that is empty
        if (it->second.total_allocations == 0)
          continue;
        // Skip anything that hasn't changed
        if (it->second.diff_allocations == 0)
          continue;
        log_allocation.info("%s on %d: "
            "total=%d total_bytes=%ld diff=%d diff_bytes=%ld",
            get_allocation_name(it->first), address_space,
            it->second.total_allocations, it->second.total_bytes,
            it->second.diff_allocations, it->second.diff_bytes);
        it->second.diff_allocations = 0;
        it->second.diff_bytes = 0;
      }
      log_allocation.info(" ");
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* Internal::get_allocation_name(AllocationType type)
    //--------------------------------------------------------------------------
    {
      switch (type)
      {
        case ARGUMENT_MAP_ALLOC:
          return "Argument Map";
        case ARGUMENT_MAP_STORE_ALLOC:
          return "Argument Map Store";
        case STORE_ARGUMENT_ALLOC:
          return "Store Argument";
        case MPI_HANDSHAKE_ALLOC:
          return "MPI Handshake";
        case GRANT_ALLOC:
          return "Grant";
        case FUTURE_ALLOC:
          return "Future";
        case FUTURE_MAP_ALLOC:
          return "Future Map";
        case PHYSICAL_REGION_ALLOC:
          return "Physical Region";
        case TRACE_ALLOC:
          return "Trace";
        case ALLOC_MANAGER_ALLOC:
          return "Allocation Manager";
        case ALLOC_INTERNAL_ALLOC:
          return "Allocation Internal";
        case TASK_ARGS_ALLOC:
          return "Task Arguments";
        case LOCAL_ARGS_ALLOC:
          return "Local Arguments";
        case REDUCTION_ALLOC:
          return "Reduction Result"; 
        case PREDICATE_ALLOC:
          return "Default Predicate";
        case FUTURE_RESULT_ALLOC:
          return "Future Result";
        case INSTANCE_MANAGER_ALLOC:
          return "Instance Manager";
        case LIST_MANAGER_ALLOC:
          return "List Reduction Manager";
        case FOLD_MANAGER_ALLOC:
          return "Fold Reduction Manager";
        case COMPOSITE_NODE_ALLOC:
          return "Composite Node";
        case TREE_CLOSE_ALLOC:
          return "Tree Close List";
        case TREE_CLOSE_IMPL_ALLOC:
          return "Tree Close Impl";
        case MATERIALIZED_VIEW_ALLOC:
          return "Materialized View";
        case REDUCTION_VIEW_ALLOC:
          return "Reduction View";
        case COMPOSITE_VIEW_ALLOC:
          return "Composite View";
        case FILL_VIEW_ALLOC:
          return "Fill View";
        case INDIVIDUAL_TASK_ALLOC:
          return "Individual Task";
        case POINT_TASK_ALLOC:
          return "Point Task";
        case INDEX_TASK_ALLOC:
          return "Index Task";
        case SLICE_TASK_ALLOC:
          return "Slice Task";
        case REMOTE_TASK_ALLOC:
          return "Remote Task";
        case INLINE_TASK_ALLOC:
          return "Inline Task";
        case MAP_OP_ALLOC:
          return "Map Op";
        case COPY_OP_ALLOC:
          return "Copy Op";
        case FENCE_OP_ALLOC:
          return "Fence Op";
        case FRAME_OP_ALLOC:
          return "Frame Op";
        case DELETION_OP_ALLOC:
          return "Deletion Op";
        case CLOSE_OP_ALLOC:
          return "Close Op";
        case DYNAMIC_COLLECTIVE_OP_ALLOC:
          return "Dynamic Collective Op";
        case FUTURE_PRED_OP_ALLOC:
          return "Future Pred Op";
        case NOT_PRED_OP_ALLOC:
          return "Not Pred Op";
        case AND_PRED_OP_ALLOC:
          return "And Pred Op";
        case OR_PRED_OP_ALLOC:
          return "Or Pred Op";
        case ACQUIRE_OP_ALLOC:
          return "Acquire Op";
        case RELEASE_OP_ALLOC:
          return "Release Op";
        case TRACE_CAPTURE_OP_ALLOC:
          return "Trace Capture Op";
        case TRACE_COMPLETE_OP_ALLOC:
          return "Trace Complete Op";
        case MUST_EPOCH_OP_ALLOC:
          return "Must Epoch Op";
        case PENDING_PARTITION_OP_ALLOC:
          return "Pending Partition Op";
        case DEPENDENT_PARTITION_OP_ALLOC:
          return "Dependent Partition Op";
        case FILL_OP_ALLOC:
          return "Fill Op";
        case ATTACH_OP_ALLOC:
          return "Attach Op";
        case DETACH_OP_ALLOC:
          return "Detach Op";
        case MESSAGE_BUFFER_ALLOC:
          return "Message Buffer";
        case EXECUTING_CHILD_ALLOC:
          return "Executing Children";
        case EXECUTED_CHILD_ALLOC:
          return "Executed Children";
        case COMPLETE_CHILD_ALLOC:
          return "Complete Children";
        case PHYSICAL_MANAGER_ALLOC:
          return "Physical Managers";
        case LOGICAL_VIEW_ALLOC:
          return "Logical Views";
        case LOGICAL_FIELD_VERSIONS_ALLOC:
          return "Logical Field Versions";
        case LOGICAL_FIELD_STATE_ALLOC:
          return "Logical Field States";
        case CURR_LOGICAL_ALLOC:
          return "Current Logical Users";
        case PREV_LOGICAL_ALLOC:
          return "Previous Logical Users";
        case VERSION_ID_ALLOC:
          return "Version IDs";
        case LOGICAL_REC_ALLOC:
          return "Recorded Logical Users";
        case CLOSE_LOGICAL_ALLOC:
          return "Close Logical Users";
        case VALID_VIEW_ALLOC:
          return "Valid Instance Views";
        case VALID_REDUCTION_ALLOC:
          return "Valid Reduction Views";
        case PENDING_UPDATES_ALLOC:
          return "Pending Updates";
        case LAYOUT_DESCRIPTION_ALLOC:
          return "Layout Description";
        case PHYSICAL_USER_ALLOC:
          return "Physical Users";
        case PHYSICAL_VERSION_ALLOC:
          return "Physical Versions";
        case MEMORY_INSTANCES_ALLOC:
          return "Memory Manager Instances";
        case MEMORY_REDUCTION_ALLOC:
          return "Memory Manager Reductions";
        case MEMORY_AVAILABLE_ALLOC:
          return "Memory Manager Available";
        case PROCESSOR_GROUP_ALLOC:
          return "Processor Groups";
        case RUNTIME_DISTRIBUTED_ALLOC:
          return "Runtime Distributed IDs";
        case RUNTIME_DIST_COLLECT_ALLOC:
          return "Distributed Collectables";
        case RUNTIME_GC_EPOCH_ALLOC:
          return "Runtime Garbage Collection Epochs";
        case RUNTIME_FUTURE_ALLOC:
          return "Runtime Futures";
        case RUNTIME_REMOTE_ALLOC:
          return "Runtime Remote Contexts";
        case TASK_INSTANCE_REGION_ALLOC:
          return "Task Physical Instances";
        case TASK_LOCAL_REGION_ALLOC:
          return "Task Local Regions";
        case TASK_INLINE_REGION_ALLOC:
          return "Task Inline Regions";
        case TASK_TRACES_ALLOC:
          return "Task Traces";
        case TASK_RESERVATION_ALLOC:
          return "Task Reservations";
        case TASK_BARRIER_ALLOC:
          return "Task Barriers";
        case TASK_LOCAL_FIELD_ALLOC:
          return "Task Local Fields";
        case TASK_INLINE_ALLOC:
          return "Task Inline Tasks";
        case SEMANTIC_INFO_ALLOC:
          return "Semantic Information";
        case DIRECTORY_ALLOC:
          return "State Directory";
        case DENSE_INDEX_ALLOC:
          return "Dense Index Set";
        case CURRENT_STATE_ALLOC:
          return "Current State";
        case PHYSICAL_STATE_ALLOC:
          return "Physical State";
        case VERSION_STATE_ALLOC:
          return "Version State";
        case TASK_IMPL_ALLOC:
          return "Task Implementation";
        case VARIANT_IMPL_ALLOC:
          return "Variant Implementation";
        case LAYOUT_CONSTRAINTS_ALLOC:
          return "Layout Constraints";
        default:
          assert(false); // should never get here
      }
      return NULL;
    }
#endif

#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
    //--------------------------------------------------------------------------
    void Internal::print_out_individual_tasks(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Build a map of the tasks based on their task IDs
      // so we can print them out in the order that they were created.
      // No need to hold the lock because we'll only ever call this
      // in the debugger.
      std::map<UniqueID,IndividualTask*> out_tasks;
      for (std::set<IndividualTask*>::const_iterator it = 
            out_individual_tasks.begin(); it !=
            out_individual_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::map<UniqueID,IndividualTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Individual Task %lld: %p %s (" IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
      }
      fflush(f);
    }

    //--------------------------------------------------------------------------
    void Internal::print_out_index_tasks(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Build a map of the tasks based on their task IDs
      // so we can print them out in the order that they were created.
      // No need to hold the lock because we'll only ever call this
      // in the debugger.
      std::map<UniqueID,IndexTask*> out_tasks;
      for (std::set<IndexTask*>::const_iterator it = 
            out_index_tasks.begin(); it !=
            out_index_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::map<UniqueID,IndexTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Index Task %lld: %p %s (" IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
      }
      fflush(f);
    }

    //--------------------------------------------------------------------------
    void Internal::print_out_slice_tasks(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Build a map of the tasks based on their task IDs
      // so we can print them out in the order that they were created.
      // No need to hold the lock because we'll only ever call this
      // in the debugger.
      std::map<UniqueID,SliceTask*> out_tasks;
      for (std::set<SliceTask*>::const_iterator it = 
            out_slice_tasks.begin(); it !=
            out_slice_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::map<UniqueID,SliceTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Slice Task %lld: %p %s (" IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
      }
      fflush(f);
    }

    //--------------------------------------------------------------------------
    void Internal::print_out_point_tasks(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      // Build a map of the tasks based on their task IDs
      // so we can print them out in the order that they were created.
      // No need to hold the lock because we'll only ever call this
      // in the debugger.
      std::map<UniqueID,PointTask*> out_tasks;
      for (std::set<PointTask*>::const_iterator it = 
            out_point_tasks.begin(); it !=
            out_point_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::map<UniqueID,PointTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Point Task %lld: %p %s (" IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
      }
      fflush(f);
    }

    //--------------------------------------------------------------------------
    void Internal::print_outstanding_tasks(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      std::map<UniqueID,TaskOp*> out_tasks;
      for (std::set<IndividualTask*>::const_iterator it = 
            out_individual_tasks.begin(); it !=
            out_individual_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::set<IndexTask*>::const_iterator it = 
            out_index_tasks.begin(); it !=
            out_index_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::set<SliceTask*>::const_iterator it = 
            out_slice_tasks.begin(); it !=
            out_slice_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::set<PointTask*>::const_iterator it = 
            out_point_tasks.begin(); it !=
            out_point_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_task_id()] = *it;
      }
      for (std::map<UniqueID,TaskOp*>::const_iterator it = 
            out_tasks.begin(); it != out_tasks.end(); it++)
      {
        Event completion = it->second->get_completion_event();
        switch (it->second->get_task_kind())
        {
          case TaskOp::INDIVIDUAL_TASK_KIND:
            {
              fprintf(f,"Outstanding Individual Task %lld: %p %s (" 
                        IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen);
              break;
            }
          case TaskOp::POINT_TASK_KIND:
            {
              fprintf(f,"Outstanding Point Task %lld: %p %s (" 
                        IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen);
              break;
            }
          case TaskOp::INDEX_TASK_KIND:
            {
              fprintf(f,"Outstanding Index Task %lld: %p %s (" 
                        IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen);
              break;
            }
          case TaskOp::SLICE_TASK_KIND:
            {
              fprintf(f,"Outstanding Slice Task %lld: %p %s (" 
                        IDFMT ",%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen);
              break;
            }
          default:
            assert(false);
        }
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
      }
      fflush(f);
    }
#endif

    //--------------------------------------------------------------------------
    LayoutConstraintID Internal::register_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      if (layout_id == AUTO_GENERATE_ID)
        layout_id = get_unique_constraint_id();
      // Now make our entry and then return the result
      LayoutConstraints *constraints = 
        legion_new<LayoutConstraints>(layout_id, this, registrar);
      AutoLock l_lock(layout_constraints_lock);
      if (layout_constraints_table.find(layout_id) != 
          layout_constraints_table.end())
      {
        log_run.error("Illegal duplicate use of layout constraint ID %ld", 
                      layout_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_DUPLICATE_CONSTRAINT_ID);
      }
      layout_constraints_table[layout_id] = constraints;
      return layout_id;
    }

    //--------------------------------------------------------------------------
    void Internal::release_layout(LayoutConstraintID layout_id,
                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      constraints->release_constraints(source);
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID,LayoutConstraints*>::iterator
          finder = layout_constraints_table.find(layout_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != layout_constraints_table.end());
#endif
        layout_constraints_table.erase(finder);
      }
      legion_delete(constraints);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID Internal::preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    { 
      if (runtime_started)
      {
        log_run.error("Illegal call to 'preregister_layout' after "
                      "the runtime has started!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_STATIC_CALL_POST_RUNTIME_START);
      }
      std::map<LayoutConstraintID,LayoutConstraintRegistrar> 
        &pending_constraints = get_pending_constraint_table();
      // See if we have to generate an ID
      if (layout_id == AUTO_GENERATE_ID)
      {
        // Find the first available layout ID
        layout_id = 1;
        for (std::map<LayoutConstraintID,LayoutConstraintRegistrar>::
              const_iterator it = pending_constraints.begin(); 
              it != pending_constraints.end(); it++)
        {
          if (layout_id != it->first)
          {
            // We've found a free one, so we can use it
            break;
          }
          else
            layout_id++;
        }
      }
      else
      {
        if (layout_id == 0)
        {
          log_run.error("Illegal use of reserved constraint ID 0");
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RESERVED_CONSTRAINT_ID);
        }
        // Check to make sure it is not already used
        std::map<LayoutConstraintID,LayoutConstraintRegistrar>::const_iterator
          finder = pending_constraints.find(layout_id);
        if (finder != pending_constraints.end())
        {
          log_run.error("Duplicate use of constraint ID %ld", layout_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_DUPLICATE_CONSTRAINT_ID);
        }
      }
      pending_constraints[layout_id] = registrar;
      return layout_id;
    }

    //--------------------------------------------------------------------------
    FieldSpace Internal::get_layout_constraint_field_space(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      return constraints->get_field_space();
    }

    //--------------------------------------------------------------------------
    void Internal::get_layout_constraints(LayoutConstraintID layout_id,
                                        LayoutConstraintSet &layout_constraints)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      layout_constraints = *constraints;
    }

    //--------------------------------------------------------------------------
    const char* Internal::get_layout_constraints_name(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      return constraints->get_name();
    }

    //--------------------------------------------------------------------------
    LayoutConstraints* Internal::find_layout_constraints(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      // See if we can find it first
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
          finder = layout_constraints_table.find(layout_id);
        if (finder != layout_constraints_table.end())
        {
          wait_on = finder->second->get_ready_event();
          if (wait_on.has_triggered())
            return finder->second;
        }
        else // make a place holder which will request the actual constraints
        {
          LayoutConstraints *constraints = 
            legion_new<LayoutConstraints>(layout_id, this);
          layout_constraints_table[layout_id] = constraints; 
          wait_on = constraints->get_ready_event();
        }
      }
      // If we didn't find it send a remote request for the constraints
      wait_on.wait();
      // When we wake up, the result should be there
      AutoLock l_lock(layout_constraints_lock);
      std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
          finder = layout_constraints_table.find(layout_id);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != layout_constraints_table.end());
#endif
      return finder->second;
    }

    /*static*/ Internal *Internal::runtime_map[(MAX_NUM_PROCS+1)];
    /*static*/ volatile RegistrationCallbackFnptr Internal::
                                              registration_callback = NULL;
    /*static*/ Processor::TaskFuncID Internal::legion_main_id = 0;
    /*static*/ int Internal::initial_task_window_size = 
                                      DEFAULT_MAX_TASK_WINDOW;
    /*static*/ unsigned Internal::initial_task_window_hysteresis =
                                      DEFAULT_TASK_WINDOW_HYSTERESIS;
    /*static*/ unsigned Internal::initial_tasks_to_schedule = 
                                      DEFAULT_MIN_TASKS_TO_SCHEDULE;
    /*static*/ unsigned Internal::superscalar_width = 
                                      DEFAULT_SUPERSCALAR_WIDTH;
    /*static*/ unsigned Internal::max_message_size = 
                                      DEFAULT_MAX_MESSAGE_SIZE;
    /*static*/ unsigned Internal::gc_epoch_size = 
                                      DEFAULT_GC_EPOCH_SIZE;
    /*static*/ bool Internal::runtime_started = false;
    /*static*/ bool Internal::runtime_backgrounded = false;
    /*static*/ bool Internal::separate_runtime_instances = false;
    /*static*/ bool Internal::record_registration = false;
    /*sattic*/ bool Internal::stealing_disabled = false;
    /*static*/ bool Internal::resilient_mode = false;
    /*static*/ bool Internal::unsafe_launch = false;
    /*static*/ bool Internal::dynamic_independence_tests = true;
    /*static*/ bool Internal::legion_spy_enabled = false;
    /*static*/ int Internal::mpi_rank = -1;
    /*static*/ unsigned Internal::mpi_rank_table[MAX_NUM_NODES];
    /*static*/ unsigned Internal::remaining_mpi_notifications = 0;
    /*static*/ UserEvent Internal::mpi_rank_event = UserEvent::NO_USER_EVENT;
#ifdef INORDER_EXECUTION
    /*static*/ bool Internal::program_order_execution = true;
#endif
#ifdef DEBUG_HIGH_LEVEL
    /*static*/ bool Internal::logging_region_tree_state = false;
    /*static*/ bool Internal::verbose_logging = false;
    /*static*/ bool Internal::logical_logging_only = false;
    /*static*/ bool Internal::physical_logging_only = false;
    /*static*/ bool Internal::check_privileges = true;
    /*static*/ bool Internal::verify_disjointness = false;
    /*static*/ bool Internal::bit_mask_logging = false;
#endif
#ifdef DEBUG_PERF
    /*static*/ unsigned long long Internal::perf_trace_tolerance = 10000; 
#endif
    /*static*/ unsigned Internal::num_profiling_nodes = 0;

#ifdef HANG_TRACE
    //--------------------------------------------------------------------------
    static void catch_hang(int signal)
    //--------------------------------------------------------------------------
    {
      assert(signal == SIGTERM);
      static int call_count = 0;
      int count = __sync_fetch_and_add(&call_count, 0);
      if (count == 0)
      {
        Internal *rt = Internal::runtime_map[1];
        const char *prefix = "";
        char file_name[1024];
        sprintf(file_name,"%strace_%d.txt", prefix, rt->address_space);
        FILE *target = fopen(file_name,"w");
        //rt->dump_processor_states(target);
        //rt->print_outstanding_tasks(target);
        rt->print_out_acquire_ops(target);
        fclose(target);
      }
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ int Internal::start(int argc, char **argv, bool background)
    //--------------------------------------------------------------------------
    {
      // Some static asserts that need to hold true for the runtime to work
      LEGION_STATIC_ASSERT(MAX_RETURN_SIZE > 0);
      LEGION_STATIC_ASSERT((1 << LEGION_FIELD_LOG2) == MAX_FIELDS);
      LEGION_STATIC_ASSERT(MAX_NUM_NODES > 0);
      LEGION_STATIC_ASSERT(MAX_NUM_PROCS > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_TASK_WINDOW > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MIN_TASKS_TO_SCHEDULE > 0);
      LEGION_STATIC_ASSERT(DEFAULT_SUPERSCALAR_WIDTH > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_MESSAGE_SIZE > 0);
      // Once we've made this call, the Legion runtime is started
      runtime_started = true;
      // Need to pass argc and argv to low-level runtime before we can record 
      // their values as they might be changed by GASNet or MPI or whatever.
      // Note that the logger isn't initialized until after this call returns 
      // which means any logging that occurs before this has undefined behavior.
      RealmRuntime realm;
#ifndef NDEBUG
      bool ok = 
#endif
        realm.init(&argc, &argv);
      assert(ok);
      
      {
	const ReductionOpTable& red_table = get_reduction_table();
	for(ReductionOpTable::const_iterator it = red_table.begin();
	    it != red_table.end();
	    it++)
	  realm.register_reduction(it->first, it->second);

        const SerdezOpTable &serdez_table = get_serdez_table();
        for (SerdezOpTable::const_iterator it = serdez_table.begin();
              it != serdez_table.end(); it++)
          realm.register_custom_serdez(it->first, it->second);
      }
      
      // Parse any inputs for the high level runtime
      {
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)

        // Set these values here before parsing the input arguments
        // so that we don't need to trust the C runtime to do 
        // static initialization properly (always risky).
        separate_runtime_instances = false;
        record_registration = false;
        stealing_disabled = false;
        resilient_mode = false;
        unsafe_launch = false;
        // We always turn this on as the Legion Spy will 
        // now understand how to handle it.
        dynamic_independence_tests = true;
#ifdef LEGION_SPY
        legion_spy_enabled = true;
#else
        legion_spy_enabled = false;
#endif
        initial_task_window_size = DEFAULT_MAX_TASK_WINDOW;
        initial_task_window_hysteresis = DEFAULT_TASK_WINDOW_HYSTERESIS;
        initial_tasks_to_schedule = DEFAULT_MIN_TASKS_TO_SCHEDULE;
        superscalar_width = DEFAULT_SUPERSCALAR_WIDTH;
        max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
        gc_epoch_size = DEFAULT_GC_EPOCH_SIZE;
#ifdef INORDER_EXECUTION
        program_order_execution = true;
#endif
        num_profiling_nodes = 0;
#ifdef DEBUG_HIGH_LEVEL
        logging_region_tree_state = false;
        verbose_logging = false;
        logical_logging_only = false;
        physical_logging_only = false;
        check_privileges = true;
        verify_disjointness = false;
        bit_mask_logging = false;
#endif
        for (int i = 1; i < argc; i++)
        {
          BOOL_ARG("-hl:separate",separate_runtime_instances);
          BOOL_ARG("-hl:registration",record_registration);
          BOOL_ARG("-hl:nosteal",stealing_disabled);
          BOOL_ARG("-hl:resilient",resilient_mode);
          BOOL_ARG("-hl:unsafe_launch",unsafe_launch);
#ifdef INORDER_EXECUTION
          if (!strcmp(argv[i],"-hl:outorder"))
            program_order_execution = false;
#endif
          INT_ARG("-hl:window", initial_task_window_size);
          INT_ARG("-hl:hysteresis", initial_task_window_hysteresis);
          INT_ARG("-hl:sched", initial_tasks_to_schedule);
          INT_ARG("-hl:width", superscalar_width);
          INT_ARG("-hl:message",max_message_size);
          INT_ARG("-hl:epoch", gc_epoch_size);
          if (!strcmp(argv[i],"-hl:no_dyn"))
            dynamic_independence_tests = false;
          BOOL_ARG("-hl:spy",legion_spy_enabled);
#ifdef DEBUG_HIGH_LEVEL
          BOOL_ARG("-hl:tree",logging_region_tree_state);
          BOOL_ARG("-hl:verbose",verbose_logging);
          BOOL_ARG("-hl:logical_only",logical_logging_only);
          BOOL_ARG("-hl:physical_only",physical_logging_only);
          BOOL_ARG("-hl:disjointness",verify_disjointness);
          BOOL_ARG("-hl:bit_masks",bit_mask_logging);
#else
          if (!strcmp(argv[i],"-hl:tree"))
          {
            log_run.warning("WARNING: Region tree state logging is "
                          "disabled.  To enable region tree state logging "
                                                  "compile in debug mode.");
          }
          if (!strcmp(argv[i],"-hl:disjointness"))
          {
            log_run.warning("WARNING: Disjointness verification for "
                      "partition creation is disabled.  To enable dynamic "
                              "disjointness testing compile in debug mode.");
          }
#endif
#ifdef DEBUG_PERF
          INT_ARG("-hl:perf_tol", perf_trace_tolerance);
#endif
          INT_ARG("-hl:prof", num_profiling_nodes);
        }
#undef INT_ARG
#undef BOOL_ARG
#ifdef DEBUG_HIGH_LEVEL
        assert(initial_task_window_hysteresis <= 100);
#endif
      }
#ifdef DEBUG_HIGH_LEVEL
      if (num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"!!! YOU ARE PROFILING IN DEBUG MODE           !!!\n");
        fprintf(stderr,"!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr,"!!! COMPILE WITH DEBUG=0 FOR PROFILING        !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
#ifdef LEGION_SPY
      if (num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"!!! YOU ARE PROFILING WITH LegionSpy ENABLED  !!!\n");
        fprintf(stderr,"!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr,"!!! COMPILE WITHOUT -DLEGION_SPY FOR PROFILING!!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#else
      if (legion_spy_enabled && (num_profiling_nodes > 0))
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"!!! YOU ARE PROFILING WITH LegionSpy ENABLED  !!!\n");
        fprintf(stderr,"!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr,"!!! RUN WITHOUT -hl:spy flag FOR PROFILING    !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
#ifdef BOUNDS_CHECKS
      if (num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"!!! YOU ARE PROFILING WITH BOUNDS_CHECKS      !!!\n");
        fprintf(stderr,"!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr,"!!! PLEASE COMPILE WITHOUT BOUNDS_CHECKS      !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
#ifdef PRIVILEGE_CHECKS
      if (num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"!!! YOU ARE PROFILING WITH PRIVILEGE_CHECKS    !!\n");
        fprintf(stderr,"!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr,"!!! PLEASE COMPILE WITHOUT PRIVILEGE_CHECKS   !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(stderr,"!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
      // Now we can set out input args
      Internal::get_input_args().argv = argv;
      Internal::get_input_args().argc = argc;
#ifdef HANG_TRACE
      signal(SIGTERM, catch_hang); 
#endif
      // For the moment, we only need to register our runtime tasks
      // We'll register everything else once the Legion runtime starts
      Event tasks_registered = register_runtime_tasks(realm);
      // Do some error checking in case we are running with separate instances
      Machine machine = Machine::get_machine();
      if (separate_runtime_instances)
      {
#ifdef TRACE_ALLOCATION
        log_run.error("Memory tracing not supported with "
                      "separate runtime instances.");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_TRACING_ALLOCATION_WITH_SEPARATE);
#endif
        // Check for utility processors
        std::set<Processor> util_procs;
        machine.get_local_processors_by_kind(util_procs, Processor::UTIL_PROC);
        if (!util_procs.empty())
        {
          log_run.error("Separate runtime instances are not "
                        "supported when running with explicit "
                        "utility processors");
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_SEPARATE_UTILITY_PROCS);
        }
      }
      // Check for exceeding the local number of processors
      // and also see if we are supposed to launch the top-level task
      Processor top_level_proc = Processor::NO_PROC;
      {
        std::set<Processor> local_procs;
        machine.get_local_processors(local_procs);
        if (local_procs.size() > MAX_NUM_PROCS)
        {
          log_run.error("Maximum number of local processors %ld exceeds "
                        "compile time maximum of %d.  Change the value "
                        "in legion_config.h and recompile.",
                        local_procs.size(), MAX_NUM_PROCS);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_MAXIMUM_PROCS_EXCEEDED);
        }
        AddressSpace local_space = local_procs.begin()->address_space();
        // If we are node 0 then we have to launch the top-level task
        if (local_space == 0)
        {
          // Find the first CPU processor
          for (std::set<Processor>::const_iterator it = local_procs.begin();
                it != local_procs.end(); it++)
          {
            Processor::Kind kind = it->kind(); 
            if (kind == Processor::LOC_PROC)
            {
              top_level_proc = *it;
              break;
            }
          }
          // If we didn't find one that is very bad
          if (!top_level_proc.exists())
          {
            log_run.error("Machine model contains no CPU processors!");
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_NO_PROCESSORS);
          }
        }
      }
      // Now perform a collective spawn to initialize the runtime everywhere
      // Save the precondition in case we are the node that needs to start
      // the top-level task.
      // If we're doing separate runtime instances we need to launch an
      // init task on every processor on all nodes, otherwise we just
      // need to launch one task on a CPU processor on every node
      Event runtime_startup_event = realm.collective_spawn_by_kind(
          (separate_runtime_instances ? Processor::NO_KIND : 
           Processor::LOC_PROC), INIT_TASK_ID, NULL, 0,
          !separate_runtime_instances, tasks_registered);
      // See if we are supposed to start the top-level task
      if (top_level_proc.exists())
      {
        Realm::ProfilingRequestSet empty_requests;
        top_level_proc.spawn(HLR_LAUNCH_TOP_LEVEL_ID, NULL, 0,
                             empty_requests, runtime_startup_event);
      }
      // If we are supposed to background this thread, then we wait
      // for the runtime to shutdown, otherwise we can now return
      if (background) // Record that the runtime was backgrounded
        runtime_backgrounded = true;
      else // Otherwise wait for realm to be shutdown
        realm.wait_for_shutdown();
      return 0;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      if (!runtime_backgrounded)
      {
        log_run.error("Illegal call to wait_for_shutdown when runtime was "
                      "not launched in background mode!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_WAIT_FOR_SHUTDOWN);
      }
      RealmRuntime::get_runtime().wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::set_top_level_task_id(
                                                  Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::configure_MPI_interoperability(int rank)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(rank >= 0);
#endif
      mpi_rank = rank;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Internal::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        log_run.error("ERROR: ReductionOpID zero is reserved.");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_RESERVED_REDOP_ID);
      }
      ReductionOpTable &red_table = Internal::get_reduction_table();
#ifdef DEBUG_HIGH_LEVEL
      if (red_table.find(redop_id) == red_table.end())
      {
        log_run.error("Invalid ReductionOpID %d",redop_id);
        assert(false);
        exit(ERROR_INVALID_REDOP_ID);
      }
#endif
      return red_table[redop_id];
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezOp* Internal::get_serdez_op(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      if (serdez_id == 0)
      {
        log_run.error("ERROR: CustomSerdezID zero is reserved.");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_RESERVED_SERDEZ_ID);
      }
      SerdezOpTable &serdez_table = Internal::get_serdez_table();
#ifdef DEBUG_HIGH_LEVEL
      if (serdez_table.find(serdez_id) == serdez_table.end())
      {
        log_run.error("Invalid CustomSerdezOpID %d", serdez_id);
        assert(false);
        exit(ERROR_INVALID_SERDEZ_ID);
      }
#endif
      return serdez_table[serdez_id];
    }

    /*static*/ const SerdezRedopFns* Internal::get_serdez_redop_fns(
                                                         ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      SerdezRedopTable &serdez_table = get_serdez_redop_table(); 
      SerdezRedopTable::const_iterator finder = serdez_table.find(redop_id);
      if (finder != serdez_table.end())
        return &(finder->second);
      return NULL;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::set_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      registration_callback = callback;
    }

    //--------------------------------------------------------------------------
    /*static*/ InputArgs& Internal::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      static InputArgs inputs = { NULL, 0 };
      return inputs;
    }

    //--------------------------------------------------------------------------
    /*static*/ Internal* Internal::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.local_id() < (MAX_NUM_PROCS+1));
#endif
      return runtime_map[p.local_id()];
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpTable& Internal::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      static ReductionOpTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezOpTable& Internal::get_serdez_table(void)
    //--------------------------------------------------------------------------
    {
      static SerdezOpTable table;
      return table;
    }
    
    //--------------------------------------------------------------------------
    /*static*/ SerdezRedopTable& Internal::get_serdez_redop_table(void)
    //--------------------------------------------------------------------------
    {
      static SerdezRedopTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Internal::register_region_projection_function(
                                          ProjectionID handle, void *func_ptr)
    //--------------------------------------------------------------------------
    {
      if (handle == 0)
      {
        log_run.error("ERROR: ProjectionID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      RegionProjectionTable &proj_table = 
                          Internal::get_region_projection_table();
      if (proj_table.find(handle) != proj_table.end())
      {
        log_run.error("ERROR: ProjectionID %d has already been used in "
                                    "the region projection table\n",handle);
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_DUPLICATE_PROJECTION_ID);
      }
      if (handle == AUTO_GENERATE_ID)
      {
        for (ProjectionID idx = 1; idx < AUTO_GENERATE_ID; idx++)
        {
          if (proj_table.find(idx) == proj_table.end())
          {
            handle = idx;
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        // We should never run out of type handles
        assert(handle != AUTO_GENERATE_ID);
#endif
      }
      proj_table[handle] = (RegionProjectionFnptr)func_ptr;  
      return handle;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Internal::
      register_partition_projection_function(ProjectionID handle, 
                                             void *func_ptr)
    //--------------------------------------------------------------------------
    {
      if (handle == 0)
      {
        log_run.error("ERROR: ProjectionID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      PartitionProjectionTable &proj_table = 
                              Internal::get_partition_projection_table();
      if (proj_table.find(handle) != proj_table.end())
      {
        log_run.error("ERROR: ProjectionID %d has already been used in "
                            "the partition projection table\n",handle);
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_DUPLICATE_PROJECTION_ID);
      }
      if (handle == AUTO_GENERATE_ID)
      {
        for (ProjectionID idx = 1; idx < AUTO_GENERATE_ID; idx++)
        {
          if (proj_table.find(idx) == proj_table.end())
          {
            handle = idx;
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        // We should never run out of type handles
        assert(handle != AUTO_GENERATE_ID);
#endif
      }
      proj_table[handle] = (PartitionProjectionFnptr)func_ptr;  
      return handle;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::deque<PendingVariantRegistration*>& 
                                       Internal::get_pending_variant_table(void)
    //--------------------------------------------------------------------------
    {
      static std::deque<PendingVariantRegistration*> pending_variant_table;
      return pending_variant_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<LayoutConstraintID,LayoutConstraintRegistrar>&
                                    Internal::get_pending_constraint_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<LayoutConstraintID,LayoutConstraintRegistrar>
                                                    pending_constraint_table;
      return pending_constraint_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID& Internal::get_current_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      static TaskID current_task_id = MAX_APPLICATION_TASK_ID;
      return current_task_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Internal::generate_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      TaskID &next_task = get_current_static_task_id(); 
      if (runtime_started)
      {
        log_run.error("Illegal call to 'generate_static_task_id' after "
                      "the runtime has been started!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_STATIC_CALL_POST_RUNTIME_START);
      }
      return next_task++;
    }

    //--------------------------------------------------------------------------
    /*static*/ VariantID Internal::preregister_variant(
                          const TaskVariantRegistrar &registrar,
                          const void *user_data, size_t user_data_size,
                          CodeDescriptor *realm,
                          bool has_ret, const char *task_name, bool check_id)
    //--------------------------------------------------------------------------
    {
      // Report an error if the runtime has already started
      if (runtime_started)
      {
        log_run.error("Illegal call to 'preregister_task_variant' after "
                      "the runtime has been started!");
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_STATIC_CALL_POST_RUNTIME_START);
      }
      if (check_id && (registrar.task_id >= get_current_static_task_id()))
      {
        log_run.error("Error preregistering task with ID %d. Exceeds the "
                      "statically set bounds on application task IDs of %d. "
                      "See %s in legion_config.h.", 
                      registrar.task_id, MAX_APPLICATION_TASK_ID, 
                      LEGION_MACRO_TO_STRING(MAX_APPLICATION_TASK_ID));
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED);
      }
      std::deque<PendingVariantRegistration*> &pending_table = 
        get_pending_variant_table();
      // Offset by the runtime tasks
      VariantID vid = TASK_ID_AVAILABLE + pending_table.size();
      pending_table.push_back(new PendingVariantRegistration(vid, has_ret,
                              registrar, user_data, user_data_size, 
                              realm, task_name));
      return vid;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionFnptr Internal::
                            find_partition_projection_function(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      const PartitionProjectionTable &table = get_partition_projection_table();
      PartitionProjectionTable::const_iterator finder = table.find(pid);
      if (finder == table.end())
      {
        log_run.error("Unable to find registered partition "
                            "projection ID %d", pid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PROJECTION_ID);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionProjectionFnptr Internal::find_region_projection_function(
                                                              ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      const RegionProjectionTable &table = get_region_projection_table();
      RegionProjectionTable::const_iterator finder = table.find(pid);
      if (finder == table.end())
      {
        log_run.error("Unable to find registered region projection "
                            "ID %d", pid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PROJECTION_ID);
      }
      return finder->second;
    }

#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    //--------------------------------------------------------------------------
    /*static*/ const char* Internal::find_privilege_task_name(void *impl)
    //--------------------------------------------------------------------------
    {
      PhysicalRegion::Impl *region = static_cast<PhysicalRegion::Impl*>(impl);
      return region->get_task_name();
    }
#endif

#ifdef BOUNDS_CHECKS
    //--------------------------------------------------------------------------
    /*static*/ void Internal::check_bounds(void *impl, ptr_t ptr)
    //--------------------------------------------------------------------------
    {
      PhysicalRegion::Impl *region = static_cast<PhysicalRegion::Impl*>(impl);
      if (!region->contains_ptr(ptr))
      {
        fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                       "pointer %d\n", region->get_task_name(), ptr.value);
        assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::check_bounds(void *impl, const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      PhysicalRegion::Impl *region = static_cast<PhysicalRegion::Impl*>(impl);
      if (!region->contains_point(dp))
      {
        switch(dp.get_dim())
        {
          case 1:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                           "1D point (%d)\n", region->get_task_name(),
                            dp.point_data[0]);
            break;
          case 2:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                           "2D point (%d,%d)\n", region->get_task_name(),
                            dp.point_data[0], dp.point_data[1]);
            break;
          case 3:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                           "3D point (%d,%d,%d)\n", region->get_task_name(),
                      dp.point_data[0], dp.point_data[1], dp.point_data[2]);
            break;
          default:
            assert(false);
        }
        assert(false);
      }
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ RegionProjectionTable& Internal::
                                              get_region_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static RegionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionTable& Internal::
                                          get_partition_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static PartitionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ Event Internal::register_runtime_tasks(RealmRuntime &realm)
    //--------------------------------------------------------------------------
    {
      // Make the code descriptors for our tasks
      CodeDescriptor init_task(Internal::initialize_runtime);
      CodeDescriptor shutdown_task(Internal::shutdown_runtime);
      CodeDescriptor hlr_task(Internal::high_level_runtime_task);
      CodeDescriptor rt_profiling_task(Internal::profiling_runtime_task);
      CodeDescriptor map_profiling_task(Internal::profiling_mapper_task);
      CodeDescriptor launch_top_level_task(Internal::launch_top_level);
      Realm::ProfilingRequestSet no_requests;
      // We'll just register these on all the processor kinds
      std::set<Event> registered_events;
      Processor::Kind kinds[4] = { Processor::TOC_PROC, Processor::LOC_PROC,
                                   Processor::UTIL_PROC, Processor::IO_PROC };
      for (unsigned idx = 0; idx < 4; idx++)
      {
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                                         INIT_TASK_ID, init_task, no_requests));
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                                 SHUTDOWN_TASK_ID, shutdown_task, no_requests));
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                                           HLR_TASK_ID, hlr_task, no_requests));
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                      HLR_LEGION_PROFILING_ID, rt_profiling_task, no_requests));
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                     HLR_MAPPER_PROFILING_ID, map_profiling_task, no_requests));
        registered_events.insert(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                  HLR_LAUNCH_TOP_LEVEL_ID, launch_top_level_task, no_requests));
      }
      if (record_registration)
      {
        log_run.print("Legion runtime initialization task "
                            "has Realm ID %d", INIT_TASK_ID);
        log_run.print("Legion runtime shutdown task has "
                            "Realm ID %d", SHUTDOWN_TASK_ID);
        log_run.print("Legion runtime meta-task has Realm ID %d", 
                      HLR_TASK_ID);
        log_run.print("Legion runtime profiling task Realm ID %d",
                      HLR_LEGION_PROFILING_ID);
        log_run.print("Legion mapper profiling task has Realm ID %d",
                      HLR_MAPPER_PROFILING_ID);
        log_run.print("Legion launch top-level task has Realm ID %d",
                      HLR_LAUNCH_TOP_LEVEL_ID);
      }
      return Event::merge_events(registered_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskFuncID Internal::get_next_available_id(void)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskFuncID available = TASK_ID_AVAILABLE;
      return available++;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::log_machine(Machine machine)
    //--------------------------------------------------------------------------
    {
      if (!legion_spy_enabled)
        return;
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      // Log processors
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        Processor::Kind k = it->kind();
        if (k == Processor::UTIL_PROC)
          LegionSpy::log_utility_processor(it->id);
        else
          LegionSpy::log_processor(it->id, k); 
      }
      // Log memories
      std::set<Memory> all_mems;
      machine.get_all_memories(all_mems);
      for (std::set<Memory>::const_iterator it = all_mems.begin();
            it != all_mems.end(); it++)
        LegionSpy::log_memory(it->id, it->capacity(), it->kind());
      // Log Proc-Mem Affinity
      for (std::set<Processor>::const_iterator pit = all_procs.begin();
            pit != all_procs.end(); pit++)
      {
        std::vector<ProcessorMemoryAffinity> affinities;
        machine.get_proc_mem_affinity(affinities, *pit);
        for (std::vector<ProcessorMemoryAffinity>::const_iterator it = 
              affinities.begin(); it != affinities.end(); it++)
        {
          LegionSpy::log_proc_mem_affinity(pit->id, it->m.id, 
                                           it->bandwidth, it->latency);
        }
      }
      // Log Mem-Mem Affinity
      for (std::set<Memory>::const_iterator mit = all_mems.begin();
            mit != all_mems.begin(); mit++)
      {
        std::vector<MemoryMemoryAffinity> affinities;
        machine.get_mem_mem_affinity(affinities, *mit);
        for (std::vector<MemoryMemoryAffinity>::const_iterator it = 
              affinities.begin(); it != affinities.end(); it++)
        {
          LegionSpy::log_mem_mem_affinity(it->m1.id, it->m2.id, 
                                          it->bandwidth, it->latency);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::initialize_runtime(
                                          const void *args, size_t arglen, 
                                          const void *userdata, size_t userlen,
                                          Processor p)
    //--------------------------------------------------------------------------
    {
      // We now know that this task will only get called once for each runtime 
      // instance that is supposed to be created which wasn't always true
      Machine machine = Machine::get_machine();
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      // not having any processors at all is a fatal error
      if (all_procs.empty())
      {
	log_run.error("Machine model contains no processors!");
	assert(false);
	exit(ERROR_NO_PROCESSORS);
      }
      // Compute the data structures necessary for
      // constructing a runtime instance
      std::set<Processor> local_procs;
      std::set<Processor> local_util_procs;
      std::set<AddressSpaceID> address_spaces;
      std::map<Processor,AddressSpaceID> proc_spaces;
      AddressSpaceID local_space_id = 0;
      if (separate_runtime_instances)
      {
        // If we are doing separate runtime instances then each
        // processor effectively gets its own address space
        local_procs.insert(p);
        AddressSpaceID sid = 0;
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++,sid++)
        {
          if (p == (*it))
            local_space_id = sid;
          address_spaces.insert(sid); 
          proc_spaces[*it] = sid;
        }
      }
      else // The normal path
      {
        local_space_id = p.address_space();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          AddressSpaceID sid = it->address_space();
          address_spaces.insert(sid);
          proc_spaces[*it] = sid;
          if (sid == local_space_id)
          {
            if (it->kind() == Processor::UTIL_PROC)
              local_util_procs.insert(*it);
            else
              local_procs.insert(*it);
          }
        }
      }
      // Set up the runtime mask for this instance
      Internal *local_rt = new Internal(machine, local_space_id, 
                                        local_procs, local_util_procs,
                                        address_spaces, proc_spaces);
      // Now set up the runtime on all of the local processors
      // and their utility processors
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        runtime_map[it->local_id()] = local_rt;
      }
      for (std::set<Processor>::const_iterator it = local_util_procs.begin();
            it != local_util_procs.end(); it++)
      {
        runtime_map[it->local_id()] = local_rt;
      }
      // If we have an MPI rank, then build the maps
      if (Internal::mpi_rank >= 0)
        local_rt->construct_mpi_rank_tables(p, Internal::mpi_rank);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::shutdown_runtime(const void *args, size_t arglen, 
                              const void *userdata, size_t userlen, Processor p)
    //--------------------------------------------------------------------------
    {
      // All we need to is delete our runtime instance
      delete get_runtime(p);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::high_level_runtime_task(
                                  const void *args, size_t arglen, 
				  const void *userdata, size_t userlen,
				  Processor p)
    //--------------------------------------------------------------------------
    {
      const char *data = (const char*)args;
      HLRTaskID tid = *((const HLRTaskID*)data);
      data += sizeof(tid);
      arglen -= sizeof(tid);
      switch (tid)
      {
        case HLR_SCHEDULER_ID:
          {
            const ProcessorManager::SchedulerArgs *sched_args = 
              (const ProcessorManager::SchedulerArgs*)args;
            Internal::get_runtime(p)->process_schedule_request(
                                                            sched_args->proc);
            break;
          }
        case HLR_MESSAGE_ID:
          {
            Internal::get_runtime(p)->process_message_task(data, arglen);
            break;
          }
        case HLR_POST_END_ID:
          {
            const SingleTask::PostEndArgs *post_end_args = 
              (const SingleTask::PostEndArgs*)args;
            post_end_args->proxy_this->post_end_task(post_end_args->result, 
                                post_end_args->result_size, true/*owned*/);
            break;
          }
        case HLR_DEFERRED_MAPPING_TRIGGER_ID:
          {
            const Operation::DeferredMappingArgs *deferred_mapping_args = 
              (const Operation::DeferredMappingArgs*)args;
            deferred_mapping_args->proxy_this->trigger_mapping();
            break;
          }
        case HLR_DEFERRED_RESOLUTION_TRIGGER_ID:
          {
            const Operation::DeferredResolutionArgs *deferred_resolution_args =
              (const Operation::DeferredResolutionArgs*)args;
            deferred_resolution_args->proxy_this->trigger_resolution();
            break;
          }
        case HLR_DEFERRED_EXECUTION_TRIGGER_ID:
          {
            const Operation::DeferredExecuteArgs *deferred_execute_args = 
              (const Operation::DeferredExecuteArgs*)args;
            deferred_execute_args->proxy_this->deferred_execute();
            break;
          }
        case HLR_DEFERRED_POST_MAPPED_ID:
          {
            const SingleTask::DeferredPostMappedArgs *post_mapped_args = 
              (const SingleTask::DeferredPostMappedArgs*)args;
            post_mapped_args->task->handle_post_mapped();
            break;
          }
        case HLR_DEFERRED_EXECUTE_ID:
          {
            const Operation::DeferredExecuteArgs *deferred_execute_args = 
              (const Operation::DeferredExecuteArgs*)args;
            deferred_execute_args->proxy_this->complete_execution();
            break;
          }
        case HLR_DEFERRED_COMPLETE_ID:
          {
            const Operation::DeferredCompleteArgs *deferred_complete_args =
              (const Operation::DeferredCompleteArgs*)args;
            deferred_complete_args->proxy_this->trigger_complete();
            break;
          }
        case HLR_DEFERRED_COMMIT_ID:
          {
            const Operation::DeferredCommitArgs *deferred_commit_args = 
              (const Operation::DeferredCommitArgs*)args;
            deferred_commit_args->proxy_this->deferred_commit(
                deferred_commit_args->gen);
            break;
          }
        case HLR_RECLAIM_LOCAL_FIELD_ID:
          {
            Deserializer derez(args, arglen+sizeof(HLRTaskID));
            derez.advance_pointer(sizeof(HLRTaskID));
            DerezCheck z(derez);
            FieldSpace handle;
            derez.deserialize(handle);
            FieldID fid;
            derez.deserialize(fid);
            Internal::get_runtime(p)->finalize_field_destroy(handle, fid);
            break; 
          }
        case HLR_DEFERRED_COLLECT_ID:
          {
            const GarbageCollectionEpoch::GarbageCollectionArgs *collect_args =
              (const GarbageCollectionEpoch::GarbageCollectionArgs*)args;
            bool done = collect_args->epoch->handle_collection(collect_args);
            if (done)
              delete collect_args->epoch;
            break;
          }
        case HLR_TRIGGER_DEPENDENCE_ID:
          {
            const SingleTask::DeferredDependenceArgs *deferred_trigger_args =
              (const SingleTask::DeferredDependenceArgs*)args;
            deferred_trigger_args->op->trigger_dependence_analysis();
            break;
          }
        case HLR_TRIGGER_OP_ID:
          {
            // Key off of args here instead of data
            const ProcessorManager::TriggerOpArgs *trigger_args = 
                            (const ProcessorManager::TriggerOpArgs*)args;
            Operation *op = trigger_args->op;
            bool mapped = op->trigger_execution();
            if (!mapped)
            {
              ProcessorManager *manager = trigger_args->manager;
              manager->add_to_local_ready_queue(op, true/*failure*/);
            }
            break;
          }
        case HLR_TRIGGER_TASK_ID:
          {
            // Key off of args here instead of data
            const ProcessorManager::TriggerTaskArgs *trigger_args = 
                          (const ProcessorManager::TriggerTaskArgs*)args;
            TaskOp *op = trigger_args->op; 
            bool mapped = op->trigger_execution();
            if (!mapped)
            {
              ProcessorManager *manager = trigger_args->manager;
              manager->add_to_ready_queue(op, true/*failure*/);
            }
            break;
          }
        case HLR_DEFERRED_RECYCLE_ID:
          {
            const DeferredRecycleArgs *deferred_recycle_args = 
              (const DeferredRecycleArgs*)args;
            Internal::get_runtime(p)->free_distributed_id(
                                        deferred_recycle_args->did);
            break;
          }
        case HLR_DEFERRED_SLICE_ID:
          {
            DeferredSlicer::handle_slice(args); 
            break;
          }
        case HLR_MUST_INDIV_ID:
          {
            MustEpochTriggerer::handle_individual(args);
            break;
          }
        case HLR_MUST_INDEX_ID:
          {
            MustEpochTriggerer::handle_index(args);
            break;
          }
        case HLR_MUST_MAP_ID:
          {
            MustEpochMapper::handle_map_task(args);
            break;
          }
        case HLR_MUST_DIST_ID:
          {
            MustEpochDistributor::handle_distribute_task(args);
            break;
          }
        case HLR_MUST_LAUNCH_ID:
          {
            MustEpochDistributor::handle_launch_task(args);
            break;
          }
        case HLR_DEFERRED_FUTURE_SET_ID:
          {
            DeferredFutureSetArgs *future_args =  
              (DeferredFutureSetArgs*)args;
            const size_t result_size = 
              future_args->task_op->check_future_size(future_args->result);
            if (result_size > 0)
              future_args->target->set_result(
                  future_args->result->get_untyped_result(),
                  result_size, false/*own*/);
            future_args->target->complete_future();
            if (future_args->target->remove_base_gc_ref(DEFERRED_TASK_REF))
              legion_delete(future_args->target);
            if (future_args->result->remove_base_gc_ref(DEFERRED_TASK_REF))
              legion_delete(future_args->result);
            future_args->task_op->complete_execution();
            break;
          }
        case HLR_DEFERRED_FUTURE_MAP_SET_ID:
          {
            DeferredFutureMapSetArgs *future_args = 
              (DeferredFutureMapSetArgs*)args;
            const size_t result_size = 
              future_args->task_op->check_future_size(future_args->result);
            const void *result = future_args->result->get_untyped_result();
            for (Domain::DomainPointIterator itr(future_args->domain); 
                  itr; itr++)
            {
              Future f = future_args->future_map->get_future(itr.p);
              if (result_size > 0)
                f.impl->set_result(result, result_size, false/*own*/);
            }
            future_args->future_map->complete_all_futures();
            if (future_args->future_map->remove_reference())
              legion_delete(future_args->future_map);
            if (future_args->result->remove_base_gc_ref(DEFERRED_TASK_REF))
              legion_delete(future_args->result);
            future_args->task_op->complete_execution();
            break;
          }
        case HLR_RESOLVE_FUTURE_PRED_ID:
          {
            FuturePredOp::ResolveFuturePredArgs *resolve_args = 
              (FuturePredOp::ResolveFuturePredArgs*)args;
            resolve_args->future_pred_op->resolve_future_predicate();
            resolve_args->future_pred_op->remove_predicate_reference();
            break;
          }
        case HLR_MPI_RANK_ID:
          {
            MPIRankArgs *margs = (MPIRankArgs*)args;
            Internal::mpi_rank_table[margs->mpi_rank] = margs->source_space;
            unsigned count = 
              __sync_fetch_and_add(&Internal::remaining_mpi_notifications, 1);
            const size_t total_ranks = 
              Machine::get_machine().get_address_space_count();
            if (count == total_ranks)
              Internal::mpi_rank_event.trigger();
            break;
          }
        case HLR_CONTRIBUTE_COLLECTIVE_ID:
          {
            Future::Impl::handle_contribute_to_collective(args);
            break;
          }
        case HLR_STATE_ANALYSIS_ID:
          {
            Operation::StateAnalysisArgs *sargs = 
              (Operation::StateAnalysisArgs*)args;
            sargs->proxy_op->trigger_remote_state_analysis(sargs->ready_event);
            break;
          }
        case HLR_MAPPER_TASK_ID:
          {
            MapperTaskArgs *margs = (MapperTaskArgs*)args;
            // Tell the mapper about the result
            Internal *rt = Internal::get_runtime(p);       
            size_t result_size = margs->future->get_untyped_size();
            const void *result = margs->future->get_untyped_result();
            rt->invoke_mapper_task_result(margs->map_id, margs->proc,
                                          margs->event, result, result_size);
            // Now indicate that we are done with the future
            if (margs->future->remove_base_gc_ref(FUTURE_HANDLE_REF))
              delete margs->future;
            // Finally tell the runtime we have one less top level task
            rt->decrement_outstanding_top_level_tasks();
            // We can also deactivate the enclosing context 
            margs->context->deactivate();
            break;
          }
        case HLR_DISJOINTNESS_TASK_ID:
          {
            RegionTreeForest::DisjointnessArgs *dargs = 
              (RegionTreeForest::DisjointnessArgs*)args;
            Internal *runtime = Internal::get_runtime(p);
            runtime->forest->compute_partition_disjointness(dargs->handle,
                                                            dargs->ready);
            break;
          }
        case HLR_PART_INDEPENDENCE_TASK_ID:
          {
            IndexSpaceNode::DynamicIndependenceArgs *dargs = 
              (IndexSpaceNode::DynamicIndependenceArgs*)args;
            IndexSpaceNode::handle_disjointness_test(
                dargs->parent, dargs->left, dargs->right);
            break;
          }
        case HLR_SPACE_INDEPENDENCE_TASK_ID:
          {
            IndexPartNode::DynamicIndependenceArgs *dargs = 
              (IndexPartNode::DynamicIndependenceArgs*)args;
            IndexPartNode::handle_disjointness_test(
                dargs->parent, dargs->left, dargs->right);
            break;
          }
        case HLR_PENDING_CHILD_TASK_ID:
          {
            IndexPartNode::handle_pending_child_task(args);
            break;
          }
        case HLR_DECREMENT_PENDING_TASK_ID:
          {
            SingleTask::DecrementArgs *dargs = 
              (SingleTask::DecrementArgs*)args;
            dargs->parent_ctx->decrement_pending();
            break;
          }
        case HLR_SEND_VERSION_STATE_TASK_ID:
          {
            VersionState::SendVersionStateArgs *vargs = 
              (VersionState::SendVersionStateArgs*)args;
            vargs->proxy_this->send_version_state(vargs->target, 
                                                  vargs->request_kind,
                                                  *(vargs->request_mask),
                                                  vargs->to_trigger);
            legion_delete(vargs->request_mask);
            break;
          }
        case HLR_ADD_TO_DEP_QUEUE_TASK_ID:
          {
            SingleTask::AddToDepQueueArgs *dargs = 
              (SingleTask::AddToDepQueueArgs*)args;
            dargs->proxy_this->add_to_dependence_queue(dargs->op,
                                                       true/*has lock*/);
            break;
          }
        case HLR_WINDOW_WAIT_TASK_ID:
          {
            SingleTask::WindowWaitArgs *wargs = 
              (SingleTask::WindowWaitArgs*)args;
            wargs->parent_ctx->perform_window_wait();
            break;
          }
        case HLR_ISSUE_FRAME_TASK_ID:
          {
            SingleTask::IssueFrameArgs *fargs = 
              (SingleTask::IssueFrameArgs*)args;
            fargs->parent_ctx->perform_frame_issue(fargs->frame, 
                                                   fargs->frame_termination);
            break;
          }
        case HLR_CONTINUATION_TASK_ID:
          {
            LegionContinuation::handle_continuation(args);
            break;
          }
        case HLR_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID:
          {
            TaskImpl::SemanticRequestArgs *req_args = 
              (TaskImpl::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID:
          {
            IndexSpaceNode::SemanticRequestArgs *req_args = 
              (IndexSpaceNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID:
          {
            IndexPartNode::SemanticRequestArgs *req_args = 
              (IndexPartNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID:
          {
            FieldSpaceNode::SemanticRequestArgs *req_args = 
              (FieldSpaceNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_FIELD_SEMANTIC_INFO_REQ_TASK_ID:
          {
            FieldSpaceNode::SemanticFieldRequestArgs *req_args = 
              (FieldSpaceNode::SemanticFieldRequestArgs*)args;
            req_args->proxy_this->process_semantic_field_request(
                  req_args->fid, req_args->tag, req_args->source, 
                  false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_REGION_SEMANTIC_INFO_REQ_TASK_ID:
          {
            RegionNode::SemanticRequestArgs *req_args = 
              (RegionNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_PARTITION_SEMANTIC_INFO_REQ_TASK_ID:
          {
            PartitionNode::SemanticRequestArgs *req_args = 
              (PartitionNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, UserEvent::NO_USER_EVENT);
            break;
          }
        case HLR_SHUTDOWN_ATTEMPT_TASK_ID:
          {
            Internal::get_runtime(p)->attempt_runtime_shutdown();
            break;
          }
        case HLR_SHUTDOWN_NOTIFICATION_TASK_ID:
          {
            ShutdownManager::NotificationArgs *notification_args = 
              (ShutdownManager::NotificationArgs*)args;
            Serializer rez;
            notification_args->manager->send_message(rez, 
                SEND_SHUTDOWN_NOTIFICATION, 
                DEFAULT_VIRTUAL_CHANNEL, true/*flush*/); 
            break;
          }
        case HLR_SHUTDOWN_RESPONSE_TASK_ID:
          {
            ShutdownManager::ResponseArgs *response_args = 
              (ShutdownManager::ResponseArgs*)args;
            Serializer rez;
            rez.serialize(response_args->result);
            response_args->target->send_message(rez,
                SEND_SHUTDOWN_RESPONSE,
                DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
            break;
          }
        default:
          assert(false); // should never get here
      }
      if (tid < HLR_MESSAGE_ID)
        Internal::get_runtime(p)->decrement_total_outstanding_tasks();
#ifdef DEBUG_SHUTDOWN_HANG
      Internal *runtime = Internal::get_runtime(p);
      __sync_fetch_and_add(&runtime->outstanding_counts[tid],-1);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::profiling_runtime_task(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      Internal *rt = Internal::get_runtime(p);
      rt->process_profiling_task(p, args, arglen);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::profiling_mapper_task(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      SingleTask::process_mapper_profiling(args, arglen);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Internal::launch_top_level(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      Internal *rt = Internal::get_runtime(p);
      rt->launch_top_level_task(p);
    }

    //--------------------------------------------------------------------------
    Event LegionContinuation::defer(Internal *runtime, Event precondition)
    //--------------------------------------------------------------------------
    {
      ContinuationArgs args;
      args.hlr_id = HLR_CONTINUATION_TASK_ID;
      args.continuation = this;
      Event done = runtime->issue_runtime_meta_task(&args, sizeof(args),
                          HLR_CONTINUATION_TASK_ID, NULL, precondition,
                          0, true/*holds reservation*/);
      return done;
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionContinuation::handle_continuation(const void *args)
    //--------------------------------------------------------------------------
    {
      ContinuationArgs *cargs = (ContinuationArgs*)args;
      cargs->continuation->execute();
    }

#ifdef TRACE_ALLOCATION
    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(
                                       AllocationType a, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Internal *rt = Internal::get_runtime(
                                  Processor::get_executing_processor());
      if (rt != NULL)
        rt->trace_allocation(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_free(AllocationType a, 
                                                 size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Internal *rt = Internal::get_runtime(
                                  Processor::get_executing_processor());
      if (rt != NULL)
        rt->trace_free(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ Internal* LegionAllocation::find_runtime(void)
    //--------------------------------------------------------------------------
    {
      return Internal::get_runtime(Processor::get_executing_processor());
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(Internal *&runtime,
                                       AllocationType a, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (runtime == NULL)
      {
        runtime = LegionAllocation::find_runtime();
        // Only happens during initialization
        if (runtime == NULL)
          return;
      }
      runtime->trace_allocation(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_free(Internal *&runtime,
                                       AllocationType a, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (runtime == NULL)
      {
        runtime = LegionAllocation::find_runtime();
        // Only happens during intialization
        if (runtime == NULL)
          return;
      }
      runtime->trace_free(a, size, elems);
    }
#endif

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

