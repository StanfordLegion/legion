/* Copyright 2015 Stanford University
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
#include "legion_logging.h"
#include "legion_profiling.h"
#ifdef HANG_TRACE
#include <signal.h>
#include <execinfo.h>
#endif

namespace LegionRuntime {

#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
  namespace Accessor {
    namespace AccessorType {
#ifdef PRIVILEGE_CHECKS
      const char* find_privilege_task_name(void *impl)
      {
        // Have to bounce this off the Runtime because C++ is stupid
        return HighLevel::Runtime::find_privilege_task_name(impl); 
      }
#endif
#ifdef BOUNDS_CHECKS
      void check_bounds(void *impl, ptr_t ptr)
      {
        // Have to bounce this off the Runtime because C++ is stupid 
        HighLevel::Runtime::check_bounds(impl, ptr);
      }
      void check_bounds(void *impl, const LowLevel::DomainPoint &dp)
      {
        // Have to bounce this off the Runtime because C++ is stupid
        HighLevel::Runtime::check_bounds(impl, dp);
      }
#endif
    };
  };
#endif
 
  namespace HighLevel {

    // Extern declarations for loggers
    extern Logger::Category log_run;
    extern Logger::Category log_task;
    extern Logger::Category log_region;
    extern Logger::Category log_index;
    extern Logger::Category log_field;
    extern Logger::Category log_inst;
    extern Logger::Category log_spy;
    extern Logger::Category log_garbage;
    extern Logger::Category log_leak;
    extern Logger::Category log_variant;
    extern Logger::Category log_allocation;

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
    Future::Impl::Impl(Runtime *rt, bool register_future, DistributedID did, 
                       AddressSpaceID own_space, AddressSpaceID loc_space,
                       Operation *o /*= NULL*/)
      : DistributedCollectable(rt, did, own_space, loc_space),
        producer_op(o), op_gen((o == NULL) ? 0 : o->get_generation()),
        ready_event(UserEvent::create_user_event()), result(NULL),
        result_size(0), empty(true), sampled(false)
    //--------------------------------------------------------------------------
    {
      if (register_future)
        runtime->register_future(did, this);
      if (producer_op != NULL)
        producer_op->add_mapping_reference(op_gen);
    }

    //--------------------------------------------------------------------------
    Future::Impl::Impl(const Future::Impl &rhs)
      : DistributedCollectable(NULL, 0, 0, 0), producer_op(NULL), op_gen(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Future::Impl::~Impl(void)
    //--------------------------------------------------------------------------
    {
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
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                              producer_op->get_parent()->get_unique_task_id(),
                              producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else {
        Processor exec_proc = Processor::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
      }
#endif
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
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else {
        Processor exec_proc = Processor::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                               prodcuer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
      }
#endif
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
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(
                              producer_op->get_parent()->get_unique_task_id(), 
                              PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else if (block) {
        Processor exec_proc = Processor::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                               producer_op->get_parent()->get_unique_task_id(),
                               producer_op->get_unique_op_id());
      }
#endif
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
      assert(owner);
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
      if (owner)
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
    void Future::Impl::notify_activate(void)
    //--------------------------------------------------------------------------
    {
      // do nothing
    }

    //--------------------------------------------------------------------------
    void Future::Impl::garbage_collect(void)
    //--------------------------------------------------------------------------
    {
      // do nothing
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // do nothing
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // do nothing
    }

    //--------------------------------------------------------------------------
    void Future::Impl::notify_new_remote(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      // do nothing 
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
      assert(owner);
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
    bool Future::Impl::send_future(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      // Two phase approach, check first to see if we need to do the send
      bool need_send;
      {
        AutoLock gc(gc_lock,1,false/*exclusive*/);
        if (remote_spaces.find(sid) != remote_spaces.end())
          need_send = false;
        else
          need_send = true;
      }
      // Need to send this first to avoid race
      bool performed_send = false;
      if (need_send)
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
        // Retake the lock and make sure we didn't lose the race
        AutoLock gc(gc_lock);
        if (remote_spaces.find(sid) == remote_spaces.end())
        {
          // Pack up the remaining information
          rez.serialize<size_t>(remote_spaces.size());
          for (std::set<AddressSpaceID>::const_iterator it = 
                remote_spaces.begin(); it != remote_spaces.end(); it++)
          {
            rez.serialize(*it);
          }
          // Actually do the send and then mark that we
          // have already sent an instance there
          runtime->send_future(sid, rez);
          remote_spaces.insert(sid);
          // Also if this is the owner mark that we already sent the future
          if (owner && send_result)
            registered_waiters.insert(sid);
          performed_send = true;
        }
      }
      if (performed_send)
        register_waiter(sid);
      // Return whether we need to send a remote 
      // reference with the packed future
      return send_remote_reference(sid);
    }

    //--------------------------------------------------------------------------
    void Future::Impl::register_waiter(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      if (owner)
      {
        bool send_result;
        {
          AutoLock gc(gc_lock);
          if (registered_waiters.find(sid) == registered_waiters.end())
          {
            send_result = ready_event.has_triggered();
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
                                        Runtime *runtime, AddressSpaceID source)
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
      future->update_remote_spaces(source); 
      if (is_complete)
      {
        future->unpack_future(derez);
        future->complete_future();
      }
      size_t num_new_spaces;
      derez.deserialize(num_new_spaces);
      for (unsigned idx = 0; idx < num_new_spaces; idx++)
      {
        AddressSpaceID new_space;
        derez.deserialize(new_space);
        future->update_remote_spaces(new_space);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Future::Impl::handle_future_result(Deserializer &derez,
                                                       Runtime *runtime)
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
                                          Deserializer &derez, Runtime *runtime)
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
        add_gc_reference();
        ContributeCollectiveArgs args;
        args.hlr_id = HLR_CONTRIBUTE_COLLECTIVE_ID;
        args.impl = this;
        args.barrier = bar;
        args.count = count;
        Processor proc = runtime->find_utility_group();
        // Spawn the task dependent on the future being ready
        proc.spawn(HLR_TASK_ID, &args, sizeof(args), ready_event);
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
      if (cargs->impl->remove_gc_reference())
        delete cargs->impl;
    }
      
    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, TaskOp *t, Runtime *rt)
      : Collectable(), context(ctx), task(t), task_gen(t->get_generation()),
        valid(true), runtime(rt), ready_event(t->get_completion_event()),
        lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, Event comp_event, Runtime *rt)
      : Collectable(), context(ctx), task(NULL), task_gen(0),
        valid(true), runtime(rt), ready_event(comp_event),
        lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, Runtime *rt)
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
      if (valid)
      {
        Event lock_event = lock.acquire(0, true/*exclusive*/);
        lock_event.wait(true/*block*/);
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
#ifdef LEGION_LOGGING
          Processor exec_proc = Processor::get_executing_processor();
          LegionLogging::log_future_wait_begin(exec_proc,
                                          context->get_unique_task_id(),
                                          task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(context->get_unique_task_id(), 
                                     PROF_BEGIN_WAIT);
#endif
          runtime->pre_wait(proc);
          ready_event.wait();
          runtime->post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_future_wait_end(exec_proc,
                                          context->get_unique_task_id(),
                                          task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(context->get_unique_task_id(), 
                                     PROF_END_WAIT);
#endif
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
                               bool leaf, Runtime *rt)
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
      // Remove any valid references we might have
      if (!leaf_region && reference.has_ref())
        reference.remove_valid_reference();
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
#ifdef LEGION_LOGGING
        LegionLogging::log_inline_wait_begin(proc,
                                             context->get_unique_task_id(),
                                             ready_event);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(context->get_unique_task_id(),
                                   PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(proc);
        ready_event.wait();
        runtime->post_wait(proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_inline_wait_end(proc,
                                           context->get_unique_task_id(),
                                           ready_event);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(context->get_unique_task_id(),
                                   PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else
      {
        Processor proc = context->get_executing_processor();
        LegionLogging::log_inline_nowait(proc,
                                         context->get_unique_task_id(),
                                         ready_event);
      }
#endif
      // Now wait for the reference to be ready
      Event ref_ready = reference.get_ready_event();
      if (!ref_ready.has_triggered())
      {
        Processor proc = context->get_executing_processor();
#ifdef LEGION_LOGGING
        LegionLogging::log_inline_wait_begin(proc,
                                             context->get_unique_task_id(),
                                             ref_ready);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(context->get_unique_task_id(),
                                   PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(proc);
        // If we need a lock for this instance taken it
        // once the reference event is ready
        if (reference.has_required_locks())
        {
          std::map<Reservation,bool> required_locks;
          reference.update_atomic_locks(required_locks, true/*exclusive*/);
          Event locked_event = ref_ready;
          for (std::map<Reservation,bool>::const_iterator it = 
                required_locks.begin(); it != required_locks.end(); it++)
          {
            locked_event = it->first.acquire(0, it->second, locked_event);
          }
          locked_event.wait();
        }
        else
          ref_ready.wait();
        runtime->post_wait(proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_inline_wait_end(proc,
                                           context->get_unique_task_id(),
                                           ref_ready);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(context->get_unique_task_id(),
                                   PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else
      {
        Processor proc = context->get_executing_processor();
        LegionLogging::log_inline_nowait(proc,
                                         context->get_unique_task_id(),
                                         ref_ready);
      }
#endif
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
            "without privleges!", fid);
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
      // Before unmapping, make sure any previous mappings have finished
      wait_until_valid();
      // Unlock our lock now that we're done
      if (reference.has_required_locks())
      {
        std::map<Reservation,bool> required_locks;
        reference.update_atomic_locks(required_locks,true/*doesn't matter*/);
        for (std::map<Reservation,bool>::const_iterator it = 
              required_locks.begin(); it != required_locks.end(); it++)
        {
          it->first.release();
        }
      }
      mapped = false;
      valid = false;
      if (trigger_on_unmap)
      {
        trigger_on_unmap = false;
        termination_event.trigger();
      }
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
    void PhysicalRegion::Impl::reset_reference(const InstanceRef &ref,
                                               UserEvent term_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapped);
#endif
      if (!leaf_region && reference.has_ref())
        reference.remove_valid_reference();
      reference = ref;
      if (!leaf_region && reference.has_ref())
        reference.add_valid_reference();
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
      legion_ready.wait(false/*block*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(state == IN_LEGION);
#endif
    }

    /////////////////////////////////////////////////////////////
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind,
                                       Runtime *rt, unsigned width, 
                                       unsigned def_mappers, bool no_steal, 
                                       unsigned max_steals)
      : runtime(rt), local_proc(proc), proc_kind(kind), 
        utility_proc(rt->find_utility_group()), superscalar_width(width), 
        stealing_disabled(no_steal), max_outstanding_steals(max_steals),
        next_local_index(0),
        task_scheduler_enabled(false), pending_shutdown(false),
        total_active_contexts(0),
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
      this->dependence_lock = Reservation::create_reservation();
      this->local_queue_lock = Reservation::create_reservation();
      this->queue_lock = Reservation::create_reservation();
      this->message_lock = Reservation::create_reservation();
      this->stealing_lock = Reservation::create_reservation();
      this->thieving_lock = Reservation::create_reservation();
      context_states.resize(DEFAULT_CONTEXTS);
      dependence_preconditions.resize(DEFAULT_CONTEXTS, Event::NO_EVENT);
      local_scheduler_preconditions.resize(superscalar_width, Event::NO_EVENT);
    }

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(const ProcessorManager &rhs)
      : runtime(NULL), local_proc(Processor::NO_PROC),
        proc_kind(Processor::LOC_PROC), utility_proc(Processor::NO_PROC),
        superscalar_width(0), stealing_disabled(false), 
        max_outstanding_steals(0), next_local_index(0),
        task_scheduler_enabled(false), pending_shutdown(false),
        total_active_contexts(0)
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
      dependence_lock.destroy_reservation();
      dependence_lock = Reservation::NO_RESERVATION;
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
    template<typename T1, void (Mapper::*CALL)(T1), bool BLOCK>
    void ProcessorManager::invoke_mapper(MapperID map_id, const T1 &arg1)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        (mapper->*CALL)(arg1);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, T (Mapper::*CALL)(T1), bool BLOCK>
    T ProcessorManager::invoke_mapper(MapperID map_id, T init, const T1 &arg1)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      T result = init;
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        result = (mapper->*CALL)(arg1);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T1, typename T2, void (Mapper::*CALL)(T1,T2), bool BLOCK>
    void ProcessorManager::invoke_mapper(MapperID map_id, 
                                         const T1 &arg1, const T2 &arg2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        (mapper->*CALL)(arg1, arg2);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, 
             T (Mapper::*CALL)(T1,T2), bool BLOCK>
    T ProcessorManager::invoke_mapper(MapperID map_id, T init, 
                                      const T1 &arg1, const T2 &arg2)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      T result = init;
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        result = (mapper->*CALL)(arg1, arg2);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T1, typename T2, 
             typename T3, void (Mapper::*CALL)(T1,T2,T3), bool BLOCK>
    void ProcessorManager::invoke_mapper(MapperID map_id, const T1 &arg1,
                                         const T2 &arg2, const T3 &arg3)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        (mapper->*CALL)(arg1, arg2, arg3);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, 
             typename T3, T (Mapper::*CALL)(T1,T2,T3), bool BLOCK>
    T ProcessorManager::invoke_mapper(MapperID map_id, T init, const T1 &arg1,
                                      const T2 &arg2, const T3 &arg3)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      T result = init;
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        result = (mapper->*CALL)(arg1, arg2, arg3);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T1, typename T2, typename T3, 
             typename T4, void (Mapper::*CALL)(T1,T2,T3,T4), bool BLOCK>
    void ProcessorManager::invoke_mapper(MapperID map_id, const T1 &arg1,
                                 const T2 &arg2, const T3 &arg3, const T4 &arg4)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        (mapper->*CALL)(arg1, arg2, arg3, arg4);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3,
             typename T4, T (Mapper::*CALL)(T1,T2,T3,T4), bool BLOCK>
    T ProcessorManager::invoke_mapper(MapperID map_id, T init, const T1 &arg1,
                                const T2 &arg2, const T3 &arg3, const T4 &arg4)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#endif
      Mapper *mapper = mapper_objects[map_id];
      std::vector<MapperMessage> messages;
      Event wait_on = defer_mapper_event[map_id];
      T result = init;
      do
      {
        if (BLOCK && wait_on.exists())
        {
          // Always send messages before waiting
          if (!messages.empty())
            send_mapper_messages(map_id, messages);
          wait_on.wait(false/*block*/);
        }
        AutoLock m_lock(mapper_locks[map_id]);
        inside_mapper_call[map_id] = true;
        result = (mapper->*CALL)(arg1, arg2, arg3, arg4);
        inside_mapper_call[map_id] = false;
        if (defer_mapper_event[map_id].exists())
        {
          wait_on = defer_mapper_event[map_id];
          defer_mapper_event[map_id] = Event::NO_EVENT;
        }
        AutoLock g_lock(message_lock);
        if (!mapper_messages[map_id].empty())
        {
          messages = mapper_messages[map_id];
          mapper_messages[map_id].clear();
        }
      } while (BLOCK && wait_on.exists());
      if (!BLOCK && wait_on.exists())
        log_run.warning(                    "Ignoring wait request in scheduling mapper call!");
      if (!messages.empty())
        send_mapper_messages(map_id, messages);
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_set_task_options(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<Task*,&Mapper::select_task_options,
                    true/*block*/>(task->map_id, task);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_pre_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<bool,Task*,&Mapper::pre_map_task,
                           true/*block*/>(task->map_id, false/*init*/, task);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_select_variant(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<Task*,&Mapper::select_task_variant,
                    true/*block*/>(task->map_id, task);
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
          wait_on.wait(false/*block*/);
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
    void ProcessorManager::invoke_mapper_failed_mapping(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const Mappable*,&Mapper::notify_mapping_failed,
                    true/*block*/>(mappable->map_id, mappable);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_notify_result(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const Mappable*,&Mapper::notify_mapping_result,
                    true/*block*/>(mappable->map_id, mappable);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_slice_domain(TaskOp *task,
                                      std::vector<Mapper::DomainSplit> &splits)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const Task*,const Domain&,std::vector<Mapper::DomainSplit>&,
                    &Mapper::slice_domain,true/*block*/>(
                              task->map_id, task, task->index_domain, splits);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_inline(Inline *op)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<bool,Inline*,&Mapper::map_inline,
                           true/*block*/>(op->map_id, false/*init*/, op);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_copy(Copy *op)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<bool,Copy*,&Mapper::map_copy,
                           true/*block*/>(op->map_id, false/*init*/, op);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_speculate(Mappable *op, bool &value)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<bool,const Mappable*,bool&,
                           &Mapper::speculate_on_predicate, true/*block*/>(
                               op->map_id, false/*init*/, op, value);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_configure_context(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<Task*,&Mapper::configure_context,
                    true/*block*/>(task->map_id, task);
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
          wait_on.wait(false/*block*/);
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
      invoke_mapper<const Mappable*,const std::set<Memory>&,Memory,
            std::vector<Memory>&,&Mapper::rank_copy_sources,true/*block*/>(
                    mappable->map_id, mappable, memories, destination, order);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_notify_profiling(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const Task*,&Mapper::notify_profiling_info,true/*block*/>(
                                                          task->map_id, task);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_must_epoch(
        const std::vector<Task*> &tasks,
        const std::vector<Mapper::MappingConstraint> &constraints,
        MapperID map_id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<bool,const std::vector<Task*>&,
                const std::vector<Mapper::MappingConstraint>&,MappingTagID,
                &Mapper::map_must_epoch,true/*block*/>(map_id, false/*init*/, 
                                                      tasks, constraints, tag);
    }

    //--------------------------------------------------------------------------
    int ProcessorManager::invoke_mapper_get_tunable_value(TaskOp *task,
                                                          TunableID tid,
                                                          MapperID map_id,
                                                          MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return invoke_mapper<int,const Task*,TunableID,MappingTagID,
                &Mapper::get_tunable_value,true/*block*/>(map_id, 0/*init*/, 
                                                          task, tid, tag);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_handle_message(MapperID map_id,
                                                        Processor source,
                                                        const void *message,
                                                        size_t length)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<Processor,const void*,size_t,
                    &Mapper::handle_message,true/*block*/>(
                                          map_id, source, message, length);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_task_result(MapperID map_id,
                                                     Event event,
                                                     const void *result,
                                                     size_t result_size)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<MapperEvent,const void*,size_t,
        &Mapper::handle_mapper_task_result,true/*block*/>(map_id, event, 
                                                          result, result_size);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_permit_task_steal(MapperID map_id,
                Processor thief, const std::vector<const Task*> &stealable,
                std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<Processor,const std::vector<const Task*>&,
            std::set<const Task*>&,&Mapper::permit_task_steal,true/*block*/>(
                        map_id, thief, stealable, to_steal);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_target_task_steal(MapperID map_id, 
                                         const std::set<Processor> &blacklist,
                                         std::set<Processor> &steal_targets)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const std::set<Processor>&,std::set<Processor>&,
                    &Mapper::target_task_steal,false/*block*/>(map_id,
                                                    blacklist, steal_targets);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_select_tasks_to_schedule(
                          MapperID map_id, const std::list<Task*> &ready_tasks)
    //--------------------------------------------------------------------------
    {
      invoke_mapper<const std::list<Task*>&,&Mapper::select_tasks_to_schedule,
                    false/*block*/>(map_id, ready_tasks);
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
      if (!pending_shutdown && (total_active_contexts > 0))
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
      utility_proc.spawn(HLR_TASK_ID, &sched_args, sizeof(sched_args));
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::notify_pending_shutdown(void)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      pending_shutdown = true;
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
      {
        AutoLock d_lock(dependence_lock);
        dependence_preconditions.resize(max_contexts, Event::NO_EVENT);
      }
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
    void ProcessorManager::add_to_dependence_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op != NULL);
#endif
      DeferredTriggerArgs args;
      args.hlr_id = HLR_TRIGGER_DEPENDENCE_ID;
      args.manager = this;
      args.op = op;
      ContextID ctx_id = op->get_parent()->get_context_id();
      AutoLock d_lock(dependence_lock);
      Event next = utility_proc.spawn(HLR_TASK_ID, &args, sizeof(args),
                                      dependence_preconditions[ctx_id]);
      dependence_preconditions[ctx_id] = next;
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
      if (!prev_failure)
      {
        AutoLock l_lock(local_queue_lock); 
        Event next = utility_proc.spawn(HLR_TASK_ID, &args, sizeof(args),
                              local_scheduler_preconditions[next_local_index]);
        local_scheduler_preconditions[next_local_index++] = next;
        if (next_local_index == superscalar_width)
          next_local_index = 0;
      }
      else
        utility_proc.spawn(HLR_TASK_ID, &args, sizeof(args));
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
        // Acquire the mapper lock and ask the mapper about scheduling
        // and then about stealing if not disabled
#if 0
        {
          AutoLock map_lock(mapper_locks[map_id]);
          if (!visible_tasks.empty())
          {
            // TODO
            mapper_objects[map_id]->select_tasks_to_schedule(ready_tasks);
          }
          if (!stealing_disabled)
          {
            AutoLock steal_lock(stealing_lock);
            std::set<Processor> &blacklist = outstanding_steal_requests[map_id];
            if (blacklist.size() < max_outstanding_steals)
            {
              std::set<Processor> steal_targets;
              // TODO
              mapper_objects[map_id]->target_task_steal(blacklist, 
                                                        steal_targets);
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
          }
        }
#endif
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
          Event wait_on = (*vis_it)->defer_mapping();
          // We give a slight priority to triggering the execution
          // of tasks relative to other runtime operations because
          // they actually have a feedback mechanism controlling
          // how far they get ahead.  We give a slight edge in priority
          // to tasks being sent remotely to get them in flight.
          // Give priority to things which are getting sent remotely
          args.op = *vis_it;
          int priority = ((*vis_it)->target_proc != local_proc) ? 2 : 1;
          utility_proc.spawn(HLR_TASK_ID, &args, sizeof(args),
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
    MemoryManager::MemoryManager(Memory m, Runtime *rt)
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
    void MemoryManager::recycle_physical_instance(InstanceManager *instance)
    //--------------------------------------------------------------------------
    {
      instance->add_resource_reference();
      AutoLock m_lock(manager_lock); 
#ifdef DEBUG_HIGH_LEVEL
      assert(available_instances.find(instance) == available_instances.end());
#endif
      available_instances.insert(instance);
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::reclaim_physical_instance(InstanceManager *instance)
    //--------------------------------------------------------------------------
    {
      bool reclaim = false;
      {
        AutoLock m_lock(manager_lock);
        std::set<InstanceManager*>::iterator finder = 
          available_instances.find(instance);
        // If we found it, remove it from the set of available resources
        if (finder != available_instances.end())
        {
          reclaim = true;
          available_instances.erase(finder);
        }
      }
      // If we are reclaiming it, remove our resource reference
      if (reclaim && instance->remove_resource_reference())
        legion_delete(instance);
      return reclaim;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance MemoryManager::find_physical_instance(size_t field_size,
                                                           const Domain &dom,
                                                           const unsigned depth,
                                                           Event &use_event)
    //--------------------------------------------------------------------------
    {
      InstanceManager *to_recycle = NULL;
      {
        AutoLock m_lock(manager_lock);
        for (std::set<InstanceManager*>::iterator it = 
              available_instances.begin(); it != 
              available_instances.end(); it++)
        {
          // To avoid deadlock it is imperative that the recycled instance
          // be used by an operation which is at the same level or higher
          // in the task graph.
          if (depth > (*it)->depth)
            continue;
          if ((*it)->match_instance(field_size, dom))
          {
            to_recycle = (*it);
            available_instances.erase(it);
            break;
          }
        }
      }
      if (to_recycle != NULL)
      {
        // If we found one, then compute the recycle event
        // and then return the 
        PhysicalInstance result = to_recycle->get_instance();
        use_event = to_recycle->get_recycle_event();
        // Remove our resource reference
        if (to_recycle->remove_resource_reference())
          legion_delete(to_recycle);
        return result;
      }
      return PhysicalInstance::NO_INST;
    }
    
    //--------------------------------------------------------------------------
    PhysicalInstance MemoryManager::find_physical_instance(
                    const std::vector<size_t> &field_sizes, const Domain &dom,
                    const size_t blocking_factor, const unsigned depth,
                    Event &use_event)
    //--------------------------------------------------------------------------
    {
      InstanceManager *to_recycle = NULL;
      {
        AutoLock m_lock(manager_lock);
        for (std::set<InstanceManager*>::iterator it = 
              available_instances.begin(); it != 
              available_instances.end(); it++)
        {
          // To avoid deadlock it is imperative that the recycled instance
          // be used by an operation which is at the same level or higher
          // in the task graph.
          if (depth > (*it)->depth)
            continue;
          if ((*it)->match_instance(field_sizes, dom, blocking_factor))
          {
            to_recycle = (*it);
            available_instances.erase(it);
            break;
          }
        }
      }
      if (to_recycle != NULL)
      {
        PhysicalInstance result = to_recycle->get_instance();
        use_event = to_recycle->get_recycle_event();
        // Remove our resource reference
        if (to_recycle->remove_resource_reference())
          legion_delete(to_recycle);
        return result;
      }
      return PhysicalInstance::NO_INST;
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
    // Message Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(AddressSpaceID remote,
                                   Runtime *rt, size_t max_message_size,
                                   const std::set<Processor> &remote_util_procs)
      : local_address_space(rt->address_space), remote_address_space(remote),
        runtime(rt), sending_buffer((char*)malloc(max_message_size)), 
        sending_buffer_size(max_message_size)
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
      // Figure out which processor to send to based on our address
      // space ID.  If there is an explicit utility processor for one
      // of the processors in our set then we use that.  Otherwise we
      // round-robin senders onto different target processors on the
      // remote node to avoid over-burdening any one of them with messages.
      {
        unsigned idx = 0;
        const unsigned target_idx = local_address_space % 
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
      // Set up the buffer for sending the first batch of messages
      // Only need to write the processor once
      *((HLRTaskID*)sending_buffer) = HLR_MESSAGE_ID;
      sending_index = sizeof(HLRTaskID);
      *((AddressSpaceID*)
          (((char*)sending_buffer)+sending_index)) = local_address_space;
      sending_index += sizeof(local_address_space);
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
    MessageManager::MessageManager(const MessageManager &rhs)
      : local_address_space(0), remote_address_space(0), runtime(NULL),
        sending_buffer(NULL), sending_buffer_size(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MessageManager::~MessageManager(void)
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
    MessageManager& MessageManager::operator=(const MessageManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_task(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, TASK_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_steal_request(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, STEAL_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_advertisement(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, ADVERTISEMENT_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_space_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_SPACE_NODE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_space_request(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_SPACE_REQUEST, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_space_return(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_SPACE_RETURN, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_partition_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_PARTITION_NODE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_partition_request(Serializer &rez, 
                                                      bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_PARTITION_REQUEST, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_partition_return(Serializer &rez,
                                                     bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_PARTITION_RETURN, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SPACE_NODE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_request(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SPACE_REQUEST, flush);
    }
    
    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_return(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SPACE_RETURN, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_logical_region_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_LOGICAL_REGION_NODE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_space_destruction(Serializer &rez, 
                                                      bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, INDEX_SPACE_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_partition_destruction(Serializer &rez, 
                                                          bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, INDEX_PARTITION_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_destruction(Serializer &rez, 
                                                      bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, FIELD_SPACE_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_logical_region_destruction(Serializer &rez, 
                                                         bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, LOGICAL_REGION_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_logical_partition_destruction(Serializer &rez, 
                                                            bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, LOGICAL_PARTITION_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_allocation(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, FIELD_ALLOCATION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_destruction(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, FIELD_DESTRUCTION_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_individual_remote_mapped(Serializer &rez, 
                                                       bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, INDIVIDUAL_REMOTE_MAPPED, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_individual_remote_complete(Serializer &rez,
                                                         bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, INDIVIDUAL_REMOTE_COMPLETE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_individual_remote_commit(Serializer &rez,
                                                       bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, INDIVIDUAL_REMOTE_COMMIT, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_slice_remote_mapped(Serializer &rez, 
                                                       bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SLICE_REMOTE_MAPPED, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_slice_remote_complete(Serializer &rez,
                                                         bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SLICE_REMOTE_COMPLETE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_slice_remote_commit(Serializer &rez,
                                                       bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SLICE_REMOTE_COMMIT, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_remove_distributed_resource(Serializer &rez,
                                                          bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, DISTRIBUTED_REMOVE_RESOURCE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_remove_distributed_remote(Serializer &rez,
                                                        bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, DISTRIBUTED_REMOVE_REMOTE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_add_distributed_remote(Serializer &rez,
                                                     bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, DISTRIBUTED_ADD_REMOTE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_remove_hierarchical_resource(Serializer &rez,
                                                           bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, HIERARCHICAL_REMOVE_RESOURCE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_remove_hierarchical_remote(Serializer &rez,
                                                         bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, HIERARCHICAL_REMOVE_REMOTE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_user(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_USER, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_atomic(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_ATOMIC, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_subscriber(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_SUBSCRIBER, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_materialized_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_MATERIALIZED_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_materialized_update(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_MATERIALIZED_UPDATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_materialized_view(Serializer &rez,bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_MATERIALIZED_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_composite_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_COMPOSITE_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_composite_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_COMPOSITE_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_composite_update(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_COMPOSITE_UPDATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_reduction_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REDUCTION_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_reduction_update(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REDUCTION_UPDATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_reduction_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_REDUCTION_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_instance_manager(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INSTANCE_MANAGER, flush); 
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_reduction_manager(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REDUCTION_MANAGER, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_region_state(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REGION_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_partition_state(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_PARTITION_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_region_state(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_REGION_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_partition_state(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_PARTITION_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_remote_references(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REMOTE_REFERENCES, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_individual_request(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDIVIDUAL_REQUEST, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_individual_return(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDIVIDUAL_RETURN, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_slice_request(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_SLICE_REQUEST, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_slice_return(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_SLICE_RETURN, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_future(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FUTURE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_future_result(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FUTURE_RESULT, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_future_subscription(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FUTURE_SUBSCRIPTION, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_make_persistent(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_MAKE_PERSISTENT, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_mapper_message(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_MAPPER_MESSAGE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_mapper_broadcast(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_MAPPER_BROADCAST, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_space_semantic_info(Serializer &rez, 
                                                        bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_SPACE_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_index_partition_semantic_info(Serializer &rez, 
                                                            bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_PARTITION_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_semantic_info(Serializer &rez, 
                                                        bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SPACE_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_semantic_info(Serializer &rez, 
                                                  bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_logical_region_semantic_info(Serializer &rez, 
                                                           bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_LOGICAL_REGION_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_logical_partition_semantic_info(Serializer &rez, 
                                                              bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_LOGICAL_PARTITION_SEMANTIC_INFO, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_free_remote_context(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FREE_REMOTE_CONTEXT, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_validate_remote_state(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_VALIDATE_REMOTE_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_invalidate_remote_state(Serializer &rez, 
                                                      bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INVALIDATE_REMOTE_STATE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::package_message(Serializer &rez, MessageKind k,
                                         bool flush)
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
          send_message(true/*complete*/);
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
            send_message(false/*complete*/);
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
        send_message(true/*complete*/);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_message(bool complete)
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
      *((MessageHeader*)(sending_buffer + sizeof(HLRTaskID) +
                          sizeof(local_address_space))) = header;
      *((unsigned*)(sending_buffer + sizeof(HLRTaskID) +
            sizeof(local_address_space) + sizeof(header))) = packaged_messages;
      // Send the message
      Event next_event = target.spawn(HLR_TASK_ID, sending_buffer,
                                      sending_index, last_message_event);
      // Update the event
      last_message_event = next_event;
      // Reset the state of the buffer
      sending_index = sizeof(HLRTaskID) + sizeof(local_address_space) + 
                      sizeof(header) + sizeof(unsigned);
      if (partial)
        header = PARTIAL_MESSAGE;
      else
        header = FULL_MESSAGE;
      packaged_messages = 0;
    }

    //--------------------------------------------------------------------------
    void MessageManager::process_message(const void *args, size_t arglen)
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
            handle_messages(num_messages, buffer, arglen);
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
            handle_messages(received_messages, receiving_buffer, 
                            receiving_index);
            receiving_index = 0;
            received_messages = 0;
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    //--------------------------------------------------------------------------
    void MessageManager::handle_messages(unsigned num_messages,
                                         const char *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_messages; idx++)
      {
        // Pull off the message kind and the size of the message
#ifdef DEBUG_HIGH_LEVEL
        assert(arglen > (sizeof(MessageKind)+sizeof(size_t)));
#endif
        MessageKind kind = *((const MessageKind*)args);
        args += sizeof(kind);
        arglen -= sizeof(kind);
        size_t message_size = *((const size_t*)args);
        args += sizeof(message_size);
        arglen -= sizeof(message_size);
#ifdef DEBUG_HIGH_LEVEL
        if (idx == (num_messages-1))
          assert(message_size == arglen);
#endif
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
          case FIELD_ALLOCATION_MESSAGE:
            {
              runtime->handle_field_allocation(derez, remote_address_space);
              break;
            }
          case FIELD_DESTRUCTION_MESSAGE:
            {
              runtime->handle_field_destruction(derez, remote_address_space);
              break;
            }
          case INDIVIDUAL_REMOTE_MAPPED:
            {
              runtime->handle_individual_remote_mapped(derez, 
                                                       remote_address_space);
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
          case DISTRIBUTED_REMOVE_RESOURCE:
            {
              runtime->handle_distributed_remove_resource(derez); 
              break;
            }
          case DISTRIBUTED_REMOVE_REMOTE:
            {
              runtime->handle_distributed_remove_remote(derez,
                                                      remote_address_space);
              break;
            }
          case DISTRIBUTED_ADD_REMOTE:
            {
              runtime->handle_distributed_add_remote(derez); 
              break;
            }
          case HIERARCHICAL_REMOVE_RESOURCE:
            {
              runtime->handle_hierarchical_remove_resource(derez);
              break;
            }
          case HIERARCHICAL_REMOVE_REMOTE:
            {
              runtime->handle_hierarchical_remove_remote(derez);
              break;
            }
          case SEND_BACK_USER:
            {
              runtime->handle_send_back_user(derez, remote_address_space);
              break;
            }
          case SEND_BACK_ATOMIC:
            {
              runtime->handle_send_back_atomic(derez, remote_address_space);
              break;
            }
          case SEND_SUBSCRIBER:
            {
              runtime->handle_send_subscriber(derez, remote_address_space);
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
          case SEND_BACK_MATERIALIZED_VIEW:
            {
              runtime->handle_send_back_materialized_view(derez, 
                                                          remote_address_space);
              break;
            }
          case SEND_COMPOSITE_VIEW:
            {
              runtime->handle_send_composite_view(derez, remote_address_space);
              break;
            }
          case SEND_BACK_COMPOSITE_VIEW:
            {
              runtime->handle_send_back_composite_view(derez, 
                                                       remote_address_space);
              break;
            }
          case SEND_COMPOSITE_UPDATE:
            {
              runtime->handle_send_composite_update(derez, 
                                                    remote_address_space);
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
          case SEND_BACK_REDUCTION_VIEW:
            {
              runtime->handle_send_back_reduction_view(derez, 
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
          case SEND_REGION_STATE:
            {
              runtime->handle_send_region_state(derez, remote_address_space);
              break;
            }
          case SEND_PARTITION_STATE:
            {
              runtime->handle_send_partition_state(derez, remote_address_space);
              break;
            }
          case SEND_BACK_REGION_STATE:
            {
              runtime->handle_send_back_region_state(derez, 
                                                     remote_address_space);
              break;
            }
          case SEND_BACK_PARTITION_STATE:
            {
              runtime->handle_send_back_partition_state(derez, 
                                                        remote_address_space);
              break;
            }
          case SEND_REMOTE_REFERENCES:
            {
              runtime->handle_send_remote_references(derez);
              break;
            }
          case SEND_INDIVIDUAL_REQUEST:
            {
              runtime->handle_individual_request(derez, remote_address_space);
              break;
            }
          case SEND_INDIVIDUAL_RETURN:
            {
              runtime->handle_individual_return(derez);
              break;
            }
          case SEND_SLICE_REQUEST:
            {
              runtime->handle_slice_request(derez, remote_address_space);
              break;
            }
          case SEND_SLICE_RETURN:
            {
              runtime->handle_slice_return(derez);
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
          case SEND_INDEX_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_index_space_semantic_info(derez);
              break;
            }
          case SEND_INDEX_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_index_partition_semantic_info(derez);
              break;
            }
          case SEND_FIELD_SPACE_SEMANTIC_INFO:
            {
              runtime->handle_field_space_semantic_info(derez);
              break;
            }
          case SEND_FIELD_SEMANTIC_INFO:
            {
              runtime->handle_field_semantic_info(derez);
              break;
            }
          case SEND_LOGICAL_REGION_SEMANTIC_INFO:
            {
              runtime->handle_logical_region_semantic_info(derez);
              break;
            }
          case SEND_LOGICAL_PARTITION_SEMANTIC_INFO:
            {
              runtime->handle_logical_partition_semantic_info(derez);
              break;
            }
          case SEND_FREE_REMOTE_CONTEXT:
            {
              runtime->handle_free_remote_context(derez);
              break;
            }
          case SEND_VALIDATE_REMOTE_STATE:
            {
              runtime->handle_validate_remote_state(derez, 
                                                    remote_address_space);
              break;
            }
          case SEND_INVALIDATE_REMOTE_STATE:
            {
              runtime->handle_invalidate_remote_state(derez,
                                                      remote_address_space);
              break;
            }
          default:
            assert(false); // should never get here
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
    void MessageManager::buffer_messages(unsigned num_messages,
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
    Event MessageManager::notify_pending_shutdown(void)
    //--------------------------------------------------------------------------
    {
      return last_message_event;
    }

    /////////////////////////////////////////////////////////////
    // Garbage Collection Epoch 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GarbageCollectionEpoch::GarbageCollectionEpoch(Runtime *rt)
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
        view->add_gc_reference();
        collections[view].insert(term);
      }
      else
        finder->second.insert(term);
    }

    //--------------------------------------------------------------------------
    void GarbageCollectionEpoch::launch(Processor utility, int priority)
    //--------------------------------------------------------------------------
    {
      // Set remaining to the total number of collections
      remaining = collections.size();
      GarbageCollectionArgs args;
      args.hlr_id = HLR_DEFERRED_COLLECT_ID;
      args.epoch = this;
      for (std::map<LogicalView*,std::set<Event> >::const_iterator it =
            collections.begin(); it != collections.end(); /*nothing*/)
      {
        args.view = it->first;
        Event precondition = Event::merge_events(it->second);
        // Avoid the deletion race by testing the condition 
        // before launching the task
        it++;
        bool done = (it == collections.end());
        utility.spawn(HLR_TASK_ID, &args, sizeof(args), precondition, priority);
        if (done)
          break;
      }
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
    // Legion Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(Machine m, AddressSpaceID unique,
                     const std::set<Processor> &locals,
                     const std::set<Processor> &local_utilities,
                     const std::set<AddressSpaceID> &address_spaces,
                     const std::map<Processor,AddressSpaceID> &processor_spaces,
                     Processor cleanup, Processor gc, Processor message)
      : high_level(new HighLevelRuntime(this)), machine(m), 
        address_space(unique), runtime_stride(address_spaces.size()),
        forest(new RegionTreeForest(this)), outstanding_top_level_tasks(1),
#ifdef SPECIALIZED_UTIL_PROCS
        cleanup_proc(cleanup), gc_proc(gc), message_proc(message),
#endif
        local_procs(locals), local_utils(local_utilities),
        memory_manager_lock(Reservation::create_reservation()),
        message_manager_lock(Reservation::create_reservation()),
        proc_spaces(processor_spaces),
        mapper_info_lock(Reservation::create_reservation()),
        unique_index_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_partition_id((unique == 0) ? runtime_stride : unique), 
        unique_field_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_tree_id((unique == 0) ? runtime_stride : unique),
        unique_region_tree_id((unique == 0) ? runtime_stride : unique),
        unique_operation_id((unique == 0) ? runtime_stride : unique),
        unique_field_id((unique == 0) ? runtime_stride : unique),
        available_lock(Reservation::create_reservation()), total_contexts(0),
        group_lock(Reservation::create_reservation()),
        distributed_id_lock(Reservation::create_reservation()),
        unique_distributed_id((unique == 0) ? runtime_stride : unique),
        distributed_collectable_lock(Reservation::create_reservation()),
        hierarchical_collectable_lock(Reservation::create_reservation()),
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
        post_close_op_lock(Reservation::create_reservation()),
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
        fill_op_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      log_run.debug("Initializing high-level runtime in address space %x",
                            address_space);
#ifdef LEGION_LOGGING
      // Initialize a logger if we have one
      {
        std::set<Processor> all_locals(local_procs.begin(), local_procs.end());
        all_locals.insert(local_utils.begin(), local_utils.end());
        LegionLogging::initialize_legion_logging(unique, all_locals);
      }
#endif
#ifdef LEGION_PROF
      {
        // See if we should disable profiling on this node
        if (Runtime::num_profiling_nodes == 0)
          LegionProf::disable_profiling();
        else if (Runtime::num_profiling_nodes > 0)
        {
          unsigned address_space_idx = 0;
          for (std::set<AddressSpaceID>::const_iterator it = 
                address_spaces.begin(); it != address_spaces.end(); it++)
          {
            if (address_space == (*it))
              break;
            address_space_idx++;
          }
          if (address_space_idx >= unsigned(Runtime::num_profiling_nodes))
            LegionProf::disable_profiling();
          else
            LegionProf::enable_profiling();
        }
        // If it's less than zero, then they are all enabled by default
        else
          LegionProf::enable_profiling();
        const std::map<Processor::TaskFuncID,TaskVariantCollection*>& table =
          Runtime::get_collection_table();
        for (std::map<Processor::TaskFuncID,TaskVariantCollection*>::
              const_iterator it = table.begin(); it != table.end(); it++)
        {
          LegionProf::register_task_variant(it->first, it->second->name);
        }
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          Processor::Kind kind = it->kind();
          assert(kind != Processor::UTIL_PROC);
          LegionProf::initialize_processor(*it, false/*util*/, kind);
        }
        for (std::set<Processor>::const_iterator it = local_utils.begin();
              it != local_utils.end(); it++)
        {
          Processor::Kind kind = it->kind();
          assert(kind == Processor::UTIL_PROC);
          LegionProf::initialize_processor(*it, true/*util*/, kind);
        }
        // Tell the profiler about all the memories and their kinds
        std::set<Memory> all_mems;
	machine.get_all_memories(all_mems);
        for (std::set<Memory>::const_iterator it = all_mems.begin();
              it != all_mems.end(); it++)
        {
          Memory::Kind kind = it->kind();
          LegionProf::initialize_memory(*it, kind);
        } 
        LegionProf::initialize_copy_processor();
      }
#endif
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

      // Before launching the top level task, see if the user requested
      // a callback to be performed before starting the application
      if (Runtime::registration_callback != NULL)
        (*Runtime::registration_callback)(machine, high_level, 
                                                local_procs);
    }

    //--------------------------------------------------------------------------
    Runtime::Runtime(const Runtime &rhs)
      : high_level(NULL), machine(rhs.machine), address_space(0), 
        runtime_stride(0), forest(NULL),
        local_procs(rhs.local_procs), proc_spaces(rhs.proc_spaces)
#ifdef SPECIALIZE_UTIL_PROCS
        , cleanup_proc(Processor::NO_PROC), gc_proc(Processor::NO_PROC),
        message_proc(Processor::NO_PROC)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Runtime::~Runtime(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      {
        std::set<Processor> all_procs;
        all_procs.insert(local_procs.begin(), local_procs.end());
        all_procs.insert(local_utils.begin(), local_utils.end());
        LegionLogging::finalize_legion_logging(all_procs);
      }
#endif
#ifdef LEGION_PROF
      {
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
#ifndef NDEBUG
          Processor::Kind kind = it->kind();
#endif
          assert(kind != Processor::UTIL_PROC);
          LegionProf::finalize_processor(*it);
        }
        for (std::set<Processor>::const_iterator it = local_utils.begin();
              it != local_utils.end(); it++)
        {
#ifndef NDEBUG
          Processor::Kind kind = it->kind();
#endif
          assert(kind == Processor::UTIL_PROC);
          LegionProf::finalize_processor(*it);
        }
        LegionProf::finalize_copy_profiler();
      }
#endif
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
      hierarchical_collectable_lock.destroy_reservation();
      hierarchical_collectable_lock = Reservation::NO_RESERVATION;
      gc_epoch_lock.destroy_reservation();
      gc_epoch_lock = Reservation::NO_RESERVATION;
      future_lock.destroy_reservation();
      future_lock = Reservation::NO_RESERVATION;
      remote_lock.destroy_reservation();
      remote_lock = Reservation::NO_RESERVATION;
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
      for (std::deque<PostCloseOp*>::const_iterator it = 
            available_post_close_ops.begin(); it !=
            available_post_close_ops.end(); it++)
      {
        legion_delete(*it);
      }
      available_post_close_ops.clear();
      post_close_op_lock.destroy_reservation();
      post_close_op_lock = Reservation::NO_RESERVATION;
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

      delete forest;

#ifdef DEBUG_HIGH_LEVEL
      if (logging_region_tree_state)
	delete tree_state_logger;
#endif
#ifdef TRACE_ALLOCATION
      allocation_lock.destroy_reservation();
      allocation_lock = Reservation::NO_RESERVATION;
      allocation_manager.clear();
#endif
    }

    //--------------------------------------------------------------------------
    Runtime& Runtime::operator=(const Runtime &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void Runtime::construct_mpi_rank_tables(Processor proc, int rank)
    //--------------------------------------------------------------------------
    {
      // Only do this on the first processor
      if (proc == *(local_procs.begin()))
      {
        // Initialize our mpi rank event
        Runtime::mpi_rank_event = UserEvent::create_user_event();
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
          it->spawn(HLR_TASK_ID, &args, sizeof(args));
          sent_targets.insert(target_space);
        }
        // Now set our own value, update the count, and see if we're done
        Runtime::mpi_rank_table[rank] = address_space;
        unsigned count = 
          __sync_add_and_fetch(&Runtime::remaining_mpi_notifications, 1);
        const size_t total_ranks = machine.get_address_space_count();
        if (count == total_ranks)
          Runtime::mpi_rank_event.trigger();
        // Wait on the event
        mpi_rank_event.wait(false/*block*/);
        // Once we've triggered, then we can build the maps
        for (unsigned local_rank = 0; local_rank < count; local_rank++)
        {
          AddressSpace local_space = Runtime::mpi_rank_table[local_rank];
#ifdef DEBUG_HIGH_LEVEL
          assert(reverse_mpi_mapping.find(local_space) == 
                 reverse_mpi_mapping.end());
#endif
          forward_mpi_mapping[local_rank] = local_space;
          reverse_mpi_mapping[local_space] = local_rank;
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::launch_top_level_task(Processor proc)
    //--------------------------------------------------------------------------
    {
      // Check to see if we should launch the top-level task
      // If we're the first address space and the first of
      // the local processors
      if ((address_space == 0) && (proc == *(local_procs.begin())))
      {
#ifdef LEGION_LOGGING
        perform_one_time_logging();
#endif
        // Get an individual task to be the top-level task
        IndividualTask *top_task = get_available_individual_task();
        // Get a remote task to serve as the top of the top-level task
        RemoteTask *top_context = find_or_init_remote_context(0/*fake uid*/);
        // Set the executing processor
        top_context->set_executing_processor(proc);
        TaskLauncher launcher(Runtime::legion_main_id, TaskArgument());
        top_task->initialize_task(top_context, launcher, 
                                  false/*check priv*/, false/*track parent*/);
        // Mark that this task is the top-level task
        top_task->top_level_task = true;
        // Set up the input arguments
        top_task->arglen = sizeof(InputArgs);
        top_task->args = malloc(top_task->arglen);
        top_task->depth = 0;
        memcpy(top_task->args,&Runtime::get_input_args(),top_task->arglen);
#ifdef DEBUG_HIGH_LEVEL
        assert(proc_managers.find(proc) != proc_managers.end());
#endif
        proc_managers[proc]->invoke_mapper_set_task_options(top_task);
        invoke_mapper_configure_context(proc, top_task);
#ifdef LEGION_LOGGING
        LegionLogging::log_top_level_task(Runtime::legion_main_id,
                                          top_task->get_unique_task_id());
#endif
#ifdef LEGION_SPY
        Runtime::log_machine(machine);
        LegionSpy::log_top_level_task(Runtime::legion_main_id,
                                      top_task->get_unique_task_id(),
                                      top_task->variants->name);
#endif
        // Put the task in the ready queue
        add_to_ready_queue(proc, top_task, false/*prev failure*/);
      }
    }

    //--------------------------------------------------------------------------
    Event Runtime::launch_mapper_task(Mapper *mapper, Processor proc, 
                                      Processor::TaskFuncID tid,
                                      const TaskArgument &arg, MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Get an individual task to be the top-level task
      IndividualTask *mapper_task = get_available_individual_task();
      // Get a remote task to serve as the top of the top-level task
      RemoteTask *map_context = find_or_init_remote_context(0/*fake uid*/);
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
      f.impl->add_gc_reference();
      // Create a meta-task to return the results to the mapper
      MapperTaskArgs args;
      args.hlr_id = HLR_MAPPER_TASK_ID;
      args.future = f.impl;
      args.map_id = map_id;
      args.proc = proc;
      args.event = result;
      Processor util = find_utility_group();
      Event pre = f.impl->get_ready_event();
      Event post = util.spawn(HLR_TASK_ID, &args, sizeof(args), pre);
      // Chain the events properly
      result.trigger(post);
      // Mark that we have another outstanding top level task
      increment_outstanding_top_level_tasks();
      // Now we can put it on the queue
      add_to_ready_queue(proc, mapper_task, false/*prev failure*/);
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::perform_one_time_logging(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      // First log information about the machine 
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      // Log all the memories
      std::set<Memory> all_mems;
      machine.get_all_memories(all_mems);
      for (std::set<Memory>::const_iterator it = all_mems.begin();
            it != all_mems.end(); it++)
      {
        Memory::Kind kind = (*it).kind();
        size_t mem_size = (*it).capacity();
        LegionLogging::log_memory(*it, kind, mem_size);
      }
      // Log processor-memory affinities
      for (std::set<Processor>::const_iterator pit = all_procs.begin();
            pit != all_procs.end(); pit++)
      {
        std::vector<ProcessorMemoryAffinity> affinities;
        machine.get_proc_mem_affinity(affinities, *pit);
        for (std::vector<ProcessorMemoryAffinity>::const_iterator it = 
              affinities.begin(); it != affinities.end(); it++)
        {
          LegionLogging::log_proc_mem_affinity(*pit, it->m,
                                               it->bandwidth,
                                               it->latency);
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
          LegionLogging::log_mem_mem_affinity(it->m1, it->m2,
                                              it->bandwidth,
                                              it->latency);
        }
      }
      // Log information about tasks and their variants
      const std::map<Processor::TaskFuncID,TaskVariantCollection*> &table = 
        Runtime::get_collection_table();
      for (std::map<Processor::TaskFuncID,TaskVariantCollection*>::
            const_iterator it = table.begin(); it != table.end(); it++)
      {
        // Leaf task properties are now on variants and not collections
        LegionLogging::log_task_collection(it->first, false/*leaf*/,
                                           it->second->idempotent,
                                           it->second->name);
        const std::map<VariantID,TaskVariantCollection::Variant> &all_vars = 
                                          it->second->get_all_variants();
        for (std::map<VariantID,TaskVariantCollection::Variant>::const_iterator 
              vit = all_vars.begin(); vit != all_vars.end(); vit++)
        {
          LegionLogging::log_task_variant(it->first,
                                          vit->second.proc_kind,
                                          vit->second.single_task,
                                          vit->second.index_space,
                                          vit->first);
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, size_t max_num_elmts)
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
      log_index.debug("Creating index space " IDFMT " in task %s "
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
#ifdef LEGION_LOGGING
      LegionLogging::log_top_index_space(ctx->get_executing_processor(),
                                         handle);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(handle.id);
#endif
      LowLevel::IndexSpace space = 
                      LowLevel::IndexSpace::create_index_space(max_num_elmts);
      forest->create_index_space(handle, Domain(space), 
                                 UNSTRUCTURED_KIND, MUTABLE);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, Domain domain)
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
      log_index.debug("Creating dummy index space " IDFMT " in task %s "
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
#ifdef LEGION_LOGGING
      LegionLogging::log_top_index_space(ctx->get_executing_processor(),
                                         handle);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(handle.id);
#endif
      forest->create_index_space(handle, domain, DENSE_ARRAY_KIND, NO_MEMORY);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, 
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
      log_index.debug("Creating dummy index space " IDFMT " in task %s "
                            "(ID %lld) for domain", 
                            handle.id, ctx->variants->name,
                            ctx->get_unique_task_id());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_top_index_space(ctx->get_executing_processor(),
                                         handle);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(handle.id);
#endif
      forest->create_index_space(handle, hull, domains,
                                 DENSE_ARRAY_KIND, NO_MEMORY);
      ctx->register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
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
      log_index.debug("Destroying index space " IDFMT " in task %s "
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
      DeletionOp *op = get_available_deletion_op();
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
    bool Runtime::finalize_index_space_destroy(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->destroy_index_space(handle, address_space);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      for (std::map<DomainPoint,ColoredPoints<ptr_t> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        LowLevel::ElementMask child_mask(num_elmts);
        const ColoredPoints<ptr_t> &pcoloring = it->second;
        for (std::set<ptr_t>::const_iterator pit = pcoloring.points.begin();
              pit != pcoloring.points.end(); pit++)
        {
          child_mask.enable(*pit,1);
        }
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator pit = 
              pcoloring.ranges.begin(); pit != pcoloring.ranges.end(); pit++)
        {
          child_mask.enable(pit->first.value, pit->second - pit->first + 1);
        }
        LowLevel::IndexSpace child_space = 
          LowLevel::IndexSpace::create_index_space(
                          parent_dom.get_index_space(), child_mask, allocable);
        new_index_spaces[it->first] = Domain(child_space);
      }
#ifdef DEBUG_HIGH_LEVEL
      if ((part_kind == DISJOINT_KIND) && verify_disjointness)
        validate_unstructured_disjointness(pid, new_index_spaces);
#endif
      ColorPoint partition_color;
      // If we have a valid color, set it now
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_index_spaces, color_space, part_kind, 
                                     allocable ? MUTABLE : NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      for (GenericPointInRectIterator<1> pir(color_range); pir; pir++)
      {
        LowLevel::ElementMask child_mask(num_elmts);
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
            child_mask.enable(*it,1);
          }
          for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
                pcoloring.ranges.begin(); it != pcoloring.ranges.end(); it++)
          {
            child_mask.enable(it->first.value, it->second-it->first+1);
          }
        }
        // Now make the index space and save the information
        LowLevel::IndexSpace child_space = 
          LowLevel::IndexSpace::create_index_space(
                          parent_dom.get_index_space(), child_mask);
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
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, disjoint,
                                         part_color);
      for (std::map<ColorPoint,Domain>::const_iterator it = 
            new_index_spaces.begin(); it != new_index_spaces.end(); it++)
      {
        LegionLogging::log_index_subspace(ctx->get_executing_processor(),
                                          pid, it->second.get_index_space(),
                                          it->first.get_index());
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     coloring, color_space, 
                                     part_kind, NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, disjoint,
                                         part_color);
      for (std::map<Color,Domain>::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        IndexSpace subspace = get_index_subspace(ctx, pid, it->first);
        LegionLogging::log_index_subspace(ctx->get_executing_processor(),
                                          pid, subspace,
                                          it->first);
      }
#endif 
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     convex_hulls, coloring,
                                     color_space, part_kind, NO_MEMORY);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, disjoint,
                                         part_color);
      for (std::map<Color,std::set<Domain> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        IndexSpace subspace = get_index_subspace(ctx, pid, it->first);
        LegionLogging::log_index_subspace(ctx->get_executing_processor(),
                                          pid, subspace,
                                          it->first);
      }
#endif 
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_index_partition(
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
        std::map<Color,LowLevel::ElementMask> child_masks;
        Domain parent_dom = forest->get_index_space_domain(parent);
        size_t parent_elmts = 
          parent_dom.get_index_space().get_valid_mask().get_num_elmts();
        for (Domain::DomainPointIterator itr(parent_dom); itr; itr++)
        {
          ptr_t cur_ptr = itr.p.get_index();
          int c = fa_coloring.read(cur_ptr);
          // Ignore all colors less than zero
          if (c >= 0)
          {
            Color color = (Color)c; 
            std::map<Color,LowLevel::ElementMask>::iterator finder = 
              child_masks.find(color);
            // Haven't made an index space for this color yet
            if (finder == child_masks.end())
            {
              child_masks[color] = LowLevel::ElementMask(parent_elmts);
              finder = child_masks.find(color);
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != child_masks.end());
#endif
            finder->second.enable(cur_ptr);
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
          std::map<Color,LowLevel::ElementMask>::const_iterator finder = 
            child_masks.find(c);
          LowLevel::IndexSpace child_space;
          if (finder != child_masks.end())
          {
            child_space = 
              LowLevel::IndexSpace::create_index_space(
                    parent_dom.get_index_space(), finder->second);
          }
          else
          {
            LowLevel::ElementMask empty_mask;
            child_space = 
              LowLevel::IndexSpace::create_index_space(
                    parent_dom.get_index_space(), empty_mask);
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
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, true/*disjoint*/,
                                         part_color);
      for (std::map<DomainPoint,Domain>::const_iterator it = 
            new_index_spaces.begin(); it != new_index_spaces.end(); it++)
      {
        IndexSpace subspace = get_index_subspace(ctx, pid, it->first);
        LegionLogging::log_index_subspace(ctx->get_executing_processor(),
                                          pid, subspace, it->first);
      }
#endif
      return pid;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_partition(Context ctx, 
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
      DeletionOp *op = get_available_deletion_op();
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
    void Runtime::finalize_index_partition_destroy(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_index_partition(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Runtime::validate_unstructured_disjointness(IndexPartition pid,
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
          const LowLevel::ElementMask &em1 = 
            it1->second.get_index_space().get_valid_mask();
          const LowLevel::ElementMask &em2 = 
            it2->second.get_index_space().get_valid_mask();
          LowLevel::ElementMask::OverlapResult result = 
            em1.overlaps_with(em2, 1/*effort level*/);
          if (result == LowLevel::ElementMask::OVERLAP_YES)
          {
            log_run.error("ERROR: colors %d and %d of partition %d "
                            "are not disjoint when they were claimed to be!",
                      it1->first.get_index(), it2->first.get_index(), pid.id);
            assert(false);
            exit(ERROR_DISJOINTNESS_TEST_FAILURE);
          }
          else if (result == LowLevel::ElementMask::OVERLAP_MAYBE)
          {
            log_run.warning("WARNING: colors %d and %d of partition "
                        "%d may not be disjoint when they were claimed to be!"
                        "(At least according to the low-level runtime.  You "
                        "might also try telling the the low-level runtime "
                        "to stop being lazy and try harder.)", 
                      it1->first.get_index(), it2->first.get_index(), pid.id);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::validate_structured_disjointness(IndexPartition pid,
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
                                  it1->first[0], it2->first[0], pid.id);
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
                            it1->first[0], it1->first[1],
                            it2->first[0], it2->first[1], pid.id);
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
                            it1->first[0], it1->first[1], it1->first[2],
                    it2->first[0], it2->first[1], it2->first[2], pid.id);
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
    void Runtime::validate_multi_structured_disjointness(IndexPartition pid,
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
                                         it1->first[0], it2->first[0], pid.id);
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
                                         it1->first[0], it1->first[1],
                                         it2->first[0], it2->first[1], pid.id);
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
                           it1->first[0], it1->first[1], it1->first[2], 
                           it2->first[0], it2->first[1], it2->first[2], pid.id);
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
    Domain Runtime::construct_convex_hull(const std::set<Domain> &domains)
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
    IndexPartition Runtime::create_equal_partition(Context ctx, 
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    IndexPartition Runtime::create_weighted_partition(Context ctx, 
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    IndexPartition Runtime::create_partition_by_union(Context ctx, 
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
                            "space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          LowLevel::IndexSpace::ISO_UNION);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    IndexPartition Runtime::create_partition_by_intersection(Context ctx, 
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
                            "index space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          LowLevel::IndexSpace::ISO_INTERSECT);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    IndexPartition Runtime::create_partition_by_difference(Context ctx, 
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
                            "index space " IDFMT " in task %s (ID %lld)", 
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      Domain color_space;
      forest->compute_pending_color_space(parent, handle1, handle2, color_space,
                                          LowLevel::IndexSpace::ISO_SUBTRACT);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    void Runtime::create_cross_product_partition(Context ctx,
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
      if (color != AUTO_GENERATE_ID)
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = get_available_pending_partition_op();
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
    IndexPartition Runtime::create_partition_by_field(Context ctx,
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
      if (color != AUTO_GENERATE_ID)
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = get_available_dependent_partition_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
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
    IndexPartition Runtime::create_partition_by_image(Context ctx,
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
      if (color != AUTO_GENERATE_ID)
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = get_available_dependent_partition_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
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
    IndexPartition Runtime::create_partition_by_preimage(Context ctx,
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
      if (color != AUTO_GENERATE_ID)
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = get_available_dependent_partition_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
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
    IndexPartition Runtime::create_pending_partition(Context ctx, 
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
      if (color != AUTO_GENERATE_ID)
        part_color = ColorPoint(color);
      forest->create_pending_partition(pid, parent, color_space, part_color,
                                       part_kind, allocable, Event::NO_EVENT,
                                       Event::NO_EVENT, true/*separate*/);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx, 
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
      PendingPartitionOp *part_op = get_available_pending_partition_op();
      part_op->initialize_index_space_union(ctx, result, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
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
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
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
      PendingPartitionOp *part_op = get_available_pending_partition_op();
      part_op->initialize_index_space_union(ctx, result, handle);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
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
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
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
      PendingPartitionOp *part_op = get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(ctx, result, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
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
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
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
      PendingPartitionOp *part_op = get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(ctx, result, handle);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
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
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_difference(Context ctx,
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
      PendingPartitionOp *part_op = get_available_pending_partition_op();
      part_op->initialize_index_space_difference(ctx, result, initial, handles);
      handle_ready.trigger(part_op->get_handle_ready());
      domain_ready.trigger(part_op->get_completion_event());
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
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(parent, "get_index_partition");
#endif
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
    IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition(parent, ColorPoint(color));
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subpartition(p, "get_index_subspace");
#endif
      IndexSpace result = forest->get_index_subspace(p, ColorPoint(color));
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid color %d for get index subspace", 
                                color);
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_COLOR); 
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(IndexPartition p, Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_subspace(p, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_multiple_domains(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "has_multiple_domains");
#endif
      return forest->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_multiple_domains(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_multiple_domains(handle);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "get_index_space_domain");
#endif
      Domain result = forest->get_index_space_domain(handle);
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_index.error("Invalid handle " IDFMT " for get index space "
                               "domain", 
                                handle.id);
        assert(false);
        exit(ERROR_INVALID_INDEX_DOMAIN);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_domain(handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domains(Context ctx, IndexSpace handle,
                                          std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "get_index_space_domains");
#endif
      forest->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domains(IndexSpace handle,
                                          std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_domains(handle, domains);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(Context ctx, 
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subpartition(p, "get_index_partition_color_space");
#endif
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
    Domain Runtime::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(sp, "get_index_space_partition_colors");
#endif
      std::set<ColorPoint> color_points;
      forest->get_index_space_partition_colors(sp, color_points);
      for (std::set<ColorPoint>::const_iterator it = color_points.begin();
            it != color_points.end(); it++)
      {
        colors.insert(it->get_index());
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(IndexSpace handle,
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
    bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subpartition(p, "is_index_partition_disjoint");
#endif
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "get_index_space_color");
#endif
      return forest->get_index_space_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subpartition(handle, "get_index_partition_color");
#endif
      return forest->get_index_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(Context ctx,   
                                               IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subpartition(handle, "get_parent_index_space");
#endif
      return forest->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_space(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "has_parent_index_partition");
#endif
      return forest->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(Context ctx,
                                                       IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_index_subspace(handle, "get_parent_index_partition");
#endif
      return forest->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(region, "safe_cast");
#endif
      if (pointer.is_null())
        return pointer;
      return ctx->perform_safe_cast(region.get_index_space(), pointer);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(region, "safe_cast");
#endif
      if (point.is_null())
        return point;
      return ctx->perform_safe_cast(region.get_index_space(), point);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::create_field_space(Context ctx)
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
#ifdef LEGION_SPY
      LegionSpy::log_field_space(space.id);
#endif
      forest->create_field_space(space);
      ctx->register_field_space_creation(space);
#ifdef LEGION_LOGGING
      LegionLogging::log_field_space(ctx->get_executing_processor(), space);
#endif
      return space;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_field_space(Context ctx, FieldSpace handle)
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
      DeletionOp *op = get_available_deletion_op();
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
    size_t Runtime::get_field_size(Context ctx, FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_field_space(handle, "get_field_size");
#endif
      return forest->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return forest->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_field_space_destroy(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_field_space(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_field_destroy(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      forest->free_field(handle, fid, address_space);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_field_destroy(FieldSpace handle, 
                                               const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      forest->free_fields(handle, to_free, address_space);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::create_logical_region(Context ctx, 
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
                              "with index space " IDFMT " and field space %x "
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
#ifdef LEGION_SPY
      LegionSpy::log_top_region(index_space.id, field_space.id, tid);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_top_region(ctx->get_executing_processor(),
                                    index_space, field_space, tid);
#endif
      forest->create_logical_region(region);
      // Register the creation of a top-level region with the context
      ctx->register_region_creation(region);
      return region;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_region(Context ctx, 
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
      log_region.debug("Deleting logical region (" IDFMT ",%x) in "
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
      DeletionOp *op = get_available_deletion_op();
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
    void Runtime::destroy_logical_partition(Context ctx, 
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
      DeletionOp *op = get_available_deletion_op();
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
    bool Runtime::finalize_logical_region_destroy(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->destroy_logical_region(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_logical_partition_destroy(
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_logical_partition(handle, address_space);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(Context ctx, 
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(parent, "get_logical_partition");
#endif
      return forest->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(LogicalRegion parent,
                                                    IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(parent, "get_logical_partition_by_color");
#endif
      return forest->get_logical_partition_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(LogicalRegion par,
                                                             Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
      {
        ctx->check_index_subpartition(handle, "get_logical_partition_by_tree");
        ctx->check_field_space(fspace, "get_logical_partition_by_tree");
      }
#endif
      return forest->get_logical_partition_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(IndexPartition part,
                                                            FieldSpace fspace,
                                                            RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_tree(part, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subpartition(parent, "get_logical_subregion");
#endif
      return forest->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(LogicalPartition parent,
                                                 IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subpartition(parent, 
                                        "get_logical_subregion_by_color");
#endif
      return forest->get_logical_subregion_by_color(parent, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(LogicalPartition par,
                                                          Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(par, ColorPoint(c));
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
      {
        ctx->check_index_subspace(handle, "get_logical_subregion_by_tree");
        ctx->check_field_space(fspace, "get_logical_subregion_by_tree");
      }
#endif
      return forest->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(IndexSpace handle,
                                                         FieldSpace fspace,
                                                         RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(handle, "get_logical_region_color");
#endif
      return forest->get_logical_region_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(Context ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subpartition(handle, "get_logical_partition_color");
#endif
      return forest->get_logical_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle).get_index();
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subpartition(handle, "get_parent_logical_region");
#endif
      return forest->get_parent_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(Context ctx, 
                                               LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(handle, "has_parent_logical_partition");
#endif
      return forest->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->has_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(Context ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx != DUMMY_CONTEXT)
        ctx->check_logical_subregion(handle, "get_parent_logical_partition");
#endif
      return forest->get_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexAllocator Runtime::create_index_allocator(Context ctx, 
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
    FieldAllocator Runtime::create_field_allocator(Context ctx, 
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
    ArgumentMap Runtime::create_argument_map(Context ctx)
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
    Future Runtime::execute_task(Context ctx, 
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
          get_available_distributed_id(), address_space, address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskVariantCollection *variants = 
            get_variant_collection(launcher.task_id);
          if (variants->return_size > 0)
          {
            log_run.error("Predicated task launch for task %s "
                                "in parent task %s (UID %lld) has non-void "
                                "return type but no default value for its "
                                "future if the task predicate evaluates to "
                                "false.  Please set either the "
                                "'predicate_false_result' or "
                                "'predicate_false_future' fields of the "
                                "TaskLauncher struct.",
                                variants->name, ctx->variants->name,
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
      IndividualTask *task = get_available_individual_task();
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
    FutureMap Runtime::execute_index_space(Context ctx, 
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
            launcher.predicate_false_future.impl->add_gc_reference();
            DeferredFutureMapSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_MAP_SET_ID;
            args.future_map = result;
            args.result = launcher.predicate_false_future.impl;
            args.domain = launcher.launch_domain;
            utility_group.spawn(HLR_TASK_ID, &args, sizeof(args), ready_event);
          }
          return FutureMap(result);
        }
        if (launcher.predicate_false_result.get_size() == 0)
        {
          // Check to make sure the task actually does expect to
          // have a void return type
          TaskVariantCollection *variants = 
            get_variant_collection(launcher.task_id);
          if (variants->return_size > 0)
          {
            log_run.error("Predicated index task launch for task %s "
                                "in parent task %s (UID %lld) has non-void "
                                "return type but no default value for its "
                                "future if the task predicate evaluates to "
                                "false.  Please set either the "
                                "'predicate_false_result' or "
                                "'predicate_false_future' fields of the "
                                "IndexLauncher struct.",
                                variants->name, ctx->variants->name,
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
      IndexTask *task = get_available_index_task();
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
    Future Runtime::execute_index_space(Context ctx, 
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
          get_available_distributed_id(), address_space, address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskVariantCollection *variants = 
            get_variant_collection(launcher.task_id);
          if (variants->return_size > 0)
          {
            log_run.error("Predicated index task launch for task %s "
                                "in parent task %s (UID %lld) has non-void "
                                "return type but no default value for its "
                                "future if the task predicate evaluates to "
                                "false.  Please set either the "
                                "'predicate_false_result' or "
                                "'predicate_false_future' fields of the "
                                "IndexLauncher struct.",
                                variants->name, ctx->variants->name,
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
      IndexTask *task = get_available_index_task();
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
    Future Runtime::execute_task(Context ctx, 
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
          get_available_distributed_id(), address_space, address_space));
      IndividualTask *task = get_available_individual_task();
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
    FutureMap Runtime::execute_index_space(Context ctx, 
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
      IndexTask *task = get_available_index_task();
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
    Future Runtime::execute_index_space(Context ctx, 
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
          get_available_distributed_id(), address_space, address_space));
      IndexTask *task = get_available_index_task();
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
    PhysicalRegion Runtime::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      MapOp *map_op = get_available_map_op();
#ifdef DEBUG_HIGH_LEVEL
      PhysicalRegion result = map_op->initialize(ctx, launcher, 
                                                 check_privileges);
      log_run.debug("Registering a map operation for region (" IDFMT 
                           ",%x,%x) in task %s (ID %lld)",
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
        log_run.error("Attempted an inline mapping of region (" IDFMT 
                            ",%x,%x) that conflicts with mapped region (" 
                            IDFMT ",%x,%x) at index %d of parent task %s "
                            "(ID %lld) that would ultimately result in "
                            "deadlock.  Instead you receive this error "
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
        log_run.error("Attempted an inline mapping of region (" 
                            IDFMT ",%x,%x) "
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
    PhysicalRegion Runtime::map_region(Context ctx, 
                    const RegionRequirement &req, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      MapOp *map_op = get_available_map_op();
#ifdef DEBUG_HIGH_LEVEL
      PhysicalRegion result = map_op->initialize(ctx, req, id, tag, 
                                                 check_privileges);
      log_run.debug("Registering a map operation for region (" 
                            IDFMT ",%x,%x) "
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
        log_run.error("Attempted an inline mapping of region (" 
                            IDFMT ",%x,%x) "
                            "that conflicts with mapped region (" 
                            IDFMT ",%x,%x) at "
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
        log_run.error("Attempted an inline mapping of region (" 
                            IDFMT ",%x,%x) "
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
    PhysicalRegion Runtime::map_region(Context ctx, unsigned idx, 
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
    void Runtime::remap_region(Context ctx, PhysicalRegion region)
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
      MapOp *map_op = get_available_map_op();
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
    void Runtime::unmap_region(Context ctx, PhysicalRegion region)
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
    void Runtime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      ctx->unmap_all_mapped_regions();
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_field(Context ctx, LogicalRegion handle,
                             LogicalRegion parent, FieldID fid,
                             const void *value, size_t value_size,
                             const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op();
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
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(),
                                           mapped_event);
        }
#endif
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
    void Runtime::fill_fields(Context ctx, LogicalRegion handle,
                              LogicalRegion parent,
                              const std::set<FieldID> &fields,
                              const void *value, size_t value_size,
                              const Predicate &pred)
    //--------------------------------------------------------------------------
    {
      FillOp *fill_op = get_available_fill_op();
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
      // Issue the copy operation
      add_to_dependence_queue(proc, fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(),
                                           mapped_event);
        }
#endif
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
    void Runtime::issue_copy_operation(Context ctx, 
                                       const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      CopyOp *copy_op = get_available_copy_op();  
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
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
      {
        std::set<Event> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(),
                                           mapped_event);
        }
#endif
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
    Predicate Runtime::create_predicate(Context ctx, const Future &f) 
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
      FuturePredOp *pred_op = get_available_future_pred_op();
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
    Predicate Runtime::predicate_not(Context ctx, const Predicate &p) 
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
      NotPredOp *pred_op = get_available_not_pred_op();
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
    Predicate Runtime::predicate_and(Context ctx, const Predicate &p1, 
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
      AndPredOp *pred_op = get_available_and_pred_op();
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
    Predicate Runtime::predicate_or(Context ctx, const Predicate &p1, 
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
      OrPredOp *pred_op = get_available_or_pred_op();
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
    Lock Runtime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      return Lock(Reservation::create_reservation());
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      ctx->destroy_user_lock(l.reservation_lock);
    }

    //--------------------------------------------------------------------------
    Grant Runtime::acquire_grant(Context ctx, 
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
    void Runtime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      grant.impl->release_grant();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::create_phase_barrier(Context ctx, unsigned arrivals) 
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
    void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
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
    PhaseBarrier Runtime::advance_phase_barrier(Context ctx, PhaseBarrier pb)
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
    DynamicCollective Runtime::create_dynamic_collective(Context ctx,
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
    void Runtime::destroy_dynamic_collective(Context ctx, DynamicCollective dc)
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
    void Runtime::arrive_dynamic_collective(Context ctx, DynamicCollective dc,
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
    void Runtime::defer_dynamic_collective_arrival(Context ctx, 
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
    Future Runtime::get_dynamic_collective_result(Context ctx, 
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
      DynamicCollectiveOp *collective = get_available_dynamic_collective_op();
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
    DynamicCollective Runtime::advance_dynamic_collective(Context ctx,
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
    void Runtime::issue_acquire(Context ctx, const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AcquireOp *acquire_op = get_available_acquire_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(),
                                           mapped_event);
        }
#endif
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
    void Runtime::issue_release(Context ctx, const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      ReleaseOp *release_op = get_available_release_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(),
                                           mapped_event);
        }
#endif
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
    void Runtime::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      FenceOp *fence_op = get_available_fence_op();
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
    void Runtime::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      FenceOp *fence_op = get_available_fence_op();
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
    void Runtime::begin_trace(Context ctx, TraceID tid)
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
      // Tracing does not work well with LegionSpy
#ifndef LEGION_SPY
      // Mark that we are starting a trace
      ctx->begin_trace(tid); 
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::end_trace(Context ctx, TraceID tid)
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
      // Tracing does not work well with LegionSpy
#ifndef LEGION_SPY
      // Mark that we are done with the trace
      ctx->end_trace(tid); 
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_frame(Context ctx)
    //--------------------------------------------------------------------------
    {
      FrameOp *frame_op = get_available_frame_op();
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
        log_task.error("Illegal Legion end trace call in leaf "
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
    FutureMap Runtime::execute_must_epoch(Context ctx, 
                                          const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      MustEpochOp *epoch_op = get_available_epoch_op();
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
          MapOp *op = get_available_map_op();
          op->initialize(ctx, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          add_to_dependence_queue(proc, op);
        }
        // Wait for all the re-mapping operations to complete
        Event mapped_event = Event::merge_events(mapped_events);
        if (!mapped_event.has_triggered())
        {
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_begin(proc,
                                               ctx->get_unique_task_id(), 
                                               mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_BEGIN_WAIT);
#endif
          pre_wait(proc);
          mapped_event.wait();
          post_wait(proc);
#ifdef LEGION_LOGGING
          LegionLogging::log_inline_wait_end(proc,
                                             ctx->get_unique_task_id(),
                                             mapped_event);
#endif
#ifdef LEGION_PROF
          LegionProf::register_event(ctx->get_unique_task_id(),
                                     PROF_END_WAIT);
#endif
        }
#ifdef LEGION_LOGGING
        else
        {
          LegionLogging::log_inline_nowait(proc,
                                           ctx->get_unique_task_id(), 
                                           mapped_event);
        }
#endif
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
    int Runtime::get_tunable_value(Context ctx, TunableID tid,
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
    Mapper* Runtime::get_mapper(Context ctx, MapperID id, Processor target)
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
    Processor Runtime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      return ctx->get_executing_processor();
    }

    //--------------------------------------------------------------------------
    void Runtime::raise_region_exception(Context ctx, 
                                               PhysicalRegion region, 
                                               bool nuclear)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& Runtime::find_forward_MPI_mapping(void)
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
    const std::map<AddressSpace,int>& Runtime::find_reverse_MPI_mapping(void)
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
    void Runtime::add_mapper(MapperID map_id, Mapper *mapper, 
                                      Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->add_mapper(map_id, mapper, true/*check*/);
      AutoLock m_lock(mapper_info_lock);
      mapper_infos[mapper] = MapperInfo(proc, map_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::replace_default_mapper(Mapper *mapper, 
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
    void Runtime::register_projection_functor(ProjectionID pid,
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
    ProjectionFunctor* Runtime::find_projection_functor(ProjectionID pid)
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
    void Runtime::attach_semantic_information(IndexSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, tag, node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, tag, node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, tag, node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle, FieldID fid,
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, fid, tag, 
                                          node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalRegion handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, tag, node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size)
    //--------------------------------------------------------------------------
    {
      NodeSet node_set;
      node_set.add(address_space);
      forest->attach_semantic_information(handle, tag, node_set, buffer, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(IndexSpace handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(IndexPartition handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(FieldSpace handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(FieldSpace handle, FieldID fid,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, fid, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(LogicalRegion handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    void Runtime::retrieve_semantic_information(LogicalPartition handle,
                                                SemanticTag tag,
                                                const void *&result, 
                                                size_t &size)
    //--------------------------------------------------------------------------
    {
      forest->retrieve_semantic_information(handle, tag, result, size);
    }

    //--------------------------------------------------------------------------
    FieldID Runtime::allocate_field(Context ctx, FieldSpace space,
                                          size_t field_size, FieldID fid,
                                          bool local)
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
#ifdef LEGION_SPY
      LegionSpy::log_field_creation(space.id, fid);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_field_creation(ctx->get_executing_processor(),
                                        space, fid, local);
#endif
      if (local)
        ctx->add_local_field(space, fid, field_size);
      else
      {
        forest->allocate_field(space, field_size, fid, local);
        ctx->register_field_creation(space, fid);
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field(Context ctx, FieldSpace space, FieldID fid)
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
      DeletionOp *op = get_available_deletion_op();
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
    void Runtime::allocate_fields(Context ctx, FieldSpace space,
                                        const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        bool local)
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
#ifdef LEGION_SPY
        LegionSpy::log_field_creation(space.id, resulting_fields[idx]);
#endif
#ifdef LEGION_LOGGING
        LegionLogging::log_field_creation(ctx->get_executing_processor(),
                                          space, resulting_fields[idx], local);
#endif
      }
      
      if (local)
        ctx->add_local_fields(space, resulting_fields, sizes);
      else
      {
        forest->allocate_fields(space, sizes, resulting_fields);
        ctx->register_field_creations(space, resulting_fields);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fields(Context ctx, FieldSpace space,
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
      DeletionOp *op = get_available_deletion_op();
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
    const std::vector<PhysicalRegion>& Runtime::begin_task(SingleTask *ctx)
    //--------------------------------------------------------------------------
    {
      return ctx->begin_task();
    }

    //--------------------------------------------------------------------------
    void Runtime::end_task(SingleTask *ctx, const void *result, 
                                 size_t result_size, bool owned)
    //--------------------------------------------------------------------------
    {
      ctx->end_task(result, result_size, owned);
    }

    //--------------------------------------------------------------------------
    const void* Runtime::get_local_args(SingleTask *ctx, 
                                        DomainPoint &point, size_t &local_size)
    //--------------------------------------------------------------------------
    {
      point = ctx->index_point;
      local_size = ctx->local_arglen;
      return ctx->local_args;
    }

    //--------------------------------------------------------------------------
    MemoryManager* Runtime::find_memory(Memory mem)
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
    void Runtime::allocate_physical_instance(PhysicalManager *instance)
    //--------------------------------------------------------------------------
    {
      find_memory(instance->memory)->allocate_physical_instance(instance);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_physical_instance(PhysicalManager *instance)
    //--------------------------------------------------------------------------
    {
      find_memory(instance->memory)->free_physical_instance(instance);
    }

    //--------------------------------------------------------------------------
    void Runtime::recycle_physical_instance(InstanceManager *inst)
    //--------------------------------------------------------------------------
    {
      find_memory(inst->memory)->recycle_physical_instance(inst);
    }

    //--------------------------------------------------------------------------
    bool Runtime::reclaim_physical_instance(InstanceManager *inst)
    //--------------------------------------------------------------------------
    {
      return find_memory(inst->memory)->reclaim_physical_instance(inst);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance Runtime::find_physical_instance(Memory mem, 
                                        size_t field_size, const Domain &dom, 
                                        const unsigned depth, Event &use_event)
    //--------------------------------------------------------------------------
    {
      return find_memory(mem)->find_physical_instance(field_size, 
                                                      dom, depth, use_event);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance Runtime::find_physical_instance(Memory mem,
        const std::vector<size_t> &field_sizes, const Domain &dom,
        const size_t blocking_factor, const unsigned depth, Event &use_event)
    //--------------------------------------------------------------------------
    {
      return find_memory(mem)->find_physical_instance(field_sizes, dom,
                                                      blocking_factor, 
                                                      depth, use_event);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::sample_allocated_space(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_allocated_space();
    }

    //--------------------------------------------------------------------------
    size_t Runtime::sample_free_space(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_free_space();
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::sample_allocated_instances(Memory mem)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory(mem);
      return manager->sample_allocated_instances();
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::sample_unmapped_tasks(Processor proc, Mapper *mapper)
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
    MessageManager* Runtime::find_messenger(AddressSpaceID sid)
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
    MessageManager* Runtime::find_messenger(Processor target)
    //--------------------------------------------------------------------------
    {
      return find_messenger(find_address_space(target));
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Runtime::find_address_space(Processor target) const
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
    void Runtime::send_task(Processor target, TaskOp *task)
    //--------------------------------------------------------------------------
    {
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
        // Put it on the queue and send it
        manager->send_task(rez, true/*flush*/);
        if (deactivate_task)
          task->deactivate();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_tasks(Processor target, 
                                   const std::set<TaskOp*> &tasks)
    //--------------------------------------------------------------------------
    {
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
          manager->send_task(rez, (idx == tasks.size()));
          // Deactivate the task if it is remote
          if (deactivate_task)
            (*it)->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_steal_request(
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
          manager->send_steal_request(rez, true/*flush*/);
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
    void Runtime::send_advertisements(const std::set<Processor> &targets,
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
          messenger->send_advertisement(rez, true/*flush*/);
          already_sent.insert(messenger);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_space_node(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_request(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_space_request(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_return(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_space_return(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_node(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_partition_node(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_request(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_partition_request(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_return(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_partition_return(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_space_node(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_space_request(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_return(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_space_return(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_node(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_logical_region_node(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_destruction(IndexSpace handle, 
                                               AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_index_space_destruction(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_destruction(IndexPartition handle, 
                                                   AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_index_partition_destruction(rez, 
                                                               false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_destruction(FieldSpace handle, 
                                               AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_field_space_destruction(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_destruction(LogicalRegion handle, 
                                                  AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_logical_region_destruction(rez, 
                                                              false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_partition_destruction(
                              LogicalPartition handle, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
      }
      find_messenger(target)->send_logical_partition_destruction(rez, 
                                                                false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_allocation(FieldSpace space, FieldID fid,
                                              size_t size, unsigned idx,
                                              AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(space);
        rez.serialize(fid);
        rez.serialize(size);
        rez.serialize(idx);
      }
      find_messenger(target)->send_field_allocation(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_destruction(FieldSpace space, FieldID fid,
                                         AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(space);
        rez.serialize(fid);
      }
      find_messenger(target)->send_field_destruction(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_mapped(Processor target,
                                        Serializer &rez, bool flush /*= true*/)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_individual_remote_mapped(rez, flush);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_complete(Processor target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_individual_remote_complete(rez, 
                                                              true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_commit(Processor target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_individual_remote_commit(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_mapped(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_slice_remote_mapped(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_complete(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_slice_remote_complete(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_commit(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_slice_remote_commit(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remove_distributed_resource(AddressSpaceID target,
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_remove_distributed_resource(rez,
                                                               true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remove_distributed_remote(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_remove_distributed_remote(rez, 
                                                             true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_add_distributed_remote(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_add_distributed_remote(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remove_hierarchical_resource(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_remove_hierarchical_resource(rez, 
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remove_hierarchical_remote(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_remove_hierarchical_remote(rez,
                                                              true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_user(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_user(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_atomic(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_atomic(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_subscriber(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_subscriber(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_materialized_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_materialized_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_materialized_update(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_materialized_update(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_materialized_view(AddressSpaceID target, 
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_materialized_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_composite_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_composite_view(rez, false/*flush*/);
    } 

    //--------------------------------------------------------------------------
    void Runtime::send_back_composite_view(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_composite_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_composite_update(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_composite_update(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_reduction_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_update(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_reduction_update(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_reduction_view(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_reduction_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_instance_manager(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_instance_manager(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_manager(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_reduction_manager(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_region_state(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_region_state(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_partition_state(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_partition_state(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_region_state(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_region_state(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_partition_state(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_partition_state(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_references(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_remote_references(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_request(AddressSpaceID target, 
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_individual_request(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_return(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_individual_return(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_request(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_slice_request(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_return(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_slice_return(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_future(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_result(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_future_result(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_subscription(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_future_subscription(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_make_persistent(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_make_persistent(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_mapper_message(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_mapper_message(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_mapper_broadcast(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_mapper_broadcast(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_space_semantic_info(rez,true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_semantic_info(AddressSpaceID target,
                                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_partition_semantic_info(rez,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_space_semantic_info(rez,true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_semantic_info(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_semantic_info(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_logical_region_semantic_info(rez,
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_partition_semantic_info(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_logical_partition_semantic_info(rez,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_free_remote_context(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_free_remote_context(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_validate_remote_state(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_validate_remote_state(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_invalidate_remote_state(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_invalidate_remote_state(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      TaskOp::process_unpack_task(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_steal(Deserializer &derez)
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
    void Runtime::handle_advertisement(Deserializer &derez)
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
    void Runtime::handle_index_space_node(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_return(derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_node(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_request(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_return(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_node(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_return(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_region_node(Deserializer &derez, 
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_destruction(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      forest->destroy_index_space(handle, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_destruction(Deserializer &derez,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      forest->destroy_index_partition(handle, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_destruction(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      forest->destroy_field_space(handle, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_region_destruction(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      forest->destroy_logical_region(handle, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_partition_destruction(Deserializer &derez,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalPartition handle;
      derez.deserialize(handle);
      forest->destroy_logical_partition(handle, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_allocation(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      size_t size;
      derez.deserialize(size);
      unsigned idx;
      derez.deserialize(idx);
      forest->allocate_field_index(handle, size, fid, idx, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_destruction(Deserializer &derez, 
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      forest->free_field(handle, fid, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_individual_remote_mapped(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_individual_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_individual_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_commit(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_slice_remote_mapped(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_slice_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_slice_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_distributed_remove_resource(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::process_remove_resource_reference(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_distributed_remove_remote(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::process_remove_remote_reference(this, source,
                                                              derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_distributed_add_remote(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::process_add_remote_reference(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_hierarchical_remove_resource(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      HierarchicalCollectable::process_remove_resource_reference(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_hierarchical_remove_remote(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      HierarchicalCollectable::process_remove_remote_reference(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_user(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalView::handle_send_back_user(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_atomic(Deserializer &derez,
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_back_atomic(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_subscriber(Deserializer &derez, 
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_send_subscriber(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_materialized_view(Deserializer &derez, 
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_materialized_view(forest, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_materialized_update(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_updates(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_materialized_view(Deserializer &derez,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_back_materialized_view(forest, derez, 
                                                           source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_composite_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      CompositeView::handle_send_composite_view(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_composite_view(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      CompositeView::handle_send_back_composite_view(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_composite_update(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      CompositeView::handle_send_composite_update(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_reduction_view(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_update(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_update(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_reduction_view(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_back_reduction_view(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_instance_manager(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceManager::handle_send_manager(forest, source, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_manager(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionManager::handle_send_manager(forest, source, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_region_state(Deserializer &derez, 
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_send_state(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_partition_state(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_send_state(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_region_state(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_send_back_state(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_partition_state(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_send_back_state(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_remote_references(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      forest->handle_remote_references(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_individual_request(Deserializer &derez, 
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndividualTask::handle_individual_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_individual_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::handle_individual_return(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_slice_request(Deserializer &derez, 
                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask::handle_slice_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_slice_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      SliceTask::handle_slice_return(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_send(Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_send(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_result(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_subscription(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Future::Impl::handle_future_subscription(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_make_persistent(Deserializer &derez,
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceManager::handle_make_persistent(derez, forest, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_mapper_message(Deserializer &derez)
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
    void Runtime::handle_mapper_broadcast(Deserializer &derez)
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
    void Runtime::handle_index_space_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_region_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_partition_semantic_info(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_semantic_info(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_free_remote_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID remote_owner_uid;
      derez.deserialize(remote_owner_uid);
      // First find it and remove it from the table
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
    void Runtime::handle_validate_remote_state(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      forest->validate_remote_state(derez, source); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_invalidate_remote_state(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      forest->invalidate_remote_state(derez, source);
    }

#ifdef SPECIALIZED_UTIL_PROCS
    //--------------------------------------------------------------------------
    Processor Runtime::get_cleanup_proc(Processor p) const
    //--------------------------------------------------------------------------
    {
      if (cleanup_proc.exists())
        return cleanup_proc;
      return p;
    }

    //--------------------------------------------------------------------------
    Processor Runtime::get_gc_proc(Processor p) const
    //--------------------------------------------------------------------------
    {
      if (gc_proc.exists())
        return gc_proc;
      return p;
    }

    //--------------------------------------------------------------------------
    Processor Runtime::get_message_proc(Processor p) const
    //--------------------------------------------------------------------------
    {
      if (message_proc.exists())
        return message_proc;
      return p;
    }
#endif

#ifdef HANG_TRACE
    //--------------------------------------------------------------------------
    void Runtime::dump_processor_states(FILE *target)
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
    void Runtime::process_schedule_request(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(local_procs.find(proc) != local_procs.end());
#endif
#ifdef LEGION_LOGGING
      // Note we can't actually trust the 'proc' variable here since
      // it may be the utility processor making this call
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      0/*unique id*/, BEGIN_SCHEDULING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(0/*unique id*/, PROF_BEGIN_SCHEDULER);
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
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Processor::get_executing_processor(),
                                      0/*unique id*/, END_SCHEDULING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(0/*unique id*/, PROF_END_SCHEDULER);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::process_message_task(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::register_event(0, PROF_BEGIN_MESSAGE);
#endif
      const char *buffer = (const char*)args;
      AddressSpaceID sender = *((const AddressSpaceID*)buffer);
      buffer += sizeof(sender);
      arglen -= sizeof(sender);
      find_messenger(sender)->process_message(buffer,arglen);
#ifdef LEGION_PROF
      LegionProf::register_event(0, PROF_END_MESSAGE);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::activate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it =
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->activate_context(context);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::deactivate_context(SingleTask *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->deactivate_context(context);
      }
    }
 
    //--------------------------------------------------------------------------
    void Runtime::execute_task_launch(Context ctx, TaskOp *task)
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
            MapOp *op = get_available_map_op();
            op->initialize(ctx, unmapped_regions[idx]);
            mapped_events.insert(op->get_completion_event());
            add_to_dependence_queue(proc, op);
          }
          // Wait for all the re-mapping operations to complete
          Event mapped_event = Event::merge_events(mapped_events);
          if (!mapped_event.has_triggered())
          {
#ifdef LEGION_LOGGING
            LegionLogging::log_inline_wait_begin(proc,
                                                 ctx->get_unique_task_id(), 
                                                 mapped_event);
#endif
#ifdef LEGION_PROF
            LegionProf::register_event(ctx->get_unique_task_id(),
                                       PROF_BEGIN_WAIT);
#endif
            pre_wait(proc);
            mapped_event.wait();
            post_wait(proc);
#ifdef LEGION_LOGGING
            LegionLogging::log_inline_wait_end(proc,
                                               ctx->get_unique_task_id(),
                                               mapped_event);
#endif
#ifdef LEGION_PROF
            LegionProf::register_event(ctx->get_unique_task_id(),
                                       PROF_END_WAIT);
#endif
          }
#ifdef LEGION_LOGGING
          else
          {
            LegionLogging::log_inline_nowait(proc,
                                             ctx->get_unique_task_id(), 
                                             mapped_event);
          }
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::add_to_dependence_queue(Processor p, Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.kind() != Processor::UTIL_PROC);
      assert(proc_managers.find(p) != proc_managers.end());
#endif
      proc_managers[p]->add_to_dependence_queue(op);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::add_to_ready_queue(Processor p, 
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
    void Runtime::add_to_local_queue(Processor p, 
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
    void Runtime::pre_wait(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      // Don't do anything for now
    }

    //--------------------------------------------------------------------------
    void Runtime::post_wait(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      // Don't do anything for now
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_pre_map_task(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_pre_map_task(task);
    }

    //--------------------------------------------------------------------------
    void Runtime::invoke_mapper_select_variant(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_select_variant(task);
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_map_task(Processor proc, SingleTask *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_task(task);
    }

    //--------------------------------------------------------------------------
    void Runtime::invoke_mapper_failed_mapping(Processor proc,
                                               Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_failed_mapping(mappable);
    }

    //--------------------------------------------------------------------------
    void Runtime::invoke_mapper_notify_result(Processor proc,
                                              Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_notify_result(mappable);
    }

    //--------------------------------------------------------------------------
    void Runtime::invoke_mapper_slice_domain(Processor proc, 
                      MultiTask *task, std::vector<Mapper::DomainSplit> &splits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_slice_domain(task, splits);
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_map_inline(Processor proc, Inline *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_inline(op);
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_map_copy(Processor proc, Copy *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_map_copy(op);
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_speculate(Processor proc, 
                                          Mappable *mappable, bool &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_speculate(mappable, value);
    }

    //--------------------------------------------------------------------------
    void Runtime::invoke_mapper_configure_context(Processor proc, TaskOp *task)
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
    bool Runtime::invoke_mapper_rank_copy_targets(Processor proc,
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
    void Runtime::invoke_mapper_rank_copy_sources(Processor proc, 
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
    void Runtime::invoke_mapper_notify_profiling(Processor proc, TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->invoke_mapper_notify_profiling(task);
    }

    //--------------------------------------------------------------------------
    bool Runtime::invoke_mapper_map_must_epoch(Processor proc,
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
    void Runtime::invoke_mapper_handle_message(Processor target,MapperID map_id,
                           Processor source, const void *message, size_t length)
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
    void Runtime::invoke_mapper_broadcast(MapperID map_id, Processor source,
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
    void Runtime::invoke_mapper_task_result(MapperID map_id, Processor proc,
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
    Processor Runtime::locate_mapper_info(Mapper *mapper, MapperID &map_id)
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
    void Runtime::handle_mapper_send_message(Mapper *mapper, Processor target,
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
    void Runtime::handle_mapper_broadcast(Mapper *mapper, const void *message,
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
    Event Runtime::launch_mapper_task(Mapper *mapper, Processor::TaskFuncID tid,
                                      const TaskArgument &arg)
    //--------------------------------------------------------------------------
    {
      MapperID map_id;
      Processor source = locate_mapper_info(mapper, map_id);
      return launch_mapper_task(mapper, source, tid, arg, map_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::defer_mapper_call(Mapper *mapper, Event wait_on)
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
    Processor Runtime::find_processor_group(const std::set<Processor> &procs)
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
    void Runtime::allocate_context(SingleTask *task)
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
    void Runtime::free_context(SingleTask *task)
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
    DistributedID Runtime::get_available_distributed_id(void)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_id_lock);
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
    void Runtime::free_distributed_id(DistributedID did)
    //--------------------------------------------------------------------------
    {
      // Don't recycle distributed IDs if we're doing LegionSpy
#ifndef LEGION_SPY
      AutoLock d_lock(distributed_id_lock);
      available_distributed_ids.push_back(did);
#endif
#ifdef DEBUG_HIGH_LEVEL
      AutoLock dist_lock(distributed_collectable_lock,1,false/*exclusive*/);
      assert(dist_collectables.find(did) == dist_collectables.end());
      AutoLock hier_lock(hierarchical_collectable_lock,1,false/*exclusive*/);
      assert(hier_collectables.find(did) == hier_collectables.end());
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::recycle_distributed_id(DistributedID did, Event recycle_event)
    //--------------------------------------------------------------------------
    {
      if (recycle_event.exists())
      {
        DeferredRecycleArgs deferred_recycle_args;
        deferred_recycle_args.hlr_id = HLR_DEFERRED_RECYCLE_ID;
        deferred_recycle_args.did = did;
        Processor proc = Processor::get_executing_processor(); 
        proc.spawn(HLR_TASK_ID, &deferred_recycle_args,
                  sizeof(deferred_recycle_args), recycle_event);
      }
      else
        free_distributed_id(did);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_distributed_collectable(DistributedID did,
                                                   DistributedCollectable *dc)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(dist_collectables.find(did) == dist_collectables.end());
#endif
      dist_collectables[did] = dc;
    }
    
    //--------------------------------------------------------------------------
    void Runtime::unregister_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(distributed_collectable_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(dist_collectables.find(did) != dist_collectables.end());
#endif
      dist_collectables.erase(did);
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::find_distributed_collectable(
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
    void Runtime::register_hierarchical_collectable(DistributedID did,
                                                    HierarchicalCollectable *hc)
    //--------------------------------------------------------------------------
    {
      AutoLock h_lock(hierarchical_collectable_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(hier_collectables.find(did) == hier_collectables.end());
#endif
      hier_collectables[did] = hc;
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_hierarchical_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock h_lock(hierarchical_collectable_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(hier_collectables.find(did) != hier_collectables.end());
#endif
      hier_collectables.erase(did);
    }

    //--------------------------------------------------------------------------
    HierarchicalCollectable* Runtime::find_hierarchical_collectable(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock h_lock(hierarchical_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,HierarchicalCollectable*>::const_iterator finder =
        hier_collectables.find(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != hier_collectables.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_future(DistributedID did, Future::Impl *impl)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_futures.find(did) == local_futures.end());
#endif
      local_futures[did] = impl;
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_future(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_futures.find(did) != local_futures.end());
#endif
      local_futures.erase(did);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_future(DistributedID did)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_lock,1,false/*exclusive*/);
      return (local_futures.find(did) != local_futures.end());
    }

    //--------------------------------------------------------------------------
    Future::Impl* Runtime::find_future(DistributedID did)
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
    Future::Impl* Runtime::find_or_create_future(DistributedID did,
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
    void Runtime::defer_collect_user(LogicalView *view, Event term_event)
    //--------------------------------------------------------------------------
    {
      GarbageCollectionEpoch *to_trigger = NULL;
      {
        AutoLock gc(gc_epoch_lock);
        current_gc_epoch->add_collection(view, term_event);
        gc_epoch_counter++;
        if (gc_epoch_counter == Runtime::gc_epoch_size)
        {
          to_trigger = current_gc_epoch;
          current_gc_epoch = new GarbageCollectionEpoch(this);
          pending_gc_epochs.insert(current_gc_epoch);
          gc_epoch_counter = 0;
        }
      }
      if (to_trigger != NULL)
      {
#ifdef SPECIALIZED_UTIL_PROCS
        Processor util = get_gc_proc(Processor::get_executing_processor());
#else
        Processor util = find_utility_group();
#endif
        to_trigger->launch(util, 0/*priority*/);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_gc_epoch(GarbageCollectionEpoch *epoch)
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
    void Runtime::increment_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      unsigned previous = 
#endif
      __sync_fetch_and_add(&outstanding_top_level_tasks,1);
#ifdef DEBUG_HIGH_LEVEL
      assert(previous > 0);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
      unsigned previous = __sync_fetch_and_sub(&outstanding_top_level_tasks,1);
#ifdef DEBUG_HIGH_LEVEL
      assert(previous > 0);
#endif
      // If there was only one left before, we're now at zero so we're done
      if (previous == 1)
        initiate_runtime_shutdown();
    }

    //--------------------------------------------------------------------------
    void Runtime::initiate_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
      log_run.spew("Computation has terminated. "
                         "Shutting down the Legion runtime...");
      // Tell all the processor managers that there is a pending shutdown
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->notify_pending_shutdown();
      }
      // Launch our last garbage collection epoch
#ifdef SPECIALIZED_UTIL_PROCS
      Processor util = get_gc_proc(Processor::get_executing_processor());
#else
      Processor util = find_utility_group();
#endif
      current_gc_epoch->launch(util, 0/*priority*/);
      // Make sure any messages that we have sent anywhere are handled
      std::set<Event> shutdown_preconditions;
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
      {
        if (message_managers[idx] != NULL)
        {
          Event last_event = message_managers[idx]->notify_pending_shutdown();
          shutdown_preconditions.insert(last_event);
        }
      }
      Event shutdown_precondition = Event::merge_events(shutdown_preconditions);
      shutdown_precondition.wait();
      // Finally shutdown the low-level runtime
      LLRuntime::get_runtime().shutdown();
    }

    //--------------------------------------------------------------------------
    IndividualTask* Runtime::get_available_individual_task(void)
    //--------------------------------------------------------------------------
    {
      IndividualTask *result = NULL;
      {
        AutoLock i_lock(individual_task_lock);
        if (!available_individual_tasks.empty())
        {
          result = available_individual_tasks.front();
          available_individual_tasks.pop_front();
        }
      }
      // Couldn't find one so make a new one
      if (result == NULL)
        result = legion_new<IndividualTask>(this);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      assert(result != NULL);
      {
        AutoLock i_lock(individual_task_lock);
        out_individual_tasks.insert(result);
      }
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    PointTask* Runtime::get_available_point_task(void)
    //--------------------------------------------------------------------------
    {
      PointTask *result = NULL;
      {
        AutoLock p_lock(point_task_lock);
        if (!available_point_tasks.empty())
        {
          result = available_point_tasks.front();
          available_point_tasks.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<PointTask>(this);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      assert(result != NULL);
      {
        AutoLock p_lock(point_task_lock);
        out_point_tasks.insert(result);
      }
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexTask* Runtime::get_available_index_task(void)
    //--------------------------------------------------------------------------
    {
      IndexTask *result = NULL;
      {
        AutoLock i_lock(index_task_lock);
        if (!available_index_tasks.empty())
        {
          result = available_index_tasks.front();
          available_index_tasks.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<IndexTask>(this);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      assert(result != NULL);
      {
        AutoLock i_lock(index_task_lock);
        out_index_tasks.insert(result);
      }
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    SliceTask* Runtime::get_available_slice_task(void)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = NULL;
      {
        AutoLock s_lock(slice_task_lock);
        if (!available_slice_tasks.empty())
        {
          result = available_slice_tasks.front();
          available_slice_tasks.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<SliceTask>(this);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      assert(result != NULL);
      {
        AutoLock s_lock(slice_task_lock);
        out_slice_tasks.insert(result);
      }
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    RemoteTask* Runtime::get_available_remote_task(void)
    //--------------------------------------------------------------------------
    {
      RemoteTask *result = NULL;
      {
        AutoLock r_lock(remote_task_lock);
        if (!available_remote_tasks.empty())
        {
          result = available_remote_tasks.front();
          available_remote_tasks.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<RemoteTask>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    InlineTask* Runtime::get_available_inline_task(void)
    //--------------------------------------------------------------------------
    {
      InlineTask *result = NULL;
      {
        AutoLock i_lock(inline_task_lock);
        if (!available_inline_tasks.empty())
        {
          result = available_inline_tasks.front();
          available_inline_tasks.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<InlineTask>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    MapOp* Runtime::get_available_map_op(void)
    //--------------------------------------------------------------------------
    {
      MapOp *result = NULL;
      {
        AutoLock m_lock(map_op_lock);
        if (!available_map_ops.empty())
        {
          result = available_map_ops.front();
          available_map_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<MapOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    CopyOp* Runtime::get_available_copy_op(void)
    //--------------------------------------------------------------------------
    {
      CopyOp *result = NULL;
      {
        AutoLock c_lock(copy_op_lock);
        if (!available_copy_ops.empty())
        {
          result = available_copy_ops.front();
          available_copy_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<CopyOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    FenceOp* Runtime::get_available_fence_op(void)
    //--------------------------------------------------------------------------
    {
      FenceOp *result = NULL;
      {
        AutoLock f_lock(fence_op_lock);
        if (!available_fence_ops.empty())
        {
          result = available_fence_ops.front();
          available_fence_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<FenceOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    FrameOp* Runtime::get_available_frame_op(void)
    //--------------------------------------------------------------------------
    {
      FrameOp *result = NULL;
      {
        AutoLock f_lock(frame_op_lock);
        if (!available_frame_ops.empty())
        {
          result = available_frame_ops.front();
          available_frame_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<FrameOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    DeletionOp* Runtime::get_available_deletion_op(void)
    //--------------------------------------------------------------------------
    {
      DeletionOp *result = NULL;
      {
        AutoLock d_lock(deletion_op_lock);
        if (!available_deletion_ops.empty())
        {
          result = available_deletion_ops.front();
          available_deletion_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<DeletionOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    InterCloseOp* Runtime::get_available_inter_close_op(void)
    //--------------------------------------------------------------------------
    {
      InterCloseOp *result = NULL;
      {
        AutoLock i_lock(inter_close_op_lock);
        if (!available_inter_close_ops.empty())
        {
          result = available_inter_close_ops.front();
          available_inter_close_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<InterCloseOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    PostCloseOp* Runtime::get_available_post_close_op(void)
    //--------------------------------------------------------------------------
    {
      PostCloseOp *result = NULL;
      {
        AutoLock p_lock(post_close_op_lock);
        if (!available_post_close_ops.empty())
        {
          result = available_post_close_ops.front();
          available_post_close_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<PostCloseOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp* Runtime::get_available_dynamic_collective_op(void)
    //--------------------------------------------------------------------------
    {
      DynamicCollectiveOp *result = NULL;
      {
        AutoLock dc_lock(dynamic_collective_op_lock);
        if (!available_dynamic_collective_ops.empty())
        {
          result = available_dynamic_collective_ops.front();
          available_dynamic_collective_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<DynamicCollectiveOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    FuturePredOp* Runtime::get_available_future_pred_op(void)
    //--------------------------------------------------------------------------
    {
      FuturePredOp *result = NULL;
      {
        AutoLock f_lock(future_pred_op_lock);
        if (!available_future_pred_ops.empty())
        {
          result = available_future_pred_ops.front();
          available_future_pred_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<FuturePredOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    NotPredOp* Runtime::get_available_not_pred_op(void)
    //--------------------------------------------------------------------------
    {
      NotPredOp *result = NULL;
      {
        AutoLock n_lock(not_pred_op_lock);
        if (!available_not_pred_ops.empty())
        {
          result = available_not_pred_ops.front();
          available_not_pred_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<NotPredOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    AndPredOp* Runtime::get_available_and_pred_op(void)
    //--------------------------------------------------------------------------
    {
      AndPredOp *result = NULL;
      {
        AutoLock a_lock(and_pred_op_lock);
        if (!available_and_pred_ops.empty())
        {
          result = available_and_pred_ops.front();
          available_and_pred_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<AndPredOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    OrPredOp* Runtime::get_available_or_pred_op(void)
    //--------------------------------------------------------------------------
    {
      OrPredOp *result = NULL;
      {
        AutoLock o_lock(or_pred_op_lock);
        if (!available_or_pred_ops.empty())
        {
          result = available_or_pred_ops.front();
          available_or_pred_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<OrPredOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    AcquireOp* Runtime::get_available_acquire_op(void)
    //--------------------------------------------------------------------------
    {
      AcquireOp *result = NULL;
      {
        AutoLock a_lock(acquire_op_lock);
        if (!available_acquire_ops.empty())
        {
          result = available_acquire_ops.front();
          available_acquire_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<AcquireOp>(this);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      assert(result != NULL);
      {
        AutoLock a_lock(acquire_op_lock);
        out_acquire_ops.insert(result);
      }
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    ReleaseOp* Runtime::get_available_release_op(void)
    //--------------------------------------------------------------------------
    {
      ReleaseOp *result = NULL;
      {
        AutoLock r_lock(release_op_lock);
        if (!available_release_ops.empty())
        {
          result = available_release_ops.front();
          available_release_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<ReleaseOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp* Runtime::get_available_capture_op(void)
    //--------------------------------------------------------------------------
    {
      TraceCaptureOp *result = NULL;
      {
        AutoLock c_lock(capture_op_lock);
        if (!available_capture_ops.empty())
        {
          result = available_capture_ops.front();
          available_capture_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<TraceCaptureOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp* Runtime::get_available_trace_op(void)
    //--------------------------------------------------------------------------
    {
      TraceCompleteOp *result = NULL;
      {
        AutoLock t_lock(trace_op_lock);
        if (!available_trace_ops.empty())
        {
          result = available_trace_ops.front();
          available_trace_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<TraceCompleteOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    MustEpochOp* Runtime::get_available_epoch_op(void)
    //--------------------------------------------------------------------------
    {
      MustEpochOp *result = NULL;
      {
        AutoLock e_lock(epoch_op_lock);
        if (!available_epoch_ops.empty())
        {
          result = available_epoch_ops.front();
          available_epoch_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<MustEpochOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp* Runtime::get_available_pending_partition_op(void)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp *result = NULL;
      {
        AutoLock p_lock(pending_partition_op_lock);
        if (!available_pending_partition_ops.empty())
        {
          result = available_pending_partition_ops.front();
          available_pending_partition_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<PendingPartitionOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp* Runtime::get_available_dependent_partition_op(void)
    //--------------------------------------------------------------------------
    {
      DependentPartitionOp *result = NULL;
      {
        AutoLock p_lock(dependent_partition_op_lock);
        if (!available_dependent_partition_ops.empty())
        {
          result = available_dependent_partition_ops.front();
          available_dependent_partition_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<DependentPartitionOp>(this);
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    FillOp* Runtime::get_available_fill_op(void)
    //--------------------------------------------------------------------------
    {
      FillOp *result = NULL;
      {
        AutoLock f_lock(fill_op_lock);
        if (!available_fill_ops.empty())
        {
          result = available_fill_ops.front();
          available_fill_ops.pop_front();
        }
      }
      if (result == NULL)
        result = legion_new<FillOp>(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_individual_task(IndividualTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(individual_task_lock);
      available_individual_tasks.push_front(task);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_individual_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_task(PointTask *task)
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
    void Runtime::free_index_task(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(index_task_lock);
      available_index_tasks.push_front(task);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_index_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_slice_task(SliceTask *task)
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
    void Runtime::free_remote_task(RemoteTask *task)
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
    void Runtime::free_inline_task(InlineTask *task)
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
    void Runtime::free_map_op(MapOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(map_op_lock);
      available_map_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_copy_op(CopyOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(copy_op_lock);
      available_copy_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fence_op(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fence_op_lock);
      available_fence_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_frame_op(FrameOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(frame_op_lock);
      available_frame_ops.push_back(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_deletion_op(DeletionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(deletion_op_lock);
      available_deletion_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_inter_close_op(InterCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inter_close_op_lock);
      available_inter_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_post_close_op(PostCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(post_close_op_lock);
      available_post_close_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_dynamic_collective_op(DynamicCollectiveOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock dc_lock(dynamic_collective_op_lock);
      available_dynamic_collective_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_future_predicate_op(FuturePredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_pred_op_lock);
      available_future_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_not_predicate_op(NotPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(not_pred_op_lock);
      available_not_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_and_predicate_op(AndPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(and_pred_op_lock);
      available_and_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_or_predicate_op(OrPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(or_pred_op_lock);
      available_or_pred_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_acquire_op(AcquireOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(acquire_op_lock);
      available_acquire_ops.push_front(op);
#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
      out_acquire_ops.erase(op);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_release_op(ReleaseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(release_op_lock);
      available_release_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_capture_op(TraceCaptureOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(capture_op_lock);
      available_capture_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_trace_op(TraceCompleteOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(trace_op_lock);
      available_trace_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_epoch_op(MustEpochOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(epoch_op_lock);
      available_epoch_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_pending_partition_op(PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(pending_partition_op_lock);
      available_pending_partition_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_dependent_partition_op(DependentPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(dependent_partition_op_lock);
      available_dependent_partition_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fill_op(FillOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fill_op_lock);
      available_fill_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    RemoteTask* Runtime::find_or_init_remote_context(UniqueID uid)
    //--------------------------------------------------------------------------
    {
      AutoLock rem_lock(remote_lock);
      std::map<UniqueID,RemoteTask*>::const_iterator finder = 
        remote_contexts.find(uid);
      if (finder != remote_contexts.end())
        return finder->second;
      // Otherwise we need to make one
      RemoteTask *result = get_available_remote_task();
      result->initialize_remote();
      // Put it in the map
      remote_contexts[uid] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_local(Processor proc) const
    //--------------------------------------------------------------------------
    {
      return (local_procs.find(proc) != local_procs.end());
    }

    //--------------------------------------------------------------------------
    IndexSpaceID Runtime::get_unique_index_space_id(void)
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
    IndexPartitionID Runtime::get_unique_index_partition_id(void)
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
    FieldSpaceID Runtime::get_unique_field_space_id(void)
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
    IndexTreeID Runtime::get_unique_index_tree_id(void)
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
    RegionTreeID Runtime::get_unique_region_tree_id(void)
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
    UniqueID Runtime::get_unique_operation_id(void)
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
    FieldID Runtime::get_unique_field_id(void)
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
    LegionErrorType Runtime::verify_requirement(
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
        if (!forest->has_node(req.region))
          return ERROR_INVALID_REGION_HANDLE;
      }
      else
      {
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
    Future Runtime::help_create_future(Operation *op /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return Future(legion_new<Future::Impl>(this, true/*register*/,
                                     get_available_distributed_id(),
                                     address_space, address_space, op));
    }

    //--------------------------------------------------------------------------
    void Runtime::help_complete_future(const Future &f)
    //--------------------------------------------------------------------------
    {
      f.impl->complete_future();
    }

    //--------------------------------------------------------------------------
    bool Runtime::help_reset_future(const Future &f)
    //--------------------------------------------------------------------------
    {
      return f.impl->reset_future();
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::generate_random_integer(void)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(random_lock);
      unsigned result = nrand48(random_state);
      return result;
    }

#ifdef TRACE_ALLOCATION 
    //--------------------------------------------------------------------------
    void Runtime::trace_allocation(AllocationType type, size_t size, int elems)
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
    void Runtime::trace_free(AllocationType type, size_t size, int elems)
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
    void Runtime::dump_allocation_info(void)
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
    /*static*/ const char* Runtime::get_allocation_name(AllocationType type)
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
        case CURR_PHYSICAL_ALLOC:
          return "Current Physical Users";
        case PREV_PHYSICAL_ALLOC:
          return "Previous Physical Users";
        case EVENT_REFERENCE_ALLOC:
          return "Instance Event References";
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
        case RUNTIME_HIER_COLLECT_ALLOC:
          return "Hierarchical Collectables";
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
        case LOGICAL_STATE_ALLOC:
          return "Logical State";
        case PHYSICAL_STATE_ALLOC:
          return "Physical State";
        default:
          assert(false); // should never get here
      }
      return NULL;
    }
#endif

#if defined(DEBUG_HIGH_LEVEL) || defined(HANG_TRACE)
    //--------------------------------------------------------------------------
    void Runtime::print_out_individual_tasks(FILE *f, int cnt /*= -1*/)
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
    void Runtime::print_out_index_tasks(FILE *f, int cnt /*= -1*/)
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
    void Runtime::print_out_slice_tasks(FILE *f, int cnt /*= -1*/)
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
    void Runtime::print_out_point_tasks(FILE *f, int cnt /*= -1*/)
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
    void Runtime::print_out_acquire_ops(FILE *f, int cnt /*= -1*/)
    //--------------------------------------------------------------------------
    {
      std::map<UniqueID,AcquireOp*> out_ops;
      for (std::set<AcquireOp*>::const_iterator it = 
            out_acquire_ops.begin(); it !=
            out_acquire_ops.end(); it++)
      {
        out_ops[(*it)->get_unique_op_id()] = *it;
      }
      for (std::map<UniqueID,AcquireOp*>::const_iterator it = 
            out_ops.begin(); it != out_ops.end(); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Acquire Op: %lld: %p (" 
                  IDFMT ",%d) triggered %d\n",
                  it->first, it->second, completion.id, completion.gen,
                  completion.has_triggered());
        if (!it->second->wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator bit = 
                it->second->wait_barriers.begin(); bit !=
                it->second->wait_barriers.end(); bit++)
          {
            Event e = bit->phase_barrier.get_previous_phase();
            fprintf(f,"Preceding barrier (" IDFMT ",%d) has triggered %d\n",
                    e.id, e.gen, e.has_triggered());
          }
        }
      }
      fflush(f);
    }

    //--------------------------------------------------------------------------
    void Runtime::print_outstanding_tasks(FILE *f, int cnt /*= -1*/)
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

    /*static*/ Runtime *Runtime::runtime_map[(MAX_NUM_PROCS+1)];
    /*static*/ volatile RegistrationCallbackFnptr Runtime::
                                              registration_callback = NULL;
    /*static*/ Processor::TaskFuncID Runtime::legion_main_id = 0;
    /*static*/ const long long Runtime::init_time = 
                                      TimeStamp::get_current_time_in_micros();
    /*static*/ int Runtime::initial_task_window_size = 
                                      DEFAULT_MAX_TASK_WINDOW;
    /*static*/ unsigned Runtime::initial_task_window_hysteresis =
                                      DEFAULT_TASK_WINDOW_HYSTERESIS;
    /*static*/ unsigned Runtime::initial_tasks_to_schedule = 
                                      DEFAULT_MIN_TASKS_TO_SCHEDULE;
    /*static*/ unsigned Runtime::initial_directory_size = 
                                      DEFAULT_MAX_DIRECTORY_SIZE;
    /*static*/ unsigned Runtime::superscalar_width = 
                                      DEFAULT_SUPERSCALAR_WIDTH;
    /*static*/ unsigned Runtime::max_message_size = 
                                      DEFAULT_MAX_MESSAGE_SIZE;
    /*static*/ unsigned Runtime::max_filter_size = 
                                      DEFAULT_MAX_FILTER_SIZE;
    /*static*/ unsigned Runtime::gc_epoch_size = 
                                      DEFAULT_GC_EPOCH_SIZE;
    /*static*/ bool Runtime::enable_imprecise_filter = false;
    /*static*/ bool Runtime::separate_runtime_instances = false;
    /*static*/ bool Runtime::record_registration = false;
    /*sattic*/ bool Runtime::stealing_disabled = false;
    /*static*/ bool Runtime::resilient_mode = false;
    /*static*/ bool Runtime::unsafe_launch = false;
    /*static*/ bool Runtime::dynamic_independence_tests = true;
    /*static*/ unsigned Runtime::shutdown_counter = 0;
    /*static*/ int Runtime::mpi_rank = -1;
    /*static*/ unsigned Runtime::mpi_rank_table[MAX_NUM_NODES];
    /*static*/ unsigned Runtime::remaining_mpi_notifications = 0;
    /*static*/ UserEvent Runtime::mpi_rank_event = UserEvent::NO_USER_EVENT;
#ifdef INORDER_EXECUTION
    /*static*/ bool Runtime::program_order_execution = true;
#endif
#ifdef DEBUG_HIGH_LEVEL
    /*static*/ bool Runtime::logging_region_tree_state = false;
    /*static*/ bool Runtime::verbose_logging = false;
    /*static*/ bool Runtime::logical_logging_only = false;
    /*static*/ bool Runtime::physical_logging_only = false;
    /*static*/ bool Runtime::check_privileges = true;
    /*static*/ bool Runtime::verify_disjointness = false;
    /*static*/ bool Runtime::bit_mask_logging = false;
#endif
#ifdef DEBUG_PERF
    /*static*/ unsigned long long Runtime::perf_trace_tolerance = 10000; 
#endif
#ifdef LEGION_PROF
    /*static*/ int Runtime::num_profiling_nodes = -1;
#endif

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
        Runtime *rt = Runtime::runtime_map[1];
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
    /*static*/ int Runtime::start(int argc, char **argv, bool background)
    //--------------------------------------------------------------------------
    {
      // Some static asserts that need to hold true for the runtime to work
      LEGION_STATIC_ASSERT(MAX_RETURN_SIZE > 0);
      LEGION_STATIC_ASSERT((1 << FIELD_LOG2) == MAX_FIELDS);
      LEGION_STATIC_ASSERT(MAX_NUM_NODES > 0);
      LEGION_STATIC_ASSERT(MAX_NUM_PROCS > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_TASK_WINDOW > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MIN_TASKS_TO_SCHEDULE > 0);
      LEGION_STATIC_ASSERT(DEFAULT_SUPERSCALAR_WIDTH > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_MESSAGE_SIZE > 0);
      // Need to pass argc and argv to low-level runtime before we can record 
      // their values as they might be changed by GASNet or MPI or whatever.
      // Note that the logger isn't initialized until after this call returns 
      // which means any logging that occurs before this has undefined behavior.
      LLRuntime ll;

#ifndef NDEBUG
      bool ok = 
#endif
        ll.init(&argc, &argv);
      assert(ok);

      // register tasks and reduction ops with LLR
      {
	const Processor::TaskIDTable& task_table =
	  get_task_table(true/*add runtime tasks*/);
	for(Processor::TaskIDTable::const_iterator it = task_table.begin();
	    it != task_table.end();
	    it++)
	  ll.register_task(it->first, it->second);
      
	const LowLevel::ReductionOpTable& red_table = get_reduction_table();
	for(LowLevel::ReductionOpTable::const_iterator it = red_table.begin();
	    it != red_table.end();
	    it++)
	  ll.register_reduction(it->first, it->second);
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
        dynamic_independence_tests = true;
        initial_task_window_size = DEFAULT_MAX_TASK_WINDOW;
        initial_task_window_hysteresis = DEFAULT_TASK_WINDOW_HYSTERESIS;
        initial_tasks_to_schedule = DEFAULT_MIN_TASKS_TO_SCHEDULE;
        initial_directory_size = DEFAULT_MAX_DIRECTORY_SIZE;
        superscalar_width = DEFAULT_SUPERSCALAR_WIDTH;
        max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
        max_filter_size = DEFAULT_MAX_FILTER_SIZE;
        gc_epoch_size = DEFAULT_GC_EPOCH_SIZE;
#ifdef INORDER_EXECUTION
        program_order_execution = true;
#endif
#ifdef LEGION_PROF
        num_profiling_nodes = -1;
#endif
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
          BOOL_ARG("-hl:imprecise",enable_imprecise_filter);
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
          INT_ARG("-hl:directory", initial_directory_size);
          INT_ARG("-hl:width", superscalar_width);
          INT_ARG("-hl:message",max_message_size);
          INT_ARG("-hl:filter", max_filter_size);
          INT_ARG("-hl:epoch", gc_epoch_size);
          if (!strcmp(argv[i],"-hl:no_dyn"))
            dynamic_independence_tests = false;
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
#ifdef LEGION_PROF
          INT_ARG("-hl:prof", num_profiling_nodes);
#else
          if (!strcmp(argv[i],"-hl:prof"))
          {
            log_run.warning("WARNING: Legion Prof is disabled.  The "
                                  "-hl:prof flag will be ignored.  Recompile "
                                  "with the -DLEGION_PROF flag to enable "
                                  "profiling.");
          }
#endif
        }
#undef INT_ARG
#undef BOOL_ARG
#ifdef DEBUG_HIGH_LEVEL
        assert(initial_task_window_hysteresis <= 100);
#endif
      }
      // Now we can set out input args
      Runtime::get_input_args().argv = argv;
      Runtime::get_input_args().argc = argc;
#ifdef HANG_TRACE
      signal(SIGTERM, catch_hang); 
#endif
      if (Runtime::record_registration)
      {
        log_run.print("High-level runtime initialization task "
                            "has low-level ID %d", INIT_FUNC_ID);
        log_run.print("High-level runtime shutdown task has "
                            "low-level ID %d", SHUTDOWN_FUNC_ID);
        log_run.print("Runtime meta-task has low-level ID %d", 
                            HLR_TASK_ID);
        std::map<Processor::TaskFuncID,TaskVariantCollection*>& 
          variant_table = Runtime::get_collection_table(); 
        for (std::map<Processor::TaskFuncID,TaskVariantCollection*>::
              const_iterator vit = variant_table.begin(); vit !=
              variant_table.end(); vit++)
        {
          TaskVariantCollection *collection = vit->second;
          for (std::map<VariantID,TaskVariantCollection::Variant>::
                const_iterator it = collection->variants.begin(); it != 
                collection->variants.end(); it++)
          {
            log_run.print("Task variant %s (ID %ld) is mapped to "
                                "low-level task ID %d", collection->name, 
                                it->first, it->second.low_id);
          }
        }
      } 
      // Kick off the low-level machine
      ll.run(0, LLRuntime::ONE_TASK_ONLY, 0, 0, background);
      // We should only make it here if the machine thread is backgrounded
      assert(background);
      if (background)
        return 0;
      else
        return -1;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      LLRuntime::get_runtime().wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_id(
                                                  Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_MPI_interoperability(int rank)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(rank >= 0);
#endif
      mpi_rank = rank;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
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
      LowLevel::ReductionOpTable &red_table = 
                                    Runtime::get_reduction_table();
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
    /*static*/ void Runtime::set_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      registration_callback = callback;
    }

    //--------------------------------------------------------------------------
    /*static*/ InputArgs& Runtime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      static InputArgs inputs = { NULL, 0 };
      return inputs;
    }

    //--------------------------------------------------------------------------
    /*static*/ Runtime* Runtime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(p.local_id() < (MAX_NUM_PROCS+1));
#endif
      return runtime_map[p.local_id()];
    }

    //--------------------------------------------------------------------------
    /*static*/ LowLevel::ReductionOpTable& Runtime::
                                                      get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      static LowLevel::ReductionOpTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::register_region_projection_function(
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
                          Runtime::get_region_projection_table();
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
    /*static*/ ProjectionID Runtime::
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
                              Runtime::get_partition_projection_table();
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
    /*static*/ TaskID Runtime::update_collection_table(
                          LowLevelFnptr low_level_ptr,
                          InlineFnptr inline_ptr,
                          TaskID uid, Processor::Kind proc_kind, 
                          bool single_task, bool index_space_task,
                          VariantID &vid, size_t return_size,
                          const TaskConfigOptions &options,
                          const char *name)
    //--------------------------------------------------------------------------
    {
      std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = 
                                        Runtime::get_collection_table();
      // See if the user wants us to find a new ID
      if (uid == AUTO_GENERATE_ID)
      {
#ifdef DEBUG_HIGH_LEVEL
        bool found = false; 
#endif
        for (unsigned idx = 0; idx < uid; idx++)
        {
          if (table.find(idx) == table.end())
          {
            uid = idx;
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(found); // If not we ran out of task ID's 2^32 tasks!
#endif
      }
      // First update the low-level task table
      Processor::TaskFuncID low_id = Runtime::get_next_available_id();
      // Add it to the low level table
      Runtime::get_task_table(false)[low_id] = low_level_ptr;
      Runtime::get_inline_table()[low_id] = inline_ptr;
      // Now see if an entry already exists in the attribute 
      // table for this uid
      if (table.find(uid) == table.end())
      {
        if (options.leaf && options.inner)
        {
          log_run.error("Task variant %s (ID %d) is not permitted to "
                              "be both inner and leaf tasks simultaneously.",
                              name, uid);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INNER_LEAF_MISMATCH);
        }
        TaskVariantCollection *collec = 
          new TaskVariantCollection(uid, name, 
                options.idempotent, return_size);
#ifdef DEBUG_HIGH_LEVEL
        assert(collec != NULL);
#endif
        table[uid] = collec;
        collec->add_variant(low_id, proc_kind, 
                            single_task, index_space_task, 
#ifdef LEGION_SPY
                            false, // no inner optimizations for analysis
#else
                            options.inner, 
#endif
                            options.leaf, 
                            vid);
      }
      else
      {
        if (table[uid]->idempotent != options.idempotent)
        {
          log_run.error("Tasks of variant %s have different idempotent "
                              "options.  All tasks of the same variant must "
                              "all be either idempotent or non-idempotent.",
                              table[uid]->name);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_IDEMPOTENT_MISMATCH);
        }
        if (table[uid]->return_size != return_size)
        {
          log_run.error("Tasks of variant %s have different return "
                              "type sizes of %ld and %ld.  All variants "
                              "must have the same return type size.",
                              table[uid]->name, table[uid]->return_size,
                              return_size);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RETURN_SIZE_MISMATCH);
        }
        if ((name != NULL) && 
            (strcmp(table[uid]->name,name) != 0))
        {
          log_run.warning("WARNING: name mismatch between variants of "
                                "task %d.  Differing names: %s %s",
                                uid, table[uid]->name, name);
        }
        if ((vid != AUTO_GENERATE_ID) && table[uid]->has_variant(vid))
        {
          log_run.warning("WARNING: Task variant collection for task %s "
                                "(ID %d) already has variant %d.  It will be "
                                "overwritten.", table[uid]->name, uid, 
                                unsigned(vid)/*dumb compiler warnings*/);
        }
        // Update the variants for the attribute
        table[uid]->add_variant(low_id, proc_kind, 
                                single_task, index_space_task, 
#ifdef LEGION_SPY
                                false, // no inner optimizations for analysis
#else
                                options.inner, 
#endif
                                options.leaf, 
                                vid);
      }
      return uid;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Runtime::update_collection_table(
                          LowLevelFnptr low_level_ptr,
                          InlineFnptr inline_ptr,
                          TaskID tid, Processor::Kind proc_kind, 
                          bool single_task, bool index_space_task,
                          VariantID vid, size_t return_size,
                          const TaskConfigOptions &options,
                          const char *name,
                          const void *user_data, size_t user_data_size)
    //--------------------------------------------------------------------------
    {
      TaskID result = update_collection_table(low_level_ptr, inline_ptr,
                                              tid, proc_kind, 
                                              single_task, index_space_task,
                                              vid, return_size, options, name);
      std::pair<TaskID,VariantID> key(tid,vid);
      void *buffer = malloc(user_data_size);
      memcpy(buffer, user_data, user_data_size);
      get_user_data_table()[key] = buffer;
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ const void* Runtime::find_user_data(TaskID tid, VariantID vid)
    //--------------------------------------------------------------------------
    {
      std::pair<TaskID,VariantID> key(tid,vid);
      const std::map<std::pair<TaskID,VariantID>,const void*> 
        &user_data_table = get_user_data_table();
      std::map<std::pair<TaskID,VariantID>,const void*>::const_iterator
        finder = user_data_table.find(key);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != user_data_table.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskVariantCollection* Runtime::get_variant_collection(
                                                      Processor::TaskFuncID tid)
    //--------------------------------------------------------------------------
    {
      std::map<Processor::TaskFuncID,TaskVariantCollection*> &task_table = 
        Runtime::get_collection_table();
      std::map<Processor::TaskFuncID,TaskVariantCollection*>::const_iterator
        finder = task_table.find(tid);
      if (finder == task_table.end())
      {
        log_run.error("Unable to find entry for Task ID %d in "
                            "the task collection table.  Did you forget "
                            "to register a task?", tid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_TASK_ID);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionFnptr Runtime::
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
    /*static*/ RegionProjectionFnptr Runtime::find_region_projection_function(
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

    //--------------------------------------------------------------------------
    /*static*/ InlineFnptr Runtime::find_inline_function(
                                                    Processor::TaskFuncID fid)
    //--------------------------------------------------------------------------
    {
      const std::map<Processor::TaskFuncID,InlineFnptr> &table = 
                                                            get_inline_table();
      std::map<Processor::TaskFuncID,InlineFnptr>::const_iterator finder = 
                                                        table.find(fid);
      if (finder == table.end())
      {
        log_run.error("Unable to find inline function with with "
                            "inline function ID %d", fid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INLINE_ID);
      }
      return finder->second;
    }

#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    //--------------------------------------------------------------------------
    /*static*/ const char* Runtime::find_privilege_task_name(void *impl)
    //--------------------------------------------------------------------------
    {
      PhysicalRegion::Impl *region = static_cast<PhysicalRegion::Impl*>(impl);
      return region->get_task_name();
    }
#endif

#ifdef BOUNDS_CHECKS
    //--------------------------------------------------------------------------
    /*static*/ void Runtime::check_bounds(void *impl, ptr_t ptr)
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
    /*static*/ void Runtime::check_bounds(void *impl, const DomainPoint &dp)
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
    /*static*/ int* Runtime::get_startup_arrivals(void)
    //--------------------------------------------------------------------------
    {
      static int startup_arrivals = 0;
      return &startup_arrivals;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskIDTable& Runtime::get_task_table(
                                            bool add_runtime_tasks /*= true*/)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskIDTable table;
      if (add_runtime_tasks)
        Runtime::register_runtime_tasks(table); 
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<Processor::TaskFuncID,InlineFnptr>& 
                                          Runtime::get_inline_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<Processor::TaskFuncID,InlineFnptr> table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<Processor::TaskFuncID,TaskVariantCollection*>& 
                                      Runtime::get_collection_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<Processor::TaskFuncID,TaskVariantCollection*> 
                                                            collection_table;
      return collection_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<std::pair<TaskID,VariantID>,const void*>&
                                      Runtime::get_user_data_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<std::pair<TaskID,VariantID>,const void*> user_data_table;
      return user_data_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionProjectionTable& Runtime::
                                              get_region_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static RegionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionTable& Runtime::
                                          get_partition_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static PartitionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::
                          register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------
    {
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_AVAILABLE; idx++)
      {
        if (table.find(idx) != table.end())
        {
          log_run.error("Task ID %d is reserved for high-level runtime "
                              "tasks",idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RESERVED_TASK_ID);
        }
      }
      table[INIT_FUNC_ID]          = Runtime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]      = Runtime::shutdown_runtime;
      table[HLR_TASK_ID]           = Runtime::high_level_runtime_task;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskFuncID Runtime::get_next_available_id(void)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskFuncID available = TASK_ID_AVAILABLE;
      return available++;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::log_machine(Machine machine)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
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
        LegionSpy::log_memory(it->id, it->capacity());
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
#endif
    }

#ifdef SPECIALIZED_UTIL_PROCS
    //--------------------------------------------------------------------------
    /*static*/ void Runtime::get_utility_processor_mapping(
                const std::set<Processor> &util_procs, Processor &cleanup_proc,
                Processor &gc_proc, Processor &message_proc)
    //--------------------------------------------------------------------------
    {
      if (util_procs.empty())
      {
        cleanup_proc = Processor::NO_PROC;
        gc_proc = Processor::NO_PROC;
        message_proc = Processor::NO_PROC;
        return;
      }
      std::set<Processor>::const_iterator set_it = util_procs.begin();
      // If we only have one utility processor then it does everything
      // otherwise we skip the first one since it does all the work
      // for actually being the utility processor for the cores
      if (util_procs.size() == 1)
      {
        cleanup_proc = *set_it;
        gc_proc = *set_it;
        message_proc = *set_it;
        return;
      }
      else
        set_it++;
      // Put the processors in a vector
      std::vector<Processor> remaining(set_it,util_procs.end());
#ifdef DEBUG_HIGH_LEVEL
      assert(!remaining.empty());
#endif
      switch (remaining.size())
      {
        case 1:
          {
            // Have the GC processor share with the actual utility
            // processor since they touch the same data structures
            gc_proc = *(util_procs.begin());
            // Use the other utility processor for the other responsibilites
            cleanup_proc = remaining[0];
            message_proc = remaining[0];
            break;
          }
        case 2:
          {
            gc_proc = remaining[0];
            cleanup_proc = remaining[1];
            message_proc = remaining[1];
            break;
          }
        default:
          {
            // Three or more
            gc_proc = remaining[0];
            cleanup_proc = remaining[1];
            message_proc = remaining[2];
            break;
          }
      }
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::initialize_runtime(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      // Always enable the idle task for any processor 
      // that is not a utility processor
      Machine machine = Machine::get_machine();
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      Processor::Kind proc_kind = p.kind();
      // Make separate runtime instances if they are requested,
      // otherwise only make a runtime instances for each of the
      // separate nodes in the machine.  To do this we exploit a
      // little bit of knowledge about the naming scheme for low-level
      // processor objects that works on both the shared and general
      // low-level runtimes.
#ifdef DEBUG_HIGH_LEVEL
      assert(p.local_id() < (MAX_NUM_PROCS+1));
#endif 
      // Figure out if we are the first processor on this node in 
      // the list of all processors. We can skip this computation 
      // if we are doing separate runtime instances.
      bool first_local_proc = false;
      if (!separate_runtime_instances)
      {
        AddressSpaceID local_addr_space = p.address_space();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          // See if this is the same address space as us
          if (it->address_space() == local_addr_space)
          {
            if ((*it) == p)
              first_local_proc = true;
            break;
          }
        }
      }
      if (separate_runtime_instances || first_local_proc)
      {
        // Compute these three data structures necessary for
        // constructing a runtime instance
        std::set<Processor> local_procs;
        std::set<Processor> local_util_procs;
        std::set<AddressSpaceID> address_spaces;
        std::map<Processor,AddressSpaceID> proc_spaces;
        AddressSpaceID local_space_id = 0;
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
          // If we are doing separate runtime instances then each
          // processor effectively gets its own address space
          local_procs.insert(p);
          AddressSpaceID sid = 0;
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++,sid++)
          {
            Processor::Kind k = it->kind();
            if (k == Processor::UTIL_PROC)
            {
              log_run.error("Separate runtime instances are not "
                                  "supported when running with explicit "
                                  "utility processors");
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_SEPARATE_UTILITY_PROCS);
            }
            if (p == (*it))
              local_space_id = sid;
            address_spaces.insert(sid); 
            proc_spaces[*it] = sid;
          }
        }
        else
        {
          std::map<unsigned,AddressSpaceID> address_space_indexes;
          // Compute an index for each address space
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++)
          {
            const unsigned space = it->address_space(); 
            std::map<unsigned,AddressSpaceID>::const_iterator finder = 
              address_space_indexes.find(space);
            if (finder == address_space_indexes.end())
            {
              AddressSpaceID index = address_space_indexes.size();
              address_space_indexes[space] = index;
              address_spaces.insert(index);
            }
            // Record our local address space
            if ((*it) == p)
              local_space_id = address_space_indexes[space];
          }
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++)
          {
            std::map<unsigned,AddressSpaceID>::const_iterator finder = 
              address_space_indexes.find(it->address_space());
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != address_space_indexes.end());
#endif
            AddressSpaceID sid = finder->second;
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
        Processor cleanup_proc = Processor::NO_PROC;
        Processor gc_proc = Processor::NO_PROC;
        Processor message_proc = Processor::NO_PROC;
#ifdef SPECIALIZED_UTIL_PROCS
        Runtime::get_utility_processor_mapping(local_util_procs,
                                               cleanup_proc, gc_proc,
                                               message_proc);
#endif
        // Set up the runtime mask for this instance
        Runtime *local_rt = new Runtime(machine, local_space_id, 
                                        local_procs, local_util_procs,
                                        address_spaces, proc_spaces,
                                        cleanup_proc, gc_proc, message_proc);
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
      }
      // Arrive at the barrier
      __sync_fetch_and_add(Runtime::get_startup_arrivals(), 1);
      // Compute the number of processors we need to wait for
      int needed_count = 0;
      {
        std::set<Processor> utility_procs;
        const unsigned local_space = p.address_space();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          if (local_space != it->address_space())
            continue;
          needed_count++;
        }
      }
      // Yes there is a race condition here on writes, but
      // everyone is going to be writing the same value
      // so it doesn't matter.
      Runtime::shutdown_counter = needed_count;
      // Have a spinning barrier here to wait for all processors
      // to finish initializing before continuing
#ifndef VALGRIND
      while (__sync_fetch_and_add(Runtime::get_startup_arrivals(), 0) 
              != needed_count) { }
#endif
      // Call in the runtime to see if we should launch the top-level task
      if (proc_kind != Processor::UTIL_PROC)
      {
        Runtime *local_rt = Runtime::get_runtime(p);
#ifdef DEBUG_HIGH_LEVEL
        assert(local_rt != NULL);
#endif
        // If we have an MPI rank, build the maps first
        if (Runtime::mpi_rank >= 0)
          local_rt->construct_mpi_rank_tables(p, Runtime::mpi_rank);
        local_rt->launch_top_level_task(p);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::shutdown_runtime(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      if (separate_runtime_instances)
        delete get_runtime(p);
      else
      {
        unsigned result = __sync_sub_and_fetch(&Runtime::shutdown_counter, 1);
        // Only delete the runtime if we're the last one to use it
        if (result == 0)
          delete get_runtime(p);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::high_level_runtime_task(
                                  const void *args, size_t arglen, Processor p)
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
            Runtime::get_runtime(p)->process_schedule_request(sched_args->proc);
            break;
          }
        case HLR_MESSAGE_ID:
          {
            Runtime::get_runtime(p)->process_message_task(data, arglen);
            break;
          }
        case HLR_POST_END_ID:
          {
            const SingleTask::PostEndArgs *post_end_args = 
              (const SingleTask::PostEndArgs*)args;
            post_end_args->proxy_this->post_end_task();
            break;
          }
        case HLR_DEFERRED_MAPPING_ID:
          {
            const Operation::DeferredMappingArgs *deferred_mapping_args = 
              (const Operation::DeferredMappingArgs*)args;
            if (deferred_mapping_args->must_epoch == NULL)
              deferred_mapping_args->proxy_this->trigger_mapping();
            else
              deferred_mapping_args->must_epoch->notify_mapping_dependence(
                  deferred_mapping_args->must_epoch_gen);
            break;
          }
        case HLR_DEFERRED_COMPLETE_ID:
          {
            const Operation::DeferredCompleteArgs *deferred_complete_args = 
              (const Operation::DeferredCompleteArgs*)args;
            deferred_complete_args->proxy_this->deferred_complete();
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
            Runtime::get_runtime(p)->finalize_field_destroy(handle, fid);
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
            const ProcessorManager::DeferredTriggerArgs *deferred_trigger_args =
              (const ProcessorManager::DeferredTriggerArgs*)args;
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
            Runtime::get_runtime(p)->free_distributed_id(
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
            if (future_args->target->remove_gc_reference())
              legion_delete(future_args->target);
            if (future_args->result->remove_gc_reference())
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
            if (future_args->result->remove_gc_reference())
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
            Runtime::mpi_rank_table[margs->mpi_rank] = margs->source_space;
            unsigned count = 
              __sync_fetch_and_add(&Runtime::remaining_mpi_notifications, 1);
            const size_t total_ranks = 
              Machine::get_machine().get_address_space_count();
            if (count == total_ranks)
              Runtime::mpi_rank_event.trigger();
            break;
          }
        case HLR_CONTRIBUTE_COLLECTIVE_ID:
          {
            Future::Impl::handle_contribute_to_collective(args);
            break;
          }
        case HLR_CHECK_STATE_ID:
          {
            TaskOp::CheckStateArgs *cargs = (TaskOp::CheckStateArgs*)args;
            cargs->task_op->check_state(cargs->ready_event);
            break;
          }
        case HLR_MAPPER_TASK_ID:
          {
            MapperTaskArgs *margs = (MapperTaskArgs*)args;
            // Tell the mapper about the result
            Runtime *rt = Runtime::get_runtime(p);       
            size_t result_size = margs->future->get_untyped_size();
            const void *result = margs->future->get_untyped_result();
            rt->invoke_mapper_task_result(margs->map_id, margs->proc,
                                          margs->event, result, result_size);
            // Now indicate that we are done with the future
            if (margs->future->remove_gc_reference())
              delete margs->future;
            // Finally tell the runtime we have one less top level task
            rt->decrement_outstanding_top_level_tasks();
            break;
          }
        case HLR_DISJOINTNESS_TASK_ID:
          {
            RegionTreeForest::DisjointnessArgs *dargs = 
              (RegionTreeForest::DisjointnessArgs*)args;
            Runtime *runtime = Runtime::get_runtime(p);
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
        default:
          assert(false); // should never get here
      }
    }

#ifdef TRACE_ALLOCATION
    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(
                                       AllocationType a, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Runtime *rt = Runtime::get_runtime(Processor::get_executing_processor());
      if (rt != NULL)
        rt->trace_allocation(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_free(AllocationType a, 
                                                 size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Runtime *rt = Runtime::get_runtime(Processor::get_executing_processor());
      if (rt != NULL)
        rt->trace_free(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ Runtime* LegionAllocation::find_runtime(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::get_runtime(Processor::get_executing_processor());
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(Runtime *&runtime,
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
    /*static*/ void LegionAllocation::trace_free(Runtime *&runtime,
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

