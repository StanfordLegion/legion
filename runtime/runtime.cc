/* Copyright 2013 Stanford University
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

namespace LegionRuntime {
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

    /////////////////////////////////////////////////////////////
    // Argument Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::Impl::Impl(void)
      : Collectable(), next(NULL), store(new ArgumentMapStore()), frozen(false)
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
      const std::map<DomainPoint,TaskArgument,DomainPoint::STLComparator> &args)
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
          delete next;
        }
      }
      else
      {
        // We're the last one in the chain being deleted,
        // so we have to delete the store as well
        delete store;
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
          std::map<DomainPoint,TaskArgument,DomainPoint::STLComparator>::
            iterator finder = arguments.find(point);
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
        std::map<DomainPoint,TaskArgument,DomainPoint::STLComparator>::
          const_iterator finder = arguments.find(point);
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
      Impl *new_impl = new Impl(store, arguments); 
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
        free(it->get_ptr());
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
      void *buffer = malloc(arg.get_size());
      memcpy(buffer, arg.get_ptr(), arg.get_size());
      TaskArgument new_arg(buffer,arg.get_size());
      values.insert(new_arg);
      return new_arg;
    }

    /////////////////////////////////////////////////////////////
    // Future Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Future::Impl::Impl(Runtime *rt, DistributedID did, 
                       AddressSpaceID own_space, AddressSpaceID loc_space,
                       TaskOp *t /*= NULL*/)
      : DistributedCollectable(rt, did, own_space, loc_space),
        task(t), task_gen((t == NULL) ? 0 : t->get_generation()),
        predicated((t == NULL) ? false : t->is_predicated()),
        ready_event(UserEvent::create_user_event()), result(NULL),
        result_size(0), empty(true), sampled(false)
    //--------------------------------------------------------------------------
    {
      runtime->register_future(did, this);
    }

    //--------------------------------------------------------------------------
    Future::Impl::Impl(const Future::Impl &rhs)
      : DistributedCollectable(NULL, 0, 0, 0), task(NULL), 
        task_gen(0), predicated(false)
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
        Processor exec_proc = Machine::get_executing_processor();
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                                      task->get_parent()->get_unique_task_id(),
                                      task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else {
        Processor exec_proc = Machine::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
      }
#endif
      if (empty)
      {
        log_run(LEVEL_ERROR,"Accessing empty future from task %s "
                            "(UID %lld)",
                            task->variants->name,
                            task->get_unique_task_id());
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
        Processor exec_proc = Machine::get_executing_processor();
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else {
        Processor exec_proc = Machine::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
      }
#endif
      if (empty)
      {
        log_run(LEVEL_ERROR,"Accessing empty future from task %s "
                            "(UID %lld)",
                            task->variants->name,
                            task->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ACCESSING_EMPTY_FUTURE);
      }
      mark_sampled();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Future::Impl::is_empty(bool block)
    //--------------------------------------------------------------------------
    {
      if (block && !ready_event.has_triggered())
      {
        Processor exec_proc = Machine::get_executing_processor();
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_begin(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(exec_proc);
        ready_event.wait();
        runtime->post_wait(exec_proc);
#ifdef LEGION_LOGGING
        LegionLogging::log_future_wait_end(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(task->get_parent()->get_unique_task_id(), 
                                   PROF_END_WAIT);
#endif
      }
#ifdef LEGION_LOGGING
      else if (block) {
        Processor exec_proc = Machine::get_executing_processor();
        LegionLogging::log_future_nowait(exec_proc,
                                       task->get_parent()->get_unique_task_id(),
                                       task->get_unique_task_id());
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
    void Future::Impl::Impl::unpack_future(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Should only happen on the owner
      // Clean out any previous results we've save
      DerezCheck z(derez);
      size_t future_size;
      derez.deserialize(future_size);
      // Handle the case where we get a double send of the
      // result once from another remote node and once
      // from the original owner
      if (result == NULL)
        result = malloc(future_size);
      if (!ready_event.has_triggered())
      {
        derez.deserialize(result,future_size);
        empty = false;
      }
#ifdef DEBUG_HIGH_LEVEL
      else
      {
        // In debug mode we need to keep the deserializer happy
        assert(result_size == future_size);
        // In theory this should just be overwriting the value
        // with the same value
        derez.deserialize(result,future_size);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Future::Impl::Impl::complete_future(void)
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
      // if we're the owner and we have a result that is complete
      // send it to the new remote future
      if (owner && ready_event.has_triggered())
      {
        Serializer rez;
        {
          rez.serialize(did);
          RezCheck z(rez);
          rez.serialize(result_size);
          rez.serialize(result,result_size);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(sid != local_space);
#endif
        runtime->send_future_result(sid, rez);
      }
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
            remote_spaces.begin(); it != remote_spaces.end(); it++)
      {
        if ((*it) != local_space)
          runtime->send_future_result(*it, rez); 
      }
    }

    //--------------------------------------------------------------------------
    bool Future::Impl::send_future(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      bool need_send;
      // Then do a quick check to see if we already sent it there
      // If we did, then we don't have 
      {
        AutoLock gc(gc_lock,1,false/*exclusive*/);
        if (remote_spaces.find(sid) != remote_spaces.end())
          need_send = false;
        else
          need_send = true;
      }
      // Need to send this first to avoid race
      if (need_send)
      {
        Serializer rez;
        {
          rez.serialize(did);
          rez.serialize(owner_space);
          if (ready_event.has_triggered())
          {
            rez.serialize(true);
            RezCheck z(rez);
            rez.serialize(result_size);
            rez.serialize(result,result_size);
          }
          else
            rez.serialize(false);
        }
        runtime->send_future(sid, rez);
      }
      // Return whether we need to send a remot reference with the packed future
      return send_remote_reference(sid);
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
      if (!runtime->has_future(did))
      {
        Future::Impl *future = new Future::Impl(runtime, did, own_space,
                                                runtime->address_space);
        future->update_remote_spaces(source);
        if (is_complete)
        {
          future->unpack_future(derez);
          future->complete_future();
        }
      }
      else
      {
        Future::Impl *future = runtime->find_future(did);
        future->update_remote_spaces(source);
        if (is_complete)
        {
          future->unpack_future(derez);
          future->complete_future();
        }
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
    }
      
    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, TaskOp *t, 
                          size_t fut_size, Runtime *rt)
      : Collectable(), context(ctx), task(t), task_gen(t->get_generation()),
        future_size(fut_size), predicated(t->is_predicated()), valid(true),
        runtime(rt), ready_event(t->get_completion_event()),
        lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(SingleTask *ctx, Runtime *rt)
      : Collectable(), context(ctx), task(NULL), task_gen(0),
        future_size(0), predicated(false), valid(false),
        runtime(rt), ready_event(Event::NO_EVENT), 
        lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    FutureMap::Impl::Impl(const FutureMap::Impl &rhs)
      : Collectable(), context(NULL), task(NULL), task_gen(0), 
        future_size(0), predicated(false), valid(false), runtime(NULL) 
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
      if (valid)
      {
        Event lock_event = lock.acquire(0, true/*exclusive*/);
        lock_event.wait(true/*block*/);
        // Check to see if we already have a future for the point
        std::map<DomainPoint,Future,DomainPoint::STLComparator>::const_iterator
          finder = futures.find(point);
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
          Processor exec_proc = Machine::get_executing_processor();
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
      for (std::map<DomainPoint,Future,
            DomainPoint::STLComparator>::const_iterator it = 
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
      for (std::map<DomainPoint,Future,
            DomainPoint::STLComparator>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        bool restart = runtime->help_reset_future(it->second);
        if (restart)
          result = true;
      }
      return result;
    }

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
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::Impl::Impl(const PhysicalRegion::Impl &rhs)
      : Collectable(), runtime(NULL), context(NULL), map_id(0), tag(0),
        ready_event(Event::NO_EVENT), mapped(false), valid(false),
        leaf_region(false), trigger_on_unmap(false)
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
        if (reference.has_required_lock())
        {
          Reservation req_lock = reference.get_required_lock();
          Event locked_event = 
            req_lock.acquire(0, true/*exclusive*/, ref_ready);
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
      return reference.get_accessor();
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
        log_inst(LEVEL_ERROR,"Requested field accessor for field %d "
            "without privleges!", fid);
        assert(false);
        exit(ERROR_INVALID_FIELD_PRIVILEGES);
      }
#endif
      return reference.get_field_accessor(fid);
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
      if (reference.has_required_lock())
      {
        Reservation req_lock = reference.get_required_lock();
        req_lock.release();
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

    /////////////////////////////////////////////////////////////
    // Physical Region Impl 
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
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind,
                                       Runtime *rt, unsigned min_out,
                                       unsigned width, unsigned def_mappers,
                                       bool no_steal, unsigned max_steals)
      : runtime(rt), local_proc(proc), proc_kind(kind), 
        utility_proc(proc.get_utility_processor()),
        explicit_utility_proc(local_proc != utility_proc),
        superscalar_width(width), min_outstanding(min_out), 
        stealing_disabled(no_steal), max_outstanding_steals(max_steals),
        current_pending(0), current_executing(false), idle_task_enabled(true),
        ready_queues(std::vector<std::list<TaskOp*> >(def_mappers)),
        mapper_objects(std::vector<Mapper*>(def_mappers,NULL)),
        mapper_locks(
            std::vector<Reservation>(def_mappers,Reservation::NO_RESERVATION))
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < def_mappers; idx++)
      {
        ready_queues[idx].clear();
        outstanding_steal_requests[idx] = std::set<Processor>();
      }
      this->idle_lock = Reservation::create_reservation();
      this->dependence_lock = Reservation::create_reservation();
      this->queue_lock = Reservation::create_reservation();
      this->stealing_lock = Reservation::create_reservation();
      this->thieving_lock = Reservation::create_reservation();
      this->gc_epoch_event = Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(const ProcessorManager &rhs)
      : runtime(NULL), local_proc(Processor::NO_PROC),
        proc_kind(Processor::LOC_PROC), utility_proc(Processor::NO_PROC),
        explicit_utility_proc(false), superscalar_width(0), min_outstanding(0), 
        stealing_disabled(false), max_outstanding_steals(0), 
        current_pending(0), current_executing(false), idle_task_enabled(true)
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
      dependence_queues.clear();
      ready_queues.clear();
      local_ready_queue.clear();
      idle_lock.destroy_reservation();
      idle_lock = Reservation::NO_RESERVATION;
      dependence_lock.destroy_reservation();
      dependence_lock = Reservation::NO_RESERVATION;
      queue_lock.destroy_reservation();
      queue_lock = Reservation::NO_RESERVATION;
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
      log_run(LEVEL_SPEW,"Adding mapper %d on processor %x", 
                          mid, local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      if (check && (mid == 0))
      {
        log_run(LEVEL_ERROR,"Invalid mapping ID.  ID 0 is reserved.");
        assert(false);
        exit(ERROR_RESERVED_MAPPING_ID);
      } 
#endif
      if (mid >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(mid+1);
        mapper_locks.resize(mid+1);
        ready_queues.resize(mid+1);
        for (unsigned int i=old_size; i<(mid+1); i++)
        {
          mapper_objects[i] = NULL;
          mapper_locks[i].destroy_reservation();
          mapper_locks[i] = Reservation::NO_RESERVATION;
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
    void ProcessorManager::invoke_mapper_set_task_options(TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      mapper_objects[task->map_id]->select_task_options(task);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_pre_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      return mapper_objects[task->map_id]->pre_map_task(task);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_select_variant(TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      mapper_objects[task->map_id]->select_task_variant(task);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      // First select the variant
      mapper_objects[task->map_id]->select_task_variant(task);
      // Then perform the mapping
      return mapper_objects[task->map_id]->map_task(task);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_failed_mapping(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mappable->map_id < mapper_objects.size());
      assert(mapper_objects[mappable->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[mappable->map_id]);     
      mapper_objects[mappable->map_id]->notify_mapping_failed(mappable);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_notify_result(Mappable *mappable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mappable->map_id < mapper_objects.size());
      assert(mapper_objects[mappable->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[mappable->map_id]);
      mapper_objects[mappable->map_id]->notify_mapping_result(mappable);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_slice_domain(TaskOp *task,
                                      std::vector<Mapper::DomainSplit> &splits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      mapper_objects[task->map_id]->slice_domain(task, 
                                                 task->index_domain, splits);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_inline(Inline *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op->map_id < mapper_objects.size());
      assert(mapper_objects[op->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[op->map_id]);
      return mapper_objects[op->map_id]->map_inline(op);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_map_copy(Copy *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op->map_id < mapper_objects.size());
      assert(mapper_objects[op->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[op->map_id]);
      return mapper_objects[op->map_id]->map_copy(op);
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::invoke_mapper_speculate(TaskOp *task, bool &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(task->map_id < mapper_objects.size());
      assert(mapper_objects[task->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[task->map_id]);
      return mapper_objects[task->map_id]->speculate_on_predicate(task, value);
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
#ifdef DEBUG_HIGH_LEVEL
      assert(mappable->map_id < mapper_objects.size());
      assert(mapper_objects[mappable->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[mappable->map_id]);
      return mapper_objects[mappable->map_id]->rank_copy_targets(mappable, 
          handle, memories, complete, max_blocking_factor, to_reuse, 
          to_create, create_one, blocking_factor);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::invoke_mapper_rank_copy_sources(Mappable *mappable,
                                           const std::set<Memory> &memories,
                                           Memory destination,
                                           std::vector<Memory> &order)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mappable->map_id < mapper_objects.size());
      assert(mapper_objects[mappable->map_id] != NULL);
#endif
      AutoLock m_lock(mapper_locks[mappable->map_id]);
      mapper_objects[mappable->map_id]->rank_copy_sources(mappable,
          memories, destination, order);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------
    {
      // Do things in the order that they can impact
      // each other so that changes can propagate as
      // quickly as possible.  For operation keep track
      // of whether there is still work to be done.
      bool disable = true;
#ifdef DYNAMIC_TESTS
      // If we're doing dynamic tests, do them first
      if (Runtime::dynamic_independence_tests)
      {
        bool more_tests = runtime->perform_dynamic_independence_tests();
        disable = disable && !more_tests;
      }
#endif
      // Then do dependence analysis
      bool more_dependences = perform_dependence_checks();
      disable = disable && !more_dependences;
      // Do any other operations next
      bool more_ops = perform_other_operations();
      disable = disable && !more_ops;
      // Finally do any mapping operations
      perform_mapping_operations();

      // Lastly we need to check to see if we should
      // disable the idle task.  Hold the idle lock
      // and check to see if we have enough outstanding
      // work on the processor.
      if (disable)
      {
        // Clearly if we're here the idle task is enabled
        AutoLock i_lock(idle_lock);
        // The condition for shutting down the idle task
        // is as follows:
        // We have no dependence analyses to perform
        //   AND
        // We have nothing in our local ready queue
        //   AND
        // (( We have enough pending tasks 
        //      AND
        //    We have a currently executing task)
        //      OR
        //   We have nothing in mapper ready queues )

        // Check to see if the dependence queue is empty
        AutoLock d_lock(dependence_lock);
        bool all_empty = true;
        for (unsigned idx = 0; idx < dependence_queues.size(); idx++)
        {
          if (!dependence_queues[idx].empty())
          {
            all_empty = false;
            break;
          }
        }
        if (all_empty)
        {
          if (local_ready_queue.empty())
          {
            // Now check to see either we have enough pending
            // or if we have nothing in our ready queues
            if (current_executing && (current_pending >= min_outstanding))
            {
              idle_task_enabled = false;
              Processor copy = local_proc;
              copy.disable_idle_task();
            }
            else
            {
              // Check to see if the ready queues are empty 
              for (unsigned idx = 0; all_empty &&
                    (idx < ready_queues.size()); idx++)
              {
                if (!ready_queues[idx].empty())
                  all_empty = false;
              }
              if (all_empty)
              {
                idle_task_enabled = false;
                Processor copy = local_proc;
                copy.disable_idle_task();
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_steal_request(Processor thief,
                                           const std::vector<MapperID> &thieves)
    //--------------------------------------------------------------------------
    {
      log_run(LEVEL_SPEW,"handling a steal request on processor %x "
                         "from processor %x", local_proc.id,thief.id);
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
          AutoLock map_lock(mapper_locks[stealer]);
          mapper_objects[stealer]->permit_task_steal(
                                        thief, stealable, to_steal_prime);
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
              temp_stolen.push_back(target);
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
          }
          else
          {
            // Always set this before putting anything on
            // the ready queue
            temp_stolen[idx]->schedule = false;
            AutoLock q_lock(queue_lock);
            ready_queues[stealer].push_front(temp_stolen[idx]);
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
          log_task(LEVEL_DEBUG,"task %s (ID %lld) stolen from processor %x "
                               "by processor %x", (*it)->variants->name,
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
      AutoLock i_lock(idle_lock);
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(idle_lock);
      current_pending++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(idle_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(current_pending > 0);
#endif
      current_pending--;
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::start_execution(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(idle_lock);
      current_executing = true;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::pause_execution(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(idle_lock);
      current_executing = false;
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_dependence_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op != NULL);
#endif
      {
        const unsigned depth = op->get_operation_depth();
        AutoLock d_lock(dependence_lock);
        // Check to see if we need to add new levels to the dependence queue
        if (depth >= dependence_queues.size())
        {
          // Add levels
          for (unsigned idx = dependence_queues.size();
                idx <= depth; idx++)
          {
            dependence_queues.push_back(std::deque<Operation*>());
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(depth < dependence_queues.size());
#endif
        dependence_queues[depth].push_back(op);
        // See if we need to start a new garbage collection epoch
        if (!gc_epoch_event.exists())
        {
          gc_epoch_trigger = UserEvent::create_user_event();
          gc_epoch_event = gc_epoch_trigger;
        }
      }
      {
        AutoLock i_lock(idle_lock);
        if (!idle_task_enabled)
        {
          idle_task_enabled = true;
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_ready_queue(TaskOp *op, bool prev_failure)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op != NULL);
      assert(op->map_id <= ready_queues.size());
#endif
      // always set this before putting something on the ready queue
      op->schedule = false; 
      {
        AutoLock q_lock(queue_lock);
        if (prev_failure)
          ready_queues[op->map_id].push_front(op);
        else
          ready_queues[op->map_id].push_back(op);
      }
      {
        AutoLock i_lock(idle_lock);
        if (!idle_task_enabled)
        {
          idle_task_enabled = true;
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_local_ready_queue(Operation *op, 
                                                    bool prev_failure)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(op != NULL);
#endif
      {
        AutoLock q_lock(queue_lock);
        if (prev_failure)
          local_ready_queue.push_front(op);
        else
          local_ready_queue.push_back(op);
      }
      {
        AutoLock i_lock(idle_lock);
        if (!idle_task_enabled)
        {
          idle_task_enabled = true;
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
      }
    }

    //--------------------------------------------------------------------------
    Event ProcessorManager::find_gc_epoch_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(dependence_lock);
      return gc_epoch_event;
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::perform_dependence_checks(void)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> ops;
      bool remaining_ops = false;
      UserEvent trigger_event;
      bool needs_trigger = false;
      {
        AutoLock d_lock(dependence_lock);
        // An important optimization here is that we always pull
        // elements off the deeper queues first which optimizes
        // for a depth-first traversal of the task/operation tree.
        unsigned handled_ops = 0;
        for (int idx = int(dependence_queues.size())-1;
              idx >= 0; idx--)
        {
          std::deque<Operation*> &current_queue = dependence_queues[idx];
          while ((handled_ops < superscalar_width) &&
                  !current_queue.empty())
          {
            ops.push_back(current_queue.front());
            current_queue.pop_front();
            handled_ops++;
          }
          remaining_ops = remaining_ops || !current_queue.empty();
          // If we know we have remaining ops and we've
          // got all the ops we need, then we can break
          if ((handled_ops == superscalar_width) && remaining_ops)
            break;
        }
        // If we no longer have any remaining dependence analyses
        // to do, then see if we have a garbage collection
        // epoch that we can trigger.
        if (!remaining_ops && gc_epoch_event.exists())
        {
          trigger_event = gc_epoch_trigger;
          needs_trigger = true;
          gc_epoch_event = Event::NO_EVENT;
        }
      }
      // Don't trigger the event while holding the lock
      // You never know what the low-level runtime might decide to do
      if (needs_trigger)
        trigger_event.trigger();

      // Ask each of the operations to issue their mapping task
      // onto the utility processor
      for (unsigned idx = 0; idx < ops.size(); idx++)
      {
        ops[idx]->trigger_dependence_analysis();
      }
      return remaining_ops;
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::perform_other_operations(void)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> ops;
      bool remaining_ops;
      {
        AutoLock q_lock(queue_lock,0,true/*exclusive*/);
        for (unsigned idx = 0; (idx < superscalar_width) && 
              !local_ready_queue.empty(); idx++)
        {
          ops.push_back(local_ready_queue.front());
          local_ready_queue.pop_front();
        }
        remaining_ops = !local_ready_queue.empty();
      }

      // Ask the operations to issue their mapping tasks onto
      // the utility processor
      for (unsigned idx = 0; idx < ops.size(); idx++)
      {
        TriggerOpArgs args;
        args.op = ops[idx];
        args.manager = this;
        utility_proc.spawn(TRIGGER_OP_ID, &args, sizeof(args));
#if 0
        bool mapped = ops[idx]->trigger_execution();
        if (!mapped)
        {
          // If we failed to perform the operation
          // then put it back on the queue
          AutoLock q_lock(queue_lock);
          local_ready_queue.push_front(ops[idx]);
        }
#endif
      }
      return remaining_ops;
    }

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
        // Pull out the current tasks for this mapping operation
        {
          AutoLock q_lock(queue_lock,1,false/*exclusive*/);
          visible_tasks.insert(visible_tasks.begin(),
               ready_queues[map_id].begin(), ready_queues[map_id].end());
        }
        // Watch me stomp all over the C++ type system here
        const std::list<Task*> &ready_tasks = 
                                *((std::list<Task*>*)(&(visible_tasks)));
        // Acquire the mapper lock and ask the mapper about scheduling
        // and then about stealing if not disabled
        {
          AutoLock map_lock(mapper_locks[map_id]);
          if (!visible_tasks.empty())
          {
            mapper_objects[map_id]->select_tasks_to_schedule(ready_tasks);
          }
          if (!stealing_disabled)
          {
            AutoLock steal_lock(stealing_lock);
            std::set<Processor> &blacklist = outstanding_steal_requests[map_id];
            if (blacklist.size() < max_outstanding_steals)
            {
              std::set<Processor> steal_targets;
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
        // Process the results first remove the operations that were
        // selected to be mapped from the queue.  Note its possible
        // that we can't actually find the task because it has been
        // stolen from the queue while we were deciding what to
        // map.  It's also possible the task is no longer in the same
        // place if the queue was prepended to.
        {
          std::list<TaskOp*> &rqueue = ready_queues[map_id];
          AutoLock q_lock(queue_lock);
          for (std::list<TaskOp*>::iterator vis_it = visible_tasks.begin(); 
                vis_it != visible_tasks.end(); /*nothing*/)
          {
            if ((*vis_it)->schedule || 
                ((*vis_it)->target_proc != local_proc))
            {
              bool found = false;
              for (std::list<TaskOp*>::iterator it = rqueue.begin();
                    it != rqueue.end(); it++)
              {
                if ((*it) == (*vis_it))
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
                vis_it++;
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
        for (std::list<TaskOp*>::iterator vis_it = visible_tasks.begin();
              vis_it != visible_tasks.end(); vis_it++)
        {
          // If we're trying to schedule or send this task somewhere
          // ask it what it wants to do
          if ((*vis_it)->schedule || ((*vis_it)->target_proc != local_proc))
          {
            bool defer = (*vis_it)->defer_mapping();
            // If the task elected to defer itself or opted to
            // not map then it is responsible for putting
            // itself back on the list of tasks to map.
            if (defer)
              continue;
            TriggerTaskArgs args;
            args.op = *vis_it;
            args.manager = this;
            utility_proc.spawn(TRIGGER_TASK_ID, &args, sizeof(args));
            continue;
#if 0
            bool executed = (*vis_it)->trigger_execution();
            if (executed)
              continue;
#endif
          }
          // Otherwise if we make it here, then we didn't map it and we
          // didn't send it so put it back on the ready queue
          (*vis_it)->schedule = false;
          AutoLock q_lock(queue_lock);
          ready_queues[map_id].push_front(*vis_it);
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
      : memory(m), capacity(rt->machine->get_memory_size(m)),
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
    void MemoryManager::recycle_physical_instance(InstanceManager *instance,
                                                  Event use_event)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock); 
#ifdef DEBUG_HIGH_LEVEL
      assert(available_instances.find(instance) == available_instances.end());
#endif
      available_instances[instance] = use_event;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::reclaim_physical_instance(InstanceManager *instance)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      std::map<InstanceManager*,Event>::iterator finder = 
        available_instances.find(instance);
      // If we didn't find it, then we can't reclaim it because someone
      // else has started using it.
      if (finder == available_instances.end())
        return false;
      // Otherwise remove it from the set of available instances
      // and indicate that we can now delete it.
      available_instances.erase(finder);
      return true;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance MemoryManager::find_physical_instance(size_t field_size,
                                                           const Domain &dom,
                                                           const unsigned depth,
                                                           Event &use_event)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      for (std::map<InstanceManager*,Event>::iterator it = 
            available_instances.begin(); it != available_instances.end(); it++)
      {
        // To avoid deadlock it is imperative that the recycled instance
        // be used by an operation which is at the same level or higher
        // in the task graph.
        if (depth > it->first->depth)
          continue;
        if (it->first->match_instance(field_size, dom))
        {
          // Set the use event, remove it from the set and return the value
          use_event = it->second;
          PhysicalInstance result = it->first->get_instance();
          // This invalidates the iterator, but we're done with it anyway
          available_instances.erase(it);
          return result;
        }
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
      AutoLock m_lock(manager_lock);
      for (std::map<InstanceManager*,Event>::iterator it = 
            available_instances.begin(); it != available_instances.end(); it++)
      {
        // To avoid deadlock it is imperative that the recycled instance
        // be used by an operation which is at the same level or higher
        // in the task graph.
        if (depth > it->first->depth)
          continue;
        if (it->first->match_instance(field_sizes, dom, blocking_factor))
        {
          // Set the use event, remove it from the set and return the value
          use_event = it->second;
          PhysicalInstance result = it->first->get_instance();
          // This invalidates the iterator, but we're done with it anyway
          available_instances.erase(it);
          return result;
        }
      }
      return PhysicalInstance::NO_INST;
    } 

    /////////////////////////////////////////////////////////////
    // Message Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(AddressSpaceID remote,
                                   Runtime *rt, size_t max_message_size,
                                   const std::set<Processor> &procs)
      : local_address_space(rt->address_space), remote_address_space(remote),
        remote_address_procs(procs), runtime(rt), 
        sending_buffer((char*)malloc(max_message_size)), 
        sending_buffer_size(max_message_size)
    //--------------------------------------------------------------------------
    {
      send_lock = Reservation::create_reservation();
      receiving_buffer_size = max_message_size;
      receiving_buffer = (char*)malloc(receiving_buffer_size);
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
                                    remote_address_procs.size();
        target = Processor::NO_PROC;
        for (std::set<Processor>::const_iterator it = 
              remote_address_procs.begin(); it !=
              remote_address_procs.end(); it++,idx++)
        {
          if (idx == target_idx)
          {
            target = it->get_utility_processor();
            break;
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(target.exists());
#endif
      }
      // Set up the buffer for sending the first batch of messages
      // Only need to write the processor once
      *((AddressSpaceID*)sending_buffer) = local_address_space;
      sending_index = sizeof(local_address_space);
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
      : local_address_space(0), remote_address_space(0),
        remote_address_procs(rhs.remote_address_procs), runtime(NULL),
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
    void MessageManager::send_index_partition_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INDEX_PARTITION_NODE, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_field_space_node(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_FIELD_SPACE_NODE, flush);
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
    void MessageManager::send_user(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_USER, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_instance_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_INSTANCE_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_back_instance_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_BACK_INSTANCE_VIEW, flush);
    }

    //--------------------------------------------------------------------------
    void MessageManager::send_reduction_view(Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      package_message(rez, SEND_REDUCTION_VIEW, flush);
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
      *((MessageHeader*)(sending_buffer+sizeof(local_address_space))) = header;
      *((unsigned*)(sending_buffer + sizeof(local_address_space) + 
            sizeof(header))) = packaged_messages;
      // Send the message
      Event next_event = target.spawn(MESSAGE_TASK_ID,sending_buffer,
                                      sending_index,last_message_event);
      // Update the event
      last_message_event = next_event;
      // Reset the state of the buffer
      sending_index = sizeof(local_address_space) + sizeof(header) + 
                      sizeof(unsigned);
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
          case SEND_INDEX_PARTITION_NODE:
            {
              runtime->handle_index_partition_node(derez, remote_address_space);
              break;
            }
          case SEND_FIELD_SPACE_NODE:
            {
              runtime->handle_field_space_node(derez, remote_address_space);
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
              runtime->handle_slice_remote_mapped(derez);
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
          case SEND_USER:
            {
              runtime->handle_send_user(derez, remote_address_space);
              break;
            }
          case SEND_INSTANCE_VIEW:
            {
              runtime->handle_send_instance_view(derez, remote_address_space);
              break;
            }
          case SEND_BACK_INSTANCE_VIEW:
            {
              runtime->handle_send_back_instance_view(derez, 
                                                      remote_address_space);
              break;
            }
          case SEND_REDUCTION_VIEW:
            {
              runtime->handle_send_reduction_view(derez, remote_address_space);
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
        while (receiving_buffer_size < (receiving_index+arglen))
          receiving_buffer_size *= 2;
        // Now realloc the memory
        void *new_ptr = realloc(receiving_buffer,receiving_buffer_size);
#ifdef DEBUG_HIGH_LEVEL
        assert(new_ptr != NULL);
#endif
        receiving_buffer = (char*)new_ptr;
      }
      // Copy the data in
      memcpy(receiving_buffer+receiving_index,args,arglen);
      receiving_index += arglen;
    }
    
    /////////////////////////////////////////////////////////////
    // Legion Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(Machine *m, AddressSpaceID unique,
                     const std::set<Processor> &locals,
                     const std::set<AddressSpaceID> &address_spaces,
                     const std::map<Processor,AddressSpaceID> &processor_spaces,
                     Processor cleanup, Processor gc, Processor message)
      : high_level(new HighLevelRuntime(this)), machine(m), 
        address_space(unique), runtime_stride(address_spaces.size()),
        forest(new RegionTreeForest(this)),
#ifdef SPECIALIZED_UTIL_PROCS
        cleanup_proc(cleanup), gc_proc(gc), message_proc(message),
#endif
        local_procs(locals), proc_spaces(processor_spaces),
        memory_manager_lock(Reservation::create_reservation()),
        unique_partition_id((unique == 0) ? runtime_stride : unique), 
        unique_field_space_id((unique == 0) ? runtime_stride : unique),
        unique_tree_id((unique == 0) ? runtime_stride : unique),
        unique_operation_id((unique == 0) ? runtime_stride : unique),
        unique_field_id((unique == 0) ? runtime_stride : unique),
        available_lock(Reservation::create_reservation()), total_contexts(0),
        distributed_id_lock(Reservation::create_reservation()),
        distributed_collectable_lock(Reservation::create_reservation()),
        hierarchical_collectable_lock(Reservation::create_reservation()),
        future_lock(Reservation::create_reservation()),
        unique_distributed_id((unique == 0) ? runtime_stride : unique),
        remote_lock(Reservation::create_reservation()),
        individual_task_lock(Reservation::create_reservation()), 
        point_task_lock(Reservation::create_reservation()),
        index_task_lock(Reservation::create_reservation()), 
        slice_task_lock(Reservation::create_reservation()),
        remote_task_lock(Reservation::create_reservation()),
        inline_task_lock(Reservation::create_reservation()),
        map_op_lock(Reservation::create_reservation()), 
        copy_op_lock(Reservation::create_reservation()), 
        fence_op_lock(Reservation::create_reservation()),
        deletion_op_lock(Reservation::create_reservation()), 
        close_op_lock(Reservation::create_reservation()), 
        future_pred_op_lock(Reservation::create_reservation()), 
        not_pred_op_lock(Reservation::create_reservation()),
        and_pred_op_lock(Reservation::create_reservation()),
        or_pred_op_lock(Reservation::create_reservation()),
        acquire_op_lock(Reservation::create_reservation()),
        release_op_lock(Reservation::create_reservation()),
        capture_op_lock(Reservation::create_reservation()),
        trace_op_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      log_run(LEVEL_DEBUG,"Initializing high-level runtime in address space %x",
                            address_space);
#ifdef LEGION_LOGGING
      // Initialize a logger if we have one
      LegionLogging::initialize_legion_logging(unique, locals);
#endif
#ifdef LEGION_PROF
      {
        std::set<Processor> handled_util_procs;
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          Processor::Kind kind = machine->get_processor_kind(*it);
          LegionProf::initialize_processor(*it, false/*util*/, kind);
          Processor util = it->get_utility_processor();
          if ((util != (*it)) && 
              (handled_util_procs.find(util) == handled_util_procs.end()))
          {
            Processor::Kind util_kind = machine->get_processor_kind(util);
            LegionProf::initialize_processor(util, true/*util*/, kind);
            handled_util_procs.insert(util);
          }
        }
        // Tell the profiler about all the memories and their kinds
        const std::set<Memory> &all_mems = machine->get_all_memories();
        for (std::set<Memory>::const_iterator it = all_mems.begin();
              it != all_mems.end(); it++)
        {
          Memory::Kind kind = machine->get_memory_kind(*it);
          LegionProf::initialize_memory(*it, kind);
        }
        // Now see if we should disable profiling on this node
        if (Runtime::num_profiling_nodes == 0)
          LegionProf::enable_profiling();
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
        }
        // If it's less than zero, then they are all enabled by default
      }
#endif
 
      // For each of the processors in our local set, construct a manager
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(machine->get_processor_kind(*it) != Processor::UTIL_PROC);
#endif
        ProcessorManager *manager = new ProcessorManager(*it,
                                    machine->get_processor_kind(*it),
                                    this, min_tasks_to_schedule,
                                    superscalar_width,
                                    DEFAULT_MAPPER_SLOTS, 
                                    stealing_disabled,
                                    machine->get_all_processors().size()-1);
        proc_managers[*it] = manager;
        manager->add_mapper(0, new DefaultMapper(machine, high_level, *it),
                            false/*needs check*/);
      }
      // For each of the other address spaces, construct a 
      // message manager for handling communication
      for (std::set<AddressSpaceID>::const_iterator it = 
            address_spaces.begin(); it != address_spaces.end(); it++)
      {
        // We don't need to make a message manager for ourself
        if (address_space == (*it))
          continue;
        // Construct the set of processors in the remote address space
        std::set<Processor> remote_procs;
        for (std::map<Processor,AddressSpaceID>::const_iterator pit = 
              processor_spaces.begin(); pit != processor_spaces.end(); pit++)
        {
          if (pit->second == (*it))
            remote_procs.insert(pit->first);
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(!remote_procs.empty());
#endif
        message_managers[(*it)] = new MessageManager(*it, this,
                                                     max_message_size,
                                                     remote_procs);
      }
      // Make the default number of contexts
      // No need to hold the lock yet because nothing is running
      for (total_contexts = 0; total_contexts < DEFAULT_CONTEXTS; 
            total_contexts++)
      {
        available_contexts.push_back(RegionTreeContext(total_contexts)); 
      }

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

      // Before launching the top level task, see if the user requested
      // a callback to be performed before starting the application
      if (Runtime::registration_callback != NULL)
        (*Runtime::registration_callback)(machine, high_level, 
                                                local_procs);
    }

    //--------------------------------------------------------------------------
    Runtime::Runtime(const Runtime &rhs)
      : high_level(NULL), machine(NULL), address_space(0), 
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
        for (std::set<Processor>::const_iterator it = local_procs.begin();
                it != local_procs.end(); it++)
        {
          all_procs.insert(*it);
          Processor util = it->get_utility_processor();
          if ((util != (*it)) && 
              (all_procs.find(util) == all_procs.end()))
          {
            all_procs.insert(util);
          }
        }
        LegionLogging::finalize_legion_logging(all_procs);
      }
#endif
#ifdef LEGION_PROF
      {
        std::set<Processor> handled_util_procs;
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          Processor::Kind kind = machine->get_processor_kind(*it);
          LegionProf::finalize_processor(*it);
          Processor util = it->get_utility_processor();
          if ((util != (*it)) && 
              (handled_util_procs.find(util) == handled_util_procs.end()))
          {
            LegionProf::finalize_processor(util);
            handled_util_procs.insert(util);
          }
        }
      }
#endif
      delete high_level;
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        delete it->second;
      }
      proc_managers.clear();
      for (std::map<AddressSpaceID,MessageManager*>::const_iterator it = 
            message_managers.begin(); it != message_managers.end(); it++)
      {
        delete it->second;
      }
      memory_manager_lock.destroy_reservation();
      memory_manager_lock = Reservation::NO_RESERVATION;
      memory_managers.clear();
      message_managers.clear();
      available_lock.destroy_reservation();
      available_lock = Reservation::NO_RESERVATION;
      distributed_id_lock.destroy_reservation();
      distributed_id_lock = Reservation::NO_RESERVATION;
      distributed_collectable_lock.destroy_reservation();
      distributed_collectable_lock = Reservation::NO_RESERVATION;
      hierarchical_collectable_lock.destroy_reservation();
      hierarchical_collectable_lock = Reservation::NO_RESERVATION;
      future_lock.destroy_reservation();
      future_lock = Reservation::NO_RESERVATION;
      remote_lock.destroy_reservation();
      remote_lock = Reservation::NO_RESERVATION;
      for (std::deque<IndividualTask*>::const_iterator it = 
            available_individual_tasks.begin(); 
            it != available_individual_tasks.end(); it++)
      {
        delete *it;
      }
      available_individual_tasks.clear();
      individual_task_lock.destroy_reservation();
      individual_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<PointTask*>::const_iterator it = 
            available_point_tasks.begin(); it != 
            available_point_tasks.end(); it++)
      {
        delete *it;
      }
      available_point_tasks.clear();
      point_task_lock.destroy_reservation();
      point_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<IndexTask*>::const_iterator it = 
            available_index_tasks.begin(); it != 
            available_index_tasks.end(); it++)
      {
        delete *it;
      }
      available_index_tasks.clear();
      index_task_lock.destroy_reservation();
      index_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<SliceTask*>::const_iterator it = 
            available_slice_tasks.begin(); it != 
            available_slice_tasks.end(); it++)
      {
        delete *it;
      }
      available_slice_tasks.clear();
      slice_task_lock.destroy_reservation();
      slice_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<RemoteTask*>::const_iterator it = 
            available_remote_tasks.begin(); it != 
            available_remote_tasks.end(); it++)
      {
        delete *it;
      }
      available_remote_tasks.clear();
      remote_task_lock.destroy_reservation();
      remote_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<InlineTask*>::const_iterator it = 
            available_inline_tasks.begin(); it !=
            available_inline_tasks.end(); it++)
      {
        delete *it;
      }
      available_inline_tasks.clear();
      inline_task_lock.destroy_reservation();
      inline_task_lock = Reservation::NO_RESERVATION;
      for (std::deque<MapOp*>::const_iterator it = 
            available_map_ops.begin(); it != 
            available_map_ops.end(); it++)
      {
        delete *it;
      }
      available_map_ops.clear();
      map_op_lock.destroy_reservation();
      map_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<CopyOp*>::const_iterator it = 
            available_copy_ops.begin(); it != 
            available_copy_ops.end(); it++)
      {
        delete *it;
      }
      available_copy_ops.clear();
      copy_op_lock.destroy_reservation();
      copy_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FenceOp*>::const_iterator it = 
            available_fence_ops.begin(); it != 
            available_fence_ops.end(); it++)
      {
        delete *it;
      }
      available_fence_ops.clear();
      fence_op_lock.destroy_reservation();
      fence_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<DeletionOp*>::const_iterator it = 
            available_deletion_ops.begin(); it != 
            available_deletion_ops.end(); it++)
      {
        delete *it;
      }
      available_deletion_ops.clear();
      deletion_op_lock.destroy_reservation();
      deletion_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<CloseOp*>::const_iterator it = 
            available_close_ops.begin(); it !=
            available_close_ops.end(); it++)
      {
        delete *it;
      }
      available_close_ops.clear();
      close_op_lock.destroy_reservation();
      close_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<FuturePredOp*>::const_iterator it = 
            available_future_pred_ops.begin(); it !=
            available_future_pred_ops.end(); it++)
      {
        delete *it;
      }
      available_future_pred_ops.clear();
      future_pred_op_lock.destroy_reservation();
      future_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<NotPredOp*>::const_iterator it = 
            available_not_pred_ops.begin(); it !=
            available_not_pred_ops.end(); it++)
      {
        delete *it;
      }
      available_not_pred_ops.clear();
      not_pred_op_lock.destroy_reservation();
      not_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<AndPredOp*>::const_iterator it = 
            available_and_pred_ops.begin(); it !=
            available_and_pred_ops.end(); it++)
      {
        delete *it;
      }
      available_and_pred_ops.clear();
      and_pred_op_lock.destroy_reservation();
      and_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<OrPredOp*>::const_iterator it = 
            available_or_pred_ops.begin(); it !=
            available_or_pred_ops.end(); it++)
      {
        delete *it;
      }
      available_or_pred_ops.clear();
      or_pred_op_lock.destroy_reservation();
      or_pred_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<AcquireOp*>::const_iterator it = 
            available_acquire_ops.begin(); it !=
            available_acquire_ops.end(); it++)
      {
        delete *it;
      }
      available_acquire_ops.clear();
      acquire_op_lock.destroy_reservation();
      acquire_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<ReleaseOp*>::const_iterator it = 
            available_release_ops.begin(); it !=
            available_release_ops.end(); it++)
      {
        delete *it;
      }
      available_release_ops.clear();
      release_op_lock.destroy_reservation();
      release_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<TraceCaptureOp*>::const_iterator it = 
            available_capture_ops.begin(); it !=
            available_capture_ops.end(); it++)
      {
        delete *it;
      }
      available_capture_ops.clear();
      capture_op_lock.destroy_reservation();
      capture_op_lock = Reservation::NO_RESERVATION;
      for (std::deque<TraceCompleteOp*>::const_iterator it = 
            available_trace_ops.begin(); it !=
            available_trace_ops.end(); it++)
      {
        delete *it;
      }
      available_trace_ops.clear();
      trace_op_lock.destroy_reservation();
      trace_op_lock = Reservation::NO_RESERVATION;

      delete forest;

#ifdef DEBUG_HIGH_LEVEL
      if (logging_region_tree_state)
	delete tree_state_logger;
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
        RemoteTask *top_context = 
          find_or_init_remote_context(top_task->get_unique_task_id());
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
    void Runtime::perform_one_time_logging(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      // First log information about the machine 
      const std::set<Processor> &all_procs = machine->get_all_processors();
      // Log all the memories
      const std::set<Memory> &all_mems = machine->get_all_memories();
      for (std::set<Memory>::const_iterator it = all_mems.begin();
            it != all_mems.end(); it++)
      {
        Memory::Kind kind = machine->get_memory_kind(*it);
        size_t mem_size = machine->get_memory_size(*it);
        LegionLogging::log_memory(*it, kind, mem_size);
      }
      // Log processor-memory affinities
      for (std::set<Processor>::const_iterator pit = all_procs.begin();
            pit != all_procs.end(); pit++)
      {
        std::vector<ProcessorMemoryAffinity> affinities;
        machine->get_proc_mem_affinity(affinities, *pit);
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
        machine->get_mem_mem_affinity(affinities, *mit);
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
    IndexSpace Runtime::create_index_space(Context ctx,
                                                    size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      IndexSpace space = IndexSpace::create_index_space(max_num_elmts);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG,"Creating index space %x in task %s (ID %lld) with "
                            "%ld maximum elements", space.id, 
                            ctx->variants->name, ctx->get_unique_task_id(), 
                            max_num_elmts);
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index space creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_top_index_space(ctx->get_executing_processor(),
                                         space);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(space.id);
#endif
      forest->create_index_space(Domain(space));
      ctx->register_index_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(domain.exists());
#endif
      // Make a dummy index space that will be associated with the domain
      IndexSpace space = domain.get_index_space(true/*create if needed*/);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG,"Creating dummy index space %x in task %s "
                            "(ID %lld) for domain", 
                            space.id, ctx->variants->name,
                            ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index space creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_top_index_space(ctx->get_executing_processor(),
                                         space);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(space.id);
#endif
      forest->create_index_space(domain);
      ctx->register_index_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Destroying index space %x in task %s (ID %lld)", 
                      handle.id, ctx->variants->name, 
                      ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index space deletion performed in leaf "
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
                                          const Coloring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(pid > 0);
      log_index(LEVEL_DEBUG,"Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", pid, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Point<1> lower_bound(coloring.begin()->first);
      Point<1> upper_bound(coloring.rbegin()->first);
      Rect<1> color_range(lower_bound,upper_bound);
      Domain color_space = Domain::from_rect<1>(color_range);
      // Perform the coloring by iterating over all the colors in the
      // range.  For unspecified colors there is nothing wrong with
      // making empty index spaces.  We do this so we can save the
      // color space as a dense 1D domain.
      std::map<Color,Domain> new_index_spaces; 
      for (GenericPointInRectIterator<1> pir(color_range); pir; pir++)
      {
        LowLevel::ElementMask 
                    child_mask(parent.get_valid_mask().get_num_elmts());
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
        IndexSpace child_space = 
          IndexSpace::create_index_space(parent, child_mask);
        new_index_spaces[finder->first] = Domain(child_space);
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
      {
        std::set<Color> current_colors;
        for (std::map<Color,Domain>::const_iterator it1 = 
              new_index_spaces.begin(); it1 != new_index_spaces.end(); it1++)
        {
          current_colors.insert(it1->first);
          for (std::map<Color,Domain>::const_iterator it2 = 
                new_index_spaces.begin(); it2 != new_index_spaces.end(); it2++)
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
              log_run(LEVEL_ERROR, "ERROR: colors %d and %d of partition %d "
                              "are not disjoint when they were claimed to be!",
                                  it1->first, it2->first, pid);
              assert(false);
              exit(ERROR_DISJOINTNESS_TEST_FAILURE);
            }
            else if (result == LowLevel::ElementMask::OVERLAP_MAYBE)
            {
              log_run(LEVEL_WARNING, "WARNING: colors %d and %d of partition "
                          "%d may not be disjoint when they were claimed to be!"
                          "(At least according to the low-level runtime.  You "
                          "might also try telling the the low-level runtime "
                          "to stop being lazy and try harder.)", 
                          it1->first, it2->first, pid);
            }
          }
        }
      }
#endif 
      forest->create_index_partition(pid, parent, disjoint,
                                 part_color, new_index_spaces, color_space);
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, disjoint,
                                         part_color);
      for (std::map<Color,Domain>::const_iterator it = 
            new_index_spaces.begin(); it != new_index_spaces.end(); it++)
      {
        LegionLogging::log_index_subspace(ctx->get_executing_processor(),
                                          pid, it->second.get_index_space(),
                                          it->first);
      }
#endif
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
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(pid > 0);
      log_index(LEVEL_DEBUG,"Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", pid, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (disjoint && verify_disjointness)
      {
        std::set<Color> current_colors;
        for (std::map<Color,Domain>::const_iterator it1 = 
              coloring.begin(); it1 != coloring.end(); it1++)
        {
          current_colors.insert(it1->first);
          for (std::map<Color,Domain>::const_iterator it2 = 
                coloring.begin(); it2 != coloring.end(); it2++)
          {
            if (current_colors.find(it2->first) != current_colors.end())
              continue;
            assert(it1->second.get_dim() == it2->second.get_dim());
            bool overlaps = false;
            switch (it1->second.get_dim())
            {
              case 1:
                {
                  Rect<1> d1 = it1->second.get_rect<1>();
                  Rect<1> d2 = it2->second.get_rect<1>();
                  overlaps = d1.overlaps(d2);
                  break;
                }
              case 2:
                {
                  Rect<2> d1 = it1->second.get_rect<2>();
                  Rect<2> d2 = it2->second.get_rect<2>();
                  overlaps = d1.overlaps(d2);
                  break;
                }
              case 3:
                {
                  Rect<3> d1 = it1->second.get_rect<3>();
                  Rect<3> d2 = it2->second.get_rect<3>();
                  overlaps = d1.overlaps(d2);
                  break;
                }
              default:
                assert(false); // should never get here
            }
            if (overlaps)
            {
              log_run(LEVEL_ERROR, "ERROR: colors %d and %d of partition %d "
                              "are not disjoint when they are claimed to be!",
                              it1->first, it2->first, pid);
              assert(false);
              exit(ERROR_DISJOINTNESS_TEST_FAILURE);
            }
          }
        }
      }
#endif
      forest->create_index_partition(pid, parent, disjoint, 
                                     part_color, coloring, color_space);
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
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(pid > 0);
      log_index(LEVEL_DEBUG,"Creating index partition %d with parent index "
                            "space %x in task %s (ID %lld)", pid, parent.id,
                            ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index partition creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // Perform the coloring
      std::map<Color,Domain> new_index_spaces;
      Domain color_space;
      // Iterate over the parent index space and make the sub-index spaces
      // for each of the different points in the space
      Accessor::RegionAccessor<Accessor::AccessorType::Generic,int> 
        fa_coloring = field_accessor.typeify<int>();
      {
        std::map<Color,LowLevel::ElementMask> child_masks;
        IndexIterator iterator(parent);
        while (iterator.has_next())
        {
          ptr_t cur_ptr = iterator.next();
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
              child_masks[color] = LowLevel::ElementMask(
                  parent.get_valid_mask().get_num_elmts());
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
          IndexSpace child_space;
          if (finder != child_masks.end())
          {
            child_space = 
              IndexSpace::create_index_space(parent, finder->second);
          }
          else
          {
            LowLevel::ElementMask empty_mask;
            child_space = IndexSpace::create_index_space(parent, empty_mask);
          }
          new_index_spaces[c] = Domain(child_space);
        }
      }
      forest->create_index_partition(pid, parent, true/*disjoint*/, 
                                     part_color, new_index_spaces, color_space);
#ifdef LEGION_LOGGING
      part_color = forest->get_index_partition_color(pid);
      LegionLogging::log_index_partition(ctx->get_executing_processor(),
                                         parent, pid, true/*disjoint*/,
                                         part_color);
      for (std::map<Color,Domain>::const_iterator it = 
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
      log_index(LEVEL_DEBUG, "Destroying index partition %x in task %s "
                             "(ID %lld)", 
                        handle, ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index partition deletion performed in "
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
    IndexPartition Runtime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = forest->get_index_partition(parent, color);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index partition performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (result == 0)
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index partitions", 
                                color);
        assert(false);
        exit(ERROR_INVALID_INDEX_SPACE_COLOR);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpace result = forest->get_index_subspace(p, color);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index subspace performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index subspace", 
                                color);
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_COLOR); 
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_space_domain(Context ctx, 
                                                    IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      Domain result = forest->get_index_space_domain(handle);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index subspace performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid handle %d for get index space domain", 
                                handle.id);
        assert(false);
        exit(ERROR_INVALID_INDEX_DOMAIN);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(Context ctx, 
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      Domain result = forest->get_index_partition_color_space(p);
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index partition color space "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid partition handle %d for get index "
                               "partition color space", p);
        assert(false);
        exit(ERROR_INVALID_INDEX_PART_DOMAIN);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index space partition colors "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      forest->get_index_space_partition_colors(sp, colors);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal is index partition disjoint "
                             "performed in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_space_color(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index space color performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_index_space_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index partition color performed "
                             "in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    ptr_t Runtime::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal safe cast operation performed "
                             "in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (pointer.is_null())
        return pointer;
      Domain domain = get_index_space_domain(ctx, region.get_index_space()); 
      DomainPoint point(pointer.value);
      if (domain.contains(point))
        return pointer;
      else
        return ptr_t::nil();
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal safe cast operation performed "
                             "in leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      if (point.is_null())
        return point;
      Domain domain = get_index_space_domain(ctx, region.get_index_space());
      if (domain.contains(point))
        return point;
      else
        return DomainPoint::nil();
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      FieldSpace space(get_unique_field_space_id());
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Creating field space %x in task %s (ID %lld)", 
                      space.id, ctx->variants->name,ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal create field space performed in leaf "
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
      log_field(LEVEL_DEBUG, "Destroying field space %x in task %s (ID %lld)", 
                    handle.id, ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal destroy field space performed in leaf "
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
      RegionTreeID tid = get_unique_tree_id();
      LogicalRegion region(tid, index_space, field_space);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Creating logical region in task %s (ID %lld) "
                              "with index space %x and field space %x in new "
                              "tree %d",
                              ctx->variants->name,ctx->get_unique_task_id(), 
                              index_space.id, field_space.id, tid);
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task "
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
      log_region(LEVEL_DEBUG, "Deleting logical region (%x,%x) in task %s "
                              "(ID %lld)",
                              handle.index_space.id, handle.field_space.id, 
                              ctx->variants->name,ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal region destruction performed in leaf "
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
      log_region(LEVEL_DEBUG, "Deleting logical partition (%x,%x) in task %s "
                              "(ID %lld)",
                              handle.index_partition, handle.field_space.id, 
                              ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal partition destruction performed in "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical partition performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_partition(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical partition performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_partition_by_color(parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical partition performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_partition_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical subregion performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical subregion performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_subregion_by_color(parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical subregion performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_subregion_by_tree(handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_region_color(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_region_color(handle);
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(Context ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get logical partition color in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return forest->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexAllocator Runtime::create_index_allocator(Context ctx, 
                                                            IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal create index allocation requested in "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal create field allocation requested in "
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
      ArgumentMap::Impl *impl = new ArgumentMap::Impl(
                                    new ArgumentMapStore());
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
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return Future(new Future::Impl(this, 
              get_available_distributed_id(), address_space, address_space));
      IndividualTask *task = get_available_individual_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute task call performed in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, launcher, check_privileges);
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %lld "
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
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return FutureMap(new FutureMap::Impl(ctx,this));
      IndexTask *task = get_available_index_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      FutureMap result = task->initialize_task(ctx, launcher, check_privileges);
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id "
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
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return Future(new Future::Impl(this, 
              get_available_distributed_id(), address_space, address_space));
      IndexTask *task = get_available_index_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, launcher, redop, 
                                            check_privileges);
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id "
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
        return Future(new Future::Impl(this,
              get_available_distributed_id(), address_space, address_space));
      IndividualTask *task = get_available_individual_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute task call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, task_id, indexes, regions, arg,
                            predicate, id, tag, check_privileges);
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %lld "
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
        return FutureMap(new FutureMap::Impl(ctx,this));
      IndexTask *task = get_available_index_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      FutureMap result = task->initialize_task(ctx, task_id, domain, indexes,
                                regions, global_arg, arg_map, predicate,
                                must_parallelism, id, tag, check_privileges);
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id "
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
        return Future(new Future::Impl(this,
              get_available_distributed_id(), address_space, address_space));
      IndexTask *task = get_available_index_task();
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal execute index space call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      Future result = task->initialize_task(ctx, task_id, domain, indexes,
                            regions, global_arg, arg_map, reduction, 
                            initial_value, predicate, must_parallelism,
                            id, tag, check_privileges);
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id "
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
      log_run(LEVEL_DEBUG, "Registering a map operation for region (%x,%x,%x) "
                           "in task %s (ID %lld)",
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
        log_run(LEVEL_ERROR,"Attempted an inline mapping of region (%x,%x,%x) "
                            "that conflicts with mapped region (%x,%x,%x) at "
                            "index %d of parent task %s (ID %lld) that would "
                            "ultimately result in deadlock.  Instead you "
                            "receive this error message.",
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
        log_run(LEVEL_ERROR,"Attempted an inline mapping of region (%x,%x,%x) "
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
      log_run(LEVEL_DEBUG, "Registering a map operation for region (%x,%x,%x) "
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
        log_run(LEVEL_ERROR,"Attempted an inline mapping of region (%x,%x,%x) "
                            "that conflicts with mapped region (%x,%x,%x) at "
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
        log_run(LEVEL_ERROR,"Attempted an inline mapping of region (%x,%x,%x) "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal remap operation performed in "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal unmap operation performed in "
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
    void Runtime::map_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      const std::vector<PhysicalRegion> &regions = ctx->get_physical_regions();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].impl->is_mapped())
          continue;
        MapOp *map_op = get_available_map_op();
        map_op->initialize(ctx, regions[idx]);
        add_to_dependence_queue(ctx->get_executing_processor(), map_op);
      }
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          regions[idx].impl->wait_until_valid();
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      const std::vector<PhysicalRegion> &regions = ctx->get_physical_regions();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].impl->is_mapped())
          regions[idx].impl->unmap_region();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx, 
                                       const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      CopyOp *copy_op = get_available_copy_op();  
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal copy operation call performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
      copy_op->initialize(ctx, launcher, check_privileges);
      log_run(LEVEL_DEBUG, "Registering a copy operation in task %s "
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
      if (f.impl->predicated)
      {
        log_run(LEVEL_ERROR,"Illegal predicate creation performed on "
                            "predicated future from task %s (ID %lld) "
                            "inside of task %s (ID %lld).",
                            f.impl->task->variants->name,
                            f.impl->task->get_unique_task_id(),
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
        log_task(LEVEL_ERROR,"Illegal predicate creation performed in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      FuturePredOp *pred_op = get_available_future_pred_op();
      pred_op->initialize(f, proc);
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
      return Predicate(pred_op);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal NOT predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      NotPredOp *pred_op = get_available_not_pred_op();
      pred_op->initialize(p);
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
      return Predicate(pred_op);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_and(Context ctx, const Predicate &p1, 
                                                  const Predicate &p2) 
    //--------------------------------------------------------------------------
    {
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal AND predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      AndPredOp *pred_op = get_available_and_pred_op();
      pred_op->initialize(p1, p2);
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
      return Predicate(pred_op);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_or(Context ctx, const Predicate &p1, 
                                                 const Predicate &p2)  
    //--------------------------------------------------------------------------
    {
      // Find the mapper for this predicate
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal OR predicate creation in "
                             "leaf task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      OrPredOp *pred_op = get_available_or_pred_op();
      pred_op->initialize(p1, p2);
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
      return Predicate(pred_op);
    }

    //--------------------------------------------------------------------------
    Lock Runtime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal lock creation in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      return Lock(Reservation::create_reservation());
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal lock destruction in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ctx->destroy_user_lock(l.reservation_lock);
    }

    //--------------------------------------------------------------------------
    Grant Runtime::acquire_grant(Context ctx, 
                                 const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal grant acquire in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
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
      return Grant(new Grant::Impl(unpack_requests));
    }

    //--------------------------------------------------------------------------
    void Runtime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal grant release in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      grant.impl->release_grant();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::create_phase_barrier(Context ctx, 
                                                        unsigned participants)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal phase barrier creation in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Barrier result = Barrier::create_barrier(participants);
      return PhaseBarrier(result, participants);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal phase barrier destruction in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      ctx->destroy_user_barrier(pb.phase_barrier);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::advance_phase_barrier(Context ctx, 
                                                         PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal phase barrier advance call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      Barrier bar = pb.phase_barrier;
      // Mark that one of the expected arrivals has arrived
      // TODO: put this back in once Sean fixes barriers
      //bar.arrive();
      Barrier new_bar = bar.advance_barrier();
      return PhaseBarrier(new_bar, pb.participant_count());
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_acquire(Context ctx, const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AcquireOp *acquire_op = get_available_acquire_op();
#ifdef DEBUG_HIGH_LEVEL
      log_run(LEVEL_DEBUG,"Issuing an acquire operation in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal acquire operation performed in leaf task"
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
      log_run(LEVEL_DEBUG,"Issuing a release operation in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal release operation performed in leaf task"
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
      log_run(LEVEL_DEBUG,"Issuing a mapping fence in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal legion mapping fence call in leaf task "
                             "%s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      fence_op->initialize(ctx, true/*mapping fence*/);
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
      log_run(LEVEL_DEBUG,"Issuing an execution fence in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal Legion execution fence call in leaf "
                             "task %s (ID %lld)",
                             ctx->variants->name, ctx->get_unique_task_id());
        assert(false);
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      fence_op->initialize(ctx, false/*mapping fence*/);
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
      log_run(LEVEL_DEBUG,"Beginning a trace in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal Legion begin trace call in leaf "
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
    void Runtime::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
 #ifdef DEBUG_HIGH_LEVEL
      log_run(LEVEL_DEBUG,"Ending a trace in task %s (ID %lld)",
                          ctx->variants->name, ctx->get_unique_task_id());
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal Legion end trace call in leaf "
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
    Mapper* Runtime::get_mapper(Context ctx, MapperID id)
    //--------------------------------------------------------------------------
    {
      Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->find_mapper(id);
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
    void Runtime::add_mapper(MapperID map_id, Mapper *mapper, 
                                      Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->add_mapper(map_id, mapper, true/*check*/);
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
    }

    //--------------------------------------------------------------------------
    FieldID Runtime::allocate_field(Context ctx, FieldSpace space,
                                          size_t field_size, FieldID fid,
                                          bool local)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (ctx->is_leaf() && !local)
      {
        log_task(LEVEL_ERROR,"Illegal non-local field allocation performed "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal field destruction performed in "
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
      if (ctx->is_leaf() && !local)
      {
        log_task(LEVEL_ERROR,"Illegal non-local field allocation performed "
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
      if (ctx->is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal field destruction performed in "
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
    void Runtime::recycle_physical_instance(InstanceManager *inst,
                                            Event use_event)
    //--------------------------------------------------------------------------
    {
      find_memory(inst->memory)->recycle_physical_instance(inst, use_event);
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
    MessageManager* Runtime::find_messenger(AddressSpaceID sid) const
    //--------------------------------------------------------------------------
    {
      std::map<AddressSpaceID,MessageManager*>::const_iterator finder = 
        message_managers.find(sid);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != message_managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    MessageManager* Runtime::find_messenger(Processor target) const
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
    void Runtime::send_index_partition_node(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_index_partition_node(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_field_space_node(rez, false/*flush*/);
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
    void Runtime::send_user(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Important note that this flush must be true in order for the
      // garbage collector to work correctly.
      find_messenger(target)->send_user(rez, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_instance_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_instance_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_instance_view(AddressSpaceID target, 
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_back_instance_view(rez, false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_reduction_view(rez, false/*flush*/);
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
    void Runtime::handle_index_partition_node(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_creation(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_node(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_node_creation(forest, derez, source);
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
    void Runtime::handle_individual_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask::process_unpack_remote_mapped(derez);
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
    void Runtime::handle_slice_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask::process_slice_mapped(derez);
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
      PhysicalView::handle_send_back_user(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_user(Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PhysicalView::handle_send_user(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_instance_view(Deserializer &derez, 
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_send_instance_view(forest, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_instance_view(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_send_back_instance_view(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_reduction_view(forest, derez, source);
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
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      0/*unique id*/, BEGIN_SCHEDULING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(0/*unique id*/, PROF_BEGIN_SCHEDULER);
#endif
      log_run(LEVEL_DEBUG,"Running scheduler on processor %x", proc.id);
      ProcessorManager *manager = proc_managers[proc];
      manager->perform_scheduling();
#ifdef DYNAMIC_TESTS
      if (dynamic_independence_tests)
        forest->perform_dynamic_tests(superscalar_width);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
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
    void Runtime::increment_pending(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->increment_pending();
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_pending(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->decrement_pending();
    }

    //--------------------------------------------------------------------------
    void Runtime::start_execution(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->start_execution();
    }

    //--------------------------------------------------------------------------
    void Runtime::pause_execution(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->pause_execution();
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
      assert(machine->get_processor_kind(p) != Processor::UTIL_PROC);
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
      assert(machine->get_processor_kind(p) != Processor::UTIL_PROC);
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
      assert(machine->get_processor_kind(p) != Processor::UTIL_PROC);
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
      proc_managers[proc]->pause_execution();
    }

    //--------------------------------------------------------------------------
    void Runtime::post_wait(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      proc_managers[proc]->start_execution();
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
                                          TaskOp *task, bool &value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(proc) != proc_managers.end());
#endif
      return proc_managers[proc]->invoke_mapper_speculate(task, value);
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
      unsigned current_contexts = total_contexts;
      __sync_fetch_and_add(&total_contexts,current_contexts);
      if (total_contexts >= MAX_CONTEXTS)
      {
        log_run(LEVEL_ERROR,"ERROR: Maximum number of allowed contexts %d "
                            "exceeded.  Please change 'MAX_CONTEXTS' at top "
                            "of legion_types.h and recompile.",
                            MAX_CONTEXTS);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_EXCEEDED_MAX_CONTEXTS);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(!available_contexts.empty());
#endif
      // Tell the forest to resize the number of available contexts
      // on all the nodes
      forest->resize_node_contexts(total_contexts);
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
    Event Runtime::find_gc_epoch_event(Processor local_proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(proc_managers.find(local_proc) != proc_managers.end());
#endif
      return proc_managers[local_proc]->find_gc_epoch_event();
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
        result = new IndividualTask(this);
#ifdef DEBUG_HIGH_LEVEL
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
        result = new PointTask(this);
#ifdef DEBUG_HIGH_LEVEL
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
        result = new IndexTask(this);
#ifdef DEBUG_HIGH_LEVEL
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
        result = new SliceTask(this);
#ifdef DEBUG_HIGH_LEVEL
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
        result = new RemoteTask(this);
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
        result = new InlineTask(this);
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
        result = new MapOp(this);
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
        result = new CopyOp(this);
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
        result = new FenceOp(this);
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
        result = new DeletionOp(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      result->activate();
      return result;
    }

    //--------------------------------------------------------------------------
    CloseOp* Runtime::get_available_close_op(void)
    //--------------------------------------------------------------------------
    {
      CloseOp *result = NULL;
      {
        AutoLock c_lock(close_op_lock);
        if (!available_close_ops.empty())
        {
          result = available_close_ops.front();
          available_close_ops.pop_front();
        }
      }
      if (result == NULL)
        result = new CloseOp(this);
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
        result = new FuturePredOp(this);
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
        result = new NotPredOp(this);
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
        result = new AndPredOp(this);
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
        result = new OrPredOp(this);
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
        result = new AcquireOp(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
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
        result = new ReleaseOp(this);
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
        result = new TraceCaptureOp(this);
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
        result = new TraceCompleteOp(this);
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
#ifdef DEBUG_HIGH_LEVEL
      out_individual_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_task(PointTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(point_task_lock);
      available_point_tasks.push_front(task);
#ifdef DEBUG_HIGH_LEVEL
      out_point_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_index_task(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(index_task_lock);
      available_index_tasks.push_front(task);
#ifdef DEBUG_HIGH_LEVEL
      out_index_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_slice_task(SliceTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(slice_task_lock);
      available_slice_tasks.push_front(task);
#ifdef DEBUG_HIGH_LEVEL
      out_slice_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_remote_task(RemoteTask *task)
    //--------------------------------------------------------------------------
    {
      // First remove it from the list of remote tasks
      {
        AutoLock rem_lock(remote_lock);
        std::map<UniqueID,RemoteTask*>::iterator finder = 
          remote_contexts.find(task->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != remote_contexts.end());
#endif
        remote_contexts.erase(finder);
      }
      // Then we can put it back on the list of available remote tasks
      AutoLock r_lock(remote_task_lock);
      available_remote_tasks.push_front(task);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_inline_task(InlineTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inline_task_lock);
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
    void Runtime::free_deletion_op(DeletionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(deletion_op_lock);
      available_deletion_ops.push_front(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_close_op(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(close_op_lock);
      available_close_ops.push_front(op);
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
      result->initialize_remote(uid);
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
    IndexPartition Runtime::get_unique_partition_id(void)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = __sync_fetch_and_add(&unique_partition_id,
                                                   runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of partitions created
      // then we are really in a bad place.
      assert(result <= unique_partition_id); 
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
    RegionTreeID Runtime::get_unique_tree_id(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeID result = __sync_fetch_and_add(&unique_tree_id,
                                                 runtime_stride);
#ifdef DEBUG_HIGH_LEVEL
      // check for overflow
      // If we have overflow on the number of region trees
      // created then we are really in a bad place.
      assert(result <= unique_tree_id);
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
    Future Runtime::help_create_future(TaskOp *task /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return Future(new Future::Impl(this, get_available_distributed_id(),
                                     address_space, address_space, task));
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

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------
    bool Runtime::perform_dynamic_independence_tests(void)
    //--------------------------------------------------------------------------
    {
      return forest->perform_dynamic_tests(superscalar_width);
    }
#endif

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void Runtime::print_out_individual_tasks(int cnt /*= 1*/)
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
            out_tasks.begin(); (it != out_tasks.end()) && (cnt >= 0); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(stdout,"Outstanding Individual Task %lld: %p %s (%x,%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        cnt--;
      }
      fflush(stdout);
    }

    //--------------------------------------------------------------------------
    void Runtime::print_out_index_tasks(int cnt /*= 1*/)
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
            out_tasks.begin(); (it != out_tasks.end()) && (cnt >= 0); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(stdout,"Outstanding Index Task %lld: %p %s (%x,%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        cnt--;
      }
      fflush(stdout);
    }

    //--------------------------------------------------------------------------
    void Runtime::print_out_slice_tasks(int cnt /*= 1*/)
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
            out_tasks.begin(); (it != out_tasks.end()) && (cnt >= 0); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(stdout,"Outstanding Slice Task %lld: %p %s (%x,%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        cnt--;
      }
      fflush(stdout);
    }

    //--------------------------------------------------------------------------
    void Runtime::print_out_point_tasks(int cnt /*= 1*/)
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
            out_tasks.begin(); (it != out_tasks.end()) && (cnt >= 0); it++)
      {
        Event completion = it->second->get_completion_event();
        fprintf(stdout,"Outstanding Point Task %lld: %p %s (%x,%d)\n",
                it->first, it->second, it->second->variants->name,
                completion.id, completion.gen); 
        cnt--;
      }
      fflush(stdout);
    }
#endif

    /*static*/ Runtime *Runtime::runtime_map[(MAX_NUM_PROCS+1)];
    /*static*/ unsigned Runtime::startup_arrivals = 0;
    /*static*/ volatile RegistrationCallbackFnptr Runtime::
                                              registration_callback = NULL;
    /*static*/ Processor::TaskFuncID Runtime::legion_main_id = 0;
    /*static*/ const long long Runtime::init_time = 
                                      TimeStamp::get_current_time_in_micros();
    /*static*/ unsigned Runtime::max_task_window_per_context = 
                                      DEFAULT_MAX_TASK_WINDOW;
    /*static*/ unsigned Runtime::min_tasks_to_schedule = 
                                      DEFAULT_MIN_TASKS_TO_SCHEDULE;
    /*static*/ unsigned Runtime::superscalar_width = 
                                      DEFAULT_SUPERSCALAR_WIDTH;
    /*static*/ unsigned Runtime::max_message_size = 
                                      DEFAULT_MAX_MESSAGE_SIZE;
    /*static*/ unsigned Runtime::max_filter_size = 
                                      DEFAULT_MAX_FILTER_SIZE;
    /*static*/ bool Runtime::separate_runtime_instances = false;
    /*sattic*/ bool Runtime::stealing_disabled = false;
    /*static*/ bool Runtime::resilient_mode = false;
    /*static*/ unsigned Runtime::shutdown_counter = 0;
#ifdef INORDER_EXECUTION
    /*static*/ bool Runtime::program_order_execution = true;
#endif
#ifdef DYNAMIC_TESTS
    /*static*/ bool Runtime::dynamic_independence_tests = true;
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
#ifdef LEGION_PROF
    /*static*/ int Runtime::num_profiling_nodes = -1;
#endif

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::start(int argc, char **argv, bool background)
    //--------------------------------------------------------------------------
    {
      // Need to pass argc and argv to low-level runtime before we can record 
      // their values as they might be changed by GASNet or MPI or whatever.
      // Note that the logger isn't initialized until after this call returns 
      // which means any logging that occurs before this has undefined behavior.
      Machine *m = new Machine(&argc, &argv, 
                      Runtime::get_task_table(true/*add runtime tasks*/), 
		      Runtime::get_reduction_table(), false/*cps style*/);
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
        startup_arrivals = 0;
        separate_runtime_instances = false;
        stealing_disabled = false;
        resilient_mode = false;
        max_task_window_per_context = DEFAULT_MAX_TASK_WINDOW;
        min_tasks_to_schedule = DEFAULT_MIN_TASKS_TO_SCHEDULE;
        superscalar_width = DEFAULT_SUPERSCALAR_WIDTH;
        max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
        max_filter_size = DEFAULT_MAX_FILTER_SIZE;
#ifdef INORDER_EXECUTION
        program_order_execution = true;
#endif
#ifdef DYNAMIC_TESTS
        dynamic_independence_tests = true;
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
          BOOL_ARG("-hl:separate",separate_runtime_instances);
          BOOL_ARG("-hl:nosteal",stealing_disabled);
          BOOL_ARG("-hl:resilient",resilient_mode);
#ifdef INORDER_EXECUTION
          if (!strcmp(argv[i],"-hl:outorder"))
            program_order_execution = false;
#endif
          INT_ARG("-hl:window", max_task_window_per_context);
          INT_ARG("-hl:sched", min_tasks_to_schedule);
          INT_ARG("-hl:width", superscalar_width);
          INT_ARG("-hl:message",max_message_size);
          INT_ARG("-hl:filter", max_filter_size);
#ifdef DYNAMIC_TESTS
          if (!strcmp(argv[i],"-hl:no_dyn"))
            dynamic_independence_tests = false;
#endif
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
            log_run(LEVEL_WARNING,"WARNING: Region tree state logging is "
                          "disabled.  To enable region tree state logging "
                                                  "compile in debug mode.");
          }
          if (!strcmp(argv[i],"-hl:disjointness"))
          {
            log_run(LEVEL_WARNING,"WARNING: Disjointness verification for "
                      "partition creation is disabled.  To enable dynamic "
                              "disjointness testing compile in debug mode.");
          }
#endif
#ifdef LEGION_PROF
          INT_ARG("-hl:prof", num_profiling_nodes);
#else
          if (!strcmp(argv[i],"-hl:prof"))
          {
            log_run(LEVEL_WARNING,"WARNING: Legion Prof is disabled.  The "
                                  "-hl:prof flag will be ignored.  Recompile "
                                  "with the -DLEGION_PROF flag to enable "
                                  "profiling.");
          }
#endif
        }
#undef INT_ARG
#undef BOOL_ARG
#ifdef DEBUG_HIGH_LEVEL
        assert(max_task_window_per_context > 0);
#endif
      }
#ifdef LEGION_PROF
      {
        const std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = 
          Runtime::get_collection_table();
        for (std::map<Processor::TaskFuncID,TaskVariantCollection*>::
              const_iterator it = table.begin(); it != table.end(); it++)
        {
          LegionProf::register_task_variant(it->first, it->second->name);
        }
      }
#endif
      // Now we can set out input args
      Runtime::get_input_args().argv = argv;
      Runtime::get_input_args().argc = argc;
      // Kick off the low-level machine
      m->run(0, Machine::ONE_TASK_ONLY, 0, 0, background);
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
      Machine *machine = Machine::get_machine();
      machine->wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_id(
                                                  Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        log_run(LEVEL_ERROR,"ERROR: ReductionOpID zero is reserved.");
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
        log_run(LEVEL_ERROR,"Invalid ReductionOpID %d",redop_id);
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
      assert((p.id & 0xffff) < (MAX_NUM_PROCS+1));
#endif
      return runtime_map[(p.id & 0xffff)];
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
        log_run(LEVEL_ERROR,"ERROR: ProjectionID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      RegionProjectionTable &proj_table = 
                          Runtime::get_region_projection_table();
      if (proj_table.find(handle) != proj_table.end())
      {
        log_run(LEVEL_ERROR,"ERROR: ProjectionID %d has already been used in "
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
        log_run(LEVEL_ERROR,"ERROR: ProjectionID zero is reserved.\n");
#ifdef DEBUG_HIGH_LEVEl
        assert(false);
#endif
        exit(ERROR_RESERVED_PROJECTION_ID);
      }
      PartitionProjectionTable &proj_table = 
                              Runtime::get_partition_projection_table();
      if (proj_table.find(handle) != proj_table.end())
      {
        log_run(LEVEL_ERROR,"ERROR: ProjectionID %d has already been used in "
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
                          VariantID vid, size_t return_size,
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
          log_run(LEVEL_ERROR,"Task variant %s (ID %d) is not permitted to "
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
          log_run(LEVEL_ERROR,"Tasks of variant %s have different idempotent "
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
          log_run(LEVEL_ERROR,"Tasks of variant %s have different return "
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
          log_run(LEVEL_WARNING,"WARNING: name mismatch between variants of "
                                "task %d.  Differing names: %s %s",
                                uid, table[uid]->name, name);
        }
        if ((vid != AUTO_GENERATE_ID) && table[uid]->has_variant(vid))
        {
          log_run(LEVEL_WARNING,"WARNING: Task variant collection for task %s "
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
        log_run(LEVEL_ERROR,"Unable to find entry for Task ID %d in "
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
        log_run(LEVEL_ERROR,"Unable to find registered partition "
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
        log_run(LEVEL_ERROR,"Unable to find registered region projection "
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
        log_run(LEVEL_ERROR,"Unable to find inline function with with "
                            "inline function ID %d", fid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INLINE_ID);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskIDTable& Runtime::get_task_table(
                                            bool add_runtime_tasks /*= true*/)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskIDTable table;
      if (add_runtime_tasks)
      {
        Runtime::register_runtime_tasks(table);
      }
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
          log_run(LEVEL_ERROR,"Task ID %d is reserved for high-level runtime "
                              "tasks",idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RESERVED_TASK_ID);
        }
      }
      table[INIT_FUNC_ID]         = Runtime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]     = Runtime::shutdown_runtime;
      table[SCHEDULER_ID]         = Runtime::schedule_runtime;
      table[MESSAGE_TASK_ID]      = Runtime::message_task;
      table[POST_END_TASK_ID]     = Runtime::post_end_task;
      table[DEFERRED_COMPLETE_ID] = Runtime::deferred_complete_task;
      table[RECLAIM_LOCAL_FID]    = Runtime::reclaim_local_field_task;
      table[DEFERRED_COLLECT_ID]  = Runtime::deferred_collect_task;
      table[TRIGGER_OP_ID]        = Runtime::trigger_op_task;
      table[TRIGGER_TASK_ID]      = Runtime::trigger_task_task;
      table[LEGION_LOGGING_ID]    = Runtime::legion_logging_task;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskFuncID Runtime::get_next_available_id(void)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskFuncID available = TASK_ID_AVAILABLE;
      return available++;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::log_machine(Machine *machine)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      std::set<Processor> utility_procs;
      const std::set<Processor> &all_procs = machine->get_all_processors();
      // Find all the utility processors
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
        utility_procs.insert(it->get_utility_processor());
      // Log utility processors
      for (std::set<Processor>::const_iterator it = utility_procs.begin();
            it != utility_procs.end(); it++)
        LegionSpy::log_utility_processor(it->id);
      // Log processors
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        Processor::Kind k = machine->get_processor_kind(*it);
        LegionSpy::log_processor(it->id, it->get_utility_processor().id, k); 
      }
      // Log memories
      const std::set<Memory> &all_mems = machine->get_all_memories();
      for (std::set<Memory>::const_iterator it = all_mems.begin();
            it != all_mems.end(); it++)
        LegionSpy::log_memory(it->id, machine->get_memory_size(*it));
      // Log Proc-Mem Affinity
      for (std::set<Processor>::const_iterator pit = all_procs.begin();
            pit != all_procs.end(); pit++)
      {
        std::vector<ProcessorMemoryAffinity> affinities;
        machine->get_proc_mem_affinity(affinities, *pit);
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
        machine->get_mem_mem_affinity(affinities, *mit);
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
      Machine *machine = Machine::get_machine();
      const std::set<Processor> &all_procs = machine->get_all_processors();
      Processor::Kind proc_kind = machine->get_processor_kind(p);
      if (proc_kind != Processor::UTIL_PROC)
        p.enable_idle_task();
      // Make separate runtime instances if they are requested,
      // otherwise only make a runtime instances for each of the
      // separate nodes in the machine.  To do this we exploit a
      // little bit of knowledge about the naming scheme for low-level
      // processor objects that works on both the shared and general
      // low-level runtimes.
#ifdef DEBUG_HIGH_LEVEL
      assert((p.id & 0xffff) < (MAX_NUM_PROCS+1));
#endif 
#ifndef SHARED_LOWLEVEL
      if (separate_runtime_instances || ((p.id & 0xffff) == 0))
#else
      if (separate_runtime_instances || (p.id == 1))
#endif
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
          // If we are doing separate runtime instances then each
          // processor effectively gets its own address space
          local_procs.insert(p);
          local_procs.insert(p.get_utility_processor());
          AddressSpaceID sid = 0;
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++,sid++)
          {
            if (p == (*it))
              local_space_id = sid;
            address_spaces.insert(sid); 
            proc_spaces[*it] = sid;
            Processor util = it->get_utility_processor();
            if (util != (*it))
            {
              log_run(LEVEL_ERROR,"Separate runtime instances are not "
                                  "supported when running with explicit "
                                  "utility processors");
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_SEPARATE_UTILITY_PROCS);
            }
          }
        }
        else
        {
#ifndef SHARED_LOWLEVEL
          std::map<unsigned,AddressSpaceID> address_space_indexes;
          // Compute an index for each address space
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++)
          {
            const unsigned space = (it->id & 0xffff0000);
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
              address_space_indexes.find(it->id & 0xffff0000);
#ifdef DEBUG_HIGH_LEVEL
            assert(finder != address_space_indexes.end());
#endif
            AddressSpaceID sid = finder->second;
            proc_spaces[*it] = sid;
            if (sid == local_space_id)
            {
              if (machine->get_processor_kind(*it) == Processor::UTIL_PROC)
                local_util_procs.insert(*it);
              else
                local_procs.insert(*it);
            }
          }
#else
          // There is only one space so let local space ID be zero
          address_spaces.insert(local_space_id);
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++)
          {
            proc_spaces[*it] = local_space_id;
            if (machine->get_processor_kind(*it) == Processor::UTIL_PROC)
              local_util_procs.insert(*it);
            else
              local_procs.insert(*it);
          }
#endif
        }
        if (local_procs.size() > MAX_NUM_PROCS)
        {
          log_run(LEVEL_ERROR,"Maximum number of local processors %ld exceeds "
                              "compile time maximum of %d.  Change the value "
                              "in legion_type.h and recompile.",
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
        Runtime *local_rt = new Runtime(machine, local_space_id, local_procs,
                                        address_spaces, proc_spaces,
                                        cleanup_proc, gc_proc, message_proc);
        // Now set up the runtime on all of the local processors
        // and their utility processors
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          runtime_map[(it->id & 0xffff)] = local_rt;
        }
        for (std::set<Processor>::const_iterator it = local_util_procs.begin();
              it != local_util_procs.end(); it++)
        {
          runtime_map[(it->id & 0xffff)] = local_rt;
        }
      }
      // Arrive at the barrier
      __sync_fetch_and_add(&Runtime::startup_arrivals, 1);
      // Compute the number of processors we need to wait for
      unsigned needed_count = 0;
      {
        std::set<Processor> utility_procs;
#ifndef SHARED_LOWLEVEL
        const unsigned local_space = 0xffff0000 & p.id;
#endif
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
#ifndef SHARED_LOWLEVEL
          if (local_space != (0xffff0000 & it->id))
            continue;
#endif
          needed_count++;
        }
      }
      // Yes there is a race condition here on writes, but
      // everyone is going to be writing the same value
      // so it doesn't matter.
      Runtime::shutdown_counter = needed_count;
      // Have a spinning barrier here to wait for all processors
      // to finish initializing before continuing
      while (__sync_fetch_and_add(&Runtime::startup_arrivals, 0) 
              != needed_count) { }
      // Call in the runtime to see if we should launch the top-level task
      if (proc_kind != Processor::UTIL_PROC)
      {
        Runtime *local_rt = Runtime::get_runtime(p);
#ifdef DEBUG_HIGH_LEVEL
        assert(local_rt != NULL);
#endif
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
    /*static*/ void Runtime::schedule_runtime(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      Runtime::get_runtime(p)->process_schedule_request(p);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::message_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      Runtime::get_runtime(p)->process_message_task(args, arglen);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::post_end_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      SingleTask *task = *((SingleTask**)args);
      task->post_end_task();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::deferred_complete_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      Operation *op = *((Operation**)args);
      op->deferred_complete();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::reclaim_local_field_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      FieldID fid;
      derez.deserialize(fid);
      Runtime::get_runtime(p)->finalize_field_destroy(handle, fid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::deferred_collect_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      PhysicalView::handle_deferred_collect(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::trigger_op_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      const ProcessorManager::TriggerOpArgs *trigger_args = 
                                  (const ProcessorManager::TriggerOpArgs*)args;
      Operation *op = trigger_args->op;
      bool mapped = op->trigger_execution();
      if (!mapped)
      {
        ProcessorManager *manager = trigger_args->manager;
        manager->add_to_local_ready_queue(op, true/*failure*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::trigger_task_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {
      const ProcessorManager::TriggerTaskArgs *trigger_args = 
                                (const ProcessorManager::TriggerTaskArgs*)args;
      TaskOp *op = trigger_args->op; 
      bool mapped = op->trigger_execution();
      if (!mapped)
      {
        ProcessorManager *manager = trigger_args->manager;
        manager->add_to_ready_queue(op, true/*failure*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::legion_logging_task(
                                  const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------
    {

    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

