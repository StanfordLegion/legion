/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// SJT: this comes first because some systems require __STDC_FORMAT_MACROS
//  to be defined before inttypes.h is included anywhere
#ifdef __MACH__
#define MASK_FMT "llx"
#else
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#define MASK_FMT PRIx64
#endif

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/legion_trace.h"
#include "legion/legion_utilities.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_context.h"
#include "legion/mapper_manager.h"
#include "legion/garbage_collection.h"
#include "mappers/default_mapper.h"
#include "mappers/test_mapper.h"
#include "mappers/replay_mapper.h"
#include "mappers/debug_mapper.h"

#include <unistd.h> // sleep for warnings

#define REPORT_DUMMY_CONTEXT(message)                        \
  REPORT_LEGION_ERROR(ERROR_DUMMY_CONTEXT_OPERATION,  message)

namespace Legion {
  namespace Internal {

    // If you add a logger, update the LEGION_EXTERN_LOGGER_DECLARATIONS
    // macro in legion_types.h
    Realm::Logger log_run("runtime");
    Realm::Logger log_task("tasks");
    Realm::Logger log_index("index_spaces");
    Realm::Logger log_field("field_spaces");
    Realm::Logger log_region("regions");
    Realm::Logger log_inst("instances");
    Realm::Logger log_variant("variants");
    Realm::Logger log_allocation("allocation");
    Realm::Logger log_prof("legion_prof");
    Realm::Logger log_garbage("legion_gc");
    Realm::Logger log_shutdown("shutdown");
    namespace LegionSpy {
      Realm::Logger log_spy("legion_spy");
    };

    __thread TaskContext *implicit_context = NULL;
    __thread AutoLock *local_lock_list = NULL;

    const LgEvent LgEvent::NO_LG_EVENT = LgEvent();
    const ApEvent ApEvent::NO_AP_EVENT = ApEvent();
    const ApUserEvent ApUserEvent::NO_AP_USER_EVENT = ApUserEvent();
    const ApBarrier ApBarrier::NO_AP_BARRIER = ApBarrier();
    const RtEvent RtEvent::NO_RT_EVENT = RtEvent();
    const RtUserEvent RtUserEvent::NO_RT_USER_EVENT = RtUserEvent();
    const RtBarrier RtBarrier::NO_RT_BARRIER = RtBarrier();
    const PredEvent PredEvent::NO_PRED_EVENT = PredEvent();

    /////////////////////////////////////////////////////////////
    // Argument Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(void)
      : Collectable(), runtime(Runtime::get_runtime(
                                        Processor::get_executing_processor())), 
        future_map(NULL), equivalent(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(const FutureMap &rhs)
      : Collectable(), runtime(Runtime::get_runtime(
                                        Processor::get_executing_processor())), 
        future_map(rhs.impl), equivalent(false)
    //--------------------------------------------------------------------------
    {
      if (future_map != NULL)
        future_map->add_base_gc_ref(FUTURE_HANDLE_REF);
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(const ArgumentMapImpl &impl)
      : Collectable(), runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // This should never ever be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    ArgumentMapImpl::~ArgumentMapImpl(void)
    //--------------------------------------------------------------------------
    {
      if ((future_map != NULL) && 
            future_map->remove_base_gc_ref(FUTURE_HANDLE_REF))
        delete (future_map);
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl& ArgumentMapImpl::operator=(const ArgumentMapImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // This should never ever be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMapImpl::has_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (future_map != NULL)
        unfreeze();
      return (arguments.find(point) != arguments.end());
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::set_point(const DomainPoint &point, 
                                const TaskArgument &arg,
                                bool replace)
    //--------------------------------------------------------------------------
    {
      if (future_map != NULL)
        unfreeze();
      std::map<DomainPoint,Future>::iterator finder = arguments.find(point);
      if (finder != arguments.end())
      {
        // If it already exists and we're not replacing it then we're done
        if (!replace)
          return;
        if (arg.get_size() > 0)
          finder->second = 
            Future::from_untyped_pointer(runtime->external,
                                         arg.get_ptr(), arg.get_size());
        else
          finder->second = Future();
      }
      else
      {
        if (arg.get_size() > 0)
          arguments[point] = 
            Future::from_untyped_pointer(runtime->external,
                                         arg.get_ptr(), arg.get_size());
        else
          arguments[point] = Future();
      }
      // If we modified things then they are no longer equivalent
      if (future_map != NULL)
      {
        equivalent = false;
        if (future_map->remove_base_gc_ref(FUTURE_HANDLE_REF))
          delete (future_map);
        future_map = NULL;
      }
    }

    //--------------------------------------------------------------------------
    bool ArgumentMapImpl::remove_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (future_map != NULL)
        unfreeze();
      std::map<DomainPoint,Future>::iterator finder = arguments.find(point);
      if (finder != arguments.end())
      {
        arguments.erase(finder);
        // If we modified things then they are no longer equivalent
        if (future_map != NULL)
        {
          equivalent = false;
          if (future_map->remove_base_gc_ref(FUTURE_HANDLE_REF))
            delete (future_map);
          future_map = NULL;
        }
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMapImpl::get_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (future_map != NULL)
        unfreeze();
      std::map<DomainPoint,Future>::const_iterator finder=arguments.find(point);
      if ((finder == arguments.end()) || (finder->second.impl == NULL))
        return TaskArgument();
      return TaskArgument(finder->second.impl->get_untyped_result(),
                          finder->second.impl->get_untyped_size());
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* ArgumentMapImpl::freeze(TaskContext *ctx)
    //--------------------------------------------------------------------------
    {
      // If we already have a future map then we are good
      if (future_map != NULL)
        return future_map;
      // If we have no futures then we can return an empty map
      if (arguments.empty())
        return NULL;
      // Otherwise we have to make a future map and set all the futures
      // We know that they are already completed 
      DistributedID did = runtime->get_available_distributed_id();
      future_map = new FutureMapImpl(ctx, runtime, did, 
                                     runtime->address_space);
      future_map->add_base_gc_ref(FUTURE_HANDLE_REF);
      future_map->set_all_futures(arguments);
      equivalent = true; // mark that these are equivalent
      return future_map;
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::unfreeze(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(future_map != NULL);
#endif
      // If they are already equivalent then we're done
      if (equivalent)
        return;
      // Otherwise we need to make them equivalent
      future_map->get_all_futures(arguments);
      equivalent = true;
    }

    /////////////////////////////////////////////////////////////
    // Future Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(Runtime *rt, bool register_now, DistributedID did,
                           AddressSpaceID own_space, Operation *o /*= NULL*/)
      : DistributedCollectable(rt, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_DC), 
          own_space, register_now),
        producer_op(o), op_gen((o == NULL) ? 0 : o->get_generation()),
#ifdef LEGION_SPY
        producer_uid((o == NULL) ? 0 : o->get_unique_op_id()),
#endif
        ready_event(Runtime::create_ap_user_event()), 
        result(NULL), result_size(0), empty(true), sampled(false)
    //--------------------------------------------------------------------------
    {
      if (producer_op != NULL)
        producer_op->add_mapping_reference(op_gen);
#ifdef LEGION_GC
      log_garbage.info("GC Future %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(const FutureImpl &rhs)
      : DistributedCollectable(NULL, 0, 0), producer_op(NULL), op_gen(0)
#ifdef LEGION_SPY
        , producer_uid(0)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void)
    //--------------------------------------------------------------------------
    {
      // don't want to leak events
      if (!ready_event.has_triggered())
        Runtime::trigger_event(ready_event);
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
    FutureImpl& FutureImpl::operator=(const FutureImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }
    
    //--------------------------------------------------------------------------
    void FutureImpl::get_void_result(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (Runtime::runtime_warnings && !silence_warnings && 
          (producer_op != NULL))
      {
        TaskContext *context = producer_op->get_context();
        if (!context->is_leaf_context())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_WAITING_FUTURE_NONLEAF,
             "Waiting on a future in non-leaf task %s "
             "(UID %lld) is a violation of Legion's deferred execution model "
             "best practices. You may notice a severe performance degradation.",
             context->get_task_name(), context->get_unique_id());
        }
      }
      if (!ready_event.has_triggered())
      {
        TaskContext *context = 
          (producer_op == NULL) ? NULL : producer_op->get_context();
        if (context != NULL)
        {
          context->begin_task_wait(false/*from runtime*/);
          ready_event.lg_wait();
          context->end_task_wait();
        }
        else
          ready_event.lg_wait();
      }
      if (empty)
        REPORT_LEGION_ERROR(ERROR_ACCESSING_EMPTY_FUTURE, 
                            "Accessing empty future! (UID %lld)",
                            (producer_op == NULL) ? 0 : 
                              producer_op->get_unique_op_id())
      mark_sampled();
    }

    //--------------------------------------------------------------------------
    void* FutureImpl::get_untyped_result(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (Runtime::runtime_warnings && !silence_warnings && 
          (producer_op != NULL))
      {
        TaskContext *context = producer_op->get_context();
        if (!context->is_leaf_context())
          REPORT_LEGION_WARNING(LEGION_WARNING_FUTURE_NONLEAF, 
             "Waiting on a future in non-leaf task %s "
             "(UID %lld) is a violation of Legion's deferred execution model "
             "best practices. You may notice a severe performance degradation.",
             context->get_task_name(), context->get_unique_id())
      }
      if (!ready_event.has_triggered())
      {
        TaskContext *context = 
          (producer_op == NULL) ? NULL : producer_op->get_context();
        if (context != NULL)
        {
          context->begin_task_wait(false/*from runtime*/);
          ready_event.lg_wait();
          context->end_task_wait();
        }
        else
          ready_event.lg_wait();
      }
      if (empty)
        REPORT_LEGION_ERROR(ERROR_ACCESSING_EMPTY_FUTURE, 
                            "Accessing empty future! (UID %lld)",
                            (producer_op == NULL) ? 0 :
                              producer_op->get_unique_op_id())
      mark_sampled();
      return result;
    }

    //--------------------------------------------------------------------------
    size_t FutureImpl::get_untyped_size(void)
    //--------------------------------------------------------------------------
    {
      // Call this first to make sure the future is ready
      get_void_result();
      return result_size;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::is_empty(bool block, bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (Runtime::runtime_warnings && !silence_warnings && 
          (producer_op != NULL))
      {
        TaskContext *context = producer_op->get_context();
        if (!context->is_leaf_context())
          REPORT_LEGION_WARNING(LEGION_WARNING_BLOCKING_EMPTY, 
              "Performing a blocking is_empty test on a "
              "in non-leaf task %s (UID %lld) is a violation of Legion's "
              "deferred execution model best practices. You may notice a "
              "severe performance degradation.", context->get_task_name(), 
              context->get_unique_id())
      }
      if (block && !ready_event.has_triggered())
      {
        TaskContext *context = 
          (producer_op == NULL) ? NULL : producer_op->get_context();
        if (context != NULL)
        {
          context->begin_task_wait(false/*from runtime*/);
          ready_event.lg_wait();
          context->end_task_wait();
        }
        else
          ready_event.lg_wait();
      }
      if (block)
        mark_sampled();
      return empty;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *args, size_t arglen, bool own)
    //--------------------------------------------------------------------------
    {
      // Should only happen on the owner
#ifdef DEBUG_LEGION
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
    void FutureImpl::unpack_future(Deserializer &derez)
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
    void FutureImpl::complete_future(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!ready_event.has_triggered());
#endif
      Runtime::trigger_event(ready_event);
      // If we're the owner send our result to any remote spaces
      if (is_owner())
        broadcast_result();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::reset_future(void)
    //--------------------------------------------------------------------------
    {
      if (ready_event.has_triggered())
        ready_event = Runtime::create_ap_user_event();
      bool was_sampled = sampled;
      sampled = false;
      return was_sampled;
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::get_boolean_value(bool &valid)
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
    void FutureImpl::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, send a gc reference back to the owner
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, remove our gc reference
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::register_dependence(Operation *consumer_op)
    //--------------------------------------------------------------------------
    {
      if (producer_op != NULL)
        consumer_op->register_dependence(producer_op, op_gen);
#ifdef DEBUG_LEGION
      else
        assert(!empty); // better not be empty if it doesn't have an op
#endif
    }

    //--------------------------------------------------------------------------
    void FutureImpl::mark_sampled(void)
    //--------------------------------------------------------------------------
    {
      sampled = true;
    }

    //--------------------------------------------------------------------------
    void FutureImpl::broadcast_result(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Need to hold the lock when reading the set of remote spaces
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      if (!registered_waiters.empty())
      {
        Serializer rez;
        {
          rez.serialize(did);
          RezCheck z(rez);
          rez.serialize(result_size);
          rez.serialize(result,result_size);
        }
        for (std::set<AddressSpaceID>::const_iterator it = 
              registered_waiters.begin(); it != registered_waiters.end(); it++)
        {
          runtime->send_future_result(*it, rez); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void FutureImpl::register_waiter(AddressSpaceID sid)
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
    void FutureImpl::record_future_registered(ReferenceMutator *creator)
    //--------------------------------------------------------------------------
    {
      // Similar to DistributedCollectable::register_with_runtime but
      // we don't actually need to do the registration since we know
      // it has already been done
#ifdef DEBUG_LEGION
      assert(!registered_with_runtime);
#endif
      registered_with_runtime = true;
      if (!is_owner())
      {
        // Send the remote registration notice
        send_remote_registration(creator);
        // Then send the subscription for this future
        register_waiter(runtime->address_space);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureImpl::handle_future_result(Deserializer &derez,
                                                 Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      FutureImpl *future = dynamic_cast<FutureImpl*>(dc);
      assert(future != NULL);
#else
      FutureImpl *future = static_cast<FutureImpl*>(dc);
#endif
      future->unpack_future(derez);
      future->complete_future();
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureImpl::handle_future_subscription(
                                         Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID subscriber;
      derez.deserialize(subscriber);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      FutureImpl *future = dynamic_cast<FutureImpl*>(dc);
      assert(future != NULL);
#else
      FutureImpl *future = static_cast<FutureImpl*>(dc);
#endif
      future->register_waiter(subscriber); 
    }

    //--------------------------------------------------------------------------
    void FutureImpl::contribute_to_collective(const DynamicCollective &dc, 
                                              unsigned count)
    //--------------------------------------------------------------------------
    {
      if (!ready_event.has_triggered())
      {
        // If we're not done then defer the operation until we are triggerd
        // First add a garbage collection reference so we don't get
        // collected while we are waiting for the contribution task to run
        add_base_gc_ref(PENDING_COLLECTIVE_REF);
        ContributeCollectiveArgs args;
        args.impl = this;
        args.dc = dc;
        args.count = count;
        // Spawn the task dependent on the future being ready
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, NULL,
                                         Runtime::protect_event(ready_event));
      }
      else // If we've already triggered, then we can do the arrival now
        Runtime::phase_barrier_arrive(dc, count, ApEvent::NO_AP_EVENT,
                                      result, result_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureImpl::handle_contribute_to_collective(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const ContributeCollectiveArgs *cargs = (ContributeCollectiveArgs*)args;
      cargs->impl->contribute_to_collective(cargs->dc, cargs->count);
      // Now remote the garbage collection reference and see if we can 
      // reclaim the future
      if (cargs->impl->remove_base_gc_ref(PENDING_COLLECTIVE_REF))
        delete cargs->impl;
    }
      
    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(TaskContext *ctx, Operation *o, Runtime *rt,
                                 DistributedID did, AddressSpaceID owner_space)
      : DistributedCollectable(rt, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_MAP_DC),  owner_space), 
        context(ctx), op(o), op_gen(o->get_generation()),
        ready_event(o->get_completion_event()), valid(true)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Future Map %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(TaskContext *ctx, Runtime *rt,
                                 DistributedID did, AddressSpaceID owner_space,
                                 bool register_now)
      : DistributedCollectable(rt, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, FUTURE_MAP_DC), 
          owner_space, register_now), 
        context(ctx), op(NULL), op_gen(0),
        ready_event(ApEvent::NO_AP_EVENT), valid(!is_owner())
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Future Map %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(const FutureMapImpl &rhs)
      : DistributedCollectable(rhs), context(NULL), op(NULL), 
        op_gen(0), valid(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::~FutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      futures.clear();
    }

    //--------------------------------------------------------------------------
    FutureMapImpl& FutureMapImpl::operator=(const FutureMapImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, send a gc reference back to the owner
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1/*count*/, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // If we are not the owner, remove our gc reference
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    Future FutureMapImpl::get_future(const DomainPoint &point, bool allow_empty)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
      {
        // See if we already have it
        {
          AutoLock m_lock(gc_lock,1,false/*exlusive*/);
          std::map<DomainPoint,Future>::const_iterator finder = 
                                                futures.find(point);
          if (finder != futures.end())
            return finder->second;
        }
        // Make an event for when we have the answer
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        // If not send a message to get it
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(point);
          rez.serialize<bool>(allow_empty);
          rez.serialize(ready_event);
        }
        runtime->send_future_map_request_future(owner_space, rez);
        ready_event.lg_wait(); 
        // When we wake up it should be here
        AutoLock m_lock(gc_lock,1,false/*exlusive*/);
        std::map<DomainPoint,Future>::const_iterator finder = 
                                              futures.find(point);
        if (allow_empty && (finder == futures.end()))
          return Future();
#ifdef DEBUG_LEGION
        assert(finder != futures.end());
#endif
        return finder->second;
      }
      else
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        // Check to make sure we are asking for something in the domain
        if (!allow_empty && (valid_points.find(point) == valid_points.end()))
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
        if (allow_empty && ready_event.exists())
          ready_event.lg_wait();
        if (valid)
        {
          Future result;
          {
            AutoLock gc(gc_lock);
            // Check to see if we already have a future for the point
            std::map<DomainPoint,Future>::const_iterator finder = 
                                                  futures.find(point);
            if (finder != futures.end())
              return finder->second;
            if (allow_empty)
              return Future();
            // Otherwise we need a future from the context to use for
            // the point that we will fill in later
            result = runtime->help_create_future(op);
            futures[point] = result;
          }
          if (Runtime::legion_spy_enabled)
            LegionSpy::log_future_creation(op->get_unique_op_id(),
                           result.impl->get_ready_event(), point);
          return result;
        }
        else if (allow_empty)
          return Future();
        else
          return runtime->help_create_future();
      }
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_future(const DomainPoint &point, FutureImpl *impl,
                                   ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner()); // should never be called on the owner node
#endif
      // Add the reference first and then set the future
      impl->add_base_gc_ref(FUTURE_HANDLE_REF, mutator);
      AutoLock g_lock(gc_lock);
      futures[point] = Future(impl, false/*need reference*/);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::get_void_result(const DomainPoint &point,
                                        bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      Future f = get_future(point);
      f.get_void_result(silence_warnings);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::wait_all_results(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      if (Runtime::runtime_warnings && !silence_warnings && 
          (context != NULL) && !context->is_leaf_context())
        REPORT_LEGION_WARNING(LEGION_WARNING_WAITING_ALL_FUTURES, 
            "Waiting for all futures in a future map in "
            "non-leaf task %s (UID %lld) is a violation of Legion's deferred "
            "execution model best practices. You may notice a severe "
            "performance degredation.", context->get_task_name(),
            context->get_unique_id())
      // Wait on the event that indicates the entire task has finished
      if (valid && !ready_event.has_triggered())
      {
        if (context != NULL)
        {
          context->begin_task_wait(false/*from runtime*/);
          ready_event.lg_wait();
          context->end_task_wait();
        }
        else
          ready_event.lg_wait();
      }
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::complete_all_futures(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(valid);
#endif
      AutoLock l_lock(gc_lock);
      for (std::map<DomainPoint,Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        runtime->help_complete_future(it->second);
      }
    }

    //--------------------------------------------------------------------------
    bool FutureMapImpl::reset_all_futures(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(valid);
#endif
      bool result = false;
      AutoLock l_lock(gc_lock);
      for (std::map<DomainPoint,Future>::const_iterator it = 
            futures.begin(); it != futures.end(); it++)
      {
        bool restart = runtime->help_reset_future(it->second);
        if (restart)
          result = true;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::get_all_futures(
                                     std::map<DomainPoint,Future> &others) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(valid);
#endif
      if (!ready_event.has_triggered())
      {
        if (context != NULL)
        {
          context->begin_task_wait(false/*from runtime*/);
          ready_event.lg_wait();
          context->end_task_wait();
        }
        else
          ready_event.lg_wait();
      }
      // No need for the lock since the map should be fixed at this point
      others = futures;
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_all_futures(
                                     const std::map<DomainPoint,Future> &others)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(!valid);
#endif
      // No need for the lock here since we're initializing
      futures = others;
      valid = true;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void FutureMapImpl::add_valid_domain(const Domain &d)
    //--------------------------------------------------------------------------
    {
      assert(is_owner());
      valid_domains.push_back(d);
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::add_valid_point(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      assert(is_owner());
      valid_points.insert(dp);
    }
#endif

    //--------------------------------------------------------------------------
    void FutureMapImpl::record_future_map_registered(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Similar to DistributedCollectable::register_with_runtime but
      // we don't actually need to do the registration since we know
      // it has already been done
#ifdef DEBUG_LEGION
      assert(!registered_with_runtime);
#endif
      registered_with_runtime = true;
      if (!is_owner())
        // Send the remote registration notice
        send_remote_registration(mutator);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureMapImpl::handle_future_map_future_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DomainPoint point;
      derez.deserialize(point);
      bool allow_empty;
      derez.deserialize(allow_empty);
      RtUserEvent done;
      derez.deserialize(done);
      
      // Should always find it since this is the owner node
      FutureMapImpl *impl = runtime->find_or_create_future_map(did, NULL, NULL);
      Future f = impl->get_future(point, allow_empty);
      if (allow_empty && (f.impl == NULL))
      {
        Runtime::trigger_event(done);
        return;
      }
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(point);
        rez.serialize(f.impl->did);
        rez.serialize(done);
      }
      runtime->send_future_map_response_future(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FutureMapImpl::handle_future_map_future_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DomainPoint point;
      derez.deserialize(point);
      DistributedID future_did;
      derez.deserialize(future_did);
      RtUserEvent done;
      derez.deserialize(done);
      
      // Should always find it since this is the source node
      FutureMapImpl *impl = runtime->find_or_create_future_map(did, NULL, NULL);
      std::set<RtEvent> done_events;
      WrapperReferenceMutator mutator(done_events);
      FutureImpl *future = runtime->find_or_create_future(future_did, &mutator);
      // Add it to the map
      impl->set_future(point, future, &mutator);
      // Trigger the done event
      if (!done_events.empty())
        Runtime::trigger_event(done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(done);
    }

    /////////////////////////////////////////////////////////////
    // Physical Region Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(const RegionRequirement &r, 
                                   ApEvent ready, bool m, TaskContext *ctx, 
                                   MapperID mid, MappingTagID t, 
                                   bool leaf, bool virt, Runtime *rt)
      : Collectable(), runtime(rt), context(ctx), map_id(mid), tag(t),
        leaf_region(leaf), virtual_mapped(virt), ready_event(ready), req(r), 
        mapped(m), valid(false), trigger_on_unmap(false), made_accessor(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(const PhysicalRegionImpl &rhs)
      : Collectable(), runtime(NULL), context(NULL), map_id(0), tag(0),
        leaf_region(false), virtual_mapped(false), 
        ready_event(ApEvent::NO_AP_EVENT), mapped(false), valid(false), 
        trigger_on_unmap(false), made_accessor(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::~PhysicalRegionImpl(void)
    //--------------------------------------------------------------------------
    {
      // If we still have a trigger on unmap, do that before
      // deleting ourselves to avoid leaking events
      if (trigger_on_unmap)
      {
        trigger_on_unmap = false;
        Runtime::trigger_event(termination_event);
      }
      if (!references.empty())
        references.remove_valid_references(PHYSICAL_REGION_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl& PhysicalRegionImpl::operator=(
                                                  const PhysicalRegionImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::wait_until_valid(bool silence_warnings, 
                                              bool warn, const char *source)
    //--------------------------------------------------------------------------
    {
      if (Runtime::runtime_warnings && !silence_warnings &&
          (context != NULL) && !context->is_leaf_context())
      {
        if (source != NULL)
          REPORT_LEGION_WARNING(LEGION_WARNING_WAITING_REGION, 
              "Waiting for a physical region to be valid "
              "for call %s in non-leaf task %s (UID %lld) is a violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation.", source,
              context->get_task_name(), context->get_unique_id())
        else
          REPORT_LEGION_WARNING(LEGION_WARNING_WAITING_REGION, 
              "Waiting for a physical region to be valid "
              "in non-leaf task %s (UID %lld) is a violation of Legion's "
              "deferred execution model best practices. You may notice a "
              "severe performance degradation.", context->get_task_name(), 
              context->get_unique_id())
      }
#ifdef DEBUG_LEGION
      assert(mapped); // should only be waiting on mapped regions
#endif
      // If we've already gone through this process we're good
      if (valid)
        return;
      if (!ready_event.has_triggered())
      {
        if (warn && !silence_warnings && (source != NULL))
          REPORT_LEGION_WARNING(LEGION_WARNING_MISSING_REGION_WAIT, 
              "Request for %s was performed on a "
              "physical region in task %s (ID %lld) without first waiting "
              "for the physical region to be valid. Legion is performing "
              "the wait for you.", source, context->get_task_name(), 
              context->get_unique_id())
        if (context != NULL)
          context->begin_task_wait(false/*from runtime*/);
        ready_event.lg_wait();
        if (context != NULL)
          context->end_task_wait();
      }
      // Now wait for the reference to be ready
      std::set<ApEvent> wait_on;
      references.update_wait_on_events(wait_on);
      ApEvent ref_ready = Runtime::merge_events(wait_on);
      bool poisoned;
      if (!ref_ready.has_triggered_faultaware(poisoned))
      {
        if (!poisoned)
        {
          if (context != NULL)
            context->begin_task_wait(false/*from runtime*/);
          ref_ready.wait_faultaware(poisoned);
          if (context != NULL)
            context->end_task_wait();
        }
      }
      valid = true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (valid)
        return true;
      if (mapped && ready_event.has_triggered())
      {
        std::set<ApEvent> wait_on;
        references.update_wait_on_events(wait_on);
        ApEvent ref_ready = Runtime::merge_events(wait_on);
        return ref_ready.has_triggered();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::is_mapped(void) const
    //--------------------------------------------------------------------------
    {
      return mapped;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::is_external_region(void) const
    //--------------------------------------------------------------------------
    {
      if (references.empty())
        return false;
      for (unsigned idx = 0; idx < references.size(); idx++)
        if (!references[idx].get_manager()->is_external_instance())
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegionImpl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      return req.region;
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        PhysicalRegionImpl::get_accessor(bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (context != NULL)
      {
        if (context->is_inner_context())
          REPORT_LEGION_ERROR(ERROR_INNER_TASK_VIOLATION, 
            "Illegal call to 'get_accessor' inside task "
            "%s (UID %lld) for a variant that was labeled as an 'inner' "
            "variant.", context->get_task_name(), context->get_unique_id())
        else if (Runtime::runtime_warnings && !silence_warnings &&
                  !context->is_leaf_context())
          REPORT_LEGION_WARNING(LEGION_WARNING_NONLEAF_ACCESSOR, 
              "Call to 'get_accessor' in non-leaf task %s "
              "(UID %lld) is a blocking operation in violation of Legion's "
              "deferred execution model best practices. You may notice a "
              "severe performance degradation.", context->get_task_name(),
              context->get_unique_id())
      }
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        if (virtual_mapped)
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_IMPLICIT_MAPPING, 
                        "Illegal implicit mapping of a virtual mapped region "
                        "in task %s (UID %lld)", context->get_task_name(),
                        context->get_unique_id())
        if (Runtime::runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(LEGION_WARNING_UNMAPPED_ACCESSOR, 
                          "Request for 'get_accessor' was "
                          "performed on an unmapped region in task %s "
                          "(UID %lld). Legion is mapping it for you. "
                          "Please try to be more careful.",
                          context->get_task_name(), context->get_unique_id())
        runtime->remap_region(context, PhysicalRegion(this));
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_LEGION
        assert(mapped);
#endif
      }
      // Wait until we are valid before returning the accessor
      wait_until_valid(silence_warnings, 
                       Runtime::runtime_warnings, "get_accessor");
      // You can only legally invoke this method when you have one instance
      if (references.size() > 1)
        REPORT_LEGION_ERROR(ERROR_DEPRECATED_METHOD_USE, 
                      "Illegal invocation of deprecated 'get_accessor' method "
                      "in task %s (ID %lld) on a PhysicalRegion containing "
                      "multiple internal instances. Use of this deprecated "
                      "method is only supported if the PhysicalRegion contains "
                      "a single physical instance.", context->get_task_name(),
                      context->get_unique_id())
      made_accessor = true;
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          result = references[0].get_accessor();
      result.set_region_untyped(this);
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(
          (LegionRuntime::AccessorPrivilege)req.get_accessor_privilege()); 
#endif
      return result;
#else // privilege or bounds checks
      return references[0].get_accessor();
#endif
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          PhysicalRegionImpl::get_field_accessor(FieldID fid, 
                                                 bool silence_warnings)
    //--------------------------------------------------------------------------
    {
      if (context != NULL)
      {
        if (context->is_inner_context())
          REPORT_LEGION_ERROR(ERROR_INNER_TASK_VIOLATION, 
            "Illegal call to 'get_field_accessor' inside "
            "task %s (UID %lld) for a variant that was labeled as an 'inner' "
            "variant.", context->get_task_name(), context->get_unique_id())
        else if (Runtime::runtime_warnings && !silence_warnings &&
                  !context->is_leaf_context())
          REPORT_LEGION_WARNING(LEGION_WARNING_NONLEAF_ACCESSOR, 
              "Call to 'get_field_accessor' in non-leaf "
              "task %s (UID %lld) is a blocking operation in violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation.", 
              context->get_task_name(), context->get_unique_id())
      }
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        if (virtual_mapped)
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_IMPLICIT_MAPPING, 
                        "Illegal implicit mapping of a virtual mapped region "
                        "in task %s (UID %lld)", context->get_task_name(),
                        context->get_unique_id())
        if (Runtime::runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(LEGION_WARNING_UNMAPPED_ACCESSOR, 
                          "Request for 'get_field_accessor' was "
                          "performed on an unmapped region in task %s "
                          "(UID %lld). Legion is mapping it for you. "
                          "Please try to be more careful.",
                          context->get_task_name(), context->get_unique_id())
        runtime->remap_region(context, PhysicalRegion(this));
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_LEGION
        assert(mapped);
#endif 
      }
      // Wait until we are valid before returning the accessor
      wait_until_valid(silence_warnings, 
                       Runtime::runtime_warnings, "get_field_acessor");
#ifdef DEBUG_LEGION
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
        REPORT_LEGION_ERROR(ERROR_INVALID_FIELD_PRIVILEGES, 
            "Requested field accessor for field %d without privileges!", fid)
#endif
      made_accessor = true;
#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic>
          result = references.get_field_accessor(fid);
      result.set_region_untyped(this);
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(
          (LegionRuntime::AccessorPrivilege)req.get_accessor_privilege());
#endif
      return result;
#else // privilege or bounds checks
      return references.get_field_accessor(fid);
#endif
    } 

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::unmap_region(void)
    //--------------------------------------------------------------------------
    {
      if (!mapped)
        return;
      wait_until_valid(true/*silence warnings*/);
      if (trigger_on_unmap)
      {
        trigger_on_unmap = false;
        // Can only do the trigger when we have actually ready
        std::set<ApEvent> wait_on;
        references.update_wait_on_events(wait_on);
        wait_on.insert(ready_event);
        Runtime::trigger_event(termination_event,
                               Runtime::merge_events(wait_on));
      }
      valid = false;
      mapped = false;
      // If we have a wait for unmapped event, then we need to wait
      // before we return, this usually occurs because we had restricted
      // coherence on the region and we have to issue copies back to 
      // the restricted instances before we are officially unmapped
      bool poisoned;
      if (wait_for_unmap.exists() && 
          !wait_for_unmap.has_triggered_faultaware(poisoned))
      {
        if (!poisoned)
        {
          if (context != NULL)
            context->begin_task_wait(false/*from runtime*/);
          wait_for_unmap.lg_wait();
          if (context != NULL)
            context->end_task_wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::remap_region(ApEvent new_ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped);
#endif
      ready_event = new_ready;
      mapped = true;
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& PhysicalRegionImpl::get_requirement(void) const
    //--------------------------------------------------------------------------
    {
      return req;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::set_reference(const InstanceRef &ref)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref.has_ref());
#endif
      references.add_instance(ref);
      ref.add_valid_reference(PHYSICAL_REGION_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::reset_references(const InstanceSet &refs,
                                       ApUserEvent term_event, ApEvent wait_for)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapped);
#endif
      if (!references.empty())
        references.remove_valid_references(PHYSICAL_REGION_REF);
      references = refs;
      if (!references.empty())
        references.add_valid_references(PHYSICAL_REGION_REF);
      termination_event = term_event;
      trigger_on_unmap = true;
      wait_for_unmap = wait_for;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalRegionImpl::get_ready_event(void) const
    //--------------------------------------------------------------------------
    {
      return ready_event;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::has_references(void) const
    //--------------------------------------------------------------------------
    {
      return !references.empty();
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_references(InstanceSet &instances) const
    //--------------------------------------------------------------------------
    {
      instances = references;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_memories(std::set<Memory>& memories) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < references.size(); idx++)
        memories.insert(references[idx].get_memory());
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_fields(std::vector<FieldID>& fields) const
    //--------------------------------------------------------------------------
    {
      // Just get these from the region requirement
      fields.insert(fields.end(), req.privilege_fields.begin(),
                    req.privilege_fields.end());
    }


#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    //--------------------------------------------------------------------------
    const char* PhysicalRegionImpl::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      return context->get_task_name();
    }
#endif

#ifdef BOUNDS_CHECKS 
    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::contains_ptr(ptr_t ptr)
    //--------------------------------------------------------------------------
    {
      if (!bounds.exists())
        bounds = runtime->forest->get_node(req.region.get_index_space())->
                    get_color_space_domain();
      DomainPoint dp(ptr.value);
      return bounds.contains(dp);
    }
    
    //--------------------------------------------------------------------------
    bool PhysicalRegionImpl::contains_point(const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      if (!bounds.exists())
        bounds = runtime->forest->get_node(req.region.get_index_space())->
                    get_color_space_domain();
      return bounds.contains(dp);
    }
#endif

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::get_bounds(void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_domain(req.region.get_index_space(),
                                      realm_is, type_tag);
    }
    
    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalRegionImpl::get_instance_info(PrivilegeMode mode, 
                                              FieldID fid, size_t field_size, 
                                              void *realm_is, TypeTag type_tag,
                                              bool silence_warnings, 
                                              bool generic_accessor,
                                              bool check_field_size,
                                              ReductionOpID redop)
    //--------------------------------------------------------------------------
    { 
      // Check the privilege mode first
      switch (mode)
      {
        case READ_ONLY:
          {
            if (!(READ_ONLY & req.privilege))
              REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                            "Error creating read-only field accessor without "
                            "read-only privileges on field %d in task %s",
                            fid, context->get_task_name())
            break;
          }
        case READ_WRITE:
          {
            if (req.privilege == WRITE_DISCARD)
            {
              if (!silence_warnings)
                REPORT_LEGION_WARNING(LEGION_WARNING_READ_DISCARD, 
                                "creating read-write accessor for "
                                "field %d in task %s which only has "
                                "WRITE_DISCARD privileges. You may be "
                                "accessing uninitialized data.",
                                fid, context->get_task_name())
            }
            else if (req.privilege != READ_WRITE)
              REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                            "Error creating read-write field accessor without "
                            "read-write privileges on field %d in task %s",
                            fid, context->get_task_name())
            break;
          }
        case WRITE_DISCARD:
          {
            if (!(WRITE_DISCARD & req.privilege))
              REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                            "Error creating write-discard field accessor "
                            "without write privileges on field %d in task %s",
                            fid, context->get_task_name())
            break;
          }
        case REDUCE:
          {
            if ((REDUCE != req.privilege) || (redop != req.redop))
            {
              if (!(REDUCE & req.privilege))
                REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                              "Error creating reduction field accessor "
                              "without reduction privileges on field %d in "
                              "task %s", fid, context->get_task_name())
              else if (redop != req.redop)
                REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                              "Error creating reduction field accessor "
                              "with mismatched reduction operators %d and %d "
                              "on field %d in task %s", redop, req.redop,
                              fid, context->get_task_name())
              else
                REPORT_LEGION_ERROR(ERROR_ACCESSOR_PRIVILEGE_CHECK, 
                              "Error creating reduction-only field accessor "
                              "for a region requirement with more than "
                              "reduction-only privileges for field %d in task "
                              "%s. Please use a read-write accessor instead.",
                              fid, context->get_task_name())
            }
            break;
          }
        default: // rest of the privileges don't matter
          break;
      }
      if (context != NULL)
      {
        if (context->is_inner_context())
          REPORT_LEGION_ERROR(ERROR_INNER_TASK_VIOLATION, 
            "Illegal accessor construction inside "
            "task %s (UID %lld) for a variant that was labeled as an 'inner' "
            "variant.", context->get_task_name(), context->get_unique_id())
        else if (Runtime::runtime_warnings && !silence_warnings &&
                  !context->is_leaf_context())
          REPORT_LEGION_WARNING(LEGION_WARNING_NONLEAF_ACCESSOR, 
              "Accessor construction in non-leaf "
              "task %s (UID %lld) is a blocking operation in violation of "
              "Legion's deferred execution model best practices. You may "
              "notice a severe performance degradation.", 
              context->get_task_name(), context->get_unique_id())
      }
      // If this physical region isn't mapped, then we have to
      // map it before we can return an accessor
      if (!mapped)
      {
        if (virtual_mapped)
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_IMPLICIT_MAPPING, 
                        "Illegal implicit mapping of a virtual mapped region "
                        "in task %s (UID %lld)", context->get_task_name(),
                        context->get_unique_id())
        if (Runtime::runtime_warnings && !silence_warnings)
          REPORT_LEGION_WARNING(LEGION_WARNING_UNMAPPED_ACCESSOR, 
                          "Accessor construction was "
                          "performed on an unmapped region in task %s "
                          "(UID %lld). Legion is mapping it for you. "
                          "Please try to be more careful.",
                          context->get_task_name(), context->get_unique_id())
        runtime->remap_region(context, PhysicalRegion(this));
        // At this point we should have a new ready event
        // and be mapped
#ifdef DEBUG_LEGION
        assert(mapped);
#endif 
      }
      if (req.privilege_fields.find(fid) == req.privilege_fields.end())
        REPORT_LEGION_ERROR(ERROR_INVALID_FIELD_PRIVILEGES, 
                       "Accessor construction for field %d in task %s "
                       "without privileges!", fid, context->get_task_name())
      if (generic_accessor && Runtime::runtime_warnings && !silence_warnings)
        REPORT_LEGION_WARNING(LEGION_WARNING_GENERIC_ACCESSOR,
                              "Using a generic accessor for accessing a "
                              "physical instance of task %s (UID %lld). "
                              "Generic accessors are very slow and are "
                              "strongly discouraged for use in high "
                              "performance code.", context->get_task_name(),
                              context->get_unique_id())
      // Get the index space to use for the accessor
      runtime->get_index_space_domain(req.region.get_index_space(),
                                      realm_is, type_tag);
      // Wait until we are valid before returning the accessor
      wait_until_valid(silence_warnings, 
                       Runtime::runtime_warnings, "Accessor Construction");
      made_accessor = true;
      for (unsigned idx = 0; idx < references.size(); idx++)
      {
        const InstanceRef &ref = references[idx];
        if (ref.is_field_set(fid))
        {
          PhysicalManager *manager = ref.get_manager();
          if (check_field_size)
          {
            const size_t actual_size = 
              manager->region_node->column_source->get_field_size(fid);
            if (actual_size != field_size)
              REPORT_LEGION_ERROR(ERROR_ACCESSOR_FIELD_SIZE_CHECK,
                            "Error creating accessor for field %d with a "
                            "type of size %zd bytes when the field was "
                            "originally allocated with a size of %zd bytes "
                            "in task %s (UID %lld)",
                            fid, field_size, actual_size, 
                            context->get_task_name(), context->get_unique_id())
          }
          return manager->get_instance();
        }
      }
      // should never get here at worst there should have been an
      // error raised earlier in this function
      assert(false);
      return PhysicalInstance::NO_INST;
    } 

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::fail_bounds_check(DomainPoint p, FieldID fid,
                                               PrivilegeMode mode)
    //--------------------------------------------------------------------------
    {
      char point_string[128];
      sprintf(point_string," (");
      for (int d = 0; d < p.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          sprintf(buffer,"%lld", p[0]);
        else
          sprintf(buffer,",%lld", p[d]);
        strcat(point_string, buffer);
      }
      strcat(point_string,")");
      switch (mode)
      {
        case READ_ONLY:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure reading point %s from "
                          "field %d in task %s\n", point_string, fid,
                          context->get_task_name())
            break;
          }
        case READ_WRITE:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure geting a reference to point %s "
                          "from field %d in task %s\n", point_string, fid,
                          context->get_task_name())
            break;
          }
        case WRITE_DISCARD:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure writing to point %s in "
                          "field %d in task %s\n", point_string, fid,
                          context->get_task_name())
            break;
          }
        case REDUCE:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure reducing to point %s in "
                          "field %d in task %s\n", point_string, fid,
                          context->get_task_name())
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::fail_bounds_check(Domain dom, FieldID fid,
                                               PrivilegeMode mode)
    //--------------------------------------------------------------------------
    {
      char rect_string[256];
      sprintf(rect_string," (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          sprintf(buffer,"%lld", dom.lo()[0]);
        else
          sprintf(buffer,",%lld", dom.lo()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string,") - (");
      for (int d = 0; d < dom.get_dim(); d++)
      {
        char buffer[32];
        if (d == 0)
          sprintf(buffer,"%lld", dom.hi()[0]);
        else
          sprintf(buffer,",%lld", dom.hi()[d]);
        strcat(rect_string, buffer);
      }
      strcat(rect_string,")");
      switch (mode)
      {
        case READ_ONLY:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure getting a read-only reference "
                          "to rect %s from field %d in task %s\n", 
                          rect_string, fid, context->get_task_name())
            break;
          }
        case READ_WRITE:
          {
            REPORT_LEGION_ERROR(ERROR_ACCESSOR_BOUNDS_CHECK, 
                          "Bounds check failure geting a reference to rect %s "
                          "from field %d in task %s\n", rect_string, fid,
                          context->get_task_name())
            break;
          }
        default:
          assert(false);
      }
    }

    /////////////////////////////////////////////////////////////
    // Grant Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GrantImpl::GrantImpl(void)
      : acquired(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    GrantImpl::GrantImpl(const std::vector<ReservationRequest> &reqs)
      : requests(reqs), acquired(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    GrantImpl::GrantImpl(const GrantImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    GrantImpl::~GrantImpl(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    GrantImpl& GrantImpl::operator=(const GrantImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void GrantImpl::register_operation(ApEvent completion_event)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      completion_events.insert(completion_event);
    }

    //--------------------------------------------------------------------------
    ApEvent GrantImpl::acquire_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      if (!acquired)
      {
        grant_event = ApEvent::NO_AP_EVENT;
        for (std::vector<ReservationRequest>::const_iterator it = 
              requests.begin(); it != requests.end(); it++)
        {
          grant_event = ApEvent(it->reservation.acquire(it->mode, 
                                                it->exclusive, grant_event));
        }
        acquired = true;
      }
      return grant_event;
    }

    //--------------------------------------------------------------------------
    void GrantImpl::release_grant(void)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(grant_lock);
      ApEvent deferred_release = Runtime::merge_events(completion_events);
      for (std::vector<ReservationRequest>::const_iterator it = 
            requests.begin(); it != requests.end(); it++)
      {
        it->reservation.release(deferred_release);
      }
    }

    //--------------------------------------------------------------------------
    void GrantImpl::pack_grant(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      ApEvent pack_event = acquire_grant();
      rez.serialize(pack_event);
    }

    //--------------------------------------------------------------------------
    void GrantImpl::unpack_grant(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      ApEvent unpack_event;
      derez.deserialize(unpack_event);
      AutoLock g_lock(grant_lock);
#ifdef DEBUG_LEGION
      assert(!acquired);
#endif
      grant_event = unpack_event;
      acquired = true;
    }

    /////////////////////////////////////////////////////////////
    // MPI Legion Handshake Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPILegionHandshakeImpl::MPILegionHandshakeImpl(bool init_mpi, int mpi_parts,
                                                   int legion_parts)
      : init_in_MPI(init_mpi), mpi_participants(mpi_parts), 
        legion_participants(legion_parts)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MPILegionHandshakeImpl::MPILegionHandshakeImpl(
                                              const MPILegionHandshakeImpl &rhs)
      : init_in_MPI(false), mpi_participants(-1), legion_participants(-1)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MPILegionHandshakeImpl::~MPILegionHandshakeImpl(void)
    //--------------------------------------------------------------------------
    {
      mpi_wait_barrier.get_barrier().destroy_barrier();
      legion_wait_barrier.get_barrier().destroy_barrier();
    }

    //--------------------------------------------------------------------------
    MPILegionHandshakeImpl& MPILegionHandshakeImpl::operator=(
                                              const MPILegionHandshakeImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::initialize(void)
    //--------------------------------------------------------------------------
    {
      mpi_wait_barrier = PhaseBarrier(ApBarrier(
            Realm::Barrier::create_barrier(legion_participants)));
      legion_wait_barrier = PhaseBarrier(ApBarrier(
            Realm::Barrier::create_barrier(mpi_participants)));
      mpi_arrive_barrier = legion_wait_barrier;
      legion_arrive_barrier = mpi_wait_barrier;
      // Advance the two wait barriers
      Runtime::advance_barrier(mpi_wait_barrier);
      Runtime::advance_barrier(legion_wait_barrier);
      // Whoever is waiting first, we have to advance their arrive barriers
      if (init_in_MPI)
      {
        Runtime::phase_barrier_arrive(legion_arrive_barrier, legion_participants);
        Runtime::advance_barrier(mpi_wait_barrier);
      }
      else
      {
        Runtime::phase_barrier_arrive(mpi_arrive_barrier, mpi_participants);
        Runtime::advance_barrier(legion_wait_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::mpi_handoff_to_legion(void)
    //--------------------------------------------------------------------------
    {
      // Just have to do our arrival
      Runtime::phase_barrier_arrive(mpi_arrive_barrier, 1);
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::mpi_wait_on_legion(void)
    //--------------------------------------------------------------------------
    {
      // When we get this call, we know we have done 
      // all the arrivals so we can advance it
      Runtime::advance_barrier(mpi_arrive_barrier);
      // Wait for mpi to be ready to run
      // Note we use the external wait to be sure 
      // we don't get drafted by the Realm runtime
      ApBarrier previous = Runtime::get_previous_phase(mpi_wait_barrier);
      if (!previous.has_triggered())
      {
        // We can't call external wait directly on the barrier
        // right now, so as a work-around we'll make an event
        // and then wait on that
        ApUserEvent wait_on = Runtime::create_ap_user_event();
        Runtime::trigger_event(wait_on, previous);
        wait_on.external_wait();
      }
      // Now we can advance our wait barrier
      Runtime::advance_barrier(mpi_wait_barrier);
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::legion_handoff_to_mpi(void)
    //--------------------------------------------------------------------------
    {
      // Just have to do our arrival
      Runtime::phase_barrier_arrive(legion_arrive_barrier, 1);
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::legion_wait_on_mpi(void)
    //--------------------------------------------------------------------------
    {
      Runtime::advance_barrier(legion_arrive_barrier);
      // Wait for Legion to be ready to run
      // No need to avoid being drafted by the
      // Realm runtime here
      legion_wait_barrier.wait();
      // Now we can advance our wait barrier
      Runtime::advance_barrier(legion_wait_barrier);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier MPILegionHandshakeImpl::get_legion_wait_phase_barrier(void)
    //--------------------------------------------------------------------------
    {
      return legion_wait_barrier;
    }

    //--------------------------------------------------------------------------
    PhaseBarrier MPILegionHandshakeImpl::get_legion_arrive_phase_barrier(void)
    //--------------------------------------------------------------------------
    {
      return legion_arrive_barrier;
    }

    //--------------------------------------------------------------------------
    void MPILegionHandshakeImpl::advance_legion_handshake(void)
    //--------------------------------------------------------------------------
    {
      Runtime::advance_barrier(legion_wait_barrier);
      Runtime::advance_barrier(legion_arrive_barrier);
    }

    /////////////////////////////////////////////////////////////
    // MPI Legion Handshake Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MPIRankTable::MPIRankTable(Runtime *rt)
      : runtime(rt), participating(int(runtime->address_space) <
          Runtime::legion_collective_participating_spaces)
    //--------------------------------------------------------------------------
    {
      // We already have our contributions for each stage so
      // we can set the inditial participants to 1
      if (participating)
      {
        sent_stages.resize(Runtime::legion_collective_stages, false);
#ifdef DEBUG_LEGION
        assert(Runtime::legion_collective_stages > 0);
#endif
        stage_notifications.resize(Runtime::legion_collective_stages, 1);
        // Stage 0 always starts with 0 notifications since we'll 
        // explictcly arrive on it
	// Special case: if we expect a stage -1 message from a 
        // non-participating space, we'll count that as part of 
        // stage 0, it will make it a negative count, but the 
        // type is 'int' so we're good
	if ((Runtime::legion_collective_stages > 0) &&
	    (runtime->address_space <
	     (runtime->total_address_spaces -
	      Runtime::legion_collective_participating_spaces)))
	  stage_notifications[0] = -1;
        else
          stage_notifications[0] = 0;
      }
      if (runtime->total_address_spaces > 1)
        done_event = Runtime::create_rt_user_event();
      // Add ourselves to the set before any exchanges start
#ifdef DEBUG_LEGION
      assert(Runtime::mpi_rank >= 0);
#endif
      forward_mapping[Runtime::mpi_rank] = runtime->address_space;
    }
    
    //--------------------------------------------------------------------------
    MPIRankTable::MPIRankTable(const MPIRankTable &rhs)
      : runtime(NULL), participating(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MPIRankTable::~MPIRankTable(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MPIRankTable& MPIRankTable::operator=(const MPIRankTable &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

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
                Runtime::legion_collective_participating_spaces) ||
              (runtime->address_space >= (runtime->total_address_spaces -
                Runtime::legion_collective_participating_spaces)))
          {
            const bool all_stages_done =  send_explicit_stage(0);
            if (all_stages_done)
              complete_exchange();
          }
        }
        else
        {
          // We are not a participating node
          // so we just have to send notification to one node
          send_explicit_stage(-1);
        }
        // Wait for our done event to be ready
        done_event.lg_wait();
      }
#ifdef DEBUG_LEGION
      assert(forward_mapping.size() == runtime->total_address_spaces);
#endif
      // Reverse the mapping
      for (std::map<int,AddressSpace>::const_iterator it = 
            forward_mapping.begin(); it != forward_mapping.end(); it++)
        reverse_mapping[it->second] = it->first;
    }

    //--------------------------------------------------------------------------
    bool MPIRankTable::send_explicit_stage(int stage)
    //--------------------------------------------------------------------------
    {
      bool all_stages_done = false;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(stage);
        AutoLock r_lock(reservation);
#ifdef DEBUG_LEGION
        {
          size_t expected_size = 1;
          for (int idx = 0; idx < stage; idx++)
            expected_size *= Runtime::legion_collective_radix;
          assert(expected_size <= forward_mapping.size());
        }
        if (stage >= 0)
        {
          assert(stage < int(sent_stages.size()));
          assert(!sent_stages[stage]);
        }
#endif
        rez.serialize<size_t>(forward_mapping.size());
        for (std::map<int,AddressSpace>::const_iterator it = 
              forward_mapping.begin(); it != forward_mapping.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        // Mark that we're sending this stage
        if (stage >= 0)
        {
#ifdef DEBUG_LEGION
          assert(stage < int(stage_notifications.size()));
          if (stage == (Runtime::legion_collective_stages-1))
            assert(stage_notifications[stage] < 
                  Runtime::legion_collective_last_radix); 
          else
            assert(stage_notifications[stage] <
                  Runtime::legion_collective_radix);
#endif
          stage_notifications[stage]++;
          sent_stages[stage] = true;
          // Check to see if all the stages are done
          all_stages_done = (stage_notifications.back() == 
                      Runtime::legion_collective_last_radix); 
          if (all_stages_done)
          {
            for (int stage = 1; 
                  stage < Runtime::legion_collective_stages; stage++)
            {
              if (stage_notifications[stage-1] == 
                    Runtime::legion_collective_radix)
                continue;
              all_stages_done = false;
              break;
            }
          }
        }
      }
      if (stage == -1)
      {
        if (participating)
        {
          // Send back to the nodes that are not participating
          AddressSpaceID target = runtime->address_space +
            Runtime::legion_collective_participating_spaces;
#ifdef DEBUG_LEGION
          assert(target < runtime->total_address_spaces);
#endif
          runtime->send_mpi_rank_exchange(target, rez);
        }
        else
        {
          // Sent to a node that is participating
          AddressSpaceID target = runtime->address_space % 
            Runtime::legion_collective_participating_spaces;
          runtime->send_mpi_rank_exchange(target, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(stage >= 0);
#endif
        if (stage == (Runtime::legion_collective_stages-1))
        {
          for (int r = 1; r < Runtime::legion_collective_last_radix; r++)
          {
            AddressSpaceID target = runtime->address_space ^
              (r << (stage * Runtime::legion_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < 
                    Runtime::legion_collective_participating_spaces);
#endif
            runtime->send_mpi_rank_exchange(target, rez);
          }
        }
        else
        {
          for (int r = 1; r < Runtime::legion_collective_radix; r++)
          {
            AddressSpaceID target = runtime->address_space ^
              (r << (stage * Runtime::legion_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < 
                    Runtime::legion_collective_participating_spaces);
#endif
            runtime->send_mpi_rank_exchange(target, rez);
          }
        }
      }
      return all_stages_done;
    }

    //--------------------------------------------------------------------------
    bool MPIRankTable::send_ready_stages(void) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(participating);
#endif
      // Iterate through the stages and send any that are ready
      // Remember that stages have to be done in order
      for (int stage = 1; stage < Runtime::legion_collective_stages; stage++)
      {
        Serializer rez;
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
          if (stage_notifications[stage-1] < Runtime::legion_collective_radix)
            return false;
          // If we get here then we can send the stage
          sent_stages[stage] = true;
#ifdef DEBUG_LEGION
          {
            size_t expected_size = 1;
            for (int idx = 0; idx < stage; idx++)
              expected_size *= Runtime::legion_collective_radix;
            assert(expected_size <= forward_mapping.size());
          }
#endif
          rez.serialize<size_t>(forward_mapping.size());
          for (std::map<int,AddressSpace>::const_iterator it = 
                forward_mapping.begin(); it != forward_mapping.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
        }
        // Now we can do the send
        if (stage == (Runtime::legion_collective_stages-1))
        {
          for (int r = 1; r < Runtime::legion_collective_last_radix; r++)
          {
            AddressSpaceID target = runtime->address_space ^
              (r << (stage * Runtime::legion_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < 
                    Runtime::legion_collective_participating_spaces);
#endif
            runtime->send_mpi_rank_exchange(target, rez);
          }
        }
        else
        {
          for (int r = 1; r < Runtime::legion_collective_radix; r++)
          {
            AddressSpaceID target = runtime->address_space ^
              (r << (stage * Runtime::legion_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < 
                    Runtime::legion_collective_participating_spaces);
#endif
            runtime->send_mpi_rank_exchange(target, rez);
          }
        }
      }
      // If we make it here, then we sent the last stage, check to see
      // if we've seen all the notifications for it
      AutoLock r_lock(reservation,1,false/*exclusive*/);
      return (stage_notifications.back() == 
                Runtime::legion_collective_last_radix);
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::handle_mpi_rank_exchange(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      int stage;
      derez.deserialize(stage);
#ifdef DEBUG_LEGION
      assert(participating || (stage == -1));
#endif
      unpack_exchange(stage, derez);
      bool all_stages_done = false;
      if (stage == -1)
      {
        if (!participating)
        {
#ifdef DEBUG_LEGION
          assert(forward_mapping.size() == runtime->total_address_spaces);
#endif
          Runtime::trigger_event(done_event);
        }
        else // we can now send our stage 0
          all_stages_done = send_explicit_stage(0); 
      }
      else
        all_stages_done = send_ready_stages();
      if (all_stages_done)
        complete_exchange();
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::unpack_exchange(int stage, Deserializer &derez)
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
#ifdef DEBUG_LEGION
	// Duplicates are possible because later messages aren't "held", but
	// they should be exact matches
	assert ((forward_mapping.count(rank) == 0) ||
		(forward_mapping[rank] == space));
#endif
	forward_mapping[rank] = space;
      }
      // A stage -1 message is counted as part of stage 0 (if it exists
      //  and we are participating)
      if ((stage == -1) && participating &&
	  (Runtime::legion_collective_stages > 0))
	stage = 0;
      if (stage >= 0)
      {
#ifdef DEBUG_LEGION
	assert(stage < int(stage_notifications.size()));
        if (stage < (Runtime::legion_collective_stages-1))
          assert(stage_notifications[stage] < 
                  Runtime::legion_collective_radix);
        else
          assert(stage_notifications[stage] < 
                  Runtime::legion_collective_last_radix);
#endif
        stage_notifications[stage]++;
      }
    }

    //--------------------------------------------------------------------------
    void MPIRankTable::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(forward_mapping.size() == runtime->total_address_spaces);
#endif
      // We are done
      Runtime::trigger_event(done_event);
      // See if we have to send a message back to a
      // non-participating node
      if ((int(runtime->total_address_spaces) > 
           Runtime::legion_collective_participating_spaces) &&
          (int(runtime->address_space) < int(runtime->total_address_spaces -
            Runtime::legion_collective_participating_spaces)))
        send_explicit_stage(-1);
    }

    /////////////////////////////////////////////////////////////
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind,
                                       Runtime *rt, unsigned def_mappers,
                                       bool no_steal, bool replay)
      : runtime(rt), local_proc(proc), proc_kind(kind), 
        stealing_disabled(no_steal), replay_execution(replay), 
        next_local_index(0), task_scheduler_enabled(false), 
        total_active_contexts(0), total_active_mappers(0)
    //--------------------------------------------------------------------------
    {
      context_states.resize(DEFAULT_CONTEXTS);
      // Find our set of visible memories
      Machine::MemoryQuery vis_mems(runtime->machine);
      vis_mems.has_affinity_to(proc);
      for (Machine::MemoryQuery::iterator it = vis_mems.begin();
            it != vis_mems.end(); it++)
        visible_memories.insert(*it);
    }

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(const ProcessorManager &rhs)
      : runtime(NULL), local_proc(Processor::NO_PROC),
        proc_kind(Processor::LOC_PROC), stealing_disabled(false), 
        replay_execution(false), next_local_index(0),
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
      mapper_states.clear();
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
    void ProcessorManager::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<MapperID,std::pair<MapperManager*,bool> >::iterator it = 
            mappers.begin(); it != mappers.end(); it++)
      {
        if (it->second.second)
          delete it->second.first;
      }
      mappers.clear();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::startup_mappers(void)
    //--------------------------------------------------------------------------
    {
      // No one can be modifying the mapper set here so 
      // there is no to hold the lock
      std::multimap<Processor,MapperID> stealing_targets;
      // See what if any stealing we should perform
      for (std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
            it = mappers.begin(); it != mappers.end(); it++)
        it->second.first->perform_stealing(stealing_targets);
      if (!stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_mapper(MapperID mid, MapperManager *m, 
                                      bool check, bool own, bool skip_replay)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (!skip_replay && replay_execution)
        return;
      log_run.spew("Adding mapper %d on processor " IDFMT "", 
                          mid, local_proc.id);
      if (check && (mid == 0))
        REPORT_LEGION_ERROR(ERROR_RESERVED_MAPPING_ID, 
                            "Invalid mapping ID. ID 0 is reserved.");
#ifndef DISABLE_PARTITION_SHIM
      if (check && (mid == PARTITION_SHIM_MAPPER_ID))
        REPORT_LEGION_ERROR(ERROR_RESERVED_MAPPING_ID,
                            "Invalid mapper ID. %d is reserved for the "
                            "partition shim mapper", PARTITION_SHIM_MAPPER_ID)
#endif
      AutoLock m_lock(mapper_lock);
      std::map<MapperID,std::pair<MapperManager*,bool> >::iterator finder = 
        mappers.find(mid);
      if (finder != mappers.end())
      {
        if (finder->second.second)
          delete finder->second.first;
        finder->second = std::pair<MapperManager*,bool>(m, own);
      }
      else
      {
        mappers[mid] = std::pair<MapperManager*,bool>(m, own); 
        AutoLock q_lock(queue_lock);
        mapper_states[mid] = MapperState();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::replace_default_mapper(MapperManager *m, bool own)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (replay_execution)
        return;
      AutoLock m_lock(mapper_lock);
      std::map<MapperID,std::pair<MapperManager*,bool> >::iterator finder = 
        mappers.find(0);
#ifdef DEBUG_LEGION
      assert(finder != mappers.end());
#endif
      if (finder->second.second)
        delete finder->second.first;
      finder->second = std::pair<MapperManager*,bool>(m, own);
    }

    //--------------------------------------------------------------------------
    MapperManager* ProcessorManager::find_mapper(MapperID mid) const 
    //--------------------------------------------------------------------------
    {
      // Easy case if we are doing replay execution
      if (replay_execution)
      {
        std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
          finder = mappers.find(0);
#ifdef DEBUG_LEGION
        assert(finder != mappers.end());
#endif
        return finder->second.first;
      }
      AutoLock m_lock(mapper_lock, 0/*mode*/, false/*exclusive*/);
      MapperManager *result = NULL;
      // We've got the lock, so do the operation
      std::map<MapperID,std::pair<MapperManager*,bool> >::const_iterator
        finder = mappers.find(mid);
      if (finder != mappers.end())
        result = finder->second.first;
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------
    {
      perform_mapping_operations(); 
      // Now re-take the lock and re-check the condition to see 
      // if the next scheduling task should be launched
      AutoLock q_lock(queue_lock);
      if ((total_active_contexts > 0) && (total_active_mappers > 0))
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
      sched_args.proc = local_proc;
      runtime->issue_runtime_meta_task(sched_args, LG_LATENCY_WORK_PRIORITY);
    } 

    //--------------------------------------------------------------------------
    void ProcessorManager::notify_deferred_mapper(MapperID map_id,
                                                  RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      MapperState &state = mapper_states[map_id];
      // Check to see if the deferral event matches the one that we have
      if (state.deferral_event == deferred_event)
      {
        // Now we can clear it
        state.deferral_event = RtEvent::NO_RT_EVENT;
        // And if we still have tasks, reactivate the mapper
        if (!state.ready_queue.empty())
          increment_active_mappers();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProcessorManager::handle_defer_mapper(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMapperSchedulerArgs *dargs = 
        (const DeferMapperSchedulerArgs*)args; 
      dargs->proxy_this->notify_deferred_mapper(dargs->map_id, 
                                                dargs->deferral_event);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::activate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_context_id();
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
      assert(!state.active);
#endif
      state.active = true;
      if (state.owned_tasks > 0)
        increment_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::deactivate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_context_id();
      // We can do this without holding the lock because we know
      // the size of this vector is fixed
      AutoLock q_lock(queue_lock); 
      ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
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
      if (!task_scheduler_enabled && (total_active_contexts == 0) &&
          (total_active_mappers > 0))
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
#ifdef DEBUG_LEGION
      assert(total_active_contexts > 0);
#endif
      total_active_contexts--;
      if (total_active_contexts == 0)
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_mappers == 0) &&
          (total_active_contexts > 0))
      {
        task_scheduler_enabled = true;
        launch_task_scheduler();
      }
      total_active_mappers++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
#ifdef DEBUG_LEGION
      assert(total_active_mappers > 0);
#endif
      total_active_mappers--;
      if (total_active_mappers == 0)
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
      std::vector<MapperID> successful_thiefs;
      for (std::vector<MapperID>::const_iterator steal_it = thieves.begin();
            steal_it != thieves.end(); steal_it++)
      {
        const MapperID stealer = *steal_it;
        // Handle a race condition here where some processors can 
        // issue steal requests to another processor before the mappers 
        // have been initialized on that processor.  There's no 
        // correctness problem for ignoring a steal request so just do that.
        MapperManager *mapper = find_mapper(stealer);
        if (mapper == NULL)
          continue;
        
        // Construct a vector of tasks eligible for stealing
        Mapper::StealRequestInput input;
        input.thief_proc = thief;
        std::vector<const Task*> &mapper_tasks = input.stealable_tasks;
        {
          AutoLock q_lock(queue_lock,1,false/*exclusive*/);
          std::list<TaskOp*> &target_list = mapper_states[stealer].ready_queue;
          for (std::list<TaskOp*>::const_iterator it = 
                target_list.begin(); it != target_list.end(); it++)
          {
            if ((*it)->is_stealable() && !(*it)->is_origin_mapped())
              mapper_tasks.push_back(*it);
          }
        }
        Mapper::StealRequestOutput output;
        // Ask the mapper what it wants to allow be stolen
        if (!mapper_tasks.empty())
          mapper->invoke_permit_steal_request(&input, &output);
        std::deque<TaskOp*> temp_stolen;
        const std::set<const Task*> &to_steal = output.stolen_tasks;
        if (!to_steal.empty())
        {
          // See if we can still get it out of the queue
          AutoLock q_lock(queue_lock);
          MapperState &map_state = mapper_states[stealer];
          std::list<TaskOp*> &target_list = map_state.ready_queue;
          for (std::set<const Task*>::const_iterator steal_it = 
                to_steal.begin(); steal_it != to_steal.end(); steal_it++)
          {
            TaskOp *target = static_cast<TaskOp*>(
                              const_cast<Task*>(*steal_it));
            bool found = false;
            for (std::list<TaskOp*>::iterator it = target_list.begin();
                  it != target_list.end(); it++)
            {
              if ((*it) == target)
              {
                target_list.erase(it);
                if (target_list.empty())
                {
                  // If we already have a deferral event then we
                  // are already not active so just clear it
                  if (map_state.deferral_event.exists())
                    map_state.deferral_event = RtEvent::NO_RT_EVENT;
                  else
                    decrement_active_mappers();
                }
                found = true;
                break;
              }
            }
            if (found)
            {
              temp_stolen.push_back(target);
              // Wait until we are no longer holding the lock
              // to mark that this is no longer an outstanding task
              ContextID ctx_id = target->get_context()->get_context_id();
              ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
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
            temp_stolen[idx]->mark_stolen();
            stolen.insert(temp_stolen[idx]);
            successful_steal = true;
            temp_stolen[idx]->deactivate_outstanding_task();
          }
          else
          {
            // Always set this before putting anything on
            // the ready queue
            ContextID ctx_id = 
              temp_stolen[idx]->get_context()->get_context_id();
            AutoLock q_lock(queue_lock);
            ContextState &state = context_states[ctx_id];
            if (state.active && (state.owned_tasks == 0))
              increment_active_contexts();
            state.owned_tasks++;
            MapperState &map_state = mapper_states[stealer];
            if (map_state.ready_queue.empty())
            {
              // Clear any deferral event if we are changing state
              map_state.deferral_event = RtEvent::NO_RT_EVENT;
              increment_active_mappers();
            }
            // Otherwise the state hasn't actually changed so
            // we can't actually clear any deferral events
            map_state.ready_queue.push_back(temp_stolen[idx]);
            map_state.added_tasks = true;
          }
        }

        if (!successful_steal) 
          mapper->process_failed_steal(thief);
        else
          successful_thiefs.push_back(stealer);
      }
      if (!stolen.empty())
      {
#ifdef DEBUG_LEGION
        for (std::set<TaskOp*>::const_iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
          log_task.debug("task %s (ID %lld) stolen from processor " IDFMT
                         " by processor " IDFMT "", (*it)->get_task_name(), 
                         (*it)->get_unique_id(), local_proc.id, thief.id);
        }
#endif
        runtime->send_tasks(thief, stolen);
        // Also have to send advertisements to the mappers that 
        // successfully stole so they know that they can try again
        std::set<Processor> thief_set;
        thief_set.insert(thief);
        for (std::vector<MapperID>::const_iterator it = 
              successful_thiefs.begin(); it != successful_thiefs.end(); it++)
          runtime->send_advertisements(thief_set, *it, local_proc);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_advertisement(Processor advertiser,
                                                 MapperID mid)
    //--------------------------------------------------------------------------
    {
      MapperManager *mapper = find_mapper(mid);
      mapper->process_advertisement(advertiser);
      // See if this mapper would like to try stealing again
      std::multimap<Processor,MapperID> stealing_targets;
      mapper->perform_stealing(stealing_targets);
      if (!stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_ready_queue(TaskOp *task)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task != NULL);
#endif
      // have to do this when we are not holding the lock
      task->activate_outstanding_task();
      // We can do this without holding the lock because the
      // vector is of a fixed size
      ContextID ctx_id = task->get_context()->get_context_id();
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_LEGION
      assert(mapper_states.find(task->map_id) != mapper_states.end());
#endif
      // Update the state for the context
      ContextState &state = context_states[ctx_id];
      if (state.active && (state.owned_tasks == 0))
        increment_active_contexts();
      state.owned_tasks++;
      // Also update the queue for the mapper
      MapperState &map_state = mapper_states[task->map_id];
      if (map_state.ready_queue.empty() || map_state.deferral_event.exists())
      {
        // Clear our deferral event since we are changing state
        map_state.deferral_event = RtEvent::NO_RT_EVENT;
        increment_active_mappers();
      }
      map_state.ready_queue.push_back(task);
      map_state.added_tasks = true;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_local_ready_queue(Operation *op, 
                                           LgPriority priority, RtEvent wait_on) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op != NULL);
#endif
      TriggerOpArgs args;
      args.op = op;
      runtime->issue_runtime_meta_task(args, priority, op, wait_on); 
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_mapping_operations(void)
    //--------------------------------------------------------------------------
    {
      std::multimap<Processor,MapperID> stealing_targets;
      std::vector<MapperID> mappers_with_stealable_work;
      std::vector<std::pair<MapperID,MapperManager*> > current_mappers;
      // Take a snapshot of our current mappers
      {
        AutoLock m_lock(mapper_lock,1,false/*exclusive*/);
        // Fast path for no deferred mappers
        current_mappers.resize(mappers.size());
        unsigned idx = 0;
        for (std::map<MapperID,std::pair<MapperManager*,bool> >::
              const_iterator it = mappers.begin(); it != 
              mappers.end(); it++, idx++)
          current_mappers[idx] = 
            std::pair<MapperID,MapperManager*>(it->first, it->second.first);
      }
      for (std::vector<std::pair<MapperID,MapperManager*> >::const_iterator
            it = current_mappers.begin(); it != current_mappers.end(); it++)
      {
        const MapperID map_id = it->first;
        MapperManager *const mapper = it->second;
        Mapper::SelectMappingInput input;
        std::list<const Task*> &visible_tasks = input.ready_tasks;
        // We also need to capture the generations here
        std::list<GenerationID> visible_generations;
        // Pull out the current tasks for this mapping operation
        {
          AutoLock q_lock(queue_lock,1,false/*exclusive*/);
          MapperState &map_state = mapper_states[map_id];
          // Only grap tasks if we don't have a deferral event
          if (!map_state.deferral_event.exists())
          {
            std::list<TaskOp*> &target_list = map_state.ready_queue;
            for (std::list<TaskOp*>::const_iterator it = 
                  target_list.begin(); it != target_list.end(); it++)
            {
              visible_tasks.push_back(*it);
              visible_generations.push_back((*it)->get_generation());
            }  
          }
          // Set our flag to check for added tasks while we do the mapper call
          map_state.added_tasks = false;
        }
        // Do this before anything else in case we don't have any tasks
        if (!stealing_disabled)
          mapper->perform_stealing(stealing_targets);
        // Nothing to do if there are no visible tasks
        if (visible_tasks.empty())
          continue;
        // Ask the mapper which tasks it would like to schedule
        Mapper::SelectMappingOutput output;
        if (!visible_tasks.empty())
          mapper->invoke_select_tasks_to_map(&input, &output);
        // If we had no entry then we better have gotten a mapper event
        if (output.map_tasks.empty() && output.relocate_tasks.empty())
        {
          const RtEvent wait_on = output.deferral_event.impl;
          if (wait_on.exists())
          {
            // Put this on the list of the deferred mappers
            AutoLock q_lock(queue_lock);
            MapperState &map_state = mapper_states[map_id];
            // We have to check to see if any new tasks were added to 
            // the ready queue while we were doing our mapper call, if 
            // they were then we need to invoke select_tasks_to_map again
            if (!map_state.added_tasks)
            {
#ifdef DEBUG_LEGION
              assert(!map_state.deferral_event.exists());
#endif
              map_state.deferral_event = wait_on;
              // Decrement the number of active mappers
              decrement_active_mappers();
            }
          }
          else // Very bad, error message
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Mapper %s failed to specify an output MapperEvent "
                          "when returning from a call to 'select_tasks_to_map' "
                          "that performed no other actions. Specifying a "
                          "MapperEvent in such situation is necessary to avoid "
                          "livelock conditions. Please return a "
                          "'deferral_event' in the 'output' struct.",
                          mapper->get_mapper_name())
          // Launch a task to remove the deferred mapper event when it triggers
          DeferMapperSchedulerArgs args;
          args.proxy_this = this;
          args.map_id = map_id;
          args.deferral_event = wait_on;
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                           NULL, wait_on);
          // We can continue because there is nothing left to do for this mapper
          continue;
        }
        // Process the results first remove the operations that were
        // selected to be mapped from the queue.  Note its possible
        // that we can't actually find the task because it has been
        // stolen from the queue while we were deciding what to
        // map.  It's also possible the task is no longer in the same
        // place if the queue was prepended to.
        {
          // First things first: filter our visible list before we are holding
          // the lock on the ready queues
          std::list<GenerationID>::iterator gen_it = 
            visible_generations.begin();
          for (std::list<const Task*>::iterator vis_it = visible_tasks.begin();
                vis_it != visible_tasks.end(); /*nothing*/)
          {
            // Check to see if this is a task that we care about
            if ((output.map_tasks.find(*vis_it) != output.map_tasks.end()) ||
                (output.relocate_tasks.find(*vis_it) != 
                 output.relocate_tasks.end()))
            {
              vis_it++; // still care about it
              gen_it++;
            }
            else // Otherwise we don't care so we can erase it
            {
              vis_it = visible_tasks.erase(vis_it);
              gen_it = visible_generations.erase(gen_it);
            }
          }
          // Reset the iterator to the start
          gen_it = visible_generations.begin();
          AutoLock q_lock(queue_lock);
          MapperState &map_state = mapper_states[map_id];
          std::list<TaskOp*> &rqueue = map_state.ready_queue;
          for (std::list<const Task*>::iterator vis_it = visible_tasks.begin(); 
                vis_it != visible_tasks.end(); gen_it++)
          {
            bool found = false;
            for (std::list<TaskOp*>::iterator it = rqueue.begin();
                  it != rqueue.end(); it++)
            {
              // In order to be the same task, they need to have the
              // same pointer and have the same generation
              if (((*it) == (*vis_it)) &&
                  ((*gen_it) == (*it)->get_generation()))
              {
                rqueue.erase(it);
                if (rqueue.empty())
                {
                  if (map_state.deferral_event.exists())
                    map_state.deferral_event = RtEvent::NO_RT_EVENT;
                  else
                    decrement_active_mappers();
                }
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
              TaskOp *task = static_cast<TaskOp*>(const_cast<Task*>(*vis_it));
              ContextID ctx_id = task->get_context()->get_context_id(); 
              ContextState &state = context_states[ctx_id];
#ifdef DEBUG_LEGION
              assert(state.owned_tasks > 0);
#endif
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
              vis_it++;
            }
          }
          if (!stealing_disabled)
          {
            for (std::list<TaskOp*>::const_iterator it =
                  rqueue.begin(); it != rqueue.end(); it++)
            {
              if ((*it)->is_stealable())
              {
                mappers_with_stealable_work.push_back(map_id);
                break;
              }
            }
          }
        }
        // Now that we've removed them from the queue, issue the
        // mapping analysis calls
        TriggerTaskArgs trigger_args;
        for (std::list<const Task*>::iterator vis_it = visible_tasks.begin();
              vis_it != visible_tasks.end(); vis_it++)
        {
          TaskOp *task = static_cast<TaskOp*>(const_cast<Task*>(*vis_it));
          // Update the target processor for this task if necessary
          std::map<const Task*,Processor>::const_iterator finder = 
            output.relocate_tasks.find(*vis_it);
          const bool send_remotely = (finder != output.relocate_tasks.end());
          if (send_remotely)
            task->set_target_proc(finder->second);
          // Mark that this task is no longer outstanding
          task->deactivate_outstanding_task();
          trigger_args.op = task;
          runtime->issue_runtime_meta_task(trigger_args,
                                           LG_THROUGHPUT_WORK_PRIORITY, task);
        }
      }

      // Advertise any work that we have
      if (!stealing_disabled && !mappers_with_stealable_work.empty())
      {
        for (std::vector<MapperID>::const_iterator it = 
              mappers_with_stealable_work.begin(); it !=
              mappers_with_stealable_work.end(); it++)
          issue_advertisements(*it);
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
      MapperManager *mapper = find_mapper(map_id);
      mapper->perform_advertisements(failed_waiters);
      if (!failed_waiters.empty())
        runtime->send_advertisements(failed_waiters, map_id, local_proc);
    }

    /////////////////////////////////////////////////////////////
    // Memory Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MemoryManager::MemoryManager(Memory m, Runtime *rt)
      : memory(m), owner_space(m.address_space()), 
        is_owner(m.address_space() == rt->address_space),
        capacity(m.capacity()), remaining_capacity(capacity), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MemoryManager::MemoryManager(const MemoryManager &rhs)
      : memory(Memory::NO_MEMORY), owner_space(0), 
        is_owner(false), capacity(0), runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);   
    }

    //--------------------------------------------------------------------------
    MemoryManager::~MemoryManager(void)
    //--------------------------------------------------------------------------
    {
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
    void MemoryManager::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      // Only need to do things if we are the owner memory
      if (is_owner)
      {
        std::vector<PhysicalManager*> instances;
        {
          AutoLock m_lock(manager_lock,1,false/*exclusive*/);
          for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
                current_instances.begin(); it != current_instances.end(); it++)
          {
            it->first->add_base_resource_ref(MEMORY_MANAGER_REF);   
            instances.push_back(it->first);
          }
        }
        std::set<RtEvent> wait_for;
        for (std::vector<PhysicalManager*>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
        {
          if ((*it)->try_active_deletion())
          {
            record_deleted_instance(*it);
            // Remove our base resource reference
            if ((*it)->remove_base_resource_ref(MEMORY_MANAGER_REF))
              delete (*it);
          }
        }
        if (!wait_for.empty())
        {
          RtEvent wait_on = Runtime::merge_events(wait_for);
          wait_on.lg_wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::finalize(void)
    //--------------------------------------------------------------------------
    {
      if (!is_owner)
        return;
      // No need for the lock, no one should be doing anything at this point
      for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
            current_instances.begin(); it != current_instances.end(); it++)
        it->first->force_deletion();
    }
    
    //--------------------------------------------------------------------------
    void MemoryManager::register_remote_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      const size_t inst_size = manager->get_instance_size();
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(current_instances.find(manager) == current_instances.end());
#endif
      // Make it valid to start since we know when we were created
      // that we were made valid to begin with
      InstanceInfo &info = current_instances[manager];
      info.instance_size = inst_size;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::unregister_remote_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
 #ifdef DEBUG_LEGION
      assert(current_instances.find(manager) != current_instances.end());
#endif     
      current_instances.erase(manager);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::activate_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
        current_instances.find(manager);
#ifdef DEBUG_LEGION
      assert(current_instances.find(manager) != current_instances.end());
      if (finder->second.current_state != VALID_STATE)
        assert(finder->second.current_state == COLLECTABLE_STATE);
#endif
      if (finder->second.current_state != VALID_STATE)
        finder->second.current_state = ACTIVE_STATE;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::deactivate_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      bool remove_reference = false;
      {
        AutoLock m_lock(manager_lock);
        std::map<PhysicalManager*,InstanceInfo>::iterator finder =
          current_instances.find(manager);
#ifdef DEBUG_LEGION
        assert(finder != current_instances.end());
        assert((finder->second.current_state == ACTIVE_STATE) ||
               (finder->second.current_state == ACTIVE_COLLECTED_STATE));
#endif
        InstanceInfo &info = finder->second;
        // See if we deleted this yet
        if (finder->second.current_state == ACTIVE_COLLECTED_STATE)
        {
          // already deferred collected this, so we can trigger 
          // the deletion now this should only happen on the owner node
#ifdef DEBUG_LEGION
          assert(is_owner);
          assert(info.deferred_collect.exists());
#endif
          Runtime::trigger_event(info.deferred_collect);
          // Now we can delete our entry because it has been deleted
          current_instances.erase(finder);
          if (is_owner)
            remove_reference = true;
        }
        else // didn't collect it yet
          info.current_state = COLLECTABLE_STATE;
      }
      // If we are the owner and this is a reduction instance
      // then let's just delete it now
      if (remove_reference)
      {
        if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete manager;
      }
      else if (is_owner && manager->is_reduction_manager())
      { 
        if (manager->try_active_deletion())
          record_deleted_instance(manager);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::validate_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
        current_instances.find(manager);
#ifdef DEBUG_LEGION
      assert(finder != current_instances.end());
      if (finder->second.current_state != VALID_STATE)
        assert(finder->second.current_state == ACTIVE_STATE);
#endif
      finder->second.current_state = VALID_STATE;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::invalidate_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
        current_instances.find(manager);
#ifdef DEBUG_LEGION
      assert(finder != current_instances.end());
      assert(finder->second.current_state == VALID_STATE);
#endif
      finder->second.current_state = ACTIVE_STATE;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::create_physical_instance(
                                const LayoutConstraintSet &constraints,
                                const std::vector<LogicalRegion> &regions,
                                MappingInstance &result, MapperID mapper_id, 
                                Processor processor, bool acquire, 
                                GCPriority priority, UniqueID creator_id,
                                bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the creation
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(CREATE_INSTANCE_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          constraints.serialize(rez);
          rez.serialize(mapper_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize(creator_id);
          rez.serialize(&success);
          rez.serialize(&result);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled in
      }
      else
      {
        // Try to make the result
        PhysicalManager *manager = allocate_physical_instance(constraints, 
                                                     regions, creator_id);
        if (manager != NULL)
        {
          if (Runtime::legion_spy_enabled)
            manager->log_instance_creation(creator_id, processor, regions);
          record_created_instance(manager, acquire, mapper_id, processor,
                                  priority, remote);
          result = MappingInstance(manager);
          success = true;
        }
      }
      return success;
    }
    
    //--------------------------------------------------------------------------
    bool MemoryManager::create_physical_instance(LayoutConstraints *constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result,MapperID mapper_id,
                                     Processor processor, bool acquire, 
                                     GCPriority priority, UniqueID creator_id,
                                     bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      if (!is_owner)
      {
        // Not the owner, send a meessage to the owner to request the creation
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(CREATE_INSTANCE_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          rez.serialize(constraints->layout_id);
          rez.serialize(mapper_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize(creator_id);
          rez.serialize(&success);
          rez.serialize(&result);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled in
      }
      else
      {
        // Try to make the instance
        PhysicalManager *manager = allocate_physical_instance(*constraints,
                                                     regions, creator_id);
        if (manager != NULL)
        {
          if (Runtime::legion_spy_enabled)
            manager->log_instance_creation(creator_id, processor, regions);
          record_created_instance(manager, acquire, mapper_id, processor,
                                  priority, remote);
          result = MappingInstance(manager);
          success = true;
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_or_create_physical_instance(
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  MappingInstance &result, bool &created, 
                                  MapperID mapper_id, Processor processor,
                                  bool acquire, GCPriority priority,
                                  bool tight_region_bounds,
                                  UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      // Set created to default to false
      created = false;
      if (!is_owner)
      {
        // See if we can find a locally valid instance first
        success = find_valid_instance(constraints, regions, result, 
                                      acquire, tight_region_bounds, remote);
        if (success)
          return true;
        // Not the owner, send a message to the owner to request creation
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_OR_CREATE_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          constraints.serialize(rez);
          rez.serialize(mapper_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(creator_id);
          rez.serialize(&success);
          rez.serialize(&result);
          rez.serialize(&created);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled in
      }
      else
      {
        // Try to find an instance first and then make one
        std::set<PhysicalManager*> candidates;
        success = find_satisfying_instance(constraints, regions, 
                   result, candidates, acquire, tight_region_bounds, remote);
        if (!success)
        {
          // If we couldn't find it, we have to make it
          PhysicalManager *manager = allocate_physical_instance(constraints, 
                                                       regions, creator_id);
          if (manager != NULL)
          {
            if (Runtime::legion_spy_enabled)
              manager->log_instance_creation(creator_id, processor, regions);
            // We're definitely going to succeed one way or another
            success = true;
            // To maintain the illusion that this is atomic we have to
            // check again to see if anything else has been registered
            // which might also satisfy the constraints
            PhysicalManager *actual_manager = 
              find_and_record(manager, constraints, regions, candidates,
                              acquire, mapper_id, processor, priority, 
                              tight_region_bounds, remote);
            // If they are still the same then we succeeded
            if (actual_manager == manager)
              created = true;
            // Save the final result
            result = MappingInstance(actual_manager);
          }
        }
        release_candidate_references(candidates);
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_or_create_physical_instance(
                                LayoutConstraints *constraints, 
                                const std::vector<LogicalRegion> &regions,
                                MappingInstance &result, bool &created,
                                MapperID mapper_id, Processor processor,
                                bool acquire, GCPriority priority, 
                                bool tight_region_bounds,
                                UniqueID creator_id, bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      // Set created to false in case we fail
      created = false;
      if (!is_owner)
      {
        // See if we can find it locally
        success = find_valid_instance(constraints, regions, result, 
                                      acquire, tight_region_bounds, remote);
        if (success)
          return true;
        // Not the owner, send a message to the owner to request creation
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_OR_CREATE_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          rez.serialize(constraints->layout_id);
          rez.serialize(mapper_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(creator_id);
          rez.serialize(&success);
          rez.serialize(&result);
          rez.serialize(&created);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled
      }
      else
      {
        // Try to find an instance first and then make one
        std::set<PhysicalManager*> candidates;
        success = find_satisfying_instance(constraints, regions, 
                   result, candidates, acquire, tight_region_bounds, remote);
        if (!success)
        {
          // If we couldn't find it, we have to make it
          PhysicalManager *manager = allocate_physical_instance(*constraints,
                                                       regions, creator_id);
          if (manager != NULL)
          {
            if (Runtime::legion_spy_enabled)
              manager->log_instance_creation(creator_id, processor, regions);
            // If we make it here we're definitely going to succeed
            success = true;
            // To maintain the illusion that this is atomic we have to
            // check again to see if anything else has been registered
            // which might also satisfy the constraints
            PhysicalManager *actual_manager = 
              find_and_record(manager, constraints, regions, candidates,
                              acquire, mapper_id, processor, priority, 
                              tight_region_bounds, remote);
            // If they are still the same then we succeeded
            if (actual_manager == manager)
              created = true;
            result = MappingInstance(actual_manager);
          }
        }
        release_candidate_references(candidates);
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_physical_instance(
                                     const LayoutConstraintSet &constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result, bool acquire, 
                                     bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      if (!is_owner)
      {
        // See if we can find it locally 
        success = find_valid_instance(constraints, regions, result, 
                                      acquire, tight_region_bounds, remote);
        if (success)
          return true;
        // Not the owner, send a message to the owner to try and find it
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_ONLY_CONSTRAINTS);
          rez.serialize(ready_event);
          rez.serialize(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          constraints.serialize(rez);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&success);
          rez.serialize(&result);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled
      }
      else
      {
        // Try to find an instance
        success = find_satisfying_instance(constraints, regions, result, 
                                  acquire, tight_region_bounds, remote);
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_physical_instance(LayoutConstraints *constraints,
                                      const std::vector<LogicalRegion> &regions,
                                      MappingInstance &result, bool acquire, 
                                      bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      volatile bool success = false;
      if (!is_owner)
      {
        // See if we can find a persistent instance
        success = find_valid_instance(constraints, regions, result, 
                                      acquire, tight_region_bounds, remote);
        if (success)
          return true;
        Serializer rez;
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(FIND_ONLY_LAYOUT);
          rez.serialize(ready_event);
          rez.serialize<size_t>(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize(regions[idx]);
          rez.serialize<bool>(acquire);
          rez.serialize(constraints->layout_id);
          rez.serialize<bool>(tight_region_bounds);
          rez.serialize(&success);
          rez.serialize(&result);
        }
        runtime->send_instance_request(owner_space, rez);
        ready_event.lg_wait();
        // When the event is triggered, everything will be filled
      }
      else
      {
        // Try to find an instance
        success = find_satisfying_instance(constraints, regions, result,
                                   acquire, tight_region_bounds, remote);
      }
      return success;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_tree_instances(RegionTreeID tree_id)
    //--------------------------------------------------------------------------
    {
      // If we're not the owner, then there is nothing to do
      if (!is_owner)
        return;
      // Take the manager lock and see if there are any managers
      // we can release now
      std::map<PhysicalManager*,bool> to_release;
      {
        AutoLock m_lock(manager_lock);
        for (std::map<PhysicalManager*,InstanceInfo>::iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // If the region for the instance is not for the tree then
          // we get to skip it
          if (it->first->region_node->handle.get_tree_id() != tree_id)
            continue;
          // If it's already been deleted, then there is nothing to do
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          // Add a resource reference for ourself
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          to_release[it->first] = 
            (it->second.min_priority == GC_NEVER_PRIORITY);
#ifdef DEBUG_LEGION
          // We might have lost a race with adding NEVER_GC_REF
          // after release the manager lock if we hit this assertion
          if (it->second.min_priority == GC_NEVER_PRIORITY)
            assert(it->second.current_state == VALID_STATE);
#endif
          it->second.mapper_priorities.clear();
          it->second.min_priority = GC_MAX_PRIORITY;
        }
      }
      for (std::map<PhysicalManager*,bool>::const_iterator it = 
            to_release.begin(); it != to_release.end(); it++)
      {
        if (it->second)
          it->first->remove_base_valid_ref(NEVER_GC_REF);
        if (it->first->try_active_deletion())
          record_deleted_instance(it->first);
        // Now we can release our resource reference
        if (it->first->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete (it->first);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::set_garbage_collection_priority(
                                PhysicalManager *manager, MapperID mapper_id, 
                                Processor processor, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      bool remove_min_reference = false;
      IgnoreReferenceMutator mutator;
      if (!is_owner)
      {
        RtUserEvent never_gc_wait;
        bool remove_never_gc_ref = false;
        std::pair<MapperID,Processor> key(mapper_id,processor);
        // Check to see if this is or is going to be a max priority instance
        if (priority == GC_NEVER_PRIORITY)
        {
          // See if we need a handback
          AutoLock m_lock(manager_lock,1,false);
          std::map<PhysicalManager*,InstanceInfo>::const_iterator finder = 
            current_instances.find(manager);
          if (finder != current_instances.end())
          {
            // If priority is already max priority, then we are done
            if (finder->second.min_priority == priority)
              return;
            // Make an event for a callback
            never_gc_wait = Runtime::create_rt_user_event();
          }
        }
        else
        {
          AutoLock m_lock(manager_lock);
          std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
            current_instances.find(manager);
          if (finder != current_instances.end())
          {
            if (finder->second.min_priority == GC_NEVER_PRIORITY)
            {
              finder->second.mapper_priorities.erase(key);
              if (finder->second.mapper_priorities.empty())
              {
                finder->second.min_priority = 0;
                remove_never_gc_ref = true;
              }
            }
          }
        }
        // Won't delete the whole manager because we still hold
        // a resource reference
        if (remove_never_gc_ref)
          manager->remove_base_valid_ref(NEVER_GC_REF);
        // We are not the owner so send a message to the owner
        // to update the priority, no need to send the manager
        // since we know we are sending to the owner node
        volatile bool success = true;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(manager->did);
          rez.serialize(mapper_id);
          rez.serialize(processor);
          rez.serialize(priority);
          rez.serialize(never_gc_wait);
          if (never_gc_wait.exists())
            rez.serialize(&success);
        }
        runtime->send_gc_priority_update(owner_space, rez);
        // In most cases, we will fire and forget, the one exception
        // is if we are waiting for a confirmation of setting max priority
        if (never_gc_wait.exists())
        {
          never_gc_wait.lg_wait();
          bool remove_duplicate = false;
          if (success)
          {
            // Add our local reference
            manager->add_base_valid_ref(NEVER_GC_REF, &mutator);
            manager->send_remote_valid_update(owner_space,NULL,1,false/*add*/);
            // Then record it
            AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
            assert(current_instances.find(manager) != current_instances.end());
#endif
            InstanceInfo &info = current_instances[manager];
            if (info.min_priority == GC_NEVER_PRIORITY)
              remove_duplicate = true; // lost the race
            else
              info.min_priority = GC_NEVER_PRIORITY;
            info.mapper_priorities[key] = GC_NEVER_PRIORITY;
          }
          if (remove_duplicate && 
              manager->remove_base_valid_ref(NEVER_GC_REF, &mutator))
            delete manager; 
        }
      }
      else
      {
        // If this a max priority, try adding the reference beforehand, if
        // it fails then we know the instance is already deleted so whatever
        if ((priority == GC_NEVER_PRIORITY) &&
            !manager->try_add_base_valid_ref(NEVER_GC_REF, &mutator,
                                             false/*must be valid*/))
          return;
        // Do the update locally 
        AutoLock m_lock(manager_lock);
        std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
          current_instances.find(manager);
        if (finder != current_instances.end())
        {
          std::map<std::pair<MapperID,Processor>,GCPriority> 
            &mapper_priorities = finder->second.mapper_priorities;
          std::pair<MapperID,Processor> key(mapper_id,processor);
          // If the new priority is NEVER_GC and we were already at NEVER_GC
          // then we need to remove the redundant reference when we are done
          if ((priority == GC_NEVER_PRIORITY) && 
              (finder->second.min_priority == GC_NEVER_PRIORITY))
            remove_min_reference = true;
          // See if we can find the current priority  
          std::map<std::pair<MapperID,Processor>,GCPriority>::iterator 
            priority_finder = mapper_priorities.find(key);
          if (priority_finder != mapper_priorities.end())
          {
            // See if it changed
            if (priority_finder->second != priority)
            {
              // Update the min if necessary
              if (priority < finder->second.min_priority)
              {
                // It decreased 
                finder->second.min_priority = priority;
              }
              // It might go up if this was (one of) the min priorities
              else if ((priority > finder->second.min_priority) &&
                       (finder->second.min_priority == priority_finder->second))
              {
                // This was (one of) the min priorities, but it 
                // is about to go up so compute the new min
                GCPriority new_min = priority;
                for (std::map<std::pair<MapperID,Processor>,GCPriority>::
                      const_iterator it = mapper_priorities.begin(); it != 
                      mapper_priorities.end(); it++)
                {
                  if (it->first == key)
                    continue;
                  // If we find another one with the same as the current 
                  // min then we know we are just going to stay the same
                  if (it->second == finder->second.min_priority)
                  {
                    new_min = it->second;
                    break;
                  }
                  if (it->second < new_min)
                    new_min = it->second;
                }
                if ((finder->second.min_priority == GC_NEVER_PRIORITY) &&
                    (new_min > GC_NEVER_PRIORITY))
                  remove_min_reference = true;
                finder->second.min_priority = new_min;
              }
              // Finally update the priority
              priority_finder->second = priority;
            }
          }
          else // previous priority was zero, see if we need to update it
          {
            mapper_priorities[key] = priority;
            if (priority < finder->second.min_priority)
              finder->second.min_priority = priority;
          }
        }
      }
      if (remove_min_reference && 
          manager->remove_base_valid_ref(NEVER_GC_REF, &mutator))
        delete manager;
    }

    //--------------------------------------------------------------------------
    RtEvent MemoryManager::acquire_instances(
                                     const std::set<PhysicalManager*> &managers,
                                     std::vector<bool> &results)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner); // should never be called on the owner
#endif
      results.resize(managers.size(), true/*all good*/);
      // Package everything up and send the request 
      RtUserEvent done = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(memory);
        rez.serialize<size_t>(managers.size());
        for (std::set<PhysicalManager*>::const_iterator it = 
              managers.begin(); it != managers.end(); it++)
          rez.serialize((*it)->did);
        rez.serialize(&results);
        rez.serialize(done);
      }
      runtime->send_acquire_request(owner_space, rez);
      return done;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_instance_request(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
#endif
      RequestKind kind;
      derez.deserialize(kind);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<LogicalRegion> regions(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
        derez.deserialize(regions[idx]);
      bool acquire;
      derez.deserialize(acquire);
      switch (kind)
      {
        case CREATE_INSTANCE_CONSTRAINTS:
          {
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            MapperID mapper_id;
            derez.deserialize(mapper_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            bool *remote_success;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            MappingInstance result;
            bool success = create_physical_instance(constraints, regions, 
                                   result, mapper_id, processor, acquire, 
                                   priority, creator_id, true/*remote*/);
            if (success)
            {
              // Send back the response starting with the instance
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
                bool min_priority = (priority == GC_NEVER_PRIORITY);
                rez.serialize<bool>(min_priority);
                if (min_priority)
                {
                  rez.serialize(mapper_id);
                  rez.serialize(processor);
                }
              }
              runtime->send_instance_response(source, rez);
            }
            else // we can just trigger the done event since we failed
              Runtime::trigger_event(to_trigger);
            break;
          }
        case CREATE_INSTANCE_LAYOUT:
          {
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            MapperID mapper_id;
            derez.deserialize(mapper_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            bool *remote_success;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            LayoutConstraints *constraints = 
              runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            bool success = create_physical_instance(constraints, regions, 
                                   result, mapper_id, processor, acquire, 
                                   priority, creator_id, true/*remote*/);
            if (success)
            {
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
                bool min_priority = (priority == GC_NEVER_PRIORITY);
                rez.serialize<bool>(min_priority);
                if (min_priority)
                {
                  rez.serialize(mapper_id);
                  rez.serialize(processor);
                }
              }
              runtime->send_instance_response(source, rez);
            }
            else // if we failed, we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_OR_CREATE_CONSTRAINTS:
          {
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            MapperID mapper_id;
            derez.deserialize(mapper_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            bool *remote_success, *remote_created;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            derez.deserialize(remote_created);
            MappingInstance result;
            bool created;
            bool success = find_or_create_physical_instance(constraints, 
                                regions, result, created, mapper_id, 
                                processor, acquire, priority, tight_bounds,
                                creator_id, true/*remote*/);
            if (success)
            {
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
                rez.serialize(remote_created);
                rez.serialize<bool>(created);
                if (created)
                {
                  bool min_priority = (priority == GC_NEVER_PRIORITY);
                  rez.serialize<bool>(min_priority);
                  if (min_priority)
                  {
                    rez.serialize(mapper_id);
                    rez.serialize(processor);
                  }
                }
              }
              runtime->send_instance_response(source, rez);
            }
            else // if we failed, we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_OR_CREATE_LAYOUT:
          {
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            MapperID mapper_id;
            derez.deserialize(mapper_id);
            Processor processor;
            derez.deserialize(processor);
            GCPriority priority;
            derez.deserialize(priority);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            UniqueID creator_id;
            derez.deserialize(creator_id);
            bool *remote_success, *remote_created;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            derez.deserialize(remote_created);
            LayoutConstraints *constraints = 
              runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            bool created;
            bool success = find_or_create_physical_instance(constraints, 
                                 regions, result, created, mapper_id, 
                                 processor, acquire, priority, tight_bounds,
                                 creator_id, true/*remote*/);
            if (success)
            {
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
                rez.serialize(remote_created);
                rez.serialize<bool>(created);
                if (created)
                {
                  bool min_priority = (priority == GC_NEVER_PRIORITY);
                  rez.serialize<bool>(min_priority);
                  if (min_priority)
                  {
                    rez.serialize(mapper_id);
                    rez.serialize(processor);
                  }
                }
              }
              runtime->send_instance_response(source, rez);
            }
            else // we failed so just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_ONLY_CONSTRAINTS:
          {
            LayoutConstraintSet constraints; 
            constraints.deserialize(derez);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            bool *remote_success;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            MappingInstance result;
            bool success = find_physical_instance(constraints, regions,
                        result, acquire, tight_bounds, true/*remote*/);
            if (success)
            {
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
              }
              runtime->send_instance_response(source, rez);
            }
            else // we failed so we can just trigger the response
              Runtime::trigger_event(to_trigger);
            break;
          }
        case FIND_ONLY_LAYOUT:
          {
            LayoutConstraintID layout_id;
            derez.deserialize(layout_id);
            bool tight_bounds;
            derez.deserialize(tight_bounds);
            bool *remote_success;
            derez.deserialize(remote_success);
            MappingInstance *remote_target;
            derez.deserialize(remote_target);
            LayoutConstraints *constraints = 
              runtime->find_layout_constraints(layout_id);
            MappingInstance result;
            bool success = find_physical_instance(constraints, regions, 
                        result, acquire, tight_bounds, true/*remote*/);
            if (success)
            {
              PhysicalManager *manager = result.impl;
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(memory);
                rez.serialize(to_trigger);
                rez.serialize(manager->did);
                rez.serialize<bool>(acquire);
                rez.serialize(remote_target);
                rez.serialize(remote_success);
                rez.serialize(kind);
              }
              runtime->send_instance_response(source, rez);
            }
            else // we failed so just trigger
              Runtime::trigger_event(to_trigger);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_instance_response(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedID did;
      derez.deserialize(did);
      bool acquire;
      derez.deserialize(acquire);
      MappingInstance *target;
      derez.deserialize(target);
      bool *success;
      derez.deserialize(success);
      RequestKind kind;
      derez.deserialize(kind);
#ifdef DEBUG_LEGION
      assert((CREATE_INSTANCE_CONSTRAINTS <= kind) &&
             (kind <= FIND_ONLY_LAYOUT));
#endif
      RtEvent manager_ready = RtEvent::NO_RT_EVENT;
      PhysicalManager *manager = 
        runtime->find_or_request_physical_manager(did, manager_ready);
      std::set<RtEvent> preconditions;
      WrapperReferenceMutator mutator(preconditions);
      // If the manager isn't ready yet, then we need to wait for it
      if (manager_ready.exists())
        manager_ready.lg_wait();
      // If we acquired on the owner node, add our own local reference
      // and then remove the remote DID
      if (acquire)
      {
        manager->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
        manager->send_remote_valid_update(source, NULL, 1, false/*add*/);
      }
      *target = MappingInstance(manager);
      *success = true;
      if ((kind == FIND_OR_CREATE_CONSTRAINTS) || 
          (kind == FIND_OR_CREATE_LAYOUT))
      {
        bool *created_ptr;
        derez.deserialize(created_ptr);
        bool created;
        derez.deserialize(created);
        *created_ptr = created;
        bool min_priority = false;
        MapperID mapper_id = 0;
        Processor processor = Processor::NO_PROC;
        if (created)
        {
          derez.deserialize(min_priority);
          if (min_priority)
          {
            derez.deserialize(mapper_id);
            derez.deserialize(processor);
          }
        }
        // Record the instance as a max priority instance
        bool remove_duplicate_valid = false;
        // No need to be safe here, we have a valid reference
        if (created && min_priority)
          manager->add_base_valid_ref(NEVER_GC_REF, &mutator);
        {
          AutoLock m_lock(manager_lock);
          std::map<PhysicalManager*,InstanceInfo>::const_iterator finder = 
            current_instances.find(manager);
          if (finder == current_instances.end())
            current_instances[manager] = InstanceInfo();
          if (created && min_priority)
          {
            std::pair<MapperID,Processor> key(mapper_id,processor);
            InstanceInfo &info = current_instances[manager];
            if (info.min_priority == GC_NEVER_PRIORITY)
              remove_duplicate_valid = true;
            else
              info.min_priority = GC_NEVER_PRIORITY;
            info.mapper_priorities[key] = GC_NEVER_PRIORITY;
          }
        }
        if (remove_duplicate_valid && 
            manager->remove_base_valid_ref(NEVER_GC_REF, &mutator))
          delete manager;
      }
      else if ((kind == CREATE_INSTANCE_CONSTRAINTS) ||
               (kind == CREATE_INSTANCE_LAYOUT))
      {
        bool min_priority;
        derez.deserialize(min_priority);
        MapperID mapper_id = 0;
        Processor processor = Processor::NO_PROC;
        if (min_priority)
        {
          derez.deserialize(mapper_id);
          derez.deserialize(processor);
        }
        bool remove_duplicate_valid = false;
        if (min_priority)
          manager->add_base_valid_ref(NEVER_GC_REF, &mutator);
        {
          std::pair<MapperID,Processor> key(mapper_id,processor);
          AutoLock m_lock(manager_lock);
          std::map<PhysicalManager*,InstanceInfo>::const_iterator finder = 
            current_instances.find(manager);
          if (finder == current_instances.end())
            current_instances[manager] = InstanceInfo();
          if (min_priority)
          {
            InstanceInfo &info = current_instances[manager];
            if (info.min_priority == GC_NEVER_PRIORITY)
              remove_duplicate_valid = true;
            else
              info.min_priority = GC_NEVER_PRIORITY;
            info.mapper_priorities[key] = GC_NEVER_PRIORITY;
          }
        }
        if (remove_duplicate_valid && 
            manager->remove_base_valid_ref(NEVER_GC_REF, &mutator))
          delete manager;
      }
      // Trigger that we are done
      if (!preconditions.empty())
        Runtime::trigger_event(to_trigger,Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_gc_priority_update(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      MapperID mapper_id;
      derez.deserialize(mapper_id);
      Processor processor;
      derez.deserialize(processor);
      GCPriority priority;
      derez.deserialize(priority);
      RtUserEvent never_gc_event;
      derez.deserialize(never_gc_event);
      // Hold our lock to make sure our allocation doesn't change
      // when getting the reference
      PhysicalManager *manager = NULL;
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        DistributedCollectable *dc = 
          runtime->weak_find_distributed_collectable(did);
        if (dc != NULL)
        {
#ifdef DEBUG_LEGION
          manager = dynamic_cast<PhysicalManager*>(dc);
#else
          manager = static_cast<PhysicalManager*>(dc);
#endif
          manager->add_base_resource_ref(MEMORY_MANAGER_REF);
        }
      }
      // If the instance was already collected, there is nothing to do
      if (manager == NULL)
      {
        if (never_gc_event.exists())
        {
          bool *success;
          derez.deserialize(success);
          // Only have to send the message back when we fail
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(memory);
            rez.serialize(success);
            rez.serialize(never_gc_event);
          }
          runtime->send_never_gc_response(source, rez);
        }
        return;
      }
      set_garbage_collection_priority(manager, mapper_id, processor, priority);
      if (never_gc_event.exists())
      {
        bool *success;
        derez.deserialize(success);
        // If we succeed we can trigger immediately, otherwise we
        // have to send back the response to fail
        if (!manager->try_add_base_valid_ref(REMOTE_DID_REF, NULL,
                                             false/*must be valid*/))
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(memory);
            rez.serialize(success);
            rez.serialize(never_gc_event);
          }
          runtime->send_never_gc_response(source, rez);
        }
        else
          Runtime::trigger_event(never_gc_event);
      }
      // Remote our reference
      if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_never_gc_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool *success;
      derez.deserialize(success);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      *success = false;
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_acquire_request(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> failures;
      size_t num_managers;
      derez.deserialize(num_managers);
      for (unsigned idx = 0; idx < num_managers; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        PhysicalManager *manager = NULL;
        // Prevent changes until we can get a resource reference
        {
          AutoLock m_lock(manager_lock,1,false/*exclusive*/);
          DistributedCollectable *dc = 
            runtime->weak_find_distributed_collectable(did);
          if (dc != NULL)
          {
#ifdef DEBUG_LEGION
            manager = dynamic_cast<PhysicalManager*>(dc);
#else
            manager = static_cast<PhysicalManager*>(dc);
#endif
            manager->add_base_resource_ref(MEMORY_MANAGER_REF);
          }
        }
        if (manager == NULL)
        {
          failures.push_back(idx);
          continue;
        }
        // Otherwise try to acquire it locally
        if (!manager->try_add_base_valid_ref(REMOTE_DID_REF, NULL,
                                             false/*needs valid*/))
        {
          failures.push_back(idx);
          if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
            delete manager;
        }
        else // just remove our reference since we succeeded
          manager->remove_base_resource_ref(MEMORY_MANAGER_REF);
      }
      std::vector<unsigned> *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      // See if we had any failures
      if (!failures.empty())
      {
        // Send back the failures
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(memory);
          rez.serialize(target);
          rez.serialize<size_t>(failures.size());
          for (unsigned idx = 0; idx < failures.size(); idx++)
            rez.serialize(idx);
          rez.serialize(to_trigger);
        }
        runtime->send_acquire_response(source, rez);
      }
      else // if we succeeded, then this is easy, just trigger
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::process_acquire_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> *target;
      derez.deserialize(target);
      size_t num_failures;
      derez.deserialize(num_failures);
      for (unsigned idx = 0; idx < num_failures; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        (*target)[index] = false;
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      Runtime::trigger_event(to_trigger);
    }
    
    //--------------------------------------------------------------------------
    bool MemoryManager::find_satisfying_instance(
                                const LayoutConstraintSet &constraints,
                                const std::vector<LogicalRegion> &regions,
                                MappingInstance &result, bool acquire, 
                                bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      std::deque<PhysicalManager*> candidates;
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          if (!it->first->meets_region_tree(regions))
            continue;
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.push_back(it->first);
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        for (std::deque<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_region_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            found = true;
            break;
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_satisfying_instance(LayoutConstraints *constraints,
                                      const std::vector<LogicalRegion> &regions,
                                      MappingInstance &result, bool acquire, 
                                      bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      std::deque<PhysicalManager*> candidates;
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          if (!it->first->meets_region_tree(regions))
            continue;
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.push_back(it->first);
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        for (std::deque<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_region_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            found = true;
            break;
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_satisfying_instance(
                                const LayoutConstraintSet &constraints,
                                const std::vector<LogicalRegion> &regions,
                                MappingInstance &result, 
                                std::set<PhysicalManager*> &candidates,
                                bool acquire, bool tight_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          if (!it->first->meets_region_tree(regions))
            continue;
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.insert(it->first);
        }
      }
      // If we have any candidates check their constraints
      if (!candidates.empty())
      {
        for (std::set<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            return true;
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool MemoryManager::find_satisfying_instance(LayoutConstraints *constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  MappingInstance &result, 
                                  std::set<PhysicalManager*> &candidates,
                                  bool acquire, bool tight_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          if (!it->first->meets_region_tree(regions))
            continue;
          candidates.insert(it->first);
        }
      }
      // If we have any candidates check their constraints
      if (!candidates.empty())
      {
        for (std::set<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            return true;
          }
        }
      }
      return false;
    }


    //--------------------------------------------------------------------------
    bool MemoryManager::find_valid_instance(
                                     const LayoutConstraintSet &constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result, bool acquire, 
                                     bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      std::deque<PhysicalManager*> candidates;
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Only consider ones that are currently valid
          if (it->second.current_state != VALID_STATE)
            continue;
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.push_back(it->first);
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        for (std::deque<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_region_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       true/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            found = true;
            break;
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }
    
    //--------------------------------------------------------------------------
    bool MemoryManager::find_valid_instance(
                                     LayoutConstraints *constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result, bool acquire, 
                                     bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while iterating here
      std::deque<PhysicalManager*> candidates;
      {
        AutoLock m_lock(manager_lock, 1, false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Only consider ones that are currently valid
          if (it->second.current_state != VALID_STATE)
            continue;
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.push_back(it->first);
        }
      }
      // If we have any candidates check their constraints
      bool found = false;
      if (!candidates.empty())
      {
        for (std::deque<PhysicalManager*>::const_iterator it = 
              candidates.begin(); it != candidates.end(); it++)
        {
          if (!(*it)->meets_regions(regions, tight_region_bounds))
            continue;
          if ((*it)->entails(constraints))
          {
            // Check to see if we need to acquire
            if (acquire)
            {
              // If we fail to acquire then keep going
              if (!(*it)->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       true/*must already be valid*/))
                continue;
            }
            // If we make it here, we succeeded
            result = MappingInstance(*it);
            found = true;
            break;
          }
        }
        release_candidate_references(candidates);
      }
      return found;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_candidate_references(
                             const std::set<PhysicalManager*> &candidates) const
    //--------------------------------------------------------------------------
    {
      for (std::set<PhysicalManager*>::const_iterator it = 
            candidates.begin(); it != candidates.end(); it++)
      {
        if ((*it)->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete (*it);
      } 
    }

    //--------------------------------------------------------------------------
    void MemoryManager::release_candidate_references(
                           const std::deque<PhysicalManager*> &candidates) const
    //--------------------------------------------------------------------------
    {
      for (std::deque<PhysicalManager*>::const_iterator it = 
            candidates.begin(); it != candidates.end(); it++)
      {
        if ((*it)->remove_base_resource_ref(MEMORY_MANAGER_REF))
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    MemoryManager::CollectableInfo<SMALLER>::CollectableInfo(PhysicalManager *m,
                                                      size_t size, GCPriority p)
      : manager(m), instance_size(size), priority(p)
    //--------------------------------------------------------------------------
    {
      if (manager != NULL)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    MemoryManager::CollectableInfo<SMALLER>::CollectableInfo(
                                                    const CollectableInfo &rhs)
      : manager(rhs.manager), instance_size(rhs.instance_size), 
        priority(rhs.priority)
    //--------------------------------------------------------------------------
    {
      if (manager != NULL)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    MemoryManager::CollectableInfo<SMALLER>::~CollectableInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((manager != NULL) && 
          manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
        delete manager;
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    MemoryManager::CollectableInfo<SMALLER>& 
      MemoryManager::CollectableInfo<SMALLER>::operator=(
                                                     const CollectableInfo &rhs)
    //--------------------------------------------------------------------------
    {
      if ((manager != NULL) && 
          manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
        delete manager;
      manager = rhs.manager;
      instance_size = rhs.instance_size;
      priority = rhs.priority;
      if (manager != NULL)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    bool MemoryManager::CollectableInfo<SMALLER>::operator<(
                                               const CollectableInfo &rhs) const
    //--------------------------------------------------------------------------
    {
      if (SMALLER)
      {
        // For smaller, we want largest instances first, then largest priorities
        if (instance_size > rhs.instance_size)
          return true;
        else if (instance_size < rhs.instance_size)
          return false;
        else
        {
          if (priority > rhs.priority)
            return true;
          else if (priority < rhs.priority)
            return false;
          else
          {
            if (((unsigned long)manager) < ((unsigned long)rhs.manager))
              return true;
            return false;
          }
        }
      }
      else
      {
        // For larger, we want smallest sizes first, then largest priorities
        if (instance_size < rhs.instance_size)
          return true;
        else if (instance_size > rhs.instance_size)
          return false;
        else
        {
          if (priority > rhs.priority)
            return true;
          else if (priority < rhs.priority)
            return false;
          else
          {
            if (((unsigned long)manager) < ((unsigned long)rhs.manager))
              return true;
            return false;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    bool MemoryManager::CollectableInfo<SMALLER>::operator==(
                                               const CollectableInfo &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((((unsigned long)manager) == ((unsigned long)rhs.manager)) &&
          (instance_size == rhs.instance_size) && (priority == rhs.priority))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::find_instances_by_state(size_t needed_size, 
                                                InstanceState state,
                std::set<CollectableInfo<true> > &smaller_instances,
                std::set<CollectableInfo<false> > &larger_instances) const
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
            current_instances.begin(); it != current_instances.end(); it++)
      {
        if (it->second.current_state != state)
          continue;
        if (it->second.instance_size >= needed_size)
          larger_instances.insert(CollectableInfo<false>(it->first,
                it->second.instance_size, it->second.min_priority));
        else
          smaller_instances.insert(CollectableInfo<true>(it->first,
                it->second.instance_size, it->second.min_priority));
      }
    }

    //--------------------------------------------------------------------------
    PhysicalManager* MemoryManager::allocate_physical_instance(
                                      const LayoutConstraintSet &constraints,
                                      const std::vector<LogicalRegion> &regions,
                                      UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
#endif
      // First, just try to make the instance as is, if it works we are done 
      InstanceBuilder builder(regions, constraints, runtime, this, creator_id);
      PhysicalManager *manager = 
                              builder.create_physical_instance(runtime->forest);
      if (manager != NULL)
        return manager;
      // If that didn't work find the set of immediately collectable regions
      // Rank them by size and then by GC priority
      // Start with all the ones larger than given size and try to 
      // delete them starting from the smallest size and highest priority
      // If that didnt' work try to delete enough from the small set to 
      // open up space.
      const size_t needed_size = builder.compute_needed_size(runtime->forest);
      std::set<CollectableInfo<true> > smaller_instances;
      std::set<CollectableInfo<false> > larger_instances;
      find_instances_by_state(needed_size, COLLECTABLE_STATE,
                              smaller_instances, larger_instances);
      size_t total_bytes_deleted = 0;
      if (!larger_instances.empty())
      {
        PhysicalManager *result = 
          delete_and_allocate<false>(builder, needed_size,
                                     total_bytes_deleted, larger_instances);
        if (result != NULL)
          return result;
        larger_instances.clear();
      }
      if (!smaller_instances.empty())
      {
        PhysicalManager *result = 
          delete_and_allocate<true>(builder, needed_size, 
                                    total_bytes_deleted, smaller_instances);
        if (result != NULL)
          return result;
        smaller_instances.clear();
      }
      // If we still haven't been able to allocate the region do the same
      // thing as above except with the active regions and deferred deletions
      find_instances_by_state(needed_size, ACTIVE_STATE, 
                              smaller_instances, larger_instances);
      if (!larger_instances.empty())
      {
        PhysicalManager *result = 
          delete_and_allocate<false>(builder, needed_size,
                                     total_bytes_deleted, larger_instances);
        if (result != NULL)
          return result;
      }
      if (!smaller_instances.empty())
      {
        PhysicalManager *result = 
          delete_and_allocate<true>(builder, needed_size,
                                    total_bytes_deleted, smaller_instances);
        if (result != NULL)
          return result;
      }
      // If we made it here well then we failed 
      return NULL;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* MemoryManager::find_and_record(PhysicalManager *manager,
                              const LayoutConstraintSet &constraints,
                              const std::vector<LogicalRegion> &regions,
                              std::set<PhysicalManager*> &candidates,
                              bool acquire, MapperID mapper_id,
                              Processor proc, GCPriority priority, 
                              bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
#endif
      // First do the insertion
      // If we're going to add a valid reference, mark this valid early
      // to avoid races with deletions
      bool early_valid = acquire || (priority == GC_NEVER_PRIORITY);
      size_t instance_size = manager->get_instance_size();
      // Since we're going to put this in the table add a reference
      if (is_owner)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
      PhysicalManager *alternate = NULL;
      do
      {
        // Check to see if we have an alternate candidate
        if (alternate != NULL)
        {
          if (!alternate->meets_regions(regions, tight_region_bounds))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          if (!alternate->entails(constraints))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          if (acquire && !alternate->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          // If we make it here then we found a better candidate
          // so we need to delete the instance that we initially made
          if (manager->try_active_deletion())
            manager->perform_deletion(RtEvent::NO_RT_EVENT);
          if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
            delete manager;
          return alternate;
        }
#ifdef DEBUG_LEGION
        assert(alternate == NULL);
#endif
        AutoLock m_lock(manager_lock);
        // Find our candidates
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          // Check if the region trees are the same
          if (!it->first->meets_region_tree(regions))
            continue;
          // If we already considered it we don't have to do it again
          if (candidates.find(it->first) != candidates.end())
            continue;
          // We found an alternate candidate so break out so we can test it
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.insert(it->first);
          alternate = it->first;
          // We found an alternate so we can break out
          break;
        }
        // If we have an alternate we might have to acquire it
        // while not holding the lock so go back around the loop
        if (alternate != NULL)
          continue;
        // If we make it here then we are good to add our instance
#ifdef DEBUG_LEGION
        assert(current_instances.find(manager) == current_instances.end());
#endif
        InstanceInfo &info = current_instances[manager];
        if (early_valid)
          info.current_state = VALID_STATE;
        info.min_priority = priority;
        info.instance_size = instance_size;
        info.mapper_priorities[
          std::pair<MapperID,Processor>(mapper_id,proc)] = priority;
        // Break out because we are done
        break;
      } while (true);
      // If we make it here we've successfully added ourselves
      // and found no satisfying instances added in between
      // Now we can add any references that we need to
      if (acquire)
      {
        if (remote)
          manager->add_base_valid_ref(REMOTE_DID_REF);
        else
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF);
      } 
      // If we have a GC_NEVER_PRIORITY then we have to add the valid reference
      if (priority == GC_NEVER_PRIORITY)
        manager->add_base_valid_ref(NEVER_GC_REF);
      return manager;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* MemoryManager::find_and_record(PhysicalManager *manager,
                              LayoutConstraints *constraints,
                              const std::vector<LogicalRegion> &regions,
                              std::set<PhysicalManager*> &candidates,
                              bool acquire, MapperID mapper_id,
                              Processor proc, GCPriority priority, 
                              bool tight_region_bounds, bool remote)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
#endif
      // First do the insertion
      // If we're going to add a valid reference, mark this valid early
      // to avoid races with deletions
      bool early_valid = acquire || (priority == GC_NEVER_PRIORITY);
      size_t instance_size = manager->get_instance_size();
      // Since we're going to put this in the table add a reference
      if (is_owner)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
      PhysicalManager *alternate = NULL;
      do
      {
        // Check to see if we have an alternate candidate
        if (alternate != NULL)
        {
          if (!alternate->meets_regions(regions, tight_region_bounds))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          if (!alternate->entails(constraints))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          if (acquire && !alternate->try_add_base_valid_ref(
                    remote ? REMOTE_DID_REF : MAPPING_ACQUIRE_REF, NULL,
                                       false/*must already be valid*/))
          {
            alternate = NULL;
            continue; // Go back around the loop
          }
          // If we make it here then we found a better candidate
          // so we need to delete the instance that we initially made
          if (manager->try_active_deletion())
            manager->perform_deletion(RtEvent::NO_RT_EVENT);
          if (manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
            delete manager;
          return alternate;
        }
#ifdef DEBUG_LEGION
        assert(alternate == NULL);
#endif
        AutoLock m_lock(manager_lock);
        // Find our candidates
        for (std::map<PhysicalManager*,InstanceInfo>::const_iterator it = 
              current_instances.begin(); it != current_instances.end(); it++)
        {
          // Skip it if has already been collected
          if (it->second.current_state == ACTIVE_COLLECTED_STATE)
            continue;
          // Check if the region trees are the same
          if (!it->first->meets_region_tree(regions))
            continue;
          // If we already considered it we don't have to do it again
          if (candidates.find(it->first) != candidates.end())
            continue;
          // We found an alternate candidate so break out so we can test it
          it->first->add_base_resource_ref(MEMORY_MANAGER_REF);
          candidates.insert(it->first);
          alternate = it->first;
          // We found an alternate so we can break out
          break;
        }
        // If we have an alternate we might have to acquire it
        // while not holding the lock so go back around the loop
        if (alternate != NULL)
          continue;
        // If we make it here then we are good to add our instance
#ifdef DEBUG_LEGION
        assert(current_instances.find(manager) == current_instances.end());
#endif
        InstanceInfo &info = current_instances[manager];
        if (early_valid)
          info.current_state = VALID_STATE;
        info.min_priority = priority;
        info.instance_size = instance_size;
        info.mapper_priorities[
          std::pair<MapperID,Processor>(mapper_id,proc)] = priority;
        // Break out because we are done
        break;
      } while (true);
      // If we make it here we've successfully added ourselves
      // and found no satisfying instances added in between
      // Now we can add any references that we need to
      if (acquire)
      {
        if (remote)
          manager->add_base_valid_ref(REMOTE_DID_REF);
        else
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF);
      }
      if (priority == GC_NEVER_PRIORITY)
        manager->add_base_valid_ref(NEVER_GC_REF);
      return manager;
    }

    //--------------------------------------------------------------------------
    void MemoryManager::record_created_instance(PhysicalManager *manager,
                           bool acquire, MapperID mapper_id, Processor p, 
                           GCPriority priority, bool remote)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner);
#endif
      // First do the insertion
      // If we're going to add a valid reference, mark this valid early
      // to avoid races with deletions
      bool early_valid = acquire || (priority == GC_NEVER_PRIORITY);
      size_t instance_size = manager->get_instance_size();
      // Since we're going to put this in the table add a reference
      if (is_owner)
        manager->add_base_resource_ref(MEMORY_MANAGER_REF);
      {
        AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
        assert(current_instances.find(manager) == current_instances.end());
#endif
        InstanceInfo &info = current_instances[manager];
        if (early_valid)
          info.current_state = VALID_STATE;
        info.min_priority = priority;
        info.instance_size = instance_size;
        info.mapper_priorities[
          std::pair<MapperID,Processor>(mapper_id,p)] = priority;
      }
      // Now we can add any references that we need to
      if (acquire)
      {
        if (remote)
          manager->add_base_valid_ref(REMOTE_DID_REF);
        else
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF);
      }
      if (priority == GC_NEVER_PRIORITY)
        manager->add_base_valid_ref(NEVER_GC_REF);
    }

    //--------------------------------------------------------------------------
    void MemoryManager::record_deleted_instance(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      RtEvent deletion_precondition;
      bool remove_reference = false;
      {
        AutoLock m_lock(manager_lock);
        std::map<PhysicalManager*,InstanceInfo>::iterator finder = 
          current_instances.find(manager);
#ifdef DEBUG_LEGION
        assert(finder != current_instances.end());
        assert(finder->second.current_state != VALID_STATE);
        assert(finder->second.current_state != ACTIVE_COLLECTED_STATE);
#endif
        // If we are still in an active mode, record the event,
        // otherwise we can delete everything now and trigger
        // the event immediately
        if (finder->second.current_state == ACTIVE_STATE)
        {
          finder->second.current_state = ACTIVE_COLLECTED_STATE;
          finder->second.deferred_collect = Runtime::create_rt_user_event();
          deletion_precondition = finder->second.deferred_collect;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(finder->second.current_state == COLLECTABLE_STATE);
#endif
          current_instances.erase(finder);
          if (is_owner)
            remove_reference = true;
        }
      }
      manager->perform_deletion(deletion_precondition);
      if (remove_reference && 
          manager->remove_base_resource_ref(MEMORY_MANAGER_REF))
        delete manager;
    }

    //--------------------------------------------------------------------------
    template<bool SMALLER>
    PhysicalManager* MemoryManager::delete_and_allocate(
                                InstanceBuilder &builder, size_t needed_size, 
                                size_t &total_bytes_deleted,
                          const std::set<CollectableInfo<SMALLER> > &instances)
    //--------------------------------------------------------------------------
    {
      for (typename std::set<CollectableInfo<SMALLER> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        PhysicalManager *target_manager = it->manager;
        // Garbage collector is not allowed to delete external instances
        if (target_manager->is_external_instance())
          continue;
        if (target_manager->try_active_deletion())
        {
          record_deleted_instance(target_manager);
          total_bytes_deleted += it->instance_size;
          // Only need to do the test if we're smaller
          if (!SMALLER || (total_bytes_deleted >= needed_size))
          {
            PhysicalManager *manager = 
              builder.create_physical_instance(runtime->forest);
            // If we succeeded we are done
            if (manager != NULL)
              return manager;
          }
        }
      }
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Virtual Channel 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(VirtualChannelKind kind, 
        AddressSpaceID local_address_space, 
        size_t max_message_size, LegionProfiler *prof)
      : sending_buffer((char*)malloc(max_message_size)), 
        sending_buffer_size(max_message_size), 
        observed_recent(true), profiler(prof)
    //--------------------------------------------------------------------------
    //
    {
      receiving_buffer_size = max_message_size;
      receiving_buffer = (char*)legion_malloc(MESSAGE_BUFFER_ALLOC,
                                              receiving_buffer_size);
#ifdef DEBUG_LEGION
      assert(sending_buffer != NULL);
      assert(receiving_buffer != NULL);
#endif
      // Set up the buffer for sending the first batch of messages
      // Only need to write the processor once
      *((LgTaskID*)sending_buffer) = LG_MESSAGE_ID;
      sending_index = sizeof(LgTaskID);
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
      last_message_event = RtEvent::NO_RT_EVENT;
      partial = false;
      // Set up the receiving buffer
      received_messages = 0;
      receiving_index = 0;
    }

    //--------------------------------------------------------------------------
    VirtualChannel::VirtualChannel(const VirtualChannel &rhs)
      : sending_buffer(NULL), sending_buffer_size(0), profiler(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VirtualChannel::~VirtualChannel(void)
    //--------------------------------------------------------------------------
    {
      free(sending_buffer);
      free(receiving_buffer);
      receiving_buffer = NULL;
      receiving_buffer_size = 0;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::package_message(Serializer &rez, MessageKind k,
                         bool flush, Runtime *runtime, Processor target, 
                         bool response, bool shutdown)
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
          send_message(true/*complete*/, runtime, target, response, shutdown);
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
            send_message(false/*complete*/, runtime, 
                         target, response, shutdown);
          remaining = sending_buffer_size - sending_index;
#ifdef DEBUG_LEGION
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
        send_message(true/*complete*/, runtime, target, response, shutdown);
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::send_message(bool complete, Runtime *runtime,
                                 Processor target, bool response, bool shutdown)
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
      const size_t base_size = sizeof(LgTaskID) + sizeof(AddressSpaceID) 
                                + sizeof(VirtualChannelKind);
      *((MessageHeader*)(sending_buffer + base_size)) = header;
      *((unsigned*)(sending_buffer + base_size + sizeof(header))) = 
                                                            packaged_messages;
      // Send the message directly there, don't go through the
      // runtime interface to avoid being counted, still include
      // a profiling request though if necessary in order to 
      // see waits on message handlers
      // Note that we don't profile on shutdown messages or we would 
      // never actually finish running
      if (!shutdown && (Runtime::num_profiling_nodes > 0) && 
          (runtime->find_address_space(target) < Runtime::num_profiling_nodes))
      {
        Realm::ProfilingRequestSet requests;
        LegionProfiler::add_message_request(requests, target);
        last_message_event = RtEvent(target.spawn(LG_TASK_ID, sending_buffer, 
            sending_index, requests, last_message_event, response ? 
              LG_LATENCY_RESPONSE_PRIORITY : LG_LATENCY_MESSAGE_PRIORITY));
      }
      else
        last_message_event = RtEvent(target.spawn(LG_TASK_ID, sending_buffer, 
              sending_index, last_message_event, response ? 
                LG_LATENCY_RESPONSE_PRIORITY : LG_LATENCY_MESSAGE_PRIORITY));
      // Reset the state of the buffer
      sending_index = base_size + sizeof(header) + sizeof(unsigned);
      if (partial)
        header = PARTIAL_MESSAGE;
      else
        header = FULL_MESSAGE;
      packaged_messages = 0;
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::confirm_shutdown(ShutdownManager *shutdown_manager,
                                          bool phase_one)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(send_lock);
      if (phase_one)
      {
        if (packaged_messages > 0)
          shutdown_manager->record_recent_message();
        // This is only sound because we know we are on the owner
        // node for the event, otherwise Realm could lie to us
        if (!last_message_event.has_triggered())
        {
          // A little hack here for slow gasnet conduits
          // If the event didn't trigger yet, make sure its just
          // because we haven't gotten the return message yet
          usleep(1000);
          if (!last_message_event.has_triggered())
            shutdown_manager->record_pending_message(last_message_event);
          else
            observed_recent = false;
        }
        else
          observed_recent = false;
      }
      else
      {
        if (observed_recent || (packaged_messages > 0)) 
          shutdown_manager->record_recent_message(); 
        else if (!last_message_event.has_triggered())
        {
          // A little hack here for slow gasnet conduits
          // If the event didn't trigger yet, make sure its just
          // because we haven't gotten the return message yet
          usleep(1000);
          if (!last_message_event.has_triggered())
            shutdown_manager->record_recent_message();
        }
      }
    }

    //--------------------------------------------------------------------------
    void VirtualChannel::process_message(const void *args, size_t arglen,
                         Runtime *runtime, AddressSpaceID remote_address_space)
    //--------------------------------------------------------------------------
    {
      // If we have a profiler we need to increment our requests count
      if (profiler != NULL)
#ifdef DEBUG_LEGION
        profiler->increment_total_outstanding_requests(
                    LegionProfiler::LEGION_PROF_MESSAGE);
#else
        profiler->increment_total_outstanding_requests();
#endif
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
                                         Runtime *runtime,
                                         AddressSpaceID remote_address_space,
                                         const char *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // For profiling if we are doing it
      unsigned long long start = 0, stop = 0;
      for (unsigned idx = 0; idx < num_messages; idx++)
      {
        // Pull off the message kind and the size of the message
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        if (idx == (num_messages-1))
          assert(message_size == arglen);
#endif
        if (profiler != NULL)
          start = Realm::Clock::current_time_in_nanoseconds();
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
          case SEND_INDEX_SPACE_SET:
            {
              runtime->handle_index_space_set(derez, remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_CHILD_REQUEST:
            {
              runtime->handle_index_space_child_request(derez, 
                                                        remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_CHILD_RESPONSE:
            {
              runtime->handle_index_space_child_response(derez);
              break;
            }
          case SEND_INDEX_SPACE_COLORS_REQUEST:
            {
              runtime->handle_index_space_colors_request(derez,
                                                         remote_address_space);
              break;
            }
          case SEND_INDEX_SPACE_COLORS_RESPONSE:
            {
              runtime->handle_index_space_colors_response(derez);
              break;
            }
          case SEND_INDEX_PARTITION_NOTIFICATION:
            {
              runtime->handle_index_partition_notification(derez);
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
              runtime->handle_index_partition_child_request(derez,
                                                          remote_address_space);
              break;
            }
          case SEND_INDEX_PARTITION_CHILD_RESPONSE:
            {
              runtime->handle_index_partition_child_response(derez);
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
          case SEND_LOCAL_FIELD_ALLOC_REQUEST:
            {
              runtime->handle_local_field_alloc_request(derez, 
                                                        remote_address_space);
              break;
            }
          case SEND_LOCAL_FIELD_ALLOC_RESPONSE:
            {
              runtime->handle_local_field_alloc_response(derez);
              break;
            }
          case SEND_LOCAL_FIELD_FREE:
            {
              runtime->handle_local_field_free(derez);
              break;
            }
          case SEND_LOCAL_FIELD_UPDATE:
            {
              runtime->handle_local_field_update(derez);
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
          case DISTRIBUTED_INVALIDATE:
            {
              runtime->handle_did_remote_invalidate(derez);
              break;
            }
          case DISTRIBUTED_DEACTIVATE:
            {
              runtime->handle_did_remote_deactivate(derez);
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
          case DISTRIBUTED_UNREGISTER:
            {
              runtime->handle_did_remote_unregister(derez);
              break;
            }
          case SEND_ATOMIC_RESERVATION_REQUEST:
            {
              runtime->handle_send_atomic_reservation_request(derez,
                                                      remote_address_space);
              break;
            }
          case SEND_ATOMIC_RESERVATION_RESPONSE:
            {
              runtime->handle_send_atomic_reservation_response(derez);
              break;
            }
          case SEND_BACK_LOGICAL_STATE:
            {
              runtime->handle_send_back_logical_state(derez);
              break;
            }
          case SEND_MATERIALIZED_VIEW:
            {
              runtime->handle_send_materialized_view(derez, 
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
          case SEND_PHI_VIEW:
            {
              runtime->handle_send_phi_view(derez, remote_address_space);
              break;
            }
          case SEND_REDUCTION_VIEW:
            {
              runtime->handle_send_reduction_view(derez, remote_address_space);
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
          case SEND_CREATE_TOP_VIEW_REQUEST:
            {
              runtime->handle_create_top_view_request(derez,
                                                      remote_address_space);
              break;
            }
          case SEND_CREATE_TOP_VIEW_RESPONSE:
            {
              runtime->handle_create_top_view_response(derez);
              break;
            }
          case SEND_SUBVIEW_DID_REQUEST:
            {
              runtime->handle_subview_did_request(derez, remote_address_space);
              break;
            }
          case SEND_SUBVIEW_DID_RESPONSE:
            {
              runtime->handle_subview_did_response(derez);
              break;
            }
          case SEND_VIEW_REQUEST:
            {
              runtime->handle_view_request(derez, remote_address_space);
              break;
            }
          case SEND_VIEW_UPDATE_REQUEST:
            {
              runtime->handle_view_update_request(derez, remote_address_space);
              break;
            }
          case SEND_VIEW_UPDATE_RESPONSE:
            {
              runtime->handle_view_update_response(derez);
              break;
            }
          case SEND_VIEW_REMOTE_UPDATE:
            {
              runtime->handle_view_remote_update(derez, remote_address_space);
              break;
            }
          case SEND_VIEW_REMOTE_INVALIDATE:
            {
              runtime->handle_view_remote_invalidate(derez);
              break;
            }
          case SEND_MANAGER_REQUEST:
            {
              runtime->handle_manager_request(derez, remote_address_space);
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
          case SEND_FUTURE_MAP_REQUEST:
            {
              runtime->handle_future_map_future_request(derez, 
                                        remote_address_space);
              break;
            }
          case SEND_FUTURE_MAP_RESPONSE:
            {
              runtime->handle_future_map_future_response(derez);
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
          case SEND_REMOTE_CONTEXT_REQUEST:
            {
              runtime->handle_remote_context_request(derez, 
                                                     remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_RESPONSE:
            {
              runtime->handle_remote_context_response(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_RELEASE:
            {
              runtime->handle_remote_context_release(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_FREE:
            {
              runtime->handle_remote_context_free(derez);
              break;
            }
          case SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST:
            {
              runtime->handle_remote_context_physical_request(derez,
                                              remote_address_space);
              break;
            }
          case SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE:
            {
              runtime->handle_remote_context_physical_response(derez);
              break;
            }
          case SEND_VERSION_OWNER_REQUEST: 
            {
              runtime->handle_version_owner_request(derez,remote_address_space);
              break;
            }
          case SEND_VERSION_OWNER_RESPONSE:
            {
              runtime->handle_version_owner_response(derez);
              break;
            }
          case SEND_VERSION_STATE_REQUEST:
            {
              runtime->handle_version_state_request(derez,remote_address_space);
              break;
            }
          case SEND_VERSION_STATE_RESPONSE:
            {
              runtime->handle_version_state_response(derez,
                                                     remote_address_space);
              break;
            }
          case SEND_VERSION_STATE_UPDATE_REQUEST:
            {
              runtime->handle_version_state_update_request(derez);
              break;
            }
          case SEND_VERSION_STATE_UPDATE_RESPONSE:
            {
              runtime->handle_version_state_update_response(derez);
              break;
            }
          case SEND_VERSION_STATE_VALID_NOTIFICATION:
            {
              runtime->handle_version_state_valid_notification(derez,
                                                 remote_address_space);
              break;
            }
          case SEND_VERSION_MANAGER_ADVANCE:
            {
              runtime->handle_version_manager_advance(derez);
              break;
            }
          case SEND_VERSION_MANAGER_INVALIDATE:
            {
              runtime->handle_version_manager_invalidate(derez);
              break;
            }
          case SEND_VERSION_MANAGER_REQUEST:
            {
              runtime->handle_version_manager_request(derez,
                                                      remote_address_space);
              break;
            }
          case SEND_VERSION_MANAGER_RESPONSE:
            {
              runtime->handle_version_manager_response(derez);
              break;
            }
          case SEND_VERSION_MANAGER_UNVERSIONED_REQUEST:
            {
              runtime->handle_version_manager_unversioned_request(derez,
                                                  remote_address_space);
              break;
            }
          case SEND_VERSION_MANAGER_UNVERSIONED_RESPONSE:
            {
              runtime->handle_version_manager_unversioned_response(derez);
              break;
            }
          case SEND_INSTANCE_REQUEST:
            {
              runtime->handle_instance_request(derez, remote_address_space);
              break;
            }
          case SEND_INSTANCE_RESPONSE:
            {
              runtime->handle_instance_response(derez, remote_address_space);
              break;
            }
          case SEND_GC_PRIORITY_UPDATE:
            {
              runtime->handle_gc_priority_update(derez, remote_address_space);
              break;
            }
          case SEND_NEVER_GC_RESPONSE:
            {
              runtime->handle_never_gc_response(derez);
              break;
            }
          case SEND_ACQUIRE_REQUEST:
            {
              runtime->handle_acquire_request(derez, remote_address_space);
              break;
            }
          case SEND_ACQUIRE_RESPONSE:
            {
              runtime->handle_acquire_response(derez);
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
          case SEND_VARIANT_BROADCAST:
            {
              runtime->handle_variant_broadcast(derez);
              break;
            }
          case SEND_CONSTRAINT_REQUEST:
            {
              runtime->handle_constraint_request(derez, remote_address_space);
              break;
            }
          case SEND_CONSTRAINT_RESPONSE:
            {
              runtime->handle_constraint_response(derez, remote_address_space);
              break;
            }
          case SEND_CONSTRAINT_RELEASE:
            {
              runtime->handle_constraint_release(derez);
              break;
            }
          case SEND_CONSTRAINT_REMOVAL:
            {
              runtime->handle_constraint_removal(derez);
              break;
            }
          case SEND_TOP_LEVEL_TASK_REQUEST:
            {
              runtime->handle_top_level_task_request(derez);
              break;
            }
          case SEND_TOP_LEVEL_TASK_COMPLETE:
            {
              runtime->handle_top_level_task_complete(derez);
              break;
            }
          case SEND_MPI_RANK_EXCHANGE:
            {
              runtime->handle_mpi_rank_exchange(derez);
              break;
            }
          case SEND_SHUTDOWN_NOTIFICATION:
            {
              // If we have a profiler, we shouldn't have incremented the
              // outstanding profiling count because we don't actually
              // do profiling requests on shutdown messages
              if (profiler != NULL)
#ifdef DEBUG_LEGION
                profiler->decrement_total_outstanding_requests(
                            LegionProfiler::LEGION_PROF_MESSAGE);
#else
                profiler->decrement_total_outstanding_requests();
#endif
              runtime->handle_shutdown_notification(derez,remote_address_space);
              break;
            }
          case SEND_SHUTDOWN_RESPONSE:
            {
              // If we have a profiler, we shouldn't have incremented the
              // outstanding profiling count because we don't actually
              // do profiling requests on shutdown messages
              if (profiler != NULL)
#ifdef DEBUG_LEGION
                profiler->decrement_total_outstanding_requests(
                            LegionProfiler::LEGION_PROF_MESSAGE);
#else
                profiler->decrement_total_outstanding_requests();
#endif
              runtime->handle_shutdown_response(derez);
              break;
            }
          default:
            assert(false); // should never get here
        }
        if (profiler != NULL)
        {
          stop = Realm::Clock::current_time_in_nanoseconds();
          profiler->record_message(kind, start, stop);
        }
        // Update the args and arglen
        args += message_size;
        arglen -= message_size;
      }
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        assert(new_buffer_size != 0); // would cause deallocation
#endif
        // Now realloc the memory
        void *new_ptr = legion_realloc(MESSAGE_BUFFER_ALLOC, receiving_buffer,
                                       receiving_buffer_size, new_buffer_size);
        receiving_buffer_size = new_buffer_size;
#ifdef DEBUG_LEGION
        assert(new_ptr != NULL);
#endif
        receiving_buffer = (char*)new_ptr;
      }
      // Copy the data in
      memcpy(receiving_buffer+receiving_index,args,arglen);
      receiving_index += arglen;
    }

    /////////////////////////////////////////////////////////////
    // Message Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MessageManager::MessageManager(AddressSpaceID remote,
                                   Runtime *rt, size_t max_message_size,
                                   const std::set<Processor> &remote_util_procs)
      : remote_address_space(remote), runtime(rt), channels((VirtualChannel*)
                      malloc(MAX_NUM_VIRTUAL_CHANNELS*sizeof(VirtualChannel))) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote != runtime->address_space);
#endif
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
#ifdef DEBUG_LEGION
        assert(target.exists());
#endif
      }
      // Initialize our virtual channels 
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
      {
        new (channels+idx) VirtualChannel((VirtualChannelKind)idx,
            rt->address_space, max_message_size, runtime->profiler);
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
    void MessageManager::send_message(Serializer &rez, MessageKind kind,
           VirtualChannelKind channel, bool flush, bool response, bool shutdown)
    //--------------------------------------------------------------------------
    {
      channels[channel].package_message(rez, kind, flush, runtime, 
                                        target, response, shutdown);
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

    //--------------------------------------------------------------------------
    void MessageManager::confirm_shutdown(ShutdownManager *shutdown_manager, 
                                          bool phase_one)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_VIRTUAL_CHANNELS; idx++)
        channels[idx].confirm_shutdown(shutdown_manager, phase_one);
    }

    /////////////////////////////////////////////////////////////
    // Shutdown Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(ShutdownPhase p, Runtime *rt, 
                                     AddressSpaceID s, unsigned r, 
                                     ShutdownManager *own)
      : phase(p), runtime(rt), source(s), radix(r), owner(own),
        needed_responses(0), result(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShutdownManager::ShutdownManager(const ShutdownManager &rhs)
      : phase(rhs.phase), runtime(NULL), source(0), radix(0), owner(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShutdownManager::~ShutdownManager(void)
    //--------------------------------------------------------------------------
    {
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
        AddressSpaceID next = start+idx;
        if (next < runtime->total_address_spaces)
          targets.push_back(next);
        else
          break;
      }
      
      if (!targets.empty())
      {
        // Set the number of needed_responses
        needed_responses = targets.size();
        Serializer rez;
        rez.serialize(this);
        rez.serialize(phase);
        for (std::vector<AddressSpaceID>::const_iterator it = 
              targets.begin(); it != targets.end(); it++)
          runtime->send_shutdown_notification(*it, rez); 
        return false;
      }
      else // no messages means we can finalize right now
      {
        finalize();
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool ShutdownManager::handle_response(bool success,
                                          const std::set<RtEvent> &to_add)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock s_lock(shutdown_lock);
        if (result && !success)
          result = false;
        wait_for.insert(to_add.begin(), to_add.end()); 
#ifdef DEBUG_LEGION
        assert(needed_responses > 0);
#endif
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
      runtime->confirm_runtime_shutdown(this, phase);
#ifdef DEBUG_SHUTDOWN_HANG
      if (!result)
      {
        LG_TASK_DESCRIPTIONS(task_descs);
        // Only need to see tasks less than this 
        for (unsigned idx = 0; idx < LG_MESSAGE_ID; idx++)
        {
          if (runtime->outstanding_counts[idx] == 0)
            continue;
          log_shutdown.info("Meta-Task %s: %d outstanding",
                task_descs[idx], runtime->outstanding_counts[idx]);
        }
      }
#endif
      if (result && (runtime->address_space == source))
      {
        log_shutdown.info("SHUTDOWN PHASE %d SUCCESS!", phase);
        if (phase != CONFIRM_SHUTDOWN)
        {
          if (phase == CONFIRM_TERMINATION)
            runtime->prepare_runtime_shutdown();
          // Do the next phase
          runtime->initiate_runtime_shutdown(source, (ShutdownPhase)(phase+1));
        }
        else
        {
          log_shutdown.info("SHUTDOWN SUCCEEDED!");
          runtime->finalize_runtime_shutdown();
        }
      }
      else if (runtime->address_space != source)
      {
#ifdef DEBUG_LEGION
        assert(owner != NULL);
#endif
        // Send the message back
        Serializer rez;
        rez.serialize(owner);
        rez.serialize<bool>(result);
        rez.serialize<size_t>(wait_for.size());
        for (std::set<RtEvent>::const_iterator it = 
              wait_for.begin(); it != wait_for.end(); it++)
          rez.serialize(*it);
        runtime->send_shutdown_response(source, rez);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!result);
#endif
        log_shutdown.info("FAILED SHUTDOWN PHASE %d! Trying again...", phase);
        RtEvent precondition;
        if (!wait_for.empty())
          precondition = Runtime::merge_events(wait_for);
        // If we failed an even phase we go back to the one before it
        RetryShutdownArgs args;
        if ((phase % 2) == 0)
          args.phase = (ShutdownPhase)(phase-1);
        else
          args.phase = phase;
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY,
                                         NULL, precondition);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownManager::handle_shutdown_notification(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShutdownManager *owner;
      derez.deserialize(owner);
      ShutdownPhase phase;
      derez.deserialize(phase);
      runtime->initiate_runtime_shutdown(source, phase, owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShutdownManager::handle_shutdown_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      ShutdownManager *shutdown_manager;
      derez.deserialize(shutdown_manager);
      bool success;
      derez.deserialize(success);
      size_t num_events;
      derez.deserialize(num_events);
      std::set<RtEvent> wait_for;
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        RtEvent event;
        derez.deserialize(event);
        wait_for.insert(event);
      }
      if (shutdown_manager->handle_response(success, wait_for))
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
    void ShutdownManager::record_recent_message(void)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      log_shutdown.info("Outstanding message on node %d", 
                        runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void ShutdownManager::record_pending_message(RtEvent pending_event)
    //--------------------------------------------------------------------------
    {
      // Instant death
      result = false;
      wait_for.insert(pending_event);
      log_shutdown.info("Pending message on node %d", runtime->address_space);
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
    void GarbageCollectionEpoch::add_collection(LogicalView *view, ApEvent term,
                                                ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalView*,std::set<ApEvent> >::iterator finder = 
        collections.find(view);
      if (finder == collections.end())
      {
        // Add a garbage collection reference to the view, it will
        // be removed in LogicalView::handle_deferred_collect
        view->add_base_gc_ref(PENDING_GC_REF, mutator);
        collections[view].insert(term);
      }
      else
        finder->second.insert(term);
    }

    //--------------------------------------------------------------------------
    RtEvent GarbageCollectionEpoch::launch(void)
    //--------------------------------------------------------------------------
    {
      // Set remaining to the total number of collections
      remaining = collections.size();
      GarbageCollectionArgs args;
      args.epoch = this;
      std::set<RtEvent> events;
      for (std::map<LogicalView*,std::set<ApEvent> >::const_iterator it =
            collections.begin(); it != collections.end(); /*nothing*/)
      {
        args.view = it->first;
        RtEvent precondition = Runtime::protect_merge_events(it->second);
        // Avoid the deletion race by testing the condition 
        // before launching the task
        it++;
        bool done = (it == collections.end());
        RtEvent e = runtime->issue_runtime_meta_task(args,
                                         LG_THROUGHPUT_WORK_PRIORITY, NULL,
                                         precondition);
        events.insert(e);
        if (done)
          break;
      }
      return Runtime::merge_events(events);
    }

    //--------------------------------------------------------------------------
    bool GarbageCollectionEpoch::handle_collection(
                                              const GarbageCollectionArgs *args)
    //--------------------------------------------------------------------------
    {
      std::map<LogicalView*,std::set<ApEvent> >::iterator finder = 
        collections.find(args->view);
#ifdef DEBUG_LEGION
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
                                  CodeDescriptor *realm, const char *task_name)
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
    void PendingVariantRegistration::perform_registration(Runtime *runtime)
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
    TaskImpl::TaskImpl(TaskID tid, Runtime *rt, const char *name/*=NULL*/)
      : task_id(tid), runtime(rt), initial_name(static_cast<char*>(
          malloc(((name == NULL) ? 64 : strlen(name) + 1) * sizeof(char)))),
        has_return_type(false), all_idempotent(false)
    //--------------------------------------------------------------------------
    {
      // Always fill in semantic info 0 with a name for the task
      if (name != NULL)
      {
        const size_t name_size = strlen(name) + 1; // for \0
        char *name_copy = (char*)legion_malloc(SEMANTIC_INFO_ALLOC, name_size);
        memcpy(name_copy, name, name_size);
        semantic_infos[NAME_SEMANTIC_TAG] = 
          SemanticInfo(name_copy, name_size, false/*mutable*/);
        if (Runtime::legion_spy_enabled)
          LegionSpy::log_task_name(task_id, name);
        // Also set the initial name to be safe
        memcpy(initial_name, name, name_size);
        // Register this task with the profiler if necessary
        if (runtime->profiler != NULL)
          runtime->profiler->register_task_kind(task_id, name, false);
      }
      else // Just set the initial name
      {
        snprintf(initial_name,64,"unnamed_task_%d", task_id);
        // Register this task with the profiler if necessary
        if (runtime->profiler != NULL)
          runtime->profiler->register_task_kind(task_id, initial_name, false);
      }
    }

    //--------------------------------------------------------------------------
    TaskImpl::TaskImpl(const TaskImpl &rhs)
      : task_id(rhs.task_id), runtime(rhs.runtime), initial_name(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskImpl::~TaskImpl(void)
    //-------------------------------------------------------------------------
    {
      for (std::map<SemanticTag,SemanticInfo>::const_iterator it = 
            semantic_infos.begin(); it != semantic_infos.end(); it++)
      {
        legion_free(SEMANTIC_INFO_ALLOC, it->second.buffer,
                    it->second.size);
      }
      semantic_infos.clear();
      free(initial_name);
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
    VariantID TaskImpl::get_unique_variant_id(void)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(task_lock);
      // VariantIDs have to uniquely identify our node so start at our
      // current runtime name and stride by the number of nodes
      VariantID result = runtime->address_space;
      if (result == 0) // Never use VariantID 0
        result = runtime->runtime_stride;
      for ( ; result <= (UINT_MAX - runtime->runtime_stride); 
            result += runtime->runtime_stride)
      {
        if (variants.find(result) != variants.end())
          continue;
        if (pending_variants.find(result) != pending_variants.end())
          continue;
        pending_variants.insert(result);
        return result;
      }
      assert(false);
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::add_variant(VariantImpl *impl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl->owner == this);
#endif
      AutoLock t_lock(task_lock);
      if (!variants.empty())
      {
        // Make sure that all the variants agree whether there is 
        // a return type or not
        if (has_return_type != impl->returns_value())
          REPORT_LEGION_ERROR(ERROR_RETURN_SIZE_MISMATCH, 
                        "Variants of task %s (ID %d) disagree on whether "
                        "there is a return type or not. All variants "
                        "of a task must agree on whether there is a "
                        "return type.", get_name(false/*need lock*/), task_id)
        if (all_idempotent != impl->is_idempotent())
          REPORT_LEGION_ERROR(ERROR_IDEMPOTENT_MISMATCH, 
                        "Variants of task %s (ID %d) have different idempotent "
                        "options.  All variants of the same task must "
                        "all be either idempotent or non-idempotent.",
                        get_name(false/*need lock*/), task_id)
      }
      else
      {
        has_return_type = impl->returns_value();
        all_idempotent  = impl->is_idempotent();
      }
      variants[impl->vid] = impl;
      // Erase the outstanding request if there is one
      outstanding_requests.erase(impl->vid);
      // Erase the pending VariantID if there is one
      pending_variants.erase(impl->vid);
    }

    //--------------------------------------------------------------------------
    VariantImpl* TaskImpl::find_variant_impl(VariantID variant_id,bool can_fail)
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
        if (can_fail)
          return NULL;
        REPORT_LEGION_ERROR(ERROR_UNREGISTERED_VARIANT, 
                            "Unable to find variant %ld of task %s!",
                            variant_id, get_name())
      }
      // Retake the lock and see if we can send a request
      RtEvent wait_on;
      RtUserEvent request_event;
      {
        AutoLock t_lock(task_lock);
        // Check to see if we lost the race
        std::map<VariantID,VariantImpl*>::const_iterator finder = 
          variants.find(variant_id);
        if (finder != variants.end())
          return finder->second;
        std::map<VariantID,RtEvent>::const_iterator out_finder = 
          outstanding_requests.find(variant_id);
        if (out_finder == outstanding_requests.end())
        {
          request_event = Runtime::create_rt_user_event();
          outstanding_requests[variant_id] = request_event;
          wait_on = request_event;
        }
        else
          wait_on = out_finder->second;
      }
      if (request_event.exists())
      {
        // Send a request to the owner node for the variant
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(task_id);
          rez.serialize(variant_id);
          rez.serialize(wait_on);
          rez.serialize(can_fail);
        }
        runtime->send_variant_request(owner_space, rez);
      }
      // Wait for the results
      wait_on.lg_wait();
      // Now we can re-take the lock and find our variant
      AutoLock t_lock(task_lock,1,false/*exclusive*/);
      std::map<VariantID,VariantImpl*>::const_iterator finder = 
        variants.find(variant_id);
      if (can_fail && (finder == variants.end()))
        return NULL;
#ifdef DEBUG_LEGION
      assert(finder != variants.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::find_valid_variants(std::vector<VariantID> &valid_variants,
                                       Processor::Kind kind) const
    //--------------------------------------------------------------------------
    {
      if (kind == Processor::NO_KIND)
      {
        AutoLock t_lock(task_lock,1,false/*exclusive*/);
        valid_variants.resize(variants.size());
        unsigned idx = 0;
        for (std::map<VariantID,VariantImpl*>::const_iterator it = 
              variants.begin(); it != variants.end(); it++, idx++)
        {
          valid_variants[idx] = it->first; 
        }
      }
      else
      {
        AutoLock t_lock(task_lock,1,false/*exclusive*/);
        for (std::map<VariantID,VariantImpl*>::const_iterator it = 
              variants.begin(); it != variants.end(); it++)
        {
          if (kind == it->second->get_processor_kind(false/*warn*/))
            valid_variants.push_back(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    const char* TaskImpl::get_name(bool needs_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        // Do the request through the semantic information
        const void *result = NULL; size_t dummy_size;
        if (retrieve_semantic_information(NAME_SEMANTIC_TAG, result, dummy_size,
                                          true/*can fail*/,false/*wait until*/))
          return reinterpret_cast<const char*>(result);
      }
      else
      {
        // If we're already holding the lock then we can just do
        // the local look-up regardless of if we're the owner or not
        std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
          semantic_infos.find(NAME_SEMANTIC_TAG);
        if (finder != semantic_infos.end())
          return reinterpret_cast<const char*>(finder->second.buffer);
      }
      // Couldn't find it so use the initial name
      return initial_name;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::attach_semantic_information(SemanticTag tag,
                                               AddressSpaceID source,
                                               const void *buffer, size_t size,
                                            bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      if ((tag == NAME_SEMANTIC_TAG) && (runtime->profiler != NULL))
        runtime->profiler->register_task_kind(task_id,(const char*)buffer,true);

      void *local = legion_malloc(SEMANTIC_INFO_ALLOC, size);
      memcpy(local, buffer, size);
      bool added = true;
      RtUserEvent to_trigger;
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
                REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                              "Inconsistent Semantic Tag value "
                              "for tag %ld with different sizes of %zd"
                              " and %zd for task impl", 
                              tag, size, finder->second.size)
              // Otherwise do a bitwise comparison
              {
                const char *orig = (const char*)finder->second.buffer;
                const char *next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                    REPORT_LEGION_ERROR(ERROR_INCONSISTENT_SEMANTIC_TAG,
                         "Inconsistent Semantic Tag value "
                                     "for tag %ld with different values at"
                                     "byte %d for task impl, %x != %x",
                                     tag, idx, orig[idx], next[idx])
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
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer = local;
            finder->second.size = size;
            to_trigger = finder->second.ready_event;
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_infos[tag] = SemanticInfo(local, size, is_mutable);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      if (added)
      {
        if (send_to_owner)
        {
          AddressSpaceID owner_space = get_owner_space();
          // if we are not the owner and the message didn't come
          // from the owner, then send it
          if ((owner_space != runtime->address_space) && 
              (source != owner_space))
          {
            if (tag == NAME_SEMANTIC_TAG)
            {
              // Special case here for task names, the user can reasonably
              // expect all tasks to have an initial name so we have to 
              // guarantee that this update is propagated before continuing
              // because otherwise we can't distinguish the case where a 
              // name hasn't propagated from one where it was never set
              RtUserEvent wait_on = Runtime::create_rt_user_event();
              send_semantic_info(owner_space, tag, buffer, size, 
                                 is_mutable, wait_on);
              wait_on.lg_wait();
            }
            else
              send_semantic_info(owner_space, tag, buffer, size, is_mutable);
          }
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
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != runtime->address_space);
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
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else // can use the canonical event
              wait_on = finder->second.ready_event; 
          }
          else if (wait_until) // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_infos[tag] = SemanticInfo(request);
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
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG, 
                      "Invalid semantic tag %ld for task implementation", tag)
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
          send_semantic_request(owner_space, tag, can_fail, wait_until,request);
        wait_on.lg_wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock t_lock(task_lock,1,false/*exclusive*/);
      std::map<SemanticTag,SemanticInfo>::const_iterator finder = 
        semantic_infos.find(tag);
      if (finder == semantic_infos.end())
      {
        if (can_fail)
          return false;
        REPORT_LEGION_ERROR(ERROR_INVALID_SEMANTIC_TAG, 
            "invalid semantic tag %ld for task implementation", tag)
      }
      result = finder->second.buffer;
      size = finder->second.size;
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_info(AddressSpaceID target, SemanticTag tag,
                                      const void *buffer, size_t size, 
                                      bool is_mutable, RtUserEvent to_trigger)
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
        rez.serialize(to_trigger);
      }
      runtime->send_task_impl_semantic_info(target, rez);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_request(AddressSpaceID target, 
             SemanticTag tag, bool can_fail, bool wait_until, RtUserEvent ready)
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
       AddressSpaceID target, bool can_fail, bool wait_until, RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(get_owner_space() == runtime->address_space);
#endif
      RtEvent precondition;
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
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_infos[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == NULL)
      {
        // this will cause a failure on the original node
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);  
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args;
          args.proxy_this = this;
          args.tag = tag;
          args.source = target;
          runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                           NULL/*op*/, precondition);
        }
      }
      else
        send_semantic_info(target, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_semantic_request(Runtime *runtime,
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
      RtUserEvent ready;
      derez.deserialize(ready);
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      impl->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_semantic_info(Runtime *runtime,
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
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      impl->attach_semantic_information(tag, source, buffer, size, 
                                        is_mutable, false/*send to owner*/);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID TaskImpl::get_owner_space(TaskID task_id,
                                                        Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (task_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskImpl::handle_variant_request(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      TaskID task_id;
      derez.deserialize(task_id);
      VariantID variant_id;
      derez.deserialize(variant_id);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool can_fail;
      derez.deserialize(can_fail);
      TaskImpl *task_impl = runtime->find_task_impl(task_id);
      VariantImpl *var_impl = task_impl->find_variant_impl(variant_id,can_fail);
      // If we can fail and there is no variant, just trigger the event
      if (can_fail && (var_impl == NULL))
        Runtime::trigger_event(done_event);
      else
        var_impl->send_variant_response(source, done_event);
    }

    /////////////////////////////////////////////////////////////
    // Variant Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VariantImpl::VariantImpl(Runtime *rt, VariantID v, TaskImpl *own, 
                           const TaskVariantRegistrar &registrar, bool ret,
                           CodeDescriptor *realm,
                           const void *udata /*=NULL*/, size_t udata_size/*=0*/)
      : vid(v), owner(own), runtime(rt), global(registrar.global_registration),
        has_return_value(ret), 
        descriptor_id(runtime->get_unique_code_descriptor_id()),
        realm_descriptor(realm),
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
      if (!Runtime::separate_runtime_instances)
      {
        Realm::ProfilingRequestSet profiling_requests;
        ready_event = ApEvent(Processor::register_task_by_kind(
            get_processor_kind(true), false/*global*/, descriptor_id, 
            *realm_descriptor, profiling_requests, user_data, user_data_size));
      }
      else
      {
        // This is a debug case for when we have one runtime instance
        // for each processor
        Processor proc = Processor::get_executing_processor();
        Realm::ProfilingRequestSet profiling_requests;
        ready_event = ApEvent(proc.register_task(descriptor_id,
            *realm_descriptor, profiling_requests, user_data, user_data_size));
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
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_GLOBAL_VARIANT_REGISTRATION, 
             "Variant %s requested global registration without "
                         "a portable implementation.", variant_name)
      if (leaf_variant && inner_variant)
        REPORT_LEGION_ERROR(ERROR_INNER_LEAF_MISMATCH, 
                      "Task variant %s (ID %ld) of task %s (ID %d) is not "
                      "permitted to be both inner and leaf tasks "
                      "simultaneously.", variant_name, vid,
                      owner->get_name(), owner->task_id)
      if (Runtime::record_registration)
        log_run.print("Task variant %s of task %s (ID %d) has Realm ID %ld",
              variant_name, owner->get_name(), owner->task_id, descriptor_id);
    }

    //--------------------------------------------------------------------------
    VariantImpl::VariantImpl(const VariantImpl &rhs) 
      : vid(rhs.vid), owner(rhs.owner), runtime(rhs.runtime), 
        global(rhs.global), has_return_value(rhs.has_return_value),
        descriptor_id(rhs.descriptor_id), realm_descriptor(rhs.realm_descriptor) 
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
    bool VariantImpl::is_no_access_region(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      bool result = false;
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.lower_bound(idx); it !=
            layout_constraints.layouts.upper_bound(idx); it++)
      {
        result = true;
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(it->second);
        if (!constraints->specialized_constraint.is_no_access())
        {
          result = false;
          break;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent VariantImpl::dispatch_task(Processor target, SingleTask *task,
                                       TaskContext *ctx, ApEvent precondition,
                                       PredEvent predicate_guard, int priority, 
                                       Realm::ProfilingRequestSet &requests)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Either it is local or it is a group that we made
      assert(runtime->is_local(target) || 
              (target.kind() == Processor::PROC_GROUP));
#endif
      // Add any profiling requests
      if (runtime->profiler != NULL)
        runtime->profiler->add_task_request(requests, owner->task_id, vid,task);
      // Increment the number of outstanding tasks
#ifdef DEBUG_LEGION
      runtime->increment_total_outstanding_tasks(task->task_id, false/*meta*/);
#else
      runtime->increment_total_outstanding_tasks();
#endif
      DETAILED_PROFILER(runtime, REALM_SPAWN_TASK_CALL);
      // If our ready event hasn't triggered, include it in the precondition
      if (predicate_guard.exists())
      {
        // Merge in the predicate guard
        ApEvent pre = Runtime::merge_events(precondition, ready_event, 
                                            ApEvent(predicate_guard));
        // Have to protect the result in case it misspeculates
        return Runtime::ignorefaults(target.spawn(descriptor_id, 
                    &ctx, sizeof(ctx), requests, pre, priority));
      }
      else
      {
        // No predicate guard
        if (!ready_event.has_triggered())
          return ApEvent(target.spawn(descriptor_id, &ctx, sizeof(ctx),requests,
                   Runtime::merge_events(precondition, ready_event), priority));
        return ApEvent(target.spawn(descriptor_id, &ctx, sizeof(ctx), requests, 
                                    precondition, priority));
      }
    }

    //--------------------------------------------------------------------------
    void VariantImpl::dispatch_inline(Processor current, InlineContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(realm_descriptor != NULL);
#endif
      const Realm::FunctionPointerImplementation *fp_impl = 
        realm_descriptor->find_impl<Realm::FunctionPointerImplementation>();
#ifdef DEBUG_LEGION
      assert(fp_impl != NULL);
#endif
      RealmFnptr inline_ptr = fp_impl->get_impl<RealmFnptr>();
      (*inline_ptr)(&ctx, sizeof(ctx), user_data, user_data_size, current);
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
        REPORT_LEGION_WARNING(LEGION_WARNING_MISSING_PROC_CONSTRAINT, 
           "NO PROCESSOR CONSTRAINT SPECIFIED FOR VARIANT"
                        " %s (ID %ld) OF TASK %s (ID %d)! ASSUMING LOC_PROC!",
                      variant_name, vid, owner->get_name(false),owner->task_id)
      return Processor::LOC_PROC;
    }

    //--------------------------------------------------------------------------
    void VariantImpl::send_variant_response(AddressSpaceID target, 
                                            RtUserEvent done)
    //--------------------------------------------------------------------------
    {
      if (!global)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_USE_OF_NON_GLOBAL_VARIANT,
                      "Illegal remote use of variant %s of task %s "
                      "which was not globally registered.",
                      variant_name, owner->get_name())
      // Package up this variant and send it over to the target 
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(owner->task_id);
        rez.serialize(vid);
        rez.serialize(done);
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
      runtime->send_variant_response(target, rez);
    }

    //--------------------------------------------------------------------------
    void VariantImpl::broadcast_variant(RtUserEvent done, AddressSpaceID origin,
                                        AddressSpaceID local)
    //--------------------------------------------------------------------------
    {
      std::vector<AddressSpaceID> targets;
      std::vector<AddressSpaceID> locals;
      const AddressSpaceID start = local * Runtime::legion_collective_radix + 1;
      for (int idx = 0; idx < Runtime::legion_collective_radix; idx++)
      {
        AddressSpaceID next = start+idx;
        if (next >= runtime->total_address_spaces)
          break;
        locals.push_back(next);
        // Convert from relative to actual address space
        AddressSpaceID actual = (origin + next) % runtime->total_address_spaces;
        targets.push_back(actual);
      }
      if (!targets.empty())
      {
        std::set<RtEvent> local_done;
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          RtUserEvent next_done = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(owner->task_id);
            rez.serialize(vid);
            rez.serialize(next_done);
            rez.serialize(origin);
            rez.serialize(locals[idx]);
          }
          runtime->send_variant_broadcast(targets[idx], rez);
          local_done.insert(next_done);
        }
        Runtime::trigger_event(done, Runtime::merge_events(local_done));
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VariantImpl::handle_variant_broadcast(Runtime *runtime,
                                                          Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID tid;
      derez.deserialize(tid);
      VariantID vid;
      derez.deserialize(vid);
      RtUserEvent done;
      derez.deserialize(done);
      AddressSpaceID origin;
      derez.deserialize(origin);
      AddressSpaceID local;
      derez.deserialize(local);
      VariantImpl *impl = runtime->find_variant_impl(tid, vid);
      impl->broadcast_variant(done, origin, local);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID VariantImpl::get_owner_space(VariantID vid,
                                                           Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (vid % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    /*static*/ void VariantImpl::handle_variant_response(Runtime *runtime,
                                                         Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      TaskVariantRegistrar registrar(task_id, false/*global*/);
      VariantID variant_id;
      derez.deserialize(variant_id);
      RtUserEvent done;
      derez.deserialize(done);
      bool has_return;
      derez.deserialize(has_return);
      size_t impl_size;
      derez.deserialize(impl_size);
      CodeDescriptor *realm_desc = new CodeDescriptor();
      {
        // Realm's serializers assume properly aligned buffers, so
        // malloc a temporary buffer here and copy the data to ensure
        // alignment.
        void *impl_buffer = malloc(impl_size);
#ifdef DEBUG_LEGION
        assert(impl_buffer);
#endif
        memcpy(impl_buffer, derez.get_current_pointer(), impl_size);
        derez.advance_pointer(impl_size);
        Realm::Serialization::FixedBufferDeserializer
          deserializer(impl_buffer, impl_size);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        bool ok =
#endif
                  realm_desc->deserialize(deserializer);
        assert(ok);
#else
        realm_desc->deserialize(deserializer);
#endif
        free(impl_buffer);
      }
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
      Runtime::trigger_event(done);
    }

    /////////////////////////////////////////////////////////////
    // Layout Constraints 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(LayoutConstraintID lay_id,FieldSpace h,
                                         Runtime *rt, AddressSpaceID owner, 
                                         AddressSpaceID local)
      : LayoutConstraintSet(), Collectable(), layout_id(lay_id), handle(h), 
        owner_space(owner), local_space(local), runtime(rt), 
        constraints_name(NULL) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(LayoutConstraintID lay_id, Runtime *rt,
                                     const LayoutConstraintRegistrar &registrar)
      : LayoutConstraintSet(registrar.layout_constraints), Collectable(),
        layout_id(lay_id), handle(registrar.handle), 
        owner_space(rt->address_space), local_space(rt->address_space),
        runtime(rt)
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
    LayoutConstraints::LayoutConstraints(LayoutConstraintID lay_id, Runtime *rt,
                                         const LayoutConstraintSet &cons,
                                         FieldSpace h)
      : LayoutConstraintSet(cons), Collectable(), layout_id(lay_id), handle(h),
        owner_space(rt->address_space), local_space(rt->address_space), 
        runtime(rt)
    //--------------------------------------------------------------------------
    {
      constraints_name = (char*)malloc(64*sizeof(char));
      snprintf(constraints_name,64,"layout constraints %ld", layout_id);
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::LayoutConstraints(const LayoutConstraints &rhs)
      : LayoutConstraintSet(rhs), Collectable(), layout_id(rhs.layout_id), 
        handle(rhs.handle), owner_space(0), local_space(0), runtime(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::RemoveFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      runtime->send_constraint_removal(target, rez);
    }

    //--------------------------------------------------------------------------
    LayoutConstraints::~LayoutConstraints(void)
    //--------------------------------------------------------------------------
    {
      if (constraints_name != NULL)
        free(constraints_name);
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
    void LayoutConstraints::send_constraint_response(AddressSpaceID target,
                                                     RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
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
      // Hold our lock when updating our se of remote instances
      AutoLock lay_lock(layout_lock);
      remote_instances.add(target);
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::update_constraints(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(constraints_name == NULL);
#endif
      size_t name_len;
      derez.deserialize(name_len);
      constraints_name = (char*)malloc(name_len);
      derez.deserialize(constraints_name, name_len);
      // unpack the constraints
      deserialize(derez); 
    }

    //--------------------------------------------------------------------------
    void LayoutConstraints::release_remote_instances(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(layout_id);
      }
      RemoveFunctor functor(rez, runtime);
      remote_instances.map(functor);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails(LayoutConstraints *constraints,
                                    unsigned total_dims)
    //--------------------------------------------------------------------------
    {
      // Check to see if the result is in the cache
      {
        AutoLock lay(layout_lock,1,false/*exclusive*/);
        std::map<LayoutConstraintID,bool>::const_iterator finder = 
          entailment_cache.find(constraints->layout_id);
        if (finder != entailment_cache.end())
          return finder->second;
      }
      // Didn't find it, so do the test for real
      bool result = entails(*constraints, total_dims);
      // Save the result in the cache
      AutoLock lay(layout_lock);
      entailment_cache[constraints->layout_id] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails(const LayoutConstraintSet &other,
                                    unsigned total_dims) const
    //--------------------------------------------------------------------------
    {
      return LayoutConstraintSet::entails(other, total_dims);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::conflicts(LayoutConstraints *constraints,
                                      unsigned total_dims)
    //--------------------------------------------------------------------------
    {
      // Check to see if the result is in the cache
      {
        AutoLock lay(layout_lock,1,false/*exclusive*/);
        std::map<LayoutConstraintID,bool>::const_iterator finder = 
          conflict_cache.find(constraints->layout_id);
        if (finder != conflict_cache.end())
          return finder->second;
      }
      // Didn't find it, so do the test for real
      bool result = conflicts(*constraints, total_dims);
      // Save the result in the cache
      AutoLock lay(layout_lock);
      conflict_cache[constraints->layout_id] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::conflicts(const LayoutConstraintSet &other,
                                      unsigned total_dims) const
    //--------------------------------------------------------------------------
    {
      return LayoutConstraintSet::conflicts(other, total_dims);
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails_without_pointer(
                            LayoutConstraints *constraints, unsigned total_dims)
    //--------------------------------------------------------------------------
    {
      // See if we have it in the cache
      {
        AutoLock lay(layout_lock,1,false/*exclusive*/);
        std::map<LayoutConstraintID,bool>::const_iterator finder = 
          no_pointer_entailment_cache.find(constraints->layout_id);
        if (finder != no_pointer_entailment_cache.end())
          return finder->second;
      }
      // Didn't find it so do the test for real
      bool result = entails_without_pointer(*constraints, total_dims);
      // Save the result in the cache
      AutoLock lay(layout_lock);
      no_pointer_entailment_cache[constraints->layout_id] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraints::entails_without_pointer(
                    const LayoutConstraintSet &other, unsigned total_dims) const
    //--------------------------------------------------------------------------
    {
      // Do all the normal entailment but don't check the pointer constraint 
      if (!specialized_constraint.entails(other.specialized_constraint))
        return false;
      if (!field_constraint.entails(other.field_constraint))
        return false;
      if (!memory_constraint.entails(other.memory_constraint))
        return false;
      if (!ordering_constraint.entails(other.ordering_constraint, total_dims))
        return false;
      for (std::vector<SplittingConstraint>::const_iterator it = 
            other.splitting_constraints.begin(); it !=
            other.splitting_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < splitting_constraints.size(); idx++)
        {
          if (splitting_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<DimensionConstraint>::const_iterator it = 
            other.dimension_constraints.begin(); it != 
            other.dimension_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < dimension_constraints.size(); idx++)
        {
          if (dimension_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<AlignmentConstraint>::const_iterator it = 
            other.alignment_constraints.begin(); it != 
            other.alignment_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < alignment_constraints.size(); idx++)
        {
          if (alignment_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<OffsetConstraint>::const_iterator it = 
            other.offset_constraints.begin(); it != 
            other.offset_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < offset_constraints.size(); idx++)
        {
          if (offset_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID LayoutConstraints::get_owner_space(
                            LayoutConstraintID layout_id, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      return (layout_id % runtime->runtime_stride);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LayoutConstraints::process_request(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool can_fail;
      derez.deserialize(can_fail);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(lay_id, can_fail);
      if (can_fail && (constraints == NULL))
        Runtime::trigger_event(done_event);
      else
        constraints->send_constraint_response(source, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID LayoutConstraints::process_response(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID lay_id;
      derez.deserialize(lay_id);
      FieldSpace handle;
      derez.deserialize(handle);
      // Make it an unpack it, then try to register it 
      LayoutConstraints *new_constraints = 
        new LayoutConstraints(lay_id, handle, runtime,
                              source, runtime->address_space);
      new_constraints->update_constraints(derez);
      if (!runtime->register_layout(new_constraints))
        delete (new_constraints);
      // Now try to register this with the runtime
      // Trigger our done event and then return it
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
      return lay_id;
    }

    /////////////////////////////////////////////////////////////
    // Identity Projection Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IdentityProjectionFunctor::IdentityProjectionFunctor(Legion::Runtime *rt)
      : ProjectionFunctor(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IdentityProjectionFunctor::~IdentityProjectionFunctor(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(const Mappable *mappable,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      return upper_bound;
    }
    
    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(const Mappable *mappable,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(upper_bound, point);
    }

    //--------------------------------------------------------------------------
    bool IdentityProjectionFunctor::is_exclusive(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    unsigned IdentityProjectionFunctor::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return 0;
    }

    /////////////////////////////////////////////////////////////
    // Projection Function 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionFunction::ProjectionFunction(ProjectionID pid, 
                                           ProjectionFunctor *func)
      : depth(func->get_depth()), is_exclusive(func->is_exclusive()),
        projection_id(pid), functor(func)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionFunction::ProjectionFunction(const ProjectionFunction &rhs)
      : depth(rhs.depth), is_exclusive(rhs.is_exclusive), 
        projection_id(rhs.projection_id), functor(rhs.functor)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ProjectionFunction::~ProjectionFunction(void)
    //--------------------------------------------------------------------------
    {
      // These can be shared in the case of multiple runtime instances
      if (!Runtime::separate_runtime_instances)
        delete functor;
    }

    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunction::project_point(Task *task, unsigned idx, 
                                     Runtime *runtime, const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = task->regions[idx];
#ifdef DEBUG_LEGION
      assert(req.handle_type != SINGULAR);
#endif
      // It's actually unsafe to evaluate projection region requirements
      // with NO_ACCESS since they can race with deletion operations for
      // the region requirement as NO_ACCESS region requirements aren't
      // recorded in the region tree
      if (req.privilege == NO_ACCESS)
        return LogicalRegion::NO_REGION;
      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == PART_PROJECTION)
        {
          LogicalRegion result = functor->project(task, idx, 
                                                  req.partition, point); 
          check_projection_partition_result(req, task, idx, result, runtime);
          return result;
        }
        else
        {
          LogicalRegion result = functor->project(task, idx, req.region, point);
          check_projection_region_result(req, task, idx, result, runtime);
          return result;
        }
      }
      else
      {
        if (req.handle_type == PART_PROJECTION)
        {
          LogicalRegion result = functor->project(task, idx, 
                                                  req.partition, point);
          check_projection_partition_result(req, task, idx, result, runtime);
          return result;
        }
        else
        {
          LogicalRegion result = functor->project(task, idx, req.region, point);
          check_projection_region_result(req, task, idx, result, runtime);
          return result;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::project_points(const RegionRequirement &req, 
     unsigned idx, Runtime *runtime, const std::vector<PointTask*> &point_tasks)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(req.handle_type != SINGULAR);
#endif
      // It's actually unsafe to evaluate projection region requirements
      // with NO_ACCESS since they can race with deletion operations for
      // the region requirement as NO_ACCESS region requirements aren't
      // recorded in the region tree
      if (req.privilege == NO_ACCESS)
      {
        for (std::vector<PointTask*>::const_iterator it =
              point_tasks.begin(); it != point_tasks.end(); it++)
        {
          (*it)->set_projection_result(idx, LogicalRegion::NO_REGION);
        }
        return;
      }

      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == PART_PROJECTION)
        {
          for (std::vector<PointTask*>::const_iterator it = 
                point_tasks.begin(); it != point_tasks.end(); it++)
          {
            LogicalRegion result = functor->project(*it, idx, req.partition, 
                                                    (*it)->get_domain_point());
            check_projection_partition_result(req, static_cast<Task*>(*it), 
                                              idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
        else
        {
          for (std::vector<PointTask*>::const_iterator it = 
                point_tasks.begin(); it != point_tasks.end(); it++)
          {
            LogicalRegion result = functor->project(*it, idx, req.region, 
                                                    (*it)->get_domain_point());
            check_projection_region_result(req, static_cast<Task*>(*it), 
                                           idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
      }
      else
      {
        if (req.handle_type == PART_PROJECTION)
        {
          for (std::vector<PointTask*>::const_iterator it = 
                point_tasks.begin(); it != point_tasks.end(); it++)
          {
            LogicalRegion result = functor->project(*it, idx, req.partition, 
                                                    (*it)->get_domain_point());
            check_projection_partition_result(req, static_cast<Task*>(*it), 
                                              idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
        else
        {
          for (std::vector<PointTask*>::const_iterator it = 
                point_tasks.begin(); it != point_tasks.end(); it++)
          {
            LogicalRegion result = functor->project(*it, idx, req.region, 
                                                    (*it)->get_domain_point());
            check_projection_region_result(req, static_cast<Task*>(*it), 
                                           idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::project_points(Operation *op, unsigned idx,
                         const RegionRequirement &req, Runtime *runtime, 
                         const std::vector<ProjectionPoint*> &points)
    //--------------------------------------------------------------------------
    {
      Mappable *mappable = op->get_mappable();
#ifdef DEBUG_LEGION
      assert(req.handle_type != SINGULAR);
      assert(mappable != NULL);
#endif
      // It's actually unsafe to evaluate projection region requirements
      // with NO_ACCESS since they can race with deletion operations for
      // the region requirement as NO_ACCESS region requirements aren't
      // recorded in the region tree
      if (req.privilege == NO_ACCESS)
      {
        for (std::vector<ProjectionPoint*>::const_iterator it =
              points.begin(); it != points.end(); it++)
        {
          (*it)->set_projection_result(idx, LogicalRegion::NO_REGION);
        }
        return;
      }

      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == PART_PROJECTION)
        {
          for (std::vector<ProjectionPoint*>::const_iterator it = 
                points.begin(); it != points.end(); it++)
          {
            LogicalRegion result = functor->project(mappable, idx, 
                req.partition, (*it)->get_domain_point());
            check_projection_partition_result(req, op, idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
        else
        {
          for (std::vector<ProjectionPoint*>::const_iterator it = 
                points.begin(); it != points.end(); it++)
          {
            LogicalRegion result = functor->project(mappable, idx, req.region,
                                                    (*it)->get_domain_point());
            check_projection_region_result(req, op, idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
      }
      else
      {
        if (req.handle_type == PART_PROJECTION)
        {
          for (std::vector<ProjectionPoint*>::const_iterator it = 
                points.begin(); it != points.end(); it++)
          {
            LogicalRegion result = functor->project(mappable, idx,
                req.partition, (*it)->get_domain_point());
            check_projection_partition_result(req, op, idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
        else
        {
          for (std::vector<ProjectionPoint*>::const_iterator it = 
                points.begin(); it != points.end(); it++)
          {
            LogicalRegion result = functor->project(mappable, idx, req.region,
                                                    (*it)->get_domain_point());
            check_projection_region_result(req, op, idx, result, runtime);
            (*it)->set_projection_result(idx, result);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_region_result(
        const RegionRequirement &req, const Task *task, unsigned idx,
        LogicalRegion result, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != req.region.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion of tree ID %d for region requirement %d "
            "of task %s (UID %lld) which is different from the upper "
            "bound node of tree ID %d", projection_id, 
            result.get_tree_id(), idx, task->get_task_name(), 
            task->get_unique_id(), req.region.get_tree_id())
#ifdef DEBUG_LEGION
      if (!runtime->forest->is_subregion(result, req.region))
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which is not a subregion of the "
            "upper bound region for region requirement %d of "
            "task %s (UID %lld)", projection_id, idx,
            task->get_task_name(), task->get_unique_id())
      const unsigned projection_depth = 
        runtime->forest->get_projection_depth(result, req.region);
      if (projection_depth != functor->get_depth())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which has projection depth %d which "
            "is different from stated projection depth of the functor "
            "which is %d for region requirement %d of task %s (ID %lld)",
            projection_id, projection_depth, functor->get_depth(),
            idx, task->get_task_name(), task->get_unique_id())
#endif
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_partition_result(
        const RegionRequirement &req, const Task *task, unsigned idx,
        LogicalRegion result, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != req.partition.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion of tree ID %d for region requirement %d "
            "of task %s (UID %lld) which is different from the upper "
            "bound node of tree ID %d", projection_id, 
            result.get_tree_id(), idx, task->get_task_name(), 
            task->get_unique_id(), req.partition.get_tree_id())
#ifdef DEBUG_LEGION
      if (!runtime->forest->is_subregion(result, req.partition))
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which is not a subregion of the "
            "upper bound region for region requirement %d of "
            "task %s (UID %lld)", projection_id, idx,
            task->get_task_name(), task->get_unique_id())
      const unsigned projection_depth = 
        runtime->forest->get_projection_depth(result, req.partition);
      if (projection_depth != functor->get_depth())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which has projection depth %d which "
            "is different from stated projection depth of the functor "
            "which is %d for region requirement %d of task %s (ID %lld)",
            projection_id, projection_depth, functor->get_depth(),
            idx, task->get_task_name(), task->get_unique_id())
#endif
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_region_result(
        const RegionRequirement &req, Operation *op, unsigned idx,
        LogicalRegion result, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != req.region.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion of tree ID %d for region requirement %d "
            "of operation %s (UID %lld) which is different from the upper "
            "bound node of tree ID %d", projection_id, 
            result.get_tree_id(), idx, op->get_logging_name(), 
            op->get_unique_op_id(), req.region.get_tree_id())
#ifdef DEBUG_LEGION
      if (!runtime->forest->is_subregion(result, req.region))
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which is not a subregion of the "
            "upper bound region for region requirement %d of "
            "operation %s (UID %lld)", projection_id, idx,
            op->get_logging_name(), op->get_unique_op_id())
      const unsigned projection_depth = 
        runtime->forest->get_projection_depth(result, req.region);
      if (projection_depth != functor->get_depth())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which has projection depth %d which "
            "is different from stated projection depth of the functor "
            "which is %d for region requirement %d of operation %s (ID %lld)",
            projection_id, projection_depth, functor->get_depth(),
            idx, op->get_logging_name(), op->get_unique_op_id())
#endif
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_partition_result(
        const RegionRequirement &req, Operation *op, unsigned idx,
        LogicalRegion result, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != req.partition.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion of tree ID %d for region requirement %d "
            "of operation %s (UID %lld) which is different from the upper "
            "bound node of tree ID %d", projection_id, 
            result.get_tree_id(), idx, op->get_logging_name(), 
            op->get_unique_op_id(), req.partition.get_tree_id())
#ifdef DEBUG_LEGION
      if (!runtime->forest->is_subregion(result, req.partition))
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which is not a subregion of the "
            "upper bound region for region requirement %d of "
            "operation %s (UID %lld)", projection_id, idx,
            op->get_logging_name(), op->get_unique_op_id())
      const unsigned projection_depth = 
        runtime->forest->get_projection_depth(result, req.partition);
      if (projection_depth != functor->get_depth())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_RESULT, 
            "Projection functor %d produced an invalid "
            "logical subregion which has projection depth %d which "
            "is different from stated projection depth of the functor "
            "which is %d for region requirement %d of operation %s (ID %lld)",
            projection_id, projection_depth, functor->get_depth(),
            idx, op->get_logging_name(), op->get_unique_op_id())
#endif
    }

    /////////////////////////////////////////////////////////////
    // Legion Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(Machine m, AddressSpaceID unique,
                     const std::set<Processor> &locals,
                     const std::set<Processor> &local_utilities,
                     const std::set<AddressSpaceID> &address_spaces,
                     const std::map<Processor,AddressSpaceID> &processor_spaces)
      : external(new Legion::Runtime(this)),
        mapper_runtime(new Legion::Mapping::MapperRuntime()),
        machine(m), address_space(unique), 
        total_address_spaces(address_spaces.size()),
        runtime_stride(address_spaces.size()), profiler(NULL),
        forest(new RegionTreeForest(this)), 
        has_explicit_utility_procs(!local_utilities.empty()), 
        prepared_for_shutdown(false),
        total_outstanding_tasks(0), outstanding_top_level_tasks(0), 
        local_procs(locals), local_utils(local_utilities),
        proc_spaces(processor_spaces),
        unique_index_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_partition_id((unique == 0) ? runtime_stride : unique), 
        unique_field_space_id((unique == 0) ? runtime_stride : unique),
        unique_index_tree_id((unique == 0) ? runtime_stride : unique),
        unique_region_tree_id((unique == 0) ? runtime_stride : unique),
        unique_operation_id((unique == 0) ? runtime_stride : unique),
        unique_field_id(MAX_APPLICATION_FIELD_ID + 
                        ((unique == 0) ? runtime_stride : unique)),
        unique_code_descriptor_id(TASK_ID_AVAILABLE +
                        ((unique == 0) ? runtime_stride : unique)),
        unique_constraint_id((unique == 0) ? runtime_stride : unique),
        unique_task_id(get_current_static_task_id()+unique),
        unique_mapper_id(get_current_static_mapper_id()+unique),
        unique_projection_id(get_current_static_projection_id()+unique),
        unique_distributed_id((unique == 0) ? runtime_stride : unique),
        gc_epoch_counter(0)
    //--------------------------------------------------------------------------
    {
      log_run.debug("Initializing high-level runtime in address space %x",
                            address_space);
      // Construct a local utility processor group
      if (local_utils.empty())
      {
        // make the utility group the set of all the local processors
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(utility_group.exists());
#endif
      Machine::ProcessorQuery all_procs(machine); 
      // For each of the processors in our local set construct a manager
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert((*it).kind() != Processor::UTIL_PROC);
#endif
        ProcessorManager *manager = new ProcessorManager(*it,
				    (*it).kind(), this,
                                    DEFAULT_MAPPER_SLOTS, 
                                    stealing_disabled,
                                    (replay_file != NULL));
        proc_managers[*it] = manager;
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
#ifdef DEBUG_LEGION
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
      allocation_tracing_count = 0;
      // Instantiate all the kinds of allocations
      for (unsigned idx = ARGUMENT_MAP_ALLOC; idx < LAST_ALLOC; idx++)
        allocation_manager[((AllocationType)idx)] = AllocationTracker();
#endif
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
      outstanding_counts.resize(LG_LAST_TASK_ID, 0);
#endif

      // Attach any accessor debug hooks for privilege or bounds checks
#ifdef PRIVILEGE_CHECKS
      LegionRuntime::Accessor::DebugHooks::find_privilege_task_name =
	&Legion::Internal::Runtime::find_privilege_task_name;
#endif
#ifdef BOUNDS_CHECKS
      LegionRuntime::Accessor::DebugHooks::check_bounds_ptr =
	&Legion::Internal::Runtime::check_bounds;
      LegionRuntime::Accessor::DebugHooks::check_bounds_dpoint =
	&Legion::Internal::Runtime::check_bounds;
#endif
    }

    //--------------------------------------------------------------------------
    Runtime::Runtime(const Runtime &rhs)
      : external(NULL), mapper_runtime(NULL), machine(rhs.machine), 
        address_space(0), total_address_spaces(0), runtime_stride(0), 
        profiler(NULL), forest(NULL), has_explicit_utility_procs(false),
        local_procs(rhs.local_procs), proc_spaces(rhs.proc_spaces)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Runtime::~Runtime(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we don't send anymore messages
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
      {
        if (message_managers[idx] != NULL)
        {
          delete message_managers[idx];
          message_managers[idx] = NULL;
        }
      }
      if (profiler != NULL)
      {
        delete profiler;
        profiler = NULL;
      }
      delete forest;
      delete external;
      delete mapper_runtime;
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        delete it->second;
      }
      proc_managers.clear();
      // Avoid duplicate deletions on these for separate runtime
      // instances by just leaking them for now
      if (!separate_runtime_instances)
      {
        for (std::map<ProjectionID,ProjectionFunction*>::
              iterator it = projection_functions.begin(); 
              it != projection_functions.end(); it++)
        {
          delete it->second;
        } 
        projection_functions.clear();
      }
      for (std::deque<IndividualTask*>::const_iterator it = 
            available_individual_tasks.begin(); 
            it != available_individual_tasks.end(); it++)
      {
        delete (*it);
      }
      available_individual_tasks.clear();
      for (std::deque<PointTask*>::const_iterator it = 
            available_point_tasks.begin(); it != 
            available_point_tasks.end(); it++)
      {
        delete (*it);
      }
      available_point_tasks.clear();
      for (std::deque<IndexTask*>::const_iterator it = 
            available_index_tasks.begin(); it != 
            available_index_tasks.end(); it++)
      {
        delete (*it);
      }
      available_index_tasks.clear();
      for (std::deque<SliceTask*>::const_iterator it = 
            available_slice_tasks.begin(); it != 
            available_slice_tasks.end(); it++)
      {
        delete (*it);
      }
      available_slice_tasks.clear();
      for (std::deque<MapOp*>::const_iterator it = 
            available_map_ops.begin(); it != 
            available_map_ops.end(); it++)
      {
        delete (*it);
      }
      available_map_ops.clear();
      for (std::deque<CopyOp*>::const_iterator it = 
            available_copy_ops.begin(); it != 
            available_copy_ops.end(); it++)
      {
        delete (*it);
      }
      available_copy_ops.clear();
      for (std::deque<FenceOp*>::const_iterator it = 
            available_fence_ops.begin(); it != 
            available_fence_ops.end(); it++)
      {
        delete (*it);
      }
      available_fence_ops.clear();
      for (std::deque<FrameOp*>::const_iterator it = 
            available_frame_ops.begin(); it !=
            available_frame_ops.end(); it++)
      {
        delete (*it);
      }
      available_frame_ops.clear();
      for (std::deque<DeletionOp*>::const_iterator it = 
            available_deletion_ops.begin(); it != 
            available_deletion_ops.end(); it++)
      {
        delete (*it);
      }
      available_deletion_ops.clear();
      for (std::deque<OpenOp*>::const_iterator it = 
            available_open_ops.begin(); it !=
            available_open_ops.end(); it++)
      {
        delete (*it);
      }
      available_open_ops.clear();
      for (std::deque<AdvanceOp*>::const_iterator it = 
            available_advance_ops.begin(); it !=
            available_advance_ops.end(); it++)
      {
        delete (*it);
      }
      available_advance_ops.clear();
      for (std::deque<InterCloseOp*>::const_iterator it = 
            available_inter_close_ops.begin(); it !=
            available_inter_close_ops.end(); it++)
      {
        delete (*it);
      }
      available_inter_close_ops.clear();
      for (std::deque<ReadCloseOp*>::const_iterator it = 
            available_read_close_ops.begin(); it != 
            available_read_close_ops.end(); it++)
      {
        delete (*it);
      }
      for (std::deque<PostCloseOp*>::const_iterator it = 
            available_post_close_ops.begin(); it !=
            available_post_close_ops.end(); it++)
      {
        delete (*it);
      }
      available_post_close_ops.clear();
      for (std::deque<VirtualCloseOp*>::const_iterator it = 
            available_virtual_close_ops.begin(); it !=
            available_virtual_close_ops.end(); it++)
      {
        delete (*it);
      }
      available_virtual_close_ops.clear();
      for (std::deque<DynamicCollectiveOp*>::const_iterator it = 
            available_dynamic_collective_ops.begin(); it !=
            available_dynamic_collective_ops.end(); it++)
      {
        delete (*it);
      }
      available_dynamic_collective_ops.end();
      for (std::deque<FuturePredOp*>::const_iterator it = 
            available_future_pred_ops.begin(); it !=
            available_future_pred_ops.end(); it++)
      {
        delete (*it);
      }
      available_future_pred_ops.clear();
      for (std::deque<NotPredOp*>::const_iterator it = 
            available_not_pred_ops.begin(); it !=
            available_not_pred_ops.end(); it++)
      {
        delete (*it);
      }
      available_not_pred_ops.clear();
      for (std::deque<AndPredOp*>::const_iterator it = 
            available_and_pred_ops.begin(); it !=
            available_and_pred_ops.end(); it++)
      {
        delete (*it);
      }
      available_and_pred_ops.clear();
      for (std::deque<OrPredOp*>::const_iterator it = 
            available_or_pred_ops.begin(); it !=
            available_or_pred_ops.end(); it++)
      {
        delete (*it);
      }
      available_or_pred_ops.clear();
      for (std::deque<AcquireOp*>::const_iterator it = 
            available_acquire_ops.begin(); it !=
            available_acquire_ops.end(); it++)
      {
        delete (*it);
      }
      available_acquire_ops.clear();
      for (std::deque<ReleaseOp*>::const_iterator it = 
            available_release_ops.begin(); it !=
            available_release_ops.end(); it++)
      {
        delete (*it);
      }
      available_release_ops.clear();
      for (std::deque<TraceCaptureOp*>::const_iterator it = 
            available_capture_ops.begin(); it !=
            available_capture_ops.end(); it++)
      {
        delete (*it);
      }
      available_capture_ops.clear();
      for (std::deque<TraceCompleteOp*>::const_iterator it = 
            available_trace_ops.begin(); it !=
            available_trace_ops.end(); it++)
      {
        delete (*it);
      }
      available_trace_ops.clear();
      for (std::deque<MustEpochOp*>::const_iterator it = 
            available_epoch_ops.begin(); it !=
            available_epoch_ops.end(); it++)
      {
        delete (*it);
      }
      available_epoch_ops.clear();
      for (std::deque<PendingPartitionOp*>::const_iterator it = 
            available_pending_partition_ops.begin(); it !=
            available_pending_partition_ops.end(); it++)
      {
        delete (*it);
      }
      available_pending_partition_ops.clear();
      for (std::deque<DependentPartitionOp*>::const_iterator it = 
            available_dependent_partition_ops.begin(); it !=
            available_dependent_partition_ops.end(); it++)
      {
        delete (*it);
      }
      available_dependent_partition_ops.clear();
      for (std::deque<FillOp*>::const_iterator it = 
            available_fill_ops.begin(); it !=
            available_fill_ops.end(); it++)
      {
        delete (*it);
      }
      available_fill_ops.clear();
      for (std::deque<AttachOp*>::const_iterator it = 
            available_attach_ops.begin(); it !=
            available_attach_ops.end(); it++)
      {
        delete (*it);
      }
      available_attach_ops.clear();
      for (std::deque<DetachOp*>::const_iterator it = 
            available_detach_ops.begin(); it !=
            available_detach_ops.end(); it++)
      {
        delete (*it);
      }
      available_detach_ops.clear();
      for (std::deque<TimingOp*>::const_iterator it = 
            available_timing_ops.begin(); it != 
            available_timing_ops.end(); it++)
      {
        delete (*it);
      }
      available_timing_ops.clear();
      for (std::map<TaskID,TaskImpl*>::const_iterator it = 
            task_table.begin(); it != task_table.end(); it++)
      {
        delete (it->second);
      }
      task_table.clear();
      // Skip this if we are in separate runtime mode
      if (!Runtime::separate_runtime_instances)
      {
        for (std::deque<VariantImpl*>::const_iterator it = 
              variant_table.begin(); it != variant_table.end(); it++)
        {
          delete (*it);
        }
      }
      variant_table.clear();
      // Skip this if we are in separate runtime mode
      if (!Runtime::separate_runtime_instances)
      {
        while (!layout_constraints_table.empty())
        {
          std::map<LayoutConstraintID,LayoutConstraints*>::iterator next_it = 
            layout_constraints_table.begin();
          LayoutConstraints *next = next_it->second;
          layout_constraints_table.erase(next_it);
          if (next->remove_reference())
            delete (next);
        }
      }
      for (std::map<Memory,MemoryManager*>::const_iterator it =
            memory_managers.begin(); it != memory_managers.end(); it++)
      {
        delete it->second;
      }
      memory_managers.clear();
#ifdef DEBUG_LEGION
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
    void Runtime::register_static_variants(void)
    //--------------------------------------------------------------------------
    {
      std::deque<PendingVariantRegistration*> &pending_variants = 
        get_pending_variant_table();
      if (!pending_variants.empty())
      {
        for (std::deque<PendingVariantRegistration*>::const_iterator it =
              pending_variants.begin(); it != pending_variants.end(); it++)
        {
          (*it)->perform_registration(this);
          // avoid races on seaparte runtime instances
          if (!Runtime::separate_runtime_instances)
            delete *it;
        }
        // avoid races on separate runtime instances
        if (!Runtime::separate_runtime_instances)
          pending_variants.clear();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_constraints(void)
    //--------------------------------------------------------------------------
    {
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
        if (!Runtime::separate_runtime_instances)
          pending_constraints.clear();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_projections(void)
    //--------------------------------------------------------------------------
    {
      std::map<ProjectionID,ProjectionFunctor*> &pending_projection_functors =
        get_pending_projection_table();
      for (std::map<ProjectionID,ProjectionFunctor*>::const_iterator it =
            pending_projection_functors.begin(); it !=
            pending_projection_functors.end(); it++)
      {
        it->second->set_runtime(external);
        register_projection_functor(it->first, it->second, true/*need check*/,
                                    true/*was preregistered*/);
      }
      register_projection_functor(0, 
          new IdentityProjectionFunctor(this->external), false/*need check*/,
                                        true/*was preregistered*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::initialize_legion_prof(void)
    //--------------------------------------------------------------------------
    {
      LG_TASK_DESCRIPTIONS(lg_task_descriptions);
      // For the profiler we want to find as many "holes" in the execution
      // as possible in which to run profiler tasks so we can minimize the
      // overhead on the application. To do this we want profiler tasks to
      // run on any processor that has a dedicated core which is either any
      // CPU processor a utility processor. There's no need to use GPU or
      // I/O processors since they share the same cores as the utility cores. 
      Machine::ProcessorQuery local_prof_procs(machine);
      local_prof_procs.local_address_space();
      std::vector<Processor> prof_procs;
      for (Machine::ProcessorQuery::iterator it = local_prof_procs.begin();
            it != local_prof_procs.end(); it++)
      {
        if (it->kind() == Processor::LOC_PROC)
          prof_procs.push_back(*it);
        else if (it->kind() == Processor::UTIL_PROC)
          prof_procs.push_back(*it);
      }
#ifdef DEBUG_LEGION
      assert(!prof_procs.empty());
#endif
      const Processor target_proc_for_profiler = prof_procs.size() > 1 ?
        Processor::create_group(prof_procs) : prof_procs.front();
      profiler = new LegionProfiler(target_proc_for_profiler,
                                    machine, this, LG_LAST_TASK_ID,
                                    lg_task_descriptions,
                                    Operation::LAST_OP_KIND,
                                    Operation::op_names,
                                    Runtime::serializer_type,
                                    Runtime::prof_logfile,
                                    total_address_spaces,
                                    Runtime::prof_footprint_threshold,
                                    Runtime::prof_target_latency);
      LG_MESSAGE_DESCRIPTIONS(lg_message_descriptions);
      profiler->record_message_kinds(lg_message_descriptions, LAST_SEND_KIND);
      MAPPER_CALL_NAMES(lg_mapper_calls);
      profiler->record_mapper_call_kinds(lg_mapper_calls, LAST_MAPPER_CALL);
#ifdef DETAILED_LEGION_PROF
      RUNTIME_CALL_DESCRIPTIONS(lg_runtime_calls);
      profiler->record_runtime_call_kinds(lg_runtime_calls, 
                                          LAST_RUNTIME_CALL_KIND);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::initialize_mappers(void)
    //--------------------------------------------------------------------------
    {
      if (Runtime::replay_file == NULL) // This is the normal path
      {
        if (enable_test_mapper)
        {
          // Make test mappers for everyone
          for (std::map<Processor,ProcessorManager*>::const_iterator it = 
                proc_managers.begin(); it != proc_managers.end(); it++)
          {
            Mapper *mapper = 
              new Mapping::TestMapper(mapper_runtime, machine, it->first);
            MapperManager *wrapper = wrap_mapper(this, mapper, 0, it->first);
            it->second->add_mapper(0, wrapper, false/*check*/, true/*owns*/);
          }
        }
        else
        {
          // Make default mappers for everyone
          for (std::map<Processor,ProcessorManager*>::const_iterator it = 
                proc_managers.begin(); it != proc_managers.end(); it++)
          {
            Mapper *mapper = 
              new Mapping::DefaultMapper(mapper_runtime, machine, it->first);
            MapperManager *wrapper = wrap_mapper(this, mapper, 0, it->first);
            it->second->add_mapper(0, wrapper, false/*check*/, true/*owns*/);
          }
#ifndef DISABLE_PARTITION_SHIM
          // Make default mappers for the partition shim 
          for (std::map<Processor,ProcessorManager*>::const_iterator it = 
                proc_managers.begin(); it != proc_managers.end(); it++)
          {
            Mapper *mapper = 
              new Mapping::DefaultMapper(mapper_runtime, machine, it->first);
            MapperManager *wrapper = wrap_mapper(this, mapper, 
                                        PARTITION_SHIM_MAPPER_ID, it->first);
            it->second->add_mapper(PARTITION_SHIM_MAPPER_ID, wrapper, 
                                   false/*check*/, true/*owns*/);
          }
#endif
          // Now ask the application what it wants to do
          if (!Runtime::registration_callbacks.empty())
          {
            log_run.info("Invoking mapper registration callback functions...");
            for (std::vector<RegistrationCallbackFnptr>::const_iterator it = 
                  Runtime::registration_callbacks.begin(); it !=
                  Runtime::registration_callbacks.end(); it++)
              (**it)(machine, external, local_procs);
            log_run.info("Finished execution of mapper registration callbacks");
          }
        }
      }
      else // This is the replay/debug path
      {
        if (legion_ldb_enabled)
        {
          // This path is not quite ready yet
          assert(false);
          for (std::map<Processor,ProcessorManager*>::const_iterator it = 
                proc_managers.begin(); it != proc_managers.end(); it++)
          {
            Mapper *mapper = new Mapping::DebugMapper(mapper_runtime, 
                                            machine, it->first, replay_file);
            MapperManager *wrapper = wrap_mapper(this, mapper, 0, it->first);
            it->second->add_mapper(0, wrapper, false/*check*/, true/*owns*/, 
                                    true/*skip replay*/);
          }
        }
        else
        {
          for (std::map<Processor,ProcessorManager*>::const_iterator it =
                proc_managers.begin(); it != proc_managers.end(); it++)
          {
            Mapper *mapper = new Mapping::ReplayMapper(mapper_runtime, 
                                            machine, it->first, replay_file);
            MapperManager *wrapper = wrap_mapper(this, mapper, 0, it->first);
            it->second->add_mapper(0, wrapper, false/*check*/, true/*owns*/,
                                    true/*skip replay*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::startup_mappers(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
        it->second->startup_mappers();
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_runtime(void)
    //--------------------------------------------------------------------------
    {
      if (profiler != NULL)
      {
        // Have the memory managers for deletion of all their instances
        for (std::map<Memory,MemoryManager*>::const_iterator it =
             memory_managers.begin(); it != memory_managers.end(); it++)
          it->second->finalize();
        profiler->finalize();
      }
    }
    
    //--------------------------------------------------------------------------
    void Runtime::launch_top_level_task(void)
    //--------------------------------------------------------------------------
    {
      // Grab the first local processor to be the initial target
#ifdef DEBUG_LEGION
      assert(!local_procs.empty());
#endif
      Processor target = *(local_procs.begin());
      // For backwards compatibility right now we'll make sure this
      // is a CPU processor if possible
      if (target.kind() != Processor::LOC_PROC)
      {
        for (std::set<Processor>::const_iterator it = 
              local_procs.begin(); it != local_procs.end(); it++)
        {
          if (it->kind() != Processor::LOC_PROC)
            continue;
          target = *it;
          break;
        }
      }
      // Get an individual task to be the top-level task
      IndividualTask *top_task = get_available_individual_task();
      // Get a remote task to serve as the top of the top-level task
      TopLevelContext *top_context = 
        new TopLevelContext(this, get_unique_operation_id());
      // Add a reference to the top level context
      top_context->add_reference();
      // Set the executing processor
      top_context->set_executing_processor(target);
      TaskLauncher launcher(Runtime::legion_main_id, TaskArgument());
      // Mark that this task is the top-level task
      top_task->set_top_level();
      top_task->initialize_task(top_context, launcher, 
                                false/*check priv*/, false/*track parent*/);
      // Set up the input arguments
      top_task->arglen = sizeof(InputArgs);
      top_task->args = malloc(top_task->arglen);
      memcpy(top_task->args,&Runtime::get_input_args(),top_task->arglen);
      // Set this to be the current processor
      top_task->set_current_proc(target);
      top_task->select_task_options();
      if (legion_spy_enabled)
      {
        Runtime::log_machine(machine);
        LegionSpy::log_top_level_task(Runtime::legion_main_id,
                                      top_task->get_unique_id(),
                                      top_task->get_task_name());
      }
      increment_outstanding_top_level_tasks();
      // Launch a task to deactivate the top-level context
      // when the top-level task is done
      TopFinishArgs args;
      args.ctx = top_context;
      ApEvent pre = top_task->get_task_completion();
      issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, NULL,
                              Runtime::protect_event(pre));
      // Put the task in the ready queue
      add_to_ready_queue(target, top_task);
    }

    //--------------------------------------------------------------------------
    ApEvent Runtime::launch_mapper_task(Mapper *mapper, Processor proc, 
                                        Processor::TaskFuncID tid,
                                        const TaskArgument &arg,MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Get an individual task to be the top-level task
      IndividualTask *mapper_task = get_available_individual_task();
      // Get a remote task to serve as the top of the top-level task
      TopLevelContext *map_context = 
        new TopLevelContext(this, get_unique_operation_id());
      map_context->add_reference();
      map_context->set_executing_processor(proc);
      TaskLauncher launcher(tid, arg, Predicate::TRUE_PRED, map_id);
      Future f = mapper_task->initialize_task(map_context, launcher, 
                                   false/*check priv*/, false/*track parent*/);
      mapper_task->set_current_proc(proc);
      mapper_task->select_task_options();
      // Create a temporary event to name the result since we 
      // have to pack it in the task that runs, but it also depends
      // on the task being reported back to the mapper
      ApUserEvent result = Runtime::create_ap_user_event();
      // Add a reference to the future impl to prevent it being collected
      f.impl->add_base_gc_ref(FUTURE_HANDLE_REF);
      // Create a meta-task to return the results to the mapper
      MapperTaskArgs args;
      args.future = f.impl;
      args.map_id = map_id;
      args.proc = proc;
      args.event = result;
      args.ctx = map_context;
      ApEvent pre = f.impl->get_ready_event();
      ApEvent post(issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, NULL,
                                           Runtime::protect_event(pre)));
      // Chain the events properly
      Runtime::trigger_event(result, post);
      // Mark that we have another outstanding top level task
      increment_outstanding_top_level_tasks();
      // Now we can put it on the queue
      add_to_ready_queue(proc, mapper_task);
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_task_result(const MapperTaskArgs *args)
    //--------------------------------------------------------------------------
    {
#if 0
      MapperManager *mapper = find_mapper(args->proc, args->map_id);
      Mapper::MapperTaskResult result;
      result.mapper_event = args->event;
      result.result = args->future->get_untyped_result();
      result.result_size = args->future->get_untyped_size();
      mapper->invoke_handle_task_result(&result);
#else
      assert(false); // update this
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space(Context ctx, const void *realm_is,
                                           TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create index space!");
      return ctx->create_index_space(forest, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::union_index_spaces(Context ctx,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create index space!");
      return ctx->union_index_spaces(forest, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::intersect_index_spaces(Context ctx,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create index space!");
      return ctx->intersect_index_spaces(forest, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::subtract_index_spaces(Context ctx,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context subtract index space!");
      return ctx->subtract_index_spaces(forest, left, right);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context destroy index space!");
      ctx->destroy_index_space(handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_index_space_destroy(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_index_space(handle, address_space);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_partition(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context destroy index partition!");
      ctx->destroy_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_index_partition_destroy(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_index_partition(handle, address_space);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_equal_partition(Context ctx, 
                                                   IndexSpace parent,
                                                   IndexSpace color_space,
                                                   size_t granularity,
                                                   Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create equal partition!");
      return ctx->create_equal_partition(forest, parent, color_space,
                                         granularity, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_union(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      IndexSpace color_space,
                                                      PartitionKind kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create partition by union!");
      IndexPartition result = 
        ctx->create_partition_by_union(forest, parent, handle1, handle2,
                                       color_space, kind, color);
      if (verify_disjointness && (kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by union in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_intersection(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      IndexSpace color_space,
                                                      PartitionKind kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create partition by intersection!");
      IndexPartition result = 
        ctx->create_partition_by_intersection(forest, parent, handle1,
                                              handle2, color_space,
                                              kind, color);
      if (verify_disjointness && (kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by intersection in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_difference(Context ctx, 
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      IndexSpace color_space,
                                                      PartitionKind kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create difference partition!");
      IndexPartition result = 
        ctx->create_partition_by_difference(forest, parent, handle1, 
                                            handle2, color_space,
                                            kind, color);
      if (verify_disjointness && (kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by difference in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    Color Runtime::create_cross_product_partitions(Context ctx,
                                                  IndexPartition handle1,
                                                  IndexPartition handle2,
                                   std::map<IndexSpace,IndexPartition> &handles,
                                                 PartitionKind kind,Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create cross product partition!");
      Color result = 
        ctx->create_cross_product_partitions(forest, handle1, handle2, 
                                             handles, kind, color);
      if (verify_disjointness && (kind == DISJOINT_KIND))
      {
        Domain color_space = get_index_partition_color_space(handle1);
        // This code will only work if the color space has type coord_t
        switch (color_space.get_dim())
        {
          case 1:
            {
              TypeTag type_tag = NT_TemplateHelper::encode_tag<1,coord_t>();
              assert(handle1.get_type_tag() == type_tag);
              break;
            }
          case 2:
            {
              TypeTag type_tag = NT_TemplateHelper::encode_tag<2,coord_t>();
              assert(handle1.get_type_tag() == type_tag);
              break;
            }
          case 3:
            {
              TypeTag type_tag = NT_TemplateHelper::encode_tag<3,coord_t>();
              assert(handle1.get_type_tag() == type_tag);
              break;
            }
          default:
            assert(false);
        }
        for (Domain::DomainPointIterator itr(color_space); itr; itr++)
        {
          IndexSpace subspace;
          switch (color_space.get_dim())
          {
            case 1:
              {
                const Point<1,coord_t> p(itr.p);
                subspace = get_index_subspace(handle1, &p, sizeof(p));
                break;
              }
            case 2:
              {
                const Point<2,coord_t> p(itr.p);
                subspace = get_index_subspace(handle1, &p, sizeof(p));
                break;
              }
            case 3:
              {
                const Point<3,coord_t> p(itr.p);
                subspace = get_index_subspace(handle1, &p, sizeof(p));
                break;
              }
            default:
              assert(false);
          }
          IndexPartition part = get_index_partition(subspace, result);
          if (!forest->is_disjoint(part))
            REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                                "Disjointness test failure for create cross "
                                "product partitions in task %s (UID %lld)",
                                ctx->get_task_name(), ctx->get_unique_id())
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::create_association(Context ctx, 
                                     LogicalRegion domain,
                                     LogicalRegion domain_parent,
                                     FieldID domain_fid,
                                     IndexSpace range,
                                     MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy create association!");
      ctx->create_association(domain, domain_parent, domain_fid, range, id,tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_restricted_partition(Context ctx,
                                                      IndexSpace parent,
                                                      IndexSpace color_space,
                                                      const void *transform,
                                                      size_t transform_size,
                                                      const void *extent,
                                                      size_t extent_size,
                                                      PartitionKind part_kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create restricted partition!");
      IndexPartition result =
        ctx->create_restricted_partition(forest, parent, color_space,
                                         transform, transform_size,
                                         extent, extent_size,
                                         part_kind, color);
      if (verify_disjointness && (part_kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create restricted "
                            "partition in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_field(Context ctx,
                                                      LogicalRegion handle,
                                                      LogicalRegion parent_priv,
                                                      FieldID fid,
                                                      IndexSpace color_space,
                                                      Color color,
                                                      MapperID id, 
                                                      MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context partition by field!");
      return ctx->create_partition_by_field(forest, handle, parent_priv, fid,
                                            color_space, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_image(Context ctx,
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color,
                                                    MapperID id, 
                                                    MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context partition by image!");
      IndexPartition result = 
        ctx->create_partition_by_image(forest, handle, projection, parent,
                                       fid, color_space, part_kind, 
                                       color, id, tag);
      if (verify_disjointness && (part_kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by image in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_image_range(Context ctx,
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color,
                                                    MapperID id, 
                                                    MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context partition by image range!");
      IndexPartition result = 
        ctx->create_partition_by_image_range(forest, handle, projection, 
                                  parent, fid, color_space, part_kind, 
                                  color, id, tag);
      if (verify_disjointness && (part_kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by image range in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_preimage(Context ctx,
                                                    IndexPartition projection,
                                                    LogicalRegion handle,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color,
                                                    MapperID id, 
                                                    MappingTagID tag)
    //--------------------------------------------------------------------------
    { 
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context partition by preimage!");
      IndexPartition result = 
        ctx->create_partition_by_preimage(forest, projection, handle,
                              parent, fid, color_space, part_kind, 
                              color, id, tag);
      if (verify_disjointness && (part_kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by preimage in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_partition_by_preimage_range(Context ctx,
                                                    IndexPartition projection,
                                                    LogicalRegion handle,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color, MapperID id,
                                                    MappingTagID tag)
    //--------------------------------------------------------------------------
    { 
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context partition by preimage range!");
      IndexPartition result = 
        ctx->create_partition_by_preimage_range(forest, projection, handle,
                                    parent, fid, color_space, part_kind, 
                                    color, id, tag);
      if (verify_disjointness && (part_kind == DISJOINT_KIND) && 
          !forest->is_disjoint(result))
        REPORT_LEGION_ERROR(ERROR_DISJOINTNESS_TEST_FAILURE,
                            "Disjointness test failure for create partition "
                            "by preimage range in task %s (UID %lld)",
                            ctx->get_task_name(), ctx->get_unique_id())
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::create_pending_partition(Context ctx, 
                                                     IndexSpace parent, 
                                                     IndexSpace color_space,
                                                     PartitionKind part_kind,
                                                     Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create pending partition!");
      return ctx->create_pending_partition(forest, parent, color_space,
                                           part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx, 
                                                 IndexPartition parent,
                                                 const void *realm_color, 
                                                 TypeTag type_tag,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create index space union!");
      return ctx->create_index_space_union(forest, parent, realm_color, 
                                           type_tag, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_union(Context ctx,
                                                 IndexPartition parent,
                                                 const void *realm_color,
                                                 TypeTag type_tag,
                                                 IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create index space union!");
      return ctx->create_index_space_union(forest, parent, realm_color, 
                                           type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                                                        IndexPartition parent,
                                                        const void *realm_color,
                                                        TypeTag type_tag,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create index space intersection!");
      return ctx->create_index_space_intersection(forest, parent, realm_color,
                                                  type_tag, handles); 
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_intersection(Context ctx,
                                                        IndexPartition parent,
                                                        const void *realm_color,
                                                        TypeTag type_tag,
                                                        IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create index space intersection!");
      return ctx->create_index_space_intersection(forest, parent, realm_color,
                                                  type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::create_index_space_difference(Context ctx,
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                                      IndexSpace initial,
                                         const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create index space difference!");
      return ctx->create_index_space_difference(forest, parent, realm_color,
                                                type_tag, initial, handles);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexPartition result = get_index_partition(parent, color);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexPartition result = forest->get_index_partition(parent, color);
#ifdef DEBUG_LEGION
      if (!result.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_INDEX_SPACE_COLOR, 
            "Invalid color %d for get index partitions", color);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(Context ctx, IndexSpace parent, 
                                      Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = has_index_partition(parent, color);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return forest->has_index_partition(parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(Context ctx, IndexPartition p, 
                                           const void *realm_color,
                                           TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexSpace result = get_index_subspace(p, realm_color, type_tag); 
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(IndexPartition p, 
                                           const void *realm_color,
                                           TypeTag type_tag) 
    //--------------------------------------------------------------------------
    {
      return forest->get_index_subspace(p, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(Context ctx, IndexPartition p,
                                     const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = has_index_subspace(p, realm_color, type_tag); 
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(IndexPartition p,
                                     const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return forest->has_index_subspace(p, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domain(Context ctx, IndexSpace handle,
                                         void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      get_index_space_domain(handle, realm_is, type_tag);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domain(IndexSpace handle, 
                                         void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_domain(handle, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(Context ctx,
                                                    IndexPartition p)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      Domain result = get_index_partition_color_space(p); 
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *part = forest->get_node(p);
      const IndexSpace color_space = part->color_space->handle;
      switch (NT_TemplateHelper::get_dim(color_space.get_type_tag()))
      {
        case 1:
          {
            DomainT<1,coord_t> color_index_space;
            forest->get_index_space_domain(color_space, &color_index_space,
                                           color_space.get_type_tag());
            return Domain(color_index_space);
          }
        case 2:
          {
            DomainT<2,coord_t> color_index_space;
            forest->get_index_space_domain(color_space, &color_index_space,
                                           color_space.get_type_tag());
            return Domain(color_index_space);
          }
        case 3:
          {
            DomainT<3,coord_t> color_index_space;
            forest->get_index_space_domain(color_space, &color_index_space,
                                           color_space.get_type_tag());
            return Domain(color_index_space);
          }
        default:
          assert(false);
      }
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_partition_color_space(IndexPartition p,
                                               void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *part = forest->get_node(p);
      const IndexSpace color_space = part->color_space->handle;
      forest->get_index_space_domain(color_space, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_partition_color_space_name(Context ctx,
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexSpace result = get_index_partition_color_space_name(p);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_partition_color_space_name(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color_space(p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(Context ctx, IndexSpace sp,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      get_index_space_partition_colors(sp, colors);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(IndexSpace handle,
                                                   std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_partition_colors(handle, colors);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = forest->is_index_partition_disjoint(p);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_disjoint(p);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_complete(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = forest->is_index_partition_complete(p);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_complete(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return forest->is_index_partition_complete(p);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_color_point(Context ctx, IndexSpace handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      forest->get_index_space_color(handle, realm_color, type_tag);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_color_point(IndexSpace handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      forest->get_index_space_color(handle, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(Context ctx, 
                                                     IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexSpaceNode *node = forest->get_node(handle);
      DomainPoint result = node->get_domain_point_color();
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = forest->get_node(handle);
      return node->get_domain_point_color();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      Color result = forest->get_index_partition_color(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(Context ctx,   
                                               IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexSpace result = forest->get_parent_index_space(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = forest->has_parent_index_partition(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexPartition result = forest->get_parent_index_partition(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_space_depth(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      unsigned result = forest->get_index_space_depth(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_space_depth(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_space_depth(handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_partition_depth(Context ctx, 
                                                IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      unsigned result = forest->get_index_partition_depth(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_partition_depth(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_index_partition_depth(handle);
    }

    //--------------------------------------------------------------------------
    bool Runtime::safe_cast(Context ctx, LogicalRegion region,
                            const void *realm_point, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context safe cast!");
      return ctx->safe_cast(forest, region.get_index_space(), 
                            realm_point, type_tag);
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create field space!");
      return ctx->create_field_space(forest); 
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_field_space(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context destroy field space!");
      ctx->destroy_field_space(handle);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(Context ctx, FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      size_t result = forest->get_field_size(handle, fid);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      return forest->get_field_size(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(Context ctx, FieldSpace handle,
                                         std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      forest->get_field_space_fields(handle, fields);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(FieldSpace handle, 
                                         std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      forest->get_field_space_fields(handle, fields);
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
      forest->free_field(handle, fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_field_destroy(FieldSpace handle, 
                                               const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> dense(to_free.begin(), to_free.end());
      forest->free_fields(handle, dense);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::create_logical_region(Context ctx, 
                                IndexSpace index_space, FieldSpace field_space)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create logical region!");
      return ctx->create_logical_region(forest, index_space, field_space); 
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context destroy logical region!");
      ctx->destroy_logical_region(handle); 
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_partition(Context ctx,LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context destroy logical partition!");
      ctx->destroy_logical_partition(handle); 
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_logical_region_destroy(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      forest->destroy_logical_region(handle, address_space);
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalPartition result = forest->get_logical_partition(parent, handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalPartition result = 
        forest->get_logical_partition_by_color(parent, c);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(LogicalRegion par,
                                                             Color c)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_by_color(par, c);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(Context ctx, 
                                              LogicalRegion parent, Color color)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = forest->has_logical_partition_by_color(parent, color);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(LogicalRegion parent, 
                                                 Color color)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_partition_by_color(parent, color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalPartition result = 
        forest->get_logical_partition_by_tree(handle, fspace, tid);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
                                                            IndexPartition part,
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalRegion result = forest->get_logical_subregion(parent, handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
             LogicalPartition parent, const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalRegion result = 
        forest->get_logical_subregion_by_color(parent, realm_color, type_tag);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(LogicalPartition par,
                                      const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_subregion_by_color(par, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(Context ctx,
             LogicalPartition parent, const void *realm_point, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = 
        forest->has_logical_subregion_by_color(parent, realm_point, type_tag);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(LogicalPartition parent, 
                                      const void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return forest->has_logical_subregion_by_color(parent, 
                                                    realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalRegion result = 
        forest->get_logical_subregion_by_tree(handle, fspace, tid);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
    void Runtime::get_logical_region_color(Context ctx, LogicalRegion handle, 
                                           void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      forest->get_logical_region_color(handle, realm_color, type_tag);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_logical_region_color(LogicalRegion handle,
                                            void *realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      forest->get_logical_region_color(handle, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_region_color_point(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      IndexSpaceNode *node = forest->get_node(handle.get_index_space());
      DomainPoint result = node->get_domain_point_color();
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_region_color_point(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode *node = forest->get_node(handle.get_index_space());
      return node->get_domain_point_color();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(Context ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      Color result = forest->get_logical_partition_color(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_logical_partition_color(handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalRegion result = forest->get_parent_logical_region(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      bool result = forest->has_parent_logical_partition(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
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
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      LogicalPartition result = forest->get_parent_logical_partition(handle);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return forest->get_parent_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    FieldAllocator Runtime::create_field_allocator(Context ctx, 
                                                            FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create field allocator!");
      return ctx->create_field_allocator(external, handle); 
    }

    //--------------------------------------------------------------------------
    ArgumentMap Runtime::create_argument_map(void)
    //--------------------------------------------------------------------------
    {
      ArgumentMapImpl *impl = new ArgumentMapImpl();
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return ArgumentMap(impl);
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_task(Context ctx, const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    { 
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context execute task!");
      return ctx->execute_task(launcher); 
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_index_space(Context ctx, 
                                           const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context execute index space!");
      return ctx->execute_index_space(launcher); 
    }

    //--------------------------------------------------------------------------
    Future Runtime::execute_index_space(Context ctx, 
                         const IndexTaskLauncher &launcher, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context execute index space!");
      return ctx->execute_index_space(launcher, redop); 
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context map region!");
      return ctx->map_region(launcher); 
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::map_region(Context ctx, unsigned idx, 
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context map region!");
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
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context remap region!");
      return ctx->remap_region(region); 
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context unmap region!");
      ctx->unmap_region(region); 
    }

    //--------------------------------------------------------------------------
    void Runtime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      ctx->unmap_all_regions();
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context fill operation!");
      ctx->fill_fields(launcher); 
    }

    //--------------------------------------------------------------------------
    void Runtime::fill_fields(Context ctx, const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context fill operation!");
      ctx->fill_fields(launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion Runtime::attach_external_resource(Context ctx, 
                                                 const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context attach external resource!");
      return ctx->attach_resource(launcher);
    }

    //--------------------------------------------------------------------------
    void Runtime::detach_external_resource(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context detach external resource!");
      ctx->detach_resource(region);
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx,const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context issue copy operation!");
      ctx->issue_copy(launcher); 
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_copy_operation(Context ctx,
                                       const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context issue copy operation!");
      ctx->issue_copy(launcher);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::create_predicate(Context ctx, const Future &f) 
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create predicate!");
      return ctx->create_predicate(f);
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create predicate not!");
      return ctx->predicate_not(p); 
    }

    //--------------------------------------------------------------------------
    Predicate Runtime::create_predicate(Context ctx, 
                                        const PredicateLauncher &launcher) 
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context create predicate!");
      return ctx->create_predicate(launcher);
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_predicate_future(Context ctx, const Predicate &p)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context get predicate future!");
      return ctx->get_predicate_future(p);
    }

    //--------------------------------------------------------------------------
    Lock Runtime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      Lock result(Reservation::create_reservation());
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      ctx->destroy_user_lock(l.reservation_lock);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    Grant Runtime::acquire_grant(Context ctx, 
                                 const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      // Kind of annoying, but we need to unpack and repack the
      // Lock type here to build new requests because the C++
      // type system is dumb with nested classes.
      std::vector<GrantImpl::ReservationRequest> 
        unpack_requests(requests.size());
      for (unsigned idx = 0; idx < requests.size(); idx++)
      {
        unpack_requests[idx] = 
          GrantImpl::ReservationRequest(requests[idx].lock.reservation_lock,
                                        requests[idx].mode,
                                        requests[idx].exclusive);
      }
      Grant result(new GrantImpl(unpack_requests));
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      grant.impl->release_grant();
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::create_phase_barrier(Context ctx, unsigned arrivals) 
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create phase barrier!");
#ifdef DEBUG_LEGION
      log_run.debug("Creating phase barrier in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      ApBarrier result(Realm::Barrier::create_barrier(arrivals));
      ctx->end_runtime_call();
      return PhaseBarrier(result);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context destroy phase barrier!");
#ifdef DEBUG_LEGION
      log_run.debug("Destroying phase barrier in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      ctx->destroy_user_barrier(pb.phase_barrier);
      ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier Runtime::advance_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context advance phase barrier!");
#ifdef DEBUG_LEGION
      log_run.debug("Advancing phase barrier in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      PhaseBarrier result = pb;
      Runtime::advance_barrier(result);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(pb.phase_barrier, result.phase_barrier);
#endif
      ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::create_dynamic_collective(Context ctx,
                                                         unsigned arrivals,
                                                         ReductionOpID redop,
                                                         const void *init_value,
                                                         size_t init_size)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context create dynamic collective!");
#ifdef DEBUG_LEGION
      log_run.debug("Creating dynamic collective in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      ApBarrier result(Realm::Barrier::create_barrier(arrivals, redop, 
                                               init_value, init_size));
      ctx->end_runtime_call();
      return DynamicCollective(result, redop);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_dynamic_collective(Context ctx, DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context destroy dynamic collective!");
#ifdef DEBUG_LEGION
      log_run.debug("Destroying dynamic collective in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      ctx->destroy_user_barrier(dc.phase_barrier);
      ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::arrive_dynamic_collective(Context ctx, DynamicCollective dc,
                                            const void *buffer, size_t size,
                                            unsigned count)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context arrive dynamic collective!");
#ifdef DEBUG_LEGION
      log_run.debug("Arrive dynamic collective in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      Runtime::phase_barrier_arrive(dc, count, ApEvent::NO_AP_EVENT, 
                                    buffer, size);
      ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    void Runtime::defer_dynamic_collective_arrival(Context ctx, 
                                                   DynamicCollective dc,
                                                   const Future &f, 
                                                   unsigned count)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context defer dynamic collective arrival!");
#ifdef DEBUG_LEGION
      log_run.debug("Defer dynamic collective arrival in "
                          "task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      // Record this future as a contribution to the collective
      // for future dependence analysis
      ctx->record_dynamic_collective_contribution(dc, f);
      f.impl->contribute_to_collective(dc, count);
      ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    Future Runtime::get_dynamic_collective_result(Context ctx, 
                                                  DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context get dynamic collective result!");
#ifdef DEBUG_LEGION
      log_run.debug("Get dynamic collective result in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      DynamicCollectiveOp *collective = 
        get_available_dynamic_collective_op();
      Future result = collective->initialize(ctx, dc);
      Processor proc = ctx->get_executing_processor();
      add_to_dependence_queue(ctx, proc, collective);
      ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective Runtime::advance_dynamic_collective(Context ctx,
                                                          DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context advance dynamic collective!");
#ifdef DEBUG_LEGION
      log_run.debug("Advancing dynamic collective in task %s (ID %lld)",
                          ctx->get_task_name(), ctx->get_unique_id());
#endif
      ctx->begin_runtime_call();
      DynamicCollective result = dc;
      Runtime::advance_barrier(result);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(dc.phase_barrier, result.phase_barrier);
#endif
      ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_acquire(Context ctx, const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context issue acquire!");
      ctx->issue_acquire(launcher); 
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_release(Context ctx, const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context issue release!");
      ctx->issue_release(launcher); 
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context issue mapping fence!");
      ctx->issue_mapping_fence(); 
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context issue execution fence!");
      ctx->issue_execution_fence(); 
    }

    //--------------------------------------------------------------------------
    void Runtime::begin_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context begin trace!");
      ctx->begin_trace(tid);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context end trace!");
      ctx->end_trace(tid); 
    }

    //--------------------------------------------------------------------------
    void Runtime::begin_static_trace(Context ctx, 
                                     const std::set<RegionTreeID> *managed)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context begin static trace!");
      ctx->begin_static_trace(managed);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_static_trace(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context end static trace!");
      ctx->end_static_trace(); 
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_frame(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context issue frame!");
      ctx->complete_frame(); 
    }

    //--------------------------------------------------------------------------
    FutureMap Runtime::execute_must_epoch(Context ctx, 
                                          const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context issue must epoch!");
      return ctx->execute_must_epoch(launcher); 
    }

    //--------------------------------------------------------------------------
    Future Runtime::issue_timing_measurement(Context ctx,
                                             const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context in timing measurement!");
      return ctx->issue_timing_measurement(launcher); 
    }

    //--------------------------------------------------------------------------
    Future Runtime::select_tunable_value(Context ctx, TunableID tid,
                                         MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context select tunable value!");
      ctx->begin_runtime_call();
#ifdef DEBUG_LEGION
      log_run.debug("Getting a value for tunable variable %d in "
                    "task %s (ID %lld)", tid, ctx->get_task_name(),
                    ctx->get_unique_id());
#endif
      FutureImpl *result = new FutureImpl(this, true/*register*/,
                              get_available_distributed_id(),
                              address_space, ctx->get_owner_task());
      // Make this here to get a local reference on it now
      Future result_future(result);
      result->add_base_gc_ref(FUTURE_HANDLE_REF);
      SelectTunableArgs args;
      args.mapper_id = mid;
      args.tag = tag;
      args.tunable_id = tid;
      if (legion_spy_enabled)
        args.tunable_index = ctx->get_tunable_index();
      args.ctx = ctx;
      args.result = result;
      issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                              ctx->get_owner_task());
      ctx->end_runtime_call();
      return result_future;
    }

    //--------------------------------------------------------------------------
    int Runtime::get_tunable_value(Context ctx, TunableID tid,
                                   MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context get tunable value!");
      ctx->begin_runtime_call();
      Future f = select_tunable_value(ctx, tid, mid, tag);
      int result = f.get_result<int>();
      if (legion_spy_enabled)
      {
        unsigned index = ctx->get_tunable_index();
        LegionSpy::log_tunable_value(ctx->get_unique_id(), index,
                                     &result, sizeof(result));
      }
      ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::perform_tunable_selection(const SelectTunableArgs *args)
    //--------------------------------------------------------------------------
    {
      // Get the mapper first
      MapperManager *mapper = find_mapper(args->ctx->get_executing_processor(),
                                          args->mapper_id);
      Mapper::SelectTunableInput input;
      Mapper::SelectTunableOutput output;
      input.tunable_id = args->tunable_id;
      input.mapping_tag = args->tag;
      output.value = NULL;
      output.size = 0;
      output.take_ownership = true;
      mapper->invoke_select_tunable_value(args->ctx->get_owner_task(), 
                                          &input, &output);
      if (legion_spy_enabled)
        LegionSpy::log_tunable_value(args->ctx->get_unique_id(), 
            args->tunable_index, output.value, output.size);
      // Set and complete the future
      if ((output.value != NULL) && (output.size > 0))
        args->result->set_result(output.value, output.size, 
                                 output.take_ownership);
      args->result->complete_future();
    }

    //--------------------------------------------------------------------------
    void* Runtime::get_local_task_variable(Context ctx, LocalVariableID id)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context get local task variable!");
      return ctx->get_local_task_variable(id);
    }

    //--------------------------------------------------------------------------
    void Runtime::set_local_task_variable(Context ctx, LocalVariableID id,
                                   const void *value, void (*destructor)(void*))
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT(
            "Illegal dummy context set local task variable!");
      ctx->set_local_task_variable(id, value, destructor);
    }

    //--------------------------------------------------------------------------
    Mapper* Runtime::get_mapper(Context ctx, MapperID id, Processor target)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      if (!target.exists())
      {
        Processor proc = ctx->get_executing_processor();
#ifdef DEBUG_LEGION
        assert(proc_managers.find(proc) != proc_managers.end());
#endif
        if (ctx != DUMMY_CONTEXT)
          ctx->end_runtime_call();
        return proc_managers[proc]->find_mapper(id)->mapper;
      }
      else
      {
        std::map<Processor,ProcessorManager*>::const_iterator finder = 
          proc_managers.find(target);
        if (finder == proc_managers.end())
          REPORT_LEGION_ERROR(ERROR_INVALID_PROCESSOR_NAME, 
           "Invalid processor " IDFMT " passed to get mapper call.", target.id);
        if (ctx != DUMMY_CONTEXT)
          ctx->end_runtime_call();
        return finder->second->find_mapper(id)->mapper;
      }
    }

    //--------------------------------------------------------------------------
    Processor Runtime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      Processor result = ctx->get_executing_processor();
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::raise_region_exception(Context ctx, 
                                         PhysicalRegion region, bool nuclear)
    //--------------------------------------------------------------------------
    {
      if (ctx != DUMMY_CONTEXT)
        ctx->begin_runtime_call();
      // TODO: implement this
      assert(false);
      if (ctx != DUMMY_CONTEXT)
        ctx->end_runtime_call();
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& Runtime::find_forward_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (mpi_rank_table == NULL)
        REPORT_LEGION_ERROR(ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED, 
             "Forward MPI mapping call not supported without "
                      "calling configure_MPI_interoperability during "
                      "start up")
#ifdef DEBUG_LEGION
      assert(!mpi_rank_table->forward_mapping.empty());
#endif
      return mpi_rank_table->forward_mapping;
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>& Runtime::find_reverse_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (mpi_rank_table == NULL)
        REPORT_LEGION_ERROR(ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED,
             "Reverse MPI mapping call not supported without "
                      "calling configure_MPI_interoperability during "
                      "start up")
#ifdef DEBUG_LEGION
      assert(!mpi_rank_table->reverse_mapping.empty());
#endif
      return mpi_rank_table->reverse_mapping;
    }

    //--------------------------------------------------------------------------
    int Runtime::find_local_MPI_rank(void)
    //-------------------------------------------------------------------------
    {
      if (mpi_rank_table == NULL)
        REPORT_LEGION_ERROR(ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED,
             "Findling local MPI rank not supported without "
                      "calling configure_MPI_interoperability during "
                      "start up")
      return mpi_rank;
    }

    //--------------------------------------------------------------------------
    void Runtime::add_mapper(MapperID map_id, Mapper *mapper, Processor proc)
    //--------------------------------------------------------------------------
    {
      // First, wrap this mapper in a mapper manager
      MapperManager *manager = wrap_mapper(this, mapper, map_id, proc);
      if (!proc.exists())
      {
        bool own = true;
        // Save it to all the managers
        for (std::map<Processor,ProcessorManager*>::const_iterator it = 
              proc_managers.begin(); it != proc_managers.end(); it++)
        {
          it->second->add_mapper(map_id, manager, true/*check*/, own);
          own = false;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(proc_managers.find(proc) != proc_managers.end());
#endif
        proc_managers[proc]->add_mapper(map_id, manager, 
                                        true/*check*/, true/*own*/);
      }
    }

    //--------------------------------------------------------------------------
    Mapping::MapperRuntime* Runtime::get_mapper_runtime(void)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime;
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      MapperID result = __sync_fetch_and_add(&unique_mapper_id, runtime_stride);
#ifdef DEBUG_LEGION
      // Check for overflow
      assert(result <= unique_mapper_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID& Runtime::get_current_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      static MapperID current_mapper_id = MAX_APPLICATION_MAPPER_ID;
      return current_mapper_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID Runtime::generate_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      MapperID &next_mapper = get_current_static_mapper_id(); 
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'generate_static_mapper_id' after "
                      "the runtime has been started!")
      return next_mapper++;
    }

    //--------------------------------------------------------------------------
    void Runtime::replace_default_mapper(Mapper *mapper, Processor proc)
    //--------------------------------------------------------------------------
    {
      // First, wrap this mapper in a mapper manager
      MapperManager *manager = wrap_mapper(this, mapper, 0, proc); 
      if (!proc.exists())
      {
        bool own = true;
        // Save it to all the managers
        for (std::map<Processor,ProcessorManager*>::const_iterator it = 
              proc_managers.begin(); it != proc_managers.end(); it++)
        {
          it->second->replace_default_mapper(manager, own);
          own = false;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(proc_managers.find(proc) != proc_managers.end());
#endif
        proc_managers[proc]->replace_default_mapper(manager, true/*own*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperManager* Runtime::wrap_mapper(Runtime *rt, Mapper *mapper,
                                                   MapperID map_id, Processor p)
    //--------------------------------------------------------------------------
    {
      MapperManager *manager = NULL;
      switch (mapper->get_mapper_sync_model())
      {
        case Mapper::CONCURRENT_MAPPER_MODEL:
          {
            manager = new ConcurrentManager(rt, mapper, map_id, p);
            break;
          }
        case Mapper::SERIALIZED_REENTRANT_MAPPER_MODEL:
          {
            manager = new SerializingManager(rt, mapper, 
                                             map_id, p, true/*reentrant*/);
            break;
          }
        case Mapper::SERIALIZED_NON_REENTRANT_MAPPER_MODEL:
          {
            manager = new SerializingManager(rt, mapper, 
                                             map_id, p, false/*reentrant*/);
            break;
          }
        default:
          assert(false);
      }
      return manager;
    }

    //--------------------------------------------------------------------------
    MapperManager* Runtime::find_mapper(Processor target, MapperID map_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target.exists());
#endif
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(target);
#ifdef DEBUG_LEGION
      assert(finder != proc_managers.end());
#endif
      return finder->second->find_mapper(map_id);
    }

    //--------------------------------------------------------------------------
    ProjectionID Runtime::generate_dynamic_projection_id(void)
    //--------------------------------------------------------------------------
    {
      ProjectionID result = 
        __sync_fetch_and_add(&unique_projection_id, runtime_stride);
#ifdef DEBUG_LEGION
      // Check for overflow
      assert(result <= unique_projection_id);
#endif
      return result;
    }
    
    //--------------------------------------------------------------------------
    /*static*/ ProjectionID& Runtime::get_current_static_projection_id(void)
    //--------------------------------------------------------------------------
    {
      static ProjectionID current_projection_id = MAX_APPLICATION_PROJECTION_ID;
      return current_projection_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::generate_static_projection_id(void)
    //--------------------------------------------------------------------------
    {
      ProjectionID &next_projection = get_current_static_projection_id();
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'generate_static_projection_id' after "
                      "the runtime has been started!");
      return next_projection++;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_projection_functor(ProjectionID pid,
                                              ProjectionFunctor *functor,
                                              bool need_zero_check,
                                              bool was_preregistered)
    //--------------------------------------------------------------------------
    {
      if (need_zero_check && (pid == 0))
        REPORT_LEGION_ERROR(ERROR_RESERVED_PROJECTION_ID, 
                            "ProjectionID zero is reserved.\n");
      if (!was_preregistered && (total_address_spaces > 1))
        REPORT_LEGION_WARNING(LEGION_WARNING_DYNAMIC_PROJECTION_REG,
                        "Projection functor %d is being dynamically "
                        "registered for a multi-node run with %d nodes. It is "
                        "currently the responsibility of the application to "
                        "ensure that this projection functor is registered on "
                        "all nodes where it will be required.",
                        pid, total_address_spaces)
      ProjectionFunction *function = new ProjectionFunction(pid, functor);
      AutoLock p_lock(projection_lock);
      // No need for a lock because these all need to be reserved at
      // registration time before the runtime starts up
      std::map<ProjectionID,ProjectionFunction*>::
        const_iterator finder = projection_functions.find(pid);
      if (finder != projection_functions.end())
        REPORT_LEGION_ERROR(ERROR_DUPLICATE_PROJECTION_ID, 
                      "ProjectionID %d has already been used in "
                      "the region projection table\n", pid)
      projection_functions[pid] = function;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_projection_function(pid, function->depth);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_projection_functor(ProjectionID pid,
                                                     ProjectionFunctor *functor)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'preregister_projection_functor' after "
                      "the runtime has started!")
      if (pid == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_PROJECTION_ID, 
                            "ProjectionID zero is reserved.\n");
      std::map<ProjectionID,ProjectionFunctor*> &pending_projection_functors =
        get_pending_projection_table();
      std::map<ProjectionID,ProjectionFunctor*>::const_iterator finder = 
        pending_projection_functors.find(pid);
      if (finder != pending_projection_functors.end())
        REPORT_LEGION_ERROR(ERROR_DUPLICATE_PROJECTION_ID, 
                      "ProjectionID %d has already been used in "
                      "the region projection table\n", pid)
      pending_projection_functors[pid] = functor;
    }

    //--------------------------------------------------------------------------
    ProjectionFunction* Runtime::find_projection_function(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(projection_lock,1,false/*exclusive*/);
      std::map<ProjectionID,ProjectionFunction*>::
        const_iterator finder = projection_functions.find(pid);
      if (finder == projection_functions.end())
        REPORT_LEGION_ERROR(ERROR_INVALID_PROJECTION_ID, 
                        "Unable to find registered region projection ID %d. "
                        "Please upgrade to using projection functors!", pid);
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(TaskID task_id, SemanticTag tag,
           const void *buffer, size_t size, bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      if ((tag == NAME_SEMANTIC_TAG) && legion_spy_enabled)
        LegionSpy::log_task_name(task_id, static_cast<const char*>(buffer));
      TaskImpl *impl = find_or_create_task_impl(task_id);
      impl->attach_semantic_information(tag, address_space, buffer, size, 
                                        is_mutable, send_to_owner);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(IndexPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(FieldSpace handle, FieldID fid,
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, fid, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalRegion handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(LogicalPartition handle, 
                                              SemanticTag tag,
                                              const void *buffer, size_t size,
                                              bool is_mutable)
    //--------------------------------------------------------------------------
    {
      forest->attach_semantic_information(handle, tag, address_space, 
                                          buffer, size, is_mutable);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(TaskID task_id,SemanticTag tag,
              const void *&result, size_t &size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = find_or_create_task_impl(task_id);
      return impl->retrieve_semantic_information(tag, result, size, 
                                                 can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(IndexSpace handle,
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
    bool Runtime::retrieve_semantic_information(IndexPartition handle,
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
    bool Runtime::retrieve_semantic_information(FieldSpace handle,
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
    bool Runtime::retrieve_semantic_information(FieldSpace handle, FieldID fid,
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
    bool Runtime::retrieve_semantic_information(LogicalRegion handle,
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
    bool Runtime::retrieve_semantic_information(LogicalPartition handle,
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
    FieldID Runtime::allocate_field(Context ctx, FieldSpace space,
                                          size_t field_size, FieldID fid,
                                          bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context allocate field!");
      return ctx->allocate_field(forest, space, field_size, 
                                 fid, local, serdez_id); 
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field(Context ctx, FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context free field!");
      ctx->free_field(space, fid); 
    }

    //--------------------------------------------------------------------------
    void Runtime::allocate_fields(Context ctx, FieldSpace space,
                                        const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context allocate fields!");
      ctx->allocate_fields(forest, space, sizes, resulting_fields, 
                           local, serdez_id); 
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fields(Context ctx, FieldSpace space,
                                    const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      if (ctx == DUMMY_CONTEXT)
        REPORT_DUMMY_CONTEXT("Illegal dummy context free fields!");
      ctx->free_fields(space, to_free); 
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      TaskID result = __sync_fetch_and_add(&unique_task_id, runtime_stride);
#ifdef DEBUG_LEGION
      // Check for overflow
      assert(result <= unique_task_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID Runtime::register_variant(const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  CodeDescriptor *realm,
                                  bool ret,VariantID vid /*= AUTO_GENERATE_ID*/,
                                  bool check_task_id /*= true*/)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out a way to make this check safe with dynamic generation
#if 0
      if (check_task_id && (registrar.task_id >= MAX_APPLICATION_TASK_ID))
        REPORT_LEGION_ERROR(ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED, 
                      "Error registering task with ID %d. Exceeds the "
                      "statically set bounds on application task IDs of %d. "
                      "See %s in legion_config.h.", 
                      registrar.task_id, MAX_APPLICATION_TASK_ID, 
                      LEGION_MACRO_TO_STRING(MAX_APPLICATION_TASK_ID))
#endif
      // First find the task implementation
      TaskImpl *task_impl = find_or_create_task_impl(registrar.task_id);
      // See if we need to make a new variant ID
      if (vid == AUTO_GENERATE_ID) // Make a variant ID to use
        vid = task_impl->get_unique_variant_id();
      else if (vid == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_VARIANT_ID,
                      "Error registering variant for task ID %d with "
                      "variant ID 0. Variant ID 0 is reserved for task "
                      "generators.", registrar.task_id)
      // Make our variant and add it to the set of variants
      VariantImpl *impl = new VariantImpl(this, vid, task_impl, 
                                          registrar, ret, realm,
                                          user_data, user_data_size);
      // Add this variant to the owner
      task_impl->add_variant(impl);
      {
        AutoLock tv_lock(task_variant_lock);
        variant_table.push_back(impl);
      }
      // If this is a global registration we need to broadcast the variant
      if (registrar.global_registration && (total_address_spaces > 1))
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        impl->broadcast_variant(done_event, address_space, 0);
        done_event.lg_wait();
      }
      if (legion_spy_enabled)
        LegionSpy::log_task_variant(registrar.task_id, vid, 
                                    impl->is_inner(), impl->is_leaf(),
                                    impl->is_idempotent(), impl->get_name());
      return vid;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Runtime::find_or_create_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tv_lock(task_variant_lock,1,false/*exclusive*/);
        std::map<TaskID,TaskImpl*>::const_iterator finder = 
          task_table.find(task_id);
        if (finder != task_table.end())
          return finder->second;
      }
      AutoLock tv_lock(task_variant_lock);
      std::map<TaskID,TaskImpl*>::const_iterator finder = 
        task_table.find(task_id);
      // Check to see if we lost the race
      if (finder == task_table.end())
      {
        TaskImpl *result = new TaskImpl(task_id, this);
        task_table[task_id] = result;
        return result;
      }
      else // Lost the race as it already exists
        return finder->second;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Runtime::find_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      AutoLock tv_lock(task_variant_lock,1,false/*exclusive*/);
      std::map<TaskID,TaskImpl*>::const_iterator finder = 
        task_table.find(task_id);
#ifdef DEBUG_LEGION
      assert(finder != task_table.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    VariantImpl* Runtime::find_variant_impl(TaskID task_id, 
                                             VariantID variant_id,bool can_fail)
    //--------------------------------------------------------------------------
    {
      TaskImpl *owner = find_or_create_task_impl(task_id);
      return owner->find_variant_impl(variant_id, can_fail);
    }

    //--------------------------------------------------------------------------
    MemoryManager* Runtime::find_memory_manager(Memory mem)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock m_lock(memory_manager_lock,1,false/*exclusive*/);
        std::map<Memory,MemoryManager*>::const_iterator finder = 
          memory_managers.find(mem);
        if (finder != memory_managers.end())
          return finder->second;
      }
      // Not there?  Take exclusive lock and check again, create if needed
      AutoLock m_lock(memory_manager_lock);
      std::map<Memory,MemoryManager*>::const_iterator finder =
        memory_managers.find(mem);
      if (finder != memory_managers.end())
        return finder->second;
      // Really do need to create it (and put it in the map)
      MemoryManager *result = new MemoryManager(mem, this);
      memory_managers[mem] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Runtime::find_address_space(Memory handle) const
    //--------------------------------------------------------------------------
    {
      // Just use the standard translation for now
      AddressSpaceID result = handle.address_space();
      return result;
    }

    //--------------------------------------------------------------------------
    MessageManager* Runtime::find_messenger(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(finder != proc_spaces.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_message(Processor target, MapperID map_id,
                                     Processor source, const void *message,
                                     size_t message_size, unsigned message_kind)
    //--------------------------------------------------------------------------
    {
      if (is_local(target))
      {
        Mapper::MapperMessage message_args;
        message_args.sender = source;
        message_args.kind = message_kind;
        message_args.message = message;
        message_args.size = message_size;
        message_args.broadcast = false;
        MapperManager *mapper = find_mapper(target, map_id);
        mapper->invoke_handle_message(&message_args);
      }
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(message_kind);
          rez.serialize(message_size);
          rez.serialize(message, message_size);
        }
        send_mapper_message(find_address_space(target), rez);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_broadcast(MapperID map_id, Processor source, 
                                    const void *message, size_t message_size, 
                                    unsigned message_kind, int radix, int index)
    //--------------------------------------------------------------------------
    {
      // First forward the message onto any remote nodes
      int base = index * radix;
      int init;
      if (Runtime::separate_runtime_instances)
      {
        std::map<Processor,Runtime*>::const_iterator finder = 
          runtime_map->find(source);
#ifdef DEBUG_LEGION
        // only works with a single process
        assert(finder != runtime_map->end()); 
#endif
        init = finder->second->address_space;
      }
      else
        init = source.address_space();
      // The runtime stride is the same as the number of nodes
      const int total_nodes = runtime_stride;
      for (int r = 1; r <= radix; r++)
      {
        int offset = base + r;
        // If we've handled all of our nodes then we are done
        if (offset >= total_nodes)
          break;
        AddressSpaceID target = (init + offset) % total_nodes;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(message_kind);
          rez.serialize(radix);
          rez.serialize(offset);
          rez.serialize(message_size);
          rez.serialize(message, message_size);
        }
        send_mapper_broadcast(target, rez);
      }
      // Then send it to all our local mappers, set will deduplicate
      std::set<MapperManager*> managers;
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        managers.insert(it->second->find_mapper(map_id));
      }
      Mapper::MapperMessage message_args;
      message_args.sender = source;
      message_args.kind = message_kind;
      message_args.message = message;
      message_args.size = message_size;
      message_args.broadcast = true;
      for (std::set<MapperManager*>::const_iterator it = 
            managers.begin(); it != managers.end(); it++)
        (*it)->invoke_handle_message(&message_args);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_task(TaskOp *task)
    //--------------------------------------------------------------------------
    {
      Processor target = task->target_proc;
      if (!target.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_TARGET_PROC, 
                      "Mapper requested invalid NO_PROC as target proc!");
      // Check to see if the target processor is still local 
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(target);
      if (finder != proc_managers.end())
      {
        // Update the current processor
        task->set_current_proc(target);
        finder->second->add_to_ready_queue(task);
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
    void Runtime::send_tasks(Processor target, const std::set<TaskOp*> &tasks)
    //--------------------------------------------------------------------------
    {
      if (!target.exists())
        REPORT_LEGION_ERROR(ERROR_INVALID_TARGET_PROC, 
                      "Mapper requested invalid NO_PROC as target proc!");
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
          (*it)->set_current_proc(target);
          finder->second->add_to_ready_queue(*it);
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
        if (it == targets.end())
          break;
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
          messenger->send_message(rez, ADVERTISEMENT_MESSAGE, 
                                  MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
          already_sent.insert(messenger);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_NODE,
                               INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_REQUEST, 
                                INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_return(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_RETURN,
            INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_set(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_SET,
              INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*return*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_child_request(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_CHILD_REQUEST,
                                INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_child_response(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_CHILD_RESPONSE,
                  INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_colors_request(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_COLORS_REQUEST,
                                INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_colors_response(AddressSpaceID target,
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,SEND_INDEX_SPACE_COLORS_RESPONSE,
                  INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_notification(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
                                  SEND_INDEX_PARTITION_NOTIFICATION, 
                                  INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_node(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_NODE,
                               INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_request(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_REQUEST,
                                INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_return(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_PARTITION_RETURN,
              INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_child_request(AddressSpaceID target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
                                SEND_INDEX_PARTITION_CHILD_REQUEST,
                                INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_child_response(AddressSpaceID target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
                                SEND_INDEX_PARTITION_CHILD_RESPONSE, 
                                INDEX_SPACE_VIRTUAL_CHANNEL, 
                                true/*flush*/, true/*response*/); 
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_node(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_NODE,
                               FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_REQUEST,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_return(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_RETURN,
            FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_alloc_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_ALLOC_REQUEST,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_alloc_notification(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_ALLOC_NOTIFICATION,
                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_top_alloc(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_TOP_ALLOC,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_free(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_FREE,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_local_field_alloc_request(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOCAL_FIELD_ALLOC_REQUEST,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_local_field_alloc_response(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOCAL_FIELD_ALLOC_RESPONSE,
                  FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_local_field_free(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOCAL_FIELD_FREE,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_local_field_update(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOCAL_FIELD_UPDATE,
                                FIELD_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_top_level_region_request(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TOP_LEVEL_REGION_REQUEST,
                                LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_top_level_region_return(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TOP_LEVEL_REGION_RETURN,
                LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_node(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_LOGICAL_REGION_NODE,
                                  LOGICAL_TREE_VIRTUAL_CHANNEL, true/*flush*/);
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
      find_messenger(target)->send_message(rez, INDEX_SPACE_DESTRUCTION_MESSAGE,
                               INDEX_SPACE_VIRTUAL_CHANNEL, true/*flush*/);
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
      find_messenger(target)->send_message(rez, 
        INDEX_PARTITION_DESTRUCTION_MESSAGE, INDEX_SPACE_VIRTUAL_CHANNEL,
                                                             true/*flush*/);
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
      find_messenger(target)->send_message(rez, 
          FIELD_SPACE_DESTRUCTION_MESSAGE, FIELD_SPACE_VIRTUAL_CHANNEL,
                                                              true/*flush*/);
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
      find_messenger(target)->send_message(rez, 
          LOGICAL_REGION_DESTRUCTION_MESSAGE, LOGICAL_TREE_VIRTUAL_CHANNEL,
                                                              true/*flush*/);
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
      find_messenger(target)->send_message(rez, 
          LOGICAL_PARTITION_DESTRUCTION_MESSAGE, LOGICAL_TREE_VIRTUAL_CHANNEL,
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_mapped(Processor target,
                                        Serializer &rez, bool flush /*= true*/)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_MAPPED,
                       DEFAULT_VIRTUAL_CHANNEL, flush, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_complete(Processor target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_COMPLETE,
                  DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_individual_remote_commit(Processor target,
                                                      Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, INDIVIDUAL_REMOTE_COMMIT,
                DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_mapped(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_MAPPED,
           DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_complete(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_COMPLETE,
             DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_slice_remote_commit(Processor target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Very important that this goes on the physical state channel
      // so that it is properly serialized with state updates
      find_messenger(target)->send_message(rez, SLICE_REMOTE_COMMIT,
           DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_registration(AddressSpaceID target, 
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_REMOTE_REGISTRATION,
                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_valid_update(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_VALID_UPDATE,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_gc_update(AddressSpaceID target,
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_GC_UPDATE,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_resource_update(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_RESOURCE_UPDATE,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_invalidate(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_INVALIDATE,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_deactivate(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_DEACTIVATE,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_add_create_reference(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_CREATE_ADD,
                                    REFERENCE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remove_create_reference(AddressSpaceID target,
                                                    Serializer &rez, bool flush)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_CREATE_REMOVE,
                                           REFERENCE_VIRTUAL_CHANNEL, flush);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_did_remote_unregister(AddressSpaceID target, 
                                         Serializer &rez, VirtualChannelKind vc)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, DISTRIBUTED_UNREGISTER,
                                           vc, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_back_logical_state(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // No need to flush, it will get flushed by the remote map return
      find_messenger(target)->send_message(rez, SEND_BACK_LOGICAL_STATE,
                                        DEFAULT_VIRTUAL_CHANNEL,false/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_atomic_reservation_request(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_ATOMIC_RESERVATION_REQUEST,
                                        VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_atomic_reservation_response(AddressSpaceID target,
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,SEND_ATOMIC_RESERVATION_RESPONSE,
                         VIEW_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_materialized_view(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MATERIALIZED_VIEW,
                                        VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_composite_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_COMPOSITE_VIEW,
                                       VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    } 

    //--------------------------------------------------------------------------
    void Runtime::send_fill_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FILL_VIEW,
                                       VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_phi_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_PHI_VIEW,
                                        VIEW_VIRTUAL_CHANNEL, true/*flush*/); 
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_view(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_VIEW,
                                       VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_instance_manager(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INSTANCE_MANAGER,
                                       MANAGER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_reduction_manager(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REDUCTION_MANAGER,
                                       MANAGER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_create_top_view_request(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CREATE_TOP_VIEW_REQUEST,
                                        VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_create_top_view_response(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CREATE_TOP_VIEW_RESPONSE,
                        VIEW_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_subview_did_request(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_SUBVIEW_DID_REQUEST,
                                        VIEW_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_subview_did_response(AddressSpaceID target,
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_SUBVIEW_DID_RESPONSE,
                    VIEW_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_view_update_request(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VIEW_UPDATE_REQUEST,
                                        UPDATE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_view_update_response(AddressSpaceID target,
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VIEW_UPDATE_RESPONSE,
                  UPDATE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_view_remote_update(AddressSpaceID target, 
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VIEW_REMOTE_UPDATE,
                                        UPDATE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_view_remote_invalidate(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VIEW_REMOTE_INVALIDATE,
                                        UPDATE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_result(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_RESULT,
            FUTURE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_subscription(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_SUBSCRIPTION,
                                        FUTURE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_map_request_future(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_MAP_REQUEST,
                                        FUTURE_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_future_map_response_future(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FUTURE_MAP_RESPONSE,
                  FUTURE_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_mapper_message(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MAPPER_MESSAGE,
                                        MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_mapper_broadcast(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MAPPER_BROADCAST,
                                         MAPPER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_task_impl_semantic_request(AddressSpaceID target,
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TASK_IMPL_SEMANTIC_REQ,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_semantic_request(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_SEMANTIC_REQ,
                                 SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_semantic_request(AddressSpaceID target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_INDEX_PARTITION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_semantic_request(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_SEMANTIC_REQ,
                                  SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_semantic_request(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SEMANTIC_REQ,
                                SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_semantic_request(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
              SEND_LOGICAL_REGION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_partition_semantic_request(
                                         AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
            SEND_LOGICAL_PARTITION_SEMANTIC_REQ, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                                                 true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_task_impl_semantic_info(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_TASK_IMPL_SEMANTIC_INFO,
              SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INDEX_SPACE_SEMANTIC_INFO,
               SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_index_partition_semantic_info(AddressSpaceID target,
                                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_INDEX_PARTITION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                             true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_space_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SPACE_SEMANTIC_INFO,
                SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_field_semantic_info(AddressSpaceID target,
                                                 Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_FIELD_SEMANTIC_INFO,
          SEMANTIC_INFO_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_region_semantic_info(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
              SEND_LOGICAL_REGION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                              true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_logical_partition_semantic_info(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
            SEND_LOGICAL_PARTITION_SEMANTIC_INFO, SEMANTIC_INFO_VIRTUAL_CHANNEL,
                                               true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_request(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REMOTE_CONTEXT_REQUEST, 
                                        CONTEXT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_response(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REMOTE_CONTEXT_RESPONSE, 
                    CONTEXT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_release(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REMOTE_CONTEXT_RELEASE,
                                     CONTEXT_VIRTUAL_CHANNEL, true/*flush*/);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_free(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_REMOTE_CONTEXT_FREE,
                                        CONTEXT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_physical_request(AddressSpaceID target,
                                                       Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_REMOTE_CONTEXT_PHYSICAL_REQUEST, 
          CONTEXT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_remote_context_physical_response(AddressSpaceID target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
          SEND_REMOTE_CONTEXT_PHYSICAL_RESPONSE,
          CONTEXT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_owner_request(AddressSpaceID target,
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_OWNER_REQUEST,
                            VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_owner_response(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_OWNER_RESPONSE,
            VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_state_response(AddressSpaceID target,
                                              Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_STATE_RESPONSE,
                    VERSION_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_state_update_request(AddressSpaceID target,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez,
                               SEND_VERSION_STATE_UPDATE_REQUEST,
                               ANALYSIS_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_state_update_response(AddressSpaceID target,
                                                     Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
                                SEND_VERSION_STATE_UPDATE_RESPONSE,
                                ANALYSIS_VIRTUAL_CHANNEL, 
                                true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_state_valid_notification(AddressSpaceID target,
                                                        Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
                                SEND_VERSION_STATE_VALID_NOTIFICATION,
                                ANALYSIS_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_advance(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VERSION_MANAGER_ADVANCE,
                                      ANALYSIS_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_invalidate(AddressSpaceID target,
                                                  Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // This comes back on the version manager channel so that it is 
      // properly ordered with respect to the version manager responses
      find_messenger(target)->send_message(rez, SEND_VERSION_MANAGER_INVALIDATE,
                                VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_request(AddressSpaceID target,
                                               Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // This goes on the analysis virtual channel so that it can 
      // be ordered with respect to advances
      find_messenger(target)->send_message(rez, SEND_VERSION_MANAGER_REQUEST,
                                       ANALYSIS_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_response(AddressSpaceID target,
                                                Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // This comes back on the version manager channel in case we need to page
      // in any version managers from remote nodes
      find_messenger(target)->send_message(rez, SEND_VERSION_MANAGER_RESPONSE,
             VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_unversioned_request(
                                         AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_VERSION_MANAGER_UNVERSIONED_REQUEST, 
          VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_version_manager_unversioned_response(
                                         AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, 
          SEND_VERSION_MANAGER_UNVERSIONED_RESPONSE, 
          VERSION_MANAGER_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::send_instance_request(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INSTANCE_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_instance_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_INSTANCE_RESPONSE,
              DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_gc_priority_update(AddressSpaceID target,Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_GC_PRIORITY_UPDATE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_never_gc_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_NEVER_GC_RESPONSE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_acquire_request(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_ACQUIRE_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::send_acquire_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_ACQUIRE_RESPONSE,
              DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_variant_request(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VARIANT_REQUEST,
                                        VARIANT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_variant_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VARIANT_RESPONSE,
              VARIANT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_variant_broadcast(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_VARIANT_BROADCAST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_constraint_request(AddressSpaceID target, 
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // This is paging in constraints so it needs its own virtual channel
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_REQUEST,
                              LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_constraint_response(AddressSpaceID target, 
                                            Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // This is paging in constraints so it needs its own virtual channel
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_RESPONSE,
        LAYOUT_CONSTRAINT_VIRTUAL_CHANNEL, true/*flush*/, true/*response*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_constraint_release(AddressSpaceID target,
                                           Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_RELEASE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_constraint_removal(AddressSpaceID target,
                                          Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_CONSTRAINT_REMOVAL,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_mpi_rank_exchange(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_MPI_RANK_EXCHANGE,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_shutdown_notification(AddressSpaceID target, 
                                             Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_SHUTDOWN_NOTIFICATION,
                                    DEFAULT_VIRTUAL_CHANNEL, true/*flush*/, 
                                    false/*response*/, true/*shutdown*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_shutdown_response(AddressSpaceID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      find_messenger(target)->send_message(rez, SEND_SHUTDOWN_RESPONSE,
                                DEFAULT_VIRTUAL_CHANNEL, true/*flush*/,
                                false/*response*/, true/*shutdown*/);
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
#ifdef DEBUG_LEGION
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
    void Runtime::handle_index_space_set(Deserializer &derez, 
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_index_space_set(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_child_request(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_child_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_child_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_node_child_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_colors_request(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_colors_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_colors_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_colors_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_notification(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_notification(forest, derez);
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
    void Runtime::handle_index_partition_child_request(Deserializer &derez,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_child_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_child_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_node_child_response(derez);
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
    void Runtime::handle_field_alloc_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_alloc_request(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_alloc_notification(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_alloc_notification(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_top_alloc(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_top_alloc(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_free(Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_free(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_local_field_alloc_request(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_local_alloc_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_local_field_alloc_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_local_alloc_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_local_field_free(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_local_free(forest, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_local_field_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RemoteContext::handle_local_field_update(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_top_level_region_request(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_top_level_request(forest, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_top_level_region_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_top_level_return(derez);   
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
    void Runtime::handle_did_remote_registration(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_registration(this,derez,source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_valid_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_valid_update(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_gc_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_gc_update(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_resource_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_resource_update(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_invalidate(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_invalidate(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_deactivate(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remote_deactivate(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_create_add(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_add_create(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_create_remove(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_did_remove_create(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_did_remote_unregister(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable::handle_unregister_collectable(this, derez);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::handle_send_back_logical_state(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode::handle_logical_state_return(this, derez); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_atomic_reservation_request(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_atomic_reservation_request(this, derez, 
                                                               source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_atomic_reservation_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_atomic_reservation_response(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_materialized_view(Deserializer &derez, 
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_send_materialized_view(this, derez, source); 
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_composite_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      CompositeView::handle_send_composite_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_fill_view(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FillView::handle_send_fill_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_phi_view(Deserializer &derez,
                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PhiView::handle_send_phi_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_view(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionView::handle_send_reduction_view(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_instance_manager(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceManager::handle_send_manager(this, source, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_send_reduction_manager(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReductionManager::handle_send_manager(this, source, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_top_view_request(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InnerContext::handle_create_top_view_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_create_top_view_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      InnerContext::handle_create_top_view_response(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_subview_did_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_subview_did_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_subview_did_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      MaterializedView::handle_subview_did_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_view_request(Deserializer &derez, 
                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LogicalView::handle_view_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_view_update_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_view_update_request(derez, this, source);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::handle_view_update_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_view_update_response(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_view_remote_update(Deserializer &derez,
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_view_remote_update(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_view_remote_invalidate(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      InstanceView::handle_view_remote_invalidate(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_manager_request(Deserializer &derez, 
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PhysicalManager::handle_manager_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FutureImpl::handle_future_result(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_subscription(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FutureImpl::handle_future_subscription(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_map_future_request(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FutureMapImpl::handle_future_map_future_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_future_map_future_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      FutureMapImpl::handle_future_map_future_response(derez, this);
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
      unsigned message_kind;
      derez.deserialize(message_kind);
      size_t message_size;
      derez.deserialize(message_size);
      const void *message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      process_mapper_message(target, map_id, source, message, 
                             message_size, message_kind);
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
      unsigned message_kind;
      derez.deserialize(message_kind);
      int radix;
      derez.deserialize(radix);
      int index;
      derez.deserialize(index);
      size_t message_size;
      derez.deserialize(message_size);
      const void *message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      process_mapper_broadcast(map_id, source, message, 
                               message_size, message_kind, radix, index);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_task_impl_semantic_request(Deserializer &derez,
                                                     AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_semantic_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_semantic_request(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_semantic_request(Deserializer &derez, 
                                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_semantic_request(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_semantic_request(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_region_semantic_request(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_partition_semantic_request(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_semantic_request(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_task_impl_semantic_info(Deserializer &derez,
                                                  AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_semantic_info(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_space_semantic_info(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_index_partition_semantic_info(Deserializer &derez, 
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexPartNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_space_semantic_info(Deserializer &derez,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_field_semantic_info(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode::handle_field_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_region_semantic_info(Deserializer &derez,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RegionNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_logical_partition_semantic_info(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PartitionNode::handle_semantic_info(forest, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_request(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      RemoteContext *target;
      derez.deserialize(target);
      InnerContext *context = find_context(context_uid);
      context->send_remote_context(source, target);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteContext *context;
      derez.deserialize(context);
      // Unpack the result
      std::set<RtEvent> preconditions;
      context->unpack_remote_context(derez, preconditions);
      // Then register it
      UniqueID context_uid = context->get_context_uid();
      register_remote_context(context_uid, context, preconditions);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_release(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      InnerContext *context = find_context(context_uid);
      context->invalidate_remote_tree_contexts(derez);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_free(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID remote_owner_uid;
      derez.deserialize(remote_owner_uid);
      unregister_remote_context(remote_owner_uid);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_physical_request(Deserializer &derez,
                                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RemoteContext::handle_physical_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_remote_context_physical_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RemoteContext::handle_physical_response(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_owner_request(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      InnerContext::handle_version_owner_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_owner_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      InnerContext::handle_version_owner_response(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_state_request(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::handle_version_state_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_state_response(Deserializer &derez,
                                                AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::handle_version_state_response(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_state_update_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_update_request(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_state_update_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_update_response(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_state_valid_notification(Deserializer &derez,
                                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionState::process_version_state_valid_notification(derez,this,source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_advance(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_remote_advance(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_invalidate(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_remote_invalidate(derez, this);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_request(Deserializer &derez,
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_unversioned_request(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_unversioned_request(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_version_manager_unversioned_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VersionManager::handle_unversioned_response(derez, this);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::handle_instance_request(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_instance_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_instance_response(Deserializer &derez,
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_instance_response(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_gc_priority_update(Deserializer &derez,
                                            AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_gc_priority_update(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_never_gc_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_never_gc_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_acquire_request(Deserializer &derez, 
                                         AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_acquire_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_acquire_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Memory target_memory;
      derez.deserialize(target_memory);
      MemoryManager *manager = find_memory_manager(target_memory);
      manager->process_acquire_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_variant_request(Deserializer &derez, 
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      TaskImpl::handle_variant_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_variant_response(Deserializer &derez) 
    //--------------------------------------------------------------------------
    {
      VariantImpl::handle_variant_response(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_variant_broadcast(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      VariantImpl::handle_variant_broadcast(this, derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_constraint_request(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints::process_request(this, derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_constraint_response(Deserializer &derez,
                                             AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      LayoutConstraintID to_remove = 
        LayoutConstraints::process_response(this, derez, source);
      // Remove the done event from our set of pending events
      AutoLock l_lock(layout_constraints_lock);
      pending_constraint_requests.erase(to_remove);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_constraint_release(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      release_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_constraint_removal(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Deserializer z(derez);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      unregister_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_top_level_task_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(address_space == 0); // should only happen on node 0
#endif
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      increment_outstanding_top_level_tasks();
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_top_level_task_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(address_space == 0); // should only happen on node 0
#endif
      decrement_outstanding_top_level_tasks();
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_mpi_rank_exchange(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(Runtime::mpi_rank_table != NULL);
#endif
      Runtime::mpi_rank_table->handle_mpi_rank_exchange(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_shutdown_notification(Deserializer &derez,
                                               AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShutdownManager::handle_shutdown_notification(derez, this, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_shutdown_response(Deserializer &derez) 
    //--------------------------------------------------------------------------
    {
      ShutdownManager::handle_shutdown_response(derez);
    }

    //--------------------------------------------------------------------------
    bool Runtime::create_physical_instance(Memory target_memory,
                                     const LayoutConstraintSet &constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result,
                                     MapperID mapper_id, Processor processor, 
                                     bool acquire, GCPriority priority,
                                     UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->create_physical_instance(constraints, regions, result,
                       mapper_id, processor, acquire, priority, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::create_physical_instance(Memory target_memory,
                                     LayoutConstraintID layout_id,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result,
                                     MapperID mapper_id, Processor processor,
                                     bool acquire, GCPriority priority,
                                     UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->create_physical_instance(constraints, regions, result,
                       mapper_id, processor, acquire, priority, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_or_create_physical_instance(Memory target_memory,
                                     const LayoutConstraintSet &constraints,
                                     const std::vector<LogicalRegion> &regions,
                                     MappingInstance &result, bool &created, 
                                     MapperID mapper_id, Processor processor,
                                     bool acquire, GCPriority priority,
                                     bool tight_bounds, UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->find_or_create_physical_instance(constraints, regions, 
                             result, created, mapper_id, processor, acquire, 
                             priority, tight_bounds, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_or_create_physical_instance(Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    MapperID mapper_id, Processor processor,
                                    bool acquire, GCPriority priority,
                                    bool tight_bounds, UniqueID creator_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->find_or_create_physical_instance(constraints, regions,
                             result, created, mapper_id, processor, acquire, 
                             priority, tight_bounds, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_physical_instance(Memory target_memory,
                                      const LayoutConstraintSet &constraints,
                                      const std::vector<LogicalRegion> &regions,
                                      MappingInstance &result, bool acquire,
                                      bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->find_physical_instance(constraints, regions, 
                             result, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_physical_instance(Memory target_memory,
                                      LayoutConstraintID layout_id,
                                      const std::vector<LogicalRegion> &regions,
                                      MappingInstance &result, bool acquire,
                                      bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      MemoryManager *manager = find_memory_manager(target_memory);
      return manager->find_physical_instance(constraints, regions, 
                                     result, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_tree_instances(RegionTreeID tree_id)
    //--------------------------------------------------------------------------
    {
      std::map<Memory,MemoryManager*> copy_managers;
      {
        AutoLock m_lock(memory_manager_lock,1,false/*exclusive*/);
        copy_managers = memory_managers;
      }
      for (std::map<Memory,MemoryManager*>::const_iterator it = 
            copy_managers.begin(); it != copy_managers.end(); it++)
        it->second->release_tree_instances(tree_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::process_schedule_request(Processor proc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
    void Runtime::process_message_task(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      const char *buffer = (const char*)args;
      AddressSpaceID sender = *((const AddressSpaceID*)buffer);
      buffer += sizeof(sender);
      arglen -= sizeof(sender);
      find_messenger(sender)->receive_message(buffer, arglen);
    }

    //--------------------------------------------------------------------------
    void Runtime::activate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it =
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->activate_context(context);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::deactivate_context(InnerContext *context)
    //--------------------------------------------------------------------------
    {
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->deactivate_context(context);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::add_to_dependence_queue(TaskContext *ctx,
                                          Processor p, Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(p.kind() != Processor::UTIL_PROC);
#endif
      // Launch the task to perform the prepipeline stage for the operation
      RtEvent precondition = op->issue_prepipeline_stage();
      if (program_order_execution)
      {
        ApEvent term_event = op->get_completion_event();
        ctx->add_to_dependence_queue(op, precondition);
        ctx->begin_task_wait(true/*from runtime*/);
        term_event.lg_wait();
        ctx->end_task_wait();
      }
      else
        ctx->add_to_dependence_queue(op, precondition);
    }
    
    //--------------------------------------------------------------------------
    void Runtime::add_to_ready_queue(Processor p, TaskOp *op, RtEvent wait_on)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(p.kind() != Processor::UTIL_PROC);
      assert(proc_managers.find(p) != proc_managers.end());
#endif
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        DeferredEnqueueArgs args;
        args.manager = proc_managers[p];
        args.task = op;
        issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY, op,wait_on);
      }
      else
        proc_managers[p]->add_to_ready_queue(op);
    }

    //--------------------------------------------------------------------------
    void Runtime::add_to_local_queue(Processor p, Operation *op, 
                                     LgPriority priority, RtEvent wait_on)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(p.kind() != Processor::UTIL_PROC);
      assert(proc_managers.find(p) != proc_managers.end());
#endif
      proc_managers[p]->add_to_local_ready_queue(op, priority, wait_on);
    }

    //--------------------------------------------------------------------------
    Processor Runtime::find_processor_group(const std::vector<Processor> &procs)
    //--------------------------------------------------------------------------
    {
      // Compute a hash of all the processor ids to avoid testing all sets 
      // Only need to worry about local IDs since all processors are
      // in this address space.
      ProcessorMask local_mask = find_processor_mask(procs);
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
    ProcessorMask Runtime::find_processor_mask(
                                            const std::vector<Processor> &procs)
    //--------------------------------------------------------------------------
    {
      ProcessorMask result;
      std::vector<Processor> need_allocation;
      {
        AutoLock p_lock(processor_mapping_lock,1,false/*exclusive*/);
        for (std::vector<Processor>::const_iterator it = procs.begin();
              it != procs.end(); it++)
        {
          std::map<Processor,unsigned>::const_iterator finder = 
            processor_mapping.find(*it);
          if (finder == processor_mapping.end())
          {
            need_allocation.push_back(*it);
            continue;
          }
          result.set_bit(finder->second);
        }
      }
      if (need_allocation.empty())
        return result;
      AutoLock p_lock(processor_mapping_lock);
      for (std::vector<Processor>::const_iterator it = 
            need_allocation.begin(); it != need_allocation.end(); it++)
      {
        // Check to make sure we didn't lose the race
        std::map<Processor,unsigned>::const_iterator finder = 
            processor_mapping.find(*it);
        if (finder != processor_mapping.end())
        {
          result.set_bit(finder->second);
          continue;
        }
        unsigned next_index = processor_mapping.size();
#ifdef DEBUG_LEGION
        assert(next_index < MAX_NUM_PROCS);
#endif
        processor_mapping[*it] = next_index;
        result.set_bit(next_index);
      }
      return result;
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
#ifdef DEBUG_LEGION
      assert(result < LEGION_DISTRIBUTED_ID_MASK);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_distributed_id(DistributedID did)
    //--------------------------------------------------------------------------
    {
      // Don't recycle distributed IDs if we're doing LegionSpy or LegionGC
#ifndef LEGION_GC
#ifndef LEGION_SPY
      AutoLock d_lock(distributed_id_lock);
      available_distributed_ids.push_back(did);
#endif
#endif
#ifdef DEBUG_LEGION
      AutoLock dist_lock(distributed_collectable_lock,1,false/*exclusive*/);
      assert(dist_collectables.find(did) == dist_collectables.end());
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent Runtime::recycle_distributed_id(DistributedID did,
                                            RtEvent recycle_event)
    //--------------------------------------------------------------------------
    {
      // Special case for did 0 on shutdown
      if (did == 0)
        return RtEvent::NO_RT_EVENT;
      did &= LEGION_DISTRIBUTED_ID_MASK;
#ifdef DEBUG_LEGION
      // Should only be getting back our own DIDs
      assert(determine_owner(did) == address_space);
#endif
      if (!recycle_event.has_triggered())
      {
        DeferredRecycleArgs deferred_recycle_args;
        deferred_recycle_args.did = did;
        return issue_runtime_meta_task(deferred_recycle_args, 
                LG_THROUGHPUT_WORK_PRIORITY, NULL, recycle_event);
      }
      else
      {
        free_distributed_id(did);
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Runtime::determine_owner(DistributedID did) const
    //--------------------------------------------------------------------------
    {
      return ((did & LEGION_DISTRIBUTED_ID_MASK) % runtime_stride);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_distributed_collectable(DistributedID did,
                                                   DistributedCollectable *dc)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      RtUserEvent to_trigger;
      {
        AutoLock dc_lock(distributed_collectable_lock);
        // If we make it here then we have the lock
#ifdef DEBUG_LEGION
        assert(dist_collectables.find(did) == dist_collectables.end());
#endif
        dist_collectables[did] = dc;
        // See if this was a pending collectable
        std::map<DistributedID,
                 std::pair<DistributedCollectable*,RtUserEvent> >::iterator 
            finder = pending_collectables.find(did);
        if (finder != pending_collectables.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second.first == dc);
#endif
          to_trigger = finder->second.second;
          pending_collectables.erase(finder);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock);
#ifdef DEBUG_LEGION
      assert(dist_collectables.find(did) != dist_collectables.end());
#endif
      dist_collectables.erase(did);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      return (dist_collectables.find(did) != dist_collectables.end());
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::find_distributed_collectable(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      const DistributedID to_find = LEGION_DISTRIBUTED_ID_FILTER(did);
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,DistributedCollectable*>::const_iterator finder = 
        dist_collectables.find(to_find);
#ifdef DEBUG_LEGION
      if (finder == dist_collectables.end())
        log_run.error("Unable to find distributed collectable %llx "
                    "with type %lld", did, LEGION_DISTRIBUTED_HELP_DECODE(did));
      assert(finder != dist_collectables.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::find_distributed_collectable(
                                              DistributedID did, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      const DistributedID to_find = LEGION_DISTRIBUTED_ID_FILTER(did);
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,DistributedCollectable*>::const_iterator finder = 
        dist_collectables.find(to_find);
      if (finder == dist_collectables.end())
      {
        // Check to see if it is in the pending set too
        std::map<DistributedID,
          std::pair<DistributedCollectable*,RtUserEvent> >::const_iterator
            pending_finder = pending_collectables.find(to_find);
        if (pending_finder != pending_collectables.end())
        {
          ready = pending_finder->second.second;
          return pending_finder->second.first;
        }
      }
#ifdef DEBUG_LEGION
      if (finder == dist_collectables.end())
        log_run.error("Unable to find distributed collectable %llx "
                    "with type %lld", did, LEGION_DISTRIBUTED_HELP_DECODE(did));
      assert(finder != dist_collectables.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::weak_find_distributed_collectable(
                                                              DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
      std::map<DistributedID,DistributedCollectable*>::const_iterator finder = 
        dist_collectables.find(did);
      if (finder == dist_collectables.end())
        return NULL;
      return finder->second;
    } 

    //--------------------------------------------------------------------------
    bool Runtime::find_pending_collectable_location(DistributedID did,
                                                    void *&location)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(dist_collectables.find(did) == dist_collectables.end());
#endif
      std::map<DistributedID,std::pair<DistributedCollectable*,RtUserEvent> >::
        const_iterator finder = pending_collectables.find(did);
      if (finder != pending_collectables.end())
      {
        location = finder->second.first;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    LogicalView* Runtime::find_or_request_logical_view(DistributedID did,
                                                       RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable *dc = NULL;
      if (LogicalView::is_materialized_did(did))
        dc = find_or_request_distributed_collectable<
         MaterializedView, SEND_VIEW_REQUEST, VIEW_VIRTUAL_CHANNEL>(did, ready);
      else if (LogicalView::is_reduction_did(did))
        dc = find_or_request_distributed_collectable<
          ReductionView, SEND_VIEW_REQUEST, VIEW_VIRTUAL_CHANNEL>(did, ready);
      else if (LogicalView::is_composite_did(did))
        dc = find_or_request_distributed_collectable<
          CompositeView, SEND_VIEW_REQUEST, VIEW_VIRTUAL_CHANNEL>(did, ready);
      else if (LogicalView::is_fill_did(did))
        dc = find_or_request_distributed_collectable<
          FillView, SEND_VIEW_REQUEST, VIEW_VIRTUAL_CHANNEL>(did, ready);
      else
        assert(false);
      // Have to static cast since the memory might not have been initialized
      return static_cast<LogicalView*>(dc);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* Runtime::find_or_request_physical_manager(
                                              DistributedID did, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable *dc = NULL;
      if (PhysicalManager::is_instance_did(did))
        dc = find_or_request_distributed_collectable<
          InstanceManager, SEND_MANAGER_REQUEST, MANAGER_VIRTUAL_CHANNEL>(did, 
                                                                        ready);
      else if (PhysicalManager::is_reduction_fold_did(did))
        dc = find_or_request_distributed_collectable<
          FoldReductionManager, SEND_MANAGER_REQUEST, MANAGER_VIRTUAL_CHANNEL>(
                                                                    did, ready);
      else if (PhysicalManager::is_reduction_list_did(did))
        dc = find_or_request_distributed_collectable<
          ListReductionManager, SEND_MANAGER_REQUEST, MANAGER_VIRTUAL_CHANNEL>(
                                                                    did, ready);
      else
        assert(false);
      // Have to static cast since the memory might not have been initialized
      return static_cast<PhysicalManager*>(dc);
    }

    //--------------------------------------------------------------------------
    VersionState* Runtime::find_or_request_version_state(DistributedID did,
                                                         RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable *dc = find_or_request_distributed_collectable<
        VersionState, SEND_VERSION_STATE_REQUEST, VERSION_VIRTUAL_CHANNEL>(did,
                                                                        ready);
      // Have to static cast since the memory might not have been initialized
      return static_cast<VersionState*>(dc);
    }

    //--------------------------------------------------------------------------
    template<typename T, MessageKind MK, VirtualChannelKind VC>
    DistributedCollectable* Runtime::find_or_request_distributed_collectable(
                                              DistributedID did, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      DistributedCollectable *result = NULL;
      {
        AutoLock d_lock(distributed_collectable_lock);
        std::map<DistributedID,DistributedCollectable*>::const_iterator finder =
          dist_collectables.find(did);
        // If we've already got it, then we are done
        if (finder != dist_collectables.end())
        {
          ready = RtEvent::NO_RT_EVENT;
          return finder->second;
        }
        // If it is already pending, we can just return the ready event
        std::map<DistributedID,std::pair<DistributedCollectable*,RtUserEvent> 
          >::const_iterator pending_finder = pending_collectables.find(did);
        if (pending_finder != pending_collectables.end())
        {
          ready = pending_finder->second.second;
          return pending_finder->second.first;
        }
        // This is the first request we've seen for this did, make it now
        // Allocate space for the result and type case
        result = (T*)legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);  
        RtUserEvent to_trigger = Runtime::create_rt_user_event();
        pending_collectables[did] = 
          std::pair<DistributedCollectable*,RtUserEvent>(result, to_trigger);
        ready = to_trigger;
      }
      AddressSpaceID target = determine_owner(did);
#ifdef DEBUG_LEGION
      assert(target != address_space); // shouldn't be sending to ourself
#endif
      // Now send the message
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
      }
      find_messenger(target)->send_message(rez, MK, VC, true/*flush*/);
      return result;
    }
    
    //--------------------------------------------------------------------------
    FutureImpl* Runtime::find_or_create_future(DistributedID did,
                                               ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK; 
      {
        AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
        std::map<DistributedID,DistributedCollectable*>::const_iterator 
          finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
#ifdef DEBUG_LEGION
          FutureImpl *result = dynamic_cast<FutureImpl*>(finder->second);
          assert(result != NULL);
#else
          FutureImpl *result = static_cast<FutureImpl*>(finder->second);
#endif
          return result;
        }
      }
      AddressSpaceID owner_space = determine_owner(did);
      FutureImpl *result = new FutureImpl(this, false/*register*/, 
                                          did, owner_space);
      // Retake the lock and see if we lost the race
      {
        AutoLock d_lock(distributed_collectable_lock);
        std::map<DistributedID,DistributedCollectable*>::const_iterator 
          finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
          // We lost the race
          delete (result);
#ifdef DEBUG_LEGION
          result = dynamic_cast<FutureImpl*>(finder->second);
          assert(result != NULL);
#else
          result = static_cast<FutureImpl*>(finder->second);
#endif
          return result;
        }
        dist_collectables[did] = result;
      }
      result->record_future_registered(mutator);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* Runtime::find_or_create_future_map(DistributedID did,
                                    TaskContext *ctx, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      {
        AutoLock d_lock(distributed_collectable_lock,1,false/*exclusive*/);
        std::map<DistributedID,DistributedCollectable*>::const_iterator 
          finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
#ifdef DEBUG_LEGION
          FutureMapImpl *result = dynamic_cast<FutureMapImpl*>(finder->second);
          assert(result != NULL);
#else
          FutureMapImpl *result = static_cast<FutureMapImpl*>(finder->second);
#endif
          return result;
        }
      }
      AddressSpaceID owner_space = determine_owner(did);
      FutureMapImpl *result = new FutureMapImpl(ctx, this, did,
                                    owner_space, false/*register now */);
      // Retake the lock and see if we lost the race
      {
        AutoLock d_lock(distributed_collectable_lock);
        std::map<DistributedID,DistributedCollectable*>::const_iterator 
          finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
          // We lost the race
          delete (result);
#ifdef DEBUG_LEGION
          result = dynamic_cast<FutureMapImpl*>(finder->second);
          assert(result != NULL);
#else
          result = static_cast<FutureMapImpl*>(finder->second);
#endif
          return result;
        }
        dist_collectables[did] = result;
      }
      result->record_future_map_registered(mutator);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::find_or_create_index_launch_space(const Domain &dom)
    //--------------------------------------------------------------------------
    {
      switch (dom.get_dim())
      {
        case 1:
          {
            DomainT<1,coord_t> is = dom;
            return find_or_create_index_launch_space(dom, &is,
                  NT_TemplateHelper::encode_tag<1,coord_t>());
          }
        case 2:
          {
            DomainT<2,coord_t> is = dom;
            return find_or_create_index_launch_space(dom, &is,
                  NT_TemplateHelper::encode_tag<2,coord_t>());
          }
        case 3:
          {
            DomainT<3,coord_t> is = dom;
            return find_or_create_index_launch_space(dom, &is,
                  NT_TemplateHelper::encode_tag<3,coord_t>());
          }
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    } 

    //--------------------------------------------------------------------------
    IndexSpace Runtime::find_or_create_index_launch_space(const Domain &dom,
                                                          const void *realm_is,
                                                          TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      const std::pair<Domain,TypeTag> key(dom, type_tag);
      {
        AutoLock is_lock(is_launch_lock,1,false/*exclusive*/);
        std::map<std::pair<Domain,TypeTag>,IndexSpace>::const_iterator finder =
          index_launch_spaces.find(key);
        if (finder != index_launch_spaces.end())
          return finder->second;
      }
      IndexSpace result(get_unique_index_space_id(),
                        get_unique_index_tree_id(), type_tag);
      DistributedID did = get_available_distributed_id();
      forest->create_index_space(result, realm_is, did);
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_top_index_space(result.id);
      // Overwrite and leak for now, don't care too much as this 
      // should occur infrequently
      AutoLock is_lock(is_launch_lock);
      index_launch_spaces[key] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::defer_collect_user(LogicalView *view, ApEvent term_event,
                                     ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      GarbageCollectionEpoch *to_trigger = NULL;
      {
        AutoLock gc(gc_epoch_lock);
        current_gc_epoch->add_collection(view, term_event, mutator);
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
        to_trigger->launch();
    }

    //--------------------------------------------------------------------------
    void Runtime::complete_gc_epoch(GarbageCollectionEpoch *epoch)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_epoch_lock);
#ifdef DEBUG_LEGION
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
      // Check to see if we are on node 0 or not
      if (address_space != 0)
      {
        // Send a message to node 0 requesting permission to 
        // lauch a new top-level task and wait on an event
        // to signal that permission has been granted
        RtUserEvent grant_event = Runtime::create_rt_user_event();
        Serializer rez;
        rez.serialize(grant_event);
        find_messenger(0)->send_message(rez, SEND_TOP_LEVEL_TASK_REQUEST,
                                        DEFAULT_VIRTUAL_CHANNEL, true/*flush*/);
        grant_event.lg_wait();
      }
      else
      {
        __sync_fetch_and_add(&outstanding_top_level_tasks,1);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_outstanding_top_level_tasks(void)
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
#ifdef DEBUG_LEGION
        assert(prev > 0);
#endif
        // Check to see if we have no more outstanding top-level tasks
        // If we don't launch a task to handle the try to shutdown the runtime 
        if (prev == 1)
          issue_runtime_shutdown_attempt();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_runtime_shutdown_attempt(void)
    //--------------------------------------------------------------------------
    {
      ShutdownManager::RetryShutdownArgs args;
      args.phase = ShutdownManager::CHECK_TERMINATION;
      // Issue this with a low priority so that other meta-tasks
      // have an opportunity to run
      issue_runtime_meta_task(args, LG_LOW_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void Runtime::initiate_runtime_shutdown(AddressSpaceID source,
                                           ShutdownManager::ShutdownPhase phase,
                                           ShutdownManager *owner)
    //--------------------------------------------------------------------------
    {
      log_shutdown.info("Received notification on node %d for phase %d",
                        address_space, phase);
      // If this is the first phase, do all our normal stuff
      if (phase == ShutdownManager::CHECK_TERMINATION)
      {
        // Launch our last garbage collection epoch and wait for it to
        // finish so we can try to have no outstanding tasks
        RtEvent gc_done;
        {
          AutoLock gc(gc_epoch_lock);
          if (current_gc_epoch != NULL)
          {
            gc_done = current_gc_epoch->launch();
            current_gc_epoch = NULL;
          }
        }
        gc_done.lg_wait();
      }
      else if ((phase == ShutdownManager::CHECK_SHUTDOWN) && 
                !prepared_for_shutdown)
      {
        // First time we check for shutdown we do the prepare for shutdown
        prepare_runtime_shutdown();  
      }
      ShutdownManager *shutdown_manager = 
        new ShutdownManager(phase, this, source, 
                            LEGION_SHUTDOWN_RADIX, owner);
      if (shutdown_manager->attempt_shutdown())
        delete shutdown_manager;
    }

    //--------------------------------------------------------------------------
    void Runtime::confirm_runtime_shutdown(ShutdownManager *shutdown_manager, 
                                           bool phase_one)
    //--------------------------------------------------------------------------
    {
      if (has_outstanding_tasks())
      {
        shutdown_manager->record_outstanding_tasks();
#ifdef DEBUG_LEGION
        LG_TASK_DESCRIPTIONS(meta_task_names);
        AutoLock out_lock(outstanding_task_lock,1,false/*exclusive*/);
        for (std::map<std::pair<unsigned,bool>,unsigned>::const_iterator it =
              outstanding_task_counts.begin(); it != 
              outstanding_task_counts.end(); it++)
        {
          if (it->second == 0)
            continue;
          if (it->first.second)
            log_shutdown.info("RT %d: %d outstanding meta task(s) %s",
                              address_space, it->second, 
                              meta_task_names[it->first.first]);
          else                
            log_shutdown.info("RT %d: %d outstanding application task(s) %d",
                              address_space, it->second, it->first.first);
        }
#endif
      }
      // Check all our message managers for outstanding messages
      for (unsigned idx = 0; idx < MAX_NUM_NODES; idx++)
      {
        if (message_managers[idx] != NULL)
          message_managers[idx]->confirm_shutdown(shutdown_manager, phase_one);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::prepare_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!prepared_for_shutdown);
#endif
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
        it->second->prepare_for_shutdown();
      for (std::map<Memory,MemoryManager*>::const_iterator it = 
            memory_managers.begin(); it != memory_managers.end(); it++)
        it->second->prepare_for_shutdown();
      forest->prepare_for_shutdown();
      prepared_for_shutdown = true;
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(address_space == 0); // only happens on node 0
#endif
      std::set<RtEvent> shutdown_events;
      // Launch tasks to shutdown all the runtime instances
      Machine::ProcessorQuery all_procs(machine);
      Realm::ProfilingRequestSet empty_requests;
      if (Runtime::separate_runtime_instances)
      {
        // If we are doing separate runtime instances, run it once on every
        // processor since we have separate runtimes for every processor
        for (Machine::ProcessorQuery::iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          shutdown_events.insert(
              RtEvent(it->spawn(SHUTDOWN_TASK_ID, NULL, 0, empty_requests)));
        }
      }
      else
      {
        // In the normal case we just have to run this once on every node
        std::set<AddressSpace> shutdown_spaces;
        for (Machine::ProcessorQuery::iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          AddressSpace space = it->address_space();
          if (shutdown_spaces.find(space) == shutdown_spaces.end())
          {
            shutdown_events.insert(
                RtEvent(it->spawn(SHUTDOWN_TASK_ID, NULL, 0, empty_requests)));
            shutdown_spaces.insert(space);
          }
        }
      }
      // Then tell Realm to shutdown when they are all done
      RtEvent shutdown_precondition = Runtime::merge_events(shutdown_events);
      RealmRuntime realm = RealmRuntime::get_runtime();
      realm.shutdown(shutdown_precondition);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      AutoLock out_lock(outstanding_task_lock);
      return (total_outstanding_tasks > 0);
#else
      return (__sync_fetch_and_add(&total_outstanding_tasks,0) != 0);
#endif
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void Runtime::increment_total_outstanding_tasks(unsigned tid, bool meta)
    //--------------------------------------------------------------------------
    {
      AutoLock out_lock(outstanding_task_lock); 
      total_outstanding_tasks++;
      std::pair<unsigned,bool> key(tid,meta);
      std::map<std::pair<unsigned,bool>,unsigned>::iterator finder = 
        outstanding_task_counts.find(key);
      if (finder == outstanding_task_counts.end())
        outstanding_task_counts[key] = 1;
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_total_outstanding_tasks(unsigned tid, bool meta)
    //--------------------------------------------------------------------------
    {
      AutoLock out_lock(outstanding_task_lock);
      assert(total_outstanding_tasks > 0);
      total_outstanding_tasks--;
      std::pair<unsigned,bool> key(tid,meta);
      std::map<std::pair<unsigned,bool>,unsigned>::iterator finder = 
        outstanding_task_counts.find(key);
      assert(finder != outstanding_task_counts.end());
      assert(finder->second > 0);
      finder->second--;
    }
#endif

    //--------------------------------------------------------------------------
    IndividualTask* Runtime::get_available_individual_task(void)
    //--------------------------------------------------------------------------
    {
      IndividualTask *result = get_available(individual_task_lock, 
                                         available_individual_tasks);
#ifdef DEBUG_LEGION
      AutoLock i_lock(individual_task_lock);
      out_individual_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PointTask* Runtime::get_available_point_task(void)
    //--------------------------------------------------------------------------
    {
      PointTask *result = get_available(point_task_lock, 
                                        available_point_tasks);
#ifdef DEBUG_LEGION
      AutoLock p_lock(point_task_lock);
      out_point_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    IndexTask* Runtime::get_available_index_task(void)
    //--------------------------------------------------------------------------
    {
      IndexTask *result = get_available(index_task_lock, 
                                       available_index_tasks);
#ifdef DEBUG_LEGION
      AutoLock i_lock(index_task_lock);
      out_index_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    SliceTask* Runtime::get_available_slice_task(void)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = get_available(slice_task_lock,
                                       available_slice_tasks);
#ifdef DEBUG_LEGION
      AutoLock s_lock(slice_task_lock);
      out_slice_tasks.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    MapOp* Runtime::get_available_map_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(map_op_lock, available_map_ops);
    }

    //--------------------------------------------------------------------------
    CopyOp* Runtime::get_available_copy_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(copy_op_lock, available_copy_ops);
    }

    //--------------------------------------------------------------------------
    IndexCopyOp* Runtime::get_available_index_copy_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(copy_op_lock, available_index_copy_ops);
    }

    //--------------------------------------------------------------------------
    PointCopyOp* Runtime::get_available_point_copy_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(copy_op_lock, available_point_copy_ops);
    }

    //--------------------------------------------------------------------------
    FenceOp* Runtime::get_available_fence_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(fence_op_lock, available_fence_ops);
    }

    //--------------------------------------------------------------------------
    FrameOp* Runtime::get_available_frame_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(frame_op_lock, available_frame_ops);
    }

    //--------------------------------------------------------------------------
    DeletionOp* Runtime::get_available_deletion_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(deletion_op_lock, available_deletion_ops);
    }

    //--------------------------------------------------------------------------
    OpenOp* Runtime::get_available_open_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(open_op_lock, available_open_ops);
    }

    //--------------------------------------------------------------------------
    AdvanceOp* Runtime::get_available_advance_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(advance_op_lock, available_advance_ops);
    }

    //--------------------------------------------------------------------------
    InterCloseOp* Runtime::get_available_inter_close_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(inter_close_op_lock, available_inter_close_ops);
    }

    //--------------------------------------------------------------------------
    ReadCloseOp* Runtime::get_available_read_close_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(read_close_op_lock, available_read_close_ops);
    }

    //--------------------------------------------------------------------------
    PostCloseOp* Runtime::get_available_post_close_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(post_close_op_lock, available_post_close_ops);
    }

    //--------------------------------------------------------------------------
    VirtualCloseOp* Runtime::get_available_virtual_close_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(virtual_close_op_lock, available_virtual_close_ops);
    }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp* Runtime::get_available_dynamic_collective_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(dynamic_collective_op_lock, 
                           available_dynamic_collective_ops);
    }

    //--------------------------------------------------------------------------
    FuturePredOp* Runtime::get_available_future_pred_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(future_pred_op_lock, available_future_pred_ops);
    }

    //--------------------------------------------------------------------------
    NotPredOp* Runtime::get_available_not_pred_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(not_pred_op_lock, available_not_pred_ops);
    }

    //--------------------------------------------------------------------------
    AndPredOp* Runtime::get_available_and_pred_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(and_pred_op_lock, available_and_pred_ops);
    }

    //--------------------------------------------------------------------------
    OrPredOp* Runtime::get_available_or_pred_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(or_pred_op_lock, available_or_pred_ops);
    }

    //--------------------------------------------------------------------------
    AcquireOp* Runtime::get_available_acquire_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(acquire_op_lock, available_acquire_ops);
    }

    //--------------------------------------------------------------------------
    ReleaseOp* Runtime::get_available_release_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(release_op_lock, available_release_ops);
    }

    //--------------------------------------------------------------------------
    TraceCaptureOp* Runtime::get_available_capture_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(capture_op_lock, available_capture_ops);
    }

    //--------------------------------------------------------------------------
    TraceCompleteOp* Runtime::get_available_trace_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(trace_op_lock, available_trace_ops);
    }

    //--------------------------------------------------------------------------
    MustEpochOp* Runtime::get_available_epoch_op(void)
    //--------------------------------------------------------------------------
    {
      MustEpochOp *result = get_available(epoch_op_lock, available_epoch_ops);
#ifdef DEBUG_LEGION
      AutoLock e_lock(epoch_op_lock);
      out_must_epoch.insert(result);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    PendingPartitionOp* Runtime::get_available_pending_partition_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(pending_partition_op_lock, 
                           available_pending_partition_ops);
    }

    //--------------------------------------------------------------------------
    DependentPartitionOp* Runtime::get_available_dependent_partition_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(dependent_partition_op_lock, 
                           available_dependent_partition_ops);
    }

    //--------------------------------------------------------------------------
    PointDepPartOp* Runtime::get_available_point_dep_part_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(dependent_partition_op_lock,
                           available_point_dep_part_ops);
    }

    //--------------------------------------------------------------------------
    FillOp* Runtime::get_available_fill_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(fill_op_lock, available_fill_ops);
    }

    //--------------------------------------------------------------------------
    IndexFillOp* Runtime::get_available_index_fill_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(fill_op_lock, available_index_fill_ops);
    }

    //--------------------------------------------------------------------------
    PointFillOp* Runtime::get_available_point_fill_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(fill_op_lock, available_point_fill_ops);
    }

    //--------------------------------------------------------------------------
    AttachOp* Runtime::get_available_attach_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(attach_op_lock, available_attach_ops);
    }

    //--------------------------------------------------------------------------
    DetachOp* Runtime::get_available_detach_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(detach_op_lock, available_detach_ops);
    }

    //--------------------------------------------------------------------------
    TimingOp* Runtime::get_available_timing_op(void)
    //--------------------------------------------------------------------------
    {
      return get_available(timing_op_lock, available_timing_ops);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_individual_task(IndividualTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(individual_task_lock);
      release_operation<false>(available_individual_tasks, task);
#ifdef DEBUG_LEGION
      out_individual_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_task(PointTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(point_task_lock);
#ifdef DEBUG_LEGION
      out_point_tasks.erase(task);
#endif
      // Note that we can safely delete point tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      release_operation<true>(available_point_tasks, task);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_index_task(IndexTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(index_task_lock);
      release_operation<false>(available_index_tasks, task);
#ifdef DEBUG_LEGION
      out_index_tasks.erase(task);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::free_slice_task(SliceTask *task)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(slice_task_lock);
#ifdef DEBUG_LEGION
      out_slice_tasks.erase(task);
#endif
      // Note that we can safely delete slice tasks because they are
      // never registered in the logical state of the region tree
      // as part of the dependence analysis. This does not apply
      // to all operation objects.
      release_operation<true>(available_slice_tasks, task);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_map_op(MapOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(map_op_lock);
      release_operation<false>(available_map_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_copy_op(CopyOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(copy_op_lock);
      release_operation<false>(available_copy_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_index_copy_op(IndexCopyOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(copy_op_lock);
      release_operation<false>(available_index_copy_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_copy_op(PointCopyOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(copy_op_lock);
      release_operation<true>(available_point_copy_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fence_op(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fence_op_lock);
      release_operation<false>(available_fence_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_frame_op(FrameOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(frame_op_lock);
      release_operation<false>(available_frame_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_deletion_op(DeletionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(deletion_op_lock);
      release_operation<false>(available_deletion_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_open_op(OpenOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(open_op_lock);
      release_operation<false>(available_open_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_advance_op(AdvanceOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(advance_op_lock);
      release_operation<false>(available_advance_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_inter_close_op(InterCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inter_close_op_lock);
      release_operation<false>(available_inter_close_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_read_close_op(ReadCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(read_close_op_lock);
      release_operation<false>(available_read_close_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_post_close_op(PostCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(post_close_op_lock);
      release_operation<false>(available_post_close_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_virtual_close_op(VirtualCloseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(virtual_close_op_lock);
      release_operation<false>(available_virtual_close_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_dynamic_collective_op(DynamicCollectiveOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock dc_lock(dynamic_collective_op_lock);
      release_operation<false>(available_dynamic_collective_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_future_predicate_op(FuturePredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(future_pred_op_lock);
      release_operation<false>(available_future_pred_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_not_predicate_op(NotPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(not_pred_op_lock);
      release_operation<false>(available_not_pred_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_and_predicate_op(AndPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(and_pred_op_lock);
      release_operation<false>(available_and_pred_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_or_predicate_op(OrPredOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(or_pred_op_lock);
      release_operation<false>(available_or_pred_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_acquire_op(AcquireOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(acquire_op_lock);
      release_operation<false>(available_acquire_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_release_op(ReleaseOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(release_op_lock);
      release_operation<false>(available_release_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_capture_op(TraceCaptureOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(capture_op_lock);
      release_operation<false>(available_capture_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_trace_op(TraceCompleteOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(trace_op_lock);
      release_operation<false>(available_trace_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_epoch_op(MustEpochOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(epoch_op_lock);
      release_operation<false>(available_epoch_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_pending_partition_op(PendingPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(pending_partition_op_lock);
      release_operation<false>(available_pending_partition_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_dependent_partition_op(DependentPartitionOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(dependent_partition_op_lock);
      release_operation<false>(available_dependent_partition_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_dep_part_op(PointDepPartOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(dependent_partition_op_lock);
      release_operation<true>(available_point_dep_part_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fill_op(FillOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fill_op_lock);
      release_operation<false>(available_fill_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_index_fill_op(IndexFillOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fill_op_lock);
      release_operation<false>(available_index_fill_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_point_fill_op(PointFillOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock f_lock(fill_op_lock);
      release_operation<true>(available_point_fill_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_attach_op(AttachOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(attach_op_lock);
      release_operation<false>(available_attach_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_detach_op(DetachOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(detach_op_lock);
      release_operation<false>(available_detach_ops, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_timing_op(TimingOp *op)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(timing_op_lock);
      release_operation<false>(available_timing_ops, op);
    }

    //--------------------------------------------------------------------------
    RegionTreeContext Runtime::allocate_region_tree_context(void)
    //--------------------------------------------------------------------------
    {
      // Try getting something off the list of available contexts
      AutoLock ctx_lock(context_lock);
      if (!available_contexts.empty())
      {
        RegionTreeContext result = available_contexts.front();
        available_contexts.pop_front();
        return result;
      }
      // If we failed to get a context, double the number of total 
      // contexts and then update the forest nodes to have the right
      // number of contexts available
      RegionTreeContext result(total_contexts);
      for (unsigned idx = 1; idx < total_contexts; idx++)
        available_contexts.push_back(RegionTreeContext(total_contexts+idx));
      // Mark that we doubled the total number of contexts
      // Very important that we do this before calling the
      // RegionTreeForest's resize method!
      total_contexts *= 2;
#ifdef DEBUG_LEGION
      assert(!available_contexts.empty());
#endif
      // Tell all the processor managers about the additional contexts
      for (std::map<Processor,ProcessorManager*>::const_iterator it = 
            proc_managers.begin(); it != proc_managers.end(); it++)
      {
        it->second->update_max_context_count(total_contexts); 
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_region_tree_context(RegionTreeContext context)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
      forest->check_context_state(context);
#endif
      AutoLock ctx_lock(context_lock);
      available_contexts.push_back(context);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_local_context(UniqueID context_uid,InnerContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((context_uid % runtime_stride) == address_space); // sanity check
#endif
      AutoLock ctx_lock(context_lock);
#ifdef DEBUG_LEGION
      assert(local_contexts.find(context_uid) == local_contexts.end());
#endif
      local_contexts[context_uid] = ctx;
    }
    
    //--------------------------------------------------------------------------
    void Runtime::unregister_local_context(UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((context_uid % runtime_stride) == address_space); // sanity check
#endif
      AutoLock ctx_lock(context_lock);
      std::map<UniqueID,InnerContext*>::iterator finder = 
        local_contexts.find(context_uid);
#ifdef DEBUG_LEGION
      assert(finder != local_contexts.end());
#endif
      local_contexts.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_remote_context(UniqueID context_uid, 
                       RemoteContext *context, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock ctx_lock(context_lock);
        std::map<UniqueID,RtUserEvent>::iterator finder = 
          pending_remote_contexts.find(context_uid);
#ifdef DEBUG_LEGION
        assert(remote_contexts.find(context_uid) == remote_contexts.end());
        assert(finder != pending_remote_contexts.end());
#endif
        to_trigger = finder->second;
        pending_remote_contexts.erase(finder);
        remote_contexts[context_uid] = context; 
      }
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      if (!preconditions.empty())
        Runtime::trigger_event(to_trigger,Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_remote_context(UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
      RemoteContext *context = NULL;
      {
        AutoLock ctx_lock(context_lock);
        std::map<UniqueID,RemoteContext*>::iterator finder = 
          remote_contexts.find(context_uid);
#ifdef DEBUG_LEGION
        assert(finder != remote_contexts.end());
#endif
        context = finder->second;
        remote_contexts.erase(finder);
      }
      // Remove our reference and delete it if we're done with it
      if (context->remove_reference())
        delete context;
    }

    //--------------------------------------------------------------------------
    InnerContext* Runtime::find_context(UniqueID context_uid,
                                      bool return_null_if_not_found /*=false*/)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent ready_event;
      {
        // Need exclusive permission since we might mutate stuff
        AutoLock ctx_lock(context_lock);
        // See if it is local first
        std::map<UniqueID,InnerContext*>::const_iterator
          local_finder = local_contexts.find(context_uid);
        if (local_finder != local_contexts.end())
          return local_finder->second;
        // Now see if it is remote
        std::map<UniqueID,RemoteContext*>::const_iterator
          remote_finder = remote_contexts.find(context_uid);
        if (remote_finder != remote_contexts.end())
          return remote_finder->second;
        // If we don't have it, see if we should send the response or not
        std::map<UniqueID,RtUserEvent>::const_iterator pending_finder = 
          pending_remote_contexts.find(context_uid);
        if (pending_finder == pending_remote_contexts.end())
        {
          // If its not here and we are supposed to return null do that
          if (return_null_if_not_found)
            return NULL;
          // Make an event to trigger for when we are done
          ready_event = Runtime::create_rt_user_event();
          pending_remote_contexts[context_uid] = ready_event; 
        }
        else // if we're going to have it we might as well wait
          wait_on = pending_finder->second;
      }
      // If there is no wait event, we have to send the message
      if (!wait_on.exists())
      {
#ifdef DEBUG_LEGION
        assert(ready_event.exists());
#endif
        // We have to send the message
        // Figure out the target
        AddressSpaceID target = get_runtime_owner(context_uid);
#ifdef DEBUG_LEGION
        assert(target != address_space);
#endif
        // Make the result
        RemoteContext *result = new RemoteContext(this, context_uid);
        // Send the message
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(context_uid);
          rez.serialize(result);
        }
        send_remote_context_request(target, rez);
        // Add a reference to the newly created context
        result->add_reference();
        // Wait for it to be ready
        ready_event.lg_wait();
        // We already know the answer cause we sent the message
        return result;
      }
      // Can't wait in some cases
      if (return_null_if_not_found && !wait_on.has_triggered())
        return NULL;
      // We wait for the results to be ready
      wait_on.lg_wait();
      // When we wake up the context should be here
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
      std::map<UniqueID,RemoteContext*>::const_iterator finder = 
        remote_contexts.find(context_uid);
#ifdef DEBUG_LEGION
      assert(finder != remote_contexts.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_local(Processor proc) const
    //--------------------------------------------------------------------------
    {
      return (local_procs.find(proc) != local_procs.end());
    }

    //--------------------------------------------------------------------------
    void Runtime::find_visible_memories(Processor proc, 
                                        std::set<Memory> &visible)
    //--------------------------------------------------------------------------
    {
      // If we cached it locally for our processors, then just go
      // ahead and get the result
      std::map<Processor,ProcessorManager*>::const_iterator finder = 
        proc_managers.find(proc);
      if (finder != proc_managers.end())
      {
        finder->second->find_visible_memories(visible);
        return;
      }
      // Otherwise look up the result
      Machine::MemoryQuery visible_memories(machine);
      // Have to handle the case where this is a processor group
      if (proc.kind() == Processor::PROC_GROUP)
      {
        std::vector<Processor> group_members;
        proc.get_group_members(group_members);
        for (std::vector<Processor>::const_iterator it = 
              group_members.begin(); it != group_members.end(); it++)
          visible_memories.has_affinity_to(*it);
      }
      else
        visible_memories.has_affinity_to(proc);
      for (Machine::MemoryQuery::iterator it = visible_memories.begin();
            it != visible_memories.end(); it++)
        visible.insert(*it);
    }

    //--------------------------------------------------------------------------
    IndexSpaceID Runtime::get_unique_index_space_id(void)
    //--------------------------------------------------------------------------
    {
      IndexSpaceID result = __sync_fetch_and_add(&unique_index_space_id,
                                                 runtime_stride);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      // check for overflow
      assert(result <= unique_field_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    CodeDescriptorID Runtime::get_unique_code_descriptor_id(void)
    //--------------------------------------------------------------------------
    {
      CodeDescriptorID result = __sync_fetch_and_add(&unique_code_descriptor_id,
                                                     runtime_stride);
#ifdef DEBUG_LEGION
      // check for overflow
      assert(result <= unique_code_descriptor_id);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID Runtime::get_unique_constraint_id(void)
    //--------------------------------------------------------------------------
    {
      LayoutConstraintID result = __sync_fetch_and_add(&unique_constraint_id,
                                                       runtime_stride);
#ifdef DEBUG_LEGION
      // check for overflow
      assert(result <= unique_constraint_id);
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
        if (req.region.get_tree_id() != req.parent.get_tree_id())
          return ERROR_INVALID_REGION_HANDLE;
      }
      else
      {
        if (!forest->has_node(req.partition))
          return ERROR_INVALID_PARTITION_HANDLE;
        if (req.partition.get_tree_id() != req.parent.get_tree_id())
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
      return Future(new FutureImpl(this, true/*register*/,
                                   get_available_distributed_id(),
                                   address_space, op));
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
            "total=%d total_bytes=%ld diff=%d diff_bytes=%lld",
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
        case STATIC_TRACE_ALLOC:
          return "Static Trace";
        case DYNAMIC_TRACE_ALLOC:
          return "Dynamic Trace";
        case ALLOC_MANAGER_ALLOC:
          return "Allocation Manager";
        case ALLOC_INTERNAL_ALLOC:
          return "Allocation Internal";
        case TASK_ARGS_ALLOC:
          return "Task Arguments";
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
        case PHI_VIEW_ALLOC:
          return "Phi View";
        case INDIVIDUAL_TASK_ALLOC:
          return "Individual Task";
        case POINT_TASK_ALLOC:
          return "Point Task";
        case INDEX_TASK_ALLOC:
          return "Index Task";
        case SLICE_TASK_ALLOC:
          return "Slice Task";
        case TOP_TASK_ALLOC:
          return "Top Level Task";
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
        case OPEN_OP_ALLOC:
          return "Open Op";
        case ADVANCE_OP_ALLOC:
          return "Advance Op";
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
        case MEMORY_GARBAGE_ALLOC:
          return "Memory Garbage Instances";
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
        case SEMANTIC_INFO_ALLOC:
          return "Semantic Information";
        case DIRECTORY_ALLOC:
          return "State Directory";
        case DENSE_INDEX_ALLOC:
          return "Dense Index Set";
        case CURRENT_STATE_ALLOC:
          return "Current State";
        case VERSION_MANAGER_ALLOC:
          return "Version Manager";
        case PHYSICAL_STATE_ALLOC:
          return "Physical State";
        case VERSION_STATE_ALLOC:
          return "Version State";
        case AGGREGATE_VERSION_ALLOC:
          return "Aggregate Version";
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

#ifdef DEBUG_LEGION
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
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::map<UniqueID,IndividualTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        ApEvent completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Individual Task %lld: %p %s (" IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id); 
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
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::map<UniqueID,IndexTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        ApEvent completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Index Task %lld: %p %s (" IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id); 
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
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::map<UniqueID,SliceTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        ApEvent completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Slice Task %lld: %p %s (" IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id); 
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
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::map<UniqueID,PointTask*>::const_iterator it = 
            out_tasks.begin(); (it != out_tasks.end()); it++)
      {
        ApEvent completion = it->second->get_completion_event();
        fprintf(f,"Outstanding Point Task %lld: %p %s (" IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id); 
        if (cnt > 0)
          cnt--;
        else if (cnt == 0)
          break;
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
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::set<IndexTask*>::const_iterator it = 
            out_index_tasks.begin(); it !=
            out_index_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::set<SliceTask*>::const_iterator it = 
            out_slice_tasks.begin(); it !=
            out_slice_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::set<PointTask*>::const_iterator it = 
            out_point_tasks.begin(); it !=
            out_point_tasks.end(); it++)
      {
        out_tasks[(*it)->get_unique_id()] = *it;
      }
      for (std::map<UniqueID,TaskOp*>::const_iterator it = 
            out_tasks.begin(); it != out_tasks.end(); it++)
      {
        ApEvent completion = it->second->get_completion_event();
        switch (it->second->get_task_kind())
        {
          case TaskOp::INDIVIDUAL_TASK_KIND:
            {
              fprintf(f,"Outstanding Individual Task %lld: %p %s (" 
                        IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id);
              break;
            }
          case TaskOp::POINT_TASK_KIND:
            {
              fprintf(f,"Outstanding Point Task %lld: %p %s (" 
                        IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id);
              break;
            }
          case TaskOp::INDEX_TASK_KIND:
            {
              fprintf(f,"Outstanding Index Task %lld: %p %s (" 
                        IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id);
              break;
            }
          case TaskOp::SLICE_TASK_KIND:
            {
              fprintf(f,"Outstanding Slice Task %lld: %p %s (" 
                        IDFMT ")\n",
                it->first, it->second, it->second->get_task_name(),
                completion.id);
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
    LayoutConstraintID Runtime::register_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      if (layout_id == AUTO_GENERATE_ID)
        layout_id = get_unique_constraint_id();
      // Now make our entry and then return the result
      LayoutConstraints *constraints = 
        new LayoutConstraints(layout_id, this, registrar);
      register_layout(constraints);
      return layout_id;
    }

    //--------------------------------------------------------------------------
    LayoutConstraints* Runtime::register_layout(FieldSpace handle,
                                                const LayoutConstraintSet &cons)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = new LayoutConstraints(
          get_unique_constraint_id(), this, cons, handle);
      register_layout(constraints);
      return constraints;
    }

    //--------------------------------------------------------------------------
    bool Runtime::register_layout(LayoutConstraints *new_constraints)
    //--------------------------------------------------------------------------
    {
      new_constraints->add_reference();
      AutoLock l_lock(layout_constraints_lock);
      std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator finder =
        layout_constraints_table.find(new_constraints->layout_id);
      if (finder != layout_constraints_table.end())
        return false;
      layout_constraints_table[new_constraints->layout_id] = new_constraints;
      return true;
    }

    //--------------------------------------------------------------------------
    void Runtime::release_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      // Check to see if this is the owner
      if (constraints->is_owner())
      {
        // Send the remove message to all the remove nodes
        constraints->release_remote_instances();
      }
      else
      {
        // Send a message to the owner asking it to do the release
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(layout_id);
        }
        send_constraint_release(constraints->owner_space, rez);
      }
      unregister_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = NULL;
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID,LayoutConstraints*>::iterator finder = 
          layout_constraints_table.find(layout_id);
        if (finder != layout_constraints_table.end())
        {
          constraints = finder->second;
          layout_constraints_table.erase(finder);
        }
      }
      if ((constraints != NULL) && constraints->remove_reference())
        delete (constraints);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID Runtime::preregister_layout(
                                     const LayoutConstraintRegistrar &registrar,
                                     LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    { 
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'preregister_layout' after "
                      "the runtime has started!");
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
          REPORT_LEGION_ERROR(ERROR_RESERVED_CONSTRAINT_ID, 
                        "Illegal use of reserved constraint ID 0");
        // Check to make sure it is not already used
        std::map<LayoutConstraintID,LayoutConstraintRegistrar>::const_iterator
          finder = pending_constraints.find(layout_id);
        if (finder != pending_constraints.end())
          REPORT_LEGION_ERROR(ERROR_DUPLICATE_CONSTRAINT_ID, 
                        "Duplicate use of constraint ID %ld", layout_id);
      }
      pending_constraints[layout_id] = registrar;
      return layout_id;
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::get_layout_constraint_field_space(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      return constraints->get_field_space();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_layout_constraints(LayoutConstraintID layout_id,
                                        LayoutConstraintSet &layout_constraints)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      layout_constraints = *constraints;
    }

    //--------------------------------------------------------------------------
    const char* Runtime::get_layout_constraints_name(
                                                   LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints *constraints = find_layout_constraints(layout_id);
      return constraints->get_name();
    }

    //--------------------------------------------------------------------------
    LayoutConstraints* Runtime::find_layout_constraints(
                        LayoutConstraintID layout_id, bool can_fail /*= false*/)
    //--------------------------------------------------------------------------
    {
      // See if we can find it first
      RtEvent wait_on;
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
          finder = layout_constraints_table.find(layout_id);
        if (finder != layout_constraints_table.end())
        {
          return finder->second;
        }
        else
        {
          // See if a request has already been issued
          std::map<LayoutConstraintID,RtEvent>::const_iterator
            wait_on_finder = pending_constraint_requests.find(layout_id);
          if (can_fail || 
              (wait_on_finder == pending_constraint_requests.end()))
          {
            // Ask for the constraints
            AddressSpaceID target = 
              LayoutConstraints::get_owner_space(layout_id, this); 
            RtUserEvent to_trigger = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(layout_id);
              rez.serialize(to_trigger);
              rez.serialize(can_fail);
            }
            // Send the message
            send_constraint_request(target, rez);
            // Only save the event to wait on if this can't fail
            if (!can_fail)
              pending_constraint_requests[layout_id] = to_trigger;
            wait_on = to_trigger;
          }
          else
            wait_on = wait_on_finder->second;
        }
      }
      // If we didn't find it send a remote request for the constraints
      wait_on.lg_wait();
      // When we wake up, the result should be there
      AutoLock l_lock(layout_constraints_lock);
      std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
          finder = layout_constraints_table.find(layout_id);
      if (finder == layout_constraints_table.end())
      {
        if (can_fail)
          return NULL;
#ifdef DEBUG_LEGION
        assert(finder != layout_constraints_table.end());
#endif
      }
      return finder->second;
    }

    /*static*/ Runtime* Runtime::the_runtime = NULL;
    /*static*/ std::map<Processor,Runtime*>* Runtime::runtime_map = NULL;
    /*static*/ std::vector<RegistrationCallbackFnptr> 
                                             Runtime::registration_callbacks;
    /*static*/ Processor::TaskFuncID Runtime::legion_main_id = 0;
    /*static*/ int Runtime::initial_task_window_size = 
                                      DEFAULT_MAX_TASK_WINDOW;
    /*static*/ unsigned Runtime::initial_task_window_hysteresis =
                                      DEFAULT_TASK_WINDOW_HYSTERESIS;
    /*static*/ unsigned Runtime::initial_tasks_to_schedule = 
                                      DEFAULT_MIN_TASKS_TO_SCHEDULE;
    /*static*/ unsigned Runtime::max_message_size = 
                                      DEFAULT_MAX_MESSAGE_SIZE;
    /*static*/ unsigned Runtime::gc_epoch_size = 
                                      DEFAULT_GC_EPOCH_SIZE;
    /*static*/ unsigned Runtime::max_local_fields = 
                                      DEFAULT_LOCAL_FIELDS;
    /*static*/ bool Runtime::runtime_started = false;
    /*static*/ bool Runtime::runtime_backgrounded = false;
    /*static*/ bool Runtime::runtime_warnings = false;
    /*static*/ bool Runtime::separate_runtime_instances = false;
    /*static*/ bool Runtime::record_registration = false;
    /*sattic*/ bool Runtime::stealing_disabled = false;
    /*static*/ bool Runtime::resilient_mode = false;
    /*static*/ bool Runtime::unsafe_launch = false;
#ifdef DEBUG_LEGION
    /*static*/ bool Runtime::unsafe_mapper = false;
#else
    /*static*/ bool Runtime::unsafe_mapper = true;
#endif
    /*static*/ bool Runtime::dynamic_independence_tests = true;
    /*static*/ bool Runtime::legion_spy_enabled = false;
    /*static*/ bool Runtime::enable_test_mapper = false;
    /*static*/ bool Runtime::legion_ldb_enabled = false;
    /*static*/ const char* Runtime::replay_file = NULL;
    /*static*/ int Runtime::legion_collective_radix = 
                                     LEGION_COLLECTIVE_RADIX; 
    /*static*/ int Runtime::legion_collective_log_radix = 0;
    /*static*/ int Runtime::legion_collective_stages = 0;
    /*static*/ int Runtime::legion_collective_participating_spaces = 0;
    /*static*/ int Runtime::legion_collective_last_radix = 
                                     LEGION_COLLECTIVE_RADIX;
    /*static*/ int Runtime::legion_collective_last_log_radix = 0;
    /*static*/ int Runtime::mpi_rank = -1;
    /*static*/ MPIRankTable* Runtime::mpi_rank_table = NULL;
    /*static*/ std::vector<MPILegionHandshake>* 
                      Runtime::pending_handshakes = NULL;
    /*static*/ bool Runtime::program_order_execution = false;
    /*static*/ bool Runtime::verify_disjointness = false;
#ifdef DEBUG_LEGION
    /*static*/ bool Runtime::logging_region_tree_state = false;
    /*static*/ bool Runtime::verbose_logging = false;
    /*static*/ bool Runtime::logical_logging_only = false;
    /*static*/ bool Runtime::physical_logging_only = false;
    /*static*/ bool Runtime::check_privileges = true;
    /*static*/ bool Runtime::bit_mask_logging = false;
#endif
    /*static*/ unsigned Runtime::num_profiling_nodes = 0;
    /*static*/ const char* Runtime::serializer_type = "binary";
    /*static*/ const char* Runtime::prof_logfile = NULL;
    /*static*/ size_t Runtime::prof_footprint_threshold = 128 << 20;
    /*static*/ size_t Runtime::prof_target_latency = 100;
    /*static*/ bool Runtime::slow_debug_ok = false;
#ifdef TRACE_ALLOCATION
    /*static*/ std::map<AllocationType,Runtime::AllocationTracker>
                                        Runtime::allocation_manager;
    /*static*/ unsigned long long Runtime::allocation_tracing_count = 0;
#endif

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::start(int argc, char **argv, bool background)
    //--------------------------------------------------------------------------
    {
      // Some static asserts that need to hold true for the runtime to work
      LEGION_STATIC_ASSERT(MAX_RETURN_SIZE > 0);
      LEGION_STATIC_ASSERT((1 << LEGION_FIELD_LOG2) == MAX_FIELDS);
      LEGION_STATIC_ASSERT(MAX_NUM_NODES > 0);
      LEGION_STATIC_ASSERT(MAX_NUM_PROCS > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_TASK_WINDOW > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MIN_TASKS_TO_SCHEDULE > 0);
      LEGION_STATIC_ASSERT(DEFAULT_MAX_MESSAGE_SIZE > 0); 

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
#ifndef DISABLE_PARTITION_SHIM
      // Preregister any partition shim task variants we need 
      PartitionShim::ColorPoints<1>::register_task();
      PartitionShim::ColorPoints<2>::register_task();
      PartitionShim::ColorPoints<3>::register_task();
      PartitionShim::ColorRects<1,1>::register_task();
      PartitionShim::ColorRects<1,2>::register_task();
      PartitionShim::ColorRects<1,3>::register_task();
      PartitionShim::ColorRects<2,1>::register_task();
      PartitionShim::ColorRects<2,2>::register_task();
      PartitionShim::ColorRects<2,3>::register_task();
      PartitionShim::ColorRects<3,1>::register_task();
      PartitionShim::ColorRects<3,2>::register_task();
      PartitionShim::ColorRects<3,3>::register_task();
#endif
      
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
        the_runtime = NULL;
        runtime_map = NULL;
        mpi_rank_table = NULL;
        runtime_warnings = false;
        separate_runtime_instances = false;
        record_registration = false;
        stealing_disabled = false;
        resilient_mode = false;
        unsafe_launch = false;
#ifdef DEBUG_LEGION
        unsafe_mapper = false;
#else
        unsafe_mapper = true;
#endif
        // We always turn this on as the Legion Spy will 
        // now understand how to handle it.
        dynamic_independence_tests = true;
#ifdef LEGION_SPY
        legion_spy_enabled = true;
#else
        legion_spy_enabled = false;
#endif
        enable_test_mapper = false;
        legion_ldb_enabled = false;
        replay_file = NULL;
        initial_task_window_size = DEFAULT_MAX_TASK_WINDOW;
        initial_task_window_hysteresis = DEFAULT_TASK_WINDOW_HYSTERESIS;
        initial_tasks_to_schedule = DEFAULT_MIN_TASKS_TO_SCHEDULE;
        max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
        gc_epoch_size = DEFAULT_GC_EPOCH_SIZE;
        max_local_fields = DEFAULT_LOCAL_FIELDS;
        program_order_execution = false;
        verify_disjointness = false;
        num_profiling_nodes = 0;
        serializer_type = "binary";
        prof_logfile = NULL;
        prof_footprint_threshold = 128 << 20;
        prof_target_latency = 100;
	slow_debug_ok = false;
        legion_collective_radix = LEGION_COLLECTIVE_RADIX;
        legion_collective_log_radix = 0;
        legion_collective_stages = 0;
        legion_collective_participating_spaces = 0;
        legion_collective_last_radix = LEGION_COLLECTIVE_RADIX;
        legion_collective_last_log_radix = 0;
#ifdef DEBUG_LEGION
        logging_region_tree_state = false;
        verbose_logging = false;
        logical_logging_only = false;
        physical_logging_only = false;
        check_privileges = true;
        bit_mask_logging = false;
#endif
        unsigned delay_start = 0;
        for (int i = 1; i < argc; i++)
        {
          BOOL_ARG("-lg:warn",runtime_warnings);
          BOOL_ARG("-lg:separate",separate_runtime_instances);
          BOOL_ARG("-lg:registration",record_registration);
          BOOL_ARG("-lg:nosteal",stealing_disabled);
          BOOL_ARG("-lg:resilient",resilient_mode);
          BOOL_ARG("-lg:unsafe_launch",unsafe_launch);
          BOOL_ARG("-lg:unsafe_mapper",unsafe_mapper);
          if (!strcmp(argv[i],"-lg:safe_mapper"))
            unsafe_mapper = false;
          BOOL_ARG("-lg:inorder",program_order_execution);
          BOOL_ARG("-lg:disjointness",verify_disjointness);
          INT_ARG("-lg:window", initial_task_window_size);
          INT_ARG("-lg:hysteresis", initial_task_window_hysteresis);
          INT_ARG("-lg:sched", initial_tasks_to_schedule);
          INT_ARG("-lg:message",max_message_size);
          INT_ARG("-lg:epoch", gc_epoch_size);
          INT_ARG("-lg:local", max_local_fields);
          if (!strcmp(argv[i],"-lg:no_dyn"))
            dynamic_independence_tests = false;
          BOOL_ARG("-lg:spy",legion_spy_enabled);
          BOOL_ARG("-lg:test",enable_test_mapper);
          INT_ARG("-lg:delay", delay_start);
          if (!strcmp(argv[i],"-lg:replay"))
          {
            replay_file = argv[++i];
            continue;
          }
          if (!strcmp(argv[i],"-lg:ldb"))
          {
            replay_file = argv[++i];
            legion_ldb_enabled = true;
            continue;
          }
#ifdef DEBUG_LEGION
          BOOL_ARG("-lg:tree",logging_region_tree_state);
          BOOL_ARG("-lg:verbose",verbose_logging);
          BOOL_ARG("-lg:logical_only",logical_logging_only);
          BOOL_ARG("-lg:physical_only",physical_logging_only);
          BOOL_ARG("-lg:bit_masks",bit_mask_logging);
#else
          if (!strcmp(argv[i],"-lg:tree"))
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_REGION_TREE_STATE_LOGGING,
                                  "Region tree state logging is "
                          "disabled.  To enable region tree state logging "
                                                  "compile in debug mode.");
          }
          if (!strcmp(argv[i],"-lg:disjointness"))
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_DISJOINTNESS_VERIFICATION,
                                  "Disjointness verification for "
                      "partition creation is disabled.  To enable dynamic "
                              "disjointness testing compile in debug mode.");
          }
#endif
          INT_ARG("-lg:prof", num_profiling_nodes);
          if (!strcmp(argv[i],"-lg:serializer"))
          {
            serializer_type = argv[++i];
            continue;
          }
          if (!strcmp(argv[i],"-lg:prof_logfile"))
          {
            prof_logfile = argv[++i];
            continue;
          }
          if (!strcmp(argv[i],"-lg:prof_footprint"))
          {
            prof_footprint_threshold = atoi(argv[++i]) << 20;
            continue;
          }
          INT_ARG("-lg:prof_latency",prof_target_latency);

	  BOOL_ARG("-lg:debug_ok",slow_debug_ok);
          
          // These are all the deprecated versions of these flag
          BOOL_ARG("-hl:separate",separate_runtime_instances);
          BOOL_ARG("-hl:registration",record_registration);
          BOOL_ARG("-hl:nosteal",stealing_disabled);
          BOOL_ARG("-hl:resilient",resilient_mode);
          BOOL_ARG("-hl:unsafe_launch",unsafe_launch);
          BOOL_ARG("-hl:unsafe_mapper",unsafe_mapper);
          if (!strcmp(argv[i],"-hl:safe_mapper"))
            unsafe_mapper = false;
          BOOL_ARG("-hl:inorder",program_order_execution);
          BOOL_ARG("-hl:disjointness",verify_disjointness);
          INT_ARG("-hl:window", initial_task_window_size);
          INT_ARG("-hl:hysteresis", initial_task_window_hysteresis);
          INT_ARG("-hl:sched", initial_tasks_to_schedule);
          INT_ARG("-hl:message",max_message_size);
          INT_ARG("-hl:epoch", gc_epoch_size);
          if (!strcmp(argv[i],"-hl:no_dyn"))
            dynamic_independence_tests = false;
          BOOL_ARG("-hl:spy",legion_spy_enabled);
          BOOL_ARG("-hl:test",enable_test_mapper);
          INT_ARG("-hl:delay", delay_start);
          if (!strcmp(argv[i],"-hl:replay"))
          {
            replay_file = argv[++i];
            continue;
          }
          if (!strcmp(argv[i],"-hl:ldb"))
          {
            replay_file = argv[++i];
            legion_ldb_enabled = true;
            continue;
          }
#ifdef DEBUG_LEGION
          BOOL_ARG("-hl:tree",logging_region_tree_state);
          BOOL_ARG("-hl:verbose",verbose_logging);
          BOOL_ARG("-hl:logical_only",logical_logging_only);
          BOOL_ARG("-hl:physical_only",physical_logging_only);
          BOOL_ARG("-hl:bit_masks",bit_mask_logging);
#else
          if (!strcmp(argv[i],"-hl:tree"))
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_REGION_TREE_STATE_LOGGING,
                                  "Region tree state logging is "
                          "disabled.  To enable region tree state logging "
                                                  "compile in debug mode.");
          }
          if (!strcmp(argv[i],"-hl:disjointness"))
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_DISJOINTNESS_VERIFICATION,
                                  "Disjointness verification for "
                      "partition creation is disabled.  To enable dynamic "
                              "disjointness testing compile in debug mode.");
          }
#endif
          INT_ARG("-hl:prof", num_profiling_nodes);
          if (!strcmp(argv[i],"-hl:serializer"))
          {
            serializer_type = argv[++i];
            continue;
          }
          if (!strcmp(argv[i],"-hl:prof_logfile"))
          {
            prof_logfile = argv[++i];
            continue;
          }
        }
        if (delay_start > 0)
          sleep(delay_start);
#undef INT_ARG
#undef BOOL_ARG
#ifdef DEBUG_LEGION
        assert(initial_task_window_hysteresis <= 100);
        assert(max_local_fields <= MAX_FIELDS);
#endif
      }
      if (legion_spy_enabled)
        LegionSpy::log_legion_spy_config();
#ifdef DEBUG_LEGION
      if ((num_profiling_nodes > 0) && !slow_debug_ok)
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
      if ((num_profiling_nodes > 0) && !slow_debug_ok)
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
      if (legion_spy_enabled && (num_profiling_nodes > 0) && !slow_debug_ok)
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
        fprintf(stderr,"!!! RUN WITHOUT -lg:spy flag FOR PROFILING    !!!\n");
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
      if ((num_profiling_nodes > 0) && !slow_debug_ok)
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
      if ((num_profiling_nodes > 0) && !slow_debug_ok)
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
      Runtime::get_input_args().argv = argv;
      Runtime::get_input_args().argc = argc;
      // For the moment, we only need to register our runtime tasks
      // We'll register everything else once the Legion runtime starts
      RtEvent tasks_registered = register_runtime_tasks(realm);
      // Do some error checking in case we are running with separate instances
      Machine machine = Machine::get_machine();
      if (separate_runtime_instances)
      {
#ifdef TRACE_ALLOCATION
        REPORT_LEGION_ERROR(ERROR_TRACING_ALLOCATION_WITH_SEPARATE, 
                      "Memory tracing not supported with "
                      "separate runtime instances.")
#endif
        // Check for utility processors
        Machine::ProcessorQuery util_procs(machine);
        util_procs.local_address_space().only_kind(Processor::UTIL_PROC);
        if (util_procs.count() > 0)
          REPORT_LEGION_ERROR(ERROR_SEPARATE_UTILITY_PROCS, 
                        "Separate runtime instances are not "
                        "supported when running with explicit "
                        "utility processors")
#ifdef DEBUG_LEGION
        assert(runtime_map == NULL);
#endif
        // Create the runtime map for everyone to use
        runtime_map = new std::map<Processor,Runtime*>();
        // Instantiate all the entries, but assign them to NULL
        // so that the map doesn't change while parallel start-up
        // is occurring
        Machine::ProcessorQuery local_procs(machine);
        local_procs.local_address_space();
        for (Machine::ProcessorQuery::iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          (*runtime_map)[*it] = NULL;
        }
      }
      // Check for exceeding the local number of processors
      // and also see if we are supposed to launch the top-level task
      Processor top_level_proc = Processor::NO_PROC;
      Processor::Kind startup_kind = Processor::NO_KIND;
      {
        Machine::ProcessorQuery local_procs(machine);
        local_procs.local_address_space();
        if (local_procs.count() > MAX_NUM_PROCS)
          REPORT_LEGION_ERROR(ERROR_MAXIMUM_PROCS_EXCEEDED, 
                        "Maximum number of local processors %zd exceeds "
                        "compile time maximum of %d.  Change the value "
                        "in legion_config.h and recompile.",
                        local_procs.count(), MAX_NUM_PROCS)
        const AddressSpace local_space = local_procs.begin()->address_space();
        // We'll prefer CPUs for startup for now, but if we get
        // a utility processor then we'll use it
        for (Machine::ProcessorQuery::iterator it = 
              local_procs.begin(); it != local_procs.end(); it++)
        {
          if (it->kind() == Processor::LOC_PROC)
          {
            startup_kind = Processor::LOC_PROC;
            if (local_space == 0)
              top_level_proc = *it;
            break;
          }
          else if ((it->kind() == Processor::UTIL_PROC) &&
                   (startup_kind == Processor::NO_KIND))
          {
            startup_kind = Processor::UTIL_PROC;
            if (local_space == 0)
              top_level_proc = *it;
            // Keep going for now to see if we have a CPU processor
          }
        }
      }
      // Check to make sure we have something to do startup
      if (startup_kind == Processor::NO_KIND)
        REPORT_LEGION_ERROR(ERROR_NO_PROCESSORS, "Machine model contains "
            "no CPU processors and no utility processors! At least one "
            "CPU or one utility processor is required for Legion.")
#ifdef DEBUG_LEGION
      // Startup kind should be a CPU or a Utility processor
      assert((startup_kind == Processor::LOC_PROC) ||
              (startup_kind == Processor::UTIL_PROC));
#endif
      // Right now we launch a dummy barrier task on all the processors
      // to ensure that Realm has started them such that any interpreter
      // processors have loaded all their code
      // TODO: Remove this once Realm gives us a startup event
      RtEvent procs_started(realm.collective_spawn_by_kind(
            Processor::NO_KIND, LG_DUMMY_BARRIER_ID, NULL, 0,
            false/*one per node*/, tasks_registered));
      // Now perform a collective spawn to initialize the runtime everywhere
      // Save the precondition in case we are the node that needs to start
      // the top-level task.
      // If we're doing separate runtime instances we need to launch an
      // init task on every processor on all nodes, otherwise we just
      // need to launch one task on a CPU processor on every node
      RtEvent runtime_startup_event(realm.collective_spawn_by_kind(
          (separate_runtime_instances ? Processor::NO_KIND : startup_kind), 
           INIT_TASK_ID, NULL, 0, !separate_runtime_instances, procs_started));
      // See if we need to do any initialization for MPI interoperability
      if (mpi_rank >= 0)
      {
        // Do another collective to construct the rank tables
        RtEvent mpi_init_event(realm.collective_spawn_by_kind(
              startup_kind, LG_MPI_INTEROP_ID, NULL, 0,
              true/*one per node*/, runtime_startup_event));
        // The mpi init event then becomes the new runtime startup event
        runtime_startup_event = mpi_init_event;
        // If we have any pending MPI handshakes, we need to initialize them now
        if (pending_handshakes != NULL)
        {
          for (std::vector<MPILegionHandshake>::const_iterator it = 
                pending_handshakes->begin(); it != 
                pending_handshakes->end(); it++)
            it->impl->initialize();
          delete pending_handshakes;
          pending_handshakes = NULL;
	  // Another (dummy) collective task launch to ensure that
	  // all ranks have initialized their handshakes before
	  // the top-level task is started
	  RtEvent startup_sync_event(realm.collective_spawn_by_kind(
                startup_kind, LG_STARTUP_SYNC_ID, NULL, 0,
                true/*one per node*/, runtime_startup_event));
	  // The startup sync event then becomes the new runtime startup event
	  runtime_startup_event = startup_sync_event;
        }
        else if (!stealing_disabled)
        {
          // Even if we don't have any pending handshakes we have to 
          // do a startup sync if stealing is enabled
          RtEvent startup_sync_event(realm.collective_spawn_by_kind(
                startup_kind, LG_STARTUP_SYNC_ID, NULL, 0,
                true/*one per node*/, runtime_startup_event));
          // The startup sync event then becomes the new runtime startup event
	  runtime_startup_event = startup_sync_event;
        }
      } 
      else if (!stealing_disabled)
      {
        // If there is no MPI interop, we still need a startup sync
        // to have the mappers kick off any stealing they are going to do
        RtEvent startup_sync_event(realm.collective_spawn_by_kind(
              (separate_runtime_instances ? Processor::NO_KIND : 
               startup_kind), LG_STARTUP_SYNC_ID, NULL, 0,
              !separate_runtime_instances, runtime_startup_event));
        // The startup sync event then becomes the new runtime startup event
        runtime_startup_event = startup_sync_event;
      }
      // See if we are supposed to start the top-level task
      if (top_level_proc.exists())
      {
        Realm::ProfilingRequestSet empty_requests;
        top_level_proc.spawn(LG_LAUNCH_TOP_LEVEL_ID, NULL, 0,
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
    /*static*/ void Runtime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      if (!runtime_backgrounded)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_WAIT_FOR_SHUTDOWN, 
                      "Illegal call to wait_for_shutdown when runtime was "
                      "not launched in background mode!");
      RealmRuntime::get_runtime().wait_for_shutdown();
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
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'configure_MPI_interoperability' after "
                      "the runtime has been started!");
#ifdef DEBUG_LEGION
      assert(rank >= 0);
#endif
      // Check to see if it was already set
      if (mpi_rank >= 0)
      {
        if (rank != mpi_rank)
          REPORT_LEGION_ERROR(ERROR_DUPLICATE_MPI_CONFIG, 
              "multiple calls to "
              "configure_MPI_interoperability with different ranks "
              "%d and %d on the same Legion runtime!", mpi_rank, rank)
        else
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_MPI_CONFIG,
                                "duplicate calls to configure_"
                                "MPI_interoperability on rank %d!", rank);
      }
      mpi_rank = rank;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::register_handshake(MPILegionHandshake &handshake)
    //--------------------------------------------------------------------------
    {
      // See if the runtime is started or not
      if (runtime_started)
      {
        // If it's started, we can just do the initialization now
        handshake.impl->initialize();
      }
      else
      {
        if (pending_handshakes == NULL)
          pending_handshakes = new std::vector<MPILegionHandshake>();
        pending_handshakes->push_back(handshake);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID, 
                      "ReductionOpID zero is reserved.")
      ReductionOpTable &red_table = Runtime::get_reduction_table();
#ifdef DEBUG_LEGION
      if (red_table.find(redop_id) == red_table.end())
        REPORT_LEGION_ERROR(ERROR_INVALID_REDOP_ID, 
                      "Invalid ReductionOpID %d",redop_id)
#endif
      return red_table[redop_id];
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezOp* Runtime::get_serdez_op(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      if (serdez_id == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_SERDEZ_ID, 
                      "CustomSerdezID zero is reserved.")
      SerdezOpTable &serdez_table = Runtime::get_serdez_table();
#ifdef DEBUG_LEGION
      if (serdez_table.find(serdez_id) == serdez_table.end())
        REPORT_LEGION_ERROR(ERROR_INVALID_SERDEZ_ID, 
                      "Invalid CustomSerdezOpID %d", serdez_id)
#endif
      return serdez_table[serdez_id];
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezRedopFns* Runtime::get_serdez_redop_fns(
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
    /*static*/ void Runtime::add_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      registration_callbacks.push_back(callback);
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
      if (separate_runtime_instances)
      {
        if (!p.exists())
          p = Processor::get_executing_processor();
#ifdef DEBUG_LEGION
        assert(runtime_map != NULL);
        assert(the_runtime == NULL);
        assert(runtime_map->find(p) != runtime_map->end());
#endif
        return (*runtime_map)[p];
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(runtime_map == NULL);
        assert(the_runtime != NULL);
#endif
        return the_runtime;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpTable& Runtime::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      static ReductionOpTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezOpTable& Runtime::get_serdez_table(void)
    //--------------------------------------------------------------------------
    {
      static SerdezOpTable table;
      return table;
    }
    
    //--------------------------------------------------------------------------
    /*static*/ SerdezRedopTable& Runtime::get_serdez_redop_table(void)
    //--------------------------------------------------------------------------
    {
      static SerdezRedopTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::deque<PendingVariantRegistration*>& 
                                       Runtime::get_pending_variant_table(void)
    //--------------------------------------------------------------------------
    {
      static std::deque<PendingVariantRegistration*> pending_variant_table;
      return pending_variant_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<LayoutConstraintID,LayoutConstraintRegistrar>&
                                    Runtime::get_pending_constraint_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<LayoutConstraintID,LayoutConstraintRegistrar>
                                                    pending_constraint_table;
      return pending_constraint_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<ProjectionID,ProjectionFunctor*>&
                                     Runtime::get_pending_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<ProjectionID,ProjectionFunctor*> pending_projection_table;
      return pending_projection_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID& Runtime::get_current_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      static TaskID current_task_id = MAX_APPLICATION_TASK_ID;
      return current_task_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Runtime::generate_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      TaskID &next_task = get_current_static_task_id(); 
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'generate_static_task_id' after "
                      "the runtime has been started!")
      return next_task++;
    }

    //--------------------------------------------------------------------------
    /*static*/ VariantID Runtime::preregister_variant(
                          const TaskVariantRegistrar &registrar,
                          const void *user_data, size_t user_data_size,
                          CodeDescriptor *code_desc, bool has_ret, 
                          const char *task_name, VariantID vid, bool check_id)
    //--------------------------------------------------------------------------
    {
      // Report an error if the runtime has already started
      if (runtime_started)
        REPORT_LEGION_ERROR(ERROR_STATIC_CALL_POST_RUNTIME_START, 
                      "Illegal call to 'preregister_task_variant' after "
                      "the runtime has been started!")
      if (check_id && (registrar.task_id >= get_current_static_task_id()))
        REPORT_LEGION_ERROR(ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED, 
                      "Error preregistering task with ID %d. Exceeds the "
                      "statically set bounds on application task IDs of %d. "
                      "See %s in legion_config.h.", 
                      registrar.task_id, MAX_APPLICATION_TASK_ID, 
                      LEGION_MACRO_TO_STRING(MAX_APPLICATION_TASK_ID))
      std::deque<PendingVariantRegistration*> &pending_table = 
        get_pending_variant_table();
      // See if we need to pick a variant
      if (vid == AUTO_GENERATE_ID)
        vid = pending_table.size() + 1;
      else if (vid == 0)
        REPORT_LEGION_ERROR(ERROR_RESERVED_VARIANT_ID,
                      "Error preregistering variant for task ID %d with "
                      "variant ID 0. Variant ID 0 is reserved for task "
                      "generators.", registrar.task_id)
      // Offset by the runtime tasks
      pending_table.push_back(new PendingVariantRegistration(vid, has_ret,
                              registrar, user_data, user_data_size, 
                              code_desc, task_name));
      return vid;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::report_fatal_message(int id,
                                          const char *file_name, const int line,
                                                  const char *message)
    //--------------------------------------------------------------------------
    {
      log_run.fatal(id, "LEGION FATAL: %s (from file %s:%d)",
                    message, file_name, line);
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(id);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ void Runtime::report_error_message(int id,
                                          const char *file_name, const int line,
                                                  const char *message)
    //--------------------------------------------------------------------------
    {
      log_run.error(id, "LEGION ERROR: %s (from file %s:%d)",
                    message, file_name, line);
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(id);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ void Runtime::report_warning_message(
                                         int id,
                                         const char *file_name, const int line,
                                         const char *message)
    //--------------------------------------------------------------------------
    {
      log_run.warning(id, "LEGION WARNING: %s (from file %s:%d)",
                      message, file_name, line);
    }

#if defined(PRIVILEGE_CHECKS) || defined(BOUNDS_CHECKS)
    //--------------------------------------------------------------------------
    /*static*/ const char* Runtime::find_privilege_task_name(void *impl)
    //--------------------------------------------------------------------------
    {
      PhysicalRegionImpl *region = static_cast<PhysicalRegionImpl*>(impl);
      return region->get_task_name();
    }
#endif

#ifdef BOUNDS_CHECKS
    //--------------------------------------------------------------------------
    /*static*/ void Runtime::check_bounds(void *impl, ptr_t ptr)
    //--------------------------------------------------------------------------
    {
      PhysicalRegionImpl *region = static_cast<PhysicalRegionImpl*>(impl);
      if (!region->contains_ptr(ptr))
      {
        fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                       "pointer %lld\n", region->get_task_name(), ptr.value);
        assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::check_bounds(void *impl, 
                                          const DomainPoint &dp)
    //--------------------------------------------------------------------------
    {
      PhysicalRegionImpl *region = static_cast<PhysicalRegionImpl*>(impl);
      if (!region->contains_point(dp))
      {
        switch(dp.get_dim())
        {
          case 1:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                           "1D point (%lld)\n", region->get_task_name(),
                            dp.point_data[0]);
            break;
          case 2:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                           "2D point (%lld,%lld)\n", region->get_task_name(),
                            dp.point_data[0], dp.point_data[1]);
            break;
          case 3:
            fprintf(stderr,"BOUNDS CHECK ERROR IN TASK %s: Accessing invalid "
                         "3D point (%lld,%lld,%lld)\n", region->get_task_name(),
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
    /*static*/ RtEvent Runtime::register_runtime_tasks(RealmRuntime &realm)
    //--------------------------------------------------------------------------
    {
      // Make the code descriptors for our tasks
      CodeDescriptor init_task(Runtime::initialize_runtime);
      CodeDescriptor shutdown_task(Runtime::shutdown_runtime);
      CodeDescriptor lg_task(Runtime::legion_runtime_task);
      CodeDescriptor rt_profiling_task(Runtime::profiling_runtime_task);
      CodeDescriptor launch_top_level_task(Runtime::launch_top_level);
      CodeDescriptor mpi_interop_task(Runtime::init_mpi_interop);
      CodeDescriptor startup_sync_task(Runtime::startup_sync);
      Realm::ProfilingRequestSet no_requests;
      // We'll just register these on all the processor kinds
      std::set<RtEvent> registered_events;
      Processor::Kind kinds[6] = { Processor::TOC_PROC, Processor::LOC_PROC,
                                   Processor::UTIL_PROC, Processor::IO_PROC,
                                   Processor::PROC_SET, Processor::OMP_PROC };
      for (unsigned idx = 0; idx < 6; idx++)
      {
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                                       INIT_TASK_ID, init_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                               SHUTDOWN_TASK_ID, shutdown_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                                         LG_TASK_ID, lg_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                    LG_LEGION_PROFILING_ID, rt_profiling_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                LG_LAUNCH_TOP_LEVEL_ID, launch_top_level_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                          LG_MPI_INTEROP_ID, mpi_interop_task, no_requests)));
        registered_events.insert(RtEvent(
            Processor::register_task_by_kind(kinds[idx], false/*global*/,
                          LG_STARTUP_SYNC_ID, startup_sync_task, no_requests)));
      }
      if (record_registration)
      {
        log_run.print("Legion dummy barrier task has Realm ID %d",
                      LG_DUMMY_BARRIER_ID);
        log_run.print("Legion runtime initialization task "
                            "has Realm ID %d", INIT_TASK_ID);
        log_run.print("Legion runtime shutdown task has "
                            "Realm ID %d", SHUTDOWN_TASK_ID);
        log_run.print("Legion runtime meta-task has Realm ID %d", 
                      LG_TASK_ID);
        log_run.print("Legion runtime profiling task Realm ID %d",
                      LG_LEGION_PROFILING_ID);
        log_run.print("Legion launch top-level task has Realm ID %d",
                      LG_LAUNCH_TOP_LEVEL_ID);
      }
      return Runtime::merge_events(registered_events);
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
      if (!legion_spy_enabled)
        return;
      std::set<Processor::Kind> proc_kinds;
      Machine::ProcessorQuery all_procs(machine);
      // Log processors
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        Processor::Kind kind = it->kind();
        if (proc_kinds.find(kind) == proc_kinds.end())
        {
          switch (kind)
          {
            case Processor::NO_KIND:
              {
                LegionSpy::log_processor_kind(kind, "NoProc");
                break;
              }
            case Processor::TOC_PROC:
              {
                LegionSpy::log_processor_kind(kind, "GPU");
                break;
              }
            case Processor::LOC_PROC:
              {
                LegionSpy::log_processor_kind(kind, "CPU");
                break;
              }
            case Processor::UTIL_PROC:
              {
                LegionSpy::log_processor_kind(kind, "Utility");
                break;
              }
            case Processor::IO_PROC:
              {
                LegionSpy::log_processor_kind(kind, "IO");
                break;
              }
            case Processor::PROC_GROUP:
              {
                LegionSpy::log_processor_kind(kind, "ProcGroup");
                break;
              }
            case Processor::PROC_SET:
              {
                LegionSpy::log_processor_kind(kind, "ProcSet");
                break;
              }
            case Processor::OMP_PROC:
              {
                LegionSpy::log_processor_kind(kind, "OpenMP");
                break;
              }
            default:
              assert(false); // unknown processor kind
          }
          proc_kinds.insert(kind);
        }
        LegionSpy::log_processor(it->id, kind);
      }
      // Log memories
      std::set<Memory::Kind> mem_kinds;
      Machine::MemoryQuery all_mems(machine);
      for (Machine::MemoryQuery::iterator it = all_mems.begin();
            it != all_mems.end(); it++)
      {
        Memory::Kind kind = it->kind();
        if (mem_kinds.find(kind) == mem_kinds.end())
        {
          switch (kind)
          {
            case GLOBAL_MEM:
              {
                LegionSpy::log_memory_kind(kind, "GASNet");
                break;
              }
            case SYSTEM_MEM:
              {
                LegionSpy::log_memory_kind(kind, "System");
                break;
              }
            case REGDMA_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Registered");
                break;
              }
            case SOCKET_MEM:
              {
                LegionSpy::log_memory_kind(kind, "NUMA");
                break;
              }
            case Z_COPY_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Zero-Copy");
                break;
              }
            case GPU_FB_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Framebuffer");
                break;
              }
            case DISK_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Disk");
                break;
              }
            case HDF_MEM:
              {
                LegionSpy::log_memory_kind(kind, "HDF");
                break;
              }
            case FILE_MEM:
              {
                LegionSpy::log_memory_kind(kind, "File");
                break;
              }
            case LEVEL3_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L3");
                break;
              }
            case LEVEL2_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L2");
                break;
              }
            case LEVEL1_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L1");
                break;
              }
            default:
              assert(false); // unknown memory kind
          }
        }
        LegionSpy::log_memory(it->id, it->capacity(), it->kind());
      }
      // Log Proc-Mem Affinity
      Machine::ProcessorQuery all_procs2(machine);
      for (Machine::ProcessorQuery::iterator pit = all_procs2.begin();
            pit != all_procs2.end(); pit++)
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
      Machine::MemoryQuery all_mems2(machine);
      for (Machine::MemoryQuery::iterator mit = all_mems2.begin();
            mit != all_mems2.begin(); mit++)
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
    /*static*/ void Runtime::initialize_runtime(
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
        REPORT_LEGION_ERROR(ERROR_NO_PROCESSORS, 
                      "Machine model contains no processors!");
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
      Runtime *local_rt = new Runtime(machine, local_space_id, 
                                      local_procs, local_util_procs,
                                      address_spaces, proc_spaces);
      if (separate_runtime_instances)
      {
#ifdef DEBUG_LEGION
        assert(local_util_procs.empty());
        assert(runtime_map != NULL);
#endif
        // Now set up the runtime on all of the local processors
        // and their utility processors
        for (std::set<Processor>::const_iterator it = local_procs.begin();
              it != local_procs.end(); it++)
        {
          std::map<Processor,Runtime*>::iterator finder = 
            runtime_map->find(*it);
#ifdef DEBUG_LEGION
          assert(finder != runtime_map->end());
          assert(finder->second == NULL);
#endif
          finder->second = local_rt;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(the_runtime == NULL);
#endif
        the_runtime = local_rt; 
      }
      // Do the rest of our initialization
      if (local_space_id < Runtime::num_profiling_nodes)
        local_rt->initialize_legion_prof();
      local_rt->register_static_variants();
      local_rt->register_static_constraints();
      local_rt->register_static_projections();
      // Initialize our one virtual manager, do this after we register
      // the static constraints so we get a valid layout constraint ID
      VirtualManager::initialize_virtual_instance(local_rt, 
                                                  0/*same across nodes*/);
      // Configure our collective settings
      if (address_spaces.size() > 1)
        configure_collective_settings(address_spaces.size());
      // If we have an MPI rank, then build the maps
      // We'll initialize the mappers after the tables are built
      if (Runtime::mpi_rank >= 0)
      {
#ifdef DEBUG_LEGION
        assert(!separate_runtime_instances);
        assert(mpi_rank_table == NULL);
#endif
        mpi_rank_table = new MPIRankTable(local_rt); 
      }
      else // We can initialize the mappers now
        local_rt->initialize_mappers(); 
      // Once we're done with this task, the Legion runtime is started
      runtime_started = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::shutdown_runtime(const void *args, size_t arglen, 
                              const void *userdata, size_t userlen, Processor p)
    //--------------------------------------------------------------------------
    {
      // We can also finalize our virtual instance unless we're running with 
      // separate runtime instances, in which case we'll just leak it
      if (!Runtime::separate_runtime_instances)
        VirtualManager::finalize_virtual_instance();
      // Finalize the runtime and then delete it
      Runtime *runtime = get_runtime(p);
      runtime->finalize_runtime();
      delete runtime;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::legion_runtime_task(
                                  const void *args, size_t arglen, 
				  const void *userdata, size_t userlen,
				  Processor p)
    //--------------------------------------------------------------------------
    {
      // We immediately bump the priority of all meta-tasks once they start
      // up to the highest level to ensure that they drain once they begin
      Processor::set_current_task_priority(LG_RUNNING_PRIORITY);
      const char *data = (const char*)args;
      LgTaskID tid = *((const LgTaskID*)data);
      data += sizeof(tid);
      arglen -= sizeof(tid);
      switch (tid)
      {
        case LG_SCHEDULER_ID:
          {
            const ProcessorManager::SchedulerArgs *sched_args = 
              (const ProcessorManager::SchedulerArgs*)args;
            Runtime::get_runtime(p)->process_schedule_request(
                                                            sched_args->proc);
            break;
          }
        case LG_MESSAGE_ID:
          {
            Runtime::get_runtime(p)->process_message_task(data, arglen);
            break;
          }
        case LG_POST_END_ID:
          {
            const InnerContext::PostEndArgs *post_end_args = 
              (const InnerContext::PostEndArgs*)args;
            post_end_args->proxy_this->post_end_task(post_end_args->result, 
                                post_end_args->result_size, true/*owned*/);
            break;
          }
        case LG_DEFERRED_READY_TRIGGER_ID:
          {
            const Operation::DeferredReadyArgs *deferred_ready_args = 
              (const Operation::DeferredReadyArgs*)args;
            deferred_ready_args->proxy_this->trigger_ready();
            break;
          }
        case LG_DEFERRED_RESOLUTION_TRIGGER_ID:
          {
            const Operation::DeferredResolutionArgs *deferred_resolution_args =
              (const Operation::DeferredResolutionArgs*)args;
            deferred_resolution_args->proxy_this->trigger_resolution();
            break;
          }
        case LG_DEFERRED_COMMIT_TRIGGER_ID:
          {
            const Operation::DeferredCommitTriggerArgs *deferred_commit_args =
              (const Operation::DeferredCommitTriggerArgs*)args;
            deferred_commit_args->proxy_this->deferred_commit_trigger(
                deferred_commit_args->gen);
            break;
          }
        case LG_DEFERRED_POST_MAPPED_ID:
          {
            const SingleTask::DeferredPostMappedArgs *post_mapped_args = 
              (const SingleTask::DeferredPostMappedArgs*)args;
            post_mapped_args->task->handle_post_mapped();
            break;
          }
        case LG_DEFERRED_EXECUTE_ID:
          {
            const Operation::DeferredExecArgs *deferred_exec_args = 
              (const Operation::DeferredExecArgs*)args;
            deferred_exec_args->proxy_this->complete_execution();
            break;
          }
        case LG_DEFERRED_EXECUTION_TRIGGER_ID:
          {
            const Operation::DeferredExecuteArgs *deferred_mapping_args = 
              (const Operation::DeferredExecuteArgs*)args;
            deferred_mapping_args->proxy_this->deferred_execute();
            break;
          }
        case LG_DEFERRED_COMPLETE_ID:
          {
            const Operation::DeferredCompleteArgs *deferred_complete_args =
              (const Operation::DeferredCompleteArgs*)args;
            deferred_complete_args->proxy_this->complete_operation();
            break;
          } 
        case LG_DEFERRED_COMMIT_ID:
          {
            const Operation::DeferredCommitArgs *deferred_commit_args = 
              (const Operation::DeferredCommitArgs*)args;
            deferred_commit_args->proxy_this->commit_operation(
                deferred_commit_args->deactivate);
            break;
          }
        case LG_DEFERRED_COLLECT_ID:
          {
            const GarbageCollectionEpoch::GarbageCollectionArgs *collect_args =
              (const GarbageCollectionEpoch::GarbageCollectionArgs*)args;
            bool done = collect_args->epoch->handle_collection(collect_args);
            if (done)
              delete collect_args->epoch;
            break;
          }
        case LG_PRE_PIPELINE_ID:
          {
            const Operation::PrepipelineArgs *pargs = 
              (const Operation::PrepipelineArgs*)args;
            pargs->proxy_this->trigger_prepipeline_stage();
            break;
          }
        case LG_TRIGGER_DEPENDENCE_ID:
          {
            const InnerContext::DeferredDependenceArgs *deferred_trigger_args =
              (const InnerContext::DeferredDependenceArgs*)args;
            deferred_trigger_args->op->execute_dependence_analysis();
            break;
          }
        case LG_TRIGGER_COMPLETE_ID:
          {
            const Operation::TriggerCompleteArgs *trigger_complete_args =
              (const Operation::TriggerCompleteArgs*)args;
            trigger_complete_args->proxy_this->trigger_complete();
            break;
          }
        case LG_TRIGGER_OP_ID:
          {
            // Key off of args here instead of data
            const ProcessorManager::TriggerOpArgs *trigger_args = 
                            (const ProcessorManager::TriggerOpArgs*)args;
            trigger_args->op->trigger_mapping();
            break;
          }
        case LG_TRIGGER_TASK_ID:
          {
            // Key off of args here instead of data
            const ProcessorManager::TriggerTaskArgs *trigger_args = 
                          (const ProcessorManager::TriggerTaskArgs*)args;
            trigger_args->op->trigger_mapping(); 
            break;
          }
        case LG_DEFER_MAPPER_SCHEDULER_TASK_ID:
          {
            ProcessorManager::handle_defer_mapper(args);
            break;
          }
        case LG_DEFERRED_RECYCLE_ID:
          {
            const DeferredRecycleArgs *deferred_recycle_args = 
              (const DeferredRecycleArgs*)args;
            Runtime::get_runtime(p)->free_distributed_id(
                                        deferred_recycle_args->did);
            break;
          }
        case LG_MUST_INDIV_ID:
          {
            MustEpochTriggerer::handle_individual(args);
            break;
          }
        case LG_MUST_INDEX_ID:
          {
            MustEpochTriggerer::handle_index(args);
            break;
          }
        case LG_MUST_MAP_ID:
          {
            MustEpochMapper::handle_map_task(args);
            break;
          }
        case LG_MUST_DIST_ID:
          {
            MustEpochDistributor::handle_distribute_task(args);
            break;
          }
        case LG_MUST_LAUNCH_ID:
          {
            MustEpochDistributor::handle_launch_task(args);
            break;
          }
        case LG_DEFERRED_FUTURE_SET_ID:
          {
            DeferredFutureSetArgs *future_args =  
              (DeferredFutureSetArgs*)args;
            const size_t result_size = 
              future_args->task_op->check_future_size(future_args->result);
            if (result_size > 0)
              future_args->target->set_result(
                  future_args->result->get_untyped_result(),
                  result_size, false/*own*/);
            if (future_args->target->remove_base_gc_ref(DEFERRED_TASK_REF))
              delete (future_args->target);
            if (future_args->result->remove_base_gc_ref(DEFERRED_TASK_REF)) 
              delete (future_args->result);
            break;
          }
        case LG_DEFERRED_FUTURE_MAP_SET_ID:
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
            if (future_args->future_map->remove_base_gc_ref(
                                                        DEFERRED_TASK_REF))
              delete (future_args->future_map);
            if (future_args->result->remove_base_gc_ref(FUTURE_HANDLE_REF))
              delete (future_args->result);
            future_args->task_op->complete_execution();
            break;
          }
        case LG_RESOLVE_FUTURE_PRED_ID:
          {
            FuturePredOp::ResolveFuturePredArgs *resolve_args = 
              (FuturePredOp::ResolveFuturePredArgs*)args;
            resolve_args->future_pred_op->resolve_future_predicate();
            resolve_args->future_pred_op->remove_predicate_reference();
            break;
          }
        case LG_CONTRIBUTE_COLLECTIVE_ID:
          {
            FutureImpl::handle_contribute_to_collective(args);
            break;
          }
        case LG_TOP_FINISH_TASK_ID:
          {
            TopFinishArgs *fargs = (TopFinishArgs*)args; 
            // Do this before deleting remote contexts
            fargs->ctx->invalidate_region_tree_contexts();
            fargs->ctx->free_remote_contexts();
            if (fargs->ctx->remove_reference())
              delete fargs->ctx;
            Runtime *runtime = Runtime::get_runtime(p);
            // Finally tell the runtime that we have one less top level task
            runtime->decrement_outstanding_top_level_tasks();
            break;
          }
        case LG_MAPPER_TASK_ID:
          {
            MapperTaskArgs *margs = (MapperTaskArgs*)args;
            Runtime *runtime = Runtime::get_runtime(p);
            runtime->process_mapper_task_result(margs);
            // Now indicate that we are done with the future
            if (margs->future->remove_base_gc_ref(FUTURE_HANDLE_REF))
              delete margs->future;
            margs->ctx->invalidate_region_tree_contexts();
            // We can also deactivate the enclosing context 
            if (margs->ctx->remove_reference())
              delete margs->ctx;
            // Finally tell the runtime we have one less top level task
            runtime->decrement_outstanding_top_level_tasks();
            break;
          }
        case LG_DISJOINTNESS_TASK_ID:
          {
            RegionTreeForest::DisjointnessArgs *dargs = 
              (RegionTreeForest::DisjointnessArgs*)args;
            Runtime *runtime = Runtime::get_runtime(p);
            runtime->forest->compute_partition_disjointness(dargs->handle,
                                                            dargs->ready);
            break;
          }
        case LG_PART_INDEPENDENCE_TASK_ID:
          {
            IndexSpaceNode::DynamicIndependenceArgs *dargs = 
              (IndexSpaceNode::DynamicIndependenceArgs*)args;
            IndexSpaceNode::handle_disjointness_test(
                dargs->parent, dargs->left, dargs->right);
            break;
          }
        case LG_SPACE_INDEPENDENCE_TASK_ID:
          {
            IndexPartNode::DynamicIndependenceArgs *dargs = 
              (IndexPartNode::DynamicIndependenceArgs*)args;
            IndexPartNode::handle_disjointness_test(
                dargs->parent, dargs->left, dargs->right);
            break;
          }
        case LG_PENDING_CHILD_TASK_ID:
          {
            IndexPartNode::handle_pending_child_task(args);
            break;
          }
        case LG_POST_DECREMENT_TASK_ID:
          {
            InnerContext::PostDecrementArgs *dargs = 
              (InnerContext::PostDecrementArgs*)args;
            Runtime::get_runtime(p)->activate_context(dargs->parent_ctx);
            break;
          }
        case LG_SEND_VERSION_STATE_UPDATE_TASK_ID:
          {
            VersionState::SendVersionStateArgs *vargs = 
              (VersionState::SendVersionStateArgs*)args;
            vargs->proxy_this->send_version_state_update(vargs->target, 
                                                  vargs->context,
                                                  *(vargs->request_mask),
                                                  vargs->request_kind,
                                                  vargs->to_trigger);
            delete (vargs->request_mask);
            break;
          }
        case LG_ISSUE_FRAME_TASK_ID:
          {
            InnerContext::IssueFrameArgs *fargs = 
              (InnerContext::IssueFrameArgs*)args;
            fargs->parent_ctx->perform_frame_issue(fargs->frame, 
                                                   fargs->frame_termination);
            break;
          }
        case LG_MAPPER_CONTINUATION_TASK_ID:
          {
            MapperContinuation::handle_continuation(args);
            break;
          }
        case LG_TASK_IMPL_SEMANTIC_INFO_REQ_TASK_ID:
          {
            TaskImpl::SemanticRequestArgs *req_args = 
              (TaskImpl::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_INDEX_SPACE_SEMANTIC_INFO_REQ_TASK_ID:
          {
            IndexSpaceNode::SemanticRequestArgs *req_args = 
              (IndexSpaceNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_INDEX_PART_SEMANTIC_INFO_REQ_TASK_ID:
          {
            IndexPartNode::SemanticRequestArgs *req_args = 
              (IndexPartNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_FIELD_SPACE_SEMANTIC_INFO_REQ_TASK_ID:
          {
            FieldSpaceNode::SemanticRequestArgs *req_args = 
              (FieldSpaceNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_FIELD_SEMANTIC_INFO_REQ_TASK_ID:
          {
            FieldSpaceNode::SemanticFieldRequestArgs *req_args = 
              (FieldSpaceNode::SemanticFieldRequestArgs*)args;
            req_args->proxy_this->process_semantic_field_request(
                  req_args->fid, req_args->tag, req_args->source, 
                  false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_REGION_SEMANTIC_INFO_REQ_TASK_ID:
          {
            RegionNode::SemanticRequestArgs *req_args = 
              (RegionNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_PARTITION_SEMANTIC_INFO_REQ_TASK_ID:
          {
            PartitionNode::SemanticRequestArgs *req_args = 
              (PartitionNode::SemanticRequestArgs*)args;
            req_args->proxy_this->process_semantic_request(
                          req_args->tag, req_args->source, 
                          false, false, RtUserEvent::NO_RT_USER_EVENT);
            break;
          }
        case LG_INDEX_SPACE_DEFER_CHILD_TASK_ID:
          {
            IndexSpaceNode::defer_node_child_request(args);
            break;
          }
        case LG_INDEX_PART_DEFER_CHILD_TASK_ID:
          {
            IndexPartNode::defer_node_child_request(args);
            break;
          }
        case LG_SELECT_TUNABLE_TASK_ID:
          {
            const SelectTunableArgs *tunable_args = 
              (const SelectTunableArgs*)args;
            Runtime::get_runtime(p)->perform_tunable_selection(tunable_args);
            // Remove the reference that we added
            if (tunable_args->result->remove_base_gc_ref(FUTURE_HANDLE_REF)) 
              delete (tunable_args->result);
            break;
          }
        case LG_DEFERRED_ENQUEUE_OP_ID:
          {
            const Operation::DeferredEnqueueArgs *deferred_enqueue_args = 
              (const Operation::DeferredEnqueueArgs*)args;
            deferred_enqueue_args->proxy_this->enqueue_ready_operation(
                RtEvent::NO_RT_EVENT, deferred_enqueue_args->priority);
            break;
          }
        case LG_DEFERRED_ENQUEUE_TASK_ID:
          {
            const DeferredEnqueueArgs *enqueue_args = 
              (const DeferredEnqueueArgs*)args;
            enqueue_args->manager->add_to_ready_queue(enqueue_args->task);
            break;
          }
        case LG_DEFER_MAPPER_MESSAGE_TASK_ID:
          {
            MapperManager::handle_deferred_message(args);
            break;
          }
        case LG_DEFER_COMPOSITE_VIEW_REF_TASK_ID:
          {
            CompositeView::handle_deferred_view_ref(args);
            break;
          }
        case LG_DEFER_COMPOSITE_VIEW_REGISTRATION_TASK_ID:
          {
            CompositeView::handle_deferred_view_registration(args);
            break;
          }
        case LG_DEFER_COMPOSITE_NODE_REF_TASK_ID:
          {
            CompositeNode::handle_deferred_node_ref(args);
            break;
          }
        case LG_DEFER_COMPOSITE_NODE_CAPTURE_TASK_ID:
          {
            CompositeNode::handle_deferred_capture(args);
            break;
          }
        case LG_CONVERT_VIEW_TASK_ID:
          {
            VersionState::process_convert_view(args);
            break;
          }
        case LG_UPDATE_VIEW_REFERENCES_TASK_ID:
          {
            VersionState::process_view_references(args);
            break;
          }
        case LG_UPDATE_VERSION_STATE_REDUCE_TASK_ID:
          {
            VersionState::process_version_state_reduction(args);
            break;
          }
        case LG_REMOVE_VERSION_STATE_REF_TASK_ID:
          {
            VersionState::process_remove_version_state_ref(args);
            break;
          }
        case LG_DEFER_RESTRICTED_MANAGER_TASK_ID:
          {
            RestrictInfo::handle_deferred_reference(args);
            break;
          }
        case LG_REMOTE_VIEW_CREATION_TASK_ID:
          {
            InnerContext::handle_remote_view_creation(args);
            break;
          }
        case LG_DEFER_DISTRIBUTE_TASK_ID:
          {
            const TaskOp::DeferDistributeArgs *dargs = 
              (const TaskOp::DeferDistributeArgs*)args;
            dargs->proxy_this->distribute_task();
            break;
          }
        case LG_DEFER_PERFORM_MAPPING_TASK_ID:
          {
            const TaskOp::DeferMappingArgs *margs = 
              (const TaskOp::DeferMappingArgs*)args;
            RtEvent wait_on = margs->proxy_this->perform_mapping(
                                                    margs->must_op);
            if (wait_on.exists())
              wait_on.lg_wait();
            break;
          }
        case LG_DEFER_LAUNCH_TASK_ID:
          {
            const TaskOp::DeferLaunchArgs *largs = 
              (const TaskOp::DeferLaunchArgs*)args;
            largs->proxy_this->launch_task();
            break;
          }
        case LG_DEFER_MAP_AND_LAUNCH_TASK_ID:
          {
            const SliceTask::DeferMapAndLaunchArgs *margs = 
              (const SliceTask::DeferMapAndLaunchArgs*)args;
            margs->proxy_this->map_and_launch();
            break;
          }
        case LG_ADD_VERSIONING_SET_REF_TASK_ID:
          {
            const VersioningSetRefArgs *ref_args = 
              (const VersioningSetRefArgs*)args;
            LocalReferenceMutator mutator;
            ref_args->state->add_base_valid_ref(ref_args->kind, &mutator);
            break;
          }
        case LG_VERSION_STATE_CAPTURE_DIRTY_TASK_ID:
          {
            VersionManager::process_capture_dirty(args);
            break;
          }
        case LG_VERSION_STATE_PENDING_ADVANCE_TASK_ID:
          {
            VersionManager::process_pending_advance(args);
            break;
          }
        case LG_DISJOINT_CLOSE_TASK_ID:
          {
            InterCloseOp::handle_disjoint_close(args);
            break;
          }
        case LG_DEFER_MATERIALIZED_VIEW_TASK_ID:
          {
            MaterializedView::handle_deferred_materialized_view(
                                  Runtime::get_runtime(p), args);
            break;
          }
        case LG_MISSPECULATE_TASK_ID:
          {
            const SingleTask::MisspeculationTaskArgs *targs = 
              (const SingleTask::MisspeculationTaskArgs*)args;
            targs->task->handle_misspeculation();
            break;
          }
        case LG_DEFER_PHI_VIEW_REF_TASK_ID:
          {
            PhiView::handle_deferred_view_ref(args);
            break;
          }
        case LG_DEFER_PHI_VIEW_REGISTRATION_TASK_ID:
          {
            PhiView::handle_deferred_view_registration(args);
            break;
          }
        case LG_TIGHTEN_INDEX_SPACE_TASK_ID:
          {
            IndexSpaceNode::handle_tighten_index_space(args);
            break;
          }
        case LG_REMOTE_PHYSICAL_REQUEST_TASK_ID:
          {
            RemoteContext::defer_physical_request(args);
            break;
          }
        case LG_REMOTE_PHYSICAL_RESPONSE_TASK_ID:
          {
            RemoteContext::defer_physical_response(args);
            break;
          }
        case LG_RETRY_SHUTDOWN_TASK_ID:
          {
            const ShutdownManager::RetryShutdownArgs *shutdown_args = 
              (const ShutdownManager::RetryShutdownArgs*)args;
            Runtime *runtime = Runtime::get_runtime(p);
            runtime->initiate_runtime_shutdown(runtime->address_space,
                                               shutdown_args->phase);
            break;
          }
        default:
          assert(false); // should never get here
      }
#ifdef DEBUG_LEGION
      if (tid < LG_MESSAGE_ID)
        Runtime::get_runtime(p)->decrement_total_outstanding_tasks(tid, 
                                                                  true/*meta*/);
#else
      if (tid < LG_MESSAGE_ID)
        Runtime::get_runtime(p)->decrement_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      Runtime *runtime = Runtime::get_runtime(p);
      __sync_fetch_and_add(&runtime->outstanding_counts[tid],-1);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::profiling_runtime_task(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      Realm::ProfilingResponse response(args, arglen);
      const ProfilingResponseBase *base = 
        (const ProfilingResponseBase*)response.user_data();
      if (base->handler == NULL)
      {
        // If we got a NULL let's assume they meant the profiler
        // this mainly happens with messages that cross nodes
        Runtime *rt = Runtime::get_runtime(p);
        rt->profiler->handle_profiling_response(response);
      }
      else
        base->handler->handle_profiling_response(response);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::launch_top_level(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      Runtime *rt = Runtime::get_runtime(p);
      rt->launch_top_level_task();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::init_mpi_interop(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mpi_rank_table != NULL);
#endif
      mpi_rank_table->perform_rank_exchange();
      // Now configure the mappers
      Runtime *rt = Runtime::get_runtime(p);
      rt->initialize_mappers();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::startup_sync(
                                   const void *args, size_t arglen, 
				   const void *userdata, size_t userlen,
				   Processor p)
    //--------------------------------------------------------------------------
    {
      log_run.debug() << "startup sync task";
      if (stealing_disabled)
        return;
      Runtime *rt = Runtime::get_runtime(p);
      rt->startup_mappers();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_collective_settings(int total_spaces)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(legion_collective_radix > 0);
#endif
      const int MultiplyDeBruijnBitPosition[32] = 
      {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
          8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
      };
      // First adjust the radix based on the number of nodes if necessary
      if (legion_collective_radix > total_spaces)
        legion_collective_radix = total_spaces;
      // Adjust the radix to the next smallest power of 2
      uint32_t radix_copy = legion_collective_radix;
      for (int i = 0; i < 5; i++)
        radix_copy |= radix_copy >> (1 << i);
      legion_collective_log_radix = 
        MultiplyDeBruijnBitPosition[(uint32_t)(radix_copy * 0x07C4ACDDU) >> 27];
      if (legion_collective_radix != (1 << legion_collective_log_radix))
        legion_collective_radix = (1 << legion_collective_log_radix);

      // Compute the number of stages
      uint32_t node_copy = total_spaces;
      for (int i = 0; i < 5; i++)
        node_copy |= node_copy >> (1 << i);
      // Now we have it log 2
      int log_nodes = 
        MultiplyDeBruijnBitPosition[(uint32_t)(node_copy * 0x07C4ACDDU) >> 27];
      // Stages round up in case of incomplete stages
      legion_collective_stages = (log_nodes + 
          legion_collective_log_radix - 1) / legion_collective_log_radix;
      int log_remainder = log_nodes % legion_collective_log_radix;
      if (log_remainder > 0)
      {
        // We have an incomplete last stage
        legion_collective_last_radix = 1 << log_remainder;
        legion_collective_last_log_radix = log_remainder;
        // Now we can compute the number of participating stages
        legion_collective_participating_spaces = 
          1 << ((legion_collective_stages - 1) * legion_collective_log_radix +
                 legion_collective_last_log_radix);
      }
      else
      {
        legion_collective_last_radix = legion_collective_radix;
        legion_collective_last_log_radix = legion_collective_log_radix;
        legion_collective_participating_spaces = 
          1 << (legion_collective_stages * legion_collective_log_radix);
      }
    }

#ifdef TRACE_ALLOCATION
    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(
                                       AllocationType a, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Runtime *rt = Runtime::the_runtime;
      if (rt != NULL)
        rt->trace_allocation(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_free(AllocationType a, 
                                                 size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      Runtime *rt = Runtime::the_runtime;
      if (rt != NULL)
        rt->trace_free(a, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ Runtime* LegionAllocation::find_runtime(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::the_runtime;
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


    //--------------------------------------------------------------------------
    /*static*/ char* BitMaskHelper::to_string(const uint64_t *bits, int count)
    //--------------------------------------------------------------------------
    {
      char *result = (char*)malloc((((count + 7) >> 3) + 1)*sizeof(char));
      assert(result != 0);
      char *p = result;
      // special case for non-multiple-of-64
      if((count & 63) != 0) {
        // each nibble (4 bits) takes one character
        int nibbles = ((count & 63) + 3) >> 2;
        sprintf(p, "%*.*" MASK_FMT, nibbles, nibbles, bits[count >> 6]);
        p += nibbles;
      }
      // rest are whole words
      int idx = (count >> 6);
      while(idx >= 0) {
        sprintf(p, "%16.16" MASK_FMT, bits[--idx]);
        p += 16;
      }
      return result;
    }

  }; // namespace Internal 
}; // namespace Legion 

// EOF

