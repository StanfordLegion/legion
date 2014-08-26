/* Copyright 2014 Stanford University
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


#include "legion_tasks.h"
#include "legion_spy.h"
#include "legion_trace.h"
#include "legion_logging.h"
#include "legion_profiling.h"
#include <algorithm>

#define PRINT_REG(reg) (reg).index_space.id,(reg).field_space.id, (reg).tree_id

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
    // Task Operation 
    /////////////////////////////////////////////////////////////
  
    //--------------------------------------------------------------------------
    TaskOp::TaskOp(Runtime *rt)
      : Task(), SpeculativeOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskOp::~TaskOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Mappable::MappableKind TaskOp::get_mappable_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_MAPPABLE;
    }

    //--------------------------------------------------------------------------
    Task* TaskOp::as_mappable_task(void) const
    //--------------------------------------------------------------------------
    {
      TaskOp *proxy_this = const_cast<TaskOp*>(this);
      return proxy_this;
    }

    //--------------------------------------------------------------------------
    Copy* TaskOp::as_mappable_copy(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Inline* TaskOp::as_mappable_inline(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Acquire* TaskOp::as_mappable_acquire(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    Release* TaskOp::as_mappable_release(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    UniqueID TaskOp::get_unique_mappable_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate_task(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      complete_received = false;
      commit_received = false;
      children_complete = false;
      children_commit = false;
      children_complete_invoked = false;
      children_commit_invoked = false;
      needs_state = false;
      arg_manager = NULL;
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate_task(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      indexes.clear();
      regions.clear();
      futures.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (args != NULL)
      {
        if (arg_manager != NULL)
        {
          // If the arg manager is not NULL then we delete the
          // argument manager and just zero out the arguments
          if (arg_manager->remove_reference())
            delete arg_manager;
          arg_manager = NULL;
        }
        else
          free(args);
        args = NULL;
        arglen = 0;
      }
      if (local_args != NULL)
      {
        free(local_args);
        local_args = NULL;
        local_arglen = 0;
      }
      early_mapped_regions.clear();
      created_regions.clear();
      created_fields.clear();
      created_field_spaces.clear();
      created_index_spaces.clear();
      deleted_regions.clear();
      deleted_fields.clear();
      deleted_field_spaces.clear();
      deleted_index_spaces.clear();
      enclosing_physical_contexts.clear();
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(task_id);
      rez.serialize(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        pack_index_space_requirement(indexes[idx], rez);
      rez.serialize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        pack_region_requirement(regions[idx], rez);
      rez.serialize(futures.size());
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        bool add_remote_reference = futures[idx].impl->send_future(target);
        rez.serialize(futures[idx].impl->did);
        rez.serialize(add_remote_reference);
      }
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize<bool>((arg_manager != NULL));
      rez.serialize(arglen);
      rez.serialize(args,arglen);
      rez.serialize(map_id);
      rez.serialize(tag);
      rez.serialize(is_index_space);
      rez.serialize(must_parallelism);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      rez.serialize(local_arglen);
      rez.serialize(local_args,local_arglen);
      rez.serialize(orig_proc);
      // No need to pack current proc, it will get set when we unpack
      rez.serialize(steal_count);
      rez.serialize(depth);
      // No need to pack remote, it will get set
      rez.serialize(speculated);
      // No need to pack premapped, must be true or can't be sent remotely
      // Can figure out variants remotely
      rez.serialize(selected_variant);
      rez.serialize(target_proc);
      // Can't be sending inline tasks remotely
      rez.serialize(spawn_task);
      rez.serialize(map_locally);
      rez.serialize(profile_task);
      rez.serialize(task_priority);
      rez.serialize(needs_state);
      rez.serialize(early_mapped_regions.size());
      for (std::map<unsigned,InstanceRef>::iterator it = 
            early_mapped_regions.begin(); it != 
            early_mapped_regions.end(); it++)
      {
        rez.serialize(it->first);
        // Need to send the region tree shape for this reference
        runtime->forest->send_tree_shape(regions[it->first], target);
        it->second.pack_reference(rez, target);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_base_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(task_id);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      indexes.resize(num_indexes);
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        unpack_index_space_requirement(indexes[idx], derez);
      size_t num_regions;
      derez.deserialize(num_regions);
      regions.resize(num_regions);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        unpack_region_requirement(regions[idx], derez);
      size_t num_futures;
      derez.deserialize(num_futures);
      futures.resize(num_futures);
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        DistributedID future_did;
        derez.deserialize(future_did);
        futures[idx] = Future(runtime->find_future(future_did));
        // See if we need to add a remote reference
        // to this future
        bool add_remote_reference;
        derez.deserialize(add_remote_reference);
        if (add_remote_reference)
          futures[idx].impl->add_held_remote_reference();
      }
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      bool has_arg_manager;
      derez.deserialize(has_arg_manager);
      derez.deserialize(arglen);
      if (arglen > 0)
      {
        if (has_arg_manager)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(arg_manager == NULL);
#endif
          arg_manager = new AllocManager(arglen);
          arg_manager->add_reference();
          args = arg_manager->get_allocation();
        }
        else
          args = malloc(arglen);
        derez.deserialize(args,arglen);
      }
      derez.deserialize(map_id);
      derez.deserialize(tag);
      derez.deserialize(is_index_space);
      derez.deserialize(must_parallelism);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      derez.deserialize(local_arglen);
      if (local_arglen > 0)
      {
        local_args = malloc(local_arglen);
        derez.deserialize(local_args,local_arglen);
      }
      derez.deserialize(orig_proc);
      derez.deserialize(steal_count);
      derez.deserialize(depth);
      derez.deserialize(speculated);
      premapped = true;
      variants = Runtime::get_variant_collection(task_id);
      derez.deserialize(selected_variant);
      derez.deserialize(target_proc);
      inline_task = false;
      derez.deserialize(spawn_task);
      derez.deserialize(map_locally);
      derez.deserialize(profile_task);
      derez.deserialize(task_priority);
      derez.deserialize(needs_state);
      size_t num_early;
      derez.deserialize(num_early);
      for (unsigned idx = 0; idx < num_early; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        early_mapped_regions[index] = 
        InstanceRef::unpack_reference(derez, runtime->forest, depth);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::process_unpack_task(Runtime *rt, 
                                                Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      Processor current;
      derez.deserialize(current);
      TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case INDIVIDUAL_TASK_KIND:
          {
            IndividualTask *task = rt->get_available_individual_task();
            if (task->unpack_task(derez, current))
              rt->add_to_ready_queue(current, task, 
                                     false/*prev fail*/);
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task();
            if (task->unpack_task(derez, current))
              rt->add_to_ready_queue(current, task,
                                     false/*prev fail*/);
            break;
          }
        case POINT_TASK_KIND:
        case REMOTE_TASK_KIND:
        case INDEX_TASK_KIND:
        default:
          assert(false); // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::mark_stolen(Processor new_target)
    //--------------------------------------------------------------------------
    {
      steal_count++;
      target_proc = new_target;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_base_task(SingleTask *ctx, bool track, 
                                      const Predicate &p,
                                      Processor::TaskFuncID tid)
    //-------------------------------------------------------------------------- 
    {
      initialize_speculation(ctx, track, regions.size(), p);
      // Fill in default values for all of the Task fields
      orig_proc = ctx->get_executing_processor();
      current_proc = orig_proc;
      steal_count = 0;
      depth = parent_ctx->depth+1;
      speculated = false;
      premapped = false;
      variants = Runtime::get_variant_collection(tid);
      selected_variant = 0;
      target_proc = orig_proc;
      inline_task = false;
      spawn_task = false;
      map_locally = false;
      profile_task = false;
      task_priority = 0;
      start_time = 0;
      stop_time = 0;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_physical_contexts(void)
    //--------------------------------------------------------------------------
    {
      enclosing_physical_contexts.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        enclosing_physical_contexts[idx] = 
          get_enclosing_physical_context(idx);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::check_empty_field_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].privilege_fields.empty())
        {
          log_task(LEVEL_WARNING,"WARNING: REGION REQUIREMENT %d OF "
                                 "TASK %s (ID %lld) HAS NO PRIVILEGE "
                                 "FIELDS! DID YOU FORGET THEM?!?",
                                 idx, variants->name, 
                                 get_unique_task_id());
        }
      }
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return variants->name;
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_complete(void) 
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(!complete_received);
        assert(!commit_received);
#endif
        complete_received = true;
        // If all our children are also complete then we are done
        task_complete = children_complete;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false; 
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(complete_received);
        assert(!commit_received);
#endif
        commit_received = true;
        // If we already received the child commit then we
        // are ready to commit this task
        task_commit = children_commit;
      }
      if (task_commit)
        trigger_task_commit();
    }

    //--------------------------------------------------------------------------
    void TaskOp::continue_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the ready queue
      runtime->add_to_ready_queue(current_proc, this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_creation(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Create a new logical region 
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(created_regions.find(handle) == created_regions.end());
#endif
      created_regions.insert(handle); 
      add_created_region(handle);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_deletion(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock o_lock(op_lock);
      std::set<LogicalRegion>::iterator finder = created_regions.find(handle);
      // See if we created this region, if so remove it from the list
      // of created regions, otherwise add it to the list of deleted
      // regions to flow backwards
      if (finder == created_regions.end())
        deleted_regions.insert(handle);
      else
      {
        created_regions.erase(finder);
        remove_created_region(handle);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_creations(const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<LogicalRegion>::const_iterator it = regs.begin();
            it != regs.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(created_regions.find(*it) == created_regions.end());
#endif
        created_regions.insert(*it);
        add_created_region(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_deletions(const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<LogicalRegion>::const_iterator it = regs.begin();
            it != regs.end(); it++)
      {
        std::set<LogicalRegion>::iterator finder = created_regions.find(*it);
        if (finder == created_regions.end())
          deleted_regions.insert(*it);
        else
        {
          created_regions.erase(finder);
          remove_created_region(*it);
        }
      }
    } 

    //--------------------------------------------------------------------------
    void TaskOp::register_field_creation(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::pair<FieldSpace,FieldID> key(handle,fid);
#ifdef DEBUG_HIGH_LEVEL
      assert(created_fields.find(key) == created_fields.end());
#endif
      created_fields.insert(key);
      add_created_field(handle, fid);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_creations(FieldSpace handle,
                                          const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        std::pair<FieldSpace,FieldID> key(handle,fields[idx]);
#ifdef DEBUG_HIGH_LEVEL
        assert(created_fields.find(key) == created_fields.end());
#endif
        created_fields.insert(key);
        add_created_field(handle, fields[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_deletions(FieldSpace handle,
                                         const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
        std::pair<FieldSpace,FieldID> key(handle,*it);
        std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
          created_fields.find(key);
        if (finder == created_fields.end())
          deleted_fields.insert(key);
        else
        {
          created_fields.erase(finder);
          remove_created_field(handle, *it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_creations(
                        const std::set<std::pair<FieldSpace,FieldID> > &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
            fields.begin(); it != fields.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(created_fields.find(*it) == created_fields.end());
#endif
        created_fields.insert(*it);
        add_created_field(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_deletions(
                        const std::set<std::pair<FieldSpace,FieldID> > &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
            fields.begin(); it != fields.end(); it++)
      {
        std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
          created_fields.find(*it);
        if (finder == created_fields.end())
          deleted_fields.insert(*it);
        else
        {
          created_fields.erase(finder);
          remove_created_field(it->first, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_creation(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(created_field_spaces.find(space) == created_field_spaces.end());
#endif
      created_field_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_deletion(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::deque<FieldID> to_delete;
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
            created_fields.begin(); it != created_fields.end(); it++)
      {
        if (it->first == space)
        {
          to_delete.push_back(it->second);
          remove_created_field(space, it->second);
        }
      }
      for (unsigned idx = 0; idx < to_delete.size(); idx++)
      {
        std::pair<FieldSpace,FieldID> key(space, to_delete[idx]);
        created_fields.erase(key);
      }
      std::set<FieldSpace>::iterator finder = created_field_spaces.find(space);
      if (finder == created_field_spaces.end())
        deleted_field_spaces.insert(space);
      else
        created_field_spaces.erase(finder);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_creations(
                                            const std::set<FieldSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<FieldSpace>::const_iterator it = spaces.begin();
            it != spaces.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(created_field_spaces.find(*it) == created_field_spaces.end());
#endif
        created_field_spaces.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_deletions(
                                            const std::set<FieldSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<FieldSpace>::const_iterator it = spaces.begin();
            it != spaces.end(); it++)
      {
        std::deque<FieldID> to_delete;
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator cit = 
              created_fields.begin(); cit != created_fields.end(); cit++)
        {
          if (cit->first == *it)
          {
            to_delete.push_back(cit->second);
            remove_created_field(*it, cit->second);
          }
        }
        for (unsigned idx = 0; idx < to_delete.size(); idx++)
        {
          std::pair<FieldSpace,FieldID> key(*it, to_delete[idx]);
          created_fields.erase(key);
        }
        std::set<FieldSpace>::iterator finder = created_field_spaces.find(*it);
        if (finder == created_field_spaces.end())
          deleted_field_spaces.insert(*it);
        else
          created_field_spaces.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    bool TaskOp::has_created_index_space(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      return (created_index_spaces.find(space) != created_index_spaces.end());
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_creation(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(created_index_spaces.find(space) == created_index_spaces.end());
#endif
      created_index_spaces.insert(space);
      add_created_index(space);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_deletion(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::set<IndexSpace>::iterator finder = created_index_spaces.find(space);
      if (finder == created_index_spaces.end())
        deleted_index_spaces.insert(space);
      else
      {
        created_index_spaces.erase(finder);
        remove_created_index(space);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_creations(
                                            const std::set<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<IndexSpace>::const_iterator it = spaces.begin();
            it != spaces.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(created_index_spaces.find(*it) == created_index_spaces.end());
#endif
        created_index_spaces.insert(*it);
        add_created_index(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_deletions(
                                            const std::set<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<IndexSpace>::const_iterator it = spaces.begin();
            it != spaces.end(); it++)
      {
        std::set<IndexSpace>::iterator finder = created_index_spaces.find(*it);
        if (finder == created_index_spaces.end())
          deleted_index_spaces.insert(*it);
        else
        {
          created_index_spaces.erase(finder);
          remove_created_index(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::return_privilege_state(TaskOp *target)
    //--------------------------------------------------------------------------
    {
      if (!created_regions.empty())
        target->register_region_creations(created_regions);
      if (!deleted_regions.empty())
        target->register_region_deletions(deleted_regions);
      if (!created_fields.empty())
        target->register_field_creations(created_fields);
      if (!deleted_fields.empty())
        target->register_field_deletions(deleted_fields);
      if (!created_field_spaces.empty())
        target->register_field_space_creations(created_field_spaces);
      if (!deleted_field_spaces.empty())
        target->register_field_space_deletions(deleted_field_spaces);
      if (!created_index_spaces.empty())
        target->register_index_space_creations(created_index_spaces);
      if (!deleted_index_spaces.empty())
        target->register_index_space_deletions(deleted_index_spaces);
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_privilege_state(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Shouldn't need the lock here since we only do this
      // while there is no one else executing
      RezCheck z(rez);
      rez.serialize(created_regions.size());
      for (std::set<LogicalRegion>::const_iterator it =
            created_regions.begin(); it != created_regions.end(); it++)
      {
        rez.serialize(*it);
        runtime->forest->send_tree_shape(*it, target);
      }
      rez.serialize(deleted_regions.size());
      for (std::set<LogicalRegion>::const_iterator it =
            deleted_regions.begin(); it != deleted_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(created_fields.size());
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it =
            created_fields.begin(); it != created_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(deleted_fields.size());
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
            deleted_fields.begin(); it != deleted_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(created_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = 
            created_field_spaces.begin(); it != 
            created_field_spaces.end(); it++)
      {
        rez.serialize(*it);
        runtime->forest->send_tree_shape(*it, target);
      }
      rez.serialize(deleted_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = 
            deleted_field_spaces.begin(); it !=
            deleted_field_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(created_index_spaces.size());
      for (std::set<IndexSpace>::const_iterator it = 
            created_index_spaces.begin(); it != 
            created_index_spaces.end(); it++)
      {
        rez.serialize(*it);
        runtime->forest->send_tree_shape(*it, target);
      }
      rez.serialize(deleted_index_spaces.size());
      for (std::set<IndexSpace>::const_iterator it = 
            deleted_index_spaces.begin(); it !=
            deleted_index_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_privilege_state(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while doing the unpack to avoid conflicting
      // with anyone else returning state
      DerezCheck z(derez);
      size_t num_created_regions;
      derez.deserialize(num_created_regions);
      for (unsigned idx = 0; idx < num_created_regions; idx++)
      {
        LogicalRegion reg;
        derez.deserialize(reg);
        created_regions.insert(reg);
      }
      size_t num_deleted_regions;
      derez.deserialize(num_deleted_regions);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
      {
        LogicalRegion reg;
        derez.deserialize(reg);
        deleted_regions.insert(reg);
      }
      size_t num_created_fields;
      derez.deserialize(num_created_fields);
      for (unsigned idx = 0; idx < num_created_fields; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        FieldID fid;
        derez.deserialize(fid);
        created_fields.insert(std::pair<FieldSpace,FieldID>(sp,fid));
      }
      size_t num_deleted_fields;
      derez.deserialize(num_deleted_fields);
      for (unsigned idx = 0; idx < num_deleted_fields; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        FieldID fid;
        derez.deserialize(fid);
        deleted_fields.insert(std::pair<FieldSpace,FieldID>(sp,fid));
      }
      size_t num_created_field_spaces;
      derez.deserialize(num_created_field_spaces);
      for (unsigned idx = 0; idx < num_created_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        created_field_spaces.insert(sp);
      }
      size_t num_deleted_field_spaces;
      derez.deserialize(num_deleted_field_spaces);
      for (unsigned idx = 0; idx < num_deleted_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        deleted_field_spaces.insert(sp);
      }
      size_t num_created_index_spaces;
      derez.deserialize(num_created_index_spaces);
      for (unsigned idx = 0; idx < num_created_index_spaces; idx++)
      {
        IndexSpace sp;
        derez.deserialize(sp);
        created_index_spaces.insert(sp);
      }
      size_t num_deleted_index_spaces;
      derez.deserialize(num_deleted_index_spaces);
      for (unsigned idx = 0; idx < num_deleted_index_spaces; idx++)
      {
        IndexSpace sp;
        derez.deserialize(sp);
        deleted_index_spaces.insert(sp);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::perform_privilege_checks(void)
    //--------------------------------------------------------------------------
    {
      // First check the index privileges
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        LegionErrorType et = parent_ctx->check_privilege(indexes[idx]);
        switch (et)
        {
          case NO_ERROR:
            break;
          case ERROR_BAD_PARENT_INDEX:
            {
              log_index(LEVEL_ERROR,"Parent task %s (ID %lld) of task %s "
                                    "(ID %lld) "
                                    "does not have an index requirement for "
                                    "index space " IDFMT " as a parent of "
                                    "child task's index requirement index %d",
                                    parent_ctx->variants->name, 
                                    parent_ctx->get_unique_task_id(),
                                    this->variants->name, get_unique_task_id(), 
                                    indexes[idx].parent.id, idx);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARENT_INDEX);
            }
          case ERROR_BAD_INDEX_PATH:
            {
              log_index(LEVEL_ERROR,"Index space " IDFMT " is not a sub-space "
                                    "of parent index space " IDFMT " for index "
                                    "requirement %d of task %s (ID %lld)",
                                    indexes[idx].handle.id, 
                                    indexes[idx].parent.id, idx,
                                    this->variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_INDEX_PATH);
            }
          case ERROR_BAD_INDEX_PRIVILEGES:
            {
              log_index(LEVEL_ERROR,"Privileges %x for index space " IDFMT 
                                    " are not a subset of privileges of parent "
                                    "task's privileges for index space "
                                    "requirement %d of task %s (ID %lld)",
                                    indexes[idx].privilege, 
                                    indexes[idx].handle.id, idx, 
                                    this->variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_INDEX_PRIVILEGES);
            }
          default:
            assert(false); // Should never happen
        }
      }
      // Now check the region requirement privileges
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Verify that the requirement is self-consistent
        FieldID bad_field;
        LegionErrorType et = runtime->verify_requirement(regions[idx], 
                                                         bad_field); 
        if ((et == NO_ERROR) && !is_index_space && 
            ((regions[idx].handle_type == PART_PROJECTION) || 
             (regions[idx].handle_type == REG_PROJECTION)))
          et = ERROR_BAD_PROJECTION_USE;
        // If that worked, then check the privileges with the parent context
        if (et == NO_ERROR)
          et = parent_ctx->check_privilege(regions[idx], bad_field);
        switch (et)
        {
          case NO_ERROR:
            break;
          case ERROR_INVALID_REGION_HANDLE:
            {
              log_region(LEVEL_ERROR, "Invalid region handle (" IDFMT ",%d,%d)"
                                    " for region requirement %d of task %s "
                                    "(ID %lld)",
                                    regions[idx].region.index_space.id, 
                                    regions[idx].region.field_space.id, 
                                    regions[idx].region.tree_id, idx, 
                                    variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_REGION_HANDLE);
            }
          case ERROR_INVALID_PARTITION_HANDLE:
            {
              log_region(LEVEL_ERROR, "Invalid partition handle (%x,%d,%d) "
                            "for partition requirement %d of task %s "
                            "(ID %lld)",
                                      regions[idx].partition.index_partition, 
                                      regions[idx].partition.field_space.id, 
                                      regions[idx].partition.tree_id, idx, 
                                      variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_PARTITION_HANDLE);
            }
          case ERROR_BAD_PROJECTION_USE:
            {
              log_region(LEVEL_ERROR,"Projection region requirement %d used "
                                      "in non-index space task %s",
                                      idx, this->variants->name);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PROJECTION_USE);
            }
          case ERROR_NON_DISJOINT_PARTITION:
            {
              log_region(LEVEL_ERROR,"Non disjoint partition selected for "
                                      "writing region requirement %d of task "
                                      "%s.  All projection partitions "
                                      "which are not read-only and not reduce "
                                      "must be disjoint", 
                                      idx, this->variants->name);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_NON_DISJOINT_PARTITION);
            }
          case ERROR_FIELD_SPACE_FIELD_MISMATCH:
            {
              FieldSpace sp = (regions[idx].handle_type == SINGULAR) || 
                (regions[idx].handle_type == REG_PROJECTION) 
                  ? regions[idx].region.field_space : 
                    regions[idx].partition.field_space;
              log_region(LEVEL_ERROR,"Field %d is not a valid field of field "
                                    "space %d for region %d of task %s "
                                    "(ID %lld)",
                                    bad_field, sp.id, idx, this->variants->name,
                                    get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
            }
          case ERROR_INVALID_INSTANCE_FIELD:
            {
              log_region(LEVEL_ERROR,"Instance field %d is not one of the "
                                      "privilege fields for region %d of "
                                      "task %s (ID %lld)",
                                      bad_field, idx, this->variants->name, 
                                      get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_INSTANCE_FIELD);
            }
          case ERROR_DUPLICATE_INSTANCE_FIELD:
            {
              log_region(LEVEL_ERROR, "Instance field %d is a duplicate for "
                                      "region %d of task %s (ID %lld)",
                                      bad_field, idx, this->variants->name, 
                                      get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_DUPLICATE_INSTANCE_FIELD);
            }
          case ERROR_BAD_PARENT_REGION:
            {
              log_region(LEVEL_ERROR,"Parent task %s (ID %lld) of task %s "
                                      "(ID %lld) does not have a region "
                                      "requirement for region (" IDFMT 
                                      ",%x,%x) as a parent of child task's "
                                      "region requirement index %d",
                                      parent_ctx->variants->name, 
                                      parent_ctx->get_unique_task_id(),
                                      this->variants->name, 
                                      get_unique_task_id(), 
                                      regions[idx].region.index_space.id,
                                      regions[idx].region.field_space.id, 
                                      regions[idx].region.tree_id, idx);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARENT_REGION);
            }
          case ERROR_BAD_REGION_PATH:
            {
              log_region(LEVEL_ERROR,"Region (" IDFMT ",%x,%x) is not a "
                                      "sub-region of parent region (" IDFMT 
                                      ",%x,%x) for region requirement %d of "
                                      "task %s (ID %lld)",
                                      regions[idx].region.index_space.id,
                                      regions[idx].region.field_space.id, 
                                      regions[idx].region.tree_id,
                                      PRINT_REG(regions[idx].parent), idx,
                                      this->variants->name, 
                                      get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_PATH);
            }
          case ERROR_BAD_PARTITION_PATH:
            {
              log_region(LEVEL_ERROR,"Partition (%x,%x,%x) is not a "
                                     "sub-partition of parent region "
                                     "(" IDFMT ",%x,%x) for region "
                                     "requirement %d task %s (ID %lld)",
                                      regions[idx].partition.index_partition, 
                                      regions[idx].partition.field_space.id, 
                                      regions[idx].partition.tree_id, 
                                      PRINT_REG(regions[idx].parent), idx,
                                      this->variants->name, 
                                      get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARTITION_PATH);
            }
          case ERROR_BAD_REGION_TYPE:
            {
              log_region(LEVEL_ERROR,"Region requirement %d of task %s "
                                     "(ID %lld) "
                                     "cannot find privileges for field %d in "
                                     "parent task",
                                      idx, this->variants->name, 
                                      get_unique_task_id(), bad_field);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_TYPE);
            }
          case ERROR_BAD_REGION_PRIVILEGES:
            {
              log_region(LEVEL_ERROR,"Privileges %x for region (" IDFMT 
                                     ",%x,%x) are not a subset of privileges " 
                                     "of parent task's privileges for "
                                     "region requirement %d of task %s "
                                     "(ID %lld)",
                                     regions[idx].privilege, 
                                     regions[idx].region.index_space.id,
                                     regions[idx].region.field_space.id, 
                                     regions[idx].region.tree_id, idx, 
                                     this->variants->name, 
                                     get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_PRIVILEGES);
            }
          case ERROR_BAD_PARTITION_PRIVILEGES:
            {
              log_region(LEVEL_ERROR,"Privileges %x for partition (%x,%x,%x) "
                                     "are not a subset of privileges of parent "
                                     "task's privileges for "
                                     "region requirement %d of task %s "
                                     "(ID %lld)",
                                     regions[idx].privilege, 
                                     regions[idx].partition.index_partition, 
                                     regions[idx].partition.field_space.id, 
                                     regions[idx].partition.tree_id, idx, 
                                     this->variants->name, 
                                     get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARTITION_PRIVILEGES);
            }
          default:
            assert(false); // Should never happen
        }
      }
    }

    //--------------------------------------------------------------------------
    InstanceRef TaskOp::find_premapped_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,InstanceRef>::const_iterator finder = 
        early_mapped_regions.find(idx);
      if (finder != early_mapped_regions.end())
        return finder->second;
      else
        return InstanceRef();
    }

    //--------------------------------------------------------------------------
    RegionTreeContext TaskOp::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->find_enclosing_physical_context(regions[idx].parent);
    }

    //--------------------------------------------------------------------------
    void TaskOp::clone_task_op_from(TaskOp *rhs, Processor p, 
                                    bool stealable, bool duplicate_args)
    //--------------------------------------------------------------------------
    {
      // From Operation
      this->parent_ctx = rhs->parent_ctx;
      if (rhs->must_epoch != NULL)
        this->set_must_epoch(rhs->must_epoch, this->must_epoch_index);
      // From Task
      this->task_id = rhs->task_id;
      this->indexes = rhs->indexes;
      this->regions = rhs->regions;
      this->futures = rhs->futures;
      this->grants = rhs->grants;
      this->wait_barriers = rhs->wait_barriers;
      this->arrive_barriers = rhs->arrive_barriers;
      this->arglen = rhs->arglen;
      if (rhs->arg_manager != NULL)
      {
        if (duplicate_args)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(arg_manager == NULL);
#endif
          this->arg_manager = new AllocManager(this->arglen); 
          this->arg_manager->add_reference();
          this->args = this->arg_manager->get_allocation();
          memcpy(this->args, rhs->args, this->arglen);
        }
        else
        {
          // No need to actually do the copy in this case
          this->arg_manager = rhs->arg_manager; 
          this->arg_manager->add_reference();
          this->args = arg_manager->get_allocation();
        }
      }
      else if (arglen > 0)
      {
        // If there is no argument manager then we do the copy no matter what
        this->args = malloc(arglen);
        memcpy(args,rhs->args,arglen);
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      this->is_index_space = rhs->is_index_space;
      this->orig_proc = rhs->orig_proc;
      this->current_proc = rhs->current_proc;
      this->steal_count = rhs->steal_count;
      this->depth = rhs->depth;
      this->speculated = rhs->speculated;
      this->premapped = rhs->premapped;
      this->variants = rhs->variants;
      this->selected_variant = rhs->selected_variant;
      this->schedule = rhs->schedule;
      this->target_proc = p; // set the target processor
      this->inline_task = rhs->inline_task;
      this->spawn_task = stealable; // set spawn to stealable
      this->map_locally = rhs->map_locally;
      this->profile_task = rhs->profile_task;
      this->task_priority = rhs->task_priority;
      // From TaskOp
      this->early_mapped_regions = rhs->early_mapped_regions;
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_grants(const std::vector<Grant> &requested_grants)
    //--------------------------------------------------------------------------
    {
      grants = requested_grants;
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(get_task_completion());
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_arrival_barriers(
                                const std::vector<PhaseBarrier> &phase_barriers) 
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhaseBarrier>::const_iterator it = 
            phase_barriers.begin(); it != phase_barriers.end(); it++)
      {
        // Update the arrival count
        // TODO: Put this back in once Sean fixes barriers
#if 0
        arrive_barriers.push_back(
            PhaseBarrier(it->phase_barrier.alter_arrival_count(1),
                         it->participants));
#else
        arrive_barriers.push_back(*it);
#endif
        // Note it is imperative we do this off the new barrier
        // generated after updating the arrival count.
        arrive_barriers.back().phase_barrier.arrive(1, get_task_completion());
#ifdef LEGION_LOGGING
        LegionLogging::log_event_dependence(
            Machine::get_executing_processor(),
            it->phase_barrier, arrive_barriers.back().phase_barrier);
        LegionLogging::log_event_dependence(
            Machine::get_executing_processor(),
            get_task_completion(), arrive_barriers.back().phase_barrier);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(it->phase_barrier,
                                        arrive_barriers.back().phase_barrier); 
        LegionSpy::log_event_dependence(get_task_completion(),
                                        arrive_barriers.back().phase_barrier);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_point_region_requirements(void)
    //--------------------------------------------------------------------------
    {
      // Update the region requirements for this point
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type == SINGULAR)
          continue;
        else if (regions[idx].handle_type == PART_PROJECTION)
        {
          // Check to see if we're doing default projection
          if (regions[idx].projection == 0)
          {
            Color subregion_color;
            switch(index_point.get_dim()) {
            case 0:
              {
                subregion_color = unsigned(index_point.get_index());
                break;
              }
            case 1:
              {
                Arrays::Rect<1> color_space = 
                  runtime->forest->get_index_partition_color_space(
                  regions[idx].partition.get_index_partition()).get_rect<1>();
                Arrays::CArrayLinearization<1> color_space_lin(color_space);
                subregion_color = 
                  (Color)(color_space_lin.image(index_point.get_point<1>()));
                break;
              }
            case 2:
              {
                Arrays::Rect<2> color_space = 
                  runtime->forest->get_index_partition_color_space(
                  regions[idx].partition.get_index_partition()).get_rect<2>();
                Arrays::CArrayLinearization<2> color_space_lin(color_space);
                subregion_color = 
                  (Color)(color_space_lin.image(index_point.get_point<2>()));
                break;
              }
            case 3:
              {
                Arrays::Rect<3> color_space = 
                  runtime->forest->get_index_partition_color_space(
                  regions[idx].partition.get_index_partition()).get_rect<3>();
                Arrays::CArrayLinearization<3> color_space_lin(color_space);
                subregion_color = 
                  (Color)(color_space_lin.image(index_point.get_point<3>()));
                break;
              }
            default:
              log_task(LEVEL_ERROR,"Projection ID 0 is invalid for tasks whose "
                                   "points are larger than three dimensional "
                                   "unsigned integers.  Points for task %s "
                                   "have elements of %d dimensions",
                                  this->variants->name, index_point.get_dim());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_IDENTITY_PROJECTION_USE);
            }
            regions[idx].region = 
              runtime->forest->get_logical_subregion_by_color(
                  regions[idx].partition, subregion_color);
          }
          else
          {
            PartitionProjectionFnptr projfn = 
              Runtime::find_partition_projection_function(
                  regions[idx].projection);
            regions[idx].region = 
              (*projfn)(regions[idx].partition,index_point,runtime->high_level);
          }
          // Update the region requirement kind 
          regions[idx].handle_type = SINGULAR;
          // Update the blocking factor as well
          regions[idx].max_blocking_factor = 
            runtime->forest->get_domain_volume(regions[idx].region);
        }
        else
        {
          // This should be region projection
#ifdef DEBUG_HIGH_LEVEL
          assert(regions[idx].handle_type == REG_PROJECTION);
#endif
          if (regions[idx].projection != 0)
          {
            RegionProjectionFnptr projfn = 
              Runtime::find_region_projection_function(
                  regions[idx].projection);
            regions[idx].region = 
              (*projfn)(regions[idx].region,index_point,runtime->high_level);
          }
          // Otherwise we are the default case in which 
          // case we don't need to do anything
          // Update the region requirement kind
          regions[idx].handle_type = SINGULAR;
          // Update the blocking factor as well
          regions[idx].max_blocking_factor = 
            runtime->forest->get_domain_volume(regions[idx].region);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TaskOp::early_map_regions(void)
    //--------------------------------------------------------------------------
    {
      // Invoke the mapper to perform the early mappings
      bool notify = runtime->invoke_mapper_pre_map_task(current_proc, this);
      std::vector<MappingRef> mapping_refs(regions.size());
      bool success = true;
      bool has_early_maps = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement &req = regions[idx];
        if (req.must_early_map || req.early_map)
        {
          if (req.handle_type == SINGULAR)
          {
            has_early_maps = true;
            RegionTreePath mapping_path;
            initialize_mapping_path(mapping_path, req, req.region); 
            mapping_refs[idx] = runtime->forest->map_physical_region(
                                       enclosing_physical_contexts[idx],
                                                      mapping_path,
                                                      req, idx, this,
                                                      current_proc,
                                                      current_proc
#ifdef DEBUG_HIGH_LEVEL
                                                      , get_logging_name()
                                                      , unique_op_id
#endif
                                                      );
            if (!mapping_refs[idx].has_ref())
            {
              success = false;
              regions[idx].mapping_failed = true;
              break;
            }
          }
          else
          {
            log_task(LEVEL_WARNING,"Ignoring request to early map region %d "
                                    "for task %s (ID %lld) which is a non-"
                                    "singular region requirement and therefore"
                                    "cannot be early mapped.",
                                    idx, variants->name, get_unique_task_id());
            req.early_map = false;
          }
        }
      }
      if (has_early_maps)
      {
        if (success)
        {
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            RegionRequirement &req = regions[idx];
            if (req.must_early_map || req.early_map)
            {
              Event term_event = get_task_completion();
#ifdef DEBUG_HIGH_LEVEL
              RegionTreePath mapping_path;
              assert(req.handle_type == SINGULAR);
              initialize_mapping_path(mapping_path, req, req.region);
#endif
              early_mapped_regions[idx] = 
                runtime->forest->register_physical_region(
                                           enclosing_physical_contexts[idx],
                                                          mapping_refs[idx],
                                                          req, idx, this,
                                                          current_proc,
                                                          term_event
#ifdef DEBUG_HIGH_LEVEL
                                                          , get_logging_name()
                                                          , unique_op_id
                                                          , mapping_path
#endif
                                                          );
#ifdef DEBUG_HIGH_LEVEL
              assert(early_mapped_regions[idx].has_ref());
#endif
              if (notify)
              {
                regions[idx].mapping_failed = false;
                regions[idx].selected_memory = 
                  early_mapped_regions[idx].get_memory();
              }
            }
          }
          if (notify)
            runtime->invoke_mapper_notify_result(current_proc, this);
        }
        else
        {
          runtime->invoke_mapper_failed_mapping(current_proc, this);
          mapping_refs.clear(); 
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_complete(void)
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock); 
#ifdef DEBUG_HIGH_LEVEL
        assert(!children_complete);
        assert(!children_commit);
#endif
        children_complete = true;
        task_complete = complete_received;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_committed(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(children_complete);
        assert(!children_commit);
#endif
        children_commit = true;
        task_commit = commit_received;
      }
      if (task_commit)
        trigger_task_commit();
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::pack_index_space_requirement(
                              const IndexSpaceRequirement &req, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.handle);
      rez.serialize(req.privilege);
      rez.serialize(req.parent);
      // no need to send verified
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::unpack_index_space_requirement(
                                IndexSpaceRequirement &req, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(req.handle);
      derez.deserialize(req.privilege);
      derez.deserialize(req.parent);
      req.verified = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::pack_region_requirement(
                                  const RegionRequirement &req, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.region);
      rez.serialize(req.partition);
      rez.serialize(req.privilege_fields.size());
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(req.instance_fields.size());
      for (std::vector<FieldID>::const_iterator it = 
            req.instance_fields.begin(); it != req.instance_fields.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(req.privilege);
      rez.serialize(req.prop);
      rez.serialize(req.parent);
      rez.serialize(req.redop);
      rez.serialize(req.tag);
      rez.serialize(req.flags);
      rez.serialize(req.handle_type);
      rez.serialize(req.projection);
      rez.serialize(req.current_instances.size());
      for (std::map<Memory,bool>::const_iterator it = 
            req.current_instances.begin(); it != 
            req.current_instances.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(req.max_blocking_factor);
      rez.serialize(req.must_early_map);
      rez.serialize(req.restricted);
      rez.serialize(req.selected_memory);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::unpack_region_requirement(
                                    RegionRequirement &req, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(req.region);
      derez.deserialize(req.partition);
      size_t num_privilege_fields;
      derez.deserialize(num_privilege_fields);
      for (unsigned idx = 0; idx < num_privilege_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        req.privilege_fields.insert(fid);
      }
      size_t num_instance_fields;
      derez.deserialize(num_instance_fields);
      for (unsigned idx = 0; idx < num_instance_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        req.instance_fields.push_back(fid);
      }
      derez.deserialize(req.privilege);
      derez.deserialize(req.prop);
      derez.deserialize(req.parent);
      derez.deserialize(req.redop);
      derez.deserialize(req.tag);
      derez.deserialize(req.flags);
      derez.deserialize(req.handle_type);
      derez.deserialize(req.projection);
      size_t num_current_instances;
      derez.deserialize(num_current_instances);
      for (unsigned idx = 0; idx < num_current_instances; idx++)
      {
        Memory mem;
        derez.deserialize(mem);
        bool full;
        derez.deserialize(full);
        req.current_instances[mem] = full;
      }
      derez.deserialize(req.max_blocking_factor);
      derez.deserialize(req.must_early_map);
      derez.deserialize(req.restricted);
      derez.deserialize(req.selected_memory);
      req.flags |= VERIFIED_FLAG;
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::pack_grant(const Grant &grant, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      grant.impl->pack_grant(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::unpack_grant(Grant &grant, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Create a new grant impl object to perform the unpack
      grant = Grant(new Grant::Impl());
      grant.impl->unpack_grant(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::pack_phase_barrier(
                                  const PhaseBarrier &barrier, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(barrier.phase_barrier);
    }  

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::unpack_phase_barrier(
                                    PhaseBarrier &barrier, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(barrier.phase_barrier);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::pack_point(Serializer &rez, const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(p.dim);
      if (p.dim == 0)
        rez.serialize(p.point_data[0]);
      else
      {
        for (int idx = 0; idx < p.dim; idx++)
          rez.serialize(p.point_data[idx]);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::unpack_point(Deserializer &derez, DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(p.dim);
      if (p.dim == 0)
        derez.deserialize(p.point_data[0]);
      else
      {
        for (int idx = 0; idx < p.dim; idx++)
          derez.deserialize(p.point_data[idx]);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::log_requirement(UniqueID uid, unsigned idx,
                                            const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      const bool reg = (req.handle_type == SINGULAR) ||
                 (req.handle_type == REG_PROJECTION);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_logical_requirement(Machine::get_executing_processor(),
                                           uid, idx, reg,
                                           reg ? req.region.index_space.id :
                                                 req.partition.index_partition,
                                           reg ? req.region.field_space.id :
                                                 req.partition.field_space.id,
                                           reg ? req.region.tree_id : 
                                                 req.partition.tree_id,
                                           req.privilege, req.prop, req.redop,
                                           req.privilege_fields);
#endif
#ifdef LEGION_SPY
      
      LegionSpy::log_logical_requirement(uid, idx, reg,
          reg ? req.region.index_space.id :
                req.partition.index_partition,
          reg ? req.region.field_space.id :
                req.partition.field_space.id,
          reg ? req.region.tree_id : 
                req.partition.tree_id,
          req.privilege, req.prop, req.redop);
      LegionSpy::log_requirement_fields(uid, idx, req.privilege_fields);
#endif
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(Runtime *rt)
      : TaskOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SingleTask::activate_single(void)
    //--------------------------------------------------------------------------
    {
      activate_task();
      num_virtual_mappings = 0;
      executing_processor = Processor::NO_PROC;
      profile_task = false;
      current_fence = NULL;
      fence_gen = 0;
      simultaneous_checked = false;
      has_simultaneous = false;
      context = RegionTreeContext();
      executed = false;
      valid_wait_event = false;
      deferred_map = Event::NO_EVENT;
      deferred_complete = Event::NO_EVENT; 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate_single(void)
    //--------------------------------------------------------------------------
    {
      deactivate_task();
      physical_instances.clear();
      local_instances.clear();
      physical_regions.clear();
      inline_regions.clear();
      virtual_mapped.clear();
      locally_mapped.clear();
      region_deleted.clear();
      index_deleted.clear();
      executing_children.clear();
      executed_children.clear();
      complete_children.clear();
      mapping_paths.clear();
      premapping_events.clear();
      for (std::set<Operation*>::const_iterator it = reclaim_children.begin();
            it != reclaim_children.end(); it++)
      {
        (*it)->deactivate();
      }
      reclaim_children.clear();
      for (std::map<TraceID,LegionTrace*>::const_iterator it = traces.begin();
            it != traces.end(); it++)
      {
        delete it->second;
      }
      traces.clear();
      // Clean up any locks and barriers that the user
      // asked us to destroy
      while (!context_locks.empty())
      {
        context_locks.back().destroy_reservation();
        context_locks.pop_back();
      }
      while (!context_barriers.empty())
      {
        context_barriers.back().destroy_barrier();
        context_barriers.pop_back();
      }
      local_fields.clear();
      while (!inline_tasks.empty())
      {
        inline_tasks.back()->deactivate();
        inline_tasks.pop_back();
      }
      if (valid_wait_event)
      {
        valid_wait_event = false;
        window_wait.trigger();
      }
      if (context.exists())
        runtime->free_context(this);
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_leaf(void) const
    //--------------------------------------------------------------------------
    {
      const TaskVariantCollection::Variant &var = 
        variants->get_variant(selected_variant);
      return var.leaf;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_inner(void) const
    //--------------------------------------------------------------------------
    {
      const TaskVariantCollection::Variant &var = 
        variants->get_variant(selected_variant);
      return var.inner;
    }

    //--------------------------------------------------------------------------
    void SingleTask::assign_context(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!context.exists());
#endif
      context = ctx;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext SingleTask::release_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      RegionTreeContext result = context;
      context = RegionTreeContext();
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext SingleTask::get_context(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      return context;
    }

    //--------------------------------------------------------------------------
    ContextID SingleTask::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      return context.get_id();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_user_lock(Reservation r)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no
      // need to hold the lock
      context_locks.push_back(r);
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_user_barrier(Barrier b)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no 
      // need to hold the lock
      context_barriers.push_back(b);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion SingleTask::get_physical_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx];
    } 

    //--------------------------------------------------------------------------
    void SingleTask::add_inline_task(InlineTask *inline_task)
    //--------------------------------------------------------------------------
    {
      inline_tasks.push_back(inline_task); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      // Remove this child from our context
      unregister_child_operation(child);
      // Check to see if the child is predicated
      // If it is wait for it to resolve
      if (child->is_predicated())
      {
        // See if the predicate speculates false, if so return false
        // and then we are done.
        if (!child->get_predicate_value(executing_processor))
          return;
      }

      // Get an available inline task
      InlineTask *inline_task = runtime->get_available_inline_task();
      inline_task->initialize_inline_task(this, child);
      // Record the inline task as one we need to deactivate
      // when we are done executing
      add_inline_task(inline_task);

      // Save the state of our physical regions
      std::vector<bool> phy_regions_mapped(physical_regions.size());
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
        phy_regions_mapped[idx] = is_region_mapped(idx);
 
      // Also save the original number of child regions
      unsigned orig_child_regions = inline_task->regions.size();

      // Find the inline function pointer for this task
      Processor::TaskFuncID low_id = 
        child->variants->get_variant(child->selected_variant).low_id;
      InlineFnptr fn = Runtime::find_inline_function(low_id);
      
      // Do the inlining
      child->perform_inlining(inline_task, fn);

      // Now when we pop back out, first see if the child made any new
      // regions and add them onto our copied regions
      size_t num_child_regions = inline_task->regions.size();
      if (num_child_regions > orig_child_regions)
      {
        for (unsigned idx = orig_child_regions; 
              idx < num_child_regions; idx++)
        {
          indexes.push_back(inline_task->indexes[idx]);
          regions.push_back(inline_task->regions[idx]);
          physical_regions.push_back(inline_task->get_physical_region(idx));
        }
      }
      // Restore any privilege information
      inline_task->return_privilege_state(this);
      // Now see if the mapping state of any of our
      // originally mapped regions has changed
      std::set<Event> wait_events;
      for (unsigned idx = 0; idx < phy_regions_mapped.size(); idx++)
      {
        if (phy_regions_mapped[idx] && !is_region_mapped(idx))
        {
          // Need to remap
          MapOp *op = runtime->get_available_map_op();
          op->initialize(this, physical_regions[idx]);
          wait_events.insert(op->get_completion_event());
          runtime->add_to_dependence_queue(executing_processor, op);
        }
        else if (!phy_regions_mapped[idx] && is_region_mapped(idx))
        {
          // Need to unmap
          physical_regions[idx].impl->unmap_region();
        }
        // Otherwise everything is still the same
      }
      if (!wait_events.empty())
      {
        Event wait_on = Event::merge_events(wait_events);
        if (!wait_on.has_triggered())
        {
          runtime->pre_wait(executing_processor);
          wait_on.wait();
          runtime->post_wait(executing_processor);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::restart_task(void)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out how to restart tasks
      assert(false);
    }

    //--------------------------------------------------------------------------
    UserEvent SingleTask::begin_premapping(RegionTreeID tid, 
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      UserEvent result = UserEvent::create_user_event();
      Event wait_on = Event::NO_EVENT;
      std::set<Event> wait_on_events;
      {
        AutoLock o_lock(op_lock);
        std::map<RegionTreeID,std::map<Event,FieldMask> >::iterator finder = 
          premapping_events.find(tid);
        if (finder == premapping_events.end())
        {
          // Couldn't find it, so add it
          premapping_events[tid][result] = mask;   
        }
        else
        {
          std::map<Event,FieldMask> &pre_events = finder->second;
          for (std::map<Event,FieldMask>::iterator it = 
                pre_events.begin(); it != pre_events.end(); it++)
          {
            // See if we have anything to wait on
            if (!(it->second * mask))
            {
              wait_on_events.insert(it->first);
              // Mark that this is no longer the most recent
              // pre-mapping operation on the overlapping fields
              it->second -= mask;
            }
          }
          // Put ourselves in the map
          pre_events[result] = mask;
        }
      }
      // Merge any events we need to wait on
      if (!wait_on_events.empty())
        wait_on = Event::merge_events(wait_on_events);
      // Since we shouldn't be holding any locks here, it should be safe
      // to wait withinout blocking the processor.
      if (wait_on.exists())
        wait_on.wait(true/*block*/);
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::end_premapping(RegionTreeID tid, UserEvent term_premap)
    //--------------------------------------------------------------------------
    {
      // Trigger our event
      term_premap.trigger();
      AutoLock o_lock(op_lock);
      // Remove it from the map
#ifdef DEBUG_HIGH_LEVEL
      assert(premapping_events.find(tid) != premapping_events.end());
#endif
      premapping_events[tid].erase(term_premap);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* SingleTask::get_instance(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < physical_instances.size());
      assert(physical_instances[idx].has_ref());
#endif
      return physical_instances[idx].get_handle().get_view()->get_manager(); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(virtual_mapped.size());
      for (unsigned idx = 0; idx < virtual_mapped.size(); idx++)
      {
        bool virt = virtual_mapped[idx];
        rez.serialize(virt);
      }
      rez.serialize(num_virtual_mappings);
      rez.serialize(executing_processor);
      rez.serialize(physical_instances.size());
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        physical_instances[idx].pack_reference(rez, target);
      }
      // Finally always send the region tree shapes for our regions
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        runtime->forest->send_tree_shape(indexes[idx], target);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->send_tree_shape(regions[idx], target);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_base_task(derez);
      size_t num_virtual;
      derez.deserialize(num_virtual);
      virtual_mapped.resize(num_virtual);
      for (unsigned idx = 0; idx < num_virtual; idx++)
      {
        bool virt;
        derez.deserialize(virt);
        virtual_mapped[idx] = virt;
      }
      derez.deserialize(num_virtual_mappings);
      derez.deserialize(executing_processor);
      size_t num_phy;
      derez.deserialize(num_phy);
      physical_instances.resize(num_phy);
      for (unsigned idx = 0; idx < num_phy; idx++)
      {
        physical_instances[idx] = 
          InstanceRef::unpack_reference(derez, runtime->forest, depth);
      }
      locally_mapped.resize(num_phy,false);
      // Initialize the mapping paths on this node
      initialize_mapping_paths(); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_mapping_paths(void)
    //--------------------------------------------------------------------------
    {
      mapping_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        initialize_mapping_path(mapping_paths[idx], regions[idx],
                                regions[idx].region);
      }
    } 

    //--------------------------------------------------------------------------
    void SingleTask::pack_parent_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Starting with this context, pack up all the enclosing local fields
      std::deque<LocalFieldInfo> locals = local_fields;
      // Get all the local fields from our enclosing contexts
      find_enclosing_local_fields(locals);
      RezCheck z(rez);
      // Now pack them all up
      size_t num_local = locals.size();
      rez.serialize(num_local);
      for (unsigned idx = 0; idx < locals.size(); idx++)
        rez.serialize(locals[idx]);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      rez.serialize(legion_spy_start);
      rez.serialize(get_task_completion());
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      Event wait_event = Event::NO_EVENT;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(executing_children.find(op) == executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
#endif
        // Put this in the list of child operations that need to map
        executing_children.insert(op);
        // Check to see if we have too many active children
        if (executing_children.size() >=
            Runtime::max_task_window_per_context)
        {
          // Check to see if we have an active wait, if not make
          // one and then wait on it
          if (!valid_wait_event)
          {
            window_wait = UserEvent::create_user_event();
            valid_wait_event = true;
          }
          wait_event = window_wait;
        }
      }
      // See if we need to preempt this task because it has exceeded
      // the maximum number of outstanding operations within its context
      if (valid_wait_event && !window_wait.has_triggered())
      {
#ifdef LEGION_LOGGING
        LegionLogging::log_timing_event(executing_processor,
                                        get_unique_task_id(), 
                                        BEGIN_WINDOW_WAIT);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(get_unique_task_id(), PROF_BEGIN_WAIT);
#endif
        runtime->pre_wait(executing_processor);
        wait_event.wait();
        runtime->post_wait(executing_processor);
#ifdef LEGION_LOGGING
        LegionLogging::log_timing_event(executing_processor,
                                        get_unique_task_id(), 
                                        END_WINDOW_WAIT);
#endif
#ifdef LEGION_PROF
        LegionProf::register_event(get_unique_task_id(),
                                   PROF_END_WAIT);
#endif
      }
      // Finally if we are performing a trace mark that the child has a trace
      if (current_trace != NULL)
        op->set_trace(current_trace);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::set<Operation*>::iterator finder = executing_children.find(op);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != executing_children.end());
      assert(executed_children.find(op) == executed_children.end());
      assert(complete_children.find(op) == complete_children.end());
#endif
      executing_children.erase(finder);
      // Now put it in the list of executing operations
      // Note this doesn't change the number of active children
      // so there's no need to trigger any window waits
      //
      // Add some hysteresis here so that we have some runway for when
      // the paused task resumes it can run for a little while.
      executed_children.insert(op);
      if (valid_wait_event && (executing_children.size() <
                                (3*Runtime::max_task_window_per_context) >> 2))
      {
        window_wait.trigger();
        valid_wait_event = false;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        std::set<Operation*>::iterator finder = executed_children.find(op);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
        assert(executing_children.find(op) == executing_children.end());
#endif
        executed_children.erase(finder);
        // Put it on the list of complete children to complete
        complete_children.insert(op);
        // See if we need to trigger the all children complete call
        if (executed && executing_children.empty() && 
            executed_children.empty() && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger)
      {
        // If we had any virtual mappings, we can now be considered mapped
        if (num_virtual_mappings > 0)
          complete_mapping();
        trigger_children_complete();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        std::set<Operation*>::iterator finder = complete_children.find(op);
#ifdef DEBUG_HIGH_LEVEL
        assert(finder != complete_children.end());
        assert(executing_children.find(op) == executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
#endif
        complete_children.erase(finder);
        // See if we need to trigger the all children commited call
        if (completed && executing_children.empty() && 
            executed_children.empty() && complete_children.empty() &&
            !children_commit_invoked)
        {
          needs_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SingleTask::unregister_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      // Remove it from everything and then see if we need to
      // trigger the window wait event
      executing_children.erase(op);
      executed_children.erase(op);
      complete_children.erase(op);
      if (valid_wait_event && (executing_children.size() < 
                                Runtime::max_task_window_per_context))
      {
        window_wait.trigger();
        valid_wait_event = false;
      }
      // No need to see if we trigger anything else because this
      // method is only called while the task is still executing
      // so 'executed' is still false.
#ifdef DEBUG_HIGH_LEVEL
      assert(!executed);
#endif
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_executing_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      return (executing_children.find(op) != executing_children.end());
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_executed_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      return (executed_children.find(op) != executed_children.end());
    }

    //--------------------------------------------------------------------------
    void SingleTask::print_children(void)
    //--------------------------------------------------------------------------
    {
      // Don't both taking the lock since this is for debugging
      // and isn't actually called anywhere
      for (std::set<Operation*>::const_iterator it =
            executing_children.begin(); it != executing_children.end(); it++)
      {
        Operation *op = *it;
        printf("Executing Child %p\n",op);
      }
      for (std::set<Operation*>::const_iterator it =
            executed_children.begin(); it != executed_children.end(); it++)
      {
        Operation *op = *it;
        printf("Executed Child %p\n",op);
      }
      for (std::set<Operation*>::const_iterator it =
            complete_children.begin(); it != complete_children.end(); it++)
      {
        Operation *op = *it;
        printf("Complete Child %p\n",op);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_reclaim_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      reclaim_children.insert(op);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (current_fence != NULL)
        op->register_dependence(current_fence, fence_gen);
    }

    //--------------------------------------------------------------------------
    void SingleTask::update_current_fence(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      if (current_fence != NULL)
        current_fence->remove_mapping_reference(fence_gen);
      current_fence = op;
      fence_gen = op->get_generation();
      current_fence->add_mapping_reference(fence_gen);
    }

    //--------------------------------------------------------------------------
    void SingleTask::begin_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
      {
        log_task(LEVEL_ERROR,"Illegal nested trace with ID %d attempted in "
                             "task %s (ID %lld)", tid, variants->name,
                             get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_NESTED_TRACE);
      }
      std::map<TraceID,LegionTrace*>::const_iterator finder = traces.find(tid);
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        current_trace = new LegionTrace(tid, this);
        traces[tid] = current_trace;
      }
      else
      {
        // Issue the mapping fence first
        runtime->issue_mapping_fence(this);
        // Now mark that we are starting a trace
        current_trace = finder->second;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      if (current_trace == NULL)
      {
        log_task(LEVEL_ERROR,"Unmatched end trace for ID %d in task %s "
                             "(ID %lld)", tid, variants->name,
                             get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNMATCHED_END_TRACE);
      }
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op();
        complete_op->initialize_complete(this);
#ifdef INORDER_EXECUTION
        Event term_event = complete_op->get_completion_event();
#endif
        runtime->add_to_dependence_queue(get_executing_processor(),complete_op);
#ifdef INORDER_EXECUTION
        if (Runtime::program_order_execution && !term_event.has_triggered())
        {
          Processor proc = get_executing_processor();
          runtime->pre_wait(proc);
          term_event.wait();
          runtime->post_wait(proc);
        }
#endif
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(); 
        capture_op->initialize_capture(this);
#ifdef INORDER_EXECUTION
        Event term_event = capture_op->get_completion_event();
#endif
        runtime->add_to_dependence_queue(get_executing_processor(), capture_op);
#ifdef INORDER_EXECUTION
        if (Runtime::program_order_execution && !term_event.has_triggered())
        {
          Processor proc = get_executing_processor();
          runtime->pre_wait(proc);
          term_event.wait();
          runtime->post_wait(proc);
        }
#endif
        // Mark that the current trace is now fixed
        current_trace->fix_trace();
      }
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_local_field(FieldSpace handle, FieldID fid, 
                                     size_t field_size)
    //--------------------------------------------------------------------------
    {
      allocate_local_field(local_fields.back());
      // Hold the lock when modifying the local_fields data structure
      // since it can be read by tasks that are being packed
      AutoLock o_lock(op_lock);
      local_fields.push_back(
                    LocalFieldInfo(handle, fid, field_size, completion_event)); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &field_sizes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.size() == field_sizes.size());
#endif
      for (unsigned idx = 0; idx < fields.size(); idx++)
        add_local_field(handle, fields[idx], field_sizes[idx]);
    }

    //--------------------------------------------------------------------------
    void SingleTask::allocate_local_field(const LocalFieldInfo &info)
    //--------------------------------------------------------------------------
    {
      // Try allocating a local field and if we succeeded then launch
      // a deferred task to reclaim the field whenever it's completion
      // event has triggered.  Otherwise it already exists on this node
      // so we are free to use it no matter what
      if (runtime->forest->allocate_field(info.handle, info.field_size,
                                          info.fid, true/*local*/))
      {
        // Successfully allocated a local field, launch a task to reclaim it
        Serializer rez;
        // Do this before the check since it gets pulled off first
        rez.serialize<HLRTaskID>(HLR_RECLAIM_LOCAL_FIELD_ID);
        {
          RezCheck z(rez);
          rez.serialize(info.handle);
          rez.serialize(info.fid);
        }
#ifdef SPECIALIZED_UTIL_PROCS
        Processor util = runtime->get_cleanup_proc(executing_processor);
#else
        Processor util = executing_processor.get_utility_processor();
#endif
        util.spawn(HLR_TASK_ID, rez.get_buffer(),
                   rez.get_used_bytes(), info.reclaim_event);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_created_index(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      indexes.push_back(IndexSpaceRequirement(handle,ALL_MEMORY,handle));
      index_deleted.push_back(false);
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_created_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      RegionRequirement new_req(handle, READ_WRITE, EXCLUSIVE, handle);
      runtime->forest->get_all_fields(handle.get_field_space(),
                                      new_req.privilege_fields);
      // Now make a new region requirement and physical region
      regions.push_back(new_req);
      // Make a new unmapped physical region
      physical_regions.push_back(PhysicalRegion(
            new PhysicalRegion::Impl(regions.back(), Event::NO_EVENT,
                 false/*mapped*/, this, map_id, tag, is_leaf(), runtime)));
      physical_instances.push_back(InstanceRef());
      local_instances.push_back(InstanceRef());
      // Mark that this region was virtually mapped so we don't
      // try to close it when we are done executing.
      virtual_mapped.push_back(true);
      locally_mapped.push_back(true);
      region_deleted.push_back(false);
      RemoteTask *outermost = find_outermost_physical_context();
      enclosing_physical_contexts.push_back(outermost->get_context());
      outermost->add_top_region(handle);
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_created_field(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      std::set<LogicalRegion> top_regions;
      runtime->forest->get_all_regions(handle, top_regions);
      for (std::set<LogicalRegion>::const_iterator it = top_regions.begin();
            it != top_regions.end(); it++)
      {
        RegionRequirement new_req(*it, READ_WRITE, EXCLUSIVE, *it);
        new_req.privilege_fields.insert(fid);
        regions.push_back(new_req);
        physical_regions.push_back(PhysicalRegion(
              new PhysicalRegion::Impl(regions.back(), Event::NO_EVENT,
                    false/*mapped*/, this, map_id, tag, is_leaf(), runtime)));
        physical_instances.push_back(InstanceRef());
        local_instances.push_back(InstanceRef());
        // Mark that the region was virtually mapped
        virtual_mapped.push_back(true);
        locally_mapped.push_back(true);
        region_deleted.push_back(false);
        RemoteTask* outermost = find_outermost_physical_context();
        enclosing_physical_contexts.push_back(outermost->get_context());
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::remove_created_index(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        if (indexes[idx].handle == handle)
        {
          index_deleted[idx] = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::remove_created_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region == handle)
        {
          region_deleted[idx] = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::remove_created_field(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region.get_field_space() == handle)
        {
          regions[idx].privilege_fields.erase(fid);
          if (regions[idx].privilege_fields.empty())
          {
            region_deleted[idx] = true;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::get_top_regions(std::vector<LogicalRegion> &top_regions)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock when getting the top regions because
      // the add_created_region method can be executing in parallel
      // and may result in the vector changing size
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      top_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif
        top_regions[idx] = regions[idx].region;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_index_space(IndexSpace handle,
                                                 Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_index_space(context, handle, op,
                                                     top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_index_partition(IndexPartition handle,
                                                     Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_index_partition(context, handle, op,
                                                         top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_field_space(FieldSpace handle,
                                                 Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_field_space(context, handle, op,
                                                     top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_fields(FieldSpace handle, Operation *op,
                                            const std::set<FieldID> &to_delete)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_fields(context, handle, to_delete,
                                                op, top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_logical_region(LogicalRegion handle,
                                                    Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_logical_region(context, handle, op,
                                                        top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_logical_partition(LogicalPartition handle,
                                                       Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
#endif
      std::vector<LogicalRegion> top_regions;
      get_top_regions(top_regions);
      for (unsigned idx = 0; idx < top_regions.size(); idx++)
      {
        runtime->forest->analyze_destroy_logical_partition(context, handle, op,
                                                           top_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    int SingleTask::has_conflicting_regions(MapOp *op, bool &parent_conflict,
                                            bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      parent_conflict = false;
      inline_conflict = false;
      // Need to hold our local lock when reading regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
#endif
      for (unsigned our_idx = 0; our_idx < regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = regions[our_idx];
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = op->requirement;
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
        {
          parent_conflict = true;
          return our_idx;
        }
      }
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = op->requirement;
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
        {
          inline_conflict = true;
          // No index for inline conflicts
          return -1;
        }
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(TaskOp *task,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      // Need to hold our local lock when reading regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
#endif
      for (unsigned our_idx = 0; our_idx < regions.size(); our_idx++)
      {
        // Skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = regions[our_idx];
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        // Check to see if any region requirements from the child have
        // a dependence on our region at location our_idx
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          const RegionRequirement &req = task->regions[idx];  
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          {
            conflicting.push_back(physical_regions[our_idx]);
            // Once we find a conflict, we don't need to check
            // against it anymore, so go onto our next region
            break;
          }
        }
      }
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        // Check to see if any region requirements from the child have
        // a dependence on our region at location our_idx
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          const RegionRequirement &req = task->regions[idx];  
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          {
            conflicting.push_back(*it);
            // Once we find a conflict, we don't need to check
            // against it anymore, so go onto our next region
            break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(CopyOp *copy,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      // Need to hold our local lock when reading regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
#endif
      for (unsigned our_idx = 0; our_idx < regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = regions[our_idx];
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        bool has_conflict = false;
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->src_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->src_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->dst_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->dst_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        if (has_conflict)
          conflicting.push_back(physical_regions[our_idx]);
      }
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        bool has_conflict = false;
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->src_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->src_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->dst_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->dst_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        if (has_conflict)
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(AcquireOp *acquire,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      // Need to hold our local lock when reading regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
#endif
      for (unsigned our_idx = 0; our_idx < regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = regions[our_idx];
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = acquire->get_requirement();
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(physical_regions[our_idx]);
      }
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = acquire->get_requirement();
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(ReleaseOp *release,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      // Need to hold our local lock when reading regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
#endif
      for (unsigned our_idx = 0; our_idx < regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = regions[our_idx];
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = release->get_requirement();
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(physical_regions[our_idx]);
      }
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->impl->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_HIGH_LEVEL
        // This better be true for a single task
        assert(our_req.handle_type == SINGULAR);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        const RegionRequirement &req = release->get_requirement();
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::check_region_dependence(RegionTreeID our_tid,
                                             IndexSpace our_space,
                                             const RegionRequirement &our_req,
                                             const RegionUsage &our_usage,
                                             const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      if ((req.handle_type == SINGULAR) || 
          (req.handle_type == REG_PROJECTION))
      {
        // If the trees are different we're done 
        if (our_tid != req.region.get_tree_id())
          return false;
        // Check to see if there is a path between
        // the index spaces
        if (runtime->forest->are_disjoint(our_space,
                                          req.region.get_index_space()))
          return false;
      }
      else
      {
        // Check if the trees are different
        if (our_tid != req.partition.get_tree_id())
          return false;
        if (runtime->forest->are_disjoint(our_space,
                  req.partition.get_index_partition()))
          return false;
      }
      // Check to see if any privilege fields overlap
      std::vector<FieldID> intersection(our_req.privilege_fields.size());
      std::vector<FieldID>::iterator intersect_it = 
        std::set_intersection(our_req.privilege_fields.begin(),
                              our_req.privilege_fields.end(),
                              req.privilege_fields.begin(),
                              req.privilege_fields.end(),
                              intersection.begin());
      intersection.resize(intersect_it - intersection.begin());
      if (intersection.empty())
        return false;
      // Finally if everything has overlapped, do a dependence analysis
      // on the privileges and coherence
      RegionUsage usage(req);
      switch (check_dependence_type(our_usage,usage))
      {
        // Only allow no-dependence, or simultaneous dependence through
        case NO_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            return false;
          }
        default:
          break;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_inline_mapped_region(PhysicalRegion &region)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      // Because of 'remap_region', this method can be called
      // both for inline regions as well as regions which were
      // initally mapped for the task.  Do a quick check to see
      // if it was an original region.  If it was then we're done.
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
      {
        if (physical_regions[idx].impl == region.impl)
          return;
      }
      inline_regions.push_back(region);
    }

    //--------------------------------------------------------------------------
    void SingleTask::unregister_inline_mapped_region(PhysicalRegion &region)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::list<PhysicalRegion>::iterator it = inline_regions.begin();
            it != inline_regions.end(); it++)
      {
        if (it->impl == region.impl)
        {
          inline_regions.erase(it);
          return;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_region_mapped(unsigned idx)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx].impl->is_mapped();
    }

    //--------------------------------------------------------------------------
    unsigned SingleTask::find_parent_region(unsigned index, TaskOp *child)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == region_deleted.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if ((regions[idx].region == child->regions[index].parent) &&
            !region_deleted[idx])
          return idx;
      }
      log_region(LEVEL_ERROR,"Parent task %s (ID %lld) of inline task %s "
                              "(ID %lld) does not have a region "
                              "requirement for region (" IDFMT ",%x,%x) "
                              "as a parent of child task's region "
                              "requirement index %d",
                              variants->name, 
                              get_unique_task_id(),
                              child->variants->name, 
                              child->get_unique_task_id(), 
                              child->regions[index].region.index_space.id,
                              child->regions[index].region.field_space.id, 
                              child->regions[index].region.tree_id, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_BAD_PARENT_REGION);
      return 0;
    }

    //--------------------------------------------------------------------------
    unsigned SingleTask::find_parent_index_region(unsigned index, TaskOp *child)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(indexes.size() == index_deleted.size());
#endif
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        if ((indexes[idx].handle == child->indexes[idx].parent) && 
              !index_deleted[idx])
          return idx;
      }
      log_index(LEVEL_ERROR,"Parent task %s (ID %lld) of inline task %s "
                            "(ID %lld) does not have an index space "
                            "requirement for index space " IDFMT " "
                            "as a parent of chlid task's index requirement "
                            "index %d",
                            variants->name,
                            get_unique_task_id(),
                            child->variants->name,
                            child->get_unique_task_id(),
                            child->indexes[index].handle.id, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_BAD_PARENT_INDEX);
      return 0;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(
                                        const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      std::vector<IndexSpaceRequirement> copy_indexes;
      {
        // Copy the indexes so we don't have to hold
        // the lock when doing this which could result
        // in double acquire of locks
        AutoLock o_lock(op_lock,1,false/*exclusive*/);
        copy_indexes = indexes;
      }
      
      // Find the parent index space
      for (std::vector<IndexSpaceRequirement>::const_iterator it = 
            copy_indexes.begin(); it != copy_indexes.end(); it++)
      {
        // Check to see if we found the requirement in the parent 
        if (it->handle == req.parent)
        {
          // Check that there is a path between the parent and the child
          std::vector<Color> path;
          if (!runtime->forest->compute_index_path(req.parent, 
                                                   req.handle, path))
            return ERROR_BAD_INDEX_PATH;
          // Now check that the privileges are less than or equal
          if (req.privilege & (~(it->privilege)))
          {
            return ERROR_BAD_INDEX_PRIVILEGES;  
          }
          return NO_ERROR;
        }
      }
      // If we didn't find it here, we have to check the added 
      // index spaces that we have
      if (has_created_index_space(req.parent))
      {
        // Still need to check that there is a path between the two
        std::vector<Color> path;
        if (!runtime->forest->compute_index_path(req.parent, req.handle, path))
          return ERROR_BAD_INDEX_PATH;
        // No need to check privileges here since it is a created space
        // which means that the parent has all privileges.
        return NO_ERROR;
      }
      return ERROR_BAD_PARENT_INDEX;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const RegionRequirement &req,
                                                FieldID &bad_field,
                                                bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
      if (req.flags & VERIFIED_FLAG)
        return NO_ERROR;
      std::vector<RegionRequirement> copy_regions;
      {
        // Make a copy of the regions so we don't have to
        // hold the lock when doing this which could result
        // in a double acquire of locks
        AutoLock o_lock(op_lock,1,false/*exclusive*/);
        copy_regions = regions;
      }
      std::set<FieldID> checking_fields = req.privilege_fields;
      for (std::vector<RegionRequirement>::const_iterator it = 
            copy_regions.begin(); it != copy_regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->handle_type == SINGULAR); // better be singular
#endif
        // Check to see if we found the requirement in the parent
        if (it->region == req.parent)
        {
          if ((req.handle_type == SINGULAR) || 
              (req.handle_type == REG_PROJECTION))
          {
            std::vector<Color> path;
            if (!runtime->forest->compute_index_path(req.parent.index_space,
                                              req.region.index_space, path))
              return ERROR_BAD_REGION_PATH;
          }
          else
          {
            std::vector<Color> path;
            if (!runtime->forest->compute_partition_path(req.parent.index_space,
                                          req.partition.index_partition, path))
              return ERROR_BAD_PARTITION_PATH;
          }
          // Now check that the types are subset of the fields
          // Note we can use the parent since all the regions/partitions
          // in the same region tree have the same field space
          bool has_fields = false;
          {
            std::vector<FieldID> to_delete;
            for (std::set<FieldID>::const_iterator fit = 
                  checking_fields.begin(); fit != checking_fields.end(); fit++)
            {
              if (it->privilege_fields.find(*fit) != it->privilege_fields.end())
              {
                to_delete.push_back(*fit);
                has_fields = true;
              }
              else if (has_created_field(req.parent.field_space, *fit))
              {
                to_delete.push_back(*fit);
              }
            }
            for (std::vector<FieldID>::const_iterator fit = to_delete.begin();
                  fit != to_delete.end(); fit++)
            {
              checking_fields.erase(*fit);
            }
          }
          // Only need to do this check if there were overlapping fields
          if (!skip_privilege && has_fields && 
              (req.privilege & (~(it->privilege))))
          {
            // Handle the special case where the parent has WRITE_DISCARD
            // privilege and the sub-task wants any other kind of privilege.  
            // This case is ok because the parent could write something
            // and then hand it off to the child.
            if (it->privilege != WRITE_DISCARD)
            {
              if ((req.handle_type == SINGULAR) || 
                  (req.handle_type == REG_PROJECTION))
                return ERROR_BAD_REGION_PRIVILEGES;
              else
                return ERROR_BAD_PARTITION_PRIVILEGES;
            }
          }
          // If we've seen all our fields, then we're done
          if (checking_fields.empty())
            return NO_ERROR;
        }
      }
      // Also check to see if it was a created region
      if (has_created_region(req.parent))
      {
        // Check that there is a path between the parent and the child
        if ((req.handle_type == SINGULAR) || 
            (req.handle_type == REG_PROJECTION))
        {
          std::vector<Color> path;
          if (!runtime->forest->compute_index_path(req.parent.index_space,
                                              req.region.index_space, path))
            return ERROR_BAD_REGION_PATH;
        }
        else
        {
          std::vector<Color> path;
          if (!runtime->forest->compute_partition_path(req.parent.index_space,
                                        req.partition.index_partition, path))
            return ERROR_BAD_PARTITION_PATH;
        }
        // No need to check the field privileges since we should have them all
        checking_fields.clear();
        // No need to check the privileges since we know we have them all
        return NO_ERROR;
      }
      if (!checking_fields.empty() && 
          (checking_fields.size() < req.privilege_fields.size()))
      {
        bad_field = *(checking_fields.begin());
        return ERROR_BAD_REGION_TYPE;
      }
      return ERROR_BAD_PARENT_REGION;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_simultaneous_coherence(void)
    //--------------------------------------------------------------------------
    {
      // If we already did the check, then return the value
      if (simultaneous_checked)
        return has_simultaneous;
      // Otherwise do the check and cache the value
      // Need the lock when reading the regions
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // If we have simultaneous coherence and there is an
        // actual physical region then there is something
        // that needs checking for child ops.
        if ((regions[idx].prop == SIMULTANEOUS) &&
            (physical_regions[idx].is_mapped()))
        {
          has_simultaneous = true;
          break;
        }
      }
      simultaneous_checked = true;
      return has_simultaneous;
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_simultaneous_restricted(
                                    RegionRequirement &child_requirement) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_simultaneous);
#endif
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].prop != SIMULTANEOUS)
          continue;
        // Also check to see if we have a physical region
        // If not then we don't need to bother with coherence
        if (!physical_regions[idx].is_mapped())
          continue;
        // Check to see if the child parent region matches this region
        if (child_requirement.parent == regions[idx].region)
        {
          // If it does, see if there are any fields which overlap
          std::vector<FieldID> intersection(
                              regions[idx].privilege_fields.size());
          std::vector<FieldID>::iterator intersect_it = 
            std::set_intersection(
                              regions[idx].privilege_fields.begin(),
                              regions[idx].privilege_fields.end(),
                              child_requirement.privilege_fields.begin(),
                              child_requirement.privilege_fields.end(),
                              intersection.begin());
          intersection.resize(intersect_it - intersection.begin());
          // If we had overlapping fields then mark that the
          // requirement is now restricted.  We might find later
          // during dependence analysis that user-level software
          // coherence changes this, but for now it is true.
          if (!intersection.empty())
          {
            child_requirement.restricted = true;
            // At this point we're done
            return;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_simultaneous_restricted(
                      std::vector<RegionRequirement> &child_requirements) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_simultaneous);
#endif
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].prop != SIMULTANEOUS)
          continue;
        if (!physical_regions[idx].is_mapped())
          continue;
        std::vector<FieldID> intersection(regions[idx].privilege_fields.size());
        for (unsigned child_idx = 0; 
              child_idx < child_requirements.size(); child_idx++)
        {
          // If the child has already been marked restricted
          // then we don't need to do the test again
          if (child_requirements[child_idx].restricted)
            continue;
          if (child_requirements[child_idx].parent == regions[idx].region)
          {
            std::vector<FieldID>::iterator intersect_it = 
              std::set_intersection(
                    regions[idx].privilege_fields.begin(),
                    regions[idx].privilege_fields.end(),
                    child_requirements[child_idx].privilege_fields.begin(),
                    child_requirements[child_idx].privilege_fields.end(),
                    intersection.begin());
            intersection.resize(intersect_it - intersection.begin());
            if (!intersection.empty())
              child_requirements[child_idx].restricted = true;
            // Reset the intersection vector
            intersection.resize(regions[idx].privilege_fields.size());
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_created_region(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      return (created_regions.find(handle) != created_regions.end());
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_created_field(FieldSpace handle, FieldID fid) const
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      if (created_fields.find(std::pair<FieldSpace,FieldID>(handle,fid))
              != created_fields.end())
        return true;
      // Otherwise, check our locally created fields to see if we have
      // privileges from there
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
      {
        if ((local_fields[idx].handle == handle) && 
            (local_fields[idx].fid == fid))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_index_subspace(IndexSpace handle, const char *caller)
    //--------------------------------------------------------------------------
    {
      // This is always called inline so no need to take the lock
      std::vector<Color> path;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        path.clear();
        if (runtime->forest->compute_index_path(
              regions[idx].region.get_index_space(), handle, path))
          return;
      }
      // Finally check the index space requirements
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        path.clear();
        if (runtime->forest->compute_index_path(indexes[idx].handle, 
                                                handle, path))
          return;
      }
      // Also check the created regions
      for (std::set<LogicalRegion>::const_iterator it = 
            created_regions.begin(); it != created_regions.end(); it++)
      {
        path.clear();
        if (runtime->forest->compute_index_path(it->get_index_space(),
                                                handle, path))
          return;
      }
      // Finally check the created index spaces
      for (std::set<IndexSpace>::const_iterator it = 
            created_index_spaces.begin(); it != 
            created_index_spaces.end(); it++)
      {
        path.clear();
        if (runtime->forest->compute_index_path(*it, handle, path))
          return;
      }
#if 0
      log_task(LEVEL_ERROR,"Invalid call of %s with index space " IDFMT 
                           "which is not a sub-space of any requested or "
                           "created index spaces in task %s (ID %lld).",
                           caller, handle.id, variants->name,
                           get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_INVALID_INDEX_SUBSPACE_REQUEST);
#else
      log_task(LEVEL_WARNING,"Invalid call of %s with index space " IDFMT 
                           "which is not a sub-space of any requested or "
                           "created index spaces in task %s (ID %lld). "
                           "This must be fixed to guarantee correct "
                           "execution in multi-node runs.",
                           caller, handle.id, variants->name,
                           get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_index_subpartition(IndexPartition handle,
                                              const char *caller)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> path;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        path.clear();
        if (runtime->forest->compute_partition_path(
              regions[idx].region.get_index_space(), handle, path))
          return;
      }
      // Finally check the index space requirements
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        path.clear();
        if (runtime->forest->compute_partition_path(
              indexes[idx].handle, handle, path))
          return;
      }
      // Also check the created regions
      for (std::set<LogicalRegion>::const_iterator it = 
            created_regions.begin(); it != created_regions.end(); it++)
      {
        path.clear();
        if (runtime->forest->compute_partition_path(
              it->get_index_space(), handle, path))
          return;
      }
      // Finally check the created index spaces
      for (std::set<IndexSpace>::const_iterator it = 
            created_index_spaces.begin(); it != 
            created_index_spaces.end(); it++)
      {
        path.clear();
        if (runtime->forest->compute_partition_path(*it, handle, path))
          return;
      }
#if 0
      log_task(LEVEL_ERROR,"Invalid call of %s with index partition %d"
                           "which is not a sub-partition of any requested "
                           "or created index spaces in task %s (ID %lld).",
                           caller, handle, variants->name,
                           get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_INVALID_INDEX_SUBPARTITION_REQUEST);
#else
      log_task(LEVEL_WARNING,"Invalid call of %s with index partition %d"
                           "which is not a sub-partition of any requested "
                           "or created index spaces in task %s (ID %lld). "
                           "This must be fixed to guarantee correct "
                           "execution in multi-node runs.",
                           caller, handle, variants->name,
                           get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_field_space(FieldSpace handle, const char *caller)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region.get_field_space() == handle)
          return;
      }
      for (std::set<FieldSpace>::const_iterator it = 
            created_field_spaces.begin(); it != 
            created_field_spaces.end(); it++)
      {
        if ((*it) == handle)
          return;
      }
#if 0
      log_task(LEVEL_ERROR,"Invalid call of %s with field space %d which is "
                           "not a field space of any requested regions and "
                           "was not created in the context of task %s "
                           "(ID %lld).", caller, handle.id, variants->name,
                           get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_INVALID_FIELD_SPACE_REQUEST);
#else
      log_task(LEVEL_WARNING,"Invalid call of %s with field space %d which is "
                           "not a field space of any requested regions and "
                           "was not created in the context of task %s "
                           "(ID %lld). This must be fixed to guarantee correct "
                           "execution in multi-node runs", caller, handle.id, 
                           variants->name, get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_logical_subregion(LogicalRegion handle, 
                                             const char *caller)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> path;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region.get_tree_id() != handle.get_tree_id())
          continue;
        path.clear();
        if (runtime->forest->compute_index_path(
              regions[idx].region.get_index_space(),
              handle.get_index_space(), path))
          return;
      }
      for (std::set<LogicalRegion>::const_iterator it = 
            created_regions.begin(); it != created_regions.end(); it++)
      {
        if (it->get_tree_id() == handle.get_tree_id())
          return;
      }
#if 0
      log_task(LEVEL_ERROR,"Invalid call of %s with logical region ("
                            IDFMT ",%d,%d) which is not a sub-region of any "
                            "requested or created regions in task %s "
                            "(ID %lld).", caller, handle.get_index_space().id,
                            handle.get_field_space().id, handle.get_tree_id(),
                            variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_INVALID_LOGICAL_SUBREGION_REQUEST);
#else
      log_task(LEVEL_WARNING,"Invalid call of %s with logical region ("
                            IDFMT ",%d,%d) which is not a sub-region of any "
                            "requested or created regions in task %s "
                            "(ID %lld). This must be fixed to guarantee "
                            "correct execution in multi-node runs.", 
                            caller, handle.get_index_space().id,
                            handle.get_field_space().id, handle.get_tree_id(),
                            variants->name, get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_logical_subpartition(LogicalPartition handle,
                                                const char *caller)
    //--------------------------------------------------------------------------
    {
      std::vector<Color> path;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region.get_tree_id() != handle.get_tree_id())
          continue;
        path.clear();
        if (runtime->forest->compute_partition_path(
              regions[idx].region.get_index_space(),
              handle.get_index_partition(), path))
          return;
      }
      for (std::set<LogicalRegion>::const_iterator it = 
            created_regions.begin(); it != created_regions.end(); it++)
      {
        if (it->get_tree_id() == handle.get_tree_id())
          return;
      }
#if 0
      log_task(LEVEL_ERROR,"Invalid call of %s with logical partition "
                           "(%d,%d,%d) which is not a sub-partition of any "
                           "requested or created regions in task %s (ID %lld).",
                           caller, handle.get_index_partition(),
                           handle.get_field_space().id, handle.get_tree_id(),
                           variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_INVALID_LOGICAL_SUBPARTITION_REQUEST);
#else
      log_task(LEVEL_WARNING,"Invalid call of %s with logical partition "
                           "(%d,%d,%d) which is not a sub-partition of any "
                           "requested or created regions in task %s (ID %lld). "
                           "This must be fixed to guarantee correct execution "
                           " in multi-node runs.",
                           caller, handle.get_index_partition(),
                           handle.get_field_space().id, handle.get_tree_id(),
                           variants->name, get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    bool SingleTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      UniqueID local_id = get_unique_task_id();
      LegionProf::register_event(local_id, PROF_BEGIN_TRIGGER);
#endif
      bool success = true;
      if (is_remote())
      {
        if (distribute_task())
        {
          // Still local
          if (is_locally_mapped())
          {
            // Remote and locally mapped means
            // we were already mapped so we can
            // just launch the task
            launch_task();
          }
          else
          {
            // Remote but still need to map
            if (perform_mapping())
            {
              launch_task();
            }
            else // failed to map
              success = false;
          }
        }
        // otherwise it was sent away
      }
      else
      {
        // Not remote
        if (is_premapped() || premap_task())
        {
          // See if we have a must epoch in which case
          // we can simply record ourselves and we are done
          if (must_epoch != NULL)
            must_epoch->register_single_task(this, must_epoch_index);
          else
          {
            // See if this task is going to be sent
            // remotely in which case we need to do the
            // mapping now, otherwise we can defer it
            // until the task ends up on the target processor
            if (is_locally_mapped() && !runtime->is_local(target_proc))
            {
              if (perform_mapping())
              {
#ifdef DEBUG_HIGH_LEVEL
                bool still_local = 
#endif
                distribute_task();
#ifdef DEBUG_HIGH_LEVEL
                assert(!still_local);
#endif
              }
              else // failed to map
                success = false; 
            }
            else
            {
              if (distribute_task())
              {
                // Still local so try mapping and launching
                if (perform_mapping())
                {
                  // Still local and mapped so
                  // we can now launch it
                  launch_task();
                }
                else // failed to map
                  success = false;
              }
            }
          }
        }
        else // failed to premap
          success = false;
      }
#ifdef LEGION_PROF
      LegionProf::register_event(local_id, PROF_END_TRIGGER);
#endif
      return success;
    } 

    //--------------------------------------------------------------------------
    void SingleTask::unmap_all_regions(void)
    //--------------------------------------------------------------------------
    {
      virtual_mapped.clear();
      region_deleted.clear();
      num_virtual_mappings = 0;
      physical_instances.clear();
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(Processor target, Event user_event,
                                     bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(mapping_paths.size() == regions.size());
      assert(enclosing_physical_contexts.size() == regions.size());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      BEGIN_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_BEGIN_MAP_ANALYSIS);
#endif
      bool map_success = true; 
      // Initialize all the region information
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].mapping_failed = false;
        regions[idx].selected_memory = Memory::NO_MEMORY;
      }
      bool notify = false;
      if (!mapper_invoked)
        notify = runtime->invoke_mapper_map_task(current_proc, this);
      // Info for virtual mappings
      virtual_mapped.resize(regions.size(),false);
      locally_mapped.resize(regions.size(),true);
      num_virtual_mappings = 0;
      // Info for actual mappings
      std::vector<MappingRef> mapping_refs(regions.size());
      physical_instances.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // First check to see if the region was premapped
        // If it was then just continue onto the next
        // since we have nothing to do
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
        {
          virtual_mapped[idx] = false;
          continue;
        }
        // Now see if the mapper asked for a virtual mapping
        // or there are no fields in which case we also give
        // them a virtual mapping
        if (regions[idx].virtual_map || regions[idx].privilege_fields.empty())
        {
          virtual_mapped[idx] = true;
          num_virtual_mappings++;
          continue;
        }
        // Otherwise we're going to do an actual mapping
        mapping_refs[idx] = runtime->forest->map_physical_region(
                                    enclosing_physical_contexts[idx],
                                                  mapping_paths[idx],
                                                  regions[idx],
                                                  idx,
                                                  this,
                                                  current_proc,
                                                  target
#ifdef DEBUG_HIGH_LEVEL
                                                  , get_logging_name()
                                                  , unique_op_id
#endif
                                                  );
        if (mapping_refs[idx].has_ref())
        {
          virtual_mapped[idx] = false;
        }
        else
        {
          // Otherwise the mapping failed so break out
          map_success = false;
          regions[idx].mapping_failed = true;
          break;
        }
      }

      if (!map_success)
      {
        // Clean up our mess
        virtual_mapped.clear();
        num_virtual_mappings = 0;
        // Finally notify the mapper about the failed mapping
        runtime->invoke_mapper_failed_mapping(current_proc, this);
      }
      else 
      {
        // Mapping succeeded, so now we can fill in our 
        // list of InstanceRefs for this task
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // See if this is a virtual mapping
          if (virtual_mapped[idx])
          {
            physical_instances[idx] = InstanceRef();
            continue;
          }
          // Now check to see if the regions was premapped
          InstanceRef premapped = find_premapped_region(idx);
          if (premapped.has_ref())
          {
#ifdef DEBUG_HIGH_LEVEL
            // Check to make sure the pre-mapped region is visible to
            // the target processor
            Machine *machine = Machine::get_machine();
            const std::set<Memory> &visible_memories = 
                                  machine->get_visible_memories(target);
            Memory premap_memory = premapped.get_memory();
            if (visible_memories.find(premap_memory) != visible_memories.end())
            {
              log_region(LEVEL_ERROR,"Illegal premapped region for logical "
                                      "region (" IDFMT ",%d,%d) index %d of "
                                      "task %s (UID %lld)!  Memory " IDFMT 
                                      " is not visible from processor " IDFMT 
                                      "!", 
                                       regions[idx].region.index_space.id, 
                                       regions[idx].region.field_space.id, 
                                       regions[idx].region.tree_id, idx, 
                                       this->variants->name, 
                                       this->get_unique_task_id(), 
                                       premap_memory.id, target.id);
              assert(false);
              exit(ERROR_INVALID_PREMAPPED_REGION_LOCATION);
            }
#endif
            physical_instances[idx] = premapped;
            continue;
          }
          // Finally, finish setting up the actual instance
#ifdef DEBUG_HIGH_LEVEL
          assert(mapping_refs[idx].has_ref());
#endif
          physical_instances[idx] = 
                  runtime->forest->register_physical_region(
                                           enclosing_physical_contexts[idx],
                                                          mapping_refs[idx],
                                                          regions[idx],
                                                          idx,
                                                          this,
                                                          current_proc,
                                                          user_event
#ifdef DEBUG_HIGH_LEVEL
                                                          , get_logging_name()
                                                          , unique_op_id
                                                          , mapping_paths[idx]
#endif
                                                          );
          if (notify)
            regions[idx].selected_memory = physical_instances[idx].get_memory();
#ifdef DEBUG_HIGH_LEVEL
          // All these better succeed since we already made the instances
          assert(physical_instances[idx].has_ref());
#endif
        }
        executing_processor = target;
        if (notify)
          runtime->invoke_mapper_notify_result(current_proc, this);
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_END_MAP_ANALYSIS);
#endif
      return map_success;
    }  

    //--------------------------------------------------------------------------
    void SingleTask::initialize_region_tree_contexts(
                      const std::vector<RegionRequirement> &clone_requirements,
                      const std::vector<UserEvent> &unmap_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
#endif
      // Initialize all of the logical contexts no matter what
      //
      // For all of the physical contexts that were mapped, initialize them
      // with a specified reference to the current instance, otherwise
      // they were a virtual reference and we can ignore it.
      local_instances.resize(regions.size(), InstanceRef());
      std::map<PhysicalManager*,LogicalView*> top_views;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        // this better be true for single tasks
        assert(regions[idx].handle_type == SINGULAR);
#endif
        runtime->forest->initialize_logical_context(context,
                                                    regions[idx].region); 
        // Only need to initialize the context if this is
        // not a leaf and it wasn't virtual mapped
        if (!virtual_mapped[idx])
        {
          local_instances[idx] = 
            runtime->forest->initialize_physical_context(context,
                clone_requirements[idx], 
                physical_instances[idx].get_handle().get_manager(),
                unmap_events[idx], 
                executing_processor, depth+1, top_views);
#ifdef DEBUG_HIGH_LEVEL
          assert(local_instances[idx].has_ref());
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Always do the logical state
        runtime->forest->invalidate_logical_context(context,
                                                    regions[idx].region);
        // Only do the physical state if we
        // actually mapped a region
        if (!virtual_mapped[idx])
          runtime->forest->invalidate_physical_context(context,
                                                       regions[idx].region);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(physical_regions.empty());
#endif 

      const TaskVariantCollection::Variant &chosen_variant =  
                                variants->get_variant(selected_variant);
      
      // STEP 1: Compute the precondition for the task launch
      std::set<Event> wait_on_events;
      // Get the set of locks that we need and sort them
      // by the order in which they get sorted by STL set
      // which should guarantee no deadlocks since all tasks
      // will take the locks in the same order.  We put all
      // the locks into the required locks.  We also track which
      // locks are reservation locks and which are future
      // locks since they have to be taken differently.
      std::map<Reservation,bool/*exlusive*/> atomic_locks;
      // If we're debugging do one last check to make sure
      // that all the memories are visible on this processor
#ifdef DEBUG_HIGH_LEVEL
      const std::set<Memory> &visible_memories = 
        runtime->machine->get_visible_memories(executing_processor);
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!virtual_mapped[idx])
        {
          // Get the event to wait on unless we are doing the inner
          // task optimization
          if (!chosen_variant.inner)
            wait_on_events.insert(physical_instances[idx].get_ready_event());
          // See if we need a lock for this region
          if (physical_instances[idx].has_required_lock())
          {
            Reservation req_lock = physical_instances[idx].get_required_lock();
            // Check to see if it is needed exclusively or not
            bool exclusive = !IS_READ_ONLY(regions[idx]);
            std::map<Reservation,bool>::iterator finder = 
              atomic_locks.find(req_lock);
            if (finder == atomic_locks.end())
              atomic_locks[req_lock] = exclusive;
            else
              finder->second = finder->second || exclusive;
          }
#ifdef DEBUG_HIGH_LEVEL
          // We can only do this check if we actually have the reference
          // which we might not if this is locally mapped.  We also don't
          // need to do this if the user has promised us that they will
          // never actually access the physical instance with an accessor.
          if (physical_instances[idx].get_handle().has_view() &&
              !(regions[idx].flags & NO_ACCESS_FLAG))
          {
            Memory inst_mem = physical_instances[idx].get_memory();
            assert(visible_memories.find(inst_mem) != visible_memories.end());
          }
#endif
        }
      }
      // Now add get all the other preconditions for the launch
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        Future::Impl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->get_ready_event());
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        Grant::Impl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      Event start_condition = Event::merge_events(wait_on_events);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      if (!start_condition.exists())
      {
        UserEvent new_start = UserEvent::create_user_event();
        new_start.trigger();
        start_condition = new_start;
      }
#endif
      // Record the dependences
#ifdef LEGION_LOGGING
      LegionLogging::log_event_dependences(
          Machine::get_executing_processor(), wait_on_events, start_condition);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_event_dependences(wait_on_events, start_condition);
#endif
      // Take all the locks in order in the proper way
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          Event next = Event::NO_EVENT;
          if (it->second)
            next = it->first.acquire(0, true/*exclusive*/,
                                         start_condition);
          else
            next = it->first.acquire(1, false/*exclusive*/,
                                         start_condition);
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
          if (!next.exists())
          {
            UserEvent new_next = UserEvent::create_user_event();
            new_next.trigger();
            next = new_next;
          }
#endif
#ifdef LEGION_LOGGING
          LegionLogging::log_event_dependence(
              Machine::get_executing_processor(), start_condition, next);
#endif
#ifdef LEGION_SPY
          LegionSpy::log_event_dependence(start_condition, next);
#endif
          start_condition = next;
        }
      }

      // STEP 2: Set up the task's context
      index_deleted.resize(indexes.size(),false);
      region_deleted.resize(regions.size(),false);
      std::vector<UserEvent>         unmap_events(regions.size());
      {
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(regions[idx].handle_type == SINGULAR);
#endif
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx])
          {
            physical_regions.push_back(PhysicalRegion(
                  new PhysicalRegion::Impl(regions[idx],
                    Event::NO_EVENT, false/*mapped*/,
                    this, map_id, tag, false/*leaf*/, runtime)));
            // Don't switch coherence modes since we virtually
            // mapped it which means we will map in the parent's
            // context
#ifdef LEGION_LOGGING
            unmap_events[idx] = UserEvent::create_user_event();
            unmap_events[idx].trigger();
#endif
          }
          else if (chosen_variant.inner)
          {
            // If this is an inner task then we don't map
            // the region with a physical region, but instead
            // we mark that the unmap event which marks when
            // the region can be used by child tasks should
            // be the ready event.
            clone_requirements[idx].copy_without_mapping_info(regions[idx]);
            localize_region_requirement(clone_requirements[idx]);
            // Also make the region requirement read-write to force
            // people to wait on the value
            clone_requirements[idx].privilege = READ_WRITE;
            physical_regions.push_back(PhysicalRegion(
                  new PhysicalRegion::Impl(regions[idx],
                    Event::NO_EVENT, false/*mapped*/,
                    this, map_id, tag, false/*leaf*/, runtime)));
            unmap_events[idx] = UserEvent::create_user_event();
            // Trigger the user event when the region is 
            // actually ready to be used
            unmap_events[idx].trigger(
                    physical_instances[idx].get_ready_event());
          }
          else
          { 
            // If this is not virtual mapped, here is where we
            // switch coherence modes from whatever they are in
            // the enclosing context to exclusive within the
            // context of this task
            clone_requirements[idx].copy_without_mapping_info(regions[idx]);
            localize_region_requirement(clone_requirements[idx]);
            physical_regions.push_back(PhysicalRegion(
                  new PhysicalRegion::Impl(clone_requirements[idx],
                    Event::NO_EVENT/*already mapped*/, true/*mapped*/,
                    this, map_id, tag, chosen_variant.leaf, runtime)));
            // Now set the reference for this physical region 
            // which is pretty much a dummy physical reference except
            // it references the same view as the outer reference
            unmap_events[idx] = UserEvent::create_user_event();
            // We reset the reference below after we've
            // initialized the local contexts and received
            // back the local instance references
          }
        }

        // If we're a leaf task and we have virtual mappings
        // then it's possible for the application to do inline
        // mappings which require a physical context
        if (!chosen_variant.leaf || (num_virtual_mappings > 0))
        {
          // Request a context from the runtime
          runtime->allocate_context(this);
#ifdef DEBUG_HIGH_LEVEL
          assert(context.exists());
          runtime->forest->check_context_state(context);
#endif
          // If we're going to do the inner task optimization
          // then when we initialize the contexts also pass in the
          // start condition so we can add a user off of which
          // all sub-users should be chained.
          initialize_region_tree_contexts(clone_requirements,
                                          unmap_events);
          if (!chosen_variant.inner)
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (!virtual_mapped[idx])
              {
                physical_regions[idx].impl->reset_reference(
                    local_instances[idx], unmap_events[idx]);
              }
            }
          }
        }
        else
        {
          // Leaf and all non-virtual mappings
          // Mark that all the local instances are empty
          local_instances.resize(regions.size(),InstanceRef());
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            physical_regions[idx].impl->reset_reference(
                physical_instances[idx], unmap_events[idx]);
          }
        }
      }

      // STEP 3: Finally we get to launch the task
      // Get the low-level task ID for the selected variant
      Processor::TaskFuncID low_id = chosen_variant.low_id;
#ifdef DEBUG_HIGH_LEVEL
      assert(this->executing_processor.exists());
      assert(low_id > 0);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef LEGION_LOGGING
        LegionLogging::log_task_instance_requirement(
                                 Machine::get_executing_processor(),
                                 get_unique_task_id(), idx, 
                                 regions[idx].region.get_index_space());
#endif
#ifdef LEGION_SPY
        LegionSpy::log_task_instance_requirement(get_unique_task_id(), idx,
                                 regions[idx].region.get_index_space().id);
#endif
      }
      {
        std::set<Event> unmap_set;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx])
            unmap_set.insert(unmap_events[idx]);
        }
        Event all_unmap_event = Event::merge_events(unmap_set);
        if (!all_unmap_event.exists())
        {
          UserEvent new_all_unmap = UserEvent::create_user_event();
          new_all_unmap.trigger();
          all_unmap_event = new_all_unmap;
        }
#ifdef LEGION_LOGGING
        // we don't log a dependence on the parent's start event, because
        // log_individual_task and log_index_space_task log this relationship
        LegionLogging::log_operation_events(
                                  Machine::get_executing_processor(),
                                  get_unique_task_id(),
                                  start_condition, get_task_completion());
#endif
#ifdef LEGION_SPY
        // Log an implicit dependence on the parent's start event
        LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
                                           start_condition);
        LegionSpy::log_op_events(get_unique_task_id(), 
                                 start_condition, all_unmap_event);
#endif
        this->legion_spy_start = start_condition; 
        // Record the start
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx])
          {
#ifdef LEGION_LOGGING
            LegionLogging::log_event_dependence(
                                  Machine::get_executing_processor(),
                                  all_unmap_event, unmap_events[idx]);
            LegionLogging::log_event_dependence(
                                  Machine::get_executing_processor(),
                                  unmap_events[idx], get_task_completion());
#endif
#ifdef LEGION_SPY
            LegionSpy::log_event_dependence(all_unmap_event, unmap_events[idx]);
            // Log an implicit dependence on the parent's start event
            LegionSpy::log_event_dependence(unmap_events[idx],
                                               get_task_completion());
#endif
          }
        }
#ifdef LEGION_LOGGING
        // as with the start event, we don't log a dependence between the
        // completion events of task/subtask, because
        // log_individual_task and log_index_space_task log this relationship
#endif
#ifdef LEGION_SPY
        LegionSpy::log_implicit_dependence(get_task_completion(),
                                           parent_ctx->get_task_completion());
#endif
      }
#endif
      // Notify the runtime that there is a new task running on the processor
      runtime->increment_pending(executing_processor);
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      LAUNCH_TASK);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_LAUNCH);
#endif
      // If this is a leaf task and we have no virtual instances
      // and the SingleTask sub-type says it is ok
      // we can trigger the task's completion event as soon as
      // the task is done running.  We first need to mark that this
      // is going to occur before actually launching the task to 
      // avoid the race.
      bool perform_chaining_optimization = false; 
      UserEvent chain_complete_event;
      if (chosen_variant.leaf && (num_virtual_mappings == 0) &&
          can_early_complete(chain_complete_event))
        perform_chaining_optimization = true;
      SingleTask *proxy_this = this; // dumb c++
      // Note there is a potential scary race condition to be aware of here: 
      // once we launch this task it's possible for this task to run and 
      // clean up before we finish the execution of this function thereby
      // invalidating this SingleTask object's fields.  This means
      // that we need to save any variables we need for after the task
      // launch here on the stack before they can be invalidated.
      Event term_event = get_task_completion();
      Event task_launch_event = executing_processor.spawn(low_id, &proxy_this,
                            sizeof(proxy_this), start_condition, task_priority);
      // Finish the chaining optimization if we're doing it
      if (perform_chaining_optimization)
        chain_complete_event.trigger(task_launch_event);
      // STEP 4: After we've launched the task, then we have to release any 
      // locks that we took for while the task was running.  
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          it->first.release(term_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& SingleTask::begin_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Task %s (ID %lld) starting on processor " IDFMT "",
                            this->variants->name, get_unique_task_id(), 
                            executing_processor.id);
      assert(regions.size() == physical_regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == region_deleted.size());
      assert(regions.size() == local_instances.size());
#endif
#ifdef LEGION_LOGGING
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (physical_instances[idx].has_ref())
        {
          LegionLogging::log_physical_user(executing_processor,
            physical_instances[idx].get_handle().get_view()->
            get_manager()->get_instance(), get_unique_task_id(), idx);
        }
      }
#endif
#ifdef LEGION_SPY
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (physical_instances[idx].has_ref())
        {
          LegionSpy::log_op_user(unique_op_id, idx, 
           physical_instances[idx].get_handle().get_view()->
                                get_manager()->get_instance().id);
        }
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      BEGIN_EXECUTION);
#endif
      // Tell the runtime that this task is now running
      // and is no longer pending
      runtime->start_execution(executing_processor);
      // Do the decrement on the processor we initially incremented
      runtime->decrement_pending(executing_processor);
      // Start the profiling if requested
      if (profile_task)
        this->start_time = (TimeStamp::get_current_time_in_micros() - 
                              Runtime::init_time);
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_BEGIN_EXECUTION);
#endif
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    void SingleTask::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
      // Tell the runtime that this task has finished executing
      runtime->pause_execution(executing_processor);

      if (profile_task)
      {
        this->stop_time = (TimeStamp::get_current_time_in_micros() -
                              Runtime::init_time);
        runtime->invoke_mapper_notify_profiling(executing_processor, this);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == region_deleted.size());
      assert(regions.size() == local_instances.size());
#endif
      // Unmap all of the physical regions which are still mapped
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (physical_regions[idx].impl->is_mapped())
          physical_regions[idx].impl->unmap_region();
      }
      // Now we can clear the physical regions since we're done using them
      physical_regions.clear();
      // Do the same thing with any residual inline mapped regions
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (it->impl->is_mapped())
          it->impl->unmap_region();
      }
      inline_regions.clear();

      // For each of the regions which were not virtual mapped, 
      // issue a close operation for them.  If we're a leaf task
      // and we had no virtual mappings then we are done.
      if (!is_leaf() || (num_virtual_mappings > 0))
      {
        for (unsigned idx = 0; idx < local_instances.size(); idx++)
        {
          if (!virtual_mapped[idx] && !region_deleted[idx]
              && !IS_READ_ONLY(regions[idx]))
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(local_instances[idx].has_ref());
#endif
            CloseOp *close_op = runtime->get_available_close_op();    
            close_op->initialize(this, idx, local_instances[idx]);
            runtime->add_to_dependence_queue(executing_processor, close_op);
          }
        }
      }
      local_instances.clear();

#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_EXECUTION);
#endif

      // Handle the future result
      handle_future(res, res_size, owned); 

#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_END_EXECUTION);
#endif
      
      // See if we want to move the rest of this computation onto
      // the utility processor
#ifdef SPECIALIZED_UTIL_PROCS
      Processor util = runtime->get_cleanup_proc(executing_processor);
#else
      Processor util = executing_processor.get_utility_processor();
#endif
      if (util != executing_processor)
      {
        PostEndArgs post_end_args;
        post_end_args.hlr_id = HLR_POST_END_ID;
        post_end_args.proxy_this = this;
        util.spawn(HLR_TASK_ID, &post_end_args, sizeof(post_end_args));
      }
      else
        post_end_task();
    }

    //--------------------------------------------------------------------------
    void SingleTask::post_end_task(void)
    //--------------------------------------------------------------------------
    {
#if defined(LEGION_PROF) || defined(LEGION_LOGGING)
      UniqueID local_id = get_unique_task_id();
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(local_id, PROF_BEGIN_POST);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      local_id,
                                      BEGIN_POST_EXEC);
#endif
      // Mark that we are done executing this operation
      complete_execution();
      // Mark that we are done executing and then see if we need to
      // trigger any of our mapping, completion, or commit methods
      bool need_complete = false;
      bool need_commit = false;
      // If we're a leaf with no virtual mappings then
      // there are guaranteed to be no children
      {
        AutoLock o_lock(op_lock);
        if (executing_children.empty() && executed_children.empty())
        {
          if (!children_complete_invoked)
          {
            need_complete = true;
            children_complete_invoked = true;
          }
          if (complete_children.empty() && 
              !children_commit_invoked)
          {
            need_commit = true;
            children_commit_invoked = true;
          }
        }
      } 
      if (need_complete)
      {
        // If we had any virtual mappings, mark that we are
        // now mapping complete since all children are mapped
        if (num_virtual_mappings > 0)
          complete_mapping();
        trigger_children_complete();
      }
      if (need_commit)
      {
        trigger_children_committed();
      } 
#ifdef LEGION_PROF
      LegionProf::register_event(local_id, PROF_END_POST);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      local_id,
                                      END_POST_EXEC);
#endif
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& SingleTask::get_physical_regions() const
    //--------------------------------------------------------------------------
    {
      return physical_regions;
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(Runtime *rt)
      : TaskOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MultiTask::activate_multi(void)
    //--------------------------------------------------------------------------
    {
      activate_task();
      sliced = false;
      redop = 0;
      reduction_op = NULL;
      reduction_state_size = 0;
      reduction_state = NULL;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate_multi(void)
    //--------------------------------------------------------------------------
    {
      deactivate_task();
      if (reduction_state != NULL)
      {
        free(reduction_state);
        reduction_state = NULL;
        reduction_state_size = 0;
      }
      // Remove our reference to any argument maps
      argument_map = ArgumentMap();
      slices.clear(); 
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void) const
    //--------------------------------------------------------------------------
    {
      return sliced;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::slice_index_space(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!sliced);
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      BEGIN_SLICING);
#endif
      sliced = true;
      spawn_task = false; // cannot steal something that has been sliced
      std::vector<Mapper::DomainSplit> splits;
      runtime->invoke_mapper_slice_domain(current_proc, this, splits);
      if (splits.empty())
      {
        log_run(LEVEL_ERROR,"Invalid mapper domain slice result for mapper %d "
                            "on processor " IDFMT " for task %s (ID %lld)",
                            map_id, current_proc.id, variants->name, 
                            get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
      }

      // Do this before here since each of the clones will slice themselves
      if (must_parallelism && (slices.size() > 1))
        must_barrier = must_barrier.alter_arrival_count(slices.size()-1);

      for (unsigned idx = 0; idx < splits.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        // Check to make sure the domain is not empty
        {
          const Domain &d = splits[idx].domain;
          switch (d.dim)
          {
            case 0:
              {
                if (d.get_volume() <= 0)
                {
                  log_run(LEVEL_ERROR,
                            "Invalid mapper domain slice result for mapper %d "
                            "on processor " IDFMT " for task %s (ID %lld). "
                            "Mapper returned an empty domain for split %d.",
                            map_id, current_proc.id, variants->name,
                            get_unique_task_id(), idx);
                  assert(false);
                  exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
                }
                break;
              }
            case 1:
              {
                Rect<1> rec = d.get_rect<1>();
                if (rec.volume() <= 0)
                {
                  log_run(LEVEL_ERROR,
                            "Invalid mapper domain slice result for mapper %d "
                            "on processor " IDFMT " for task %s (ID %lld).  "
                            "Mapper returned an empty domain for split %d.",
                            map_id, current_proc.id, variants->name, 
                            get_unique_task_id(), idx);
                  assert(false);
                  exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
                }
                break;
              }
            case 2:
              {
                Rect<2> rec = d.get_rect<2>();
                if (rec.volume() <= 0)
                {
                  log_run(LEVEL_ERROR,
                            "Invalid mapper domain slice result for mapper %d "
                            "on processor " IDFMT " for task %s (ID %lld).  "
                            "Mapper returned an empty domain for split %d.",
                            map_id, current_proc.id, variants->name, 
                            get_unique_task_id(), idx);
                  assert(false);
                  exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
                }
                break;
              }
            case 3:
              {
                Rect<3> rec = d.get_rect<3>();
                if (rec.volume() <= 0)
                {
                  log_run(LEVEL_ERROR,
                            "Invalid mapper domain slice result for mapper %d "
                            "on processor " IDFMT " for task %s (ID %lld).  "
                            "Mapper returned an empty domain for split %d.",
                            map_id, current_proc.id, variants->name, 
                            get_unique_task_id(), idx);
                  assert(false);
                  exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
                }
                break;
              }
            default:
              assert(false);
          }
        }
#endif
        SliceTask *slice = this->clone_as_slice_task(splits[idx].domain,
                                                     splits[idx].proc,
                                                     splits[idx].recurse,
                                                     splits[idx].stealable,
                                                     splits.size());
        slices.push_back(slice);
      }

#ifdef LEGION_LOGGING
      UniqueID local_id = get_unique_task_id();
#endif
      bool success = trigger_slices(); 
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      local_id, END_SLICING);
#endif
      // If we succeeded and this is an intermediate slice task
      // then we can reclaim it, otherwise, if it is the original
      // index task then we want to keep it around. Note it is safe
      // to call get_task_kind here despite the cleanup race because
      // it is a static property of the object.
      if (success && (get_task_kind() == SLICE_TASK_KIND))
        deactivate();
      return success;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::trigger_slices(void)
    //--------------------------------------------------------------------------
    {
      DeferredSlicer slicer(this);
      return slicer.trigger_slices(slices);
    }

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(MultiTask *rhs, const Domain &d,
                                     Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      this->clone_task_op_from(rhs, p, stealable, false/*duplicate*/);
      this->index_domain = d;
      this->must_parallelism = rhs->must_parallelism;
      this->sliced = !recurse;
      if (must_parallelism)
        this->must_barrier = rhs->must_barrier;
      this->argument_map = rhs->argument_map;
      this->redop = rhs->redop;
      if (this->redop != 0)
      {
        this->reduction_op = rhs->reduction_op;
        initialize_reduction_state();
      }
    }

    //--------------------------------------------------------------------------
    bool MultiTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      UniqueID local_id = get_unique_task_id();
      LegionProf::register_event(local_id, PROF_BEGIN_TRIGGER);
#endif
      bool success = true;
      if (is_remote())
      {
        // distribute, slice, then map/launch
        if (distribute_task())
        {
          // Still local
          if (is_sliced())
          {
            if (is_locally_mapped())
            {
              launch_task();
            }
            else
            {
              // Try mapping and launching
              success = map_and_launch();
            }
          }
          else
            success = slice_index_space();
        }
      }
      else
      {
        // Not remote, make sure it is premapped
        if (is_premapped() || premap_task())
        {
          if (is_locally_mapped())
          {
            if (is_sliced())
            {
              if (must_epoch != NULL)
                register_must_epoch();
              else
              {
                // See if we're going to send it
                // remotely.  If so we need to do
                // the mapping now.  Otherwise we
                // can defer the mapping until we get
                // on the target processor.
                if (!runtime->is_local(target_proc))
                {
                  if (perform_mapping())
                  {
#ifdef DEBUG_HIGH_LEVEL
                    bool still_local = 
#endif
                    distribute_task();
#ifdef DEBUG_HIGH_LEVEL
                    assert(!still_local);
#endif
                  }
                  else // failed to map
                    success = false;
                }
                else
                {
                  // We know that it is staying on one
                  // of our local processors.  If it is
                  // still this processor then map and run it
                  if (distribute_task())
                  {
                    // Still local so we can map and launch it
                    success = map_and_launch();
                  }
                }
              }
            }
            else
              success = slice_index_space();
          }
          else
          {
            if (distribute_task())
            {
              // Still local try slicing, mapping, and launching
              if (is_sliced())
              {
                success = map_and_launch();
              }
              else
                success = slice_index_space();
            }
          }
        }
        else // failed to premap
          success = false; 
      }
#ifdef LEGION_PROF
      LegionProf::register_event(local_id, PROF_END_TRIGGER);
#endif
      return success;
    } 

    //--------------------------------------------------------------------------
    void MultiTask::add_created_index(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::add_created_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::add_created_field(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::remove_created_index(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::remove_created_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::remove_created_field(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      // Do nothing
    }

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer &rez, bool pack_args, 
                                    AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(sliced);
      rez.serialize(must_barrier);
      rez.serialize(redop);
      if (pack_args)
        argument_map.impl->pack_arguments(rez, index_domain);
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez, bool unpack_args)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_base_task(derez); 
      derez.deserialize(sliced);
      derez.deserialize(must_barrier);
      derez.deserialize(redop);
      if (redop > 0)
      {
        reduction_op = Runtime::get_reduction_op(redop);
        initialize_reduction_state();
      }
      if (unpack_args)
      {
        argument_map = 
          ArgumentMap(new ArgumentMap::Impl(new ArgumentMapStore()));
        argument_map.impl->unpack_arguments(derez);
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::initialize_reduction_state(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reduction_op != NULL);
      assert(reduction_op->is_foldable);
      assert(reduction_state == NULL);
#endif
      reduction_state_size = reduction_op->sizeof_rhs;
      reduction_state = malloc(reduction_state_size);
      reduction_op->init(reduction_state, 1);
    }

    //--------------------------------------------------------------------------
    void MultiTask::fold_reduction_future(const void *result, 
                                          size_t result_size, 
                                          bool owner, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Apply the reduction operation
#ifdef DEBUG_HIGH_LEVEL
      assert(reduction_op != NULL);
      assert(reduction_op->is_foldable);
      assert(reduction_state != NULL);
      assert(result_size == reduction_op->sizeof_rhs);
#endif
      // Perform the reduction
      reduction_op->fold(reduction_state, result, 1, exclusive);

      // If we're the owner, then free the memory
      if (owner)
        free(const_cast<void*>(result));
    }

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(const IndividualTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask& IndividualTask::operator=(const IndividualTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_single();
      future_store = NULL;
      future_size = 0;
      orig_task = this;
      remote_completion_event = get_completion_event();
      remote_unique_id = get_unique_task_id();
      sent_remotely = false;
      top_level_task = false;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // If we are remote then we need to deactivate our parent context
      if (is_remote() || top_level_task)
        parent_ctx->deactivate();
      deactivate_single();
      if (future_store != NULL)
      {
        free(future_store);
        future_store = NULL;
        future_size = 0;
      }
      // Remove our reference on the future
      result = Future();
      privilege_paths.clear();
      // Read this before freeing the task
      // Should be safe, but we'll be careful
      bool is_top_level_task = top_level_task;
      runtime->free_individual_task(this);
      // If we are the top-level-task and we are deactivated then
      // it is now safe to shutdown the machine
      if (is_top_level_task)
        runtime->initiate_runtime_shutdown();
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::initialize_task(SingleTask *ctx,
                                           const TaskLauncher &launcher,
                                           bool check_privileges,
                                           bool track /*=true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions = launcher.region_requirements;
      regions.resize(launcher.region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].copy_without_mapping_info(
            launcher.region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      futures = launcher.futures;
      grants = launcher.grants;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.argument.get_size();
      if (arglen > 0)
      {
        args = malloc(arglen);
        memcpy(args,launcher.argument.get_ptr(),arglen);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      index_point = launcher.point;
      is_index_space = false;
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      remote_contexts = enclosing_physical_contexts;
      remote_outermost_context = 
        find_outermost_physical_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_outermost_context.exists());
#endif
      initialize_paths(); 
      // Get a future from the parent context to use as the result
      result = Future(new Future::Impl(runtime, true/*register*/, 
            runtime->get_available_distributed_id(), runtime->address_space,
            runtime->address_space, this));
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_individual_task(parent_ctx->get_executing_processor(),
                                         parent_ctx->get_unique_task_id(),
                                         unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_individual_task(parent_ctx->get_unique_task_id(),
                                     unique_op_id,
                                     task_id,
                                     variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::initialize_task(SingleTask *ctx,
              Processor::TaskFuncID tid,
              const std::vector<IndexSpaceRequirement> &index_requirements,
              const std::vector<RegionRequirement> &region_requirements,
              const TaskArgument &arg,
              const Predicate &pred,
              MapperID mid, MappingTagID t,
              bool check_privileges, bool track /*=true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = tid;
      indexes = index_requirements;
      regions = region_requirements;
      regions.resize(region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      { 
        regions[idx].copy_without_mapping_info(region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      arglen = arg.get_size();
      if (arglen > 0)
      {
        args = malloc(arglen);
        memcpy(args,arg.get_ptr(),arglen);
      }
      map_id = mid;
      tag = t;
      is_index_space = false;
      initialize_base_task(ctx, track, pred, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      remote_contexts = enclosing_physical_contexts;
      remote_outermost_context = 
        find_outermost_physical_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_outermost_context.exists());
#endif
      initialize_paths();
      result = Future(new Future::Impl(runtime, true/*register*/,
            runtime->get_available_distributed_id(), runtime->address_space,
            runtime->address_space, this));
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_individual_task(parent_ctx->get_executing_processor(),
                                         parent_ctx->get_unique_task_id(),
                                         unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_individual_task(parent_ctx->get_unique_task_id(),
                                     unique_op_id,
                                     task_id,
                                     variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return result;
    }  

    //--------------------------------------------------------------------------
    void IndividualTask::initialize_paths(void)
    //--------------------------------------------------------------------------
    {
      privilege_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        initialize_privilege_path(privilege_paths[idx], regions[idx]);
      }
      initialize_mapping_paths();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(privilege_paths.size() == regions.size());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(), 
                                      BEGIN_DEPENDENCE_ANALYSIS);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_BEGIN_DEP_ANALYSIS);
#endif
      begin_dependence_analysis();
      RegionTreeContext ctx = parent_ctx->get_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(ctx, this, idx, 
                                    regions[idx], privilege_paths[idx]);
      }
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_DEPENDENCE_ANALYSIS);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_END_DEP_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void IndividualTask::report_aliased_requirements(unsigned idx1, 
                                                     unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 0
      log_run(LEVEL_ERROR,"Aliased region requirements for individual tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are aliased.", idx1, idx2, variants->name,
                          get_unique_task_id(), parent_ctx->variants->name,
                          parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_ALIASED_REGION_REQUIREMENTS);
#else
      log_run(LEVEL_WARNING,"Region requirements %d and %d of individual task "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "aliased.  This behavior is currently undefined. "
                            "You better really know what you are doing.",
                            idx1, idx2, variants->name, get_unique_task_id(),
                            parent_ctx->variants->name, 
                            parent_ctx->get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      if (premapped)
        return true;
      premapped = true;
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing_physical_contexts.size() == regions.size());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      BEGIN_PRE_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), 
                                 PROF_BEGIN_PREMAP_ANALYSIS);
#endif
      // All regions need to be premapped no matter what
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Check to see if we already premapped this region
        // If not then we need to do it now
        if (!regions[idx].premapped)
        {
          if (!runtime->forest->premap_physical_region(
                                       enclosing_physical_contexts[idx],
                                       privilege_paths[idx], regions[idx], 
                                       this, parent_ctx, 
                                       parent_ctx->get_executing_processor()
#ifdef DEBUG_HIGH_LEVEL
                                       , idx, get_logging_name(), unique_op_id
#endif
                                       ))
          {
            // Failed to premap, break out and try again later
            premapped = false;
            break;
          }
          else
          {
            regions[idx].premapped = true;
          }
        }
      }
      if (premapped)
        premapped = early_map_regions();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_PRE_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(),
                                 PROF_END_PREMAP_ANALYSIS);
#endif
      return premapped;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      if (premapped)
        return true;
      else
        return premap_task();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::defer_mapping(void)
    //--------------------------------------------------------------------------
    {
      // If we are a stealable task and we are remote then we need
      // to request the information necessary to map
      if (needs_state)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_remote());
#endif
        needs_state = false;
        // Send a message to the original processor requesting
        // the mapping infomration
        Serializer rez;
        {
          RezCheck z(rez); 
          rez.serialize(orig_task);
          IndividualTask *proxy_this = this;
          rez.serialize(proxy_this);
        }
        runtime->send_individual_request(
            runtime->find_address_space(orig_proc), rez);
        return true;
      }
      // No need to defer otherwise
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (target_proc != current_proc)
      {
        runtime->send_task(target_proc, this);
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
      // Before we try mapping the task, ask the mapper to pick a task variant
      runtime->invoke_mapper_select_variant(current_proc, this);
#ifdef DEBUG_HIGH_LEVEL
      if (!variants->has_variant(selected_variant))
      {
        log_task(LEVEL_ERROR,"Invalid task variant %ld selected by mapper "
                             "for task %s (ID %lld)", selected_variant,
                             variants->name, get_unique_task_id());
        assert(false);
        exit(ERROR_INVALID_VARIANT_SELECTION);
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_task_instance_variant(
                                  Machine::get_executing_processor(),
                                  get_unique_task_id(),
                                  selected_variant);
#endif
      // Now try to do the mapping, we can just use our completion
      // event since we know this task will object will be active
      // throughout the duration of the computation
      bool map_success = map_all_regions(target_proc, 
                                         get_task_completion(), mapper_invoked);
      // If we mapped, then we are no longer stealable
      if (map_success)
        spawn_task = false;
      // If we succeeded in mapping and everything was mapped
      // then we get to mark that we are done mapping
      if (map_success && (num_virtual_mappings == 0))
      {
        if (is_remote())
        {
          // Send back the message saying that we finished mapping
          Serializer rez;
          pack_remote_mapped(rez);
          runtime->send_individual_remote_mapped(orig_proc, rez);
        }
        // Mark that we have completed mapping
        complete_mapping();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_locally) && spawn_task);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::can_early_complete(UserEvent &chain_event)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return false;
      // Otherwise we're going to do it mark that we
      // don't need to trigger the underlying completion event.
      // Note we need to do this now to avoid any race condition.
      need_completion_trigger = false;
      chain_event = completion_event;
      return true;
    }

    //--------------------------------------------------------------------------
    Event IndividualTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return remote_completion_event;
      else
        return completion_event;
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndividualTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDIVIDUAL_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext IndividualTask::find_enclosing_physical_context(
                                                          LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock when accessing these data structures
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == virtual_mapped.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if ((regions[idx].region == parent) && !region_deleted[idx])
        {
          if (!virtual_mapped[idx])
            return context;
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(parent_ctx != NULL);
#endif
            return enclosing_physical_contexts[idx];
          }
        }
      }
      // If we get here that means that our privilege checking framework
      // is failing to catch a case where we don't actually have privileges.
      assert(false);
      return RegionTreeContext();
    }

    //--------------------------------------------------------------------------
    RemoteTask* IndividualTask::find_outermost_physical_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->find_outermost_physical_context();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      unique_op_id, COMPLETE_OPERATION);
#endif
      if (!is_remote())
      {
        // Pass back our created and deleted operations
        if (!top_level_task)
          return_privilege_state(parent_ctx);

        // The future has already been set so just trigger it
        result.impl->complete_future();
      }
      else
      {
        // Send back any messages to say that we are complete
        if (num_virtual_mappings > 0)
        {
          Serializer rez;
          pack_remote_mapped(rez);
          runtime->send_individual_remote_mapped(orig_proc,rez,false/*flush*/);
        }
        Serializer rez;
        pack_remote_complete(rez);
        runtime->send_individual_remote_complete(orig_proc,rez);
      }
      // Invalidate any state that we had 
      if (context.exists() && (!is_leaf() || (num_virtual_mappings > 0)))
        invalidate_region_tree_contexts();
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_COMPLETE);
#endif
      // Mark that this operation is complete
      complete_operation();
      // See if we need to trigger that our children are complete
      // Note it is only safe to do this if we were not sent remotely
      bool need_commit = false;
      if (!sent_remotely)
      {
        AutoLock o_lock(op_lock);
        if (complete_children.empty() && !children_commit_invoked)
        {
          need_commit = true;
          children_commit_invoked = true;
        }
      }
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez);
        runtime->send_individual_remote_commit(orig_proc,rez);
      }
      commit_operation();
      // Finally we can deactivate this task now that it has commited
      deactivate();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future(const void *res, size_t res_size,
                                       bool owned)
    //--------------------------------------------------------------------------
    {
      // Save our future value so we can set it or send it back later
      if (is_remote())
      {
        if (owned)
        {
          future_store = const_cast<void*>(res);
          future_size = res_size;
        }
        else
        {
          future_size = res_size;
          future_store = malloc(future_size);
          memcpy(future_store,res,future_size);
        }
      }
      else
      {
        // Set our future, but don't trigger it yet
        if (must_epoch == NULL)
          result.impl->set_result(res, res_size, owned);
        else
          must_epoch->set_future(index_point, res, res_size, owned);
      }
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are stealable, if not and we have not
      // yet been sent remotely, then send the state
      AddressSpaceID addr_target = runtime->find_address_space(target);
      if (!spawn_task && !is_remote())
      {
        needs_state = false;
        send_remote_state(addr_target);
      }
      else
        needs_state = true;
      RezCheck z(rez);
      pack_single_task(rez, addr_target);
      rez.serialize(orig_task);
      rez.serialize(remote_completion_event);
      rez.serialize(remote_unique_id);
      rez.serialize(remote_outermost_context);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == remote_contexts.size());
#endif
      for (unsigned idx = 0; idx < remote_contexts.size(); idx++)
        rez.serialize(remote_contexts[idx]);
      parent_ctx->pack_parent_task(rez);
      // Mark that we sent this task remotely
      sent_remotely = true;
      // If this task is remote, then deactivate it, otherwise
      // we're local so we don't want to be deactivated for when
      // return messages get sent back.
      return is_remote();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_single_task(derez);
      derez.deserialize(orig_task);
      derez.deserialize(remote_completion_event);
      derez.deserialize(remote_unique_id);
      derez.deserialize(remote_outermost_context);
      current_proc = current;
      remote_contexts.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        derez.deserialize(remote_contexts[idx]);
      RemoteTask *remote_ctx = 
        runtime->find_or_init_remote_context(remote_unique_id);
      remote_ctx->unpack_parent_task(derez);
      // Add our enclosing parent regions to the list of 
      // top regions maintained by the remote context
      for (unsigned idx = 0; idx < regions.size(); idx++)
        remote_ctx->add_top_region(regions[idx].parent);
      enclosing_physical_contexts.resize(regions.size());
      // Mark that all of our enclosing physical contexts are marked
      // by the remote version
      for (unsigned idx = 0; idx < regions.size(); idx++)
        enclosing_physical_contexts[idx] = remote_ctx->get_context();
      // Now save the remote context as our parent context
      parent_ctx = remote_ctx;
      // Quick check to see if we've been sent back to our original node
      if (!is_remote())
      {
        // If we were sent back then mark that we are no longer remote
        sent_remotely = false;
        // Put the original instance back on the mapping queue and
        // deactivate this version of the task
        runtime->add_to_ready_queue(current_proc, orig_task, 
                                    false/*prev fail*/);
        deactivate();
        return false;
      }
      // Check to see if we had no virtual mappings and everything
      // was pre-mapped and we're remote then we can mark this
      // task as being mapped
      if (is_locally_mapped() && (num_virtual_mappings == 0))
        complete_mapping();
#ifdef LEGION_LOGGING
      LegionLogging::log_point_point(Machine::get_executing_processor(),
                                     remote_unique_id,
                                     get_unique_task_id());
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_point_point(remote_unique_id, get_unique_task_id());
#endif
      // Return true to add ourselves to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::find_enclosing_local_fields(
                                              std::deque<LocalFieldInfo> &infos)
    //--------------------------------------------------------------------------
    {
      // Ask the same for our parent context
      parent_ctx->find_enclosing_local_fields(infos);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_inlining(SingleTask *ctx, InlineFnptr fn) 
    //--------------------------------------------------------------------------
    {
      // See if there is anything that we need to wait on before running
      std::set<Event> wait_on_events;
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        Future::Impl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->ready_event);
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        Grant::Impl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      Event start_condition = Event::merge_events(wait_on_events); 

      // See if we need to wait for anything
      if (start_condition.exists() && !start_condition.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        runtime->pre_wait(proc);
        start_condition.wait();
        runtime->post_wait(proc);
      }

      // Run the task  
      (*fn)(this, ctx->get_physical_regions(), ctx, 
            runtime->high_level, future_store, future_size);
      // Save the future result and trigger it, mark that the
      // future owns the result.
      result.impl->set_result(future_store,future_size,true);
      future_store = NULL;
      future_size = 0;
      result.impl->complete_future();

      // Trigger our completion event
      completion_event.trigger();
      // Now we're done, someone else will deactivate us
    }  

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_mapped(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_contexts.size() <= regions.size());
      assert(remote_contexts.size() <= enclosing_physical_contexts.size());
      assert(remote_contexts.size() <= mapping_paths.size());
#endif
      // First send back any region tree state that can be sent back
      // This includes everything for which we have remote context information
      // and which was locally mapped.  Note this includes things that are
      // virtually mapped for which we own the state here.
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      std::set<PhysicalManager*> needed_managers;
      for (unsigned idx = 0; idx < remote_contexts.size(); idx++)
      {
        if (locally_mapped[idx])
        {
          runtime->forest->send_back_physical_state(
                                    enclosing_physical_contexts[idx],
                                    remote_contexts[idx],
                                    mapping_paths[idx],
                                    regions[idx], target,
                                    needed_managers);
        }
      }
      if (!needed_managers.empty())
        runtime->forest->send_remote_references(needed_managers, target);
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Nothing more to unpack, we know everything is mapped
      // so tell everyone that we are mapped
      if (!is_locally_mapped())
        complete_mapping();
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer &rez) 
    //--------------------------------------------------------------------------
    {
      // First send back any created region tree state.  Note to do
      // this we can start at the index of the remote contexts since we
      // know that any additional region beyond this are ones for which
      // we have created the privileges.
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      std::set<PhysicalManager*> needed_managers;
      for (unsigned idx = remote_contexts.size(); idx < regions.size(); idx++)
      {
        if (locally_mapped[idx] && !region_deleted[idx])
        {
          RegionTreePath path;
          // Initialize a path to use
          initialize_mapping_path(path, regions[idx], regions[idx].parent);
          runtime->forest->send_back_physical_state(
                                    enclosing_physical_contexts[idx],
                                    remote_outermost_context,
                                    path, regions[idx], target,
                                    needed_managers);
        }
      }
      if (!needed_managers.empty())
        runtime->forest->send_remote_references(needed_managers, target);
      // Send back the pointer to the task instance, then serialize
      // everything else that needs to be sent back
      rez.serialize(orig_task);
      RezCheck z(rez);
      // Pack the privilege state
      pack_privilege_state(rez, target);
      // Then pack the future result
      {
        RezCheck z2(rez);
        rez.serialize(future_size);
        rez.serialize(future_store,future_size);
      }
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // First unpack the privilege state
      unpack_privilege_state(derez);
      // Unpack the future result
      if (must_epoch == NULL)
        result.impl->unpack_future(derez);
      else
        must_epoch->unpack_future(index_point, derez);
      // Mark that we have both finished executing and that our
      // children are complete
      complete_execution();
#ifdef DEBUG_HIGH_LEVEL
      // No need for a lock here since we know
      // that it is a single task
      assert(!children_complete_invoked);
      children_complete_invoked = true;
#endif
      trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_commit(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      // Don't need the lock here since we know that
      // this is an individual task
      assert(!children_commit_invoked);
      children_commit_invoked = true;
#endif
      trigger_children_committed();
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::send_remote_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Send state for anything that was not premapped and has not
      // already been mapped
      std::map<LogicalView*,FieldMask> needed_views;
      std::set<PhysicalManager*> needed_managers;
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing_physical_contexts.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        runtime->forest->send_tree_shape(indexes[idx], target);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Can skip any early mapped regions
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
        {
          // Anything that is sending a reference will pack
          // its own region tree shape
          continue;
        }
        // Skip anything that is already mapped
        if ((idx < physical_instances.size()) &&
            physical_instances[idx].has_ref())
        {
          // Anything that is already mapped will send its own
          // region tree state when packing the reference
          continue;
        }
        // Otherwise we need to send the state
        runtime->forest->send_physical_state(enclosing_physical_contexts[idx],
                                             regions[idx],
                                             remote_unique_id,
                                             target,
                                             needed_views,
                                             needed_managers); 
      }
      if (!needed_views.empty() || !needed_managers.empty())
        runtime->forest->send_remote_references(needed_views,
                                                needed_managers, target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::handle_individual_request(Runtime *runtime,
                                                          Deserializer &derez,
                                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask *local_task;
      derez.deserialize(local_task);
      IndividualTask *remote_task;
      derez.deserialize(remote_task);
      // Send the state back to the source
      local_task->send_remote_state(source);
      // Now send a message back to the remote task saying the
      // state has been sent
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(remote_task);
      }
      runtime->send_individual_return(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::handle_individual_return(Runtime *runtime,
                                                          Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask *local_task;
      derez.deserialize(local_task);
      // State is now local so put it on the ready queue
      runtime->add_to_ready_queue(local_task->current_proc, local_task,
                                  false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_mapped(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_mapped(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_complete(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_commit(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_commit(derez);
    }

    /////////////////////////////////////////////////////////////
    // Point Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointTask::PointTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask::PointTask(const PointTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask& PointTask::operator=(const PointTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_single();
      slice_owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_single();
      runtime->free_point_task(this);
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are always ready to map since they had to be
      // premapped first anyway
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::defer_mapping(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are never sent anywhere
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::perform_mapping(bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
      // For point tasks we use the point termination event which as the
      // end event for this task since point tasks can be moved and
      // the completion event is therefore not guaranteed to survive
      // the length of the task's execution
      bool map_success = map_all_regions(target_proc, 
                                         point_termination, mapper_invoked);
      // If we succeeded in mapping and had no virtual mappings
      // then we are done mapping
      if (map_success && (num_virtual_mappings == 0))
      {
        // Tell our owner that we mapped
        slice_owner->record_child_mapped();
        // Mark that we ourselves have mapped
        complete_mapping();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::can_early_complete(UserEvent &chain_event)
    //--------------------------------------------------------------------------
    {
      chain_event = point_termination;
      return true;
    }

    //--------------------------------------------------------------------------
    Event PointTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      return point_termination;
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind PointTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POINT_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext PointTask::find_enclosing_physical_context(
                                                          LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock when accessing these data structures
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == virtual_mapped.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if ((regions[idx].region == parent) && !region_deleted[idx])
        {
          if (!virtual_mapped[idx])
            return context;
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(parent_ctx != NULL);
#endif
            return parent_ctx->find_enclosing_physical_context(
                                                regions[idx].parent);
          }
        }
      }
      // If we get here that means that our privilege checking framework
      // is failing to catch a case where we don't actually have privileges.
      assert(false);
      return RegionTreeContext();
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_inlining(SingleTask *ctx, InlineFnptr fn)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteTask* PointTask::find_outermost_physical_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->find_outermost_physical_context();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      unique_op_id, COMPLETE_OPERATION);
#endif
      // If we had any virtual mappings, we can now be considered mapped
      if (num_virtual_mappings > 0)
        slice_owner->record_child_mapped();

      // Pass back our created and deleted operations 
      slice_owner->return_privileges(this);

      slice_owner->record_child_complete();

      // Since this point is now complete we know
      // that we can trigger it. Note we don't need to do
      // this if we're a leaf task with no virtual mappings
      // because we would have performed the leaf task
      // early complete chaining operation.
      if (!is_leaf() || (num_virtual_mappings > 0))
        point_termination.trigger();

      // Invalidate any context that we had so that the child
      // operations can begin committing
      if (context.exists() && (!is_leaf() || (num_virtual_mappings > 0)))
        invalidate_region_tree_contexts();
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_COMPLETE);
#endif 
      // Mark that this operation is now complete
      complete_operation();
      // See if we need to trigger that our children are complete
      bool need_commit = false;
      {
        AutoLock o_lock(op_lock);
        if (complete_children.empty() && !children_commit_invoked)
        {
          need_commit = true;
          children_commit_invoked = true;
        }
      }
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Tell our slice owner that we're done
      slice_owner->record_child_committed();
      // Commit this operation
      commit_operation();
      // Then we get to deactivate ourselves
      deactivate();
    }

    //--------------------------------------------------------------------------
    bool PointTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_single_task(rez, runtime->find_address_space(target));
      rez.serialize(point_termination); 
      // Return false since point tasks should always be deactivated
      // once they are sent to a remote node
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_single_task(derez);
      derez.deserialize(point_termination);
      current_proc = current;
      // Check to see if we had no virtual mappings and everything
      // was pre-mapped and we're remote then we can mark this
      // task as being mapped
      if (is_locally_mapped() && (num_virtual_mappings == 0))
      {
        slice_owner->record_child_mapped();
        complete_mapping();
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_slice_point(Machine::get_executing_processor(),
                                     slice_owner->get_unique_task_id(),
                                     get_unique_task_id(), index_point);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::find_enclosing_local_fields(
                                              std::deque<LocalFieldInfo> &infos)
    //--------------------------------------------------------------------------
    {
      // Ask the same for our parent context
      parent_ctx->find_enclosing_local_fields(infos);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future(const void *res, size_t res_size, bool owner)
    //--------------------------------------------------------------------------
    {
      slice_owner->handle_future(index_point, res, res_size, owner);
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(SliceTask *owner)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      compute_point_region_requirements();
      // Make a new termination event for this point
      point_termination = UserEvent::create_user_event();
      // Finally compute the paths for this point task
      mapping_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (owner->regions[idx].handle_type == PART_PROJECTION)
        {
          initialize_mapping_path(mapping_paths[idx], regions[idx],
                                  owner->regions[idx].partition);
        }
        else
        {
          initialize_mapping_path(mapping_paths[idx], regions[idx],
                                  owner->regions[idx].region);
        }
      }
    } 

    //--------------------------------------------------------------------------
    void PointTask::send_back_remote_state(AddressSpaceID target, unsigned idx,
                                           RegionTreeContext remote_context,
                                   std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
      assert(idx < mapping_paths.size());
      assert(idx < enclosing_physical_contexts.size());
#endif
      if (locally_mapped[idx])
      {
        runtime->forest->send_back_physical_state(
                                    enclosing_physical_contexts[idx],
                                    remote_context,
                                    mapping_paths[idx],
                                    regions[idx], target,
                                    needed_managers);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::send_back_created_state(AddressSpaceID target, 
                                            unsigned start,
                                            RegionTreeContext remote_outermost,
                                    std::set<PhysicalManager*> &needed_managers)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = start; idx < regions.size(); idx++)
      {
        if (locally_mapped[idx] && !region_deleted[idx])
        {
          RegionTreePath path;
          // Initialize a path to use
          initialize_mapping_path(path, regions[idx], regions[idx].parent);
          runtime->forest->send_back_physical_state(
                                          enclosing_physical_contexts[idx],
                                          remote_outermost, path,
                                          regions[idx], target,
                                          needed_managers); 
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Wrapper Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    WrapperTask::WrapperTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    WrapperTask::~WrapperTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void WrapperTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::defer_mapping(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::perform_mapping(bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::can_early_complete(UserEvent &chain_event)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void WrapperTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void WrapperTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void WrapperTask::perform_inlining(SingleTask *ctx, InlineFnptr fn)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void WrapperTask::handle_future(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void WrapperTask::activate_wrapper(void)
    //--------------------------------------------------------------------------
    {
      activate_single();
    }

    //--------------------------------------------------------------------------
    void WrapperTask::deactivate_wrapper(void)
    //--------------------------------------------------------------------------
    {
      deactivate_single();
    }

    /////////////////////////////////////////////////////////////
    // Remote Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTask::RemoteTask(Runtime *rt)
      : WrapperTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteTask::RemoteTask(const RemoteTask &rhs)
      : WrapperTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteTask::~RemoteTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteTask& RemoteTask::operator=(const RemoteTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_wrapper();
      runtime->allocate_context(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
      runtime->forest->check_context_state(context);
#endif
    }

    //--------------------------------------------------------------------------
    void RemoteTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // Before deactivating the context, clean it out
      for (std::set<LogicalRegion>::const_iterator it = 
            top_level_regions.begin(); it != top_level_regions.end(); it++)
      {
        runtime->forest->invalidate_physical_context(context, *it); 
      }
      top_level_regions.clear();
      deactivate_wrapper();
      // Context is freed in deactivate single
      runtime->free_remote_task(this);
    }
    
    //--------------------------------------------------------------------------
    void RemoteTask::initialize_remote(UniqueID uid)
    //--------------------------------------------------------------------------
    {
      unique_op_id = uid;
    } 

    //--------------------------------------------------------------------------
    void RemoteTask::unpack_parent_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_local;
      derez.deserialize(num_local);
      std::deque<LocalFieldInfo> temp_local(num_local);
      for (unsigned idx = 0; idx < num_local; idx++)
      {
        derez.deserialize(temp_local[idx]);
        allocate_local_field(temp_local[idx]);
      }
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      derez.deserialize(legion_spy_start);
      derez.deserialize(remote_legion_spy_completion);
#endif
      // Now put them on the local fields list, hold the lock
      // while modifying the data structure
      AutoLock o_lock(op_lock);
      for (unsigned idx = 0; idx < num_local; idx++)
        local_fields.push_back(temp_local[idx]);
    }

    //--------------------------------------------------------------------------
    RegionTreeContext RemoteTask::find_enclosing_physical_context(
                                                          LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // This will always contains the virtual context
      return context;
    }

    //--------------------------------------------------------------------------
    RemoteTask* RemoteTask::find_outermost_physical_context(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    Event RemoteTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
#if !defined(LEGION_LOGGING) && !defined(LEGION_SPY)
      // should never be called
      assert(false);
      return Event::NO_EVENT;
#else
      return remote_legion_spy_completion;
#endif
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind RemoteTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return REMOTE_TASK_KIND;
    } 

    //--------------------------------------------------------------------------
    void RemoteTask::find_enclosing_local_fields(
                                              std::deque<LocalFieldInfo> &infos)
    //--------------------------------------------------------------------------
    {
      // No need to go up since we are the uppermost task on this runtime
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    } 

    //--------------------------------------------------------------------------
    void RemoteTask::add_top_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      top_level_regions.insert(handle);
    }

    /////////////////////////////////////////////////////////////
    // Inline Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineTask::InlineTask(Runtime *rt)
      : WrapperTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineTask::InlineTask(const InlineTask &rhs)
      : WrapperTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InlineTask::~InlineTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineTask& InlineTask::operator=(const InlineTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void InlineTask::initialize_inline_task(SingleTask *enc, TaskOp *clone)
    //--------------------------------------------------------------------------
    {
      enclosing = enc;
      indexes = clone->indexes;
      regions = clone->regions;
      physical_regions.resize(regions.size());
      // Now update the parent regions so that they are valid with
      // respect to the outermost context
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        unsigned index = enclosing->find_parent_index_region(idx, this);
        indexes[idx].parent = enclosing->indexes[index].parent;
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        unsigned index = enclosing->find_parent_region(idx, this);
        regions[idx].parent = enclosing->regions[index].parent;
        physical_regions[idx] = enclosing->get_physical_region(index);
      }
    }

    //--------------------------------------------------------------------------
    void InlineTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_wrapper();
      enclosing = NULL;
    }

    //--------------------------------------------------------------------------
    void InlineTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_wrapper();
      runtime->free_inline_task(this);
    }

    //--------------------------------------------------------------------------
    RegionTreeContext InlineTask::get_context(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_context();
    }

    //--------------------------------------------------------------------------
    ContextID InlineTask::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_context_id();
    }

    //--------------------------------------------------------------------------
    RegionTreeContext InlineTask::find_enclosing_physical_context(
                                                          LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      if (created_regions.find(parent) != created_regions.end())
      {
        return find_outermost_physical_context()->get_context();
      }
      return enclosing->find_enclosing_physical_context(parent);
    }

    //--------------------------------------------------------------------------
    RemoteTask* InlineTask::find_outermost_physical_context(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->find_outermost_physical_context();
    }

    //--------------------------------------------------------------------------
    Event InlineTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_task_completion();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind InlineTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_task_kind();
    }

    //--------------------------------------------------------------------------
    void InlineTask::find_enclosing_local_fields(
                                              std::deque<LocalFieldInfo> &infos)
    //--------------------------------------------------------------------------
    {
      enclosing->find_enclosing_local_fields(infos);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_operation(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_executed(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_complete(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_commit(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::unregister_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->unregister_child_operation(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_reclaim_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_reclaim_operation(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_fence_dependence(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::update_current_fence(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      enclosing->update_current_fence(op);
    }

    /////////////////////////////////////////////////////////////
    // Index Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(Runtime *rt)
      : MultiTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(const IndexTask &rhs)
      : MultiTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask& IndexTask::operator=(const IndexTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_multi();
      reduction_op = NULL;
      slice_fraction = Fraction<long long>(0,1); // empty fraction
      total_points = 0;
      mapped_points = 0;
      complete_points = 0;
      committed_points = 0;
      complete_received = false;
      commit_received = false; 
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_multi();
      if (must_parallelism)
        must_barrier.destroy_barrier();
      privilege_paths.clear();
      // Remove our reference to the argument map
      argument_map = ArgumentMap();
      // Remove our reference to the future map
      future_map = FutureMap();
      // Remove our reference to the reduction future
      reduction_future = Future();
      runtime->free_index_task(this);
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(SingleTask *ctx,
                                         const IndexLauncher &launcher,
                                         bool check_privileges,
                                         bool track /*= true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions.resize(launcher.region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].copy_without_mapping_info(
            launcher.region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(launcher.argument_map.impl->freeze());
      map_id = launcher.map_id;
      tag = launcher.tag;
      is_index_space = true;
      must_parallelism = launcher.must_parallelism;
      if (must_parallelism)
        must_barrier = Barrier::create_barrier(1);
      index_domain = launcher.launch_domain;
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      initialize_paths();
      annotate_early_mapped_regions();
      future_map = FutureMap(new FutureMap::Impl(ctx, this, runtime));
#ifdef DEBUG_HIGH_LEVEL
      future_map.impl->add_valid_domain(index_domain);
#endif
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_index_space_task(parent_ctx->get_executing_processor(),
                                          parent_ctx->get_unique_task_id(),
                                          unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_index_task(parent_ctx->get_unique_task_id(),
                                unique_op_id, task_id,
                                variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return future_map;
    }

    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(SingleTask *ctx,
                                      const IndexLauncher &launcher,
                                      ReductionOpID redop_id, 
                                      bool check_privileges,
                                      bool track /*= true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions.resize(launcher.region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].copy_without_mapping_info(
            launcher.region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(launcher.argument_map.impl->freeze());
      map_id = launcher.map_id;
      tag = launcher.tag;
      is_index_space = true;
      must_parallelism = launcher.must_parallelism;
      if (must_parallelism)
        must_barrier = Barrier::create_barrier(1);
      index_domain = launcher.launch_domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      if (!reduction_op->is_foldable)
      {
        log_run(LEVEL_ERROR,"Reduction operation %d for index task launch %s "
                            "(ID %lld) is not foldable.",
                            redop, variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNFOLDABLE_REDUCTION_OP);
      }
      else
        initialize_reduction_state();
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      initialize_paths();
      annotate_early_mapped_regions();
      reduction_future = Future(new Future::Impl(runtime, true/*register*/,
            runtime->get_available_distributed_id(), runtime->address_space,
            runtime->address_space, this));
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_index_space_task(parent_ctx->get_executing_processor(),
                                          parent_ctx->get_unique_task_id(),
                                          unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_index_task(parent_ctx->get_unique_task_id(),
                                unique_op_id, task_id,
                                variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(SingleTask *ctx,
            Processor::TaskFuncID tid,
            const Domain &domain,
            const std::vector<IndexSpaceRequirement> &index_requirements,
            const std::vector<RegionRequirement> &region_requirements,
            const TaskArgument &global_arg,
            const ArgumentMap &arg_map,
            const Predicate &pred,
            bool must,
            MapperID mid, MappingTagID t,
            bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = tid;
      indexes = index_requirements;
      regions.resize(region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].copy_without_mapping_info(region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(arg_map.impl->freeze());
      map_id = mid;
      tag = t;
      is_index_space = true;
      must_parallelism = must;
      if (must_parallelism)
        must_barrier = Barrier::create_barrier(1);
      index_domain = domain;
      initialize_base_task(ctx, true/*track*/, pred, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      initialize_paths();
      annotate_early_mapped_regions();
      future_map = FutureMap(new FutureMap::Impl(ctx, this, runtime));
#ifdef DEBUG_HIGH_LEVEL
      future_map.impl->add_valid_domain(index_domain);
#endif
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_index_space_task(parent_ctx->get_executing_processor(),
                                          parent_ctx->get_unique_task_id(),
                                          unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_index_task(parent_ctx->get_unique_task_id(),
                                unique_op_id, task_id,
                                variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return future_map;
    }
    
    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(SingleTask *ctx,
            Processor::TaskFuncID tid,
            const Domain &domain,
            const std::vector<IndexSpaceRequirement> &index_requirements,
            const std::vector<RegionRequirement> &region_requirements,
            const TaskArgument &global_arg,
            const ArgumentMap &arg_map,
            ReductionOpID redop_id,
            const TaskArgument &init_value,
            const Predicate &pred,
            bool must,
            MapperID mid, MappingTagID t,
            bool check_privileges)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = tid;
      indexes = index_requirements;
      regions.resize(region_requirements.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].copy_without_mapping_info(region_requirements[idx]);
        regions[idx].initialize_mapping_fields();
      }
      if (parent_ctx->has_simultaneous_coherence())
        parent_ctx->check_simultaneous_restricted(regions);
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(arg_map.impl->freeze());
      map_id = mid;
      tag = t;
      is_index_space = true;
      must_parallelism = must;
        if (must_parallelism)
        must_barrier = Barrier::create_barrier(1);
      index_domain = domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      if (!reduction_op->is_foldable)
      {
        log_run(LEVEL_ERROR,"Reduction operation %d for index task launch %s "
                            "(ID %lld) is not foldable.",
                            redop, variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNFOLDABLE_REDUCTION_OP);
      }
      else
        initialize_reduction_state();
      initialize_base_task(ctx, true/*track*/, pred, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_physical_contexts();
      initialize_paths();
      annotate_early_mapped_regions();
      reduction_future = Future(new Future::Impl(runtime, true/*register*/,
            runtime->get_available_distributed_id(), runtime->address_space,
            runtime->address_space, this));
      check_empty_field_requirements();
#ifdef LEGION_LOGGING
      LegionLogging::log_index_space_task(parent_ctx->get_executing_processor(),
                                          parent_ctx->get_unique_task_id(),
                                          unique_op_id, task_id, tag);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_index_task(parent_ctx->get_unique_task_id(),
                                unique_op_id, task_id,
                                variants->name);
#endif
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
      }
#endif
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_paths(void)
    //--------------------------------------------------------------------------
    {
      privilege_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        initialize_privilege_path(privilege_paths[idx], regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::annotate_early_mapped_regions(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_WRITE(regions[idx]) && (regions[idx].handle_type == SINGULAR))
          regions[idx].must_early_map = true;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(privilege_paths.size() == regions.size());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(), 
                                      BEGIN_DEPENDENCE_ANALYSIS);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_BEGIN_DEP_ANALYSIS);
#endif
      begin_dependence_analysis();
      RegionTreeContext ctx = parent_ctx->get_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(ctx, this, idx, 
                                    regions[idx], privilege_paths[idx]);
      }
      end_dependence_analysis();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_DEPENDENCE_ANALYSIS);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), PROF_END_DEP_ANALYSIS);
#endif
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_aliased_requirements(unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 0
      log_run(LEVEL_ERROR,"Aliased region requirements for index tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are aliased.", idx1, idx2, variants->name,
                          get_unique_task_id(), parent_ctx->variants->name,
                          parent_ctx->get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_ALIASED_REGION_REQUIREMENTS);
#else
      log_run(LEVEL_WARNING,"Region requirements %d and %d of individual task "
                            "%s (UID %lld) in parent task %s (UID %lld) are "
                            "aliased.  This behavior is currently undefined. "
                            "You better really know what you are doing.",
                            idx1, idx2, variants->name, get_unique_task_id(),
                            parent_ctx->variants->name, 
                            parent_ctx->get_unique_task_id());
#endif
    }

    //--------------------------------------------------------------------------
    bool IndexTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      if (premapped)
        return true;
      premapped = true;
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing_physical_contexts.size() == regions.size());
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      BEGIN_PRE_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(), 
                                 PROF_BEGIN_PREMAP_ANALYSIS);
#endif
      // All regions need to be premapped no matter what
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Check to see if we already premapped this region
        // If not then we need to do it now
        if (!regions[idx].premapped)
        {
          if (!runtime->forest->premap_physical_region(
                                       enclosing_physical_contexts[idx],
                                       privilege_paths[idx], regions[idx], 
                                       this, parent_ctx,
                                       parent_ctx->get_executing_processor()
#ifdef DEBUG_HIGH_LEVEL
                                       , idx, get_logging_name(), unique_op_id
#endif
                                       ))
          {
            // Failed to premap, break out and try again later
            premapped = false;
            break;
          }
          else
          {
            regions[idx].premapped = true;
          }
        }
      }
      if (premapped)
        premapped = early_map_regions();
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      get_unique_task_id(),
                                      END_PRE_MAPPING);
#endif
#ifdef LEGION_PROF
      LegionProf::register_event(get_unique_task_id(),
                                 PROF_END_PREMAP_ANALYSIS);
#endif
      return premapped;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called since index tasks are never stealable
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::defer_mapping(void)
    //--------------------------------------------------------------------------
    {
      // never need to defer an index task mapping since it never goes anywhere
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_locally_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
#ifdef DEBUG_HIGH_LEVEL
        assert(slices.empty());
#endif
        // We're never actually run
        return false;
      }
      else
      {
        if (!is_sliced() && (target_proc != current_proc))
        {
          // Make a slice copy and send it away
          SliceTask *clone = clone_as_slice_task(index_domain, target_proc,
                                                 true/*needs slice*/,
                                                 spawn_task, 1LL);
          runtime->send_task(target_proc, clone);
          return false; // We have now been sent away
        }
        else
          return true; // Still local so we can be sliced
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
      // This will only get called if we had slices that failed to map locally
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      bool map_success = true;
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        bool slice_success = (*it)->trigger_execution();
        if (!slice_success)
        {
          // Didn't succeed, leave it on the list for next time
          map_success = false;
          it++;
        }
        else
        {
          // Succeeded, so take it off the list
          it = slices.erase(it);
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // Index space tasks are never stealable, they must first be
      // split into slices which can then be stolen.  Note that slicing
      // always happens after premapping so we know stealing is safe.
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // This should only ever be called if we had slices which failed to map
#ifdef DEBUG_HIGH_LEVEL
      assert(is_sliced());
      assert(!slices.empty());
#endif
      return trigger_slices();
    }

    //--------------------------------------------------------------------------
    Event IndexTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      return get_completion_event();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndexTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDEX_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      unique_op_id, COMPLETE_OPERATION);
      LegionLogging::log_operation_events(
                                  Machine::get_executing_processor(),
                                  get_unique_task_id(),
                                  Event::NO_EVENT, get_task_completion());
#endif

      // Return back our privileges
      return_privilege_state(parent_ctx);

      // Trigger all the futures or set the reduction future result
      // and then trigger it
      if (redop != 0)
      {
        reduction_future.impl->set_result(reduction_state,
                                          reduction_state_size, 
                                          false/*owner*/);
        reduction_future.impl->complete_future();
      }
      else
        future_map.impl->complete_all_futures();

      complete_operation();
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Mark that this operation is now committed
      commit_operation();
      // Now we get to deactivate this task
      deactivate();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_inlining(SingleTask *ctx, InlineFnptr fn)
    //--------------------------------------------------------------------------
    {
      // must parallelism not allowed to be inlined
      if (must_parallelism)
      {
        log_task(LEVEL_ERROR,"Illegal attempt to inline must-parallelism "
                             "task %s (ID %lld)",
                             variants->name, get_unique_task_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_MUST_PARALLEL_INLINE);
      }
      // See if there is anything to wait for
      std::set<Event> wait_on_events;
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        Future::Impl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->ready_event);
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        Grant::Impl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      Event start_condition = Event::merge_events(wait_on_events); 

      // See if we need to wait for anything
      if (start_condition.exists() && !start_condition.has_triggered())
      {
        Processor proc = ctx->get_executing_processor();
        runtime->pre_wait(proc);
        start_condition.wait();
        runtime->post_wait(proc);
      }

      // Enumerate all of the points of our index space and run
      // the task for each one of them either saving or reducing their futures
      const std::vector<PhysicalRegion> &phy_regions = 
                                                  ctx->get_physical_regions();
      for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
      {
        index_point = itr.p; 
        compute_point_region_requirements();
        // Get our local args
        TaskArgument local = argument_map.impl->get_point(index_point);
        local_args = local.get_ptr();
        local_arglen = local.get_size();
        void *result;
        size_t result_size;
        (*fn)(this, phy_regions, ctx, runtime->high_level, result, result_size);
        if (redop == 0)
        {
          Future f = future_map.impl->get_future(index_point);
          f.impl->set_result(result,result_size,true/*owner*/);
        }
        else
          fold_reduction_future(result, result_size, 
                                true/*owner*/, true/*exclusive*/);
      }
      if (redop == 0)
        future_map.impl->complete_all_futures();
      else
      {
        reduction_future.impl->set_result(reduction_state,
                                          reduction_state_size,false/*owner*/);
        reduction_future.impl->complete_future();
      }

      // Trigger all our events event
      completion_event.trigger();
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(const Domain &d, Processor p,
                                              bool recurse, bool stealable,
                                              long long scale_denominator)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx, 
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_multi_from(this, d, p, recurse, stealable);
      result->enclosing_physical_contexts = this->enclosing_physical_contexts;
      result->remote_contexts = this->enclosing_physical_contexts;
      result->remote_outermost_context = 
        parent_ctx->find_outermost_physical_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(result->remote_outermost_context.exists());
#endif
      result->index_complete = this->completion_event;
      result->denominator = scale_denominator;
      result->index_owner = this;
#ifdef LEGION_LOGGING
      LegionLogging::log_index_slice(Machine::get_executing_processor(),
                                     unique_op_id, result->get_unique_op_id());
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(result->task_id, result->get_unique_task_id(),
                                result->index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_index_slice(get_unique_task_id(), 
                                 result->get_unique_task_id());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_future(const DomainPoint &point, const void *result,
                                  size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock when doing this since it could
      // be going in parallel with other users
      if (reduction_op != NULL)
        fold_reduction_future(result, result_size, owner, false/*exclusive*/);
      else
      {
        if (must_epoch == NULL)
        {
          Future f = future_map.get_future(point);
          f.impl->set_result(result, result_size, owner);
        }
        else
          must_epoch->set_future(point, result, result_size, owner);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_mapped(unsigned points, long long denom)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        total_points += points;
        mapped_points += points;
        slice_fraction.add(Fraction<long long>(1,denom));
        // Already know that mapped points is the same as total points
        if (slice_fraction.is_whole())
          need_trigger = true;
      }
      if (need_trigger)
      {
        complete_mapping();
        complete_execution();
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(unsigned points)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        complete_points += points;
#ifdef DEBUG_HIGH_LEVEL
        assert(!complete_received);
        assert(complete_points <= total_points);
#endif
        if (slice_fraction.is_whole() && 
            (complete_points == total_points) &&
            !children_complete_invoked)
        {
          need_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_commit(unsigned points)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        committed_points += points;
#ifdef DEBUG_HIGH_LEVEL
        assert(committed_points <= total_points);
#endif
        if (slice_fraction.is_whole() &&
            (committed_points == total_points) && 
            !children_commit_invoked)
        {
          need_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_committed();
    } 

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      long long denom;
      derez.deserialize(denom);
      return_slice_mapped(points, denom);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      // Hold the lock when unpacking the privileges
      {
        AutoLock o_lock(op_lock);
        unpack_privilege_state(derez);
      }
      if (redop != 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(reduction_op != NULL);
        assert(reduction_state_size == reduction_op->sizeof_rhs);
#endif
        const void *reduc_ptr = derez.get_current_pointer();
        fold_reduction_future(reduc_ptr, reduction_state_size,
                              false /*owner*/, false/*exclusive*/);
        // Advance the pointer on the deserializer
        derez.advance_pointer(reduction_state_size);
      }
      else
      {
        for (unsigned idx = 0; idx < points; idx++)
        {
          DomainPoint p;
          unpack_point(derez, p);
          if (must_epoch == NULL)
          {
            Future f = future_map.impl->get_future(p);
            f.impl->unpack_future(derez);
          }
          else
            must_epoch->unpack_future(p, derez);
        }
      }
      return_slice_complete(points);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      return_slice_commit(points);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_mapped(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    void IndexTask::send_remote_state(AddressSpaceID target, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      // Send state for anything that was not premapped and has not
      // already been mapped
      std::map<LogicalView*,FieldMask> needed_views;
      std::set<PhysicalManager*> needed_managers;
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing_physical_contexts.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        runtime->forest->send_tree_shape(indexes[idx], target);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Can skip any early mapped regions
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
        {
          // Anything packing up a reference will send its own
          // region tree shape
          continue;
        }
        // Otherwise we need to send the state
        runtime->forest->send_physical_state(enclosing_physical_contexts[idx],
                                             regions[idx],
                                             uid, target,
                                             needed_views,
                                             needed_managers);
      }
      // If we had any needed views or needed managers send 
      // their remote references
      if (!needed_views.empty() || !needed_managers.empty())
        runtime->forest->send_remote_references(needed_views,
                                                needed_managers, target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::handle_slice_request(Runtime *rt, 
                                                    Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *owner_task;
      derez.deserialize(owner_task);
      SliceTask *remote_slice;
      derez.deserialize(remote_slice);
      UniqueID uid;
      derez.deserialize(uid);
      // Send the state
      owner_task->send_remote_state(source, uid);
      // Send the message back saying the state is sent
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(remote_slice);
      }
      rt->send_slice_return(source, rez);
    }

    /////////////////////////////////////////////////////////////
    // Slice Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(Runtime *rt)
      : MultiTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(const SliceTask &rhs)
      : MultiTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask& SliceTask::operator=(const SliceTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void SliceTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_multi();
      reclaim = false;
      index_complete = Event::NO_EVENT;
      mapping_index = 0;
      num_unmapped_points = 0;
      num_uncomplete_points = 0;
      num_uncommitted_points = 0;
      denominator = 0;
      index_owner = NULL;
      remote_unique_id = get_unique_task_id();
      locally_mapped = false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // If we're remote, then deactivate our parent context since
      // it is a remote context
      if (is_remote())
        parent_ctx->deactivate();
      deactivate_multi();
      points.clear();
      for (std::map<DomainPoint,std::pair<void*,size_t>,
            DomainPoint::STLComparator>::const_iterator it = 
            temporary_futures.begin(); it != temporary_futures.end(); it++)
      {
        free(it->second.first);
      }
      temporary_futures.clear();
      remote_contexts.clear();
      runtime->free_slice_task(this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing. We've already been sanitized by our index task owner.
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_locally_mapped());
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::defer_mapping(void)
    //--------------------------------------------------------------------------
    {
      // If we are a stealable task and we are remote then we need
      // to request the information necessary to map
      if (needs_state)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_remote());
#endif
        // Mark that this task is no longer stealable
        needs_state = false;
        // Send a message to the original processor requesting
        // the mapping infomration
        Serializer rez;
        {
          RezCheck z(rez); 
          rez.serialize(index_owner);
          SliceTask *proxy_this = this;
          rez.serialize(proxy_this);
          rez.serialize(remote_unique_id);
        }
        runtime->send_individual_request(
            runtime->find_address_space(orig_proc), rez);
        return true;
      }
      // No need to defer otherwise
      return false;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Quick out in case this slice task is to be reclaimed
      if (reclaim)
        return true;
      if (target_proc != current_proc)
      {
        runtime->send_task(target_proc,this);
        // The runtime will deactivate this task
        // after it has been sent
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::perform_mapping(bool mapper_invoked)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // If slices are empty, this is a leaf slice so we can do the
      // normal mapping procedure
      if (slices.empty())
      {
        // Check to see if we already enumerated all the points, if
        // not then do so now
        if (points.empty())
          enumerate_points();

        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          bool point_success = points[idx]->perform_mapping(mapper_invoked);
          if (!point_success)
          {
            // Failed to map, so unmap everything up to this point
            for (unsigned i = 0; i < idx; i++)
              points[i]->unmap_all_regions();
            map_success = false;
            // Reset the number of unmapped points
            num_unmapped_points = points.size();
            break;
          }
        }

        // Check to see if we all mapped then we are ready to 
        // trigger that all the operations have mapped
        if (map_success)
        {
          // If we succeeded in mapping we are no longer stealable
          spawn_task = false;
        }
      }
      else
      {
        // This case only occurs if this is an intermediate slice and
        // its sub-slices failed to map, so try to remap them.
        for (std::list<SliceTask*>::iterator it = slices.begin();
              it != slices.end(); /*nothing*/)
        {
          bool slice_success = (*it)->trigger_execution();
          if (!slice_success)
          {
            map_success = false;
            it++;
          }
          else
          {
            // Remote it from the list since it succeeded
            it = slices.erase(it);
          }
        }
        // If we succeeded in mapping all our remaining
        // slices then mark that this task can be reclaimed
        if (map_success)
          reclaim = true;
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      // Quick out in case we are reclaiming this task
      if (reclaim)
      {
        deactivate();
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(!points.empty());
#endif
      // Launch all of our child points
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->launch_task();
      // If we're doing must parallelism, mark that everyone is ready to run
      if (must_parallelism)
        must_barrier.arrive();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_locally) && spawn_task);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // Mark that this task is no longer stealable.  Once we start
      // executing things onto a specific processor slices cannot move.
      spawn_task = false;
      // First enumerate all of our points if we haven't already done so
      if (points.empty())
        enumerate_points();
#ifdef DEBUG_HIGH_LEVEL
      assert(!points.empty());
      assert(mapping_index <= points.size());
#endif
      // Watch out for the race condition here of all the points
      // finishing and cleaning up this context before we are done
      // Pull everything onto the stack that we need.
      bool has_barrier = must_parallelism;
      Barrier must_bar;
      if (must_parallelism)
        must_bar = must_barrier;

      // Now try mapping and then launching all the points starting
      // at the index of the last known good index
      // Copy the points onto the stack to avoid them being
      // cleaned up while we are still iterating through the loop
      std::vector<PointTask*> local_points(points.size()-mapping_index);
      for (unsigned idx = mapping_index; idx < points.size(); idx++)
        local_points[idx-mapping_index] = points[idx];
      for (std::vector<PointTask*>::const_iterator it = local_points.begin();
            it != local_points.end(); it++)
      {
        PointTask *next_point = *it;
        bool point_success = next_point->perform_mapping();
        if (!point_success)
        {
          map_success = false;    
          break;
        }
        else
        {
          // Otherwise update the mapping index and then launch
          // the point (it is imperative that these happen in this order!)
          mapping_index++;
          // Once we call this function on the last point it
          // is possible that this slice task object can be recycled
          next_point->launch_task();
        }
      }

      if (map_success)
      {
        // Trigger the must barrier once everyone is launched
        if (has_barrier)
          must_bar.arrive();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    Event SliceTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      return index_complete;
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind SliceTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return SLICE_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are stealable or not yet fully sliced,
      // if both are false and we're not remote, then we can send the state
      // now or check to see if we are remotely mapped
      AddressSpaceID addr_target = runtime->find_address_space(target);
      if (!spawn_task && sliced && !is_remote())
      {
        needs_state = false;
        if (points.empty() || (num_unmapped_points > 0))
        {
          // Just send the state now
          index_owner->send_remote_state(addr_target, remote_unique_id);
          locally_mapped = false;
        }
        else
        {
          // Otherwise we are locally mapped so there is 
          // no need to send the state 
          locally_mapped = true;
          // We do still need to send the tree shapes though
          for (unsigned idx = 0; idx < indexes.size(); idx++)
          {
            runtime->forest->send_tree_shape(indexes[idx], addr_target);
          }
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            runtime->forest->send_tree_shape(regions[idx], addr_target);
          }
        }
      }
      else
        needs_state = true;
      RezCheck z(rez);
      // Preamble used in TaskOp::unpack
      rez.serialize(points.size());
      pack_multi_task(rez, points.empty(), addr_target);
      rez.serialize(denominator);
      rez.serialize(index_owner);
      rez.serialize(index_complete);
      rez.serialize(remote_unique_id);
      rez.serialize(remote_outermost_context);
      rez.serialize(locally_mapped);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_contexts.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        rez.serialize(remote_contexts[idx]);
      }
      parent_ctx->pack_parent_task(rez);
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->pack_task(rez, target);
        // Once we're done packing the task we can deactivate it
        points[idx]->deactivate();
      }
      // Always return true for slice tasks since they should
      // always be deactivated after they are sent somewhere else
      return true;
    }
    
    //--------------------------------------------------------------------------
    bool SliceTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez, (num_points==0));
      current_proc = current;
      derez.deserialize(denominator);
      derez.deserialize(index_owner);
      derez.deserialize(index_complete);
      derez.deserialize(remote_unique_id); 
      derez.deserialize(remote_outermost_context);
      derez.deserialize(locally_mapped);
      remote_contexts.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        derez.deserialize(remote_contexts[idx]);
      RemoteTask *remote_ctx = 
        runtime->find_or_init_remote_context(remote_unique_id);
      remote_ctx->unpack_parent_task(derez);
      // Add our parent regions to the list of top regions
      for (unsigned idx = 0; idx < regions.size(); idx++)
        remote_ctx->add_top_region(regions[idx].parent);
      // Quick check to see if we ended up back on the original node
      if (!is_remote())
      {
        // Otherwise we can deactivate the remote ctx and use
        // our original parent context
        remote_ctx->deactivate();
        parent_ctx = index_owner->parent_ctx;
        // If this happens we already have our state
        needs_state = false;
        // We also have our enclosing contexts
        enclosing_physical_contexts = index_owner->enclosing_physical_contexts;
      }
      else
      {
        parent_ctx = remote_ctx;
        enclosing_physical_contexts.resize(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
          enclosing_physical_contexts[idx] = remote_ctx->get_context();
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_slice_slice(Machine::get_executing_processor(),
                                     remote_unique_id, get_unique_task_id());
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, get_unique_task_id(), index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_slice_slice(remote_unique_id, get_unique_task_id());
#endif
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = runtime->get_available_point_task(); 
        point->slice_owner = this;
        point->unpack_task(derez, current);
        point->parent_ctx = parent_ctx;
        point->enclosing_physical_contexts = enclosing_physical_contexts;
        points.push_back(point);
#ifdef LEGION_LOGGING
        LegionLogging::log_slice_point(Machine::get_executing_processor(),
                                       get_unique_task_id(),
                                       point->get_unique_task_id(),
                                       point->index_point);
#endif
#ifdef LEGION_PROF
        LegionProf::register_task(task_id, point->get_unique_task_id(),
                                  point->index_point);
#endif
#ifdef LEGION_SPY
        LegionSpy::log_slice_point(get_unique_task_id(), 
                                   point->get_unique_task_id(),
                                   point->index_point);
#endif
      }
      // Return true to add this to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_inlining(SingleTask *ctx, InlineFnptr fn)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SliceTask* SliceTask::clone_as_slice_task(const Domain &d, Processor p,
                                              bool recurse, bool stealable,
                                              long long scale_denominator)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx, 
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_multi_from(this, d, p, recurse, stealable);
      result->enclosing_physical_contexts = this->enclosing_physical_contexts;
      result->remote_contexts = this->remote_contexts;
      result->remote_outermost_context = this->remote_outermost_context;
      result->index_complete = this->index_complete;
      result->denominator = this->denominator * scale_denominator;
      result->index_owner = this->index_owner;
#ifdef LEGION_LOGGING
      LegionLogging::log_slice_slice(Machine::get_executing_processor(),
                                     unique_op_id, result->get_unique_op_id());
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(result->task_id, result->get_unique_task_id(),
                                result->index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_slice_slice(get_unique_task_id(), 
                                 result->get_unique_task_id());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const DomainPoint &point, const void *result,
                                  size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      // If we're remote, just handle it ourselves, otherwise pass
      // it back to the enclosing index owner
      if (is_remote())
      {
        if (redop != 0)
          fold_reduction_future(result, result_size, owner, false/*exclusive*/);
        else
        {
          // Store it in our temporary futures
#ifdef DEBUG_HIGH_LEVEL
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          if (owner)
            temporary_futures[point] = 
              std::pair<void*,size_t>(const_cast<void*>(result),result_size);
          else
          {
            void *copy = malloc(result_size);
            memcpy(copy,result,result_size);
            temporary_futures[point] = 
              std::pair<void*,size_t>(copy,result_size);
          }
        }
      }
      else
        index_owner->handle_future(point, result, result_size, owner);
    }

    //--------------------------------------------------------------------------
    void SliceTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(must_epoch != NULL);
#endif
      if (points.empty())
        enumerate_points();
      must_epoch->register_slice_task(this);
      for (unsigned idx = 0; idx < points.size(); idx++)
        must_epoch->register_single_task(points[idx], must_epoch_index);
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(const DomainPoint &p)
    //--------------------------------------------------------------------------
    {
      PointTask *result = runtime->get_available_point_task();
      result->initialize_base_task(parent_ctx,
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_task_op_from(this, this->target_proc, 
                                 false/*stealable*/, true/*duplicate*/);
      result->enclosing_physical_contexts = this->enclosing_physical_contexts;
      result->is_index_space = true;
      result->must_parallelism = this->must_parallelism;
      result->index_domain = this->index_domain;
      result->index_point = p;
      // Now figure out our local point information
      result->initialize_point(this);
      // Get our local arguments from the argument map
      // Note we don't need to copy it since the arugment
      // map will live as long as the slice task which is
      // as long or longer than the lifetime of a point task
      TaskArgument arg = argument_map.impl->get_point(p);
      result->local_arglen = arg.get_size();
      if (result->local_arglen > 0)
      {
        result->local_args = malloc(result->local_arglen);
        memcpy(result->local_args,arg.get_ptr(),
                result->local_arglen);
      }
#ifdef LEGION_LOGGING
      LegionLogging::log_slice_point(Machine::get_executing_processor(),
                                     unique_op_id,
                                     result->get_unique_op_id(),
                                     result->index_point);
#endif
#ifdef LEGION_PROF
      LegionProf::register_task(task_id, 
                                result->get_unique_task_id(), 
                                result->index_point);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_slice_point(get_unique_task_id(), 
                                 result->get_unique_task_id(),
                                 result->index_point);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      // Before we enumerate the points, ask the mapper to pick the
      // task variant to be used for all of these points so when
      // we clone each of the point tasks, they will get the right
      // variant information to run on this processor
      runtime->invoke_mapper_select_variant(current_proc, this);
#ifdef DEBUG_HIGH_LEVEL
      if (!variants->has_variant(selected_variant))
      {
        log_task(LEVEL_ERROR,"Invalid task variant %ld selected by mapper "
                             "for task %s (ID %lld)", selected_variant,
                             variants->name, get_unique_task_id());
        assert(false);
        exit(ERROR_INVALID_VARIANT_SELECTION);
      }
#endif
#ifdef LEGION_LOGGING
      LegionLogging::log_task_instance_variant(
                                  Machine::get_executing_processor(),
                                  get_unique_task_id(),
                                  selected_variant);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(index_domain.get_volume() > 0);
#endif
      // Enumerate all the points
      for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
      {
        PointTask *next_point = clone_as_point_task(itr.p);
        points.push_back(next_point);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(index_domain.get_volume() == points.size());
#endif
      mapping_index = 0;
      // Mark how many points we have
      num_unmapped_points = points.size();
      num_uncomplete_points = points.size();
      num_uncommitted_points = points.size();
    } 

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_LOGGING
      LegionLogging::log_timing_event(Machine::get_executing_processor(),
                                      unique_op_id, COMPLETE_OPERATION);
#endif
      trigger_slice_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      trigger_slice_commit();
    }

    //--------------------------------------------------------------------------
    void SliceTask::return_privileges(PointTask *point)
    //--------------------------------------------------------------------------
    {
      // If we're remote, pass our privileges back to ourself
      // otherwise pass them directly back to the index owner
      if (is_remote())
        point->return_privilege_state(this);
      else
        point->return_privilege_state(index_owner);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_child_mapped(void)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(num_unmapped_points > 0);
#endif
        num_unmapped_points--;
        if (num_unmapped_points == 0)
          needs_trigger = true;
      }
      if (needs_trigger)
        trigger_slice_mapped();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_child_complete(void)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(num_uncomplete_points > 0);
#endif
        num_uncomplete_points--;
        if ((num_uncomplete_points == 0) && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_child_committed(void)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(num_uncommitted_points > 0);
#endif
        num_uncommitted_points--;
        if ((num_uncommitted_points == 0) && !children_commit_invoked)
        {
          needs_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_mapped(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        // Only need to send something back if this wasn't mapped locally
        if (!is_locally_mapped())
        {
          Serializer rez;
          pack_remote_mapped(rez);
          runtime->send_slice_remote_mapped(orig_proc, rez);
        }
      }
      else
      {
        index_owner->return_slice_mapped(points.size(), denominator);
      }
      complete_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_complete(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        // Send back the message saying that this slice is complete
        Serializer rez;
        pack_remote_complete(rez);
        runtime->send_slice_remote_complete(orig_proc, rez);
      }
      else
      {
        index_owner->return_slice_complete(points.size());
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_commit(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez);
        runtime->send_slice_remote_commit(orig_proc, rez);
      }
      else
      {
        // created and deleted privilege information already passed back

        // futures already sent back

        index_owner->return_slice_commit(points.size());
      }
      commit_operation();
      // After we're done with this, then we can reclaim oursleves
      deactivate();
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_mapped(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Send back any state owned by this slice.  For read-only and reduce
      // requirements, we can just send back the entire state from this slice.
      // For any read-write requirements that were not early mapped they have
      // to be projection in which case we ask each of the children to send
      // them back individually.
      if (!locally_mapped)
      {
        AddressSpaceID target = runtime->find_address_space(orig_proc);
        std::set<PhysicalManager*> needed_managers;
        for (unsigned idx = 0; idx < remote_contexts.size(); idx++)
        {
          if (early_mapped_regions.find(idx) != early_mapped_regions.end())
            continue;
          if (IS_WRITE(regions[idx]))
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((regions[idx].handle_type == PART_PROJECTION) ||
                    (regions[idx].handle_type == REG_PROJECTION));
            assert(idx < remote_contexts.size());
#endif
            // Ask each of the points to send back their remote state
            for (std::deque<PointTask*>::const_iterator it = 
                  points.begin(); it != points.end(); it++)
            {
              (*it)->send_back_remote_state(target, idx, 
                  remote_contexts[idx], needed_managers);
            }
          }
          else
          {
            // Since this state is going to get merged we can send
            // the whole state back from this region requirement
            RegionTreePath path;
            if (regions[idx].handle_type == PART_PROJECTION)
            {
              runtime->forest->initialize_path(
                  regions[idx].partition.get_index_partition(),
                  regions[idx].partition.get_index_partition(),
                                                         path);
            }
            else
            {
              runtime->forest->initialize_path(
                  regions[idx].region.get_index_space(),
                  regions[idx].region.get_index_space(),
                                                   path);
            }
            runtime->forest->send_back_physical_state(
                                      enclosing_physical_contexts[idx],
                                      remote_contexts[idx],
                                      path, regions[idx], target,
                                      needed_managers);
          }
        }
        if (!needed_managers.empty())
          runtime->forest->send_remote_references(needed_managers, target);
      }
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(denominator);
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_complete(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Send back any created state that our point tasks made
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      std::set<PhysicalManager*> needed_managers;
      for (std::deque<PointTask*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
      {
        (*it)->send_back_created_state(target, remote_contexts.size(),
                             remote_outermost_context, needed_managers);
      }
      if (!needed_managers.empty())
        runtime->forest->send_remote_references(needed_managers, target);
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      // Serialize the privilege state
      pack_privilege_state(rez, target); 
      // Now pack up the future results
      if (redop != 0)
      {
        // Don't need to pack the size since they already 
        // know it on the other side
        rez.serialize(reduction_state,reduction_state_size);
      }
      else
      {
        // Already know how many futures we are packing 
#ifdef DEBUG_HIGH_LEVEL
        assert(temporary_futures.size() == points.size());
#endif
        for (std::map<DomainPoint,std::pair<void*,size_t>,
              DomainPoint::STLComparator>::const_iterator it =
              temporary_futures.begin(); it != temporary_futures.end(); it++)
        {
          pack_point(rez, it->first);
          RezCheck z2(rez);
          rez.serialize(it->second.second);
          rez.serialize(it->second.first,it->second.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_commit(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_slice_return(Runtime *rt, 
                                                   Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SliceTask *slice_task;
      derez.deserialize(slice_task);
      rt->add_to_ready_queue(slice_task->current_proc, slice_task,
                             false/*prev fail*/);
    }

    /////////////////////////////////////////////////////////////
    // Slice Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredSlicer::DeferredSlicer(MultiTask *own)
      : owner(own)
    //--------------------------------------------------------------------------
    {
      slice_lock = Reservation::create_reservation();
    }

    //--------------------------------------------------------------------------
    DeferredSlicer::DeferredSlicer(const DeferredSlicer &rhs)
      : owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeferredSlicer::~DeferredSlicer(void)
    //--------------------------------------------------------------------------
    {
      slice_lock.destroy_reservation();
      slice_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    DeferredSlicer& DeferredSlicer::operator=(const DeferredSlicer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool DeferredSlicer::trigger_slices(std::list<SliceTask*> &slices)
    //--------------------------------------------------------------------------
    {
      // Watch out for the cleanup race with some acrobatics here
      // to handle the case where the iterator is invalidated
      std::set<Event> wait_events;
      {
        Processor exec_proc = Machine::get_executing_processor();
        Processor util_proc = exec_proc.get_utility_processor();
        std::list<SliceTask*>::const_iterator it = slices.begin();
        DeferredSliceArgs args;
        args.hlr_id = HLR_DEFERRED_SLICE_ID;
        args.slicer = this;
        while (true) 
        {
          args.slice = *it;
          it++;
          bool done = (it == slices.end()); 
          Event wait = util_proc.spawn(HLR_TASK_ID, &args, sizeof(args)); 
          if (wait.exists())
            wait_events.insert(wait);
          if (done)
            break;
        }
      }

      // Now we wait for the slices to trigger, note we do not
      // block on the event allowing the utility processor to 
      // perform other operations
      if (!wait_events.empty())
      {
        Event sliced_event = Event::merge_events(wait_events);
        sliced_event.wait(false/*block*/);
      }

      bool success = failed_slices.empty();
      // If there were some slices that didn't succeed, then we
      // need to clean up the ones that did so we know when
      // which ones to re-trigger when we try again later. Otherwise
      // if all the slices succeeded, then the normal deactivation of
      // this task will clean up the slices. Note that if we have
      // at least one slice that didn't succeed then we know this
      // task cannot be deactivated.
      if (!success)
      {
        for (std::list<SliceTask*>::iterator it = slices.begin();
              it != slices.end(); /*nothing*/)
        {
          if (failed_slices.find(*it) == failed_slices.end())
            it = slices.erase(it);
          else
            it++;
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    void DeferredSlicer::perform_slice(SliceTask *slice)
    //--------------------------------------------------------------------------
    {
      if (!slice->trigger_execution())
      {
        AutoLock s_lock(slice_lock);
        failed_slices.insert(slice);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void DeferredSlicer::handle_slice(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredSliceArgs *slice_args = (const DeferredSliceArgs*)args;
      slice_args->slicer->perform_slice(slice_args->slice);
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

#undef PRINT_REG

// EOF

