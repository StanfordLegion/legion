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


#include "region_tree.h"
#include "legion_tasks.h"
#include "legion_spy.h"
#include "legion_trace.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"
#include <algorithm>

#define PRINT_REG(reg) (reg).index_space.id,(reg).field_space.id, (reg).tree_id

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

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
    UniqueID TaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    int TaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_remote(void) const
    //--------------------------------------------------------------------------
    {
      if (local_cached)
        return !is_local;
      if (!orig_proc.exists())
        is_local = runtime->is_local(parent_ctx->get_executing_processor());
      else
        is_local = runtime->is_local(orig_proc);
      local_cached = true;
      return !is_local;
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_current_proc(Processor current)
    //--------------------------------------------------------------------------
    {
      // Always clear target_proc and the mapper when setting a new current proc
      target_proc = Processor::NO_PROC;
      mapper = NULL;
      current_proc = current;
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
      stealable = false;
      map_locally = false;
      local_cached = false;
      arg_manager = NULL;
      target_proc = Processor::NO_PROC;
      mapper = NULL;
      orig_proc = Processor::NO_PROC; // for is_remote
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
            legion_delete(arg_manager);
          arg_manager = NULL;
        }
        else
          legion_free(TASK_ARGS_ALLOC, args, arglen);
        args = NULL;
        arglen = 0;
      }
      if (local_args != NULL)
      {
        legion_free(LOCAL_ARGS_ALLOC, local_args, local_arglen);
        local_args = NULL;
        local_arglen = 0;
      }
      early_mapped_regions.clear();
      atomic_locks.clear();
      created_regions.clear();
      created_fields.clear();
      created_field_spaces.clear();
      created_index_spaces.clear();
      deleted_regions.clear();
      deleted_fields.clear();
      deleted_field_spaces.clear();
      deleted_index_spaces.clear();
      parent_req_indexes.clear();
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
        futures[idx].impl->send_future(target);
        rez.serialize(futures[idx].impl->did);
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
      // No need to pack remote, it will get set
      rez.serialize(speculated);
      rez.serialize(map_locally);
      if (map_locally)
      {
        rez.serialize<size_t>(atomic_locks.size());
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      // Can't be sending inline tasks remotely
      rez.serialize(early_mapped_regions.size());
      for (std::map<unsigned,InstanceSet>::iterator it = 
            early_mapped_regions.begin(); it != 
            early_mapped_regions.end(); it++)
      {
        rez.serialize(it->first);
        it->second.pack_references(rez, target);
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
          arg_manager = legion_new<AllocManager>(arglen);
          arg_manager->add_reference();
          args = arg_manager->get_allocation();
        }
        else
          args = legion_malloc(TASK_ARGS_ALLOC, arglen);
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
        local_args = legion_malloc(LOCAL_ARGS_ALLOC, local_arglen);
        derez.deserialize(local_args,local_arglen);
      }
      derez.deserialize(orig_proc);
      derez.deserialize(steal_count);
      derez.deserialize(speculated);
      derez.deserialize(map_locally);
      if (map_locally)
      {
        size_t num_atomic;
        derez.deserialize(num_atomic);
        for (unsigned idx = 0; idx < num_atomic; idx++)
        {
          Reservation lock;
          derez.deserialize(lock);
          derez.deserialize(atomic_locks[lock]);
        }
      }
      size_t num_early;
      derez.deserialize(num_early);
      for (unsigned idx = 0; idx < num_early; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        early_mapped_regions[index].unpack_references(runtime, derez); 
      }
      // Parent requirement indexes don't mean anything remotely
      parent_req_indexes.resize(regions.size(), 0);
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
            IndividualTask *task = rt->get_available_individual_task(false);
            if (task->unpack_task(derez, current))
              rt->add_to_ready_queue(current, task, 
                                     false/*prev fail*/);
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task(false);
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
    void TaskOp::mark_stolen(void)
    //--------------------------------------------------------------------------
    {
      steal_count++;
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
      speculated = false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::check_empty_field_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].privilege_fields.empty())
        {
          log_task.warning("WARNING: REGION REQUIREMENT %d OF "
                           "TASK %s (ID %lld) HAS NO PRIVILEGE "
                           "FIELDS! DID YOU FORGET THEM?!?",
                           idx, get_task_name(), get_unique_id());
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::check_future_size(FutureImpl *impl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      const size_t result_size = impl->get_untyped_size();
      // TODO: figure out a way to put this check back in with dynamic task
      // registration where we might not know the return size until later
#if 0
      if (result_size != variants->return_size)
      {
        log_run.error("Predicated task launch for task %s "
                      "in parent task %s (UID %lld) has predicated "
                      "false future of size %ld bytes, but the "
                      "expected return size is %ld bytes.",
                      get_task_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(),
                      result_size, variants->return_size);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_PREDICATE_RESULT_SIZE_MISMATCH);
      }
#endif
      return result_size;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::select_task_options(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::TaskOptions options;
      options.initial_proc = current_proc;
      options.inline_task = false;
      options.stealable = false;
      options.map_locally = false;
      mapper->invoke_select_task_options(this, &options);
      target_proc = options.initial_proc;
      stealable = options.stealable;
      map_locally = options.map_locally;
      return options.inline_task;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_logging_name(void)
    //--------------------------------------------------------------------------
    {
      return get_task_name();
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TaskOp::get_operation_kind(void)
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return regions.size();
    }

    //--------------------------------------------------------------------------
    Mappable* TaskOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
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
    void TaskOp::resolve_true(void)
    //--------------------------------------------------------------------------
    {
      // Put this on the ready queue
      runtime->add_to_ready_queue(current_proc, this, false/*prev fail*/);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::speculate(bool &value)
    //--------------------------------------------------------------------------
    {
      Mapper::SpeculativeOutput output;
      if (mapper == NULL)  
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_task_speculate(this, &output);
      if (output.speculate)
        value = output.speculative_value;
      return output.speculate;
    }

    //--------------------------------------------------------------------------
    void TaskOp::select_sources(const InstanceRef &target,
                                const InstanceSet &sources,
                                std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances);
      input.region_req_index = current_mapping_index;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_select_task_sources(this, &input, &output);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_atomic_locks(Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Only one region should be in the process of being analyzed
      // at a time so there is no need to hold the operation lock
      std::map<Reservation,bool>::iterator finder = atomic_locks.find(lock);
      if (finder != atomic_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        atomic_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    unsigned TaskOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < parent_req_indexes.size());
#endif
      return parent_req_indexes[idx];
    }

    //--------------------------------------------------------------------------
    VersionInfo& TaskOp::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return (*(new VersionInfo()));
    }

    //--------------------------------------------------------------------------
    RegionTreePath& TaskOp::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return (*(new RegionTreePath()));
    }

    //--------------------------------------------------------------------------
    void TaskOp::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_inline_task(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called except by inherited types
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskOp::begin_inline_task(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *(reinterpret_cast<std::vector<PhysicalRegion>*>(NULL));
    }

    //--------------------------------------------------------------------------
    void TaskOp::end_inline_task(const void *result, 
                                 size_t result_size, bool owned)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RegionTreeContext TaskOp::get_parent_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      assert(idx < regions.size());
#endif
      if (!is_remote())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < parent_req_indexes.size());
#endif
        return parent_ctx->find_enclosing_context(parent_req_indexes[idx]);
      }
      // This is remote, so just return the context of the remote parent
      return parent_ctx->get_context(); 
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_version_infos(Serializer &rez,
                                    std::vector<VersionInfo> &infos)
    //--------------------------------------------------------------------------
    {
      if (!is_locally_mapped())
      {
        RezCheck z(rez);
        AddressSpaceID local_space = runtime->address_space;
#ifdef DEBUG_HIGH_LEVEL
        assert(infos.size() == regions.size());
#endif
        for (unsigned idx = 0; idx < infos.size(); idx++)
        {
          infos[idx].pack_version_info(rez, local_space, 
                                       get_parent_context(idx).get_id()); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_version_infos(Deserializer &derez,
                                      std::vector<VersionInfo> &infos)
    //--------------------------------------------------------------------------
    {
      if (!is_locally_mapped())
      {
        DerezCheck z(derez);
        infos.resize(regions.size());
        for (unsigned idx = 0; idx < infos.size(); idx++)
        {
          infos[idx].unpack_version_info(derez);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_restrict_infos(Serializer &rez,
                                     std::vector<RestrictInfo> &infos)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      size_t count = 0;
      for (unsigned idx = 0; idx < infos.size(); idx++)
      {
        if (infos[idx].has_restrictions())
          count++;
      }
      rez.serialize(count);
      if (count > 0)
      {
        rez.serialize(runtime->address_space);
        for (unsigned idx = 0; idx < infos.size(); idx++)
        {
          if (infos[idx].has_restrictions())
          {
            rez.serialize(idx);
            infos[idx].pack_info(rez);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_restrict_infos(Deserializer &derez,
                                       std::vector<RestrictInfo> &infos)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // Always resize the restrictions
      infos.resize(regions.size());
      size_t num_restrictions;
      derez.deserialize(num_restrictions);
      if (num_restrictions > 0)
      {
        AddressSpaceID source;
        derez.deserialize(source);
        for (unsigned idx = 0; idx < num_restrictions; idx++)
        {
          unsigned index;
          derez.deserialize(index);
#ifdef DEBUG_HIGH_LEVEL
          assert(index < infos.size());
#endif
          infos[index].unpack_info(derez, source, runtime->forest);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->increment_outstanding();
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->decrement_outstanding();
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
      rez.serialize<size_t>(created_regions.size());
      for (std::set<LogicalRegion>::const_iterator it =
            created_regions.begin(); it != created_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize<size_t>(deleted_regions.size());
      for (std::set<LogicalRegion>::const_iterator it =
            deleted_regions.begin(); it != deleted_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize<size_t>(created_fields.size());
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it =
            created_fields.begin(); it != created_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(deleted_fields.size());
      for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
            deleted_fields.begin(); it != deleted_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(created_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = 
            created_field_spaces.begin(); it != 
            created_field_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize<size_t>(deleted_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = 
            deleted_field_spaces.begin(); it !=
            deleted_field_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize<size_t>(created_index_spaces.size());
      for (std::set<IndexSpace>::const_iterator it = 
            created_index_spaces.begin(); it != 
            created_index_spaces.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize<size_t>(deleted_index_spaces.size());
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
              log_index.error("Parent task %s (ID %lld) of task %s "
                              "(ID %lld) "
                              "does not have an index requirement for "
                              "index space %x as a parent of "
                              "child task's index requirement index %d",
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id(), get_task_name(),
                              get_unique_id(), indexes[idx].parent.id, idx);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARENT_INDEX);
            }
          case ERROR_BAD_INDEX_PATH:
            {
              log_index.error("Index space %x is not a sub-space "
                              "of parent index space %x for index "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].handle.id, 
                              indexes[idx].parent.id, idx,
                              get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_INDEX_PATH);
            }
          case ERROR_BAD_INDEX_PRIVILEGES:
            {
              log_index.error("Privileges %x for index space %x "
                              " are not a subset of privileges of parent "
                              "task's privileges for index space "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].privilege, 
                              indexes[idx].handle.id, idx, 
                              get_task_name(), get_unique_id());
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
              log_region.error("Invalid region handle (%x,%d,%d)"
                              " for region requirement %d of task %s "
                              "(ID %lld)",
                              regions[idx].region.index_space.id, 
                              regions[idx].region.field_space.id, 
                              regions[idx].region.tree_id, idx, 
                              get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_REGION_HANDLE);
            }
          case ERROR_INVALID_PARTITION_HANDLE:
            {
              log_region.error("Invalid partition handle (%x,%d,%d) "
                               "for partition requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].partition.index_partition.id, 
                               regions[idx].partition.field_space.id, 
                               regions[idx].partition.tree_id, idx, 
                               get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_PARTITION_HANDLE);
            }
          case ERROR_BAD_PROJECTION_USE:
            {
              log_region.error("Projection region requirement %d used "
                                "in non-index space task %s",
                                idx, get_task_name());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PROJECTION_USE);
            }
          case ERROR_NON_DISJOINT_PARTITION:
            {
              log_region.error("Non disjoint partition selected for "
                                "writing region requirement %d of task "
                                "%s.  All projection partitions "
                                "which are not read-only and not reduce "
                                "must be disjoint", 
                                idx, get_task_name());
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
              log_region.error("Field %d is not a valid field of field "
                              "space %d for region %d of task %s "
                              "(ID %lld)",
                              bad_field, sp.id, idx, get_task_name(),
                              get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
            }
          case ERROR_INVALID_INSTANCE_FIELD:
            {
              log_region.error("Instance field %d is not one of the "
                                "privilege fields for region %d of "
                                "task %s (ID %lld)",
                                bad_field, idx, get_task_name(), 
                                get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_INSTANCE_FIELD);
            }
          case ERROR_DUPLICATE_INSTANCE_FIELD:
            {
              log_region.error("Instance field %d is a duplicate for "
                                "region %d of task %s (ID %lld)",
                                bad_field, idx, get_task_name(), 
                                get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_DUPLICATE_INSTANCE_FIELD);
            }
          case ERROR_BAD_PARENT_REGION:
            {
              log_region.error("Parent task %s (ID %lld) of task %s "
                                "(ID %lld) does not have a region "
                                "requirement for region " 
                                "(%x,%x,%x) as a parent of child task's "
                                "region requirement index %d",
                                parent_ctx->get_task_name(), 
                                parent_ctx->get_unique_id(),
                                get_task_name(), get_unique_id(),
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
              log_region.error("Region (%x,%x,%x) is not a "
                                "sub-region of parent region "
                                "(%x,%x,%x) for region requirement %d of "
                                "task %s (ID %lld)",
                                regions[idx].region.index_space.id,
                                regions[idx].region.field_space.id, 
                                regions[idx].region.tree_id,
                                PRINT_REG(regions[idx].parent), idx,
                                get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_PATH);
            }
          case ERROR_BAD_PARTITION_PATH:
            {
              log_region.error("Partition (%x,%x,%x) is not a "
                               "sub-partition of parent region "
                               "(%x,%x,%x) for region "
                               "requirement %d task %s (ID %lld)",
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id, 
                               regions[idx].partition.tree_id, 
                               PRINT_REG(regions[idx].parent), idx,
                               get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_PARTITION_PATH);
            }
          case ERROR_BAD_REGION_TYPE:
            {
              log_region.error("Region requirement %d of task %s "
                                     "(ID %lld) "
                                     "cannot find privileges for field %d in "
                                     "parent task",
                                      idx, get_task_name(), 
                                      get_unique_id(), bad_field);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_TYPE);
            }
          case ERROR_BAD_REGION_PRIVILEGES:
            {
              log_region.error("Privileges %x for region " 
                               "(%x,%x,%x) are not a subset of privileges " 
                               "of parent task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege, 
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id, 
                               regions[idx].region.tree_id, idx, 
                               get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_BAD_REGION_PRIVILEGES);
            }
          case ERROR_BAD_PARTITION_PRIVILEGES:
            {
              log_region.error("Privileges %x for partition (%x,%x,%x) "
                               "are not a subset of privileges of parent "
                               "task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege, 
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id, 
                               regions[idx].partition.tree_id, idx, 
                               get_task_name(), get_unique_id());
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
    void TaskOp::find_early_mapped_region(unsigned idx, InstanceSet &ref)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,InstanceSet>::const_iterator finder = 
        early_mapped_regions.find(idx);
      if (finder != early_mapped_regions.end())
        ref = finder->second;
    }

    //--------------------------------------------------------------------------
    void TaskOp::clone_task_op_from(TaskOp *rhs, Processor p, 
                                    bool can_steal, bool duplicate_args)
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
          this->arg_manager = legion_new<AllocManager>(this->arglen); 
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
        this->args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,rhs->args,arglen);
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      this->is_index_space = rhs->is_index_space;
      this->orig_proc = rhs->orig_proc;
      this->current_proc = rhs->current_proc;
      this->steal_count = rhs->steal_count;
      this->stealable = can_steal;
      this->speculated = rhs->speculated;
      // Premapping should never get cloned
      this->map_locally = rhs->map_locally;
      // From TaskOp
      this->atomic_locks = rhs->atomic_locks;
      this->early_mapped_regions = rhs->early_mapped_regions;
      this->parent_req_indexes = rhs->parent_req_indexes;
      this->target_proc = p;
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
        arrive_barriers.push_back(*it);
        // Note it is imperative we do this off the new barrier
        // generated after updating the arrival count.
        arrive_barriers.back().phase_barrier.arrive(1, get_task_completion());
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(it->phase_barrier,
                                        arrive_barriers.back().phase_barrier); 
        LegionSpy::log_event_dependence(get_task_completion(),
                                        arrive_barriers.back().phase_barrier);
#endif
      }
    }

    //--------------------------------------------------------------------------
    bool TaskOp::compute_point_region_requirements(MinimalPoint *mp/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool all_invalid = true;
      // Update the region requirements for this point
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type == PART_PROJECTION)
        {
          if (mp != NULL)
          {
            // If we have a minimal point we should be able to find it
            regions[idx].region = mp->find_logical_region(idx);
          }
          else
          {
            // Check to see if we're doing default projection
            if (regions[idx].projection == 0)
            {
              if (index_point.get_dim() > 3)
              {
                log_task.error("Projection ID 0 is invalid for tasks whose "
                               "points are larger than three dimensional "
                               "unsigned integers.  Points for task %s "
                               "have elements of %d dimensions",
                                get_task_name(), index_point.get_dim());
#ifdef DEBUG_HIGH_LEVEL
                assert(false);
#endif
                exit(ERROR_INVALID_IDENTITY_PROJECTION_USE);
              }
              regions[idx].region = 
                runtime->forest->get_logical_subregion_by_color(
                    regions[idx].partition, ColorPoint(index_point));
            }
            else
            {
              ProjectionFunctor *functor = 
                runtime->find_projection_functor(regions[idx].projection);
              if (functor == NULL)
              {
                PartitionProjectionFnptr projfn = 
                  Runtime::find_partition_projection_function(
                      regions[idx].projection);
                regions[idx].region = 
                  (*projfn)(regions[idx].partition,
                            index_point,runtime->external);
              }
              else
                regions[idx].region = 
                  functor->project(DUMMY_CONTEXT, this, idx,
                                   regions[idx].partition, index_point);
            }
          }
          // Update the region requirement kind 
          regions[idx].handle_type = SINGULAR;
        }
        else if (regions[idx].handle_type == REG_PROJECTION)
        {
          if (mp != NULL)
          {
            // If the minimal point is not null then it should have it
            regions[idx].region = mp->find_logical_region(idx);
          }
          else
          {
            if (regions[idx].projection != 0)
            {
              ProjectionFunctor *functor = 
                runtime->find_projection_functor(regions[idx].projection);
              if (functor == NULL)
              {
                RegionProjectionFnptr projfn = 
                  Runtime::find_region_projection_function(
                      regions[idx].projection);
                regions[idx].region = 
                 (*projfn)(regions[idx].region,index_point,runtime->external);
              }
              else
                regions[idx].region = 
                  functor->project(DUMMY_CONTEXT, this, idx, 
                                   regions[idx].region, index_point);
            }
          }
          // Otherwise we are the default case in which 
          // case we don't need to do anything
          // Update the region requirement kind
          regions[idx].handle_type = SINGULAR;
        }
        // Always check to see if there are any restrictions
        if (has_restrictions(idx, regions[idx].region))
          regions[idx].flags |= RESTRICTED_FLAG;
        // Check to see if the region is a NO_REGION,
        // if it is then switch the privilege to NO_ACCESS
        if (regions[idx].region == LogicalRegion::NO_REGION)
          regions[idx].privilege = NO_ACCESS;
        else
          all_invalid = false;
      }
      // Return true if this point has any valid region requirements
      return (!all_invalid);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::early_map_regions(std::set<Event> &applied_conditions,
                                   const std::vector<unsigned> &must_premap)
    //--------------------------------------------------------------------------
    {
      Mapper::PremapTaskInput input;
      Mapper::PremapTaskOutput output;
      // Set up the inputs and outputs 
      std::vector<InstanceSet> valid_instances(must_premap.size());
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < must_premap.size(); idx++)
      {
        VersionInfo &version_info = get_version_info(must_premap[idx]);
        RegionTreeContext req_ctx = get_parent_context(must_premap[idx]);
        RegionTreePath &privilege_path = get_privilege_path(must_premap[idx]);
        InstanceSet &valid = valid_instances[idx];    
        // Do the premapping
        runtime->forest->physical_traverse_path(req_ctx, privilege_path,
                                                regions[must_premap[idx]],
                                                version_info, this, 
                                                true/*find valid*/, valid
#ifdef DEBUG_HIGH_LEVEL
                                                , must_premap[idx]
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // If we need visible instances, filter them as part of the conversion
        if (regions[idx].is_no_access())
          prepare_for_mapping(valid, input.valid_instances[must_premap[idx]]);
        else
          prepare_for_mapping(valid, visible_memories, 
                              input.valid_instances[must_premap[idx]]);
      }
      // Now invoke the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, &input, &output);
      // Now do the registration
      for (unsigned idx = 0; idx < must_premap.size(); idx++)
      {
        VersionInfo &version_info = get_version_info(must_premap[idx]);
        RegionTreeContext req_ctx = get_parent_context(must_premap[idx]);
        InstanceSet chosen_instances;
        // If this is restricted then we know what the answer is so
        // just ignore whatever the mapper did
        if (regions[idx].is_restricted())
        {
          // Since we know we are on the owner node, we know we can
          // always ask our parent context to find the restricted instances
          parent_ctx->get_physical_references(must_premap[idx], 
                                              chosen_instances);
        }
        else
        {
          // Otherwise this was not restricted, so do what the mapper wants
          std::map<unsigned,std::vector<MappingInstance> >::const_iterator 
            finder = output.premapped_instances.find(must_premap[idx]);
          if (finder == output.premapped_instances.end())
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper failed to map required premap "
                          "region requirement %d of task %s (ID %lld) launched "
                          "in parent task %s (ID %lld).", 
                          mapper->get_mapper_name(), must_premap[idx], 
                          get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          std::vector<FieldID> missing_fields;
          int composite_index = runtime->forest->physical_convert_mapping(
              regions[must_premap[idx]], finder->second, valid_instances[idx],
              chosen_instances, missing_fields);
          if (composite_index >= 0)
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper requested composite instance "
                          "creation on region requirement %d of task %s "
                          "(ID %lld) launched in parent task %s (ID %lld).",
                          mapper->get_mapper_name(), must_premap[idx],
                          get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          if (!missing_fields.empty())
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper failed to specify instances "
                          "for %ld fields of region requirement %d of task %s "
                          "(ID %lld) launched in parent task %s (ID %lld). "
                          "The missing fields are listed below.",
                          mapper->get_mapper_name(), missing_fields.size(),
                          must_premap[idx], get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
            for (std::vector<FieldID>::const_iterator it = 
                  missing_fields.begin(); it != missing_fields.end(); it++)
            {
              const void *name; size_t name_size;
              runtime->retrieve_semantic_information(
                  regions[must_premap[idx]].region.get_field_space(), *it,
                  NAME_SEMANTIC_TAG, name, name_size, false, false);
              log_run.error("MIssing instance for field %s (FieldID: %d)",
                            static_cast<const char*>(name), *it);
            }
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          if (!Runtime::unsafe_mapper)
          {
            for (unsigned check_idx = 0; 
                  check_idx < chosen_instances.size(); check_idx++)
            {
              if (!runtime->forest->is_valid_mapping(
                    chosen_instances[check_idx], regions[must_premap[idx]]))
              {
                log_run.error("Invalid mapper output from invocation of "
                              "'premap_task' on mapper %s. Mapper specified an "
                              "instance region requirement %d of task %s "
                              "(ID %lld) that does not meet the logical region "
                              "requirement. Task was launched in task %s "
                              "(ID %lld).", mapper->get_mapper_name(), 
                              must_premap[idx], get_task_name(), 
                              get_unique_id(), parent_ctx->get_task_name(), 
                              parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
        }
        // Set the current mapping index before doing anything that
        // could result in the generation of a copy
        set_current_mapping_index(must_premap[idx]);
        // Passed all the error checking tests so register it
        runtime->forest->physical_register_only(req_ctx, 
                              regions[must_premap[idx]], version_info, 
                              this, completion_event, chosen_instances
#ifdef DEBUG_HIGH_LEVEL
                              , must_premap[idx], get_logging_name()
                              , unique_op_id
#endif
                              );
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      if (is_locally_mapped())
        return false;
      if (!is_remote())
        return early_map_task();
      else
        return true;
    }

    //--------------------------------------------------------------------------
    void TaskOp::record_aliased_region_requirements(LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      for (unsigned i = 1; i < regions.size(); i++)
      {
        for (unsigned j = 0; j < i; j++)
        {
          // Check tree ID first
          if (regions[i].parent.get_tree_id() != 
              regions[j].parent.get_tree_id())
            continue;
          // Now check if the regions/partitions are aliased
          if (regions[i].handle_type == PART_PROJECTION)
          {
            IndexPartition part1 = regions[i].partition.get_index_partition();
            if (regions[j].handle_type == PART_PROJECTION)
            {
              IndexPartition part2 = regions[j].partition.get_index_partition();
              if (runtime->forest->are_disjoint(part1, part2))
                continue;
            }
            else
            {
              IndexSpace space2 = regions[j].region.get_index_space();
              if (runtime->forest->are_disjoint(space2, part1))
                continue;
            }
          }
          else
          {
            IndexSpace space1 = regions[i].region.get_index_space();
            if (regions[j].handle_type == PART_PROJECTION)
            {
              IndexPartition part2 = regions[j].partition.get_index_partition();
              if (runtime->forest->are_disjoint(space1, part2))
                continue;
            }
            else
            {
              IndexSpace space2 = regions[j].region.get_index_space();
              if (runtime->forest->are_disjoint(space1, space2))
                continue;
            }
          }
          // Regions are aliased, see if there are overlapping fields
          bool overlap = false;
          if (regions[i].privilege_fields.size() < 
              regions[j].privilege_fields.size())
          {
            for (std::set<FieldID>::const_iterator it = 
                  regions[i].privilege_fields.begin(); it !=
                  regions[i].privilege_fields.end(); it++)
            {
              if (regions[j].privilege_fields.find(*it) !=
                  regions[j].privilege_fields.end())
              {
                overlap = true;
                break;
              }
            }
          }
          else
          {
            for (std::set<FieldID>::const_iterator it = 
                  regions[j].privilege_fields.begin(); it !=
                  regions[j].privilege_fields.end(); it++)
            {
              if (regions[i].privilege_fields.find(*it) !=
                  regions[i].privilege_fields.end())
              {
                overlap = true;
                break;
              }
            }
          }
          if (overlap)
            trace->record_aliased_requirements(j,i);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_parent_indexes(void)
    //--------------------------------------------------------------------------
    {
      parent_req_indexes.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        int parent_index = 
          parent_ctx->find_parent_region_req(regions[idx]);
        if (parent_index < 0)
        {
          log_region.error("Parent task %s (ID %lld) of task %s "
                           "(ID %lld) does not have a region "
                           "requirement for region "
                           "(%x,%x,%x) as a parent of child task's "
                           "region requirement index %d",
                           parent_ctx->get_task_name(), 
                           parent_ctx->get_unique_id(),
                           get_task_name(), get_unique_id(),
                           regions[idx].parent.index_space.id,
                           regions[idx].parent.field_space.id, 
                           regions[idx].parent.tree_id, idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_BAD_PARENT_REGION);
        }
        parent_req_indexes[idx] = parent_index;
      }
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
      grant = Grant(legion_new<GrantImpl>());
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
      const bool reg = (req.handle_type == SINGULAR) ||
                 (req.handle_type == REG_PROJECTION);

      LegionSpy::log_logical_requirement(uid, idx, reg,
          reg ? req.region.index_space.id :
                req.partition.index_partition.id,
          reg ? req.region.field_space.id :
                req.partition.field_space.id,
          reg ? req.region.tree_id : 
                req.partition.tree_id,
          req.privilege, req.prop, req.redop);
      LegionSpy::log_requirement_fields(uid, idx, req.privilege_fields);
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
      executing_processor = Processor::NO_PROC;
      current_fence = NULL;
      fence_gen = 0;
      context = RegionTreeContext();
      initial_region_count = 0;
      valid_wait_event = false;
      deferred_map = Event::NO_EVENT;
      deferred_complete = Event::NO_EVENT; 
      pending_done = Event::NO_EVENT;
      last_registration = Event::NO_EVENT;
      dependence_precondition = Event::NO_EVENT;
      profiling_done = Event::NO_EVENT;
      current_trace = NULL;
      task_executed = false;
      outstanding_children_count = 0;
      outstanding_subtasks = 0;
      pending_subtasks = 0;
      pending_frames = 0;
      context_order_event = Event::NO_EVENT;
      // Set some of the default values for a context
      context_configuration.max_window_size = 
        Runtime::initial_task_window_size;
      context_configuration.hysteresis_percentage = 
        Runtime::initial_task_window_hysteresis;
      context_configuration.max_outstanding_frames = -1;
      context_configuration.min_tasks_to_schedule = 
        Runtime::initial_tasks_to_schedule;
      context_configuration.min_frames_to_schedule = 0;
      selected_variant = 0;
      task_priority = 0;
      perform_postmap = false;
      leaf_cached = false;
      inner_cached = false;
      has_virtual_instances_result = false;
      has_virtual_instances_cached = false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate_single(void)
    //--------------------------------------------------------------------------
    {
      deactivate_task();
      target_processors.clear();
      physical_instances.clear();
      physical_regions.clear();
      inline_regions.clear();
      virtual_mapped.clear();
      region_deleted.clear();
      index_deleted.clear();
      executing_children.clear();
      executed_children.clear();
      complete_children.clear();
      safe_cast_domains.clear();
      restricted_trees.clear();
      frame_events.clear();
      for (std::map<TraceID,LegionTrace*>::const_iterator it = traces.begin();
            it != traces.end(); it++)
      {
        legion_delete(it->second);
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
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_subtasks == 0);
      assert(pending_subtasks == 0);
      assert(pending_frames == 0);
#endif
      if (context.exists())
        runtime->free_context(this);
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_leaf(void) const
    //--------------------------------------------------------------------------
    {
      if (!leaf_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_leaf_result = var->is_leaf();
        leaf_cached = true;
      }
      return is_leaf_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_inner(void) const
    //--------------------------------------------------------------------------
    {
      if (!inner_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_inner_result = var->is_inner();
        inner_cached = true;
      }
      return is_inner_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::has_virtual_instances(void) const
    //--------------------------------------------------------------------------
    {
      if (!has_virtual_instances_cached)
      {
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (physical_instances[idx].has_composite_ref())
          {
            has_virtual_instances_result = true;
            break;
          }
        }
        has_virtual_instances_cached = true;
      }
      return has_virtual_instances_result;
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
    UniqueID SingleTask::get_context_uid(void) const
    //--------------------------------------------------------------------------
    {
      // For most single tasks, this is always the answer, the
      // exception is for RemoteTask objects which override this method
      return unique_op_id;
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
    void SingleTask::get_physical_references(unsigned idx, InstanceSet &set)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < physical_instances.size());
#endif
      set = physical_instances[idx];
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
      InlineTask *inline_task = runtime->get_available_inline_task(true);
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

      // Pick a variant to use for executing this task
      VariantImpl *variant = select_inline_variant(child);    
      
      // Do the inlining
      child->perform_inlining(inline_task, variant);

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
          MapOp *op = runtime->get_available_map_op(true);
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
        Event wait_on = Runtime::merge_events<true>(wait_events);
        if (!wait_on.has_triggered())
          wait_on.wait();
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
    const std::vector<PhysicalRegion>& SingleTask::get_physical_regions(void) 
                                                                           const
    //--------------------------------------------------------------------------
    {
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      if (map_locally)
      {
        rez.serialize(selected_variant);
        rez.serialize<size_t>(target_processors.size());
        for (unsigned idx = 0; idx < target_processors.size(); idx++)
          rez.serialize(target_processors[idx]);
      }
      rez.serialize<size_t>(physical_instances.size());
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        physical_instances[idx].pack_references(rez, target);
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_base_task(derez);
      if (map_locally)
      {
        derez.deserialize(selected_variant);
        size_t num_target_processors;
        derez.deserialize(num_target_processors);
        target_processors.resize(num_target_processors);
        for (unsigned idx = 0; idx < num_target_processors; idx++)
          derez.deserialize(target_processors[idx]);
      }
      size_t num_phy;
      derez.deserialize(num_phy);
      physical_instances.resize(num_phy);
      for (unsigned idx = 0; idx < num_phy; idx++)
        physical_instances[idx].unpack_references(runtime, derez);
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_parent_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Starting with this context, pack up all the enclosing local fields
      LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked locals = 
                                                                  local_fields;
      // Get all the local fields from our enclosing contexts
      find_enclosing_local_fields(locals);
      RezCheck z(rez);
      int depth = get_depth();
      rez.serialize(depth);
      // Now pack them all up
      size_t num_local = locals.size();
      rez.serialize(num_local);
      for (unsigned idx = 0; idx < locals.size(); idx++)
        rez.serialize(locals[idx]);
#ifdef LEGION_SPY
      rez.serialize(legion_spy_start);
      rez.serialize(get_task_completion());
#endif
    }

    //-------------------------------------------------------------------------
    void SingleTask::pack_remote_ctx_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Assume we are the owner in this case
      UniqueID remote_owner_proxy = get_unique_id();
      rez.serialize(remote_owner_proxy);
      SingleTask *proxy_this = this;
      rez.serialize(proxy_this);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_new_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      // If we are performing a trace mark that the child has a trace
      if (current_trace != NULL)
        op->set_trace(current_trace, !current_trace->is_fixed());
      int outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,1);
      // Only need to check if we are not tracing by frames
      if ((context_configuration.max_outstanding_frames <= 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count >= context_configuration.max_window_size))
      {
        // Launch a window-wait task and then wait on the event 
        WindowWaitArgs args;
        args.hlr_id = HLR_WINDOW_WAIT_TASK_ID;
        args.parent_ctx = this;
        Event wait_done = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                                HLR_WINDOW_WAIT_TASK_ID, this);
        wait_done.wait();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_window_wait(void)
    //--------------------------------------------------------------------------
    {
      Event wait_event = Event::NO_EVENT;
      {
        // Take the lock and make sure we didn't lose the race
        AutoLock o_lock(op_lock);
        // We can read this without locking because we know the application
        // task isn't running if we are here and the lock serializes us
        // with all the other meta-tasks
        if (outstanding_children_count >= context_configuration.max_window_size)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!valid_wait_event);
#endif
          window_wait = UserEvent::create_user_event();
          valid_wait_event = true;
          wait_event = window_wait;
        }
      }
      if (wait_event.exists() && !wait_event.has_triggered())
        wait_event.wait();
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_to_dependence_queue(Operation *op, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (!has_lock)
      {
        Event lock_acquire = 
          op_lock.acquire(0, true/*exclusive*/, last_registration); 
        if (!lock_acquire.has_triggered())
        {
          AddToDepQueueArgs args;
          args.hlr_id = HLR_ADD_TO_DEP_QUEUE_TASK_ID;
          args.proxy_this = this;
          args.op = op;
          last_registration = runtime->issue_runtime_meta_task(&args, 
             sizeof(args), HLR_ADD_TO_DEP_QUEUE_TASK_ID, op, lock_acquire);
          return;
        }
      }
      // We have the lock
      if (op->is_tracking_parent())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(executing_children.find(op) == executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
#endif       
        executing_children.insert(op);
      }
      // Issue the next dependence analysis task
      DeferredDependenceArgs args;
      args.hlr_id = HLR_TRIGGER_DEPENDENCE_ID;
      args.op = op;
      Event next = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                      HLR_TRIGGER_DEPENDENCE_ID, op,
                                      dependence_precondition);
      dependence_precondition = next;
      // Now we can release the lock
      op_lock.release();
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
      int outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,-1);
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_count >= 0);
#endif
      if (valid_wait_event && (context_configuration.max_window_size > 0) &&
          (outstanding_count <=
           int(context_configuration.hysteresis_percentage * 
               context_configuration.max_window_size / 100)))
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
        if (task_executed && executing_children.empty() && 
            executed_children.empty() && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_complete();
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
      int outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,-1);
#ifdef DEBUG_HIGH_LEVEL
      assert(outstanding_count >= 0);
#endif
      if (valid_wait_event && (context_configuration.max_window_size > 0) &&
          (outstanding_count <=
           int(context_configuration.hysteresis_percentage * 
               context_configuration.max_window_size / 100)))
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
    void SingleTask::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (current_fence != NULL)
      {
#ifdef LEGION_SPY
        // Can't prune when doing legion spy
        op->register_dependence(current_fence, fence_gen);
        unsigned num_regions = op->get_region_count();
        if (num_regions > 0)
        {
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            LegionSpy::log_mapping_dependence(
                get_unique_op_id(), current_fence_uid, 0,
                op->get_unique_op_id(), idx, TRUE_DEPENDENCE);
          }
        }
        else
          LegionSpy::log_mapping_dependence(
              get_unique_op_id(), current_fence_uid, 0,
              op->get_unique_op_id(), 0, TRUE_DEPENDENCE);
#else
        // If we can prune it then go ahead and do so
        // No need to remove the mapping reference because 
        // the fence has already been committed
        if (op->register_dependence(current_fence, fence_gen))
          current_fence = NULL;
#endif
      }
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
#ifdef LEGION_SPY
      current_fence_uid = op->get_unique_op_id();
#endif
    }

    //--------------------------------------------------------------------------
    void SingleTask::begin_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
      {
        log_task.error("Illegal nested trace with ID %d attempted in "
                       "task %s (ID %lld)", tid, get_task_name(),
                       get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_NESTED_TRACE);
      }
      std::map<TraceID,LegionTrace*>::const_iterator finder = traces.find(tid);
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        current_trace = legion_new<LegionTrace>(tid, this);
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
        log_task.error("Unmatched end trace for ID %d in task %s "
                       "(ID %lld)", tid, get_task_name(),
                       get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNMATCHED_END_TRACE);
      }
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op(true);
        complete_op->initialize_complete(this);
#ifdef INORDER_EXECUTION
        Event term_event = complete_op->get_completion_event();
#endif
        runtime->add_to_dependence_queue(get_executing_processor(),complete_op);
#ifdef INORDER_EXECUTION
        if (Runtime::program_order_execution && !term_event.has_triggered())
          term_event.wait();
#endif
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(true); 
        capture_op->initialize_capture(this);
#ifdef INORDER_EXECUTION
        Event term_event = capture_op->get_completion_event();
#endif
        runtime->add_to_dependence_queue(get_executing_processor(), capture_op);
#ifdef INORDER_EXECUTION
        if (Runtime::program_order_execution && !term_event.has_triggered())
          term_event.wait();
#endif
        // Mark that the current trace is now fixed
        current_trace->fix_trace();
      }
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void SingleTask::issue_frame(FrameOp *frame, Event frame_termination)
    //--------------------------------------------------------------------------
    {
      // This happens infrequently enough that we can just issue
      // a meta-task to see what we should do without holding the lock
      if (context_configuration.max_outstanding_frames > 0)
      {
        IssueFrameArgs args;
        args.hlr_id = HLR_ISSUE_FRAME_TASK_ID;
        args.parent_ctx = this;
        args.frame = frame;
        args.frame_termination = frame_termination;
        // We know that the issuing is done in order because we block after
        // we launch this meta-task which blocks the application task
        Event wait_on = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                              HLR_ISSUE_FRAME_TASK_ID, this); 
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_frame_issue(FrameOp *frame,Event frame_termination)
    //--------------------------------------------------------------------------
    {
      Event wait_on = Event::NO_EVENT;
      Event previous = Event::NO_EVENT;
      {
        AutoLock o_lock(op_lock);
        const size_t current_frames = frame_events.size();
        if (current_frames > 0)
          previous = frame_events.back();
        if (current_frames > 
            (size_t)context_configuration.max_outstanding_frames)
          wait_on = frame_events[current_frames - 
                                 context_configuration.max_outstanding_frames];
        frame_events.push_back(frame_termination); 
      }
      frame->set_previous(previous);
      if (!wait_on.has_triggered())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    void SingleTask::finish_frame(Event frame_termination)
    //--------------------------------------------------------------------------
    {
      // Pull off all the frame events until we reach ours
      if (context_configuration.max_outstanding_frames > 0)
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(frame_events.front() == frame_termination);
#endif
        frame_events.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_outstanding(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((context_configuration.min_tasks_to_schedule == 0) || 
             (context_configuration.min_frames_to_schedule == 0));
      assert((context_configuration.min_tasks_to_schedule > 0) || 
             (context_configuration.min_frames_to_schedule > 0));
#endif
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
        if ((outstanding_subtasks == 0) && 
            (((context_configuration.min_tasks_to_schedule > 0) && 
              (pending_subtasks < 
               context_configuration.min_tasks_to_schedule)) ||
             ((context_configuration.min_frames_to_schedule > 0) &&
              (pending_frames < 
               context_configuration.min_frames_to_schedule))))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
        outstanding_subtasks++;
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_outstanding(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((context_configuration.min_tasks_to_schedule == 0) || 
             (context_configuration.min_frames_to_schedule == 0));
      assert((context_configuration.min_tasks_to_schedule > 0) || 
             (context_configuration.min_frames_to_schedule > 0));
#endif
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(outstanding_subtasks > 0);
#endif
        outstanding_subtasks--;
        if ((outstanding_subtasks == 0) && 
            (((context_configuration.min_tasks_to_schedule > 0) &&
              (pending_subtasks < 
               context_configuration.min_tasks_to_schedule)) ||
             ((context_configuration.min_frames_to_schedule > 0) &&
              (pending_frames < 
               context_configuration.min_frames_to_schedule))))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
        pending_subtasks++;
        if ((outstanding_subtasks > 0) &&
            (pending_subtasks == context_configuration.min_tasks_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are schedule based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(pending_subtasks > 0);
#endif
        if ((outstanding_subtasks > 0) &&
            (pending_subtasks == context_configuration.min_tasks_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
        pending_subtasks--;
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
        pending_frames++;
        if ((outstanding_subtasks > 0) &&
            (pending_frames == context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      Event wait_on = Event::NO_EVENT;
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(pending_frames > 0);
#endif
        if ((outstanding_subtasks > 0) &&
            (pending_frames == context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = UserEvent::create_user_event();
          context_order_event = to_trigger;
        }
        pending_frames--;
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        to_trigger.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_local_field(FieldSpace handle, FieldID fid, 
                                     size_t field_size,CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      allocate_local_field(local_fields.back());
      // Hold the lock when modifying the local_fields data structure
      // since it can be read by tasks that are being packed
      AutoLock o_lock(op_lock);
      local_fields.push_back(
          LocalFieldInfo(handle, fid, field_size, completion_event, serdez_id));
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &field_sizes,
                                      CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(fields.size() == field_sizes.size());
#endif
      for (unsigned idx = 0; idx < fields.size(); idx++)
        add_local_field(handle, fields[idx], field_sizes[idx], serdez_id);
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
                              info.fid, true/*local*/, info.serdez_id))
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
        runtime->issue_runtime_meta_task(rez.get_buffer(),
            rez.get_used_bytes(), HLR_RECLAIM_LOCAL_FIELD_ID,
            this, info.reclaim_event);
      }
    }

    //--------------------------------------------------------------------------
    ptr_t SingleTask::perform_safe_cast(IndexSpace handle, ptr_t pointer)
    //--------------------------------------------------------------------------
    {
      DomainPoint point(pointer.value);
      std::map<IndexSpace,Domain>::const_iterator finder = 
                                              safe_cast_domains.find(handle);
      if (finder != safe_cast_domains.end())
      {
        if (finder->second.contains(point))
          return pointer;
        else
          return ptr_t::nil();
      }
      Domain domain = runtime->get_index_space_domain(this, handle);
      // Save the result
      safe_cast_domains[handle] = domain;
      if (domain.contains(point))
        return pointer;
      else
        return ptr_t::nil();
    }
    
    //--------------------------------------------------------------------------
    DomainPoint SingleTask::perform_safe_cast(IndexSpace handle, 
                                              const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      std::map<IndexSpace,Domain>::const_iterator finder = 
                                              safe_cast_domains.find(handle);
      if (finder != safe_cast_domains.end())
      {
        if (finder->second.contains(point))
          return point;
        else
          return DomainPoint::nil();
      }
      Domain domain = runtime->get_index_space_domain(this, handle);
      // Save the result
      safe_cast_domains[handle] = domain;
      if (domain.contains(point))
        return point;
      else
        return DomainPoint::nil();
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
      runtime->forest->get_field_space_fields(handle.get_field_space(),
                                              new_req.instance_fields);
      new_req.privilege_fields.insert(new_req.instance_fields.begin(),
                                      new_req.instance_fields.end());
      // Now make a new region requirement and physical region
      regions.push_back(new_req);
      // Make a new unmapped physical region
      physical_regions.push_back(PhysicalRegion(
            legion_new<PhysicalRegionImpl>(regions.back(), Event::NO_EVENT,
                 false/*mapped*/, this, map_id, tag, is_leaf(), runtime)));
      physical_instances.push_back(InstanceSet());
      // Mark that this region was virtually mapped so we don't
      // try to close it when we are done executing.
      virtual_mapped.push_back(true);
      region_deleted.push_back(false);
      RemoteTask *outermost = find_outermost_context();
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
              legion_new<PhysicalRegionImpl>(regions.back(), Event::NO_EVENT,
                    false/*mapped*/, this, map_id, tag, is_leaf(), runtime)));
        physical_instances.push_back(InstanceSet());
        // Mark that the region was virtually mapped
        virtual_mapped.push_back(true);
        region_deleted.push_back(false);
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
      const RegionRequirement &req = op->get_requirement(); 
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int SingleTask::has_conflicting_regions(AttachOp *attach,
                                            bool &parent_conflict,
                                            bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = attach->get_requirement();
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int SingleTask::has_conflicting_internal(const RegionRequirement &req,
                                             bool &parent_conflict,
                                             bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      parent_conflict = false;
      inline_conflict = false;
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the regions data structure
      // but we are here so we aren't mutating
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
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the regions data structure
      // but we are here so we aren't mutating
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
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the regions data structure
      // but we are here so we aren't mutating
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
      const RegionRequirement &req = acquire->get_requirement();
      find_conflicting_internal(req, conflicting); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(ReleaseOp *release,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = release->get_requirement();
      find_conflicting_internal(req, conflicting);      
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(DependentPartitionOp *partition,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = partition->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_internal(const RegionRequirement &req,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the regions data structure
      // but we are here so we aren't mutating
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
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(FillOp *fill,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = fill->get_requirement();
      find_conflicting_internal(req, conflicting);
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
      // Don't need the lock because this is only accessed from 
      // the executing task context
      //
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
      // Don't need the lock because this is only accessed from the
      // executed task context
      for (std::list<PhysicalRegion>::iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
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
    int SingleTask::find_parent_region_req(const RegionRequirement &req,
                                           bool check_privilege /*= true*/)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // First check that the regions match
        if ((regions[idx].region != req.parent) || region_deleted[idx])
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((req.privilege & regions[idx].privilege) != req.privilege))
          continue;
        // Finally check that all the fields are contained
        bool dominated = true;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it !=
              req.privilege_fields.end(); it++)
        {
          if (regions[idx].privilege_fields.find(*it) ==
              regions[idx].privilege_fields.end())
          {
            dominated = false;
            break;
          }
        }
        if (!dominated)
          continue;
        return int(idx);
      }
      return -1;
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
      log_region.error("Parent task %s (ID %lld) of inline task %s "
                        "(ID %lld) does not have a region "
                        "requirement for region (%x,%x,%x) "
                        "as a parent of child task's region "
                        "requirement index %d", get_task_name(),
                        get_unique_id(), child->get_task_name(),
                        child->get_unique_id(), 
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
      log_index.error("Parent task %s (ID %lld) of inline task %s "
                            "(ID %lld) does not have an index space "
                            "requirement for index space %x "
                            "as a parent of chlid task's index requirement "
                            "index %d", get_task_name(), get_unique_id(),
                            child->get_task_name(), child->get_unique_id(),
                            child->indexes[index].handle.id, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_BAD_PARENT_INDEX);
      return 0;
    }

    //--------------------------------------------------------------------------
    PrivilegeMode SingleTask::find_parent_privilege_mode(unsigned idx)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      return regions[idx].privilege;
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
          std::vector<ColorPoint> path;
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
        std::vector<ColorPoint> path;
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
            std::vector<ColorPoint> path;
            if (!runtime->forest->compute_index_path(req.parent.index_space,
                                              req.region.index_space, path))
              return ERROR_BAD_REGION_PATH;
          }
          else
          {
            std::vector<ColorPoint> path;
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
          std::vector<ColorPoint> path;
          if (!runtime->forest->compute_index_path(req.parent.index_space,
                                              req.region.index_space, path))
            return ERROR_BAD_REGION_PATH;
        }
        else
        {
          std::vector<ColorPoint> path;
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
    bool SingleTask::has_tree_restriction(RegionTreeID tid, 
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // No need for the lock because we know that this processo
      // is serialized by the dependence analysis stage
      LegionMap<RegionTreeID,FieldMask>::aligned::const_iterator finder = 
        restricted_trees.find(tid);
      if ((finder != restricted_trees.end()) &&
          (!(finder->second * mask)))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_tree_restriction(RegionTreeID tid,
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock because we know access to this data
      // structure is serialized by the mapping process
      LegionMap<RegionTreeID,FieldMask>::aligned::iterator finder = 
        restricted_trees.find(tid);
      if (finder == restricted_trees.end())
        restricted_trees[tid] = mask;
      else
        finder->second |= mask;
    } 

    //--------------------------------------------------------------------------
    RegionTreeContext SingleTask::find_enclosing_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // See if this is one of our original regions or if it is a new one
      if (idx < initial_region_count)
        return context;
      else
        return find_outermost_context()->get_context();
    } 

    //--------------------------------------------------------------------------
    bool SingleTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
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
        if (early_map_task())
        {
          // See if we have a must epoch in which case
          // we can simply record ourselves and we are done
          if (must_epoch != NULL)
          {
            must_epoch->register_single_task(this, must_epoch_index);
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(!target_processors.empty());
#endif
            // See if this task is going to be sent
            // remotely in which case we need to do the
            // mapping now, otherwise we can defer it
            // until the task ends up on the target processor
            if (is_locally_mapped() && target_proc.exists() &&
                !runtime->is_local(target_proc))
            {
              if (perform_mapping())
              {
#ifdef DEBUG_HIGH_LEVEL
#ifndef NDEBUG
                bool still_local = 
#endif
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
      return success;
    } 

    //--------------------------------------------------------------------------
    void SingleTask::initialize_map_task_input(Mapper::MapTaskInput &input,
                                               Mapper::MapTaskOutput &output,
                                      std::vector<RegionTreeContext> &enclosing,
                                      std::vector<InstanceSet> &valid)
    //--------------------------------------------------------------------------
    {
      // Fill in our set of enclosing contexts if we need to
      if (enclosing.empty())
      {
        enclosing.resize(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
          enclosing[idx] = get_parent_context(idx);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing.size() == regions.size());
#endif
      // Do the traversals for all the non-early mapped regions and find
      // their valid instances, then fill in the mapper input structure
      valid.resize(regions.size());
      input.valid_instances.resize(regions.size());
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any early mapped regions
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
        {
          input.premapped_regions.push_back(idx);
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
          continue;
        InstanceSet &current_valid = valid[idx];
        perform_physical_traversal(idx, enclosing[idx], current_valid);
        // Now we can prepare this for mapping,
        // filter for visible memories if necessary
        if (regions[idx].is_no_access())
          prepare_for_mapping(current_valid, input.valid_instances[idx]);
        else
          prepare_for_mapping(current_valid, visible_memories,
                              input.valid_instances[idx]);
      }
      // Prepare the output too
      output.chosen_instances.resize(regions.size());
      output.chosen_variant = 0;
      output.postmap_task = false;
      output.task_priority = 0;
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_map_task_output(Mapper::MapTaskInput &input,
                                              Mapper::MapTaskOutput &output,
                                      std::vector<RegionTreeContext> &enclosing,
                                      std::vector<InstanceSet> &valid,
                                      bool must_epoch_map /*= false*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(enclosing.size() == regions.size());
#endif
      // first check the processors to make sure they are all on the
      // same node and of the same kind
      if (!Runtime::unsafe_mapper)
        validate_target_processors(output.target_procs, must_epoch_map);
      target_processors = output.target_procs;
      // fill in virtual_mapped
      virtual_mapped.resize(regions.size(),false);
      // Convert all the outputs into our set of physical instances and
      // validate them by checking the following properites:
      // - all are either pure virtual or pure physical 
      // - no missing fields
      // - all satisfy the region requirement
      // - all are visible from all the target processors
      physical_instances.resize(regions.size());
      // If we're doing safety checks, we need the set of memories
      // visible from all the target processors
      std::set<Memory> visible_memories;
      if (!Runtime::unsafe_mapper)
        find_visible_memories(visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // If it was early mapped, that was easy
        std::map<unsigned,InstanceSet>::const_iterator finder = 
          early_mapped_regions.find(idx);
        if (finder != early_mapped_regions.end())
        {
          physical_instances[idx] = finder->second;
          // Check to see if it is visible or not from the target processors
          if (!Runtime::unsafe_mapper && !regions[idx].is_no_access())
          {
            for (unsigned idx2 = 0; idx2 < finder->second.size(); idx2++)
            {
              Memory mem = finder->second[idx2].get_memory();
              if (visible_memories.find(mem) == visible_memories.end())
              {
                // Not visible from all target processors
                // Different error messages depending on the cause
                if (regions[idx].is_restricted())
                  log_run.error("Invalid mapper output from invocation of '%s' "
                                "on mapper %s. Mapper selected processor(s) "
                                "which restricted instance of region "
                                "requirement %d in memory " IDFMT " is not "
                                "visible for task %s (ID %lld).",
                                (must_epoch_map ? "map_must_epoch" : 
                                 "map_task"), mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id());
                else
                  log_run.error("Invalid mapper output from invocation of '%s' "
                                "on mapper %s. Mapper selected processor(s) " 
                                "for which premapped instance of region "
                                "requirement %d in memory " IDFMT " is not "
                                "visible for task %s (ID %lld).", 
                                (must_epoch_map ? "map_must_epoch" : 
                                  "map_task"), mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
        {
          virtual_mapped[idx] = true;
          continue;
        }
        // Do the conversion
        InstanceSet &result = physical_instances[idx];
        std::vector<FieldID> missing_fields;
        int composite_idx = 
          runtime->forest->physical_convert_mapping(regions[idx],
                                output.chosen_instances[idx], valid[idx], 
                                result, missing_fields);
        if (composite_idx >= 0)
        {
          // Everything better be all virtual or all real
          if (result.size() > 1)
          {
            log_run.error("Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Mapper specified mixed composite and "
                          "concrete instances for region requirement %d of "
                          "task %s (ID %lld). Only full concrete instances "
                          "or a single composite instance is supported.",
                          (must_epoch_map ? "map_must_epoch" : "map_task"),
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          if (IS_REDUCE(regions[idx]))
          {
            log_run.error("Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Illegal composite mapping requested on "
                          "region requirement %d of task %s (UID %lld) which "
                          "has only reduction privileges.", 
                          (must_epoch_map ? "map_must_epoch" : "map_task"), 
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_ILLEGAL_REDUCTION_VIRTUAL_MAPPING);
          }
          virtual_mapped[idx] = true;
        }
        if (!missing_fields.empty())
        {
          log_run.error("Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper failed to specify an instance for "
                        "%ld fields of region requirement %d on task %s "
                        "(ID %lld). The missing fields are listed below.",
                        (must_epoch_map ? "map_must_epoch" : "map_task"),
                        mapper->get_mapper_name(), missing_fields.size(),
                        idx, get_task_name(), get_unique_id());
          for (std::vector<FieldID>::const_iterator it = 
                missing_fields.begin(); it != missing_fields.end(); it++)
          {
            const void *name; size_t name_size;
            runtime->retrieve_semantic_information(
                regions[idx].region.get_field_space(), *it, NAME_SEMANTIC_TAG,
                name, name_size, false, false);
            log_run.error("Missing instance for field %s (FieldID: %d)",
                          static_cast<const char*>(name), *it);
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        // Skip checks if the mapper promises it is safe
        if (Runtime::unsafe_mapper)
          continue;
        for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
        {
          if (!runtime->forest->is_valid_mapping(result[idx2], regions[idx]))
          {
            // Doesn't satisfy the region requirement
            log_run.error("Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Mapper specified instance that does "
                          "not meet region requirement %d for task %s "
                          "(ID %lld).", (must_epoch_map ? "map_must_epoch" : 
                            "map_task"), mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
        if (!regions[idx].is_no_access())
        {
          for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
          {
            Memory mem = result[idx2].get_memory();
            if (visible_memories.find(mem) == visible_memories.end())
            {
              // Not visible from all target processors
              log_run.error("Invalid mapper output from invocation of '%s' on "
                            "mapper %s. Mapper selected an instance for region "
                            "requirement %d in memory " IDFMT " which is not "
                            "visible from the target processors for task %s "
                            "(ID %lld).", (must_epoch_map ? "map_must_epoch" : 
                              "map_task"), mapper->get_mapper_name(), idx,
                            mem.id, get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_INVALID_MAPPER_OUTPUT);
            }
          }
        }
      }
      early_mapped_regions.clear();
      // See whether the mapper picked a variant or a generator
      VariantImpl *variant_impl = NULL;
      if (output.chosen_variant > 0)
      {
        variant_impl = runtime->find_variant_impl(task_id, 
                                output.chosen_variant, true/*can fail*/);
      }
      else
      {
        // TODO: invoke a generator if one exists
        assert(false); 
      }
      if (variant_impl == NULL)
      {
        // If we couldn't find or make a variant that is bad
        log_run.error("Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper failed to specify a valid "
                      "task variant or generator capable of create a variant "
                      "implementation of task %s (ID %lld).",
                      (must_epoch_map ? "map_must_epoch" : "map_task"),
                      mapper->get_mapper_name(), get_task_name(),
                      get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      // Now that we know which variant to use, we can validate it
      if (!Runtime::unsafe_mapper)
        validate_variant_selection(variant_impl, must_epoch_map);
      // Record anything else that needs to be recorded 
      selected_variant = output.chosen_variant;
      task_priority = output.task_priority;
      perform_postmap = output.postmap_task;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(Event local_termination_event,
                                     MustEpochOp *must_epoch_op /*=NULL*/)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionTreeContext> enclosing_contexts(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        enclosing_contexts[idx] = get_parent_context(idx);
      // If we were already mapped by a must epoch operation, then we
      // are already know that our instance set is valid, so we can
      // skip actually calling the mapper
      if (must_epoch_op == NULL)
      {
        Mapper::MapTaskInput input;
        Mapper::MapTaskOutput output;
        // Initialize the mapping input which also does all the traversal
        // down to the target nodes
        std::vector<InstanceSet> valid_instances(regions.size());
        initialize_map_task_input(input, output, 
                                  enclosing_contexts, valid_instances);
        // Now we can invoke the mapper to do the mapping
        if (mapper == NULL)
          mapper = runtime->find_mapper(current_proc, map_id);
        mapper->invoke_map_task(this, &input, &output);
        // Now we can convert the mapper output into our physical instances
        finalize_map_task_output(input, output, 
                                 enclosing_contexts, valid_instances);
      }
      // Now that we are here, apply our state to the region tree
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
          continue;
        // See if we have to do any virtual mapping before registering
        if (virtual_mapped[idx])
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances[idx].size() == 1);
          assert(physical_instances[idx][0].is_composite_ref());
#endif
          runtime->forest->map_virtual_region(enclosing_contexts[idx],
                                              regions[idx],
                                              physical_instances[idx][0],
                                              get_version_info(idx)
#ifdef DEBUG_HIGH_LEVEL
                                              , idx, get_logging_name()
                                              , unique_op_id
#endif
                                              );
        }
        // Set the current mapping index before doing anything
        // that sould result in a copy
        set_current_mapping_index(idx);
        // apply the results of the mapping to the tree
        runtime->forest->physical_register_only(enclosing_contexts[idx],
                                    regions[idx], get_version_info(idx), 
                                    this, local_termination_event, 
                                    physical_instances[idx]
#ifdef DEBUG_HIGH_LEVEL
                                    , idx, get_logging_name()
                                    , unique_op_id
#endif
                                    );
      }
      if (perform_postmap)
        perform_post_mapping();
      // See if we need to invoke a post-map mapper call for this task
      return true;
    }  

    //--------------------------------------------------------------------------
    void SingleTask::perform_post_mapping(void)
    //--------------------------------------------------------------------------
    {
      Mapper::PostMapInput input;
      Mapper::PostMapOutput output;
      input.mapped_regions.resize(regions.size());
      input.valid_instances.resize(regions.size());
      output.chosen_instances.resize(regions.size());
      std::vector<InstanceSet> postmap_valid(regions.size());
      std::vector<RegionTreeContext> enclosing_contexts(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        enclosing_contexts[idx] = get_parent_context(idx);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Don't need to actually traverse very far, but we do need the
        // valid instances for all the regions
        RegionTreePath path;
        initialize_mapping_path(path, regions[idx], regions[idx].region);
        runtime->forest->physical_traverse_path(enclosing_contexts[idx],
                              path, regions[idx], get_version_info(idx), 
                              this, true/*valid*/, postmap_valid[idx]
#ifdef DEBUG_HIGH_LEVEL
                              , idx, get_logging_name()
                              , unique_op_id
#endif
                              );
        // No need to filter these because they are on the way out
        prepare_for_mapping(postmap_valid[idx], input.valid_instances[idx]);  
        prepare_for_mapping(physical_instances[idx], input.mapped_regions[idx]);
      }
      // Now we can do the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_post_map_task(this, &input, &output);
      // Check and register the results
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (output.chosen_instances.empty())
          continue;
        RegionRequirement &req = regions[idx];
        if (has_restrictions(idx, req.region))
        {
          log_run.warning("WARNING: Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "is restricted. The request is being ignored.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_NO_ACCESS(req))
        {
          log_run.warning("WARNING: Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has NO_ACCESS privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_REDUCE(req))
        {
          log_run.warning("WARNING: Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has REDUCE privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        // Convert the post-mapping  
        InstanceSet result;
        bool had_composite = 
          runtime->forest->physical_convert_postmapping(req,
              output.chosen_instances[idx], postmap_valid[idx], result); 
        if (had_composite)
        {
          log_run.warning("WARNING: Mapper %s requested a composite "
                          "instance be created for region requirement %d "
                          "of task %s (ID %lld) for a post mapping. The "
                          "request is being ignored.",
                          mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        // Register this with a no-event so that the instance can
        // be used as soon as it is valid from the copy to it
        runtime->forest->physical_register_only(enclosing_contexts[idx],
                          regions[idx], get_version_info(idx),
                          this, Event::NO_EVENT/*done immediately*/,
                          result
#ifdef DEBUG_HIGH_LEVEL
                          , idx, get_logging_name(), unique_op_id
#endif
                          );
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_region_tree_contexts(
                      const std::vector<RegionRequirement> &clone_requirements,
                      const std::vector<UserEvent> &unmap_events,
                      std::set<Event> &preconditions)
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
      std::map<PhysicalManager*,InstanceView*> top_views;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        // this better be true for single tasks
        assert(regions[idx].handle_type == SINGULAR);
#endif
        // If this is a NO_ACCESS or had no privilege fields we can skip this
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
          continue;
        // Only need to initialize the context if this is
        // not a leaf and it wasn't virtual mapped
        if (!virtual_mapped[idx])
        {
          runtime->forest->initialize_current_context(context,
              clone_requirements[idx], physical_instances[idx],
              unmap_events[idx], get_depth()+1, 
              unique_op_id, top_views);
#ifdef DEBUG_HIGH_LEVEL
          assert(!physical_instances[idx].empty());
#endif
          // If we need to add restricted coherence, do that now
          // Not we only need to do this for non-virtually mapped task
          if ((regions[idx].prop == SIMULTANEOUS) ||
              has_restrictions(idx, regions[idx].region)) 
            runtime->forest->restrict_user_coherence(context, this, 
                      regions[idx].region, regions[idx].privilege_fields);
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances[idx].has_composite_ref());
#endif
          const InstanceRef &ref = physical_instances[idx].get_composite_ref();
          CompositeView *composite_view = ref.get_composite_view();
          // First get any events necessary to make this view local
          if (!ref.is_local())
            composite_view->make_local(preconditions);
          // There is something really scary here so pay attention!
          // We're about to put a composite view from one context into
          // a different context. This composite view has captured
          // certain version numbers internally in its version info,
          // or possibly nested version infos. In theory this could
          // cause issues for the physical analysis since it sometimes
          // uses version numbers to avoid catching dependences when 
          // version numbers are the same. This would be really bad
          // if we tried to do this with version numbers from different
          // contexts. However, we know that it will never happen because
          // the physical analysis only permits this optimization for
          // WAR and WAW dependences, but composite instances are only
          // ever being read from, so all the dependences it will catch
          // are true dependences, therefore making it safe. :)
          runtime->forest->initialize_current_context(context,
              clone_requirements[idx], physical_instances[idx], composite_view);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(initial_region_count <= regions.size());
#endif
      for (unsigned idx = 0; idx < initial_region_count; idx++)
      {
        runtime->forest->invalidate_current_context(context,
                                                    regions[idx].region,
                                                    false/*logical only*/);
      }
      for (unsigned idx = initial_region_count; idx < regions.size(); idx++)
      {
        runtime->forest->invalidate_current_context(context,
                                                    regions[idx].region,
                                                    true/*logical only*/);
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
      VariantImpl *variant = 
        runtime->find_variant_impl(task_id, selected_variant);
      // STEP 1: Compute the precondition for the task launch
      std::set<Event> wait_on_events;
      // If we're debugging do one last check to make sure
      // that all the memories are visible on this processor
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!virtual_mapped[idx])
        {
          InstanceSet &instances = physical_instances[idx];
          // Get the event to wait on unless we are doing the inner
          // task optimization
          if (!variant->is_inner())
            instances.update_wait_on_events(wait_on_events);
        }
      }
      // Now add get all the other preconditions for the launch
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        FutureImpl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->get_ready_event());
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        GrantImpl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }

      // STEP 2: Set up the task's context
      index_deleted.resize(indexes.size(),false);
      region_deleted.resize(regions.size(),false);
      std::vector<UserEvent> unmap_events(regions.size());
      {
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(regions[idx].handle_type == SINGULAR);
#endif
          // Convert any WRITE_ONLY or WRITE_DISCARD privleges to READ_WRITE
          // This is necessary for any sub-operations which may need to rely
          // on our privileges for determining their own privileges such
          // as inline mappings or acquire and release operations
          if (regions[idx].privilege == WRITE_DISCARD)
            regions[idx].privilege = READ_WRITE;
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx])
          {
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            physical_regions.push_back(PhysicalRegion(
                  legion_new<PhysicalRegionImpl>(regions[idx],
                    Event::NO_EVENT, false/*mapped*/,
                    this, map_id, tag, false/*leaf*/, runtime)));
            // Don't switch coherence modes since we virtually
            // mapped it which means we will map in the parent's
            // context
          }
          else if (variant->is_inner())
          {
            // If this is an inner task then we don't map
            // the region with a physical region, but instead
            // we mark that the unmap event which marks when
            // the region can be used by child tasks should
            // be the ready event.
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            // Also make the region requirement read-write to force
            // people to wait on the value
            clone_requirements[idx].privilege = READ_WRITE;
            physical_regions.push_back(PhysicalRegion(
                  legion_new<PhysicalRegionImpl>(regions[idx],
                    Event::NO_EVENT, false/*mapped*/,
                    this, map_id, tag, false/*leaf*/, runtime)));
            unmap_events[idx] = UserEvent::create_user_event();
            // Trigger the user event when the region is 
            // actually ready to be used
            std::set<Event> ready_events;
            physical_instances[idx].update_wait_on_events(ready_events);
            Event precondition = Runtime::merge_events<false>(ready_events);
            Runtime::trigger_event<false>(unmap_events[idx], precondition);
          }
          else
          { 
            // If this is not virtual mapped, here is where we
            // switch coherence modes from whatever they are in
            // the enclosing context to exclusive within the
            // context of this task
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            physical_regions.push_back(PhysicalRegion(
                  legion_new<PhysicalRegionImpl>(clone_requirements[idx],
                    Event::NO_EVENT/*already mapped*/, true/*mapped*/,
                    this, map_id, tag, variant->is_leaf(), runtime)));
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
        if (!variant->is_leaf() || has_virtual_instances())
        {
          // Request a context from the runtime
          runtime->allocate_context(this);
          initial_region_count = regions.size();
          // Have the mapper configure the properties of the context
          context_configuration.max_window_size = 
            Runtime::initial_task_window_size;
          context_configuration.hysteresis_percentage = 
            Runtime::initial_task_window_hysteresis;
          context_configuration.max_outstanding_frames = 2; 
          context_configuration.min_tasks_to_schedule = 
            Runtime::initial_tasks_to_schedule;
          context_configuration.min_frames_to_schedule = 0;
          if (mapper == NULL)
            mapper = runtime->find_mapper(current_proc, map_id);
          mapper->invoke_configure_context(this, &context_configuration);
          // Do a little bit of checking on the output.  Make
          // sure that we only set one of the two cases so we
          // are counting by frames or by outstanding tasks.
          if ((context_configuration.min_tasks_to_schedule == 0) && 
              (context_configuration.min_frames_to_schedule == 0))
          {
            log_run.error("Invalid mapper output from call 'configure_context' "
                          "on mapper %s. One of 'min_tasks_to_schedule' and "
                          "'min_frames_to_schedule' must be non-zero for task "
                          "%s (ID %lld)", mapper->get_mapper_name(),
                          get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_INVALID_CONTEXT_CONFIGURATION);
          }
          // If we're counting by frames set min_tasks_to_schedule to zero
          if (context_configuration.min_frames_to_schedule > 0)
            context_configuration.min_tasks_to_schedule = 0;
          // otherwise we know min_frames_to_schedule is zero
#ifdef DEBUG_HIGH_LEVEL
          assert(context.exists());
          runtime->forest->check_context_state(context);
#endif
          // If we're going to do the inner task optimization
          // then when we initialize the contexts also pass in the
          // start condition so we can add a user off of which
          // all sub-users should be chained.
          initialize_region_tree_contexts(clone_requirements,
                                          unmap_events, wait_on_events);
          if (!variant->is_inner())
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (!virtual_mapped[idx])
              {
                physical_regions[idx].impl->reset_references(
                    physical_instances[idx], unmap_events[idx]);
              }
            }
          } 
        }
        else
        {
          // Leaf and all non-virtual mappings
          // Mark that all the local instances are empty
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!virtual_mapped[idx])
            {
              physical_regions[idx].impl->reset_references(
                  physical_instances[idx], unmap_events[idx]);
            }
          }
        }
      }
      // Merge together all the events for the start condition 
      Event start_condition = Runtime::merge_events<false>(wait_on_events);
      // Take all the locks in order in the proper way
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          start_condition = 
            Runtime::acquire_reservation<false>(it->first, it->second,
                                                start_condition);
        }
      }
      // STEP 3: Finally we get to launch the task
#ifdef LEGION_SPY
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        LegionSpy::log_task_instance_requirement(get_unique_id(), idx,
                                 regions[idx].region.get_index_space().id);
      }
      {
        std::set<Event> unmap_set;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx])
            unmap_set.insert(unmap_events[idx]);
        }
        Event all_unmap_event = Runtime::merge_events<false>(unmap_set);
        // Log an implicit dependence on the parent's start event
        LegionSpy::log_implicit_dependence(parent_ctx->get_start_event(),
                                           start_condition);
        LegionSpy::log_op_events(get_unique_id(), 
                                 start_condition, all_unmap_event);
        this->legion_spy_start = start_condition; 
        // Record the start
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx])
          {
            LegionSpy::log_event_dependence(all_unmap_event, unmap_events[idx]);
            // Log an implicit dependence on the parent's start event
            LegionSpy::log_event_dependence(unmap_events[idx],
                                               get_task_completion());
          }
        }
        LegionSpy::log_implicit_dependence(get_task_completion(),
                                           parent_ctx->get_task_completion());
      }
#endif
      // Mark that we have an outstanding task in this context 
      parent_ctx->increment_pending();
      // If this is a leaf task and we have no virtual instances
      // and the SingleTask sub-type says it is ok
      // we can trigger the task's completion event as soon as
      // the task is done running.  We first need to mark that this
      // is going to occur before actually launching the task to 
      // avoid the race.
      bool perform_chaining_optimization = false; 
      UserEvent chain_complete_event;
      if (variant->is_leaf() && !has_virtual_instances() &&
          can_early_complete(chain_complete_event))
        perform_chaining_optimization = true;
      // Note there is a potential scary race condition to be aware of here: 
      // once we launch this task it's possible for this task to run and 
      // clean up before we finish the execution of this function thereby
      // invalidating this SingleTask object's fields.  This means
      // that we need to save any variables we need for after the task
      // launch here on the stack before they can be invalidated.
      Event term_event = get_task_completion();
#ifdef DEBUG_HIGH_LEVEL
      assert(!target_processors.empty());
#endif
      Processor launch_processor = target_processors[0];
      if (target_processors.size() > 1)
      {
        // Find the processor group for all the target processors
        launch_processor = runtime->find_processor_group(target_processors);
      }
      Realm::ProfilingRequestSet profiling_requests;
      // If the mapper requested profiling add that now too
      // TODO: fill in the profiling requests
#if 0
      if (profile_task)
      {
        // Make a user event for signaling when we've reporting profiling
        MapperProfilingInfo info;
        info.task = this;
        info.profiling_done = UserEvent::create_user_event();
        Realm::ProfilingRequest &req = profiling_requests.add_request(
                                        runtime->find_utility_group(),
                                        HLR_MAPPER_PROFILING_ID, 
                                        &info, sizeof(info));
        req.add_measurement<
          Realm::ProfilingMeasurements::OperationTimeline>();
        // Record the event for when we are done profiling
        profiling_done = info.profiling_done;
      } 
#endif
      Event task_launch_event = variant->dispatch_task(launch_processor, this,
                          start_condition, task_priority, profiling_requests);
      // Finish the chaining optimization if we're doing it
      if (perform_chaining_optimization)
        Runtime::trigger_event<false>(chain_complete_event, task_launch_event);
      // STEP 4: After we've launched the task, then we have to release any 
      // locks that we took for while the task was running.  
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          Runtime::release_reservation<false>(it->first, term_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& SingleTask::begin_task(void)
    //--------------------------------------------------------------------------
    {
      // Switch over the executing processor to the one
      // that has actually been assigned to run this task.
      executing_processor = Processor::get_executing_processor();
#ifdef DEBUG_HIGH_LEVEL
      log_task.debug("Task %s (ID %lld) starting on processor " IDFMT "",
                    get_task_name(), get_unique_id(), executing_processor.id);
      assert(regions.size() == physical_regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == region_deleted.size());
#endif
#ifdef LEGION_SPY
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        for (unsigned idx2 = 0; idx2 < physical_instances[idx].size(); idx2++)
          LegionSpy::log_op_user(unique_op_id, idx, 
              physical_instances[idx][idx2].get_manager()->get_instance().id);
      }
      LegionSpy::log_op_proc_user(unique_op_id, executing_processor.id);
#endif
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      {
        DecrementArgs decrement_args;
        decrement_args.hlr_id = HLR_DECREMENT_PENDING_TASK_ID;
        decrement_args.parent_ctx = parent_ctx;
        pending_done = runtime->issue_runtime_meta_task(&decrement_args, 
            sizeof(decrement_args), HLR_DECREMENT_PENDING_TASK_ID, this);
      }
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    void SingleTask::notify_profiling_results(Realm::ProfilingResponse &results)
    //--------------------------------------------------------------------------
    {
      // TODO: Save the results into the task profiling info 
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::process_mapper_profiling(const void *buffer, 
                                                         size_t size)
    //--------------------------------------------------------------------------
    {
      Realm::ProfilingResponse response(buffer, size);
#ifdef DEBUG_HIGH_LEVEL
      assert(response.user_data_size() == sizeof(MapperProfilingInfo));
#endif
      const MapperProfilingInfo *info = 
        (const MapperProfilingInfo*)response.user_data();
      // Record the results
      info->task->notify_profiling_results(response);
      // Then trigger the event saying we are done
      info->profiling_done.trigger();
    }

    //--------------------------------------------------------------------------
    void SingleTask::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == region_deleted.size());
#endif
      // Quick check to make sure the user didn't forget to end a trace
      if (current_trace != NULL)
      {
        log_task.error("Task %s (UID %lld) failed to end trace before exiting!",
                        get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INCOMPLETE_TRACE);
      }
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

      if (!is_leaf() || has_virtual_instances())
      {
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (IS_READ_ONLY(regions[idx]) || IS_NO_ACCESS(regions[idx]) ||
              region_deleted[idx])
            continue;
          if (!virtual_mapped[idx])
          {
            if (!is_leaf())
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!physical_instances[idx].empty());
#endif
              PostCloseOp *close_op = 
                runtime->get_available_post_close_op(true);
              close_op->initialize(this, idx);
              runtime->add_to_dependence_queue(executing_processor, close_op);
            }
          }
          else if (physical_instances[idx].has_composite_ref())
          {
            // Make a virtual close op to close up the instance
            VirtualCloseOp *close_op = 
              runtime->get_available_virtual_close_op(true);
            close_op->initialize(this, idx);
            runtime->add_to_dependence_queue(executing_processor, close_op);
          }
        }
      } 
      // See if we want to move the rest of this computation onto
      // the utility processor. We also need to be sure that we have 
      // registered all of our operations before we can do the post end task
      if (runtime->has_explicit_utility_procs || 
          !last_registration.has_triggered())
      {
        PostEndArgs post_end_args;
        post_end_args.hlr_id = HLR_POST_END_ID;
        post_end_args.proxy_this = this;
        post_end_args.result_size = res_size;
        // If it is not owned make a copy
        if (!owned)
        {
          post_end_args.result = malloc(res_size);
          memcpy(post_end_args.result, res, res_size);
        }
        else
          post_end_args.result = const_cast<void*>(res);
        runtime->issue_runtime_meta_task(&post_end_args, sizeof(post_end_args),
                                     HLR_POST_END_ID, this, last_registration);
      }
      else
        post_end_task(res, res_size, owned);
    }

    //--------------------------------------------------------------------------
    void SingleTask::post_end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
      // Handle the future result
      handle_future(res, res_size, owned);
      // If we weren't a leaf task, compute the conditions for being mapped
      // which is that all of our children are now mapped
      // Also test for whether we need to trigger any of our child
      // complete or committed operations before marking that we
      // are done executing
      bool need_complete = false;
      bool need_commit = false;
      if (!is_leaf())
      {
        std::set<Event> preconditions;
        {
          AutoLock o_lock(op_lock);
          // Only need to do this for executing and executed children
          // We know that any complete children are done
          for (std::set<Operation*>::const_iterator it = 
                executing_children.begin(); it != 
                executing_children.end(); it++)
          {
            preconditions.insert((*it)->get_mapped_event());
          }
          for (std::set<Operation*>::const_iterator it = 
                executed_children.begin(); it != executed_children.end(); it++)
          {
            preconditions.insert((*it)->get_mapped_event());
          }
#ifdef DEBUG_HIGH_LEVEL
          assert(!task_executed);
#endif
          // Now that we know the last registration has taken place we
          // can mark that we are done executing
          task_executed = true;
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
        if (!preconditions.empty())
          handle_post_mapped(Runtime::merge_events<true>(preconditions));
        else
          handle_post_mapped();
      }
      else
      {
        // Handle the leaf task case
        AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(!task_executed);
#endif
        // Now that we know the last registration has taken place we
        // can mark that we are done executing
        task_executed = true;
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
      // Mark that we are done executing this operation
      // We're not actually done until we have registered our pending
      // decrement of our parent task and recorded any profiling
      if (!pending_done.has_triggered() || !profiling_done.has_triggered())
      {
        Event exec_precondition = 
          Runtime::merge_events<true>(pending_done, profiling_done);
        complete_execution(exec_precondition);
      }
      else
        complete_execution();
      if (need_complete)
        trigger_children_complete();
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SingleTask::unmap_all_mapped_regions(void)
    //--------------------------------------------------------------------------
    {
      // Unmap any of our original physical instances
      for (std::vector<PhysicalRegion>::const_iterator it = 
            physical_regions.begin(); it != physical_regions.end(); it++)
      {
        if (it->impl->is_mapped())
          it->impl->unmap_region();
      }
      // Also unmap any of our inline mapped physical regions
      for (LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::
            tracked::const_iterator it = inline_regions.begin();
            it != inline_regions.end(); it++)
      {
        if (it->impl->is_mapped())
          it->impl->unmap_region();
      }
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
      minimal_points_assigned = 0;
      redop = 0;
      reduction_op = NULL;
      serdez_redop_fns = NULL;
      reduction_state_size = 0;
      reduction_state = NULL;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate_multi(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->profiler != NULL)
        runtime->profiler->register_multi_task(this, task_id);
      deactivate_task();
      if (reduction_state != NULL)
      {
        legion_free(REDUCTION_ALLOC, reduction_state, reduction_state_size);
        reduction_state = NULL;
        reduction_state_size = 0;
      }
      for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
            minimal_points.begin(); it != minimal_points.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->second != NULL);
#endif
        delete it->second;
      }
      minimal_points.clear(); 
      slices.clear(); 
      version_infos.clear();
      restrict_infos.clear();
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
      sliced = true;
      stealable = false; // cannot steal something that has been sliced
      Mapper::SliceTaskInput input;
      Mapper::SliceTaskOutput output;
      input.domain = index_domain;
      output.verify_correctness = false;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_slice_task(this, &input, &output);
      if (output.slices.empty())
      {
        log_run.error("Invalid mapper output from invocation of 'slice_task' "
                      "call on mapper %s. Mapper failed to specify an slices "
                      "for task %s (ID %lld).", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
      }

#ifdef DEBUG_HIGH_LEVEL
      assert(minimal_points_assigned == 0);
#endif
      for (unsigned idx = 0; idx < output.slices.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        // Check to make sure the domain is not empty
        const Domain &d = output.slices[idx].domain;
        bool empty = false;
        switch (d.dim)
        {
          case 0:
            {
              if (d.get_volume() <= 0)
                empty = true;
              break;
            }
          case 1:
            {
              Rect<1> rec = d.get_rect<1>();
              if (rec.volume() <= 0)
                empty = true;
              break;
            }
          case 2:
            {
              Rect<2> rec = d.get_rect<2>();
              if (rec.volume() <= 0)
                empty = true;
              break;
            }
          case 3:
            {
              Rect<3> rec = d.get_rect<3>();
              if (rec.volume() <= 0)
                empty = true;
              break;
            }
          default:
            assert(false);
        }
        if (empty)
        {
          log_run.error("Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned an empty slice for task "
                        "%s (ID %lld).", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id());
          assert(false);
          exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
        }
#endif
        SliceTask *slice = this->clone_as_slice_task(output.slices[idx].domain,
                                                   output.slices[idx].proc,
                                                   output.slices[idx].recurse,
                                                   output.slices[idx].stealable,
                                                   output.slices.size());
        slices.push_back(slice);
      }
      // If the volumes don't match, then something bad happend in the mapper
      if (minimal_points_assigned != minimal_points.size())
      {
        log_run.error("Invalid mapper output from invocation of 'slice_task' "
                      "on mapper %s. Mapper returned slices with a total "
                      "volume %d that does not match the expected volume of "
                      "%ld when slicing task %s (ID %lld).", 
                      mapper->get_mapper_name(), minimal_points_assigned,
                      index_domain.get_volume(), 
                      get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
      }
      else
        minimal_points.clear();
      bool success = trigger_slices(); 
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
      this->redop = rhs->redop;
      if (this->redop != 0)
      {
        this->reduction_op = rhs->reduction_op;
        this->serdez_redop_fns = rhs->serdez_redop_fns;
        initialize_reduction_state();
      }
      // Take ownership of all the points
      rhs->assign_points(this, d);
      this->restrict_infos = rhs->restrict_infos;
      // Copy over the version infos that we need, we can skip this if
      // we are remote and locally mapped
      if (!is_remote() || !is_locally_mapped())
      {
        this->version_infos.resize(rhs->version_infos.size());
        for (unsigned idx = 0; idx < this->version_infos.size(); idx++)
        {
          if (IS_NO_ACCESS(regions[idx]))
            continue;
          if (regions[idx].handle_type != SINGULAR)
          {
            VersionInfo &local_info = this->version_infos[idx];
            const VersionInfo &rhs_info = rhs->version_infos[idx];
            for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                  minimal_points.begin(); it != minimal_points.end(); it++)
            {
              local_info.clone_version_info(runtime->forest, 
                  it->second->find_logical_region(idx), rhs_info,
                  IS_WRITE(regions[idx]));
            }
          }
          else // non-projection we can copy over the normal way
            this->version_infos[idx] = rhs->version_infos[idx];
        }
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::assign_points(MultiTask *target, const Domain &d)
    //--------------------------------------------------------------------------
    {
      for (Domain::DomainPointIterator itr(d); itr; itr++)
      {
        std::map<DomainPoint,MinimalPoint*>::iterator finder = 
          minimal_points.find(itr.p);
        if (finder == minimal_points.end())
        {
          log_run.error("Invalid mapper domain slice result for mapper %d "
                        "on processor " IDFMT " for task %s (ID %lld). "
                        "Mapper returned slices with additional points "
                        "beyond the original index space.", map_id,
                        current_proc.id, get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
        }
        if (finder->second == NULL)
        {
          log_run.error("Invalid mapper domain slice result for mapper %d "
                        "on processor " IDFMT " for task %s (ID %lld). "
                        "Mapper returned overlapping slices.", map_id,
                        current_proc.id, get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
        }
        target->add_point(itr.p, finder->second);
        finder->second = NULL; // mark null to avoid duplicate gives
        minimal_points_assigned++;
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::add_point(const DomainPoint &p, MinimalPoint *point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(minimal_points.find(p) == minimal_points.end());
#endif
      minimal_points[p] = point;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
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
        // Not remote
        if (early_map_task())
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
                if (target_proc.exists() && !runtime->is_local(target_proc))
                {
                  if (perform_mapping())
                  {
#ifdef DEBUG_HIGH_LEVEL
#ifndef NDEBUG
                    bool still_local = 
#endif
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
    void MultiTask::pack_multi_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(sliced);
      rez.serialize(redop);
      rez.serialize<size_t>(minimal_points.size());
      for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
            minimal_points.begin(); it != minimal_points.end(); it++)
      {
        pack_point(rez, it->first);
        it->second->pack(rez);
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_base_task(derez); 
      derez.deserialize(sliced);
      derez.deserialize(redop);
      if (redop > 0)
      {
        reduction_op = Runtime::get_reduction_op(redop);
        serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
        initialize_reduction_state();
      }
      size_t num_points;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint p;
        unpack_point(derez, p);
        MinimalPoint *point = new MinimalPoint();
        point->unpack(derez);
        minimal_points[p] = point;
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
      reduction_state = legion_malloc(REDUCTION_ALLOC, reduction_state_size);
      // If we need to initialize specially, then we do that with a serdez fn
      if (serdez_redop_fns != NULL)
        (*(serdez_redop_fns->init_fn))(reduction_op, reduction_state, 
                                       reduction_state_size);
      else
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
#endif
      // Perform the reduction, see if we have to do serdez reductions
      if (serdez_redop_fns != NULL)
      {
        // Need to hold the lock to make the serialize/deserialize
        // process atomic
        AutoLock o_lock(op_lock);
        (*(serdez_redop_fns->fold_fn))(reduction_op, reduction_state,
                                       reduction_state_size, result);
      }
      else
        reduction_op->fold(reduction_state, result, 1, exclusive);

      // If we're the owner, then free the memory
      if (owner)
        free(const_cast<void*>(result));
    } 

    //--------------------------------------------------------------------------
    VersionInfo& MultiTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    void MultiTask::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < version_infos.size());
#endif
      version_infos[idx].recapture_state();
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
      predicate_false_result = NULL;
      predicate_false_size = 0;
      orig_task = this;
      remote_owner_uid = 0;
      remote_parent_ctx = NULL;
      remote_completion_event = get_completion_event();
      remote_unique_id = get_unique_id();
      sent_remotely = false;
      top_level_task = false;
      is_inline = false;
      has_remote_subtasks = false;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // If we are the top_level task then deactivate our parent context
      if (top_level_task)
        parent_ctx->deactivate();
      deactivate_single();
      if (!remote_instances.empty())
      {
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
        }
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          if (it->first == runtime->address_space)
            runtime->release_remote_context(local_uid);
          else
            runtime->send_free_remote_context(it->first, rez);
        }
        remote_instances.clear();
      }
      if (future_store != NULL)
      {
        legion_free(FUTURE_RESULT_ALLOC, future_store, future_size);
        future_store = NULL;
        future_size = 0;
      }
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      // Remove our reference on the future
      result = Future();
      predicate_false_future = Future();
      rerun_analysis_requirements.clear();
      privilege_paths.clear();
      version_infos.clear();
      restrict_infos.clear();
      map_applied_conditions.clear();
      // Read this before freeing the task
      // Should be safe, but we'll be careful
      bool is_top_level_task = top_level_task;
      runtime->free_individual_task(this);
      // If we are the top-level-task and we are deactivated then
      // it is now safe to shutdown the machine
      if (is_top_level_task)
        runtime->decrement_outstanding_top_level_tasks();
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
        regions[idx] = launcher.region_requirements[idx];
      futures = launcher.futures;
      grants = launcher.grants;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.argument.get_size();
      if (arglen > 0)
      {
        args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,launcher.argument.get_ptr(),arglen);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      index_point = launcher.point;
      is_index_space = false;
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      remote_owner_uid = ctx->get_unique_id();
      remote_parent_ctx = parent_ctx;
      if (launcher.predicate != Predicate::TRUE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
          predicate_false_future = launcher.predicate_false_future;
        else
        {
          predicate_false_size = launcher.predicate_false_result.get_size();
          if (predicate_false_size == 0)
          {
            // TODO: Put this check back in
#if 0
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
                                  get_task_name(), ctx->get_task_name(),
                                  ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
            }
#endif
          }
          else
          {
            // TODO: Put this check back in
#if 0
            if (predicate_false_size != variants->return_size)
            {
              log_run.error("Predicated task launch for task %s "
                                 "in parent task %s (UID %lld) has predicated "
                                 "false return type of size %ld bytes, but the "
                                 "expected return size is %ld bytes.",
                                 get_task_name(), parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 predicate_false_size, variants->return_size);
#ifdef DEBUG_HIGH_LEVEL
              assert(false);
#endif
              exit(ERROR_PREDICATE_RESULT_SIZE_MISMATCH);
            }
#endif
#ifdef DEBUG_HIGH_LEVEL
            assert(predicate_false_result == NULL);
#endif
            predicate_false_result = 
              legion_malloc(PREDICATE_ALLOC, predicate_false_size);
            memcpy(predicate_false_result, 
                   launcher.predicate_false_result.get_ptr(),
                   predicate_false_size);
          }
        }
      }
      if (check_privileges)
        perform_privilege_checks();
      remote_outermost_context = 
        find_outermost_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_outermost_context.exists());
#endif
      initialize_paths(); 
      // Get a future from the parent context to use as the result
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(!top_level_task), 
            runtime->address_space, runtime->address_space, this));
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                       unique_op_id,
                                       task_id, get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
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
        regions[idx] = region_requirements[idx];
      arglen = arg.get_size();
      if (arglen > 0)
      {
        args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,arg.get_ptr(),arglen);
      }
      map_id = mid;
      tag = t;
      is_index_space = false;
      initialize_base_task(ctx, track, pred, task_id);
      remote_owner_uid = ctx->get_unique_id();
      remote_parent_ctx = parent_ctx;
      if (check_privileges)
        perform_privilege_checks();
      remote_outermost_context = 
        find_outermost_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_outermost_context.exists());
#endif
      initialize_paths();
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(!top_level_task), 
            runtime->address_space, runtime->address_space, this));
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                       unique_op_id,
                                       task_id, get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
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
    }

    //--------------------------------------------------------------------------
    void IndividualTask::set_top_level(void)
    //--------------------------------------------------------------------------
    {
      this->top_level_task = true;
      // Top-level tasks never do dependence analysis, so we
      // need to complete those stages now
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(privilege_paths.size() == regions.size());
#endif
      // First compute the parent indexes
      compute_parent_indexes();
      begin_dependence_analysis();
      // If we are tracing we need to record any aliased region requirements
      if (is_tracing())
        record_aliased_region_requirements(get_trace());
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->impl != NULL);
#endif
        it->impl->register_dependence(this);
#ifdef LEGION_SPY
        if (it->impl->producer_op != NULL)
          LegionSpy::log_mapping_dependence(
              parent_ctx->get_unique_id(), it->impl->producer_uid, 0,
              get_unique_id(), 0, TRUE_DEPENDENCE);
#endif
      }
      // Also have to register any dependences on our predicate
      register_predicate_dependence();
      version_infos.resize(regions.size());
      restrict_infos.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(this, idx, regions[idx], 
                                                     version_infos[idx],
                                                     restrict_infos[idx],
                                                     privilege_paths[idx]);
      }
      // See if we have any requirements that interferred with a close
      // operation that was generated by a later region requirement
      // and therefore needs to be re-analyzed
      if (!rerun_analysis_requirements.empty())
      {
        // Make a local copy to avoid invalidating the iterator
        std::vector<unsigned> rerun(rerun_analysis_requirements.begin(),
                                    rerun_analysis_requirements.end());
#ifdef DEBUG_HIGH_LEVEL
        rerun_analysis_requirements.clear();
#endif
        for (std::vector<unsigned>::const_iterator it = 
              rerun.begin(); it != rerun.end(); it++)
        {
          // Clear out the version infos so we get new data
          VersionInfo &version_info = version_infos[*it];
          version_info.release();
          version_info.clear();
          runtime->forest->perform_dependence_analysis(this, *it, regions[*it],
                                                       version_info,
                                                       restrict_infos[*it],
                                                       privilege_paths[*it]);
          // If we still have re-run requirements, then we have
          // interfering region requirements so warn the user
          if (!rerun_analysis_requirements.empty())
          {
            for (std::set<unsigned>::const_iterator it2 = 
                  rerun_analysis_requirements.begin(); it2 != 
                  rerun_analysis_requirements.end(); it2++)
            {
              report_interfering_requirements(*it, *it2);
            }
            rerun_analysis_requirements.clear();
          }
        }
      }
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions; 
      if (is_remote())
      {
        // If we're remote and locally mapped, we are done
        if (is_locally_mapped())
        {
          ready_event.trigger();
          return;
        }
        // Otherwise request state for anything 
        // that was not early mapped 
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          if (early_mapped_regions.find(idx) == early_mapped_regions.end())
            version_infos[idx].make_local(preconditions, runtime->forest,
                                          get_parent_context(idx).get_id());
        }
      }
      else
      {
        // We're still local, see if we are locally mapped or not
        if (is_locally_mapped())
        {
          // If we're locally mapping, we need everything now
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
          {
            version_infos[idx].make_local(preconditions, runtime->forest,
                                          get_parent_context(idx).get_id());
          }
        }
        else
        {
          // We only early map restricted regions for individual tasks
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
          {
            if (!regions[idx].is_restricted())
              continue;
            version_infos[idx].make_local(preconditions, runtime->forest,
                                          get_parent_context(idx).get_id());
          }
        }
      }
      if (preconditions.empty())
        ready_event.trigger();
      else
        Runtime::trigger_event<true>(ready_event,
            Runtime::merge_events<true>(preconditions));
    }

    //--------------------------------------------------------------------------
    void IndividualTask::report_interfering_requirements(unsigned idx1, 
                                                         unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 0
      log_run.error("Aliased region requirements for individual tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are interfering.", idx1, idx2, get_task_name(),
                          get_unique_id(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_ALIASED_REGION_REQUIREMENTS);
#else
      log_run.warning("Region requirements %d and %d of individual task "
                      "%s (UID %lld) in parent task %s (UID %lld) are "
                      "interfering.  This behavior is currently "
                      "undefined. You better really know what you are "
                      "doing.", idx1, idx2, get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id());
#endif
    }

    //--------------------------------------------------------------------------
    void IndividualTask::report_interfering_close_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
      rerun_analysis_requirements.insert(idx);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      bool trigger = true;
      // Set the future to the false result
      if (predicate_false_future.impl != NULL)
      {
        Event wait_on = predicate_false_future.impl->get_ready_event();
        if (wait_on.has_triggered())
        {
          const size_t result_size = 
            check_future_size(predicate_false_future.impl);
          if (result_size > 0)
            result.impl->set_result(
                predicate_false_future.impl->get_untyped_result(),
                result_size, false/*own*/);
        }
        else
        {
          // Add references so they aren't garbage collected
          result.impl->add_base_gc_ref(DEFERRED_TASK_REF);
          predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF);
          Runtime::DeferredFutureSetArgs args;
          args.hlr_id = HLR_DEFERRED_FUTURE_SET_ID;
          args.target = result.impl;
          args.result = predicate_false_future.impl;
          args.task_op = this;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_FUTURE_SET_ID,
                                           this, wait_on);
          trigger = false;
        }
      }
      else
      {
        if (predicate_false_size > 0)
          result.impl->set_result(predicate_false_result,
                                  predicate_false_size, false/*own*/);
      }
      // Then clean up this task instance
      if (trigger)
        complete_execution();
      complete_mapping();
      trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // For individual tasks we always early map restricted regions
      std::vector<unsigned> early_map_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].is_restricted())
          early_map_indexes.push_back(idx);
      }
      if (!early_map_indexes.empty())
        return early_map_regions(map_applied_conditions, early_map_indexes);
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(MustEpochOp *must_epoch_owner/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // Now try to do the mapping, we can just use our completion
      // event since we know this task will object will be active
      // throughout the duration of the computation
      bool map_success = map_all_regions(get_task_completion(), 
                                         must_epoch_owner);
      if (map_success)
      {
        // If we mapped, then we are no longer stealable
        stealable = false;
        // Also flush out physical regions
        if (is_remote())
        {
          AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
            if (!virtual_mapped[idx])
              version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                           owner_space, map_applied_conditions);
        }
        else
        {
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
            if (!virtual_mapped[idx])
              version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                runtime->address_space, map_applied_conditions);
        }
        // If we succeeded in mapping and everything was mapped
        // then we get to mark that we are done mapping
        if (is_leaf())
        {
          Event applied_condition = Event::NO_EVENT;
          if (!map_applied_conditions.empty())
          {
            applied_condition = 
              Runtime::merge_events<true>(map_applied_conditions);
            map_applied_conditions.clear();
          }
          if (is_remote())
          {
            // Send back the message saying that we finished mapping
            Serializer rez;
            // Only need to send back the pointer to the task instance
            rez.serialize(orig_task);
            rez.serialize(applied_condition);
            runtime->send_individual_remote_mapped(orig_proc, rez);
          }
          // Mark that we have completed mapping
          complete_mapping(applied_condition);
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_locally) && stealable);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::has_restrictions(unsigned idx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < restrict_infos.size());
#endif
      // We know that if there are any restrictions they directly apply
      return restrict_infos[idx].has_restrictions();
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
    void IndividualTask::return_virtual_instance(unsigned index,
                                                 InstanceSet &refs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(refs.size() == 1);
      assert(refs[0].is_composite_ref());
#endif
      RegionTreeContext virtual_ctx = get_parent_context(index);
      // Put this in an instance set and then register it
      // Have to control access to the version info data structure
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(virtual_mapped[index]);
#endif
      runtime->forest->physical_register_only(virtual_ctx, regions[index],
                                              version_infos[index], this,
                                              Event::NO_EVENT, refs
#ifdef DEBUG_HIGH_LEVEL
                                              , index, get_logging_name()
                                              , unique_op_id
#endif
                                              );
    }

    //--------------------------------------------------------------------------
    VersionInfo& IndividualTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndividualTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < privilege_paths.size());
#endif
      return privilege_paths[idx];
    }

    //--------------------------------------------------------------------------
    void IndividualTask::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < version_infos.size());
#endif
      version_infos[idx].recapture_state();
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
    RemoteTask* IndividualTask::find_outermost_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->find_outermost_context();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::has_remote_state(void) const
    //--------------------------------------------------------------------------
    {
      return has_remote_subtasks;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_remote_state(void)
    //--------------------------------------------------------------------------
    {
      // Monotonic so no need to hold the lock 
      has_remote_subtasks = true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_remote_instance(AddressSpaceID remote_instance,
                                                RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      if (!is_remote())
      {
        // Pass back our created and deleted operations
        if (!top_level_task)
          return_privilege_state(parent_ctx);
#ifdef DEBUG_HIGH_LEVEL
        else
        {
          // Pass back the leaked top-level regions so that the outtermost
          // context knows how to clear its state when it is cleaning up
          // Only need to do this in debug case since it won't matter otherwise
          RemoteTask *outer = static_cast<RemoteTask*>(parent_ctx);
          for (std::set<LogicalRegion>::const_iterator it = 
                created_regions.begin(); it != created_regions.end(); it++)
          {
            outer->add_top_region(*it);
          }
        }
#endif

        // The future has already been set so just trigger it
        result.impl->complete_future();
      }
      else
      {
        Serializer rez;
        pack_remote_complete(rez);
        runtime->send_individual_remote_complete(orig_proc,rez);
      }
      
      // Invalidate any state that we had if we didn't already
      if (context.exists() && (!is_leaf() || has_virtual_instances()))
        invalidate_region_tree_contexts();
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
      // Mark that this operation is complete
      complete_operation();
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
      // We can release our version infos now
      for (std::vector<VersionInfo>::iterator it = version_infos.begin();
            it != version_infos.end(); it++)
      {
        it->release();
      }
      commit_operation(true/*deactivate*/);
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
          future_store = legion_malloc(FUTURE_RESULT_ALLOC, future_size);
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
    void IndividualTask::handle_post_mapped(Event mapped_precondition)
    //--------------------------------------------------------------------------
    {
      // If this is either a remote task or we have virtual mappings, then
      // we need to wait before completing our mapping
      if ((is_remote() || has_virtual_instances()) && 
          !mapped_precondition.has_triggered())
      {
        SingleTask::DeferredPostMappedArgs args;
        args.hlr_id = HLR_DEFERRED_POST_MAPPED_ID;
        args.task = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_POST_MAPPED_ID,
                                         this, mapped_precondition);
        return;
      }
      // If we have any virtual instances then we need to apply
      // the changes for them now
      if (has_virtual_instances())
      {
        if (is_remote())
        {
          AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
          for (unsigned idx = 0; idx < physical_instances.size(); idx++)
          {
            if (!physical_instances[idx].has_composite_ref())
              continue;
#ifdef DEBUG_HIGH_LEVEL
            assert(virtual_mapped[idx]);
#endif
            version_infos[idx].apply_mapping(
                get_parent_context(idx).get_id(),
                owner_space, map_applied_conditions);
          }
        }
        else
        {
          for (unsigned idx = 0; idx < physical_instances.size(); idx++)
          {
            if (!physical_instances[idx].has_composite_ref())
              continue;
#ifdef DEBUG_HIGH_LEVEL
            assert(virtual_mapped[idx]);
#endif
            version_infos[idx].apply_mapping(
                get_parent_context(idx).get_id(),
                runtime->address_space, map_applied_conditions);
          }
        }
      }
      if (!is_remote())
      {
        if (!map_applied_conditions.empty())
        {
          map_applied_conditions.insert(mapped_precondition);
          complete_mapping(Runtime::merge_events<true>(map_applied_conditions));
        }
        else 
          complete_mapping(mapped_precondition);
        return;
      }
      Event applied_condition = Event::NO_EVENT;
      if (!map_applied_conditions.empty())
        applied_condition = Runtime::merge_events<true>(map_applied_conditions);
      // Send back the message saying that we finished mapping
      Serializer rez;
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
      rez.serialize(applied_condition);
      runtime->send_individual_remote_mapped(orig_proc, rez);
      // Now we can complete this task
      complete_mapping(applied_condition);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      // Notify our enclosing parent task that we are being sent 
      // remotely if we are not locally mapped because now there
      // will be remote state
      if (!is_remote() && !is_locally_mapped())
        parent_ctx->record_remote_state();
      // Check to see if we are stealable, if not and we have not
      // yet been sent remotely, then send the state now
      AddressSpaceID addr_target = runtime->find_address_space(target);
      RezCheck z(rez);
      pack_single_task(rez, addr_target);
      rez.serialize(orig_task);
      rez.serialize(remote_completion_event);
      rez.serialize(remote_unique_id);
      rez.serialize(remote_outermost_context);
      rez.serialize(remote_owner_uid);
      rez.serialize(remote_parent_ctx);
      parent_ctx->pack_parent_task(rez);
      if (!is_locally_mapped())
        pack_version_infos(rez, version_infos);
      pack_restrict_infos(rez, restrict_infos);
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
      derez.deserialize(remote_owner_uid);
      derez.deserialize(remote_parent_ctx);
      RemoteTask *remote_ctx = 
        runtime->find_or_init_remote_context(remote_owner_uid, orig_proc,
                                             remote_parent_ctx);
      remote_ctx->unpack_parent_task(derez);
      if (!is_locally_mapped())
        unpack_version_infos(derez, version_infos);
      unpack_restrict_infos(derez, restrict_infos);
      // Add our enclosing parent regions to the list of 
      // top regions maintained by the remote context
      for (unsigned idx = 0; idx < regions.size(); idx++)
        remote_ctx->add_top_region(regions[idx].parent);
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
      if (is_locally_mapped() && is_leaf())
        complete_mapping();
      // If we're remote, we've already resolved speculation for now
      resolve_speculation();
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_point_point(remote_unique_id, get_unique_id());
      // Return true to add ourselves to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::find_enclosing_local_fields(
           LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos)
    //--------------------------------------------------------------------------
    {
      // Ask the same for our parent context
      parent_ctx->find_enclosing_local_fields(infos);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_inlining(SingleTask *ctx, VariantImpl *variant)
    //--------------------------------------------------------------------------
    {
      // See if there is anything that we need to wait on before running
      std::set<Event> wait_on_events;
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        FutureImpl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->ready_event);
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        GrantImpl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      Event start_condition = Runtime::merge_events<true>(wait_on_events); 

      // See if we need to wait for anything
      if (start_condition.exists() && !start_condition.has_triggered())
        start_condition.wait();

      // Run the task  
      Processor current = parent_ctx->get_executing_processor();
      // Set the context to be the current inline context
      parent_ctx = ctx;
      // Mark that we are an inline task
      is_inline = true;
      variant->dispatch_inline(current, this); 
    }  

    //--------------------------------------------------------------------------
    bool IndividualTask::is_inline_task(void) const
    //--------------------------------------------------------------------------
    {
      return is_inline;
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& IndividualTask::begin_inline_task(void)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_physical_regions();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::end_inline_task(const void *res, 
                                         size_t res_size, bool owned) 
    //--------------------------------------------------------------------------
    {
      // Save the future result and trigger it
      result.impl->set_result(res, res_size, owned);
      result.impl->complete_future();
      // Trigger our completion event
      completion_event.trigger();
      // Now we're done, someone else will deactivate us
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      Event applied;
      derez.deserialize(applied);
      if (applied.exists())
        map_applied_conditions.insert(applied);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events<true>(map_applied_conditions));
      else
        complete_mapping();
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer &rez) 
    //--------------------------------------------------------------------------
    {
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      // First send back any created region tree state.  Note to do
      // this we can start at the index of the version infos since we
      // know that any additional region beyond this are ones for which
      // we have created the privileges.
      for (unsigned idx = version_infos.size(); idx < regions.size(); idx++)
      {
        if (!region_deleted[idx])
        {
          runtime->forest->send_back_logical_state(get_parent_context(idx),
                                                   remote_outermost_context,
                                                   regions[idx], target);
        }
      }
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
      // Point tasks never have to resolve speculation
      resolve_speculation();
      slice_owner = NULL;
      has_remote_subtasks = false;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_single();
      if (!remote_instances.empty())
      {
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
        }
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          if (it->first == runtime->address_space)
            runtime->release_remote_context(local_uid);
          else
            runtime->send_free_remote_context(it->first, rez);
        }
        remote_instances.clear();
      }
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
    void PointTask::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are always done with early mapping
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are never sent anywhere
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::perform_mapping(MustEpochOp *must_epoch_owner/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // For point tasks we use the point termination event which as the
      // end event for this task since point tasks can be moved and
      // the completion event is therefore not guaranteed to survive
      // the length of the task's execution
      bool map_success = map_all_regions(point_termination, must_epoch_owner);
      // If we succeeded in mapping and had no virtual mappings
      // then we are done mapping
      if (map_success && is_leaf()) 
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
    bool PointTask::has_restrictions(unsigned idx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return slice_owner->has_restrictions(idx, handle);
    }

    //--------------------------------------------------------------------------
    bool PointTask::can_early_complete(UserEvent &chain_event)
    //--------------------------------------------------------------------------
    {
      chain_event = point_termination;
      return true;
    }

    //--------------------------------------------------------------------------
    void PointTask::return_virtual_instance(unsigned index, InstanceSet &refs)
    //--------------------------------------------------------------------------
    {
      slice_owner->return_virtual_instance(index, refs);
    }

    //--------------------------------------------------------------------------
    VersionInfo& PointTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    void PointTask::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      slice_owner->recapture_version_info(idx);
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_inline_task(void) const
    //--------------------------------------------------------------------------
    {
      // We are never an inline task
      return false;
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
    void PointTask::perform_inlining(SingleTask *ctx, VariantImpl *variant)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteTask* PointTask::find_outermost_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->find_outermost_context();
    }

    //--------------------------------------------------------------------------
    bool PointTask::has_remote_state(void) const
    //--------------------------------------------------------------------------
    {
      return has_remote_subtasks;
    }

    //--------------------------------------------------------------------------
    void PointTask::record_remote_state(void)
    //--------------------------------------------------------------------------
    {
      // Monotonic so no need to hold the lock
      has_remote_subtasks = true;
    }

    //--------------------------------------------------------------------------
    void PointTask::record_remote_instance(AddressSpaceID remote_instance,
                                           RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      // Pass back our created and deleted operations 
      slice_owner->return_privileges(this);

      slice_owner->record_child_complete();

      // Since this point is now complete we know
      // that we can trigger it. Note we don't need to do
      // this if we're a leaf task with no virtual mappings
      // because we would have performed the leaf task
      // early complete chaining operation.
      if (!is_leaf() || has_virtual_instances())
        point_termination.trigger();

      // Invalidate any context that we had so that the child
      // operations can begin committing
      if (context.exists() && (!is_leaf() || has_virtual_instances()))
        invalidate_region_tree_contexts();
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
      // Mark that this operation is now complete
      complete_operation();
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Commit this operation
      // Don't deactivate ourselves, our slice will do that for us
      commit_operation(false/*deactivate*/);
      // Then tell our slice owner that we're done
      slice_owner->record_child_committed();
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
      if (is_locally_mapped() && is_leaf())
      {
        slice_owner->record_child_mapped();
        complete_mapping();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::find_enclosing_local_fields(
           LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos)
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
    void PointTask::handle_post_mapped(Event mapped_precondition)
    //--------------------------------------------------------------------------
    {
      if (!mapped_precondition.has_triggered())
      {
        SingleTask::DeferredPostMappedArgs args;
        args.hlr_id = HLR_DEFERRED_POST_MAPPED_ID;
        args.task = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_POST_MAPPED_ID,
                                         this, mapped_precondition);
        return;
      }
      slice_owner->record_child_mapped();
      // Now we can complete this point task
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(SliceTask *owner, MinimalPoint *mp)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      compute_point_region_requirements(mp);
      // Get our argument
      mp->assign_argument(local_args, local_arglen);
      // Make a new termination event for this point
      point_termination = UserEvent::create_user_event();
    } 

    //--------------------------------------------------------------------------
    void PointTask::send_back_created_state(AddressSpaceID target, 
                                            unsigned start,
                                            RegionTreeContext remote_outermost)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = start; idx < regions.size(); idx++)
      {
        if (!region_deleted[idx])
        {
          runtime->forest->send_back_logical_state(get_parent_context(idx),
                                                   remote_outermost,
                                                   regions[idx], target);
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
    void WrapperTask::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool WrapperTask::early_map_task(void)
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
    bool WrapperTask::perform_mapping(MustEpochOp *owner)
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
    bool WrapperTask::has_restrictions(unsigned idx, LogicalRegion handle)
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
    void WrapperTask::return_virtual_instance(unsigned index, InstanceSet &refs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
    void WrapperTask::perform_inlining(SingleTask *ctx, VariantImpl *variant)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void WrapperTask::handle_future(const void *res, size_t res_size,bool owned)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void WrapperTask::handle_post_mapped(Event mapped_precondition)
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
      parent_task = NULL;
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
    int RemoteTask::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_wrapper();
      context = RegionTreeContext();
      remote_owner_uid = 0;
      remote_parent_ctx = NULL;
      depth = -1;
      is_top_level_context = false;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // Before deactivating the context, clean it out
      if (!top_level_regions.empty())
      {
        for (std::set<LogicalRegion>::const_iterator it = 
              top_level_regions.begin(); it != top_level_regions.end(); it++)
        {
          runtime->forest->invalidate_current_context(context, *it, 
                                                  false/*logical users only*/); 
        }
      }
      if (!remote_instances.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_top_level_context);
#endif
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
        }
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          if (it->first == runtime->address_space)
            runtime->release_remote_context(local_uid);
          else
            runtime->send_free_remote_context(it->first, rez);
        }
        remote_instances.clear();
      }
      top_level_regions.clear();
      if (context.exists())
        runtime->free_context(this);
      deactivate_wrapper();
      // Context is freed in deactivate single
      runtime->free_remote_task(this);
    }
    
    //--------------------------------------------------------------------------
    void RemoteTask::initialize_remote(UniqueID uid, SingleTask *remote_parent,
                                       bool is_top_level)
    //--------------------------------------------------------------------------
    {
      remote_owner_uid = uid;
      remote_parent_ctx = remote_parent;
      is_top_level_context = is_top_level;
      runtime->allocate_context(this);
#ifdef DEBUG_HIGH_LEVEL
      assert(context.exists());
      runtime->forest->check_context_state(context);
#endif
    } 

    //--------------------------------------------------------------------------
    void RemoteTask::unpack_parent_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(depth);
      size_t num_local;
      derez.deserialize(num_local);
      std::deque<LocalFieldInfo> temp_local(num_local);
      for (unsigned idx = 0; idx < num_local; idx++)
      {
        derez.deserialize(temp_local[idx]);
        allocate_local_field(temp_local[idx]);
      }
#ifdef LEGION_SPY
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
    RemoteTask* RemoteTask::find_outermost_context(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteTask::get_context_uid(void) const
    //--------------------------------------------------------------------------
    {
      return remote_owner_uid;
    }

    //--------------------------------------------------------------------------
    bool RemoteTask::has_remote_state(void) const
    //--------------------------------------------------------------------------
    {
      return true; // Definitely does!
    }

    //--------------------------------------------------------------------------
    void RemoteTask::record_remote_state(void)
    //--------------------------------------------------------------------------
    {
      // Should only see this call if it is the top-level context
#ifdef DEBUG_HIGH_LEVEL
      assert(is_top_level_context);
#endif
    }

    //--------------------------------------------------------------------------
    void RemoteTask::record_remote_instance(AddressSpaceID remote_instance,
                                            RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
      // should only see this call if it is the top-level context
#ifdef DEBUG_HIGH_LEVEL
      assert(is_top_level_context);
#endif
      AutoLock o_lock(op_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    Event RemoteTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_SPY
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
           LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos)
    //--------------------------------------------------------------------------
    {
      // No need to go up since we are the uppermost task on this runtime
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    } 

    //--------------------------------------------------------------------------
    void RemoteTask::pack_remote_ctx_info(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(remote_owner_uid);
      rez.serialize(remote_parent_ctx);
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
      parent_ctx = enclosing; 
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
      compute_parent_indexes();
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
    RemoteTask* InlineTask::find_outermost_context(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->find_outermost_context();
    }

    //--------------------------------------------------------------------------
    bool InlineTask::has_remote_state(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void InlineTask::record_remote_state(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InlineTask::record_remote_instance(AddressSpaceID remote_inst,
                                            RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
           LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos)
    //--------------------------------------------------------------------------
    {
      enclosing->find_enclosing_local_fields(infos);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        infos.push_back(local_fields[idx]);
    }

    //--------------------------------------------------------------------------
    void InlineTask::register_new_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_new_child_operation(op);
    }

    //--------------------------------------------------------------------------
    void InlineTask::add_to_dependence_queue(Operation *op, bool has_lock)
    //--------------------------------------------------------------------------
    {
      enclosing->add_to_dependence_queue(op, has_lock);
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
      serdez_redop_fns = NULL;
      slice_fraction = Fraction<long long>(0,1); // empty fraction
      total_points = 0;
      mapped_points = 0;
      complete_points = 0;
      committed_points = 0;
      complete_received = false;
      commit_received = false; 
      predicate_false_result = NULL;
      predicate_false_size = 0;
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_multi();
      privilege_paths.clear();
      if (!locally_mapped_slices.empty())
      {
        for (std::deque<SliceTask*>::const_iterator it = 
              locally_mapped_slices.begin(); it != 
              locally_mapped_slices.end(); it++)
        {
          (*it)->deactivate();
        }
        locally_mapped_slices.clear();
      }
      // Remove our reference to the argument map
      argument_map = ArgumentMap();
      // Remove our reference to the future map
      future_map = FutureMap();
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      predicate_false_future = Future();
      // Remove our reference to the reduction future
      reduction_future = Future();
      rerun_analysis_requirements.clear();
      map_applied_conditions.clear();
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
        regions[idx] = launcher.region_requirements[idx];
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
        arg_manager = legion_new<AllocManager>(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(launcher.argument_map.impl->freeze());
      map_id = launcher.map_id;
      tag = launcher.tag;
      is_index_space = true;
      must_parallelism = launcher.must_parallelism;
      index_domain = launcher.launch_domain;
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(launcher.predicate_false_future,
                             launcher.predicate_false_result);
      if (check_privileges)
        perform_privilege_checks();
      initialize_paths();
      annotate_early_mapped_regions();
      future_map = FutureMap(legion_new<FutureMapImpl>(ctx, this, runtime));
#ifdef DEBUG_HIGH_LEVEL
      future_map.impl->add_valid_domain(index_domain);
#endif
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
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
        regions[idx] = launcher.region_requirements[idx];
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
        arg_manager = legion_new<AllocManager>(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(launcher.argument_map.impl->freeze());
      map_id = launcher.map_id;
      tag = launcher.tag;
      is_index_space = true;
      must_parallelism = launcher.must_parallelism;
      index_domain = launcher.launch_domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->is_foldable)
      {
        log_run.error("Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_UNFOLDABLE_REDUCTION_OP);
      }
      else
        initialize_reduction_state();
      initialize_base_task(ctx, track, launcher.predicate, task_id);
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(launcher.predicate_false_future,
                             launcher.predicate_false_result);
      if (check_privileges)
        perform_privilege_checks();
      initialize_paths();
      annotate_early_mapped_regions();
      reduction_future = Future(legion_new<FutureImpl>(runtime,
            true/*register*/, runtime->get_available_distributed_id(true), 
            runtime->address_space, runtime->address_space, this));
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
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
        regions[idx] = region_requirements[idx];
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = legion_new<AllocManager>(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(arg_map.impl->freeze());
      map_id = mid;
      tag = t;
      is_index_space = true;
      must_parallelism = must;
      index_domain = domain;
      initialize_base_task(ctx, true/*track*/, pred, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_paths();
      annotate_early_mapped_regions();
      future_map = FutureMap(legion_new<FutureMapImpl>(ctx, this, runtime));
#ifdef DEBUG_HIGH_LEVEL
      future_map.impl->add_valid_domain(index_domain);
#endif
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
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
        regions[idx] = region_requirements[idx];
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg_manager == NULL);
#endif
        arg_manager = legion_new<AllocManager>(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, global_arg.get_ptr(), arglen);
      }
      argument_map = ArgumentMap(arg_map.impl->freeze());
      map_id = mid;
      tag = t;
      is_index_space = true;
      must_parallelism = must;
      index_domain = domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->is_foldable)
      {
        log_run.error("Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id());
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
      initialize_paths();
      annotate_early_mapped_regions();
      reduction_future = Future(legion_new<FutureImpl>(runtime, 
            true/*register*/, runtime->get_available_distributed_id(true), 
            runtime->address_space, runtime->address_space, this));
      check_empty_field_requirements();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  get_task_name());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        }
      }
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_predicate(const Future &pred_future,
                                         const TaskArgument &pred_arg)
    //--------------------------------------------------------------------------
    {
      if (pred_future.impl != NULL)
        predicate_false_future = pred_future;
      else
      {
        predicate_false_size = pred_arg.get_size();
        if (predicate_false_size == 0)
        {
          // TODO: Reenable this error if we want to track predicate defaults
#if 0
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
                          get_task_name(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
#endif
        }
        else
        {
          // TODO: Reenable this error if we want to track predicate defaults
#if 0
          if (predicate_false_size != variants->return_size)
          {
            log_run.error("Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has predicated "
                          "false return type of size %ld bytes, but the "
                          "expected return size is %ld bytes.",
                          get_task_name(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          predicate_false_size, variants->return_size);
#ifdef DEBUG_HIGH_LEVEL
            assert(false);
#endif
            exit(ERROR_PREDICATE_RESULT_SIZE_MISMATCH);
          }
#endif
#ifdef DEBUG_HIGH_LEVEL
          assert(predicate_false_result == NULL);
#endif
          predicate_false_result = 
            legion_malloc(PREDICATE_ALLOC, predicate_false_size);
          memcpy(predicate_false_result, pred_arg.get_ptr(),
                 predicate_false_size);
        }
      }
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
          regions[idx].flags |= MUST_PREMAP_FLAG;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(privilege_paths.size() == regions.size());
#endif
      // First compute the parent indexes
      compute_parent_indexes();
      // Enumerate our points
      enumerate_points();
      begin_dependence_analysis();
      // If we are tracing we need to record any aliased region requirements
      if (is_tracing())
        record_aliased_region_requirements(get_trace());
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->impl != NULL);
#endif
        it->impl->register_dependence(this);
#ifdef LEGION_SPY
        if (it->impl->producer_op != NULL)
          LegionSpy::log_mapping_dependence(
              parent_ctx->get_unique_id(), it->impl->producer_uid, 0,
              get_unique_id(), 0, TRUE_DEPENDENCE);
#endif
      }
      // Also have to register any dependences on our predicate
      register_predicate_dependence();
      version_infos.resize(regions.size());
      restrict_infos.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(this, idx, regions[idx], 
                                                     version_infos[idx],
                                                     restrict_infos[idx],
                                                     privilege_paths[idx]);
      }
      // See if we have any requirements that interferred with a close
      // operation that was generated by a later region requirement
      // and therefore needs to be re-analyzed
      if (!rerun_analysis_requirements.empty())
      {
        // Make a local copy to avoid invalidating the iterator
        std::vector<unsigned> rerun(rerun_analysis_requirements.begin(),
                                    rerun_analysis_requirements.end());
#ifdef DEBUG_HIGH_LEVEL
        rerun_analysis_requirements.clear();
#endif
        for (std::vector<unsigned>::const_iterator it = 
              rerun.begin(); it != rerun.end(); it++)
        {
          // Clear out the version infos so we get new data
          VersionInfo &version_info = version_infos[*it];
          version_info.release();
          version_info.clear();
          runtime->forest->perform_dependence_analysis(this, *it, regions[*it],
                                                       version_info,
                                                       restrict_infos[*it],
                                                       privilege_paths[*it]);
          // If we still have re-run requirements, then we have
          // interfering region requirements so warn the user
          if (!rerun_analysis_requirements.empty())
          {
            for (std::set<unsigned>::const_iterator it2 = 
                  rerun_analysis_requirements.begin(); it2 != 
                  rerun_analysis_requirements.end(); it2++)
            {
              report_interfering_requirements(*it, *it2);
            }
            rerun_analysis_requirements.clear();
          }
        }
      }
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      std::set<Event> preconditions;
      if (is_locally_mapped())
      {
        // If we're locally mapped, request everyone's state
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
          version_infos[idx].make_local(preconditions, runtime->forest, 
                                        get_parent_context(idx).get_id());
      }
      else
      {
        // Otherwise we only need to request state for early mapped regions
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          const RegionRequirement &req = regions[idx];
          // We need to request state for any early mapped regions, either
          // because they are restricted or we actually need to early map them
          if (req.is_restricted() || req.must_premap())
            version_infos[idx].make_local(preconditions, runtime->forest, 
                                          get_parent_context(idx).get_id());
        }
      }
      if (preconditions.empty())
        ready_event.trigger();
      else
        Runtime::trigger_event<true>(ready_event,
            Runtime::merge_events<true>(preconditions));
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_requirements(unsigned idx1,unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 0
      log_run.error("Aliased region requirements for index tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are interfering.", idx1, idx2, get_task_name(),
                          get_unique_id(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_ALIASED_REGION_REQUIREMENTS);
#else
      log_run.warning("Region requirements %d and %d of index task "
                      "%s (UID %lld) in parent task %s (UID %lld) are "
                      "interfering.  This behavior is currently undefined. "
                      "You better really know what you are doing.",
                      idx1, idx2, get_task_name(), get_unique_id(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#endif
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_close_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
      rerun_analysis_requirements.insert(idx);
    }

    //--------------------------------------------------------------------------
    FatTreePath* IndexTask::compute_fat_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
      assert(regions[idx].handle_type != SINGULAR);
#endif
      FatTreePath *result = NULL; 
      std::map<IndexTreeNode*,FatTreePath*> storage;
      bool overlap = false;
      const bool overlap_is_bad = IS_WRITE(regions[idx]);
      if (regions[idx].handle_type == REG_PROJECTION)
      {
        IndexSpace parent_space = regions[idx].region.get_index_space();
        if (overlap_is_bad)
        {
          for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                minimal_points.begin(); it != minimal_points.end(); it++)
          {
            LogicalRegion dst = it->second->find_logical_region(idx);
            result = runtime->forest->compute_fat_path(dst.get_index_space(),
                               parent_space, storage, true/*test*/, overlap);
            if (overlap)
              break;
          }
        }
        else
        {
          for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                minimal_points.begin(); it != minimal_points.end(); it++)
          {
            LogicalRegion dst = it->second->find_logical_region(idx);
            result = runtime->forest->compute_fat_path(dst.get_index_space(),
                               parent_space, storage, false/*test*/, overlap);
          }
        }
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == PART_PROJECTION);
#endif
        IndexPartition parent_partition = 
                        regions[idx].partition.get_index_partition();
        if (overlap_is_bad)
        {
          for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                minimal_points.begin(); it != minimal_points.end(); it++)
          {
            LogicalRegion dst = it->second->find_logical_region(idx);
            result = runtime->forest->compute_fat_path(dst.get_index_space(),
                            parent_partition, storage, true/*test*/, overlap);
            if (overlap)
              break;
          }
        }
        else
        {
          for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                minimal_points.begin(); it != minimal_points.end(); it++)
          {
            LogicalRegion dst = it->second->find_logical_region(idx);
            result = runtime->forest->compute_fat_path(dst.get_index_space(),
                            parent_partition, storage, false/*test*/, overlap);
          }
        }
      }
      if (overlap_is_bad && overlap)
      {
        log_task.error("Index Space Task %s (ID %lld) violated the disjoint "
                       "projection region requirement assumption for region "
                       "requirement %d!", get_task_name(), 
                       get_unique_id(), idx);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_BAD_PROJECTION_USE);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndexTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < privilege_paths.size());
#endif
      return privilege_paths[idx];
    }

    //--------------------------------------------------------------------------
    void IndexTask::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      bool trigger = true;
      // Fill in the index task map with the default future value
      if (redop == 0)
      {
        // Handling the future map case
        if (predicate_false_future.impl != NULL)
        {
          Event wait_on = predicate_false_future.impl->get_ready_event();
          if (wait_on.has_triggered())
          {
            const size_t result_size = 
              check_future_size(predicate_false_future.impl);
            const void *result = 
              predicate_false_future.impl->get_untyped_result();
            for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
            {
              Future f = future_map.get_future(itr.p);
              if (result_size > 0)
                f.impl->set_result(result, result_size, false/*own*/);
            }
          }
          else
          {
            // Add references so things won't be prematurely collected
            future_map.impl->add_reference();
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF);
            Runtime::DeferredFutureMapSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_MAP_SET_ID;
            args.future_map = future_map.impl;
            args.result = predicate_false_future.impl;
            args.domain = index_domain;
            args.task_op = this;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_FUTURE_MAP_SET_ID,
                                             this, wait_on);
            trigger = false;
          }
        }
        else
        {
          for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
          {
            Future f = future_map.get_future(itr.p);
            if (predicate_false_size > 0)
              f.impl->set_result(predicate_false_result,
                                 predicate_false_size, false/*own*/);
          }
        }
      }
      else
      {
        // Handling a reduction case
        if (predicate_false_future.impl != NULL)
        {
          Event wait_on = predicate_false_future.impl->get_ready_event();
          if (wait_on.has_triggered())
          {
            const size_t result_size = 
                        check_future_size(predicate_false_future.impl);
            if (result_size > 0)
              reduction_future.impl->set_result(
                  predicate_false_future.impl->get_untyped_result(),
                  result_size, false/*own*/);
          }
          else
          {
            // Add references so they aren't garbage collected 
            reduction_future.impl->add_base_gc_ref(DEFERRED_TASK_REF);
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF);
            Runtime::DeferredFutureSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_SET_ID;
            args.target = reduction_future.impl;
            args.result = predicate_false_future.impl;
            args.task_op = this;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_FUTURE_SET_ID,
                                             this, wait_on);
            trigger = false;
          }
        }
        else
        {
          if (predicate_false_size > 0)
            reduction_future.impl->set_result(predicate_false_result,
                                  predicate_false_size, false/*own*/);
        }
      }
      // Then clean up this task execution
      if (trigger)
        complete_execution();
      complete_mapping();
      trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      std::vector<unsigned> early_map_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &req = regions[idx];
        if (req.is_restricted() || req.must_premap())
          early_map_indexes.push_back(idx);
      }
      if (!early_map_indexes.empty())
        return early_map_regions(map_applied_conditions, early_map_indexes);
      return true;
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
        if (!is_sliced() && target_proc.exists() && 
            (target_proc != current_proc))
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(minimal_points_assigned == 0);
#endif
          // Make a slice copy and send it away
          SliceTask *clone = clone_as_slice_task(index_domain, target_proc,
                                                 true/*needs slice*/,
                                                 stealable, 1LL);
#ifdef DEBUG_HIGH_LEVEL
          assert(minimal_points_assigned == minimal_points.size());
#endif
          minimal_points.clear();
          runtime->send_task(clone);
          return false; // We have now been sent away
        }
        else
          return true; // Still local so we can be sliced
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(MustEpochOp *owner/*=NULL*/)
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
    bool IndexTask::has_restrictions(unsigned idx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Handle the case of inline tasks
      if (restrict_infos.empty())
        return false;
      if (restrict_infos[idx].has_restrictions())
        return runtime->forest->has_restrictions(handle, restrict_infos[idx],
                                                 regions[idx].privilege_fields);
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
      // Return back our privileges
      return_privilege_state(parent_ctx);

      // Trigger all the futures or set the reduction future result
      // and then trigger it
      if (redop != 0)
      {
        if (speculation_state != RESOLVE_FALSE_STATE)
          reduction_future.impl->set_result(reduction_state,
                                            reduction_state_size, 
                                            false/*owner*/);
        reduction_future.impl->complete_future();
      }
      else
        future_map.impl->complete_all_futures();

      complete_operation();
      if (speculation_state == RESOLVE_FALSE_STATE)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // We can release our version infos now
      for (std::vector<VersionInfo>::iterator it = version_infos.begin();
            it != version_infos.end(); it++)
      {
        it->release();
      }
      // Mark that this operation is now committed
      commit_operation(true/*deactivate*/);
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
    void IndexTask::perform_inlining(SingleTask *ctx, VariantImpl *variant)
    //--------------------------------------------------------------------------
    {
      // must parallelism not allowed to be inlined
      if (must_parallelism)
      {
        log_task.error("Illegal attempt to inline must-parallelism "
                       "task %s (ID %lld)",
                       get_task_name(), get_unique_id());
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_ILLEGAL_MUST_PARALLEL_INLINE);
      }
      // See if there is anything to wait for
      std::set<Event> wait_on_events;
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        FutureImpl *impl = futures[idx].impl; 
        wait_on_events.insert(impl->ready_event);
      }
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        GrantImpl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	Event e = wait_barriers[idx].phase_barrier.get_previous_phase();
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      Event start_condition = Runtime::merge_events<true>(wait_on_events); 

      // See if we need to wait for anything
      if (start_condition.exists() && !start_condition.has_triggered())
        start_condition.wait();

      // Enumerate all of the points of our index space and run
      // the task for each one of them either saving or reducing their futures
      Processor current = parent_ctx->get_executing_processor();
      // Save the context to be the current inline context
      parent_ctx = ctx;
      // Make a copy of our region requirements
      std::vector<RegionRequirement> copy_requirements(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        copy_requirements[idx] = regions[idx];
      bool first = true;
      for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
      {
        // If this is not the first we have to restore the region
        // requirements from copy that we made before hand
        if (!first)
        {
          for (unsigned idx = 0; idx < regions.size(); idx++)
            regions[idx] = copy_requirements[idx];
        }
        else
          first = false;
        index_point = itr.p; 
        compute_point_region_requirements();
        // Get our local args
        TaskArgument local = argument_map.impl->get_point(index_point);
        local_args = local.get_ptr();
        local_arglen = local.get_size();
        variant->dispatch_inline(current, this);
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
    bool IndexTask::is_inline_task(void) const
    //--------------------------------------------------------------------------
    {
      // We are always an inline task if we are getting called here
      return true;
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& IndexTask::begin_inline_task(void)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_physical_regions();
    }

    //--------------------------------------------------------------------------
    void IndexTask::end_inline_task(const void *res, size_t res_size,bool owned)
    //--------------------------------------------------------------------------
    {
      if (redop == 0)
      {
        Future f = future_map.impl->get_future(index_point);
        f.impl->set_result(res, res_size, owned);
      }
      else
        fold_reduction_future(res, res_size, owned, true/*exclusive*/);
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(const Domain &d, Processor p,
                                              bool recurse, bool stealable,
                                              long long scale_denominator)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(false); 
      result->initialize_base_task(parent_ctx, 
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_multi_from(this, d, p, recurse, stealable);
      result->remote_outermost_context = 
        parent_ctx->find_outermost_context()->get_context();
#ifdef DEBUG_HIGH_LEVEL
      assert(result->remote_outermost_context.exists());
#endif
      result->index_complete = this->completion_event;
      result->denominator = scale_denominator;
      result->index_owner = this;
      result->remote_owner_uid = parent_ctx->get_unique_id();
      result->remote_parent_ctx = parent_ctx;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_index_slice(get_unique_id(), 
                                   result->get_unique_id());
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
    void IndexTask::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
      {
        MinimalPoint *point = new MinimalPoint();
        // Find the argument for this point if it exists
        TaskArgument arg = argument_map.impl->get_point(itr.p);
        point->add_argument(arg, false/*own*/);
        minimal_points[itr.p] = point;
      }
      // Figure out which requirements are projection and update them
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type == SINGULAR)
          continue;
        else if (regions[idx].handle_type == PART_PROJECTION)
        {
          // Check to see if we're doing default projection
          if (regions[idx].projection == 0)
          {
            for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                  minimal_points.begin(); it != minimal_points.end(); it++)
            {
              if (it->first.get_dim() > 3)
              {
                log_task.error("Projection ID 0 is invalid for tasks whose "
                               "points are larger than three dimensional "
                               "unsigned integers.  Points for task %s "
                               "have elements of %d dimensions",
                               get_task_name(), it->first.get_dim());
#ifdef DEBUG_HIGH_LEVEL
                assert(false);
#endif
                exit(ERROR_INVALID_IDENTITY_PROJECTION_USE);
              }
              it->second->add_projection_region(idx, 
                runtime->forest->get_logical_subregion_by_color(
                    regions[idx].partition, ColorPoint(it->first)));
            }
          }
          else
          {
            ProjectionFunctor *functor = 
              runtime->find_projection_functor(regions[idx].projection);
            if (functor == NULL)
            {
              PartitionProjectionFnptr projfn = 
                  Runtime::find_partition_projection_function(
                      regions[idx].projection);
              for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                    minimal_points.begin(); it != minimal_points.end(); it++)
              {
                it->second->add_projection_region(idx,  
                    (*projfn)(regions[idx].partition,
                              it->first,runtime->external));
              }
            }
            else
            {
              for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                    minimal_points.begin(); it != minimal_points.end(); it++)
              {
                it->second->add_projection_region(idx,
                    functor->project(DUMMY_CONTEXT, this, idx,
                                     regions[idx].partition, it->first));
              }
            }
          }
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(regions[idx].handle_type == REG_PROJECTION);
#endif
          if (regions[idx].projection != 0)
          {
            ProjectionFunctor *functor = 
              runtime->find_projection_functor(regions[idx].projection);
            if (functor == NULL)
            {
              RegionProjectionFnptr projfn = 
                Runtime::find_region_projection_function(
                    regions[idx].projection);
              for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                    minimal_points.begin(); it != minimal_points.end(); it++)
              {
                it->second->add_projection_region(idx, 
                  (*projfn)(regions[idx].region,
                            it->first, runtime->external));
              }
            }
            else
            {
              for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
                    minimal_points.begin(); it != minimal_points.end(); it++)
              {
                it->second->add_projection_region(idx, 
                  functor->project(DUMMY_CONTEXT, this, idx, 
                                   regions[idx].region, it->first));
              }
            }
          }
          else
          {
            // Otherwise we are the default case in which 
            // case we don't need to do anything
            // Update the region requirement kind
            // to be singular since all points will use 
            // the same logical region
            regions[idx].handle_type = SINGULAR;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_locally_mapped_slice(SliceTask *local_slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      locally_mapped_slices.push_back(local_slice);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_mapped(unsigned points, long long denom,
                                        Event applied_condition)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      bool trigger_children_completed = false;
      bool trigger_children_commit = false;
      {
        AutoLock o_lock(op_lock);
        total_points += points;
        mapped_points += points;
        slice_fraction.add(Fraction<long long>(1,denom));
        if (applied_condition.exists())
          map_applied_conditions.insert(applied_condition);
        // Already know that mapped points is the same as total points
        if (slice_fraction.is_whole())
        {
          need_trigger = true;
          if ((complete_points == total_points) &&
              !children_complete_invoked)
          {
            trigger_children_completed = true;
            children_complete_invoked = true;
          }
          if ((committed_points == total_points) &&
              !children_commit_invoked)
          {
            trigger_children_commit = true;
            children_commit_invoked = true;
          }
        }
      }
      if (need_trigger)
      {
        // At this point, we know that we are mapped, see if we have
        // any locally mapped slices which we need to apply changes
        // Note we do this here while we're not holding the lock
        if (!locally_mapped_slices.empty())
        {
          for (std::deque<SliceTask*>::const_iterator it = 
                locally_mapped_slices.begin(); it != 
                locally_mapped_slices.end(); it++)
          {
            (*it)->apply_local_version_infos(map_applied_conditions);
          }
        }
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          Event map_condition = 
            Runtime::merge_events<true>(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
      }
      if (trigger_children_completed)
        trigger_children_complete();
      if (trigger_children_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(unsigned points)
    //--------------------------------------------------------------------------
    {
      bool trigger_execution = false;
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        complete_points += points;
#ifdef DEBUG_HIGH_LEVEL
        assert(!complete_received);
        assert(complete_points <= total_points);
#endif
        if (slice_fraction.is_whole() && 
            (complete_points == total_points))
        {
          trigger_execution = true;
          if (!children_complete_invoked)
          {
            need_trigger = true;
            children_complete_invoked = true;
          }
        }
      }
      if (trigger_execution)
        complete_execution();
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
    void IndexTask::unpack_slice_mapped(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      long long denom;
      derez.deserialize(denom);
      Event applied_condition;
      derez.deserialize(applied_condition);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!IS_WRITE(regions[idx]))
          continue;
        if (regions[idx].handle_type != SINGULAR)
        {
          std::vector<LogicalRegion> handles(points); 
          for (unsigned pidx = 0; pidx < points; pidx++)
            derez.deserialize(handles[pidx]);
        }
        // otherwise it was locally mapped so we are already done
      }
      return_slice_mapped(points, denom, applied_condition);
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
    /*static*/ void IndexTask::process_slice_mapped(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_mapped(derez, source);
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
      // Slice tasks never have to resolve speculation
      resolve_speculation();
      reclaim = false;
      index_complete = Event::NO_EVENT;
      mapping_index = 0;
      num_unmapped_points = 0;
      num_uncomplete_points = 0;
      num_uncommitted_points = 0;
      denominator = 0;
      index_owner = NULL;
      remote_owner_uid = 0;
      remote_parent_ctx = NULL;
      remote_unique_id = get_unique_id();
      locally_mapped = false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      if (!version_infos.empty())
      {
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
          version_infos[idx].release();
        version_infos.clear();
      }
      deactivate_multi();
      // Deactivate all our points 
      for (std::deque<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        (*it)->deactivate(); 
      }
      points.clear();
      for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator it = 
            temporary_futures.begin(); it != temporary_futures.end(); it++)
      {
        legion_free(FUTURE_RESULT_ALLOC, it->second.first, it->second.second);
      }
      temporary_futures.clear();
      temporary_virtual_refs.clear();
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
    void SliceTask::trigger_remote_state_analysis(UserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      // If we are locally mapped, we are done no matter what
      if (is_locally_mapped())
      {
        ready_event.trigger();
        return;
      }
      // Otherwise we just need to request state for any non-eary mapped regions
      std::set<Event> preconditions;
      for (unsigned idx = 0; idx < version_infos.size(); idx++)
      {
        if (early_mapped_regions.find(idx) == early_mapped_regions.end())
          version_infos[idx].make_local(preconditions, runtime->forest,
                                        get_parent_context(idx).get_id());
      }
      if (preconditions.empty())
        ready_event.trigger();
      else
        Runtime::trigger_event<true>(ready_event,
            Runtime::merge_events<true>(preconditions));
    }

    //--------------------------------------------------------------------------
    void SliceTask::resolve_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // Slices are already done with early mapping 
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::prewalk_slice(void)
    //--------------------------------------------------------------------------
    {
      // Premap all regions that were not early mapped
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // If we've already premapped it then we are done
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
          continue;
        RegionTreePath privilege_path;
        initialize_privilege_path(privilege_path, regions[idx]);
        // Walk the path down to the upper bound region without getting
        // any of the valid instances
        InstanceSet empty_set;
        runtime->forest->physical_traverse_path(get_parent_context(idx),
                                        privilege_path, regions[idx],
                                        version_infos[idx], this,
                                        false/*find valid*/, empty_set
#ifdef DEBUG_HIGH_LEVEL
                                        , idx, get_logging_name(), unique_op_id
#endif
                                        );
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::apply_local_version_infos(std::set<Event> &map_conditions)
    //--------------------------------------------------------------------------
    {
      // We know we are local
      AddressSpaceID owner_space = runtime->address_space; 
      for (unsigned idx = 0; idx < version_infos.size(); idx++)
      {
        version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                         owner_space, map_conditions);
      }
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Quick out in case this slice task is to be reclaimed
      if (reclaim)
        return true;
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        // The runtime will deactivate this task
        // after it has been sent
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::perform_mapping(MustEpochOp *epoch_owner/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // Walk the path down to the upper bounds for all points first
      prewalk_slice();
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
#ifdef DEBUG_HIGH_LEVEL
          bool point_success = 
#endif
            points[idx]->perform_mapping(epoch_owner);
#ifdef DEBUG_HIGH_LEVEL
          assert(point_success);
#endif
        }
        // If we succeeded in mapping we are no longer stealable
        stealable = false;
      }
      else
      {
        // This case only occurs if this is an intermediate slice and
        // its sub-slices failed to map, so try to remap them.
        for (std::list<SliceTask*>::iterator it = slices.begin();
              it != slices.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          bool slice_success = 
#endif
            (*it)->trigger_execution();
#ifdef DEBUG_HIGH_LEVEL
          assert(slice_success);
#endif
          
        }
        slices.clear();
        // If we succeeded in mapping all our remaining
        // slices then mark that this task can be reclaimed
        reclaim = true;
      }
      return true;
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
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_locally) && stealable);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::has_restrictions(unsigned idx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < restrict_infos.size());
#endif
        if (restrict_infos[idx].has_restrictions())
          return runtime->forest->has_restrictions(handle, restrict_infos[idx],
                                                 regions[idx].privilege_fields);
        return false;
      }
      else
        return index_owner->has_restrictions(idx, handle);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // Walk the path down to the upper bounds for all points first
      prewalk_slice();
      // Mark that this task is no longer stealable.  Once we start
      // executing things onto a specific processor slices cannot move.
      stealable = false;
      // First enumerate all of our points if we haven't already done so
      if (points.empty())
        enumerate_points();
#ifdef DEBUG_HIGH_LEVEL
      assert(!points.empty());
      assert(mapping_index <= points.size());
#endif
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
#ifdef DEBUG_HIGH_LEVEL
        bool point_success = 
#endif
          next_point->perform_mapping();
#ifdef DEBUG_HIGH_LEVEL
        assert(point_success);
#endif
        // Update the mapping index and then launch
        // the point (it is imperative that these happen in this order!)
        mapping_index++;
        // Once we call this function on the last point it
        // is possible that this slice task object can be recycled
        next_point->launch_task();
      }
      return true;
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
      // Notify our enclosing parent task that we are being sent 
      // remotely if we are not locally mapped because now there
      // will be remote state
      if (!is_remote() && !is_locally_mapped())
        parent_ctx->record_remote_state();
      // Check to see if we are stealable or not yet fully sliced,
      // if both are false and we're not remote, then we can send the state
      // now or check to see if we are remotely mapped
      AddressSpaceID addr_target = runtime->find_address_space(target);
      RezCheck z(rez);
      // Preamble used in TaskOp::unpack
      rez.serialize(points.size());
      pack_multi_task(rez, addr_target);
      rez.serialize(denominator);
      rez.serialize(index_owner);
      rez.serialize(index_complete);
      rez.serialize(remote_unique_id);
      rez.serialize(remote_outermost_context);
      rez.serialize(locally_mapped);
      rez.serialize(remote_owner_uid);
      rez.serialize(remote_parent_ctx);
      parent_ctx->pack_parent_task(rez);
      if (!is_locally_mapped())
        pack_version_infos(rez, version_infos);
      if (is_remote())
        pack_restrict_infos(rez, restrict_infos);
      else
        index_owner->pack_restrict_infos(rez, index_owner->restrict_infos);
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->pack_task(rez, target);
      }
      bool deactivate_now = true;
      if (!is_remote() && is_locally_mapped())
      {
        // If we're not remote and locally mapped then we need
        // to hold onto these version infos until we are done
        // with the whole index space task, so tell our owner
        index_owner->record_locally_mapped_slice(this);
        deactivate_now = false;
      }
      else
      {
        // Release our version infos
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
          version_infos[idx].release();
        version_infos.clear();
      }
      // Always return true for slice tasks since they should
      // always be deactivated after they are sent somewhere else
      return deactivate_now;
    }
    
    //--------------------------------------------------------------------------
    bool SliceTask::unpack_task(Deserializer &derez, Processor current)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez);
      current_proc = current;
      derez.deserialize(denominator);
      derez.deserialize(index_owner);
      derez.deserialize(index_complete);
      derez.deserialize(remote_unique_id); 
      derez.deserialize(remote_outermost_context);
      derez.deserialize(locally_mapped);
      derez.deserialize(remote_owner_uid);
      derez.deserialize(remote_parent_ctx);
      RemoteTask *remote_ctx = 
        runtime->find_or_init_remote_context(remote_owner_uid, orig_proc,
                                             remote_parent_ctx);
      remote_ctx->unpack_parent_task(derez);
      if (!is_locally_mapped())
        unpack_version_infos(derez, version_infos);
      unpack_restrict_infos(derez, restrict_infos);
      // Add our parent regions to the list of top regions
      for (unsigned idx = 0; idx < regions.size(); idx++)
        remote_ctx->add_top_region(regions[idx].parent);
      // Quick check to see if we ended up back on the original node
      if (!is_remote())
        parent_ctx = index_owner->parent_ctx;
      else
        parent_ctx = remote_ctx;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_slice_slice(remote_unique_id, get_unique_id());
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = runtime->get_available_point_task(false); 
        point->slice_owner = this;
        point->unpack_task(derez, current);
        point->parent_ctx = parent_ctx;
        points.push_back(point);
        if (Runtime::legion_spy_enabled)
          LegionSpy::log_slice_point(get_unique_id(), 
                                     point->get_unique_id(),
                                     point->index_point);
      }
      // Return true to add this to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_inlining(SingleTask *ctx, VariantImpl *variant)
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
      SliceTask *result = runtime->get_available_slice_task(false); 
      result->initialize_base_task(parent_ctx, 
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_multi_from(this, d, p, recurse, stealable);
      result->remote_outermost_context = this->remote_outermost_context;
      result->index_complete = this->index_complete;
      result->denominator = this->denominator * scale_denominator;
      result->index_owner = this->index_owner;
      result->remote_owner_uid = this->remote_owner_uid;
      result->remote_parent_ctx = this->remote_parent_ctx;
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_slice_slice(get_unique_id(), 
                                   result->get_unique_id());
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
          {
            // Hold the lock to protect the data structure
            AutoLock o_lock(op_lock);
            temporary_futures[point] = 
              std::pair<void*,size_t>(const_cast<void*>(result),result_size);
          }
          else
          {
            void *copy = legion_malloc(FUTURE_RESULT_ALLOC, result_size);
            memcpy(copy,result,result_size);
            // Hold the lock to protect the data structure
            AutoLock o_lock(op_lock);
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
      {
        PointTask *point = points[idx];
        must_epoch->register_single_task(point, must_epoch_index);
      }
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(const DomainPoint &p,
                                              MinimalPoint *mp)
    //--------------------------------------------------------------------------
    {
      PointTask *result = runtime->get_available_point_task(false);
      result->initialize_base_task(parent_ctx,
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_task_op_from(this, this->target_proc, 
                                 false/*stealable*/, true/*duplicate*/);
      result->is_index_space = true;
      result->must_parallelism = this->must_parallelism;
      result->index_domain = this->index_domain;
      result->index_point = p;
      // Now figure out our local point information
      result->initialize_point(this, mp);
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_slice_point(get_unique_id(), 
                                   result->get_unique_id(),
                                   result->index_point);
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_domain.get_volume() > 0);
#endif
      // Enumerate all the points
      for (std::map<DomainPoint,MinimalPoint*>::const_iterator it = 
            minimal_points.begin(); it != minimal_points.end(); it++)
      {
        PointTask *next_point = clone_as_point_task(it->first, it->second);
        points.push_back(next_point);
        // We can now delete our old minimal points
        delete it->second;
      }
      minimal_points.clear();
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
    void SliceTask::return_virtual_instance(unsigned index, InstanceSet &refs)
    //--------------------------------------------------------------------------
    {
      // Add it to our state
#ifdef DEBUG_HIGH_LEVEL
      assert(refs.size() == 1);
      assert(refs[0].is_composite_ref());
#endif
      RegionTreeContext virtual_ctx = get_parent_context(index);
      // Have to control access to the version info data structure
      AutoLock o_lock(op_lock);
      // Hold a reference so it doesn't get deleted
      temporary_virtual_refs.push_back(refs[0]);
      runtime->forest->physical_register_only(virtual_ctx, regions[index],
                                              version_infos[index], this,
                                              Event::NO_EVENT, refs
#ifdef DEBUG_HIGH_LEVEL
                                              , index, get_logging_name()
                                              , unique_op_id
#endif
                                              );
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
      // No matter what, flush out our physical states
      Event applied_condition = Event::NO_EVENT;
      if (!version_infos.empty())
      {
        std::set<Event> applied_conditions;
        AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                           owner_space, applied_conditions);
        }
        if (!applied_conditions.empty())
          applied_condition = Runtime::merge_events<true>(applied_conditions);
      }
      if (is_remote())
      {
        bool has_nonleaf_point = false;
        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          if (!points[idx]->is_leaf())
          {
            has_nonleaf_point = true;
            break;
          }
        }

        // Only need to send something back if this wasn't mapped locally
        // wclee: also need to send back if there were some non-leaf point tasks
        // because they haven't recorded themselves as mapped
        if (!is_locally_mapped() || has_nonleaf_point)
        {
          Serializer rez;
          pack_remote_mapped(rez, applied_condition);
          runtime->send_slice_remote_mapped(orig_proc, rez);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!IS_WRITE(regions[idx]))
            continue;
          if (regions[idx].handle_type != SINGULAR)
          {
            // Construct a set of regions for all the children
            std::vector<LogicalRegion> handles(points.size());
            for (unsigned pidx = 0; pidx < points.size(); pidx++)
              handles[pidx] = points[pidx]->regions[idx].region;
          }
          // otherwise it was locally mapped so we are already done
        }
        index_owner->return_slice_mapped(points.size(), denominator, 
                                         applied_condition);
      }
      complete_mapping(applied_condition);
      complete_execution();
      // Now that we've mapped, we can remove any composite references
      // that we are holding
      if (!temporary_virtual_refs.empty())
        temporary_virtual_refs.clear();
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
      // We can release our version infos now
      for (std::vector<VersionInfo>::iterator it = version_infos.begin();
            it != version_infos.end(); it++)
      {
        it->release();
      }
      version_infos.clear();
      commit_operation(true/*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_mapped(Serializer &rez, Event applied_condition)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(denominator);
      rez.serialize(applied_condition);
      // Also pack up any regions names we need for doing invalidations
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!IS_WRITE(regions[idx]))
          continue;
        if (regions[idx].handle_type == SINGULAR)
          continue;
        for (unsigned pidx = 0; pidx < points.size(); pidx++)
          rez.serialize(points[pidx]->regions[idx].region);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_complete(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Send back any created state that our point tasks made
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      for (std::deque<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        (*it)->send_back_created_state(target, version_infos.size(),
                                       remote_outermost_context);
      }
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize<size_t>(points.size());
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
        for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator it =
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
      UserEvent ready_event;
      derez.deserialize(ready_event);
      ready_event.trigger();
    }

    /////////////////////////////////////////////////////////////
    // Deferred Slicer 
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
        std::list<SliceTask*>::const_iterator it = slices.begin();
        DeferredSliceArgs args;
        args.hlr_id = HLR_DEFERRED_SLICE_ID;
        args.slicer = this;
        while (true) 
        {
          args.slice = *it;
          it++;
          bool done = (it == slices.end()); 
          Event wait = owner->runtime->issue_runtime_meta_task(&args, 
                                sizeof(args), HLR_DEFERRED_SLICE_ID, owner);
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
        Event sliced_event = Runtime::merge_events<true>(wait_events);
        sliced_event.wait();
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

    /////////////////////////////////////////////////////////////
    // Minimal Point 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MinimalPoint::MinimalPoint(void)
      : arg(NULL), arglen(0), own_arg(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MinimalPoint::MinimalPoint(const MinimalPoint &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MinimalPoint::~MinimalPoint(void)
    //--------------------------------------------------------------------------
    {
      if (own_arg)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(arg != NULL);
        assert(arglen > 0);
#endif
        free(arg);
      }
    }

    //--------------------------------------------------------------------------
    MinimalPoint& MinimalPoint::operator=(const MinimalPoint &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MinimalPoint::add_projection_region(unsigned idx, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(projections.find(idx) == projections.end());
#endif
      projections[idx] = handle;
    }
    
    //--------------------------------------------------------------------------
    void MinimalPoint::add_argument(const TaskArgument &argument, bool own)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(arg == NULL);
#endif
      arg = argument.get_ptr();
      arglen = argument.get_size();
      own_arg = own;
    }

    //--------------------------------------------------------------------------
    void MinimalPoint::assign_argument(void *&local_arg, size_t &local_arglen)
    //--------------------------------------------------------------------------
    {
      // If we own it, we can just give it      
      if (own_arg)
      {
        local_arg = arg;
        local_arglen = arglen;
        arg = 0;
        arglen = 0;
        own_arg = false;
      }
      else if (arg != NULL)
      {
        local_arglen = arglen;
        local_arg = malloc(arglen);
        memcpy(local_arg, arg, arglen);
      }
      // Otherwise there is no argument so we are done
    }

    //--------------------------------------------------------------------------
    LogicalRegion MinimalPoint::find_logical_region(unsigned index)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,LogicalRegion>::const_iterator finder = 
        projections.find(index);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != projections.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void MinimalPoint::pack(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(projections.size());
      for (std::map<unsigned,LogicalRegion>::const_iterator it = 
            projections.begin(); it != projections.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize(arglen);
      if (arglen > 0)
        rez.serialize(arg, arglen);
    }

    //--------------------------------------------------------------------------
    void MinimalPoint::unpack(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_projections;
      derez.deserialize(num_projections);
      for (unsigned idx = 0; idx < num_projections; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        derez.deserialize(projections[index]);
      }
      derez.deserialize(arglen);
      if (arglen > 0)
      {
        arg = malloc(arglen);
        derez.deserialize(arg, arglen);
        own_arg = true;
      }
    }

  }; // namespace Internal 
}; // namespace Legion 

#undef PRINT_REG

// EOF

