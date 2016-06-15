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
#include "legion_analysis.h"
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(current.exists());
      assert(runtime->is_local(current));
#endif
      // Always clear target_proc and the mapper when setting a new current proc
      mapper = NULL;
      current_proc = current;
      target_proc = current;
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
      must_epoch = NULL;
      must_epoch_task = false;
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
      created_index_partitions.clear();
      deleted_regions.clear();
      deleted_fields.clear();
      deleted_field_spaces.clear();
      deleted_index_spaces.clear();
      deleted_index_partitions.clear();
      parent_req_indexes.clear();
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_must_epoch(MustEpochOp *epoch, unsigned index,
                                bool do_registration)
    //--------------------------------------------------------------------------
    {
      Operation::set_must_epoch(epoch, do_registration);
      must_epoch_index = index;
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_BASE_TASK_CALL);
      // pack all the user facing data first
      pack_base_external_task(rez, target); 
      RezCheck z(rez);
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
    void TaskOp::unpack_base_task(Deserializer &derez,
                                  std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_BASE_TASK_CALL);
      // unpack all the user facing data
      unpack_base_external_task(derez); 
      DerezCheck z(derez);
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
        early_mapped_regions[index].unpack_references(runtime, this, derez, 
                                                      ready_events);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_external_task(Serializer &rez, AddressSpaceID target)
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
      // If we are remote we can just do the normal pack
      for (unsigned idx = 0; idx < futures.size(); idx++)
        rez.serialize(futures[idx].impl->did);
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
      rez.serialize(must_epoch_task);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      rez.serialize(local_arglen);
      rez.serialize(local_args,local_arglen);
      rez.serialize(orig_proc);
      // No need to pack current proc, it will get set when we unpack
      rez.serialize(steal_count);
      // No need to pack remote, it will get set
      rez.serialize(speculated);
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_base_external_task(Deserializer &derez)
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
        futures[idx] = Future(
            runtime->find_or_create_future(future_did, this));
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
#ifdef DEBUG_LEGION
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
      derez.deserialize(must_epoch_task);
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
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              if (!ready_events.empty())
              {
                RtEvent ready = Runtime::merge_events(ready_events);
                rt->add_to_ready_queue(current, task, false/*prev fail*/,ready);
              }
              else
                rt->add_to_ready_queue(current, task, false/*prev fail*/);
            }
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task(false);
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              if (!ready_events.empty())
              {
                RtEvent ready = Runtime::merge_events(ready_events);
                rt->add_to_ready_queue(current, task, false/*prev fail*/,ready);
              }
              else
                rt->add_to_ready_queue(current, task, false/*prev fail*/);
            }
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
      parent_task = ctx; // initialize the parent task
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
        if (regions[idx].privilege != NO_ACCESS && 
            regions[idx].privilege_fields.empty())
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
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      const size_t result_size = impl->get_untyped_size();
      // TODO: figure out a way to put this check back in with dynamic task
      // registration where we might not know the return size until later
#ifdef PERFORM_PREDICATE_SIZE_CHECKS
      if (result_size != variants->return_size)
      {
        log_run.error("Predicated task launch for task %s "
                      "in parent task %s (UID %lld) has predicated "
                      "false future of size %ld bytes, but the "
                      "expected return size is %ld bytes.",
                      get_task_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id(),
                      result_size, variants->return_size);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
    const std::vector<VersionInfo>* TaskOp::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return NULL;
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
      if (!is_remote())
      {
        if (idx < parent_req_indexes.size())
          return parent_ctx->find_enclosing_context(parent_req_indexes[idx]);
        else
          return parent_ctx->find_outermost_context()->get_context();
      }
      // This is remote, so just return the context of the remote parent
      return parent_ctx->get_context(); 
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_version_infos(Serializer &rez,
                                    std::vector<VersionInfo> &infos,
                                    const std::vector<bool> &full_version_infos)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
#ifdef DEBUG_LEGION
      assert(infos.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < infos.size(); idx++)
      {
        rez.serialize<bool>(full_version_infos[idx]);
        if (full_version_infos[idx])
          infos[idx].pack_version_info(rez);
        else
          infos[idx].pack_version_numbers(rez);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_version_infos(Deserializer &derez,
                                      std::vector<VersionInfo> &infos)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      infos.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool full_info;
        derez.deserialize(full_info);
        if (full_info)
          infos[idx].unpack_version_info(derez);
        else
          infos[idx].unpack_version_numbers(derez, runtime->forest);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(created_regions.find(handle) == created_regions.end());
#endif
      created_regions.insert(handle); 
      add_created_region(handle);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_deletion(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      {
        AutoLock o_lock(op_lock);
        std::set<LogicalRegion>::iterator finder = created_regions.find(handle);
        // See if we created this region, if so remove it from the list
        // of created regions, otherwise add it to the list of deleted
        // regions to flow backwards
        if (finder != created_regions.end())
        {
          created_regions.erase(finder);
          finalize = true;
        }
        else
          deleted_regions.insert(handle);
      }
      if (finalize)
        runtime->finalize_logical_region_destroy(handle);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_region_creations(const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<LogicalRegion>::const_iterator it = regs.begin();
            it != regs.end(); it++)
      {
#ifdef DEBUG_LEGION
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
      std::vector<LogicalRegion> to_finalize;
      {
        AutoLock o_lock(op_lock);
        for (std::set<LogicalRegion>::const_iterator it = regs.begin();
              it != regs.end(); it++)
        {
          std::set<LogicalRegion>::iterator finder = created_regions.find(*it);
          if (finder != created_regions.end())
          {
            created_regions.erase(finder);
            to_finalize.push_back(*it);
          }
          else
            deleted_regions.insert(*it);
        }
      }
      if (!to_finalize.empty())
      {
        for (std::vector<LogicalRegion>::const_iterator it = 
              to_finalize.begin(); it != to_finalize.end(); it++)
          runtime->finalize_logical_region_destroy(*it);
      }
    } 

    //--------------------------------------------------------------------------
    void TaskOp::register_field_creation(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::pair<FieldSpace,FieldID> key(handle,fid);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      std::set<FieldID> to_finalize;
      {
        AutoLock o_lock(op_lock);
        for (std::set<FieldID>::const_iterator it = to_free.begin();
              it != to_free.end(); it++)
        {
          std::pair<FieldSpace,FieldID> key(handle,*it);
          std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
            created_fields.find(key);
          if (finder != created_fields.end())
          {
            created_fields.erase(finder);
            to_finalize.insert(*it);
          }
          else
            deleted_fields.insert(key);
        }
      }
      if (!to_finalize.empty())
        runtime->finalize_field_destroy(handle, to_finalize);
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
#ifdef DEBUG_LEGION
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
      std::map<FieldSpace,std::set<FieldID> > to_finalize;
      {
        AutoLock o_lock(op_lock);
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
              fields.begin(); it != fields.end(); it++)
        {
          std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
            created_fields.find(*it);
          if (finder != created_fields.end())
          {
            created_fields.erase(finder);
            to_finalize[it->first].insert(it->second);
          }
          else
            deleted_fields.insert(*it);
        }
      }
      if (!to_finalize.empty())
      {
        for (std::map<FieldSpace,std::set<FieldID> >::const_iterator it = 
              to_finalize.begin(); it != to_finalize.end(); it++)
        {
          runtime->finalize_field_destroy(it->first, it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_creation(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(created_field_spaces.find(space) == created_field_spaces.end());
#endif
      created_field_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_field_space_deletion(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock o_lock(op_lock);
        std::deque<FieldID> to_delete;
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
              created_fields.begin(); it != created_fields.end(); it++)
        {
          if (it->first == space)
            to_delete.push_back(it->second);
        }
        for (unsigned idx = 0; idx < to_delete.size(); idx++)
        {
          std::pair<FieldSpace,FieldID> key(space, to_delete[idx]);
          created_fields.erase(key);
        }
        std::set<FieldSpace>::iterator finder = 
          created_field_spaces.find(space);
        if (finder != created_field_spaces.end())
        {
          created_field_spaces.erase(finder);
          finalize = true;
        }
        else
          deleted_field_spaces.insert(space);
      }
      if (finalize)
        runtime->finalize_field_space_destroy(space);
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
#ifdef DEBUG_LEGION
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
      std::vector<FieldSpace> to_finalize;
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
              to_delete.push_back(cit->second);
          }
          for (unsigned idx = 0; idx < to_delete.size(); idx++)
          {
            std::pair<FieldSpace,FieldID> key(*it, to_delete[idx]);
            created_fields.erase(key);
          }
          std::set<FieldSpace>::iterator finder = created_field_spaces.find(*it);
          if (finder != created_field_spaces.end())
          {
            created_field_spaces.erase(finder);
            to_finalize.push_back(*it);
          }
          else
            deleted_field_spaces.insert(*it);
        }
      }
      if (!to_finalize.empty())
      {
        for (std::vector<FieldSpace>::const_iterator it = to_finalize.begin();
              it != to_finalize.end(); it++)
          runtime->finalize_field_space_destroy(*it);
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
#ifdef DEBUG_LEGION
      assert(created_index_spaces.find(space) == created_index_spaces.end());
#endif
      created_index_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_deletion(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock o_lock(op_lock);
        std::set<IndexSpace>::iterator finder = 
          created_index_spaces.find(space);
        if (finder != created_index_spaces.end())
        {
          created_index_spaces.erase(finder);
          finalize = true;
        }
        else
          deleted_index_spaces.insert(space);
      }
      if (finalize)
        runtime->finalize_index_space_destroy(space);
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
#ifdef DEBUG_LEGION
        assert(created_index_spaces.find(*it) == created_index_spaces.end());
#endif
        created_index_spaces.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_space_deletions(
                                            const std::set<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> to_finalize;
      {
        AutoLock o_lock(op_lock);
        for (std::set<IndexSpace>::const_iterator it = spaces.begin();
              it != spaces.end(); it++)
        {
          std::set<IndexSpace>::iterator finder = 
            created_index_spaces.find(*it);
          if (finder != created_index_spaces.end())
          {
            created_index_spaces.erase(finder);
            to_finalize.push_back(*it);
          }
          else
            deleted_index_spaces.insert(*it);
        }
      }
      if (!to_finalize.empty())
      {
        for (std::vector<IndexSpace>::const_iterator it = to_finalize.begin();
              it != to_finalize.end(); it++)
          runtime->finalize_index_space_destroy(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_partition_creation(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(created_index_partitions.find(handle) == 
             created_index_partitions.end());
#endif
      created_index_partitions.insert(handle);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_partition_deletion(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock o_lock(op_lock);
        std::set<IndexPartition>::iterator finder = 
          created_index_partitions.find(handle);
        if (finder != created_index_partitions.end())
        {
          created_index_partitions.erase(finder);
          finalize = true;
        }
        else
          deleted_index_partitions.insert(handle);
      }
      if (finalize)
        runtime->finalize_index_partition_destroy(handle);
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_partition_creations(
                                          const std::set<IndexPartition> &parts)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      for (std::set<IndexPartition>::const_iterator it = parts.begin();
            it != parts.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(created_index_partitions.find(*it) == 
               created_index_partitions.end());
#endif
        created_index_partitions.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::register_index_partition_deletions(
                                          const std::set<IndexPartition> &parts)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexPartition> to_finalize;
      {
        AutoLock o_lock(op_lock);
        for (std::set<IndexPartition>::const_iterator it = parts.begin();
              it != parts.end(); it++)
        {
          std::set<IndexPartition>::iterator finder = 
            created_index_partitions.find(*it);
          if (finder != created_index_partitions.end())
          {
            created_index_partitions.erase(finder);
            to_finalize.push_back(*it);
          }
          else
            deleted_index_partitions.insert(*it);
        }
      }
      if (!to_finalize.empty())
      {
        for (std::vector<IndexPartition>::const_iterator it = 
              to_finalize.begin(); it != to_finalize.end(); it++)
          runtime->finalize_index_partition_destroy(*it);
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
      if (!created_index_partitions.empty())
        target->register_index_partition_creations(created_index_partitions);
      if (!deleted_index_partitions.empty())
        target->register_index_partition_deletions(deleted_index_partitions);
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_privilege_state(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Shouldn't need the lock here since we only do this
      // while there is no one else executing
      RezCheck z(rez);
      rez.serialize<size_t>(created_regions.size());
      if (!created_regions.empty())
      {
        for (std::set<LogicalRegion>::const_iterator it =
              created_regions.begin(); it != created_regions.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(deleted_regions.size());
      if (!deleted_regions.empty())
      {
        for (std::set<LogicalRegion>::const_iterator it =
              deleted_regions.begin(); it != deleted_regions.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(created_fields.size());
      if (!created_fields.empty())
      {
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it =
              created_fields.begin(); it != created_fields.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(deleted_fields.size());
      if (!deleted_fields.empty())
      {
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
              deleted_fields.begin(); it != deleted_fields.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(created_field_spaces.size());
      if (!created_field_spaces.empty())
      {
        for (std::set<FieldSpace>::const_iterator it = 
              created_field_spaces.begin(); it != 
              created_field_spaces.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(deleted_field_spaces.size());
      if (!deleted_field_spaces.empty())
      {
        for (std::set<FieldSpace>::const_iterator it = 
              deleted_field_spaces.begin(); it !=
              deleted_field_spaces.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(created_index_spaces.size());
      if (!created_index_spaces.empty())
      {
        for (std::set<IndexSpace>::const_iterator it = 
              created_index_spaces.begin(); it != 
              created_index_spaces.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(deleted_index_spaces.size());
      if (!deleted_index_spaces.empty())
      {
        for (std::set<IndexSpace>::const_iterator it = 
              deleted_index_spaces.begin(); it !=
              deleted_index_spaces.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(created_index_partitions.size());
      if (!created_index_partitions.empty())
      {
        for (std::set<IndexPartition>::const_iterator it = 
              created_index_partitions.begin(); it !=
              created_index_partitions.end(); it++)
        {
          rez.serialize(*it);
        }
      }
      rez.serialize<size_t>(deleted_index_partitions.size());
      if (!deleted_index_partitions.empty())
      {
        for (std::set<IndexPartition>::const_iterator it = 
              deleted_index_partitions.begin(); it !=
              deleted_index_partitions.end(); it++)
        {
          rez.serialize(*it);
        }
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
      size_t num_created_index_partitions;
      derez.deserialize(num_created_index_partitions);
      for (unsigned idx = 0; idx < num_created_index_partitions; idx++)
      {
        IndexPartition ip;
        derez.deserialize(ip);
        created_index_partitions.insert(ip);
      }
      size_t num_deleted_index_partitions;
      derez.deserialize(num_deleted_index_partitions);
      for (unsigned idx = 0; idx < num_deleted_index_partitions; idx++)
      {
        IndexPartition ip;
        derez.deserialize(ip);
        deleted_index_partitions.insert(ip);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::perform_privilege_checks(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TASK_PRIVILEGE_CHECK_CALL);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_PARTITION_HANDLE);
            }
          case ERROR_BAD_PROJECTION_USE:
            {
              log_region.error("Projection region requirement %d used "
                                "in non-index space task %s",
                                idx, get_task_name());
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, CLONE_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(p.exists());
#endif
      // From Operation
      this->parent_ctx = rhs->parent_ctx;
      // Don't register this an operation when setting the must epoch info
      if (rhs->must_epoch != NULL)
        this->set_must_epoch(rhs->must_epoch, rhs->must_epoch_index,
                             false/*do registration*/);
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
#ifdef DEBUG_LEGION
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
      this->parent_task = rhs->parent_task;
      // Premapping should never get cloned
      this->map_locally = rhs->map_locally;
      // From TaskOp
      this->atomic_locks = rhs->atomic_locks;
      this->early_mapped_regions = rhs->early_mapped_regions;
      this->parent_req_indexes = rhs->parent_req_indexes;
      this->current_proc = rhs->current_proc;
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
      ApEvent arrive_pre = get_task_completion();
      for (std::vector<PhaseBarrier>::const_iterator it = 
            phase_barriers.begin(); it != phase_barriers.end(); it++)
      {
        // Update the arrival count
        arrive_barriers.push_back(*it);
        // Note it is imperative we do this off the new barrier
        // generated after updating the arrival count.
        Runtime::phase_barrier_arrive(*it, 1/*count*/, arrive_pre);
      }
    }

    //--------------------------------------------------------------------------
    bool TaskOp::compute_point_region_requirements(MinimalPoint *mp/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, COMPUTE_POINT_REQUIREMENTS_CALL);
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
#ifdef DEBUG_LEGION
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
      // Log our requirements that we computed
      if (Runtime::legion_spy_enabled)
      {
        UniqueID our_uid = get_unique_id();
        for (unsigned idx = 0; idx < regions.size(); idx++)
          log_requirement(our_uid, idx, regions[idx]);
      }
      // Return true if this point has any valid region requirements
      return (!all_invalid);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::early_map_regions(std::set<RtEvent> &applied_conditions,
                                   const std::vector<unsigned> &must_premap)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, EARLY_MAP_REGIONS_CALL);
      Mapper::PremapTaskInput input;
      Mapper::PremapTaskOutput output;
      // Initialize this to not have a new target processor
      output.new_target_proc = Processor::NO_PROC;
      // Set up the inputs and outputs 
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (std::vector<unsigned>::const_iterator it = must_premap.begin();
            it != must_premap.end(); it++)
      {
        InstanceSet valid;    
        VersionInfo &version_info = get_version_info(*it);
        RegionTreeContext req_ctx = get_parent_context(*it);
        RegionTreePath &privilege_path = get_privilege_path(*it);
        // Do the premapping
        runtime->forest->physical_traverse_path(req_ctx, privilege_path,
                                                regions[*it],
                                                version_info, this, *it, 
                                                true/*find valid*/, 
                                                applied_conditions, valid
#ifdef DEBUG_LEGION
                                                , get_logging_name()
                                                , unique_op_id
#endif
                                                );
        // If we need visible instances, filter them as part of the conversion
        if (regions[*it].is_no_access())
          prepare_for_mapping(valid, input.valid_instances[*it]);
        else
          prepare_for_mapping(valid, visible_memories, 
                              input.valid_instances[*it]);
      }
      // Now invoke the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, &input, &output);
      // See if we need to update the new target processor
      if (output.new_target_proc.exists())
        this->target_proc = output.new_target_proc;
      // Now do the registration
      for (std::vector<unsigned>::const_iterator it = must_premap.begin();
            it != must_premap.end(); it++)
      {
        VersionInfo &version_info = get_version_info(*it);
        RegionTreeContext req_ctx = get_parent_context(*it);
        InstanceSet &chosen_instances = early_mapped_regions[*it];
        // If this is restricted then we know what the answer is so
        // just ignore whatever the mapper did
        if (regions[*it].is_restricted())
        {
          // Since we know we are on the owner node, we know we can
          // always ask our parent context to find the restricted instances
          parent_ctx->get_physical_references(
              parent_req_indexes[*it], chosen_instances);
        }
        else
        {
          // Otherwise this was not restricted, so do what the mapper wants
          std::map<unsigned,std::vector<MappingInstance> >::const_iterator 
            finder = output.premapped_instances.find(*it);
          if (finder == output.premapped_instances.end())
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper failed to map required premap "
                          "region requirement %d of task %s (ID %lld) launched "
                          "in parent task %s (ID %lld).", 
                          mapper->get_mapper_name(), *it, 
                          get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          RegionTreeID bad_tree = 0;
          std::vector<FieldID> missing_fields;
          std::vector<PhysicalManager*> unacquired;
          int composite_index = runtime->forest->physical_convert_mapping(
              this, regions[*it], finder->second, 
              chosen_instances, bad_tree, missing_fields,
              Runtime::unsafe_mapper ? NULL : get_acquired_instances_ref(),
              unacquired, !Runtime::unsafe_mapper);
          if (bad_tree > 0)
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper provided an instanced from "
                          "region tree %d for use in satisfying region "
                          "requirement %d of task %s (ID %lld) whose region "
                          "is from region tree %d.", mapper->get_mapper_name(),
                          bad_tree, *it, get_task_name(), get_unique_id(), 
                          regions[*it].region.get_tree_id());
#ifdef DEBUG_LEGION
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
                          *it, get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(), 
                          parent_ctx->get_unique_id());
            for (std::vector<FieldID>::const_iterator it = 
                  missing_fields.begin(); it != missing_fields.end(); it++)
            {
              const void *name; size_t name_size;
              runtime->retrieve_semantic_information(
                  regions[*it].region.get_field_space(), *it,
                  NAME_SEMANTIC_TAG, name, name_size, false, false);
              log_run.error("Missing instance for field %s (FieldID: %d)",
                            static_cast<const char*>(name), *it);
            }
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
          if (!unacquired.empty())
          {
            std::map<PhysicalManager*,std::pair<unsigned,bool> > 
              *acquired_instances = get_acquired_instances_ref();
            for (std::vector<PhysicalManager*>::const_iterator uit = 
                  unacquired.begin(); uit != unacquired.end(); uit++)
            {
              if (acquired_instances->find(*uit) == acquired_instances->end())
              {
                log_run.error("Invalid mapper output from 'premap_task' "
                              "invocation on mapper %s. Mapper selected "
                              "physical instance for region requirement "
                              "%d of task %s (ID %lld) which has already "
                              "been collected. If the mapper had properly "
                              "acquired this instance as part of the mapper "
                              "call it would have detected this. Please "
                              "update the mapper to abide by proper mapping "
                              "conventions.", mapper->get_mapper_name(),
                              (*it), get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
            // If we did successfully acquire them, still issue the warning
            log_run.warning("WARNING: mapper %s failed to acquire instances "
                            "for region requirement %d of task %s (ID %lld) "
                            "in 'premap_task' call. You may experience "
                            "undefined behavior as a consequence.",
                            mapper->get_mapper_name(), *it, 
                            get_task_name(), get_unique_id());
          }
          if (composite_index >= 0)
          {
            log_run.error("Invalid mapper output from 'premap_task' invocation "
                          "on mapper %s. Mapper requested composite instance "
                          "creation on region requirement %d of task %s "
                          "(ID %lld) launched in parent task %s (ID %lld).",
                          mapper->get_mapper_name(), *it,
                          get_task_name(), get_unique_id(),
                          parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          } 
          if (Runtime::legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, *it,
                                                  regions[*it],
                                                  chosen_instances);
          if (!Runtime::unsafe_mapper)
          {
            std::vector<LogicalRegion> regions_to_check(1, 
                                          regions[*it].region);
            for (unsigned check_idx = 0; 
                  check_idx < chosen_instances.size(); check_idx++)
            {
              if (!chosen_instances[check_idx].get_manager()->meets_regions(
                                                            regions_to_check))
              {
                log_run.error("Invalid mapper output from invocation of "
                              "'premap_task' on mapper %s. Mapper specified an "
                              "instance region requirement %d of task %s "
                              "(ID %lld) that does not meet the logical region "
                              "requirement. Task was launched in task %s "
                              "(ID %lld).", mapper->get_mapper_name(), *it, 
                              get_task_name(), get_unique_id(), 
                              parent_ctx->get_task_name(), 
                              parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
        }
        // Set the current mapping index before doing anything that
        // could result in the generation of a copy
        set_current_mapping_index(*it);
        // Passed all the error checking tests so register it
        runtime->forest->physical_register_only(req_ctx, 
                              regions[*it], version_info, this, *it,
                              completion_event, (regions.size() > 1), 
                              applied_conditions, chosen_instances
#ifdef DEBUG_LEGION
                              , get_logging_name(), unique_op_id
#endif
                              );
        // Now apply our mapping
        AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
        version_info.apply_mapping(req_ctx.get_id(), owner_space,
                                   applied_conditions);
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
      DETAILED_PROFILER(runtime, RECORD_ALIASED_REQUIREMENTS_CALL);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
          req.privilege, req.prop, req.redop, req.parent.index_space.id);
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
      DETAILED_PROFILER(runtime, ACTIVATE_SINGLE_CALL);
      activate_task();
      executing_processor = Processor::NO_PROC;
      current_fence = NULL;
      fence_gen = 0;
      context = RegionTreeContext();
      valid_wait_event = false;
      deferred_map = RtEvent::NO_RT_EVENT;
      deferred_complete = RtEvent::NO_RT_EVENT; 
      pending_done = RtEvent::NO_RT_EVENT;
      last_registration = RtEvent::NO_RT_EVENT;
      dependence_precondition = RtEvent::NO_RT_EVENT;
      profiling_done = RtEvent::NO_RT_EVENT;
      current_trace = NULL;
      task_executed = false;
      outstanding_children_count = 0;
      outstanding_subtasks = 0;
      pending_subtasks = 0;
      pending_frames = 0;
      context_order_event = RtEvent::NO_RT_EVENT;
      // Set some of the default values for a context
      context_configuration.max_window_size = 
        Runtime::initial_task_window_size;
      context_configuration.hysteresis_percentage = 
        Runtime::initial_task_window_hysteresis;
      context_configuration.max_outstanding_frames = 0;
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
      DETAILED_PROFILER(runtime, DEACTIVATE_SINGLE_CALL);
      deactivate_task();
      target_processors.clear();
      physical_instances.clear();
      physical_regions.clear();
      created_requirements.clear();
      inline_regions.clear();
      virtual_mapped.clear();
      no_access_regions.clear();
      executing_children.clear();
      executed_children.clear();
      complete_children.clear();
      safe_cast_domains.clear();
      restricted_trees.clear();
      frame_events.clear();
      map_applied_conditions.clear();
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
        Realm::Barrier bar = context_barriers.back();
        bar.destroy_barrier();
        context_barriers.pop_back();
      }
      local_fields.clear();
      if (valid_wait_event)
      {
        valid_wait_event = false;
        Runtime::trigger_event(window_wait);
      }
      // Clean up our instance top views
      if (!instance_top_views.empty())
      {
        for (std::map<PhysicalManager*,InstanceView*>::const_iterator it = 
              instance_top_views.begin(); it != instance_top_views.end(); it++)
        {
          it->first->unregister_active_context(this);
          if (it->second->remove_base_resource_ref(CONTEXT_REF))
            LogicalView::delete_logical_view(it->second);
        }
        instance_top_views.clear();
      }
#ifdef DEBUG_LEGION
      assert(pending_top_views.empty());
      assert(outstanding_subtasks == 0);
      assert(pending_subtasks == 0);
      assert(pending_frames == 0);
#endif
      if (context.exists())
        runtime->free_local_context(this);
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
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (virtual_mapped[idx])
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
    bool SingleTask::is_created_region(unsigned index) const
    //--------------------------------------------------------------------------
    {
      return (index >= regions.size());
    }

    //--------------------------------------------------------------------------
    void SingleTask::update_no_access_regions(void)
    //--------------------------------------------------------------------------
    {
      no_access_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        no_access_regions[idx] = IS_NO_ACCESS(regions[idx]) || 
                                  regions[idx].privilege_fields.empty();
    }

    //--------------------------------------------------------------------------
    void SingleTask::assign_context(RegionTreeContext ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!context.exists());
#endif
      context = ctx;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext SingleTask::release_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      return context;
    }

    //--------------------------------------------------------------------------
    ContextID SingleTask::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
    void SingleTask::destroy_user_barrier(ApBarrier b)
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
#ifdef DEBUG_LEGION
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx];
    } 

    //--------------------------------------------------------------------------
    void SingleTask::get_physical_references(unsigned idx, InstanceSet &set)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(idx < physical_instances.size());
#endif
      set = physical_instances[idx];
    }

    //--------------------------------------------------------------------------
    VariantImpl* SingleTask::select_inline_variant(TaskOp *child,
                                                   InlineTask *inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SELECT_INLINE_VARIANT_CALL);
      Mapper::SelectVariantInput input;
      Mapper::SelectVariantOutput output;
      input.processor = current_proc;
      input.chosen_instances.resize(child->regions.size());
      // Compute the parent indexes since we're going to need them
      child->compute_parent_indexes();
      // Find the instances for this child
      for (unsigned idx = 0; idx < child->regions.size(); idx++)
      {
        // We can get access to physical_regions without the
        // lock because we know we are running in the application
        // thread in order to do this inlining
        unsigned local_index = child->find_parent_index(idx); 
#ifdef DEBUG_LEGION
        assert(local_index < physical_regions.size());
#endif
        InstanceSet instances;
        physical_regions[local_index].impl->get_references(instances);
        std::vector<MappingInstance> &mapping_instances = 
          input.chosen_instances[idx];
        mapping_instances.resize(instances.size());
        for (unsigned idx2 = 0; idx2 < instances.size(); idx2++)
        {
          mapping_instances[idx2] = 
            MappingInstance(instances[idx2].get_manager());
        }
      }
      output.chosen_variant = 0;
      // Always do this with the child mapper
      MapperManager *child_mapper = runtime->find_mapper(current_proc, 
                                                         child->map_id);
      child_mapper->invoke_select_task_variant(child, &input, &output);
      VariantImpl *variant_impl= runtime->find_variant_impl(child->task_id,
                                  output.chosen_variant, true/*can fail*/);
      if (variant_impl == NULL)
      {
        log_run.error("Invalid mapper output from invoction of "
                      "'select_task_variant' on mapper %s. Mapper selected "
                      "an invalidate variant ID %ld for inlining of task %s "
                      "(UID %lld).", child_mapper->get_mapper_name(),
                      output.chosen_variant, child->get_task_name(), 
                      child->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (!Runtime::unsafe_mapper)
        inline_task->validate_variant_selection(child_mapper, variant_impl, 
                                                "select_task_variant");
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    void SingleTask::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INLINE_CHILD_TASK_CALL);
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

      // Save the state of our physical regions
      std::vector<bool> phy_regions_mapped(physical_regions.size());
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
        phy_regions_mapped[idx] = is_region_mapped(idx);
 
      // Also save the original number of child regions
      unsigned orig_child_regions = inline_task->regions.size();

      // Pick a variant to use for executing this task
      VariantImpl *variant = select_inline_variant(child, inline_task);    
      
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
      std::set<ApEvent> wait_events;
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
        ApEvent wait_on = Runtime::merge_events(wait_events);
        if (!wait_on.has_triggered())
          wait_on.wait();
      }
      // Now we can deactivate our inline task
      inline_task->deactivate();
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
      DETAILED_PROFILER(runtime, PACK_SINGLE_TASK_CALL);
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
    void SingleTask::unpack_single_task(Deserializer &derez,
                                        std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_SINGLE_TASK_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events);
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
        physical_instances[idx].unpack_references(runtime, this,
                                                  derez, ready_events);
      virtual_mapped.resize(regions.size());
      for (unsigned idx = 0; idx < num_phy; idx++)
        virtual_mapped[idx] = physical_instances[idx].has_composite_ref();
      update_no_access_regions();
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_remote_context(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_REMOTE_CONTEXT_CALL);
      int depth = get_depth();
      rez.serialize(depth);
      // See if we need to pack up base task information
      pack_base_external_task(rez, target);
      // Pack up the version numbers only 
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        VersionInfo &info = get_version_info(idx);
        info.pack_version_numbers(rez);
      }
      // Pack up our virtual mapping information
      std::vector<unsigned> virtual_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (virtual_mapped[idx])
          virtual_indexes.push_back(idx);
      }
      rez.serialize<size_t>(virtual_indexes.size());
      for (unsigned idx = 0; idx < virtual_indexes.size(); idx++)
        rez.serialize(virtual_indexes[idx]);
      // Now pack up any local fields 
      LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked locals = 
                                                                  local_fields;
      find_enclosing_local_fields(locals);
      size_t num_local = locals.size();
      rez.serialize(num_local);
      for (unsigned idx = 0; idx < locals.size(); idx++)
        rez.serialize(locals[idx]);
      rez.serialize(get_task_completion());
      rez.serialize(get_context_uid());
      // Can happen if the top-level task is sent remotely
      if (parent_ctx != NULL)
        rez.serialize(parent_ctx->get_context_uid());
      else
        rez.serialize<UniqueID>(0);
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_remote_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      assert(false); // should only be called for RemoteTask
    }

    //--------------------------------------------------------------------------
    void SingleTask::send_back_created_state(AddressSpaceID target, 
                                             unsigned start,
                                             RegionTreeContext remote_outermost)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        RegionRequirement &req = created_requirements[idx];
        // If it was deleted, then we don't care
        if (created_regions.find(req.region) == created_regions.end())
          continue;
        FieldSpace fs = req.region.get_field_space();
        bool all_fields_deleted = true;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it != 
              req.privilege_fields.end(); it++)
        {
          if (created_fields.find(std::pair<FieldSpace,FieldID>(fs,*it)) != 
              created_fields.end())
          {
            all_fields_deleted = false;
            break;
          }
        }
        if (all_fields_deleted)
          continue;
        unsigned index = regions.size() + idx;
        runtime->forest->send_back_logical_state(get_parent_context(index),
                       remote_outermost, created_requirements[idx], target);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_new_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      // If we are performing a trace mark that the child has a trace
      if (current_trace != NULL)
        op->set_trace(current_trace, !current_trace->is_fixed());
      unsigned outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,1);
      // Only need to check if we are not tracing by frames
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count >= context_configuration.max_window_size))
      {
        // Try taking the lock first and see if we succeed
        RtEvent precondition = 
          Runtime::acquire_rt_reservation(op_lock, true/*exclusive*/);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // Launch a window-wait task and then wait on the event 
          WindowWaitArgs args;
          args.hlr_id = HLR_WINDOW_WAIT_TASK_ID;
          args.parent_ctx = this;  
          RtEvent wait_done = 
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_WINDOW_WAIT_TASK_ID, 
                                             HLR_RESOURCE_PRIORITY,
                                             this, precondition);
          wait_done.wait();
        }
        else // we can do the wait inline
          perform_window_wait();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_window_wait(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_event;
      // We already hold our lock from the callsite above
      if (outstanding_children_count >= context_configuration.max_window_size)
      {
#ifdef DEBUG_LEGION
        assert(!valid_wait_event);
#endif
        window_wait = Runtime::create_rt_user_event();
        valid_wait_event = true;
        wait_event = window_wait;
      }
      // Release our lock now
      op_lock.release();
      if (wait_event.exists() && !wait_event.has_triggered())
        wait_event.wait();
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_to_dependence_queue(Operation *op, bool has_lock)
    //--------------------------------------------------------------------------
    {
      if (!has_lock)
      {
        RtEvent lock_acquire = Runtime::acquire_rt_reservation(op_lock, 
                                true/*exclusive*/, last_registration);
        if (!lock_acquire.has_triggered())
        {
          AddToDepQueueArgs args;
          args.hlr_id = HLR_ADD_TO_DEP_QUEUE_TASK_ID;
          args.proxy_this = this;
          args.op = op;
          last_registration = 
            runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                             HLR_ADD_TO_DEP_QUEUE_TASK_ID,
                                             HLR_RESOURCE_PRIORITY,
                                             op, lock_acquire);
          return;
        }
      }
      // We have the lock
      if (op->is_tracking_parent())
      {
#ifdef DEBUG_LEGION
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
      RtEvent next = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                      HLR_TRIGGER_DEPENDENCE_ID, 
                                      HLR_LATENCY_PRIORITY, op,
                                      dependence_precondition);
      dependence_precondition = next;
      // Now we can release the lock
      op_lock.release();
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
        std::set<Operation*>::iterator finder = executing_children.find(op);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        assert(outstanding_count >= 0);
#endif
        if (valid_wait_event && (context_configuration.max_window_size > 0) &&
            (outstanding_count <=
             int(context_configuration.hysteresis_percentage * 
                 context_configuration.max_window_size / 100)))
        {
          to_trigger = window_wait;
          valid_wait_event = false;
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        std::set<Operation*>::iterator finder = executed_children.find(op);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
        // Remove it from everything and then see if we need to
        // trigger the window wait event
        executing_children.erase(op);
        executed_children.erase(op);
        complete_children.erase(op);
        int outstanding_count = 
          __sync_add_and_fetch(&outstanding_children_count,-1);
#ifdef DEBUG_LEGION
        assert(outstanding_count >= 0);
#endif
        if (valid_wait_event && (context_configuration.max_window_size > 0) &&
            (outstanding_count <=
             int(context_configuration.hysteresis_percentage * 
                 context_configuration.max_window_size / 100)))
        {
          to_trigger = window_wait;
          valid_wait_event = false;
        }
        // No need to see if we trigger anything else because this
        // method is only called while the task is still executing
        // so 'executed' is still false.
#ifdef DEBUG_LEGION
        assert(!executed);
#endif
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
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
    void SingleTask::perform_fence_analysis(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext ctx = get_context();
      // Do our internal regions first
      for (unsigned idx = 0; idx < regions.size(); idx++)
        runtime->forest->perform_fence_analysis(ctx, op, 
                                        regions[idx].region, true/*dominate*/);
      // Now see if we have any created regions
      std::vector<LogicalRegion> created_regions;
      {
        AutoLock o_lock(op_lock,1,false/*exclusive*/);
        if (created_requirements.empty())
          return;
        created_regions.resize(created_requirements.size());
        for (unsigned idx = 0; idx < created_requirements.size(); idx++)
          created_regions[idx] = created_requirements[idx].region;
      }
      // These get analyzed in the outermost context since they are
      // created regions
      RegionTreeContext outermost = find_outermost_context()->get_context();
      for (unsigned idx = 0; idx < created_regions.size(); idx++)
        runtime->forest->perform_fence_analysis(outermost, op, 
                                    created_regions[idx], true/*dominate*/);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNMATCHED_END_TRACE);
      }
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op(true);
        complete_op->initialize_complete(this);
        runtime->add_to_dependence_queue(get_executing_processor(),complete_op);
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(true); 
        capture_op->initialize_capture(this);
        runtime->add_to_dependence_queue(get_executing_processor(), capture_op);
        // Mark that the current trace is now fixed
        current_trace->fix_trace();
      }
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void SingleTask::issue_frame(FrameOp *frame, ApEvent frame_termination)
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
        RtEvent wait_on = runtime->issue_runtime_meta_task(&args, sizeof(args),
                                              HLR_ISSUE_FRAME_TASK_ID, 
                                              HLR_THROUGHPUT_PRIORITY, this);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_frame_issue(FrameOp *frame,
                                         ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on, previous;
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
    void SingleTask::finish_frame(ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      // Pull off all the frame events until we reach ours
      if (context_configuration.max_outstanding_frames > 0)
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(frame_events.front() == frame_termination);
#endif
        frame_events.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_outstanding(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((context_configuration.min_tasks_to_schedule == 0) || 
             (context_configuration.min_frames_to_schedule == 0));
      assert((context_configuration.min_tasks_to_schedule > 0) || 
             (context_configuration.min_frames_to_schedule > 0));
#endif
      RtEvent wait_on;
      RtUserEvent to_trigger;
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
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
        }
        outstanding_subtasks++;
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_outstanding(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((context_configuration.min_tasks_to_schedule == 0) || 
             (context_configuration.min_frames_to_schedule == 0));
      assert((context_configuration.min_tasks_to_schedule > 0) || 
             (context_configuration.min_frames_to_schedule > 0));
#endif
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
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
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
        pending_subtasks++;
        if ((outstanding_subtasks > 0) &&
            (pending_subtasks == context_configuration.min_tasks_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent SingleTask::decrement_pending(SingleTask *child) const
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduled by frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return RtEvent::NO_RT_EVENT;
      // This may involve waiting, so always issue it as a meta-task 
      DecrementArgs decrement_args;
      decrement_args.hlr_id = HLR_DECREMENT_PENDING_TASK_ID;
      decrement_args.parent_ctx = const_cast<SingleTask*>(this);
      RtEvent precondition = 
        Runtime::acquire_rt_reservation(op_lock, true/*exclusive*/);
      return runtime->issue_runtime_meta_task(&decrement_args, 
          sizeof(decrement_args), HLR_DECREMENT_PENDING_TASK_ID, 
          HLR_RESOURCE_PRIORITY, child, precondition);
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent to_trigger;
      // We already hold the lock from the dispatch site (see above)
#ifdef DEBUG_LEGION
      assert(pending_subtasks > 0);
#endif
      if ((outstanding_subtasks > 0) &&
          (pending_subtasks == context_configuration.min_tasks_to_schedule))
      {
        wait_on = context_order_event;
        to_trigger = Runtime::create_rt_user_event();
        context_order_event = to_trigger;
      }
      pending_subtasks--;
      // Release the lock before doing the trigger or the wait
      op_lock.release();
      // Do anything that we need to do
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
        pending_frames++;
        if ((outstanding_subtasks > 0) &&
            (pending_frames == context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->deactivate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(pending_frames > 0);
#endif
        if ((outstanding_subtasks > 0) &&
            (pending_frames == context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
        }
        pending_frames--;
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        Runtime::trigger_event(to_trigger);
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
          LocalFieldInfo(handle, fid, field_size, 
            Runtime::protect_event(completion_event), serdez_id));
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_local_fields(FieldSpace handle,
                                      const std::vector<FieldID> &fields,
                                      const std::vector<size_t> &field_sizes,
                                      CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
                                       info.fid, info.serdez_id, true/*local*/))
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
            HLR_LATENCY_PRIORITY, this, info.reclaim_event);
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
      created_requirements.push_back(new_req);
      // Make a new unmapped physical region
      physical_regions.push_back(PhysicalRegion(
            legion_new<PhysicalRegionImpl>(created_requirements.back(), 
              ApEvent::NO_AP_EVENT, false/*mapped*/, this, map_id, tag, 
              is_leaf(), runtime)));
      RemoteTask *outermost = find_outermost_context();
      outermost->add_top_region(handle);
      // Log this requirement for legion spy if necessary
      if (Runtime::legion_spy_enabled)
        log_created_requirement(created_requirements.size() - 1);
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_created_field(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      std::set<LogicalRegion> top_regions;
      runtime->forest->get_all_regions(handle, top_regions);
      RemoteTask *outermost = find_outermost_context();
      for (std::set<LogicalRegion>::const_iterator it = top_regions.begin();
            it != top_regions.end(); it++)
      {
        RegionRequirement new_req(*it, READ_WRITE, EXCLUSIVE, *it);
        new_req.privilege_fields.insert(fid);
        created_requirements.push_back(new_req);
        physical_regions.push_back(PhysicalRegion(
              legion_new<PhysicalRegionImpl>(created_requirements.back(), 
                ApEvent::NO_AP_EVENT, false/*mapped*/, this, map_id, tag, 
                is_leaf(), runtime)));
        outermost->add_top_region(*it);
        if (Runtime::legion_spy_enabled)
          log_created_requirement(created_requirements.size() - 1);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::log_created_requirement(unsigned index)
    //--------------------------------------------------------------------------
    {
      log_requirement(unique_op_id, regions.size() + index, 
                      created_requirements[index]);
      std::vector<MappingInstance> instances(1, 
          Mapping::PhysicalInstance::get_virtual_instance());
      RegionTreeID bad_tree; std::vector<FieldID> missing_fields;
      std::vector<PhysicalManager*> unacquired;
      InstanceSet instance_set;
      runtime->forest->physical_convert_mapping(this, 
          created_requirements[index], instances, instance_set, bad_tree, 
          missing_fields, NULL, unacquired, false/*do acquire_checks*/);
      runtime->forest->log_mapping_decision(unique_op_id,
          regions.size() + index, created_requirements[index], instance_set);
    }

    //--------------------------------------------------------------------------
    void SingleTask::get_top_regions(
                         std::map<LogicalRegion,RegionTreeContext> &top_regions)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext outermost = find_outermost_context()->get_context();
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->handle_type == SINGULAR);
#endif
        if (top_regions.find(it->region) != top_regions.end())
          continue;
        top_regions[it->region] = context;
      }
      // Need to hold the lock when getting the top regions because
      // the add_created_region method can be executing in parallel
      // and may result in the vector changing size
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
           created_requirements.begin(); it != created_requirements.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->handle_type == SINGULAR);
#endif
        if (top_regions.find(it->region) != top_regions.end())
          continue;
        // If it is something that we made, then the context is
        // the outermost context
        top_regions[it->region] = outermost;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_index_space(IndexSpace handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      // Iterate through our region requirements and find the
      // ones we interfere with
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.index_space.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(handle, it->region.index_space))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_index_path(it->region.index_space, 
                                                handle, dummy_path))
          req.region = LogicalRegion(it->region.get_tree_id(), handle, 
                                     it->region.get_field_space());
        else
          req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
      // Now do the same thing for the created requirements
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.index_space.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(handle, it->region.index_space))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_index_path(it->region.index_space,
                                                handle, dummy_path))
          req.region = LogicalRegion(it->region.get_tree_id(), handle, 
                                     it->region.get_field_space());
        else
          req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_index_partition(IndexPartition handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      // Iterate through our region requirements and find the
      // ones we interfere with
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.index_space.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(it->region.index_space, handle))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_partition_path(it->region.index_space,
                                                    handle, dummy_path))
        {
          req.partition = LogicalPartition(it->region.get_tree_id(), handle,
                                           it->region.get_field_space());
          req.handle_type = PART_PROJECTION;
        }
        else
        {
          req.region = it->region;
          req.handle_type = SINGULAR;
        }
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        parent_req_indexes.push_back(parent_index);
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.index_space.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(it->region.index_space, handle))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_partition_path(it->region.index_space,
                                                    handle, dummy_path))
        {
          req.partition = LogicalPartition(it->region.get_tree_id(), handle,
                                           it->region.get_field_space());
          req.handle_type = PART_PROJECTION;
        }
        else
        {
          req.region = it->region;
          req.handle_type = SINGULAR;
        }
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        parent_req_indexes.push_back(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_field_space(FieldSpace handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        if (it->region.get_field_space() != handle)
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        if (it->region.get_field_space() != handle)
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_fields(FieldSpace handle,
                                            const std::set<FieldID> &to_delete,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        if (it->region.get_field_space() != handle)
          continue;
        std::set<FieldID> overlapping_fields;
        for (std::set<FieldID>::const_iterator fit = to_delete.begin();
              fit != to_delete.end(); fit++)
        {
          if (it->privilege_fields.find(*fit) != it->privilege_fields.end())
            overlapping_fields.insert(*fit);
        }
        if (overlapping_fields.empty())
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = overlapping_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        if (it->region.get_field_space() != handle)
          continue;
        std::set<FieldID> overlapping_fields;
        for (std::set<FieldID>::const_iterator fit = to_delete.begin();
              fit != to_delete.end(); fit++)
        {
          if (it->privilege_fields.find(*fit) != it->privilege_fields.end())
            overlapping_fields.insert(*fit);
        }
        if (overlapping_fields.empty())
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = overlapping_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_logical_region(LogicalRegion handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(handle.get_index_space(), 
                                          it->region.index_space))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_index_path(it->region.index_space,
                                  handle.get_index_space(), dummy_path))
          req.region = handle;
        else
          req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(handle.get_index_space(), 
                                          it->region.index_space))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_index_path(it->region.index_space,
                                  handle.get_index_space(), dummy_path))
          req.region = handle;
        else
          req.region = it->region;
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        req.handle_type = SINGULAR;
        parent_req_indexes.push_back(parent_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::analyze_destroy_logical_partition(LogicalPartition handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(context.exists());
#endif
      unsigned parent_index = 0;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(it->region.index_space,
                                          handle.get_index_partition())) 
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_partition_path(it->region.index_space,
                                  handle.get_index_partition(), dummy_path))
        {
          req.partition = handle;
          req.handle_type = PART_PROJECTION;
        }
        else
        {
          req.region = it->region;
          req.handle_type = SINGULAR;
        }
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        parent_req_indexes.push_back(parent_index);
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++)
      {
        // Different index space trees means we can skip
        if (handle.get_tree_id() != it->region.get_tree_id())
          continue;
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(it->region.index_space,
                                          handle.get_index_partition())) 
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<ColorPoint> dummy_path;
        // See if we dominate the deleted instance
        if (runtime->forest->compute_partition_path(it->region.index_space,
                                  handle.get_index_partition(), dummy_path))
        {
          req.partition = handle;
          req.handle_type = PART_PROJECTION;
        }
        else
        {
          req.region = it->region;
          req.handle_type = SINGULAR;
        }
        req.parent = it->region;
        req.privilege = READ_WRITE;
        req.prop = EXCLUSIVE;
        req.privilege_fields = it->privilege_fields;
        parent_req_indexes.push_back(parent_index);
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
      DETAILED_PROFILER(runtime, HAS_CONFLICTING_INTERNAL_CALL);
      parent_conflict = false;
      inline_conflict = false;
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data 
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = 
          physical_regions[our_idx].impl->get_requirement();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data 
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // Skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = 
          physical_regions[our_idx].impl->get_requirement();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data 
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = 
          physical_regions[our_idx].impl->get_requirement();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = acquire->get_requirement();
      find_conflicting_internal(req, conflicting); 
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(ReleaseOp *release,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = release->get_requirement();
      find_conflicting_internal(req, conflicting);      
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_regions(DependentPartitionOp *partition,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = partition->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_conflicting_internal(const RegionRequirement &req,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      // No need to hold our lock here because we are the only ones who
      // could possibly be doing any mutating of the physical_regions data 
      // structure but we are here so we aren't mutating
      for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
      {
        // skip any regions which are not mapped
        if (!physical_regions[our_idx].impl->is_mapped())
          continue;
        const RegionRequirement &our_req = 
          physical_regions[our_idx].impl->get_requirement();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
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
      DETAILED_PROFILER(runtime, CHECK_REGION_DEPENDENCE_CALL);
      if ((req.handle_type == SINGULAR) || 
          (req.handle_type == REG_PROJECTION))
      {
        // If the trees are different we're done 
        if (our_tid != req.region.get_tree_id())
          return false;
        // Check to see if there is a path between
        // the index spaces
        std::vector<ColorPoint> path;
        if (!runtime->forest->compute_index_path(our_space,
                         req.region.get_index_space(),path))
          return false;
      }
      else
      {
        // Check if the trees are different
        if (our_tid != req.partition.get_tree_id())
          return false;
        std::vector<ColorPoint> path;
        if (!runtime->forest->compute_partition_path(our_space,
                     req.partition.get_index_partition(), path))
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
#ifdef DEBUG_LEGION
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx].impl->is_mapped();
    }

    //--------------------------------------------------------------------------
    void SingleTask::clone_requirement(unsigned idx, RegionRequirement &target)
    //--------------------------------------------------------------------------
    {
      if (idx >= regions.size())
      {
        idx -= regions.size();
        AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        assert(idx < created_requirements.size());
#endif
        target = created_requirements[idx];
      }
      else
        target = regions[idx];
    }

    //--------------------------------------------------------------------------
    int SingleTask::find_parent_region_req(const RegionRequirement &req,
                                           bool check_privilege /*= true*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_PARENT_REGION_REQ_CALL);
      // We can check most of our region requirements without the lock
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &our_req = regions[idx];
        // First check that the regions match
        if (our_req.region != req.parent)
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((req.privilege & our_req.privilege) != req.privilege))
          continue;
        // Finally check that all the fields are contained
        bool dominated = true;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it !=
              req.privilege_fields.end(); it++)
        {
          if (our_req.privilege_fields.find(*it) ==
              our_req.privilege_fields.end())
          {
            dominated = false;
            break;
          }
        }
        if (!dominated)
          continue;
        return int(idx);
      }
      // The created region requirements have to be checked while holding
      // the lock since they are subject to mutation by the application
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        const RegionRequirement &our_req = created_requirements[idx];
        // First check that the regions match
        if (our_req.region != req.parent)
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((req.privilege & our_req.privilege) != req.privilege))
          continue;
        // Finally check that all the fields are contained
        bool dominated = true;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it !=
              req.privilege_fields.end(); it++)
        {
          if (our_req.privilege_fields.find(*it) ==
              our_req.privilege_fields.end())
          {
            dominated = false;
            break;
          }
        }
        if (!dominated)
          continue;
        // Include the offset by the number of base requirements
        return int(regions.size() + idx);
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    unsigned SingleTask::find_parent_region(unsigned index, TaskOp *child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_PARENT_REGION_CALL);
      // We can check these without the lock
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region == child->regions[index].parent)
          return idx;
      }
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        if (created_requirements[idx].region == child->regions[index].parent)
          return (regions.size() + idx);
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
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_BAD_PARENT_REGION);
      return 0;
    }

    //--------------------------------------------------------------------------
    unsigned SingleTask::find_parent_index_region(unsigned index, TaskOp *child)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        if ((indexes[idx].handle == child->indexes[idx].parent))
          return idx;
      }
      log_index.error("Parent task %s (ID %lld) of inline task %s "
                            "(ID %lld) does not have an index space "
                            "requirement for index space %x "
                            "as a parent of chlid task's index requirement "
                            "index %d", get_task_name(), get_unique_id(),
                            child->get_task_name(), child->get_unique_id(),
                            child->indexes[index].handle.id, index);
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_BAD_PARENT_INDEX);
      return 0;
    }

    //--------------------------------------------------------------------------
    PrivilegeMode SingleTask::find_parent_privilege_mode(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx < regions.size())
        return regions[idx].privilege;
      idx -= regions.size();
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(idx < created_requirements.size());
#endif
      return created_requirements[idx].privilege;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(
                                        const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
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
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
      if (req.flags & VERIFIED_FLAG)
        return NO_ERROR;
      std::set<FieldID> checking_fields = req.privilege_fields;
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
#ifdef DEBUG_LEGION
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
      if (idx < regions.size())
        return context;
      else
        return find_outermost_context()->get_context();
    } 

    //--------------------------------------------------------------------------
    SingleTask* SingleTask::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      return parent_ctx;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TRIGGER_SINGLE_CALL);
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
#ifdef DEBUG_LEGION
            assert(target_proc.exists());
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
#ifdef DEBUG_LEGION
#ifndef NDEBUG
                bool still_local = 
#endif
#endif
                distribute_task();
#ifdef DEBUG_LEGION
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
                                               MustEpochOp *must_epoch_owner,
                                const std::vector<RegionTreeContext> &enclosing,
                                      std::vector<InstanceSet> &valid)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INITIALIZE_MAP_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(enclosing.size() == regions.size());
#endif
      // Do the traversals for all the non-early mapped regions and find
      // their valid instances, then fill in the mapper input structure
      valid.resize(regions.size());
      input.valid_instances.resize(regions.size());
      output.chosen_instances.resize(regions.size());
      // If we have must epoch owner, we have to check for any 
      // constrained mappings which must be heeded
      if (must_epoch_owner != NULL)
        must_epoch_owner->must_epoch_map_task_callback(this, input, output);
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any early mapped regions
        std::map<unsigned,InstanceSet>::const_iterator early_mapped_finder = 
          early_mapped_regions.find(idx);
        if (early_mapped_finder != early_mapped_regions.end())
        {
          input.premapped_regions.push_back(idx);
          // Still fill in the valid regions so that mappers can use
          // the instance names for constraints
          prepare_for_mapping(early_mapped_finder->second, 
                              input.valid_instances[idx]);
          // We can also copy them over to the output too
          output.chosen_instances[idx] = input.valid_instances[idx];
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
          continue;
        // Always have to do the traversal at this point to mark open children
        InstanceSet &current_valid = valid[idx];
        perform_physical_traversal(idx, enclosing[idx], current_valid);
        // See if we've already got an output from a must-epoch mapping
        if (!output.chosen_instances[idx].empty())
        {
#ifdef DEBUG_LEGION
          assert(must_epoch_owner != NULL);
#endif
          // We can skip this since we already know the result
          continue;
        }
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
                                              MustEpochOp *must_epoch_owner,
                                const std::vector<RegionTreeContext> &enclosing,
                                      std::vector<InstanceSet> &valid)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FINALIZE_MAP_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(enclosing.size() == regions.size());
#endif
      // first check the processors to make sure they are all on the
      // same node and of the same kind, if we know we have a must epoch
      // owner then we also know there is only one valid choice
      if (must_epoch_owner == NULL)
      {
        if (output.target_procs.empty())
        {
          log_run.warning("Empty output target_procs from call to 'map_task' "
                          "by mapper %s for task %s (ID %lld). Adding the "
                          "'target_proc' " IDFMT " as the default.",
                          mapper->get_mapper_name(), get_task_name(),
                          get_unique_id(), this->target_proc.id);
          output.target_procs.push_back(this->target_proc);
        }
        else if (Runtime::separate_runtime_instances && 
                  (output.target_procs.size() > 1))
        {
          // Ignore additional processors in separate runtime instances
          output.target_procs.resize(1);
        }
        if (!Runtime::unsafe_mapper)
          validate_target_processors(output.target_procs);
        // Special case for when we run in hl:separate mode
        if (Runtime::separate_runtime_instances)
        {
          target_processors.resize(1);
          target_processors[0] = this->target_proc;
        }
        else // the common case
          target_processors = output.target_procs;
      }
      else
      {
        if (output.target_procs.size() > 1)
        {
          log_run.warning("Ignoring suprious additional target processors "
                          "requested in 'map_task' for task %s (ID %lld) "
                          "by mapper %s because task is part of a must "
                          "epoch launch.", get_task_name(), get_unique_id(),
                          mapper->get_mapper_name());
        }
        if (!output.target_procs.empty() && 
                 (output.target_procs[0] != this->target_proc))
        {
          log_run.warning("Ignoring processor request of " IDFMT " for "
                          "task %s (ID %lld) by mapper %s because task "
                          "has already been mapped to processor " IDFMT
                          " as part of a must epoch launch.", 
                          output.target_procs[0].id, get_task_name(), 
                          get_unique_id(), mapper->get_mapper_name(),
                          this->target_proc.id);
        }
        // Only one valid choice in this case, ignore everything else
        target_processors.push_back(this->target_proc);
      }
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
        runtime->find_visible_memories(target_proc, visible_memories);
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
                                "map_task", mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id());
                else
                  log_run.error("Invalid mapper output from invocation of '%s' "
                                "on mapper %s. Mapper selected processor(s) " 
                                "for which premapped instance of region "
                                "requirement %d in memory " IDFMT " is not "
                                "visible for task %s (ID %lld).", 
                                "map_task", mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
          if (Runtime::legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, idx,
                                                  regions[idx],
                                                  physical_instances[idx]);
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (no_access_regions[idx])
          continue;
        // Do the conversion
        InstanceSet &result = physical_instances[idx];
        RegionTreeID bad_tree = 0;
        std::vector<FieldID> missing_fields;
        std::vector<PhysicalManager*> unacquired;
        bool free_acquired = false;
        std::map<PhysicalManager*,std::pair<unsigned,bool> > *acquired = NULL;
        // Get the acquired instances only if we are checking
        if (!Runtime::unsafe_mapper)
        {
          if (this->must_epoch != NULL)
          {
            acquired = new std::map<PhysicalManager*,
                     std::pair<unsigned,bool> >(*get_acquired_instances_ref());
            free_acquired = true;
            // Merge the must epoch owners acquired instances too 
            // if we need to check for all our instances being acquired
            std::map<PhysicalManager*,std::pair<unsigned,bool> > 
              *epoch_acquired = this->must_epoch->get_acquired_instances_ref();
            if (epoch_acquired != NULL)
              acquired->insert(epoch_acquired->begin(), epoch_acquired->end());
          }
          else
            acquired = get_acquired_instances_ref();
        }
        int composite_idx = 
          runtime->forest->physical_convert_mapping(this, regions[idx],
                output.chosen_instances[idx], result, bad_tree, missing_fields,
                acquired, unacquired, !Runtime::unsafe_mapper);
        if (free_acquired)
          delete acquired;
        if (bad_tree > 0)
        {
          log_run.error("Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper specified an instance from region "
                        "tree %d for use with region requirement %d of task "
                        "%s (ID %lld) whose region is from tree %d.",
                        "map_task", mapper->get_mapper_name(), bad_tree,
                        idx, get_task_name(), get_unique_id(),
                        regions[idx].region.get_tree_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        if (!missing_fields.empty())
        {
          log_run.error("Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper failed to specify an instance for "
                        "%ld fields of region requirement %d on task %s "
                        "(ID %lld). The missing fields are listed below.",
                        "map_task", mapper->get_mapper_name(), 
                        missing_fields.size(), idx, get_task_name(), 
                        get_unique_id());
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
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,std::pair<unsigned,bool> > 
            *acquired_instances = get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator it = 
                unacquired.begin(); it != unacquired.end(); it++)
          {
            if (acquired_instances->find(*it) == acquired_instances->end())
            {
              log_run.error("Invalid mapper output from 'map_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_MAPPER_OUTPUT);
            }
          }
          // Event if we did successfully acquire them, still issue the warning
          log_run.warning("WARNING: mapper %s failed to acquire instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'map_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
        }
        // See if they want a virtual mapping
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
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_ILLEGAL_REDUCTION_VIRTUAL_MAPPING);
          }
          virtual_mapped[idx] = true;
        } 
        if (Runtime::legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, idx,
                                                regions[idx],
                                                physical_instances[idx]);
        // Skip checks if the mapper promises it is safe
        if (Runtime::unsafe_mapper)
          continue;
        // If this is anything other than a virtual mapping, check that
        // the instances align with the privileges
        if (!virtual_mapped[idx])
        {
          std::vector<LogicalRegion> regions_to_check(1, regions[idx].region);
          for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
          {
            if (!result[idx2].get_manager()->meets_regions(regions_to_check))
            {
              // Doesn't satisfy the region requirement
              log_run.error("Invalid mapper output from invocation of '%s' on "
                            "mapper %s. Mapper specified instance that does "
                            "not meet region requirement %d for task %s "
                            "(ID %lld). The index space for the instance has "
                            "insufficient space for the requested logical "
                            "region.", "map_task", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
                log_run.error("Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected an instance for "
                              "region requirement %d in memory " IDFMT " "
                              "which is not visible from the target processors "
                              "for task %s (ID %lld).", "map_task", 
                              mapper->get_mapper_name(), idx, mem.id, 
                              get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
          // If this is a reduction region requirement make sure all the 
          // managers are reduction instances
          if (IS_REDUCE(regions[idx]))
          {
            std::map<PhysicalManager*,std::pair<unsigned,bool> > 
              *acquired = get_acquired_instances_ref();
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              if (!result[idx2].get_manager()->is_reduction_manager())
              {
                log_run.error("Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper failed to choose a "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which has "
                              "reduction privileges.", "map_task", 
                              mapper->get_mapper_name(), idx,
                              get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
              std::map<PhysicalManager*,std::pair<unsigned,bool> >::
                const_iterator finder = acquired->find(
                    result[idx2].get_manager());
#ifdef DEBUG_LEGION
              assert(finder != acquired->end());
#endif
              if (!finder->second.second)
              {
                log_run.error("Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper made an illegal decision "
                              "to re-use a reduction instance for region "
                              "requirement %d of task %s (ID %lld). Reduction "
                              "instances are not currently permitted to be "
                              "recycled.", "map_task",mapper->get_mapper_name(),
                              idx, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
            }
          }
          else
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              if (!result[idx2].get_manager()->is_instance_manager())
              {
                log_run.error("Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected illegal "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which "
                              "does not have reduction privileges.", "map_task",
                              mapper->get_mapper_name(), idx, 
                              get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
                assert(false);
#endif
                exit(ERROR_INVALID_MAPPER_OUTPUT);
              }
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
        log_run.error("Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper specified an invalid task variant "
                      "of ID 0 for task %s (ID %lld), but Legion does not yet "
                      "support task generators.", "map_task", 
                      mapper->get_mapper_name(), 
                      get_task_name(), get_unique_id());
        // TODO: invoke a generator if one exists
#ifdef DEBUG_LEGION
        assert(false); 
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      if (variant_impl == NULL)
      {
        // If we couldn't find or make a variant that is bad
        log_run.error("Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper failed to specify a valid "
                      "task variant or generator capable of create a variant "
                      "implementation of task %s (ID %lld).",
                      "map_task", mapper->get_mapper_name(), get_task_name(),
                      get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      // Now that we know which variant to use, we can validate it
      if (!Runtime::unsafe_mapper)
        validate_variant_selection(mapper, variant_impl, "map_task"); 
      // Record anything else that needs to be recorded 
      selected_variant = output.chosen_variant;
      task_priority = output.task_priority;
      perform_postmap = output.postmap_task;
    }

    //--------------------------------------------------------------------------
    void SingleTask::validate_target_processors(
                                 const std::vector<Processor> &processors) const
    //--------------------------------------------------------------------------
    {
      // Make sure that they are all on the same node and of the same kind
      Processor::Kind kind = this->target_proc.kind();
      AddressSpace space = this->target_proc.address_space();
      for (unsigned idx = 0; idx < processors.size(); idx++)
      {
        const Processor &proc = processors[idx];
        if (proc.kind() != kind)
        {
          log_run.error("Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is of kind %s when mapping task %s "
                        "(ID %lld), but the target processor " IDFMT " has "
                        "kind %s. Only one kind of processor is permitted.",
                        mapper->get_mapper_name(), proc.id, 
                        Processor::get_kind_name(proc.kind()), get_task_name(),
                        get_unique_id(), this->target_proc.id, 
                        Processor::get_kind_name(kind));
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        if (proc.address_space() != space)
        {
          log_run.error("Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is in address space %d when mapping "
                        "task %s (ID %lld) but the target processor " IDFMT 
                        "is in address space %d. All target processors must "
                        "be in the same address space.", 
                        mapper->get_mapper_name(), proc.id,
                        proc.address_space(), get_task_name(), get_unique_id(), 
                        this->target_proc.id, space);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::validate_variant_selection(MapperManager *local_mapper,
                          VariantImpl *impl, const char *mapper_call_name) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, VALIDATE_VARIANT_SELECTION_CALL);
      // Check the layout constraints first
      const TaskLayoutConstraintSet &layout_constraints = 
        impl->get_layout_constraints();
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.begin(); it != 
            layout_constraints.layouts.end(); it++)
      {
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(it->second);
        const InstanceSet &instances = physical_instances[it->first]; 
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          PhysicalManager *manager = instances[idx].get_manager();
          if (manager->conflicts(constraints))
          {
            log_run.error("Invalid mapper output. Mapper %s selected variant "
                          "%ld for task %s (ID %lld). But instance selected "
                          "for region requirement %d fails to satisfy the "
                          "corresponding constraints.", 
                          local_mapper->get_mapper_name(), impl->vid,
                          get_task_name(), get_unique_id(), it->first);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_MAPPER_OUTPUT);
          }
        }
      }
      // Now we can test against the execution constraints
      const ExecutionConstraintSet &execution_constraints = 
        impl->get_execution_constraints();
      // TODO: Check ISA, resource, and launch constraints
      // First check the processor constraint
      if (execution_constraints.processor_constraint.is_valid() &&
          (execution_constraints.processor_constraint.get_kind() != 
           this->target_proc.kind()))
      {
        log_run.error("Invalid mapper output. Mapper %s selected variant %ld "
                      "for task %s (ID %lld). However, this variant has a "
                      "processor constraint for processors of kind %s, but "
                      "the target processor " IDFMT " is of kind %s.",
                      local_mapper->get_mapper_name(),impl->vid,get_task_name(),
                      get_unique_id(), Processor::get_kind_name(
                        execution_constraints.processor_constraint.get_kind()),
                      this->target_proc.id, Processor::get_kind_name(
                        this->target_proc.kind()));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      // Then check the colocation constraints
      for (std::vector<ColocationConstraint>::const_iterator con_it = 
            execution_constraints.colocation_constraints.begin(); con_it !=
            execution_constraints.colocation_constraints.end(); con_it++)
      {
        if (con_it->indexes.size() < 2)
          continue;
        if (con_it->fields.empty())
          continue;
        // First check to make sure that all these region requirements have
        // the same region tree ID.
        bool first = true;
        FieldSpace handle = FieldSpace::NO_SPACE;
        std::vector<InstanceSet*> instances(con_it->indexes.size());
        unsigned idx = 0;
        for (std::set<unsigned>::const_iterator it = con_it->indexes.begin();
              it != con_it->indexes.end(); it++, idx++)
        {
#ifdef DEBUG_LEGION
          assert(regions[*it].handle_type == SINGULAR);
          for (std::set<FieldID>::const_iterator fit = con_it->fields.begin();
                fit != con_it->fields.end(); fit++)
          {
            if (regions[*it].privilege_fields.find(*fit) ==
                regions[*it].privilege_fields.end())
            {
              log_run.error("Invalid location constraint. Location constraint "
                            "specifies field %d which is not included in "
                            "region requirement %d of task %s (ID %lld).",
                            *fit, *it, get_task_name(), get_unique_id());
              assert(false);
            }
          }
#endif
          if (first)
          {
            handle = regions[*it].region.get_field_space();
            first = false;
          }
          else
          {
            if (regions[*it].region.get_field_space() != handle)
            {
              log_run.error("Invalid mapper output. Mapper %s selected variant "
                            "%ld for task %s (ID %lld). However, this variant "
                            "has colocation constraints for indexes %d and %d "
                            "which have region requirements with different "
                            "field spaces which is illegal.",
                            local_mapper->get_mapper_name(), impl->vid, 
                            get_task_name(), get_unique_id(), 
                            *(con_it->indexes.begin()), *it);
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_MAPPER_OUTPUT);
            }
          }
          instances[idx] = const_cast<InstanceSet*>(&physical_instances[*it]);
        }
        // Now do the test for colocation
        unsigned bad1 = 0, bad2 = 0; 
        if (!runtime->forest->are_colocated(instances, handle, 
                                            con_it->fields, bad1, bad2))
        {
          // Used for translating the indexes back from their linearized form
          std::vector<unsigned> lin_indexes(con_it->indexes.begin(),
                                            con_it->indexes.end());
          log_run.error("Invalid mapper output. Mapper %s selected variant "
                        "%ld for task %s (ID %lld). However, this variant "
                        "requires that region requirements %d and %d be "
                        "co-located for some set of field, but they are not.",
                        local_mapper->get_mapper_name(), impl->vid, 
                        get_task_name(), get_unique_id(), lin_indexes[bad1],
                        lin_indexes[bad2]);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invoke_mapper(MustEpochOp *must_epoch_owner,
                       const std::vector<RegionTreeContext> &enclosing_contexts)
    //--------------------------------------------------------------------------
    {
      Mapper::MapTaskInput input;
      Mapper::MapTaskOutput output;
      // Initialize the mapping input which also does all the traversal
      // down to the target nodes
      std::vector<InstanceSet> valid_instances(regions.size());
      initialize_map_task_input(input, output, must_epoch_owner, 
                                enclosing_contexts, valid_instances);
      // Now we can invoke the mapper to do the mapping
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_map_task(this, &input, &output);
      // Now we can convert the mapper output into our physical instances
      finalize_map_task_output(input, output, must_epoch_owner, 
                               enclosing_contexts, valid_instances);
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(ApEvent local_termination_event,
                                     MustEpochOp *must_epoch_op /*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MAP_ALL_REGIONS_CALL);
#ifdef LEGION_SPY
      {
        ApEvent local_completion = get_completion_event();
        // Yes, these events actually trigger in the opposite order, but
        // it is the logical entailement that is important here
        if (local_completion != local_termination_event)
          LegionSpy::log_event_dependence(local_completion, 
                                          local_termination_event);
      }
#endif
      std::vector<RegionTreeContext> enclosing_contexts(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        enclosing_contexts[idx] = get_parent_context(idx);
      // Now do the mapping call
      invoke_mapper(must_epoch_op, enclosing_contexts);
      // After we've got our results, apply the state to the region tree
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
          continue;
        if (no_access_regions[idx])
          continue;
        // See if we have to do any virtual mapping before registering
        if (virtual_mapped[idx])
        {
#ifdef DEBUG_LEGION
          assert(physical_instances[idx].size() == 1);
          assert(physical_instances[idx][0].get_manager()->
                                      is_virtual_instance());
#endif
          // We choose different target contexts depending on whether
          // we are locally mapping and are remote or not. If we're 
          // locally mapping and being sent to a remote processor then
          // we will use our parent context for now and translate later
          // on our destination node. Otherwise we can just use ourself
          SingleTask *target_ctx = this;
          if (is_locally_mapped() && !runtime->is_local(target_proc))
            target_ctx = parent_ctx;
          runtime->forest->map_virtual_region(enclosing_contexts[idx],
                                              regions[idx],
                                              physical_instances[idx][0],
                                              get_version_info(idx),
                                              target_ctx, this,
                                              false/*needs fields*/
#ifdef DEBUG_LEGION
                                              , idx, get_logging_name()
                                              , unique_op_id
#endif
                                              );
          // No need to register this instance in the context
          // because it doesn't represent a change of state
          continue;
        }
        // Set the current mapping index before doing anything
        // that sould result in a copy
        set_current_mapping_index(idx);
        // apply the results of the mapping to the tree
        runtime->forest->physical_register_only(enclosing_contexts[idx],
                                    regions[idx], get_version_info(idx), 
                                    this, idx, local_termination_event, 
                                    (regions.size() > 1)/*defer add users*/,
                                    map_applied_conditions,
                                    physical_instances[idx]
#ifdef DEBUG_LEGION
                                    , get_logging_name()
                                    , unique_op_id
#endif
                                    );
      }
      // If we had more than one region requirement when now have to
      // record our users because we skipped that during traversal
      if (regions.size() > 1)
      {
        // C++ type system suckiness
        std::deque<InstanceSet> &phy_inst_ref = 
          *(reinterpret_cast<std::deque<InstanceSet>*>(&physical_instances));
        runtime->forest->physical_register_users(this,
            local_termination_event, regions, virtual_mapped, 
            *get_version_infos(), phy_inst_ref, map_applied_conditions);
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
                              this, idx, true/*valid*/, 
                              map_applied_conditions, postmap_valid[idx]
#ifdef DEBUG_LEGION
                              , get_logging_name()
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
        RegionTreeID bad_tree = 0;
        std::vector<PhysicalManager*> unacquired;
        bool had_composite = 
          runtime->forest->physical_convert_postmapping(this, req,
                              output.chosen_instances[idx], result, bad_tree,
                              Runtime::unsafe_mapper ? NULL : 
                                get_acquired_instances_ref(),
                              unacquired, !Runtime::unsafe_mapper);
        if (bad_tree > 0)
        {
          log_run.error("Invalid mapper output from 'postmap_task' invocation "
                        "on mapper %s. Mapper provided an instance from region "
                        "tree %d for use in satisfying region requirement %d "
                        "of task %s (ID %lld) whose region is from region tree "
                        "%d.", mapper->get_mapper_name(), bad_tree, idx,
                        get_task_name(), get_unique_id(), 
                        regions[idx].region.get_tree_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_OUTPUT);
        }
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,std::pair<unsigned,bool> > 
            *acquired_instances = get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator uit = 
                unacquired.begin(); uit != unacquired.end(); uit++)
          {
            if (acquired_instances->find(*uit) == acquired_instances->end())
            {
              log_run.error("Invalid mapper output from 'postmap_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_MAPPER_OUTPUT);
            }
          }
          // If we did successfully acquire them, still issue the warning
          log_run.warning("WARNING: mapper %s failed to acquires instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'postmap_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
        }
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
        if (!Runtime::unsafe_mapper)
        {
          std::vector<LogicalRegion> regions_to_check(1, 
                                        regions[idx].region);
          for (unsigned check_idx = 0; check_idx < result.size(); check_idx++)
          {
            if (!result[check_idx].get_manager()->meets_regions(
                                                      regions_to_check))
            {
              log_run.error("Invalid mapper output from invocation of "
                            "'postmap_task' on mapper %s. Mapper specified an "
                            "instance region requirement %d of task %s "
                            "(ID %lld) that does not meet the logical region "
                            "requirement.", mapper->get_mapper_name(), idx, 
                            get_task_name(), get_unique_id()); 
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_INVALID_MAPPER_OUTPUT);
            }
          }
        }
        // Register this with a no-event so that the instance can
        // be used as soon as it is valid from the copy to it
        runtime->forest->physical_register_only(enclosing_contexts[idx],
                          regions[idx], get_version_info(idx), this, idx,
                          ApEvent::NO_AP_EVENT/*done immediately*/, 
                          true/*defer add users*/, 
                          map_applied_conditions, result
#ifdef DEBUG_LEGION
                          , get_logging_name(), unique_op_id
#endif
                          );
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_region_tree_contexts(
                      const std::vector<RegionRequirement> &clone_requirements,
                      const std::vector<ApUserEvent> &unmap_events,
                      std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INITIALIZE_REGION_TREE_CONTEXTS_CALL);
#ifdef DEBUG_LEGION
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == no_access_regions.size());
#endif
      // Initialize all of the logical contexts no matter what
      //
      // For all of the physical contexts that were mapped, initialize them
      // with a specified reference to the current instance, otherwise
      // they were a virtual reference and we can ignore it.
      std::map<PhysicalManager*,InstanceView*> top_views;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_LEGION
        // this better be true for single tasks
        assert(regions[idx].handle_type == SINGULAR);
#endif
        // If this is a NO_ACCESS or had no privilege fields we can skip this
        if (no_access_regions[idx])
          continue;
        // Only need to initialize the context if this is
        // not a leaf and it wasn't virtual mapped
        if (!virtual_mapped[idx])
        {
          runtime->forest->initialize_current_context(context,
              clone_requirements[idx], physical_instances[idx],
              unmap_events[idx], this, idx, top_views);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
          assert(physical_instances[idx].has_composite_ref());
#endif
          const InstanceRef &ref = physical_instances[idx].get_composite_ref();
          CompositeView *composite_view = ref.get_composite_view();
          // First get any events necessary to make this view local
          // If we locally mapped and are now remote, we need to translate
          // this composite instance so that its views are specific to 
          // our context
          if (is_locally_mapped() && is_remote())
          {
            CompositeCloser closer(context.get_id(),get_version_info(idx),this);
            DeferredView *translated_view = 
              composite_view->simplify(closer, ref.get_valid_fields());
#ifdef DEBUG_LEGION
            assert(translated_view->is_composite_view());
#endif
            composite_view = translated_view->as_composite_view();
          }
          runtime->forest->initialize_current_context(context,
              clone_requirements[idx], physical_instances[idx], 
              this, idx, composite_view);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INVALIDATE_REGION_TREE_CONTEXTS_CALL);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->invalidate_current_context(context,
                                                    regions[idx].region,
                                                    false/*logical only*/);
      }
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        runtime->forest->invalidate_current_context(context,
                  created_requirements[idx].region, true/*logical only*/);
      }
    }

    //--------------------------------------------------------------------------
    InstanceView* SingleTask::create_instance_top_view(PhysicalManager *manager,
                                                  AddressSpaceID request_source)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CREATE_INSTANCE_TOP_VIEW_CALL);
      // First check to see if we are the owner node for this manager
      // if not we have to send the message there since the context
      // on that node is actually the point of serialization
      if (!manager->is_owner())
      {
        InstanceView *volatile result = NULL;
        RtUserEvent wait_on = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize<UniqueID>(get_context_uid());
          rez.serialize(manager->did);
          rez.serialize<InstanceView**>(const_cast<InstanceView**>(&result));
          rez.serialize(wait_on); 
        }
        runtime->send_create_top_view_request(manager->owner_space, rez);
        wait_on.wait();
#ifdef DEBUG_LEGION
        assert(result != NULL); // when we wake up we should have the result
#endif
        return result;
      }
      // Check to see if we already have the 
      // instance, if we do, return it, otherwise make it and save it
      RtEvent wait_on;
      {
        AutoLock o_lock(op_lock);
        std::map<PhysicalManager*,InstanceView*>::const_iterator finder = 
          instance_top_views.find(manager);
        if (finder != instance_top_views.end())
          // We've already got the view, so we are done
          return finder->second;
        // See if someone else is already making it
        std::map<PhysicalManager*,RtUserEvent>::iterator pending_finder =
          pending_top_views.find(manager);
        if (pending_finder == pending_top_views.end())
          // mark that we are making it
          pending_top_views[manager] = RtUserEvent::NO_RT_USER_EVENT;
        else
        {
          // See if we are the first one to follow
          if (!pending_finder->second.exists())
            pending_finder->second = Runtime::create_rt_user_event();
          wait_on = pending_finder->second;
        }
      }
      if (wait_on.exists())
      {
        // Someone else is making it so we just have to wait for it
        wait_on.wait();
        // Retake the lock and read out the result
        AutoLock o_lock(op_lock, 1, false/*exclusive*/);
        std::map<PhysicalManager*,InstanceView*>::const_iterator finder = 
            instance_top_views.find(manager);
#ifdef DEBUG_LEGION
        assert(finder != instance_top_views.end());
#endif
        return finder->second;
      }
      InstanceView *result = 
        manager->create_instance_top_view(this, request_source);
      result->add_base_resource_ref(CONTEXT_REF);
      // We've got the results, if we have any virtual mappings we have
      // to see if this instance can be used to satisfy a virtual mapping
      // We can also skip reduction views because we know they are not
      // permitted to cross context boundaries
      if (has_virtual_instances() && !result->is_reduction_view())
      {
        InstanceView *parent_context_view = NULL;
        const LogicalRegion &handle = manager->region_node->handle;
        // Nasty corner case, see if there are any other region requirements
        // which this instance can also be used for and are also virtual 
        // mapped and therefore we need to do this analysis now before we
        // release the instance as ready for the context
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // If it's not virtual mapped we can keep going
          if (!virtual_mapped[idx] || no_access_regions[idx])
            continue;
          // Different tree IDs then we can keep going
          if (regions[idx].region.get_tree_id() != handle.get_tree_id())
            continue;
          // See if we have a path
          std::vector<ColorPoint> path;
          if (!runtime->forest->compute_index_path(handle.get_index_space(), 
                                regions[idx].region.get_index_space(), path))
            continue;
          // See if we have any overlapping fields
          bool has_overlapping_fields = false;
          for (std::set<FieldID>::const_iterator it = 
                regions[idx].privilege_fields.begin(); it !=
                regions[idx].privilege_fields.end(); it++)
          {
            if (manager->layout->has_field(*it))
            {
              has_overlapping_fields = true;
              break;
            }
          }
          if (!has_overlapping_fields)
            continue;
          // We definitely need to do the analysis, since we are a
          // sub-region and have overlapping fields, get the parent
          // context view if we haven't done so already
          if (parent_context_view == NULL)
          {
            // Note that this recursion ensures that we handle nested
            // virtual mappings correctly
            SingleTask *parent_context = find_parent_context();
            parent_context_view = 
              parent_context->create_instance_top_view(manager, request_source);
          }
          // Now we clone across for the given region requirement
          VersionInfo &info = get_version_info(idx); 
          runtime->forest->convert_views_into_context(regions[idx], this, idx,
                     info, parent_context_view, result, get_task_completion(), 
                     path, map_applied_conditions);
        }
      }
      // Record the result and trigger any user event to signal that the
      // view is ready
      RtUserEvent to_trigger;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(instance_top_views.find(manager) == 
                instance_top_views.end());
#endif
        instance_top_views[manager] = result;
        std::map<PhysicalManager*,RtUserEvent>::iterator pending_finder =
          pending_top_views.find(manager);
#ifdef DEBUG_LEGION
        assert(pending_finder != pending_top_views.end());
#endif
        to_trigger = pending_finder->second;
        pending_top_views.erase(pending_finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::notify_instance_deletion(PhysicalManager *deleted,
                                              GenerationID old_gen)
    //--------------------------------------------------------------------------
    {
      InstanceView *removed = NULL;
      {
        AutoLock o_lock(op_lock);
        // If we are no longer the same generation, then we can ignore this
        if (old_gen < gen)
          return;
        std::map<PhysicalManager*,InstanceView*>::iterator finder =  
          instance_top_views.find(deleted);
#ifdef DEBUG_LEGION
        assert(finder != instance_top_views.end());
#endif
        removed = finder->second;
        instance_top_views.erase(finder);
      }
      if (removed->remove_base_resource_ref(CONTEXT_REF))
        LogicalView::delete_logical_view(removed);
    }

    //--------------------------------------------------------------------------
    void SingleTask::convert_virtual_instance_top_views(
                   const std::map<AddressSpaceID,RemoteTask*> &remote_instances)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CONVERT_VIRTUAL_INSTANCE_TOP_VIEW_CALL);
      // If we have no virtual mapped regions and we have no created
      // region requirements then we are done
      if (created_requirements.empty())
        return;
      // If we have any remote instances we need to send a message to them
      // that they have to do this for themselves before we are mapped
      if (!remote_instances.empty())
      {
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          RtUserEvent remote_done = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->second);
            rez.serialize<size_t>(created_requirements.size());
            for (unsigned idx = 0; idx < created_requirements.size(); idx++)
              pack_region_requirement(created_requirements[idx], rez);
            rez.serialize(remote_done); 
          }
          runtime->send_remote_convert_virtual_instances(it->first, rez);
          map_applied_conditions.insert(remote_done);
        }
      }
      // Now do it for ourself, we have to make a copy because deletions
      // can still come back while we are iterating over our instances
      std::vector<PhysicalManager*> copy_top_views;
      {
        AutoLock o_lock(op_lock,1,false/*exclusive*/);
        for (std::map<PhysicalManager*,InstanceView*>::const_iterator it = 
              instance_top_views.begin(); it != instance_top_views.end(); it++)
        {
          // Try adding a valid reference to make sure it doesn't get
          // collected while we are trying to do this, we're on the owner
          // node so this doesn't need to be valid initially
          if (it->first->try_add_base_valid_ref(CONTEXT_REF, NULL, false))
            copy_top_views.push_back(it->first);
          // If we couldn't add the valid ref that means this instance
          // is definitely going to be collected
        }
      }
      for (std::vector<PhysicalManager*>::const_iterator it = 
            copy_top_views.begin(); it != copy_top_views.end(); it++)
      {
        InstanceView *parent_context_view = NULL;
        const LogicalRegion &handle = (*it)->region_node->handle;
        // If we had any created region requirements we need to see 
        // if we have any interfering users for those requirements
        if (!created_requirements.empty())
        {
          const AddressSpaceID local_space = runtime->address_space;
          for (unsigned idx = 0; idx < created_requirements.size(); idx++)
          {
            if (created_requirements[idx].region.get_tree_id() != 
                 handle.get_tree_id())
              continue;
            // We are guaranteed to have a path since we know a created
            // requirement is always for the top of the region tree
            // See if we have any overlapping fields
            bool has_overlapping_fields = false;
            for (std::set<FieldID>::const_iterator fit = 
                  created_requirements[idx].privilege_fields.begin(); fit !=
                  created_requirements[idx].privilege_fields.end(); fit++)
            {
              if ((*it)->layout->has_field(*fit))
              {
                has_overlapping_fields = true;
                break;
              }
            }
            if (!has_overlapping_fields)
              continue;
            if (parent_context_view == NULL)
            {
              SingleTask *parent_context = find_parent_context();
              parent_context_view = 
                parent_context->create_instance_top_view(*it, local_space);
            }
            // Do the conversion back out
            VersionInfo dummy_info;
            runtime->forest->convert_views_from_context(
                                                      created_requirements[idx],
                                                      this, regions.size()+idx,
                                                      dummy_info,
                                                      parent_context_view,
                                                      get_task_completion(),
                                                      true/*initial user*/,
                                                      map_applied_conditions);
          }
        }
        // Then we can remove our valid reference on the manager
        if ((*it)->remove_base_valid_ref(CONTEXT_REF))
          PhysicalManager::delete_physical_manager(*it); 
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::handle_create_top_view_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      InstanceView **target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      // Get the context first
      SingleTask *context = runtime->find_context(context_uid);
      // Find the manager too, we know we are local so it should already
      // be registered in the set of distributed IDs
      DistributedCollectable *dc = 
        runtime->find_distributed_collectable(manager_did);
#ifdef DEBUG_LEGION
      PhysicalManager *manager = dynamic_cast<PhysicalManager*>(dc);
      assert(manager != NULL);
#else
      PhysicalManager *manager = static_cast<PhysicalManager*>(dc);
#endif
      InstanceView *result = context->create_instance_top_view(manager, source);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(result->did);
        rez.serialize(target);
        rez.serialize(to_trigger);
      }
      runtime->send_create_top_view_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::handle_create_top_view_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID result_did;
      derez.deserialize(result_did);
      InstanceView **target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtEvent ready;
      LogicalView *view = 
        runtime->find_or_request_logical_view(result_did, ready);
      // Have to static cast since it might not be ready
      *target = static_cast<InstanceView*>(view);
      if (ready.exists())
        Runtime::trigger_event(to_trigger, ready);
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, LAUNCH_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == no_access_regions.size());
      assert(physical_regions.empty());
#endif 
      VariantImpl *variant = 
        runtime->find_variant_impl(task_id, selected_variant);
      // STEP 1: Compute the precondition for the task launch
      std::set<ApEvent> wait_on_events;
      // Get the event to wait on unless we are 
      // doing the inner task optimization
      if (!variant->is_inner())
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx] && !no_access_regions[idx])
            physical_instances[idx].update_wait_on_events(wait_on_events);
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
	ApEvent e = 
          Runtime::get_previous_phase(wait_barriers[idx].phase_barrier);
        wait_on_events.insert(e);
      }

      // STEP 2: Set up the task's context
      std::vector<ApUserEvent> unmap_events(regions.size());
      {
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(regions[idx].handle_type == SINGULAR);
#endif
          // Convert any WRITE_ONLY or WRITE_DISCARD privleges to READ_WRITE
          // This is necessary for any sub-operations which may need to rely
          // on our privileges for determining their own privileges such
          // as inline mappings or acquire and release operations
          if (regions[idx].privilege == WRITE_DISCARD)
            regions[idx].privilege = READ_WRITE;
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx] || no_access_regions[idx])
          {
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            physical_regions.push_back(PhysicalRegion(
                  legion_new<PhysicalRegionImpl>(regions[idx],
                    ApEvent::NO_AP_EVENT, false/*mapped*/,
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
                    ApEvent::NO_AP_EVENT, false/*mapped*/,
                    this, map_id, tag, false/*leaf*/, runtime)));
            unmap_events[idx] = Runtime::create_ap_user_event();
            // Trigger the user event when the region is 
            // actually ready to be used
            std::set<ApEvent> ready_events;
            physical_instances[idx].update_wait_on_events(ready_events);
            ApEvent precondition = Runtime::merge_events(ready_events);
            Runtime::trigger_event(unmap_events[idx], precondition);
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
                    ApEvent::NO_AP_EVENT/*already mapped*/, true/*mapped*/,
                    this, map_id, tag, variant->is_leaf(), runtime)));
            // Now set the reference for this physical region 
            // which is pretty much a dummy physical reference except
            // it references the same view as the outer reference
            unmap_events[idx] = Runtime::create_ap_user_event();
            // We reset the reference below after we've
            // initialized the local contexts and received
            // back the local instance references
          }
          // Make sure you have the metadata for the region with no access priv
          if (no_access_regions[idx])
            runtime->forest->get_node(clone_requirements[idx].region);
        }

        // If we're a leaf task and we have virtual mappings
        // then it's possible for the application to do inline
        // mappings which require a physical context
        if (!variant->is_leaf() || has_virtual_instances())
        {
          // Request a context from the runtime
          runtime->allocate_local_context(this);
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
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_CONTEXT_CONFIGURATION);
          }
          // If we're counting by frames set min_tasks_to_schedule to zero
          if (context_configuration.min_frames_to_schedule > 0)
            context_configuration.min_tasks_to_schedule = 0;
          // otherwise we know min_frames_to_schedule is zero
#ifdef DEBUG_LEGION
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
            for (unsigned idx = 0; idx < physical_regions.size(); idx++)
            {
              if (!virtual_mapped[idx] && !no_access_regions[idx])
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
          for (unsigned idx = 0; idx < physical_regions.size(); idx++)
          {
            if (!virtual_mapped[idx] && !no_access_regions[idx])
            {
              physical_regions[idx].impl->reset_references(
                  physical_instances[idx], unmap_events[idx]);
            }
          }
        }
      }
      // Merge together all the events for the start condition 
      ApEvent start_condition = Runtime::merge_events(wait_on_events);
      // Take all the locks in order in the proper way
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          start_condition = Runtime::acquire_ap_reservation(it->first, 
                                          it->second, start_condition);
        }
      }
      // STEP 3: Finally we get to launch the task
      // Mark that we have an outstanding task in this context 
      parent_ctx->increment_pending();
      // If this is a leaf task and we have no virtual instances
      // and the SingleTask sub-type says it is ok
      // we can trigger the task's completion event as soon as
      // the task is done running.  We first need to mark that this
      // is going to occur before actually launching the task to 
      // avoid the race.
      bool perform_chaining_optimization = false; 
      ApUserEvent chain_complete_event;
      if (variant->is_leaf() && !has_virtual_instances() &&
          can_early_complete(chain_complete_event))
        perform_chaining_optimization = true;
      // Note there is a potential scary race condition to be aware of here: 
      // once we launch this task it's possible for this task to run and 
      // clean up before we finish the execution of this function thereby
      // invalidating this SingleTask object's fields.  This means
      // that we need to save any variables we need for after the task
      // launch here on the stack before they can be invalidated.
      ApEvent term_event = get_task_completion();
#ifdef DEBUG_LEGION
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
#ifdef LEGION_SPY
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_operation_events(get_unique_id(), start_condition, 
                                        completion_event);
#endif
      ApEvent task_launch_event = variant->dispatch_task(launch_processor, this,
                            start_condition, task_priority, profiling_requests);
      // Finish the chaining optimization if we're doing it
      if (perform_chaining_optimization)
        Runtime::trigger_event(chain_complete_event, task_launch_event);
      // STEP 4: After we've launched the task, then we have to release any 
      // locks that we took for while the task was running.  
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          Runtime::release_reservation(it->first, term_event);
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
#ifdef DEBUG_LEGION
      log_task.debug("Task %s (ID %lld) starting on processor " IDFMT "",
                    get_task_name(), get_unique_id(), executing_processor.id);
      assert(regions.size() == physical_regions.size());
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == no_access_regions.size());
#endif
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      pending_done = parent_ctx->decrement_pending(this);
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
#ifdef DEBUG_LEGION
      assert(response.user_data_size() == sizeof(MapperProfilingInfo));
#endif
      const MapperProfilingInfo *info = 
        (const MapperProfilingInfo*)response.user_data();
      // Record the results
      info->task->notify_profiling_results(response);
      // Then trigger the event saying we are done
      Runtime::trigger_event(info->profiling_done);
    }

    //--------------------------------------------------------------------------
    void SingleTask::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((regions.size() + 
                created_requirements.size()) == physical_regions.size());
#endif
      // Quick check to make sure the user didn't forget to end a trace
      if (current_trace != NULL)
      {
        log_task.error("Task %s (UID %lld) failed to end trace before exiting!",
                        get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INCOMPLETE_TRACE);
      }
      // Unmap all of the physical regions which are still mapped
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
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
        // Note that this loop doesn't handle create regions
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          // We also don't need to close up read-only instances
          if (no_access_regions[idx] || IS_READ_ONLY(regions[idx]))
            continue;
          if (!virtual_mapped[idx])
          {
            if (!is_leaf())
            {
#ifdef DEBUG_LEGION
              assert(!physical_instances[idx].empty());
#endif
              PostCloseOp *close_op = 
                runtime->get_available_post_close_op(true);
              close_op->initialize(this, idx);
              runtime->add_to_dependence_queue(executing_processor, close_op);
            }
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(physical_instances[idx].has_composite_ref());
#endif
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
        // Give these high priority too since they are cleaning up 
        // and will allow other tasks to run
        runtime->issue_runtime_meta_task(&post_end_args, sizeof(post_end_args),
               HLR_POST_END_ID, HLR_LATENCY_PRIORITY, this, last_registration);
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
        std::set<RtEvent> preconditions;
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
#ifdef DEBUG_LEGION
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
          handle_post_mapped(Runtime::merge_events(preconditions));
        else
          handle_post_mapped();
      }
      else
      {
        // Handle the leaf task case
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
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
        RtEvent exec_precondition = 
          Runtime::merge_events(pending_done, profiling_done);
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
      DETAILED_PROFILER(runtime, ACTIVATE_MULTI_CALL);
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
      DETAILED_PROFILER(runtime, DEACTIVATE_MULTI_CALL);
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_INDEX_SPACE_CALL);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
      }

#ifdef DEBUG_LEGION
      assert(minimal_points_assigned == 0);
#endif
      for (unsigned idx = 0; idx < output.slices.size(); idx++)
      {
        const Mapper::TaskSlice &slice = output.slices[idx]; 
        if (!slice.proc.exists())
        {
          log_run.error("Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned a slice for task "
                        "%s (ID %lld) with an invalid processor " IDFMT ".",
                        mapper->get_mapper_name(), get_task_name(),
                        get_unique_id(), slice.proc.id);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_DOMAIN_SLICE);
        }
#ifdef DEBUG_LEGION
        // Check to make sure the domain is not empty
        const Domain &d = slice.domain;
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
        SliceTask *new_slice = this->clone_as_slice_task(slice.domain,
                                                         slice.proc,
                                                         slice.recurse,
                                                         slice.stealable,
                                                         output.slices.size());
        slices.push_back(new_slice);
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, CLONE_MULTI_CALL);
      this->clone_task_op_from(rhs, p, stealable, false/*duplicate*/);
      this->index_domain = d;
      this->must_epoch_task = rhs->must_epoch_task;
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(minimal_points.find(p) == minimal_points.end());
#endif
      minimal_points[p] = point;
    }

    //--------------------------------------------------------------------------
    bool MultiTask::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MULTI_TRIGGER_EXECUTION_CALL);
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
#ifdef DEBUG_LEGION
#ifndef NDEBUG
                    bool still_local = 
#endif
#endif
                    distribute_task();
#ifdef DEBUG_LEGION
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
    void MultiTask::pack_multi_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_MULTI_CALL);
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
    void MultiTask::unpack_multi_task(Deserializer &derez,
                                      std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_MULTI_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events); 
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* MultiTask::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      return &version_infos;
    }

    //--------------------------------------------------------------------------
    void MultiTask::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, ACTIVATE_INDIVIDUAL_CALL);
      activate_single();
      future_store = NULL;
      future_size = 0;
      predicate_false_result = NULL;
      predicate_false_size = 0;
      orig_task = this;
      remote_owner_uid = 0;
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
      DETAILED_PROFILER(runtime, DEACTIVATE_INDIVIDUAL_CALL);
      // If we are the top_level task then deactivate our parent context
      const bool is_local_task = !is_remote();
      if (top_level_task && is_local_task)
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
          runtime->send_remote_context_free(it->first, rez);
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
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances); 
      acquired_instances.clear();
      // Read this before freeing the task
      // Should be safe, but we'll be careful
      const bool is_top_level_task = top_level_task;
      runtime->free_individual_task(this);
      // If we are the top-level-task and we are deactivated then
      // it is now safe to shutdown the machine
      if (is_top_level_task && is_local_task)
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
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
            }
#endif
          }
          else
          {
            // TODO: Put this check back in
#ifdef PERFORM_PREDICATE_SIZE_CHECKS
            if (predicate_false_size != variants->return_size)
            {
              log_run.error("Predicated task launch for task %s "
                                 "in parent task %s (UID %lld) has predicated "
                                 "false return type of size %ld bytes, but the "
                                 "expected return size is %ld bytes.",
                                 get_task_name(), parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 predicate_false_size, variants->return_size);
#ifdef DEBUG_LEGION
              assert(false);
#endif
              exit(ERROR_PREDICATE_RESULT_SIZE_MISMATCH);
            }
#endif
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(remote_outermost_context.exists());
#endif
      initialize_paths(); 
      // Get a future from the parent context to use as the result
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(!top_level_task), 
            runtime->address_space, runtime->address_space, 
            RtUserEvent::NO_RT_USER_EVENT, this));
      check_empty_field_requirements();
      update_no_access_regions();
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
      if (check_privileges)
        perform_privilege_checks();
      remote_outermost_context = 
        find_outermost_context()->get_context();
#ifdef DEBUG_LEGION
      assert(remote_outermost_context.exists());
#endif
      initialize_paths();
      result = Future(legion_new<FutureImpl>(runtime, true/*register*/,
            runtime->get_available_distributed_id(!top_level_task), 
            runtime->address_space, runtime->address_space, 
            RtUserEvent::NO_RT_USER_EVENT, this));
      check_empty_field_requirements();
      update_no_access_regions();
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
    void IndividualTask::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_REMOTE_STATE_ANALYSIS_CALL);
#ifdef DEBUG_LEGION
      assert(version_infos.size() == virtual_mapped.size());
#endif
      std::set<RtEvent> preconditions; 
      if (is_remote())
      {
        if (is_locally_mapped())
        {
          // See if we have any to unpack and make local
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
          {
            if (version_infos[idx].is_packed())
              version_infos[idx].make_local(preconditions,this,runtime->forest);
          }
        }
        else
        {
          // Otherwise request state for anything 
          // that was not early mapped or was virtually mapped 
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
          {
            if (early_mapped_regions.find(idx) == early_mapped_regions.end()) 
              version_infos[idx].make_local(preconditions,this,runtime->forest);
          }
        }
      }
      else
      {
        // We're still local, see if we are locally mapped or not
        if (is_locally_mapped())
        {
          // If we're locally mapping, we need everything now
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
            version_infos[idx].make_local(preconditions, this, runtime->forest);
        }
        else
        {
          // We only early map restricted regions for individual tasks
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
          {
            if (!regions[idx].is_restricted())
              continue;
            version_infos[idx].make_local(preconditions, this, runtime->forest);
          }
        }
      }
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
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
#ifdef DEBUG_LEGION
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
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                IndividualTask::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
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
        ApEvent wait_on = predicate_false_future.impl->get_ready_event();
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
          result.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
          predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
          Runtime::DeferredFutureSetArgs args;
          args.hlr_id = HLR_DEFERRED_FUTURE_SET_ID;
          args.target = result.impl;
          args.result = predicate_false_future.impl;
          args.task_op = this;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
                                           HLR_DEFERRED_FUTURE_SET_ID,
                                           HLR_LATENCY_PRIORITY, this, 
                                           Runtime::protect_event(wait_on));
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
      // "mapping" does not change the physical state
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement &req = regions[idx];
	VersionInfo &version_info = version_infos[idx];
	RegionTreeContext req_ctx =
	  parent_ctx->find_enclosing_context(parent_req_indexes[idx]);
	// don't bother if this wasn't going to change mapping state anyway
	// only requirements with write privileges bump version numbers
	if(!IS_WRITE(req))
	  continue;
	version_info.apply_mapping(req_ctx.get_id(),
				   runtime->address_space,
				   map_applied_conditions,
				   true /*copy previous*/);
      }      
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
      {
        bool result = early_map_regions(map_applied_conditions, 
                                        early_map_indexes);
        if (!acquired_instances.empty())
          release_acquired_instances(acquired_instances);
        return result;
      }
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
      DETAILED_PROFILER(runtime, INDIVIDUAL_PERFORM_MAPPING_CALL);
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
            if (!virtual_mapped[idx] && !no_access_regions[idx])
              version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                           owner_space, map_applied_conditions);
        }
        else
        {
          for (unsigned idx = 0; idx < version_infos.size(); idx++)
            if (!virtual_mapped[idx] && !no_access_regions[idx])
              version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                runtime->address_space, map_applied_conditions);
        }
        // If we succeeded in mapping and everything was mapped
        // then we get to mark that we are done mapping
        if (is_leaf())
        {
          RtEvent applied_condition;
          if (!map_applied_conditions.empty())
          {
            applied_condition = Runtime::merge_events(map_applied_conditions);
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
          if (!acquired_instances.empty())
            release_acquired_instances(acquired_instances);
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
#ifdef DEBUG_LEGION
      assert(idx < restrict_infos.size());
#endif
      // We know that if there are any restrictions they directly apply
      return restrict_infos[idx].has_restrictions();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::can_early_complete(ApUserEvent &chain_event)
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
      DETAILED_PROFILER(runtime, INDIVIDUAL_RETURN_VIRTUAL_CALL);
#ifdef DEBUG_LEGION
      assert(refs.size() == 1);
      assert(refs[0].is_composite_ref());
#endif
      RegionTreeContext virtual_ctx = get_parent_context(index);
#ifdef DEBUG_LEGION
      assert(virtual_mapped[index]);
#endif
      runtime->forest->physical_register_only(virtual_ctx, regions[index],
                                              version_infos[index], this,
                                              index, ApEvent::NO_AP_EVENT,
                                              false/*defer add users*/, 
                                              map_applied_conditions, refs
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
      // Apply our version state information, put map applied information
      // in a temporary data structure and then hold the lock when merging
      // it back into map_applied conditions
      std::set<RtEvent> temp_map_applied_conditions;
      if (is_remote())
      {
        AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
        version_infos[index].apply_mapping(virtual_ctx.get_id(),
                                    owner_space, temp_map_applied_conditions);
      }
      else
        version_infos[index].apply_mapping(virtual_ctx.get_id(),
                         runtime->address_space, temp_map_applied_conditions);
      if (!temp_map_applied_conditions.empty())
      {
        AutoLock o_lock(op_lock);
        map_applied_conditions.insert(temp_map_applied_conditions.begin(),
                                      temp_map_applied_conditions.end());
      }
    }

    //--------------------------------------------------------------------------
    VersionInfo& IndividualTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* IndividualTask::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      return &version_infos;
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndividualTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < privilege_paths.size());
#endif
      return privilege_paths[idx];
    }

    //--------------------------------------------------------------------------
    void IndividualTask::recapture_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      version_infos[idx].recapture_state();
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualTask::get_task_completion(void) const
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
#ifdef DEBUG_LEGION
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
    void IndividualTask::send_remote_context(AddressSpaceID remote_instance,
                                             RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_instance != runtime->address_space);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        pack_remote_context(rez, remote_instance);
      }
      runtime->send_remote_context_response(remote_instance, rez);
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMPLETE_CALL);
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      if (!is_remote())
      {
        // Pass back our created and deleted operations
        if (!top_level_task)
          return_privilege_state(parent_ctx);
        else
        {
          // Pass back the leaked top-level regions so that the outtermost
          // context knows how to clear its state when it is cleaning up
          RemoteTask *outer = static_cast<RemoteTask*>(parent_ctx);
          for (std::deque<RegionRequirement>::const_iterator it = 
                created_requirements.begin(); it != 
                created_requirements.end(); it++)
          {
            outer->add_top_region(it->region);
          }
        }
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
      if (must_epoch != NULL)
        must_epoch->notify_subop_complete(this);
      // Mark that this operation is complete
      complete_operation();
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMMIT_CALL);
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
      if (must_epoch != NULL)
        must_epoch->notify_subop_commit(this);
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
    void IndividualTask::handle_post_mapped(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_POST_MAPPED_CALL);
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
                                         HLR_LATENCY_PRIORITY,
                                         this, mapped_precondition);
        return;
      }
      // We used to have to apply our virtual state here, but that is now
      // done when the virtual instances are returned in return_virtual_task
      // If we have any virtual instances then we need to apply
      // the changes for them now
      if (!is_remote())
      {
        if (!acquired_instances.empty())
          release_acquired_instances(acquired_instances);
        // Handle remaining state flowing back out for virtual mappings
        // and newly created regions and fields, don't have to do this
        // though if we are at the top of the task tree
        if (!top_level_task)
          convert_virtual_instance_top_views(remote_instances);
        if (!map_applied_conditions.empty())
        {
          map_applied_conditions.insert(mapped_precondition);
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        }
        else 
          complete_mapping(mapped_precondition);
        return;
      }
      RtEvent applied_condition;
      if (!map_applied_conditions.empty())
        applied_condition = Runtime::merge_events(map_applied_conditions);
      // Send back the message saying that we finished mapping
      Serializer rez;
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
      rez.serialize(applied_condition);
      runtime->send_individual_remote_mapped(orig_proc, rez);
      // Now we can complete this task
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      complete_mapping(applied_condition);
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_physical_traversal(unsigned idx, 
                                      RegionTreeContext ctx, InstanceSet &valid)
    //--------------------------------------------------------------------------
    {
      runtime->forest->physical_traverse_path(ctx, privilege_paths[idx],
                                              regions[idx], 
                                              version_infos[idx], this, idx, 
                                              true/*find valid*/, 
                                              map_applied_conditions, valid
#ifdef DEBUG_LEGION
                                              , get_logging_name() 
                                              , get_unique_id()
#endif
                                              );
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_TASK_CALL);
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
      rez.serialize(top_level_task);
      if (!is_locally_mapped())
      {
        // Indicate we have to send full version infos for all requirements
        std::vector<bool> full_version_infos(regions.size(), true);
        pack_version_infos(rez, version_infos, full_version_infos);
      }
      else // Only virtual mapped regions require full infos
        pack_version_infos(rez, version_infos, virtual_mapped);
      pack_restrict_infos(rez, restrict_infos);
      // Mark that we sent this task remotely
      sent_remotely = true;
      // If this task is remote, then deactivate it, otherwise
      // we're local so we don't want to be deactivated for when
      // return messages get sent back.
      return is_remote();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::unpack_task(Deserializer &derez, Processor current,
                                     std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      derez.deserialize(orig_task);
      derez.deserialize(remote_completion_event);
      derez.deserialize(remote_unique_id);
      derez.deserialize(remote_outermost_context);
      set_current_proc(current);
      derez.deserialize(remote_owner_uid);
      derez.deserialize(top_level_task);
      unpack_version_infos(derez, version_infos);
      unpack_restrict_infos(derez, restrict_infos);
      // Quick check to see if we've been sent back to our original node
      if (!is_remote())
      {
#ifdef DEBUG_LEGION
        // Need to make the deserializer happy in debug mode
        derez.advance_pointer(derez.get_remaining_bytes());
#endif
        // If we were sent back then mark that we are no longer remote
        sent_remotely = false;
        // Put the original instance back on the mapping queue and
        // deactivate this version of the task
        runtime->add_to_ready_queue(current_proc, orig_task, 
                                    false/*prev fail*/);
        deactivate();
        return false;
      }
      // Figure out what our parent context is
      parent_ctx = runtime->find_context(remote_owner_uid);
      // Set our parent task for the user
      parent_task = parent_ctx;
      // Check to see if we had no virtual mappings and everything
      // was pre-mapped and we're remote then we can mark this
      // task as being mapped
      if (is_locally_mapped() && is_leaf())
        complete_mapping();
      // If we're remote, we've already resolved speculation for now
      resolve_speculation();
      if (Runtime::legion_spy_enabled)
      {
        LegionSpy::log_point_point(remote_unique_id, get_unique_id());
#ifdef LEGION_SPY
        LegionSpy::log_event_dependence(completion_event, 
                                        remote_completion_event);
#endif
      }
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
      std::set<ApEvent> wait_on_events;
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
        ApEvent e = 
          Runtime::get_previous_phase(wait_barriers[idx].phase_barrier);
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      ApEvent start_condition = Runtime::merge_events(wait_on_events); 

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
      Runtime::trigger_event(completion_event);
      // Now we're done, someone else will deactivate us
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_mapped(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RtEvent applied;
      derez.deserialize(applied);
      if (applied.exists())
        map_applied_conditions.insert(applied);
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_REMOTE_COMPLETE_CALL);
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      send_back_created_state(target, regions.size(), remote_outermost_context);
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
      if (top_level_task)
      {
        rez.serialize<size_t>(created_requirements.size());
        for (unsigned idx = 0; idx < created_requirements.size(); idx++)
          pack_region_requirement(created_requirements[idx], rez);
      }
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_REMOTE_COMPLETE_CALL);
      DerezCheck z(derez);
      // First unpack the privilege state
      unpack_privilege_state(derez);
      // Unpack the future result
      if (must_epoch == NULL)
        result.impl->unpack_future(derez);
      else
        must_epoch->unpack_future(index_point, derez);
      if (top_level_task)
      {
        size_t num_created;
        derez.deserialize(num_created);
        created_requirements.resize(num_created);
        for (unsigned idx = 0; idx < num_created; idx++)
          unpack_region_requirement(created_requirements[idx], derez);
      }
      // Mark that we have both finished executing and that our
      // children are complete
      complete_execution();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, POINT_ACTIVATE_CALL);
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
      DETAILED_PROFILER(runtime, POINT_DEACTIVATE_CALL);
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(
            this->slice_owner->get_unique_op_id(),
            this->get_unique_op_id());
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
          runtime->send_remote_context_free(it->first, rez);
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
        if (!map_applied_conditions.empty())
        {
          RtEvent done = Runtime::merge_events(map_applied_conditions);
          slice_owner->record_child_mapped(done);
          complete_mapping(done);
        }
        else
        {
          // Tell our owner that we mapped
          slice_owner->record_child_mapped(RtEvent::NO_RT_EVENT);
          // Mark that we ourselves have mapped
          complete_mapping();
        }
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
    bool PointTask::can_early_complete(ApUserEvent &chain_event)
    //--------------------------------------------------------------------------
    {
      chain_event = point_termination;
      return true;
    }

    //--------------------------------------------------------------------------
    void PointTask::return_virtual_instance(unsigned index, InstanceSet &refs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index < regions.size());
#endif
      slice_owner->return_virtual_instance(index, refs, regions[index]);
    }

    //--------------------------------------------------------------------------
    VersionInfo& PointTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* PointTask::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_version_infos();
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
    ApEvent PointTask::get_task_completion(void) const
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
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     PointTask::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_acquired_instances_ref();
    }

    //--------------------------------------------------------------------------
    RemoteTask* PointTask::find_outermost_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
    void PointTask::send_remote_context(AddressSpaceID remote_instance,
                                        RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_instance != runtime->address_space);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        pack_remote_context(rez, remote_instance);
      }
      runtime->send_remote_context_response(remote_instance, rez);
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_COMPLETE_CALL);
      // Pass back our created and deleted operations 
      slice_owner->return_privileges(this);

      slice_owner->record_child_complete();

      // Since this point is now complete we know
      // that we can trigger it. Note we don't need to do
      // this if we're a leaf task with no virtual mappings
      // because we would have performed the leaf task
      // early complete chaining operation.
      if (!is_leaf() || has_virtual_instances())
        Runtime::trigger_event(point_termination);

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
      DETAILED_PROFILER(runtime, POINT_TASK_COMMIT_CALL);
      // Commit this operation
      // Don't deactivate ourselves, our slice will do that for us
      commit_operation(false/*deactivate*/);
      // Then tell our slice owner that we're done
      slice_owner->record_child_committed();
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_physical_traversal(unsigned idx,
                                      RegionTreeContext ctx, InstanceSet &valid)
    //--------------------------------------------------------------------------
    {
      // We only need to traverse from the upper bound region because our
      // slice already traversed down to the upper bound for all points
      RegionTreePath traversal_path;
      const RegionRequirement &orig_req = slice_owner->regions[idx];
      if (orig_req.handle_type == PART_PROJECTION)
        runtime->forest->initialize_path(regions[idx].region.get_index_space(),
            orig_req.partition.get_index_partition(), traversal_path);
      else
        runtime->forest->initialize_path(regions[idx].region.get_index_space(),
            orig_req.region.get_index_space(), traversal_path);
      runtime->forest->physical_traverse_path(ctx, traversal_path, regions[idx],
                                             slice_owner->get_version_info(idx),
                                             this, idx, true/*find valid*/,
                                             map_applied_conditions, valid
#ifdef DEBUG_LEGION
                                             , get_logging_name()
                                             , get_unique_id()
#endif
                                             );
    }

    //--------------------------------------------------------------------------
    bool PointTask::pack_task(Serializer &rez, Processor target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_PACK_TASK_CALL);
      RezCheck z(rez);
      pack_single_task(rez, runtime->find_address_space(target));
      rez.serialize(point_termination); 
      // Return false since point tasks should always be deactivated
      // once they are sent to a remote node
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      derez.deserialize(point_termination);
      set_current_proc(current);
      // Get the context information from our slice owner
      parent_ctx = slice_owner->get_parent();
      parent_task = parent_ctx;
      // Check to see if we had no virtual mappings and everything
      // was pre-mapped and we're remote then we can mark this
      // task as being mapped
      if (is_locally_mapped() && is_leaf())
      {
        slice_owner->record_child_mapped(RtEvent::NO_RT_EVENT);
        complete_mapping();
      }
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(completion_event, point_termination);
#endif
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
    void PointTask::handle_post_mapped(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_POST_MAPPED_CALL);
      if (!mapped_precondition.has_triggered())
      {
        SingleTask::DeferredPostMappedArgs args;
        args.hlr_id = HLR_DEFERRED_POST_MAPPED_ID;
        args.task = this;
        runtime->issue_runtime_meta_task(&args, sizeof(args),
                                         HLR_DEFERRED_POST_MAPPED_ID,
                                         HLR_LATENCY_PRIORITY,
                                         this, mapped_precondition);
        return;
      }
      // Handle remaining state flowing back out for virtual mappings
      // and newly created regions and fields
      convert_virtual_instance_top_views(remote_instances);
      if (!map_applied_conditions.empty())
      {
        RtEvent done = Runtime::merge_events(map_applied_conditions);
        slice_owner->record_child_mapped(done);
        complete_mapping(done);
      }
      else
      {
        slice_owner->record_child_mapped(RtEvent::NO_RT_EVENT);
        // Now we can complete this point task
        complete_mapping();
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(SliceTask *owner, MinimalPoint *mp)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      compute_point_region_requirements(mp);
      update_no_access_regions();
      // Get our argument
      mp->assign_argument(local_args, local_arglen);
      // Make a new termination event for this point
      point_termination = Runtime::create_ap_user_event();
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
    bool WrapperTask::can_early_complete(ApUserEvent &chain_event)
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
    void WrapperTask::perform_physical_traversal(unsigned idx,
                                      RegionTreeContext ctx, InstanceSet &valid)
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
    bool WrapperTask::unpack_task(Deserializer &derez, Processor current,
                                  std::set<RtEvent> &ready_events)
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
    void WrapperTask::handle_post_mapped(RtEvent mapped_precondition)
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
    int RemoteTask::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::activate(void)
    //--------------------------------------------------------------------------
    { 
      DETAILED_PROFILER(runtime, REMOTE_TASK_ACTIVATE_CALL);
      activate_wrapper();
      parent_ctx = NULL;
      parent_task = NULL;
      context = RegionTreeContext();
      remote_owner_uid = 0;
      parent_context_uid = 0;
      depth = -1;
      is_top_level_context = false;
      remote_completion_event = ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REMOTE_TASK_DEACTIVATE_CALL);
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
      top_level_regions.clear();
      deactivate_wrapper();
      if (!remote_instances.empty())
      {
#ifdef DEBUG_LEGION
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
          runtime->send_remote_context_free(it->first, rez);
        }
        remote_instances.clear();
      }
      version_infos.clear();
      // Context is freed in deactivate single
      runtime->free_remote_task(this);
    }

    //--------------------------------------------------------------------------
    void RemoteTask::initialize_remote(UniqueID context_uid, bool is_top_level)
    //--------------------------------------------------------------------------
    {
      remote_owner_uid = context_uid;
      is_top_level_context = is_top_level;
      runtime->allocate_local_context(this);
#ifdef DEBUG_LEGION
      assert(context.exists());
      runtime->forest->check_context_state(context);
#endif
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
    VersionInfo& RemoteTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* RemoteTask::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      return &version_infos;
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
#ifdef DEBUG_LEGION
      assert(is_top_level_context);
#endif
    }

    //--------------------------------------------------------------------------
    void RemoteTask::send_remote_context(AddressSpaceID remote_instance,
                                         RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_instance != runtime->address_space);
#endif
      // should only see this call if it is the top-level context
#ifdef DEBUG_LEGION
      assert(is_top_level_context);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        pack_remote_context(rez, remote_instance);
      }
      runtime->send_remote_context_response(remote_instance, rez);
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    SingleTask* RemoteTask::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      // See if we already have it
      if (parent_ctx != NULL)
        return parent_ctx;
#ifdef DEBUG_LEGION
      assert(parent_context_uid != 0);
#endif
      // THIS IS ONLY SAFE BECAUSE THIS FUNCTION IS NEVER CALLED BY
      // A MESSAGE IN THE CONTEXT_VIRTUAL_CHANNEL
      parent_ctx = runtime->find_context(parent_context_uid);
      parent_task = parent_ctx;
      return parent_ctx;
    }

    //--------------------------------------------------------------------------
    ApEvent RemoteTask::get_task_completion(void) const
    //--------------------------------------------------------------------------
    {
      return remote_completion_event;
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
    void RemoteTask::unpack_remote_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REMOTE_UNPACK_CONTEXT_CALL);
      derez.deserialize(depth);
      unpack_base_external_task(derez);
      version_infos.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        version_infos[idx].unpack_version_numbers(derez, runtime->forest);
      virtual_mapped.resize(regions.size(), false);
      size_t num_virtual;
      derez.deserialize(num_virtual);
      for (unsigned idx = 0; idx < num_virtual; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        virtual_mapped[index] = true;
      }
      update_no_access_regions();
      size_t num_local;
      derez.deserialize(num_local);
      local_fields.resize(num_local);
      for (unsigned idx = 0; idx < num_local; idx++)
      {
        derez.deserialize(local_fields[idx]);
        allocate_local_field(local_fields[idx]);
      }
      derez.deserialize(remote_completion_event);
      derez.deserialize(remote_owner_uid);
      derez.deserialize(parent_context_uid);
      // See if we can find our parent task, if not don't worry about it
      // DO NOT CHANGE THIS UNLESS YOU THINK REALLY HARD ABOUT VIRTUAL 
      // CHANNELS AND HOW CONTEXT META-DATA IS MOVED!
      parent_ctx = runtime->find_context(parent_context_uid, true/*can fail*/);
      parent_task = parent_ctx;
      // Add our regions to the set of top-level regions
      for (unsigned idx = 0; idx < regions.size(); idx++)
        top_level_regions.insert(regions[idx].region);
    }

    //--------------------------------------------------------------------------
    void RemoteTask::add_top_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      top_level_regions.insert(handle);
    }
    
    //--------------------------------------------------------------------------
    void RemoteTask::convert_virtual_instances(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_created;
      derez.deserialize(num_created);
      created_requirements.resize(num_created);
      for (unsigned idx = 0; idx < num_created; idx++)
        unpack_region_requirement(created_requirements[idx], derez);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      std::map<AddressSpaceID,RemoteTask*> empty_remote;
      convert_virtual_instance_top_views(empty_remote);
      if (!map_applied_conditions.empty())
        Runtime::trigger_event(to_trigger,
                               Runtime::merge_events(map_applied_conditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTask::handle_convert_virtual_instances(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteTask *local_context;
      derez.deserialize(local_context);
      local_context->convert_virtual_instances(derez);
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
      orig_proc = clone->orig_proc;
      current_proc = clone->current_proc;
      target_proc = clone->target_proc;
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
        if (index < enclosing->regions.size())
        {
          regions[idx].parent = enclosing->regions[index].parent;
          physical_regions[idx] = enclosing->get_physical_region(index);
        }
        else
        {
          // This is a created requirements, so we have to make a copy
          RegionRequirement copy;
          enclosing->clone_requirement(index, copy);
          regions[idx].parent = copy.parent;
          // physical regions are empty becaue they are virtual
        }
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
    void InlineTask::send_remote_context(AddressSpaceID remote_inst,
                                         RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent InlineTask::get_task_completion(void) const
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
      DETAILED_PROFILER(runtime, INDEX_ACTIVATE_CALL);
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
      DETAILED_PROFILER(runtime, INDEX_DEACTIVATE_CALL);
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
#ifdef DEBUG_LEGION
      assert(acquired_instances.empty());
#endif
      acquired_instances.clear();
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
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
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
      index_domain = launcher.launch_domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->is_foldable)
      {
        log_run.error("Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
            runtime->address_space, runtime->address_space, 
            RtUserEvent::NO_RT_USER_EVENT, this));
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
      regions = region_requirements;
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
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
      index_domain = domain;
      initialize_base_task(ctx, true/*track*/, pred, task_id);
      if (check_privileges)
        perform_privilege_checks();
      initialize_paths();
      annotate_early_mapped_regions();
      future_map = FutureMap(legion_new<FutureMapImpl>(ctx, this, runtime));
#ifdef DEBUG_LEGION
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
      regions = region_requirements;
      arglen = global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
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
      index_domain = domain;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->is_foldable)
      {
        log_run.error("Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
            runtime->address_space, runtime->address_space, 
            RtUserEvent::NO_RT_USER_EVENT, this));
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
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
#endif
        }
        else
        {
          // TODO: Reenable this error if we want to track predicate defaults
#ifdef PERFORM_PREDICATE_SIZE_CHECKS
          if (predicate_false_size != variants->return_size)
          {
            log_run.error("Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has predicated "
                          "false return type of size %ld bytes, but the "
                          "expected return size is %ld bytes.",
                          get_task_name(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id(),
                          predicate_false_size, variants->return_size);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_PREDICATE_RESULT_SIZE_MISMATCH);
          }
#endif
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
    void IndexTask::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_REMOTE_STATE_ANALYSIS_CALL);
      std::set<RtEvent> preconditions;
      if (is_locally_mapped())
      {
        // If we're locally mapped, request everyone's state
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
          version_infos[idx].make_local(preconditions, this, runtime->forest);
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
            version_infos[idx].make_local(preconditions, this, runtime->forest);
        }
      }
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, INDEX_COMPUTE_FAT_PATH_CALL);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_BAD_PROJECTION_USE);
      }
#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndexTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          ApEvent wait_on = predicate_false_future.impl->get_ready_event();
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
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF,
                                                         this);
            Runtime::DeferredFutureMapSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_MAP_SET_ID;
            args.future_map = future_map.impl;
            args.result = predicate_false_future.impl;
            args.domain = index_domain;
            args.task_op = this;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_FUTURE_MAP_SET_ID,
                                             HLR_LATENCY_PRIORITY, this, 
                                             Runtime::protect_event(wait_on));
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
          ApEvent wait_on = predicate_false_future.impl->get_ready_event();
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
            reduction_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, 
                                                         this);
            Runtime::DeferredFutureSetArgs args;
            args.hlr_id = HLR_DEFERRED_FUTURE_SET_ID;
            args.target = reduction_future.impl;
            args.result = predicate_false_future.impl;
            args.task_op = this;
            runtime->issue_runtime_meta_task(&args, sizeof(args),
                                             HLR_DEFERRED_FUTURE_SET_ID,
                                             HLR_LATENCY_PRIORITY, this, 
                                             Runtime::protect_event(wait_on));
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
      assert(0 && "TODO: advance mapping states if you care");
      complete_mapping();
      trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_EARLY_MAP_TASK_CALL);
      std::vector<unsigned> early_map_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &req = regions[idx];
        if (req.is_restricted() || req.must_premap())
          early_map_indexes.push_back(idx);
      }
      if (!early_map_indexes.empty())
      {
        bool result = early_map_regions(map_applied_conditions, 
                                        early_map_indexes);
        if (!acquired_instances.empty())
          release_acquired_instances(acquired_instances);
        return result;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_DISTRIBUTE_CALL);
      if (is_locally_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
          assert(minimal_points_assigned == 0);
#endif
          // Make a slice copy and send it away
          SliceTask *clone = clone_as_slice_task(index_domain, target_proc,
                                                 true/*needs slice*/,
                                                 stealable, 1LL);
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, INDEX_PERFORM_MAPPING_CALL);
      // This will only get called if we had slices that failed to map locally
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(is_sliced());
      assert(!slices.empty());
#endif
      return trigger_slices();
    }

    //--------------------------------------------------------------------------
    ApEvent IndexTask::get_task_completion(void) const
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
      DETAILED_PROFILER(runtime, INDEX_COMPLETE_CALL);
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
      if (must_epoch != NULL)
        must_epoch->notify_subop_complete(this);
      complete_operation();
      if (speculation_state == RESOLVE_FALSE_STATE)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_COMMIT_CALL);
      // We can release our version infos now
      for (std::vector<VersionInfo>::iterator it = version_infos.begin();
            it != version_infos.end(); it++)
      {
        it->release();
      }
      if (must_epoch != NULL)
        must_epoch->notify_subop_commit(this);
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
    bool IndexTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
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
      DETAILED_PROFILER(runtime, INDEX_PERFORM_INLINING_CALL);
      // See if there is anything to wait for
      std::set<ApEvent> wait_on_events;
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
	ApEvent e = 
          Runtime::get_previous_phase(wait_barriers[idx].phase_barrier);
        wait_on_events.insert(e);
      }
      // Merge together all the events for the start condition 
      ApEvent start_condition = Runtime::merge_events(wait_on_events); 

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
      Runtime::trigger_event(completion_event);
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
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     IndexTask::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(const Domain &d, Processor p,
                                              bool recurse, bool stealable,
                                              long long scale_denominator)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_CLONE_AS_SLICE_CALL);
      SliceTask *result = runtime->get_available_slice_task(false); 
      result->initialize_base_task(parent_ctx, 
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_multi_from(this, d, p, recurse, stealable);
      result->remote_outermost_context = 
        parent_ctx->find_outermost_context()->get_context();
#ifdef DEBUG_LEGION
      assert(result->remote_outermost_context.exists());
#endif
      result->index_complete = this->completion_event;
      result->denominator = scale_denominator;
      result->index_owner = this;
      result->remote_owner_uid = parent_ctx->get_unique_id();
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_index_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
                                                result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_future(const DomainPoint &point, const void *result,
                                  size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_HANDLE_FUTURE);
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
    void IndexTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void IndexTask::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_ENUMERATE_POINTS_CALL);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
                                        RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_MAPPED_CALL);
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
          RtEvent map_condition = Runtime::merge_events(map_applied_conditions);
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
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMPLETE_CALL);
      bool trigger_execution = false;
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        complete_points += points;
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMMIT_CALL);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        committed_points += points;
#ifdef DEBUG_LEGION
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
      RtEvent applied_condition;
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_ACTIVATE_CALL);
      activate_multi();
      // Slice tasks never have to resolve speculation
      resolve_speculation();
      reclaim = false;
      index_complete = ApEvent::NO_AP_EVENT;
      mapping_index = 0;
      num_unmapped_points = 0;
      num_uncomplete_points = 0;
      num_uncommitted_points = 0;
      denominator = 0;
      index_owner = NULL;
      remote_owner_uid = 0;
      remote_unique_id = get_unique_id();
      locally_mapped = false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DEACTIVATE_CALL);
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
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      acquired_instances.clear();
      map_applied_conditions.clear();
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
    void SliceTask::trigger_remote_state_analysis(RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_REMOTE_STATE_ANALYSIS_CALL);
      std::set<RtEvent> preconditions;
      if (is_locally_mapped())
      {
        // See if we have any version infos that still need to be unpacked
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          if (version_infos[idx].is_packed())
            version_infos[idx].make_local(preconditions, this, runtime->forest);
        }
      }
      else
      {
        // Otherwise we just need to request state for 
        // any non-eary mapped regions
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          if (early_mapped_regions.find(idx) == early_mapped_regions.end())
            version_infos[idx].make_local(preconditions, this, runtime->forest);
        }
      }
      if (preconditions.empty())
        Runtime::trigger_event(ready_event);
      else
        Runtime::trigger_event(ready_event,
                               Runtime::merge_events(preconditions));
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
      DETAILED_PROFILER(runtime, SLICE_PREWALK_CALL);
      // Premap all regions that were not early mapped
      std::set<RtEvent> empty_conditions;
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
                                        version_infos[idx], this, idx,
                                        false/*find valid*/, 
                                        empty_conditions, empty_set
#ifdef DEBUG_LEGION
                                        , get_logging_name(), unique_op_id
#endif
                                        );
      }
#ifdef DEBUG_LEGION
      assert(empty_conditions.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void SliceTask::apply_local_version_infos(std::set<RtEvent> &map_conditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_APPLY_VERSION_INFO_CALL);
      // We know we are local
      AddressSpaceID owner_space = runtime->address_space; 
      for (unsigned idx = 0; idx < version_infos.size(); idx++)
      {
        version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                         owner_space, map_conditions);
      }
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     SliceTask::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DISTRIBUTE_CALL);
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
      DETAILED_PROFILER(runtime, SLICE_PERFORM_MAPPING_CALL);
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
#ifdef DEBUG_LEGION
          bool point_success = 
#endif
            points[idx]->perform_mapping(epoch_owner);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
          bool slice_success = 
#endif
            (*it)->trigger_execution();
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_LAUNCH_CALL);
      // Quick out in case we are reclaiming this task
      if (reclaim)
      {
        deactivate();
        return;
      }
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_MAP_AND_LAUNCH_CALL);
      // Walk the path down to the upper bounds for all points first
      prewalk_slice();
      // Mark that this task is no longer stealable.  Once we start
      // executing things onto a specific processor slices cannot move.
      stealable = false;
      // First enumerate all of our points if we haven't already done so
      if (points.empty())
        enumerate_points();
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
        bool point_success = 
#endif
          next_point->perform_mapping();
#ifdef DEBUG_LEGION
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
    ApEvent SliceTask::get_task_completion(void) const
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
      DETAILED_PROFILER(runtime, SLICE_PACK_TASK_CALL);
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
      if (is_locally_mapped() && (num_unmapped_points == 0))
      {
        // If we've mapped everything and there are no virtual mappings
        // then we can just send the version numbers
        std::vector<bool> full_version_infos(regions.size(), false);
        pack_version_infos(rez, version_infos, full_version_infos);
      }
      else
      {
        // Otherwise we have to send all the version infos, we could try
        // and figure out which subset of region requirements have full
        // or partial virtual mappings, but that might be expensive
        std::vector<bool> full_version_infos(regions.size(), true);
        pack_version_infos(rez, version_infos, full_version_infos);
      }
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
    bool SliceTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez, ready_events);
      set_current_proc(current);
      derez.deserialize(denominator);
      derez.deserialize(index_owner);
      derez.deserialize(index_complete);
      derez.deserialize(remote_unique_id); 
      derez.deserialize(remote_outermost_context);
      derez.deserialize(locally_mapped);
      derez.deserialize(remote_owner_uid);
      unpack_version_infos(derez, version_infos);
      unpack_restrict_infos(derez, restrict_infos);
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_slice_slice(remote_unique_id, get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(remote_unique_id,
            get_unique_op_id());
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      // Check to see if we ended up back on the original node
      // We have to do this before unpacking the points
      if (is_remote())
        parent_ctx = runtime->find_context(remote_owner_uid);
      else
        parent_ctx = index_owner->parent_ctx;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = runtime->get_available_point_task(false); 
        point->slice_owner = this;
        point->unpack_task(derez, current, ready_events);
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
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_SLICE_CALL);
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
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_slice_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
            result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const DomainPoint &point, const void *result,
                                  size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_HANDLE_FUTURE_CALL);
      // If we're remote, just handle it ourselves, otherwise pass
      // it back to the enclosing index owner
      if (is_remote())
      {
        if (redop != 0)
          fold_reduction_future(result, result_size, owner, false/*exclusive*/);
        else
        {
          // Store it in our temporary futures
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_POINT_CALL);
      PointTask *result = runtime->get_available_point_task(false);
      result->initialize_base_task(parent_ctx,
                                   false/*track*/, Predicate::TRUE_PRED,
                                   this->task_id);
      result->clone_task_op_from(this, this->target_proc, 
                                 false/*stealable*/, true/*duplicate*/);
      result->is_index_space = true;
      result->must_epoch_task = this->must_epoch_task;
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
      DETAILED_PROFILER(runtime, SLICE_ENUMERATE_POINTS_CALL);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
    void SliceTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
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
    void SliceTask::return_virtual_instance(unsigned index, InstanceSet &refs,
                                            const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_RETURN_VIRTUAL_CALL);
      // Add it to our state
#ifdef DEBUG_LEGION
      assert(refs.size() == 1);
      assert(refs[0].is_composite_ref());
#endif
      RegionTreeContext virtual_ctx = get_parent_context(index);
      std::set<RtEvent> empty_conditions;
      // Have to control access to the version info data structure
      AutoLock o_lock(op_lock);
      // Hold a reference so it doesn't get deleted
      temporary_virtual_refs.push_back(refs[0]);
      runtime->forest->physical_register_only(virtual_ctx, req,
                                              version_infos[index], this,
                                              index, ApEvent::NO_AP_EVENT,
                                              false/*defer add users*/, 
                                              empty_conditions, refs
#ifdef DEBUG_LEGION
                                              , get_logging_name()
                                              , unique_op_id
#endif
                                              );
#ifdef DEBUG_LEGION
      assert(empty_conditions.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_child_mapped(RtEvent child_complete)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (child_complete.exists())
          map_applied_conditions.insert(child_complete);
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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
      DETAILED_PROFILER(runtime, SLICE_MAPPED_CALL);
      // No matter what, flush out our physical states
      RtEvent applied_condition;
      if (!is_remote() || !is_locally_mapped())
      {
        AddressSpaceID owner_space = runtime->find_address_space(orig_proc);
        for (unsigned idx = 0; idx < version_infos.size(); idx++)
        {
          version_infos[idx].apply_mapping(get_parent_context(idx).get_id(),
                                           owner_space, map_applied_conditions);
        }
        if (!map_applied_conditions.empty())
          applied_condition = Runtime::merge_events(map_applied_conditions);
      }
      else if (!map_applied_conditions.empty())
        applied_condition = Runtime::merge_events(map_applied_conditions);
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
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
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
      DETAILED_PROFILER(runtime, SLICE_COMPLETE_CALL);
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
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
      DETAILED_PROFILER(runtime, SLICE_COMMIT_CALL);
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
    void SliceTask::pack_remote_mapped(Serializer &rez, 
                                       RtEvent applied_condition)
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
        (*it)->send_back_created_state(target, regions.size(),
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
#ifdef DEBUG_LEGION
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
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      Runtime::trigger_event(ready_event);
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
      std::set<RtEvent> wait_events;
      {
        std::list<SliceTask*>::const_iterator it = slices.begin();
        DeferredSliceArgs args;
        args.hlr_id = HLR_DEFERRED_SLICE_ID;
        args.slicer = this;
        while (true) 
        {
          args.slice = *it;
          // Have to update this before launching the task to avoid 
          // the clean-up race
          it++;
          bool done = (it == slices.end()); 
          RtEvent wait = 
            owner->runtime->issue_runtime_meta_task(&args, sizeof(args), 
                                                    HLR_DEFERRED_SLICE_ID, 
                                                    HLR_LATENCY_PRIORITY, args.slice);
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
        RtEvent sliced_event = Runtime::merge_events(wait_events);
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
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
      assert(projections.find(idx) == projections.end());
#endif
      projections[idx] = handle;
    }
    
    //--------------------------------------------------------------------------
    void MinimalPoint::add_argument(const TaskArgument &argument, bool own)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
#ifdef DEBUG_LEGION
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

