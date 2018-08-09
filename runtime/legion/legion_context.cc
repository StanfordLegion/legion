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

#include "legion/runtime.h"
#include "legion/legion_tasks.h"
#include "legion/legion_trace.h"
#include "legion/legion_context.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Task Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(Runtime *rt, TaskOp *owner,
                             const std::vector<RegionRequirement> &reqs)
      : runtime(rt), owner_task(owner), regions(reqs),
        executing_processor(Processor::NO_PROC), total_tunable_count(0), 
        overhead_tracker(NULL), task_executed(false),
        has_inline_accessor(false), mutable_priority(false),
        children_complete_invoked(false), children_commit_invoked(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(const TaskContext &rhs)
      : runtime(NULL), owner_task(NULL), regions(rhs.regions)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
      // Clean up any local variables that we have
      if (!task_local_variables.empty())
      {
        for (std::map<LocalVariableID,
                      std::pair<void*,void (*)(void*)> >::iterator it = 
              task_local_variables.begin(); it != 
              task_local_variables.end(); it++)
        {
          if (it->second.second != NULL)
            (*it->second.second)(it->second.first);
        }
      }
    }

    //--------------------------------------------------------------------------
    TaskContext& TaskContext::operator=(const TaskContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID TaskContext::get_context_uid(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_unique_op_id();
    }

    //--------------------------------------------------------------------------
    int TaskContext::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_depth();
    }

    //--------------------------------------------------------------------------
    Task* TaskContext::get_task(void)
    //--------------------------------------------------------------------------
    {
      return owner_task;
    }

    //--------------------------------------------------------------------------
    TaskContext* TaskContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
#endif
      return owner_task->get_context();
    }

    //--------------------------------------------------------------------------
    bool TaskContext::is_leaf_context(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool TaskContext::is_inner_context(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void TaskContext::add_physical_region(const RegionRequirement &req,
                                   bool mapped, MapperID mid, MappingTagID tag,
                                   ApUserEvent unmap_event, bool virtual_mapped,
                                   const InstanceSet &physical_instances)
    //--------------------------------------------------------------------------
    {
      PhysicalRegionImpl *impl = new PhysicalRegionImpl(req, 
          ApEvent::NO_AP_EVENT, mapped, this, mid, tag, 
          is_leaf_context(), virtual_mapped, runtime);
      physical_regions.push_back(PhysicalRegion(impl));
      if (mapped)
        impl->reset_references(physical_instances, unmap_event);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion TaskContext::get_physical_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < regions.size()); // should be one of our original regions
#endif
      return physical_regions[idx];
    } 

    //--------------------------------------------------------------------------
    void TaskContext::destroy_user_lock(Reservation r)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no
      // need to hold the lock
      context_locks.push_back(r);
    }

    //--------------------------------------------------------------------------
    void TaskContext::destroy_user_barrier(ApBarrier b)
    //--------------------------------------------------------------------------
    {
      // Can only be called from user land so no 
      // need to hold the lock
      context_barriers.push_back(b);
    } 

    //--------------------------------------------------------------------------
    void TaskContext::add_created_region(LogicalRegion handle, bool task_local)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      RegionRequirement new_req(handle, READ_WRITE, EXCLUSIVE, handle);
      // Put a region requirement with no fields in the list of
      // created requirements, we know we can add any fields for
      // this field space in the future since we own all privileges
      // Now make a new region requirement and physical region
      created_requirements.push_back(new_req);
      // Created regions always return privileges that they make
      returnable_privileges.push_back(!task_local);
    }

    //--------------------------------------------------------------------------
    void TaskContext::log_created_requirements(void)
    //--------------------------------------------------------------------------
    {
      std::vector<MappingInstance> instances(1, 
            Mapping::PhysicalInstance::get_virtual_instance());
      const UniqueID unique_op_id = get_unique_id();
      const size_t original_size = 
        (owner_task == NULL) ? 0 : owner_task->regions.size();
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        TaskOp::log_requirement(unique_op_id, original_size + idx,
                                created_requirements[idx]);
        // Skip it if there are no privilege fields
        if (created_requirements[idx].privilege_fields.empty())
          continue;
        InstanceSet instance_set;
        std::vector<PhysicalManager*> unacquired;  
        RegionTreeID bad_tree; std::vector<FieldID> missing_fields;
        runtime->forest->physical_convert_mapping(owner_task, 
            created_requirements[idx], instances, instance_set, bad_tree, 
            missing_fields, NULL, unacquired, false/*do acquire_checks*/);
        runtime->forest->log_mapping_decision(unique_op_id,
            original_size + idx, created_requirements[idx], instance_set);
      }
    } 

    //--------------------------------------------------------------------------
    void TaskContext::register_region_creation(LogicalRegion handle,
                                               bool task_local)
    //--------------------------------------------------------------------------
    {
      // Create a new logical region 
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock priv_lock(privilege_lock);
#ifdef DEBUG_LEGION
      assert(created_regions.find(handle) == created_regions.end());
#endif
      created_regions[handle] = task_local; 
      add_created_region(handle, task_local);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_region_deletion(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      {
        AutoLock priv_lock(privilege_lock);
        std::map<LogicalRegion,bool>::iterator finder =
          created_regions.find(handle);
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
    void TaskContext::register_field_creation(FieldSpace handle, 
                                              FieldID fid, bool local)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      std::pair<FieldSpace,FieldID> key(handle,fid);
#ifdef DEBUG_LEGION
      assert(created_fields.find(key) == created_fields.end());
#endif
      created_fields[key] = local;
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_creations(FieldSpace handle, bool local,
                                          const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        std::pair<FieldSpace,FieldID> key(handle,fields[idx]);
#ifdef DEBUG_LEGION
        assert(created_fields.find(key) == created_fields.end());
#endif
        created_fields[key] = local;
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_deletions(FieldSpace handle,
                                         const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      std::set<FieldID> to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::set<FieldID>::const_iterator it = to_free.begin();
              it != to_free.end(); it++)
        {
          std::pair<FieldSpace,FieldID> key(handle,*it);
          std::map<std::pair<FieldSpace,FieldID>,bool>::iterator finder = 
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
    void TaskContext::register_field_space_creation(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
#ifdef DEBUG_LEGION
      assert(created_field_spaces.find(space) == created_field_spaces.end());
#endif
      created_field_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_space_deletion(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock priv_lock(privilege_lock);
        std::deque<FieldID> to_delete;
        for (std::map<std::pair<FieldSpace,FieldID>,bool>::const_iterator it =
              created_fields.begin(); it != created_fields.end(); it++)
        {
          if (it->first.first == space)
            to_delete.push_back(it->first.second);
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
    bool TaskContext::has_created_index_space(IndexSpace space) const
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      return (created_index_spaces.find(space) != created_index_spaces.end());
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_index_space_creation(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
#ifdef DEBUG_LEGION
      assert(created_index_spaces.find(space) == created_index_spaces.end());
#endif
      created_index_spaces.insert(space);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_index_space_deletion(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock priv_lock(privilege_lock);
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
    void TaskContext::register_index_partition_creation(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
#ifdef DEBUG_LEGION
      assert(created_index_partitions.find(handle) == 
             created_index_partitions.end());
#endif
      created_index_partitions.insert(handle);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_index_partition_deletion(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      {
        AutoLock priv_lock(privilege_lock);
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
    bool TaskContext::was_created_requirement_deleted(
                                             const RegionRequirement &req) const
    //--------------------------------------------------------------------------
    {
      // No need to worry about deleted field creation requirements here
      // since this method is only called for requirements with returnable
      // privileges and therefore we just need to see if the region is
      // still in the set of created regions.
      return (created_regions.find(req.region) == created_regions.end());
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_region_creations(
                                       const std::map<LogicalRegion,bool> &regs)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      for (std::map<LogicalRegion,bool>::const_iterator it = regs.begin();
            it != regs.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(created_regions.find(it->first) == created_regions.end());
#endif
        created_regions[it->first] = it->second;
        add_created_region(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_region_deletions(
                                            const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      std::vector<LogicalRegion> to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::set<LogicalRegion>::const_iterator it = regs.begin();
              it != regs.end(); it++)
        {
          std::map<LogicalRegion,bool>::iterator finder =
            created_regions.find(*it);
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
    void TaskContext::register_field_creations(
                     const std::map<std::pair<FieldSpace,FieldID>,bool> &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      for (std::map<std::pair<FieldSpace,FieldID>,bool>::const_iterator it = 
            fields.begin(); it != fields.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(created_fields.find(it->first) == created_fields.end());
#endif
        created_fields.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_deletions(
                        const std::set<std::pair<FieldSpace,FieldID> > &fields)
    //--------------------------------------------------------------------------
    {
      std::map<FieldSpace,std::set<FieldID> > to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
              fields.begin(); it != fields.end(); it++)
        {
          std::map<std::pair<FieldSpace,FieldID>,bool>::iterator finder = 
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
    void TaskContext::register_field_space_creations(
                                            const std::set<FieldSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
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
    void TaskContext::register_field_space_deletions(
                                            const std::set<FieldSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldSpace> to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::set<FieldSpace>::const_iterator it = spaces.begin();
              it != spaces.end(); it++)
        {
          std::deque<FieldID> to_delete;
          for (std::map<std::pair<FieldSpace,FieldID>,bool>::const_iterator cit 
                = created_fields.begin(); cit != created_fields.end(); cit++)
          {
            if (cit->first.first == *it)
              to_delete.push_back(cit->first.second);
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
    void TaskContext::register_index_space_creations(
                                            const std::set<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
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
    void TaskContext::register_index_space_deletions(
                                            const std::set<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexSpace> to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
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
    void TaskContext::register_index_partition_creations(
                                          const std::set<IndexPartition> &parts)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
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
    void TaskContext::register_index_partition_deletions(
                                          const std::set<IndexPartition> &parts)
    //--------------------------------------------------------------------------
    {
      std::vector<IndexPartition> to_finalize;
      {
        AutoLock priv_lock(privilege_lock);
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
    void TaskContext::analyze_destroy_fields(FieldSpace handle,
                                             const std::set<FieldID> &to_delete,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
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
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
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
    void TaskContext::analyze_destroy_logical_region(LogicalRegion handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
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
        std::vector<LegionColor> dummy_path;
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
      std::deque<RegionRequirement> local_requirements;
      std::deque<unsigned> parent_indexes;
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        for (std::deque<RegionRequirement>::const_iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); it++, parent_index++)
        {
          // Different index space trees means we can skip
          if (handle.get_tree_id() != it->region.get_tree_id())
            continue;
          local_requirements.push_back(*it);
          parent_indexes.push_back(parent_index);
        }
      }
      if (local_requirements.empty())
        return;
      unsigned local_index = 0;
      for (std::deque<RegionRequirement>::const_iterator it = 
            local_requirements.begin(); it != 
            local_requirements.end(); it++, local_index++)
      {
        // Disjoint index spaces means we can skip
        if (runtime->forest->are_disjoint(handle.get_index_space(), 
                                          it->region.index_space))
          continue;
        delete_reqs.resize(delete_reqs.size()+1);
        RegionRequirement &req = delete_reqs.back();
        std::vector<LegionColor> dummy_path;
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
        parent_req_indexes.push_back(parent_indexes[local_index]);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::analyze_destroy_logical_partition(LogicalPartition handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
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
        std::vector<LegionColor> dummy_path;
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
      std::deque<RegionRequirement> local_requirements;
      std::deque<unsigned> parent_indexes;
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        for (std::deque<RegionRequirement>::const_iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); it++, parent_index++)
        {
          // Different index space trees means we can skip
          if (handle.get_tree_id() != it->region.get_tree_id())
            continue;
          local_requirements.push_back(*it);
          parent_indexes.push_back(parent_index);
        }
      }
      if (local_requirements.empty())
        return;
      unsigned local_index = 0;
      for (std::deque<RegionRequirement>::const_iterator it = 
            local_requirements.begin(); it != 
            local_requirements.end(); it++, local_index++)
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
        std::vector<LegionColor> dummy_path;
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
        parent_req_indexes.push_back(parent_indexes[local_index]);
      }
    }

    //--------------------------------------------------------------------------
    int TaskContext::has_conflicting_regions(MapOp *op, bool &parent_conflict,
                                             bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = op->get_requirement(); 
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int TaskContext::has_conflicting_regions(AttachOp *attach,
                                             bool &parent_conflict,
                                             bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = attach->get_requirement();
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int TaskContext::has_conflicting_internal(const RegionRequirement &req,
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
        if (!physical_regions[our_idx].is_mapped())
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
        if (!it->is_mapped())
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
    void TaskContext::find_conflicting_regions(TaskOp *task,
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
        if (!physical_regions[our_idx].is_mapped())
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
        if (!it->is_mapped())
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
    void TaskContext::find_conflicting_regions(CopyOp *copy,
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
        if (!physical_regions[our_idx].is_mapped())
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
        if (!it->is_mapped())
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
    void TaskContext::find_conflicting_regions(AcquireOp *acquire,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = acquire->get_requirement();
      find_conflicting_internal(req, conflicting); 
    }

    //--------------------------------------------------------------------------
    void TaskContext::find_conflicting_regions(ReleaseOp *release,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = release->get_requirement();
      find_conflicting_internal(req, conflicting);      
    }

    //--------------------------------------------------------------------------
    void TaskContext::find_conflicting_regions(DependentPartitionOp *partition,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = partition->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void TaskContext::find_conflicting_internal(const RegionRequirement &req,
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
        if (!physical_regions[our_idx].is_mapped())
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
        if (!it->is_mapped())
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
    void TaskContext::find_conflicting_regions(FillOp *fill,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = fill->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::check_region_dependence(RegionTreeID our_tid,
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
        std::vector<LegionColor> path;
        if (!runtime->forest->compute_index_path(our_space,
                         req.region.get_index_space(),path))
          return false;
      }
      else
      {
        // Check if the trees are different
        if (our_tid != req.partition.get_tree_id())
          return false;
        std::vector<LegionColor> path;
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
    void TaskContext::register_inline_mapped_region(PhysicalRegion &region)
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
    void TaskContext::unregister_inline_mapped_region(PhysicalRegion &region)
    //--------------------------------------------------------------------------
    {
      // Don't need the lock because this is only accessed from the
      // executed task context
      for (std::list<PhysicalRegion>::iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (it->impl == region.impl)
        {
          if (runtime->runtime_warnings && !has_inline_accessor)
            has_inline_accessor = it->impl->created_accessor();
          inline_regions.erase(it);
          return;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TaskContext::safe_cast(RegionTreeForest *forest, IndexSpace handle,
                                const void *realm_point, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      // Check to see if we already have the pointer for the node
      std::map<IndexSpace,IndexSpaceNode*>::const_iterator finder =
        safe_cast_spaces.find(handle);
      if (finder == safe_cast_spaces.end())
      {
        safe_cast_spaces[handle] = forest->get_node(handle);
        finder = safe_cast_spaces.find(handle);
      }
      return finder->second->contains_point(realm_point, type_tag);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::is_region_mapped(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx].is_mapped();
    }

    //--------------------------------------------------------------------------
    void TaskContext::clone_requirement(unsigned idx, RegionRequirement &target)
    //--------------------------------------------------------------------------
    {
      if (idx >= regions.size())
      {
        idx -= regions.size();
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        assert(idx < created_requirements.size());
#endif
        target = created_requirements[idx];
      }
      else
        target = regions[idx];
    }

    //--------------------------------------------------------------------------
    int TaskContext::find_parent_region_req(const RegionRequirement &req,
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
            ((PRIV_ONLY(req) & our_req.privilege) != PRIV_ONLY(req)))
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
      const FieldSpace fs = req.parent.get_field_space(); 
      // The created region requirements have to be checked while holding
      // the lock since they are subject to mutation by the application
      // We might also mutate it so we take the lock in exclusive mode
      AutoLock priv_lock(privilege_lock);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        RegionRequirement &our_req = created_requirements[idx];
        // First check that the regions match
        if (our_req.region != req.parent)
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((PRIV_ONLY(req) & our_req.privilege) != PRIV_ONLY(req)))
          continue;
        // If this is a returnable privilege requiremnt that means
        // that we made this region so we always have privileges
        // on any fields for that region, just add them and be done
        if (returnable_privileges[idx])
        {
          our_req.privilege_fields.insert(req.privilege_fields.begin(),
                                          req.privilege_fields.end());
          return int(regions.size() + idx);
        }
        // Finally check that all the fields are contained
        bool dominated = true;
        for (std::set<FieldID>::const_iterator it = 
              req.privilege_fields.begin(); it !=
              req.privilege_fields.end(); it++)
        {
          if (our_req.privilege_fields.find(*it) ==
              our_req.privilege_fields.end())
          {
            // Check to see if this is a field we made
            std::pair<FieldSpace,FieldID> key(fs, *it);
            if (created_fields.find(key) != created_fields.end())
            {
              // We made it so we can add it to the requirement
              // and continue on our way
              our_req.privilege_fields.insert(*it);
              continue;
            }
            // Otherwise we don't have privileges
            dominated = false;
            break;
          }
        }
        if (!dominated)
          continue;
        // Include the offset by the number of base requirements
        return int(regions.size() + idx);
      }
      // Method of last resort, check to see if we made all the fields
      // if we did, then we can make a new requirement for all the fields
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        std::pair<FieldSpace,FieldID> key(fs, *it);
        // Didn't make it so we don't have privileges anywhere
        if (created_fields.find(key) == created_fields.end())
          return -1;
      }
      // If we get here then we can make a new requirement
      // which has non-returnable privileges
      // Get the top level region for the region tree
      RegionNode *top = runtime->forest->get_tree(req.parent.get_tree_id());
      RegionRequirement new_req(top->handle, READ_WRITE, EXCLUSIVE,top->handle);
      created_requirements.push_back(new_req);
      // Add our fields
      created_requirements.back().privilege_fields.insert(
          req.privilege_fields.begin(), req.privilege_fields.end());
      // This is not a returnable privilege requirement
      returnable_privileges.push_back(false);
      return int(regions.size() + created_requirements.size() - 1);
    }

    //--------------------------------------------------------------------------
    unsigned TaskContext::find_parent_region(unsigned index, TaskOp *child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_PARENT_REGION_CALL);
      // We can check these without the lock
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region == child->regions[index].parent)
          return idx;
      }
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        if (created_requirements[idx].region == child->regions[index].parent)
          return (regions.size() + idx);
      }
      REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                          "Parent task %s (ID %lld) of inline task %s "
                        "(ID %lld) does not have a region "
                        "requirement for region (%x,%x,%x) "
                        "as a parent of child task's region "
                        "requirement index %d", get_task_name(),
                        get_unique_id(), child->get_task_name(),
                        child->get_unique_id(), 
                        child->regions[index].region.index_space.id,
                        child->regions[index].region.field_space.id, 
                        child->regions[index].region.tree_id, index)
      return 0;
    }

    //--------------------------------------------------------------------------
    unsigned TaskContext::find_parent_index_region(unsigned index,TaskOp *child)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < owner_task->indexes.size(); idx++)
      {
        if ((owner_task->indexes[idx].handle == child->indexes[idx].parent))
          return idx;
      }
      REPORT_LEGION_ERROR(ERROR_PARENT_TASK_INLINE,
                          "Parent task %s (ID %lld) of inline task %s "
                            "(ID %lld) does not have an index space "
                            "requirement for index space %x "
                            "as a parent of chlid task's index requirement "
                            "index %d", get_task_name(), get_unique_id(),
                            child->get_task_name(), child->get_unique_id(),
                            child->indexes[index].handle.id, index)
      return 0;
    }

    //--------------------------------------------------------------------------
    PrivilegeMode TaskContext::find_parent_privilege_mode(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx < regions.size())
        return regions[idx].privilege;
      idx -= regions.size();
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(idx < created_requirements.size());
#endif
      return created_requirements[idx].privilege;
    }

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege(
                                         const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
      if (req.verified)
        return NO_ERROR;
      // Find the parent index space
      for (std::vector<IndexSpaceRequirement>::const_iterator it = 
            owner_task->indexes.begin(); it != owner_task->indexes.end(); it++)
      {
        // Check to see if we found the requirement in the parent 
        if (it->handle == req.parent)
        {
          // Check that there is a path between the parent and the child
          std::vector<LegionColor> path;
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
        std::vector<LegionColor> path;
        if (!runtime->forest->compute_index_path(req.parent, req.handle, path))
          return ERROR_BAD_INDEX_PATH;
        // No need to check privileges here since it is a created space
        // which means that the parent has all privileges.
        return NO_ERROR;
      }
      return ERROR_BAD_PARENT_INDEX;
    }

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege(const RegionRequirement &req,
                                                 FieldID &bad_field,
                                                 int &bad_index,
                                                 bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
#ifdef DEBUG_LEGION
      assert(bad_index < 0);
#endif
      if (req.flags & VERIFIED_FLAG)
        return NO_ERROR;
      // Copy privilege fields for check
      std::set<FieldID> privilege_fields(req.privilege_fields);
      unsigned index = 0;
      // Try our original region requirements first
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++, index++)
      {
        LegionErrorType et = 
          check_privilege_internal(req, *it, privilege_fields, bad_field,
                                   index, bad_index, skip_privilege);
        // No error so we are done
        if (et == NO_ERROR)
          return et;
        // Something other than bad parent region is a real error
        if (et != ERROR_BAD_PARENT_REGION)
          return et;
        // Otherwise we just keep going
      }
      // If none of that worked, we now get to try the created requirements
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++, index++)
      {
        const RegionRequirement &created_req = created_requirements[idx];
        LegionErrorType et = 
          check_privilege_internal(req, created_req, privilege_fields, 
                                   bad_field, index, bad_index, skip_privilege);
        // No error so we are done
        if (et == NO_ERROR)
          return et;
        // Something other than bad parent region is a real error
        if (et != ERROR_BAD_PARENT_REGION)
          return et;
        // If we got a BAD_PARENT_REGION, see if this a returnable
        // privilege in which case we know we have privileges on all fields
        if (created_req.privilege_fields.empty())
        {
          // Still have to check the parent region is right
          if (req.parent == created_req.region)
            return NO_ERROR;
        }
        // Otherwise we just keep going
      }
      // Finally see if we created all the fields in which case we know
      // we have privileges on all their regions
      const FieldSpace sp = req.parent.get_field_space();
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        std::pair<FieldSpace,FieldID> key(sp, *it);
        // If we don't find the field, then we are done
        if (created_fields.find(key) == created_fields.end())
          return ERROR_BAD_PARENT_REGION;
      }
      // Check that the parent is the root of the tree, if not it is bad
      RegionNode *parent_region = runtime->forest->get_node(req.parent);
      if (parent_region->parent != NULL)
        return ERROR_BAD_PARENT_REGION;
      // Otherwise we have privileges on these fields for all regions
      // so we are good on privileges
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege_internal(
        const RegionRequirement &req, const RegionRequirement &our_req,
        std::set<FieldID>& privilege_fields, FieldID &bad_field, 
        int local_index, int &bad_index, bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(our_req.handle_type == SINGULAR); // better be singular
#endif
      // Check to see if we found the requirement in the parent
      if (our_req.region == req.parent)
      {
        // If we make it in here then we know we have at least found
        // the parent name so we can set the bad index
        bad_index = local_index;
        bad_field = AUTO_GENERATE_ID; // set it to an invalid field
        if ((req.handle_type == SINGULAR) || 
            (req.handle_type == REG_PROJECTION))
        {
          std::vector<LegionColor> path;
          if (!runtime->forest->compute_index_path(req.parent.index_space,
                                            req.region.index_space, path))
            return ERROR_BAD_REGION_PATH;
        }
        else
        {
          std::vector<LegionColor> path;
          if (!runtime->forest->compute_partition_path(req.parent.index_space,
                                        req.partition.index_partition, path))
            return ERROR_BAD_PARTITION_PATH;
        }
        // Now check that the types are subset of the fields
        // Note we can use the parent since all the regions/partitions
        // in the same region tree have the same field space
        for (std::set<FieldID>::const_iterator fit = 
              privilege_fields.begin(); fit != 
              privilege_fields.end(); )
        {
          if (our_req.privilege_fields.find(*fit) != 
              our_req.privilege_fields.end())
          {
            // Only need to do this check if there were overlapping fields
            if (!skip_privilege && (PRIV_ONLY(req) & (~(our_req.privilege))))
            {
              if ((req.handle_type == SINGULAR) || 
                  (req.handle_type == REG_PROJECTION))
                return ERROR_BAD_REGION_PRIVILEGES;
              else
                return ERROR_BAD_PARTITION_PRIVILEGES;
            }
            privilege_fields.erase(fit++);
          }
          else
            ++fit;
        }
      }

      if (!privilege_fields.empty()) 
      {
        bad_field = *(privilege_fields.begin());
        return ERROR_BAD_PARENT_REGION;
      }
        // If we make it here then we are good
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    LogicalRegion TaskContext::find_logical_region(unsigned index)
    //--------------------------------------------------------------------------
    {
      if (index < regions.size())
        return regions[index].region;
      index -= regions.size();
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(index < created_requirements.size());
#endif
      return created_requirements[index].region;
    } 

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskContext::begin_task(
                                                           Legion::Runtime *&rt)
    //--------------------------------------------------------------------------
    {
      implicit_context = this;
      implicit_runtime = this->runtime;
      rt = this->runtime->external;
      task_profiling_provenance = owner_task->get_unique_op_id();
      if (overhead_tracker != NULL)
        previous_profiling_time = Realm::Clock::current_time_in_nanoseconds();
      // Switch over the executing processor to the one
      // that has actually been assigned to run this task.
      executing_processor = Processor::get_executing_processor();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_task_processor(get_unique_id(), executing_processor.id);
#ifdef DEBUG_LEGION
      log_task.debug("Task %s (ID %lld) starting on processor " IDFMT "",
                    get_task_name(), get_unique_id(), executing_processor.id);
      assert(regions.size() == physical_regions.size());
#endif
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      pending_done = owner_task->get_context()->decrement_pending(owner_task);
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    void TaskContext::begin_misspeculation(void)
    //--------------------------------------------------------------------------
    {
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      pending_done = owner_task->get_context()->decrement_pending(owner_task);
    }

    //--------------------------------------------------------------------------
    void TaskContext::end_misspeculation(const void *result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      // Mark that we are done executing this operation
      owner_task->complete_execution();
      // Grab some information before doing the next step in case it
      // results in the deletion of 'this'
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      const TaskID owner_task_id = owner_task->task_id;
#endif
      Runtime *runtime_ptr = runtime;
      // Call post end task
      post_end_task(result, result_size, false/*owner*/);
#ifdef DEBUG_LEGION
      runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                     false/*meta*/);
#else
      runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    void TaskContext::initialize_overhead_tracker(void)
    //--------------------------------------------------------------------------
    {
      // Make an overhead tracker
#ifdef DEBUG_LEGION
      assert(overhead_tracker == NULL);
#endif
      overhead_tracker = new 
        Mapping::ProfilingMeasurements::RuntimeOverhead();
    }

    //--------------------------------------------------------------------------
    void TaskContext::unmap_all_regions(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalRegion>::const_iterator it = 
            physical_regions.begin(); it != physical_regions.end(); it++)
      {
        if (it->is_mapped())
          it->impl->unmap_region();
      }
      // Also unmap any of our inline mapped physical regions
      for (LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::
            tracked::const_iterator it = inline_regions.begin();
            it != inline_regions.end(); it++)
      {
        if (it->is_mapped())
          it->impl->unmap_region();
      }
    } 

    //--------------------------------------------------------------------------
    void TaskContext::remap_unmapped_regions(LegionTrace *trace,
                            const std::vector<PhysicalRegion> &unmapped_regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unmapped_regions.empty());
#endif
      if ((trace != NULL) && trace->is_static_trace())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RUNTIME_REMAPPING,
          "Illegal runtime remapping in static trace inside of "
                      "task %s (UID %lld). Static traces must perfectly "
                      "manage their physical mappings with no runtime help.",
                      get_task_name(), get_unique_id())
      std::set<ApEvent> mapped_events;
      for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
      {
        const ApEvent ready = remap_region(unmapped_regions[idx]);
        if (ready.exists())
          mapped_events.insert(ready);
      }
      // Wait for all the re-mapping operations to complete
      const ApEvent mapped_event = Runtime::merge_events(mapped_events);
      if (mapped_event.has_triggered())
        return;
      begin_task_wait(true/*from runtime*/);
      mapped_event.wait();
      end_task_wait();
    }

    //--------------------------------------------------------------------------
    void* TaskContext::get_local_task_variable(LocalVariableID id)
    //--------------------------------------------------------------------------
    {
      std::map<LocalVariableID,std::pair<void*,void (*)(void*)> >::
        const_iterator finder = task_local_variables.find(id);
      if (finder == task_local_variables.end())
        REPORT_LEGION_ERROR(ERROR_UNABLE_FIND_TASK_LOCAL,
          "Unable to find task local variable %d in task %s "
                      "(UID %lld)", id, get_task_name(), get_unique_id())  
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    void TaskContext::set_local_task_variable(LocalVariableID id,
                                              const void *value,
                                              void (*destructor)(void*))
    //--------------------------------------------------------------------------
    {
      std::map<LocalVariableID,std::pair<void*,void (*)(void*)> >::iterator
        finder = task_local_variables.find(id);
      if (finder != task_local_variables.end())
      {
        // See if we need to clean things up first
        if (finder->second.second != NULL)
          (*finder->second.second)(finder->second.first);
        finder->second = 
          std::pair<void*,void (*)(void*)>(const_cast<void*>(value),destructor);
      }
      else
        task_local_variables[id] = 
          std::pair<void*,void (*)(void*)>(const_cast<void*>(value),destructor);
    }

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    RtEvent TaskContext::update_previous_mapped_event(RtEvent next)
    //--------------------------------------------------------------------------
    {
      RtEvent result = previous_mapped_event;
      previous_mapped_event = next;
      return result;
    }
#endif
    /////////////////////////////////////////////////////////////
    // Inner Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InnerContext::InnerContext(Runtime *rt, TaskOp *owner, bool full_inner,
                               const std::vector<RegionRequirement> &reqs,
                               const std::vector<unsigned> &parent_indexes,
                               const std::vector<bool> &virt_mapped,
                               UniqueID uid, bool remote)
      : TaskContext(rt, owner, reqs), 
        tree_context(rt->allocate_region_tree_context()), context_uid(uid), 
        remote_context(remote), full_inner_context(full_inner),
        parent_req_indexes(parent_indexes), virtual_mapped(virt_mapped), 
        total_children_count(0), total_close_count(0), total_summary_count(0),
        outstanding_children_count(0), outstanding_prepipeline(0),
        outstanding_dependence(false), outstanding_post_task(0),
        current_trace(NULL),previous_trace(NULL),
        valid_wait_event(false), outstanding_subtasks(0), pending_subtasks(0), 
        pending_frames(0), currently_active_context(false),
        current_mapping_fence(NULL), mapping_fence_gen(0), 
        current_mapping_fence_index(0), current_execution_fence_index(0) 
    //--------------------------------------------------------------------------
    {
      // Set some of the default values for a context
      context_configuration.max_window_size = 
        runtime->initial_task_window_size;
      context_configuration.hysteresis_percentage = 
        runtime->initial_task_window_hysteresis;
      context_configuration.max_outstanding_frames = 0;
      context_configuration.min_tasks_to_schedule = 
        runtime->initial_tasks_to_schedule;
      context_configuration.min_frames_to_schedule = 0;
      context_configuration.meta_task_vector_width = 
        runtime->initial_meta_task_vector_width;
      context_configuration.mutable_priority = false;
#ifdef DEBUG_LEGION
      assert(tree_context.exists());
      runtime->forest->check_context_state(tree_context);
#endif
      // If we have an owner, clone our local fields from its context
      if (owner != NULL)
      {
        TaskContext *owner_ctx = owner_task->get_context();
#ifdef DEBUG_LEGION
        InnerContext *parent_ctx = dynamic_cast<InnerContext*>(owner_ctx);
        assert(parent_ctx != NULL);
#else
        InnerContext *parent_ctx = static_cast<InnerContext*>(owner_ctx);
#endif
        parent_ctx->clone_local_fields(local_fields);
      }
      if (!remote_context)
        runtime->register_local_context(context_uid, this);
    }

    //--------------------------------------------------------------------------
    InnerContext::InnerContext(const InnerContext &rhs)
      : TaskContext(NULL, NULL, rhs.regions), tree_context(rhs.tree_context),
        context_uid(0), remote_context(false), full_inner_context(false),
        parent_req_indexes(rhs.parent_req_indexes), 
        virtual_mapped(rhs.virtual_mapped)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InnerContext::~InnerContext(void)
    //--------------------------------------------------------------------------
    {
      if (!remote_instances.empty())
        free_remote_contexts();
      for (std::map<TraceID,DynamicTrace*>::const_iterator it = traces.begin();
            it != traces.end(); it++)
      {
        if (it->second->remove_reference())
          delete (it->second);
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
      if (valid_wait_event)
      {
        valid_wait_event = false;
        Runtime::trigger_event(window_wait);
      }
      // No need for the lock here since we're being cleaned up
      if (!local_fields.empty())
      {
        for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator 
              it = local_fields.begin(); it != local_fields.end(); it++)
        {
          const std::vector<LocalFieldInfo> &infos = it->second;
          std::vector<FieldID> to_free;
          std::vector<unsigned> indexes;
          for (unsigned idx = 0; idx < infos.size(); idx++)
          {
            if (infos[idx].ancestor)
              continue;
            to_free.push_back(infos[idx].fid); 
            indexes.push_back(infos[idx].index);
          }
          if (!to_free.empty())
            runtime->forest->free_local_fields(it->first, to_free, indexes);
        }
      }
      // Unregister ourselves from any tracking contexts that we might have
      if (!region_tree_owners.empty())
      {
        for (std::map<RegionTreeNode*,
                      std::pair<AddressSpaceID,bool> >::const_iterator it =
              region_tree_owners.begin(); it != region_tree_owners.end(); it++)
          it->first->unregister_tracking_context(this);
        region_tree_owners.clear();
      }
#ifdef DEBUG_LEGION
      assert(pending_top_views.empty());
      assert(outstanding_subtasks == 0);
      assert(pending_subtasks == 0);
      assert(pending_frames == 0);
#endif
      if (!remote_context)
        runtime->unregister_local_context(context_uid);
    }

    //--------------------------------------------------------------------------
    InnerContext& InnerContext::operator=(const InnerContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext InnerContext::get_context(void) const
    //--------------------------------------------------------------------------
    {
      return tree_context;
    }

    //--------------------------------------------------------------------------
    ContextID InnerContext::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
      return tree_context.get_id();
    }

    //--------------------------------------------------------------------------
    UniqueID InnerContext::get_context_uid(void) const
    //--------------------------------------------------------------------------
    {
      return context_uid;
    }

    //--------------------------------------------------------------------------
    int InnerContext::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_depth();
    }

    //--------------------------------------------------------------------------
    bool InnerContext::is_inner_context(void) const
    //--------------------------------------------------------------------------
    {
      return full_inner_context;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID InnerContext::get_version_owner(RegionTreeNode *node,
                                                   AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock tree_lock(tree_owner_lock); 
      // See if we've already assigned it
      std::map<RegionTreeNode*,std::pair<AddressSpaceID,bool> >::iterator
        finder = region_tree_owners.find(node);
      // If we already assigned it then we are done
      if (finder != region_tree_owners.end())
      {
        // If it is remote only, see if it gets to stay that way
        if (finder->second.second && (source == runtime->address_space))
          finder->second.second = false; // no longer remote only
        return finder->second.first;
      }
      // Otherwise assign it to the source
      region_tree_owners[node] = 
        std::pair<AddressSpaceID,bool>(source, 
                                      (source != runtime->address_space));
      node->register_tracking_context(this);
      return source;
    } 

    //--------------------------------------------------------------------------
    void InnerContext::notify_region_tree_node_deletion(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      AutoLock tree_lock(tree_owner_lock);
#ifdef DEBUG_LEGION
      assert(region_tree_owners.find(node) != region_tree_owners.end());
#endif
      region_tree_owners.erase(node);
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      // If this is one of our original region requirements then
      // we can do the analysis in our original context
      const size_t owner_size = regions.size();
      if (index < owner_size)
        return this;
      // Otherwise we need to see if this going to be one of our
      // region requirements that returns privileges or not. If
      // it is then we do the analysis in the outermost context
      // otherwise we do it locally in our own context. We need
      // to hold the operation lock to look at this data structure.
      index -= owner_size;
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(index < returnable_privileges.size());
#endif
      if (returnable_privileges[index])
        return find_outermost_local_context();
      return this;
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_physical_context(unsigned index,
                                                          LogicalRegion *handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == parent_req_indexes.size());
#endif     
      const unsigned owner_size = virtual_mapped.size();
      if (index < owner_size)
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
                                            parent_req_indexes[index], handle);
        else // We mapped a physical instance so we're it
        {
          if (handle != NULL)
            *handle = regions[index].region;
          return this;
        }
      }
      else // We created it
      {
        // Check to see if this has returnable privileges or not
        // If they are not returnable, then we can just be the 
        // context for the handling the meta-data management, 
        // otherwise if they are returnable then the top-level
        // context has to provide global guidance about which
        // node manages the meta-data.
        index -= owner_size;
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        if ((index >= returnable_privileges.size()) || 
            returnable_privileges[index])
          return find_top_context();
        else
        {
          if (handle != NULL)
            *handle = created_requirements[index].region;
          return this;
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_parent_version_info(unsigned index, unsigned depth,
                     const FieldMask &version_mask, InnerContext *context,
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      assert(regions.size() == virtual_mapped.size()); 
#endif
      // If this isn't one of our original region requirements then 
      // we don't have any versions that the child won't discover itself
      // Same if the region was not virtually mapped
      if ((index >= virtual_mapped.size()) || !virtual_mapped[index])
        return;
      // We now need to clone any version info from the parent into the child
      const VersionInfo &parent_info = owner_task->get_version_info(index);  
      parent_info.clone_to_depth(depth, version_mask, context, 
                                 version_info, ready_events);
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_outermost_local_context(
                                                         InnerContext *previous)
    //--------------------------------------------------------------------------
    {
      TaskContext *parent = find_parent_context();
      if (parent != NULL)
        return parent->find_outermost_local_context(this);
#ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_top_context(void)
    //--------------------------------------------------------------------------
    {
      return find_parent_context()->find_top_context();
    }

    //--------------------------------------------------------------------------
    void InnerContext::pack_remote_context(Serializer &rez, 
                                           AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_REMOTE_CONTEXT_CALL);
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
#endif
      rez.serialize<bool>(false); // not the top-level context
      int depth = get_depth();
      rez.serialize(depth);
      // See if we need to pack up base task information
      owner_task->pack_external_task(rez, target);
#ifdef DEBUG_LEGION
      assert(regions.size() == parent_req_indexes.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
        rez.serialize(parent_req_indexes[idx]);
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
      // Pack up the version numbers only 
      const std::vector<VersionInfo> *version_infos = 
        owner_task->get_version_infos();
#ifdef DEBUG_LEGION
      assert(version_infos->size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const VersionInfo &info = (*version_infos)[idx];
        // If we're virtually mapped, we need all the information
        if (virtual_mapped[idx])
          info.pack_version_info(rez);
        else
          info.pack_version_numbers(rez);
      }
      rez.serialize(owner_task->get_task_completion());
      rez.serialize(find_parent_context()->get_context_uid());
      // Finally pack the local field infos
      AutoLock local_lock(local_field_lock,1,false/*exclusive*/);
      rez.serialize<size_t>(local_fields.size());
      for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator 
            it = local_fields.begin(); it != local_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize<size_t>(it->second.size());
        for (unsigned idx = 0; idx < it->second.size(); idx++)
          rez.serialize(it->second[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::unpack_remote_context(Deserializer &derez,
                                           std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      assert(false); // should only be called for RemoteTask
    }

    //--------------------------------------------------------------------------
    void InnerContext::send_back_created_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (created_requirements.empty())
        return;
#ifdef DEBUG_LEGION
      assert(created_requirements.size() == returnable_privileges.size());
#endif
      UniqueID target_context_uid = find_parent_context()->get_context_uid();
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        // Skip anything that doesn't have returnable privileges
        if (!returnable_privileges[idx])
          continue;
        const RegionRequirement &req = created_requirements[idx];
        // If it was deleted then we don't care
        if (was_created_requirement_deleted(req))
          continue;
        runtime->forest->send_back_logical_state(tree_context, 
                        target_context_uid, req, target);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(RegionTreeForest *forest,
                                         const void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id(), type_tag);
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task%s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id()); 
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);
      forest->create_index_space(handle, realm_is, did); 
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::union_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      AutoRuntimeCall call(this); 
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id(), 
                        spaces[0].get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task%s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id()); 
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);
      forest->create_union_space(handle, owner_task, spaces, did); 
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::intersect_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      AutoRuntimeCall call(this); 
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id(), 
                        spaces[0].get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task%s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id()); 
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);
      forest->create_intersection_space(handle, owner_task, spaces, did);
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::subtract_index_spaces(RegionTreeForest *forest,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id(), 
                        left.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task%s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id()); 
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);
      forest->create_difference_space(handle, owner_task, left, right, did);
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_index_space_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_partition(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index partition %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_index_part_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_equal_partition(
                                                      RegionTreeForest *forest,
                                                      IndexSpace parent,
                                                      IndexSpace color_space,
                                                      size_t granularity,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);  
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating equal partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_equal_partition(this, pid, granularity);
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space,
                      partition_color, DISJOINT_COMPLETE_KIND, did, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_union(
                                          RegionTreeForest *forest,
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          IndexSpace color_space,
                                          PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating union partition %d with parent index "
                      "space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_tree_id() != handle1.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create "
                        "partition by union!", handle1.id, parent.id)
      if (parent.get_tree_id() != handle2.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create "
                        "partition by union!", handle2.id, parent.id)
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_union_partition(this, pid, handle1, handle2);
      ApEvent term_event = part_op->get_completion_event();
      // If either partition is aliased the result is aliased
      if ((kind == COMPUTE_KIND) || (kind == COMPUTE_COMPLETE_KIND) ||
          (kind == COMPUTE_INCOMPLETE_KIND))
      {
        // If one of these partitions is aliased then the result is aliased
        IndexPartNode *p1 = forest->get_node(handle1);
        if (p1->is_disjoint(true/*from app*/))
        {
          IndexPartNode *p2 = forest->get_node(handle2);
          if (!p2->is_disjoint(true/*from app*/))
          {
            if (kind == COMPUTE_KIND)
              kind = ALIASED_KIND;
            else if (kind == COMPUTE_COMPLETE_KIND)
              kind = ALIASED_COMPLETE_KIND;
            else
              kind = ALIASED_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == COMPUTE_KIND)
            kind = ALIASED_KIND;
          else if (kind == COMPUTE_COMPLETE_KIND)
            kind = ALIASED_COMPLETE_KIND;
          else
            kind = ALIASED_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space, 
                                        partition_color, kind, did, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
                                              RegionTreeForest *forest,
                                              IndexSpace parent,
                                              IndexPartition handle1,
                                              IndexPartition handle2,
                                              IndexSpace color_space,
                                              PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating intersection partition %d with parent "
                      "index space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_tree_id() != handle1.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create partition by "
                        "intersection!", handle1.id, parent.id)
      if (parent.get_tree_id() != handle2.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create partition by "
                        "intersection!", handle2.id, parent.id)
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_intersection_partition(this, pid, handle1, handle2);
      ApEvent term_event = part_op->get_completion_event();
      // If either partition is disjoint then the result is disjoint
      if ((kind == COMPUTE_KIND) || (kind == COMPUTE_COMPLETE_KIND) ||
          (kind == COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p1 = forest->get_node(handle1);
        if (!p1->is_disjoint(true/*from app*/))
        {
          IndexPartNode *p2 = forest->get_node(handle2);
          if (p2->is_disjoint(true/*from app*/))
          {
            if (kind == COMPUTE_KIND)
              kind = DISJOINT_KIND;
            else if (kind == COMPUTE_COMPLETE_KIND)
              kind = DISJOINT_COMPLETE_KIND;
            else
              kind = DISJOINT_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == COMPUTE_KIND)
            kind = DISJOINT_KIND;
          else if (kind == COMPUTE_COMPLETE_KIND)
            kind = DISJOINT_COMPLETE_KIND;
          else
            kind = DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space,
                                        partition_color, kind, did, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_difference(
                                                  RegionTreeForest *forest,
                                                  IndexSpace parent,
                                                  IndexPartition handle1,
                                                  IndexPartition handle2,
                                                  IndexSpace color_space,
                                                  PartitionKind kind, 
                                                  Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating difference partition %d with parent "
                      "index space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_tree_id() != handle1.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by difference!",
                              handle1.id, parent.id)
      if (parent.get_tree_id() != handle2.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                              "index tree as IndexSpace %d in create "
                              "partition by difference!",
                              handle2.id, parent.id)
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_difference_partition(this, pid, handle1, handle2);
      ApEvent term_event = part_op->get_completion_event();
      // If the left-hand-side is disjoint the result is disjoint
      if ((kind == COMPUTE_KIND) || (kind == COMPUTE_COMPLETE_KIND) ||
          (kind == COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p1 = forest->get_node(handle1);
        if (p1->is_disjoint(true/*from app*/))
        {
          if (kind == COMPUTE_KIND)
            kind = DISJOINT_KIND;
          else if (kind == COMPUTE_COMPLETE_KIND)
            kind = DISJOINT_COMPLETE_KIND;
          else
            kind = DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space, 
                                        partition_color, kind, did, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    Color InnerContext::create_cross_product_partitions(
                                                      RegionTreeForest *forest,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                   std::map<IndexSpace,IndexPartition> &handles,
                                                      PartitionKind kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating cross product partitions in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
      if (handle1.get_tree_id() != handle2.get_tree_id())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
          "IndexPartition %d is not part of the same "
                              "index tree as IndexPartition %d in create "
                              "cross product partitions!",
                              handle1.id, handle2.id)
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_cross_product(handle1, handle2, handles, kind, 
                                           partition_color, term_event);
      part_op->initialize_cross_product(this, handle1, handle2,partition_color);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return partition_color;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_association(LogicalRegion domain,
                                          LogicalRegion domain_parent,
                                          FieldID domain_fid,
                                          IndexSpace range,
                                          MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating association in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      part_op->initialize_by_association(this, domain, domain_parent, 
                                         domain_fid, range, id, tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_association call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_restricted_partition(
                                              RegionTreeForest *forest,
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
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating restricted partition in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color; 
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_restricted_partition(this, pid, transform, 
                                transform_size, extent, extent_size);
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space,
                                        part_color, part_kind, did, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_field(
                                              RegionTreeForest *forest,
                                              LogicalRegion handle,
                                              LogicalRegion parent_priv,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              Color color,
                                              MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexSpace parent = handle.get_index_space(); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by field in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition 
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space,
                                    part_color, DISJOINT_KIND, did, term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_field(this, pid, handle, parent_priv, fid, id,tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_partition_by_field call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image(
                                                    RegionTreeForest *forest,
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
      AutoRuntimeCall call(this); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_tree_id(), handle.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by image in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event(); 
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, handle, color_space,
                                        part_color, part_kind, did, term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image(this, pid, projection, parent, fid, id, tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_partition_by_image call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image_range(
                                                    RegionTreeForest *forest,
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
      AutoRuntimeCall call(this); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_tree_id(), handle.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by image range in task %s (ID %lld)",
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, handle, color_space,
                                        part_color, part_kind, did, term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image_range(this, pid, projection, parent, 
                                         fid, id, tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_partition_by_image_range call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage(
                                                  RegionTreeForest *forest,
                                                  IndexPartition projection,
                                                  LogicalRegion handle,
                                                  LogicalRegion parent,
                                                  FieldID fid,
                                                  IndexSpace color_space,
                                                  PartitionKind part_kind,
                                                  Color color,
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id(),
                         parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by preimage in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(); 
      ApEvent term_event = part_op->get_completion_event();
      // If the source of the preimage is disjoint then the result is disjoint
      // Note this only applies here and not to range
      if ((part_kind == COMPUTE_KIND) || (part_kind == COMPUTE_COMPLETE_KIND) ||
          (part_kind == COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p = forest->get_node(projection);
        if (p->is_disjoint(true/*from app*/))
        {
          if (part_kind == COMPUTE_KIND)
            part_kind = DISJOINT_KIND;
          else if (part_kind == COMPUTE_COMPLETE_KIND)
            part_kind = DISJOINT_COMPLETE_KIND;
          else
            part_kind = DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, 
                                       handle.get_index_space(), color_space, 
                                       part_color, part_kind, did, term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage(this, pid, projection, handle, 
                                      parent, fid, id, tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_partition_by_preimage call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage_range(
                                                  RegionTreeForest *forest,
                                                  IndexPartition projection,
                                                  LogicalRegion handle,
                                                  LogicalRegion parent,
                                                  FieldID fid,
                                                  IndexSpace color_space,
                                                  PartitionKind part_kind,
                                                  Color color,
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id(),
                         parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by preimage range in task %s "
                      "(ID %lld)", get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(); 
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = forest->create_pending_partition(pid, 
                                       handle.get_index_space(), color_space, 
                                       part_color, part_kind, did, term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage_range(this, pid, projection, handle,
                                            parent, fid, id, tag);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around create_partition_by_preimage_range call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_pending_partition(
                                                RegionTreeForest *forest,
                                                IndexSpace parent,
                                                IndexSpace color_space, 
                                                PartitionKind part_kind,
                                                Color color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating pending partition in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != AUTO_GENERATE_ID)
        part_color = color;
      ApUserEvent partition_ready = Runtime::create_ap_user_event();
      RtEvent safe = forest->create_pending_partition(pid, parent, color_space, 
                 part_color, part_kind, did, partition_ready, partition_ready);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = forest->find_pending_space(parent, realm_color, 
                                                     type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_union(this, result, handles);
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = forest->find_pending_space(parent, realm_color, 
                                                     type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_union(this, result, handle);
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = forest->find_pending_space(parent, realm_color, 
                                                     type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(this, result, handles);
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = forest->find_pending_space(parent, realm_color, 
                                                     type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(this, result, handle);
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_difference(
                                                    RegionTreeForest *forest,
                                                    IndexPartition parent,
                                                    const void *realm_color,
                                                    TypeTag type_tag,
                                                    IndexSpace initial,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space difference in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = forest->find_pending_space(parent, realm_color, 
                                                     type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_difference(this, result, initial,handles);
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpace InnerContext::create_field_space(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FieldSpace space(runtime->get_unique_field_space_id());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_field.debug("Creating field space %x in task %s (ID %lld)", 
                      space.id, get_task_name(), get_unique_id());
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_field_space(space.id);

      forest->create_field_space(space, did);
      register_field_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_field_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_field.debug("Destroying field space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_field_space_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    FieldID InnerContext::allocate_field(RegionTreeForest *forest,
                                         FieldSpace space, size_t field_size,
                                         FieldID fid, bool local,
                                         CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (fid == AUTO_GENERATE_ID)
        fid = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
      else if (fid >= MAX_APPLICATION_FIELD_ID)
      {
        REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FILED,
          "Task %s (ID %lld) attempted to allocate a field with "
                       "ID %d which exceeds the MAX_APPLICATION_FIELD_ID bound "
                       "set in legion_config.h", get_task_name(),
                       get_unique_id(), fid);
        assert(false);
      }
#endif

      if (runtime->legion_spy_enabled)
        LegionSpy::log_field_creation(space.id, fid, field_size);

      std::set<RtEvent> done_events;
      if (local)
      {
        // See if we've exceeded our local field allocations 
        // for this field space
        AutoLock local_lock(local_field_lock);
        std::vector<LocalFieldInfo> &infos = local_fields[space];
        if (infos.size() == runtime->max_local_fields)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_LOCAL_FIELDS,
            "Exceeded maximum number of local fields in "
                        "context of task %s (UID %lld). The maximum "
                        "is currently set to %d, but can be modified "
                        "with the -lg:local flag.", get_task_name(),
                        get_unique_id(), runtime->max_local_fields)
        std::set<unsigned> current_indexes;
        for (std::vector<LocalFieldInfo>::const_iterator it = 
              infos.begin(); it != infos.end(); it++)
          current_indexes.insert(it->index);
        std::vector<FieldID> fields(1, fid);
        std::vector<size_t> sizes(1, field_size);
        std::vector<unsigned> new_indexes;
        if (!forest->allocate_local_fields(space, fields, sizes, serdez_id, 
                                           current_indexes, new_indexes))
          REPORT_LEGION_ERROR(ERROR_UNABLE_ALLOCATE_LOCAL_FIELD,
            "Unable to allocate local field in context of "
                        "task %s (UID %lld) due to local field size "
                        "fragmentation. This situation can be improved "
                        "by increasing the maximum number of permitted "
                        "local fields in a context with the -lg:local "
                        "flag.", get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
        assert(new_indexes.size() == 1);
#endif
        // Only need the lock here when modifying since all writes
        // to this data structure are serialized
        infos.push_back(LocalFieldInfo(fid, field_size, serdez_id, 
                                       new_indexes[0], false));
        AutoLock rem_lock(remote_lock,1,false/*exclusive*/);
        // Have to send notifications to any remote nodes
        for (std::map<AddressSpaceID,RemoteContext*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->second);
            rez.serialize<size_t>(1); // field space count
            rez.serialize(space);
            rez.serialize<size_t>(1); // field count
            rez.serialize(infos.back());
            rez.serialize(done_event);
          }
          runtime->send_local_field_update(it->first, rez);
          done_events.insert(done_event);
        }
      }
      else
        forest->allocate_field(space, field_size, fid, serdez_id);
      register_field_creation(space, fid, local);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_field_deletion(this, space, fid);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_fields(RegionTreeForest *forest, 
                                       FieldSpace space,
                                       const std::vector<size_t> &sizes,
                                       std::vector<FieldID> &resulting_fields,
                                       bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
        else if (resulting_fields[idx] >= MAX_APPLICATION_FIELD_ID)
        {
          REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD,
            "Task %s (ID %lld) attempted to allocate a field with "
                         "ID %d which exceeds the MAX_APPLICATION_FIELD_ID "
                         "bound set in legion_config.h", get_task_name(),
                         get_unique_id(), resulting_fields[idx]);
          assert(false);
        }
#endif

        if (runtime->legion_spy_enabled)
          LegionSpy::log_field_creation(space.id, 
                                        resulting_fields[idx], sizes[idx]);
      }
      std::set<RtEvent> done_events;
      if (local)
      {
        // See if we've exceeded our local field allocations 
        // for this field space
        AutoLock local_lock(local_field_lock);
        std::vector<LocalFieldInfo> &infos = local_fields[space];
        if ((infos.size() + sizes.size()) > runtime->max_local_fields)
          REPORT_LEGION_ERROR(ERROR_EXCEEDED_MAXIMUM_NUMBER_LOCAL_FIELDS,
            "Exceeded maximum number of local fields in "
                        "context of task %s (UID %lld). The maximum "
                        "is currently set to %d, but can be modified "
                        "with the -lg:local flag.", get_task_name(),
                        get_unique_id(), runtime->max_local_fields)
        std::set<unsigned> current_indexes;
        for (std::vector<LocalFieldInfo>::const_iterator it = 
              infos.begin(); it != infos.end(); it++)
          current_indexes.insert(it->index);
        std::vector<unsigned> new_indexes;
        if (!forest->allocate_local_fields(space, resulting_fields, sizes, 
                                  serdez_id, current_indexes, new_indexes))
          REPORT_LEGION_ERROR(ERROR_UNABLE_ALLOCATE_LOCAL_FIELD,
            "Unable to allocate local field in context of "
                        "task %s (UID %lld) due to local field size "
                        "fragmentation. This situation can be improved "
                        "by increasing the maximum number of permitted "
                        "local fields in a context with the -lg:local "
                        "flag.", get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
        assert(new_indexes.size() == resulting_fields.size());
#endif
        // Only need the lock here when writing since we know all writes
        // are serialized and we only need to worry about interfering readers
        const unsigned offset = infos.size();
        for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
          infos.push_back(LocalFieldInfo(resulting_fields[idx], 
                     sizes[idx], serdez_id, new_indexes[idx], false));
        // Have to send notifications to any remote nodes 
        AutoLock rem_lock(remote_lock,1,false/*exclusive*/);
        for (std::map<AddressSpaceID,RemoteContext*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          RtUserEvent done_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->second);
            rez.serialize<size_t>(1); // field space count
            rez.serialize(space);
            rez.serialize<size_t>(resulting_fields.size()); // field count
            for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
              rez.serialize(infos[offset+idx]);
            rez.serialize(done_event);
          }
          runtime->send_local_field_update(it->first, rez);
          done_events.insert(done_event);
        }
      }
      else
        forest->allocate_fields(space, sizes, resulting_fields, serdez_id);
      register_field_creations(space, local, resulting_fields);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_fields(FieldSpace space, 
                                   const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_field_deletions(this, space, to_free);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::create_logical_region(RegionTreeForest *forest,
                                                      IndexSpace index_space,
                                                      FieldSpace field_space,
                                                      bool task_local)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      RegionTreeID tid = runtime->get_unique_region_tree_id();
      LogicalRegion region(tid, index_space, field_space);
#ifdef DEBUG_LEGION
      log_region.debug("Creating logical region in task %s (ID %lld) with "
                       "index space %x and field space %x in new tree %d",
                       get_task_name(), get_unique_id(), 
                       index_space.id, field_space.id, tid);
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_region(index_space.id, field_space.id, tid);

      forest->create_logical_region(region);
      // Register the creation of a top-level region with the context
      register_region_creation(region, task_local);
      if (task_local)
      {
#ifdef DEBUG_LEGION
        assert(local_regions.find(region) == local_regions.end());
#endif
        local_regions.insert(region);
      }
      return region;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_region.debug("Deleting logical region (%x,%x) in task %s (ID %lld)",
                       handle.index_space.id, handle.field_space.id, 
                       get_task_name(), get_unique_id());
#endif
      // If this is a local region remove it from our set
      std::set<LogicalRegion>::iterator finder = local_regions.find(handle);
      if (finder != local_regions.end())
        local_regions.erase(finder);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_logical_region_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_logical_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_region.debug("Deleting logical partition (%x,%x) in task %s "
                       "(ID %lld)", handle.index_partition.id, 
                       handle.field_space.id, get_task_name(), get_unique_id());
#endif
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_logical_partition_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    FieldAllocator InnerContext::create_field_allocator(
                                   Legion::Runtime *external, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      return FieldAllocator(handle, this, external);
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
          return launcher.predicate_false_future;
        // Otherwise check to see if we have a value
        FutureImpl *result = new FutureImpl(runtime, true/*register*/,
          runtime->get_available_distributed_id(), runtime->address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskImpl *impl = runtime->find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
            REPORT_LEGION_ERROR(ERROR_PREDICATED_TASK_LAUNCH_FOR_TASK,
              "Predicated task launch for task %s in parent "
                          "task %s (UID %lld) has non-void return type "
                          "but no default value for its future if the task "
                          "predicate evaluates to false.  Please set either "
                          "the 'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "TaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id())
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      IndividualTask *task = runtime->get_available_individual_task();
#ifdef DEBUG_LEGION
      Future result = 
        task->initialize_task(this, launcher, runtime->check_privileges);
      log_task.debug("Registering new single task with unique id %lld "
                      "and task %s (ID %lld) with high level runtime in "
                      "addresss space %d",
                      task->get_unique_id(), task->get_task_name(), 
                      task->get_unique_id(), runtime->address_space);
#else
      Future result = task->initialize_task(this, launcher,
                                            false/*check privileges*/);
#endif
      execute_task_launch(task, false/*index*/, current_trace, 
                          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_index_space(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        FutureMap result = execute_must_epoch(epoch_launcher);
        return result;
      }
      AutoRuntimeCall call(this);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        FutureMapImpl *result = new FutureMapImpl(this, runtime,
            runtime->get_available_distributed_id(),
            runtime->address_space);
        if (launcher.predicate_false_future.impl != NULL)
        {
          ApEvent ready_event = 
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
            result->add_base_gc_ref(DEFERRED_TASK_REF);
            launcher.predicate_false_future.impl->add_base_gc_ref(
                                                FUTURE_HANDLE_REF);
            TaskOp::DeferredFutureMapSetArgs args(result,
                launcher.predicate_false_future.impl, 
                launcher.launch_domain, owner_task);
            runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                           Runtime::protect_event(ready_event));
          }
          return FutureMap(result);
        }
        if (launcher.predicate_false_result.get_size() == 0)
        {
          // Check to make sure the task actually does expect to
          // have a void return type
          TaskImpl *impl = runtime->find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
            REPORT_LEGION_ERROR(ERROR_PREDICATED_INDEX_TASK_LAUNCH,
              "Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "IndexTaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id())
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
      if (launcher.launch_domain.exists() && 
          (launcher.launch_domain.get_volume() == 0))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_EMPTY_INDEX_TASK_LAUNCH,
          "Ignoring empty index task launch in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        return FutureMap();
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space = find_index_launch_space(launcher.launch_domain);
      IndexTask *task = runtime->get_available_index_task();
#ifdef DEBUG_LEGION
      FutureMap result = 
        task->initialize_task(this, launcher, launch_space,
                              runtime->check_privileges);
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#else
      FutureMap result = task->initialize_task(this, launcher, launch_space,
                                               false/*check privileges*/);
#endif
      execute_task_launch(task, true/*index*/, current_trace, 
                          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_index_space(const IndexTaskLauncher &launcher,
                                             ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
        assert(false); // TODO: add support for this
      AutoRuntimeCall call(this);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
          return launcher.predicate_false_future;
        // Otherwise check to see if we have a value
        FutureImpl *result = new FutureImpl(runtime, true/*register*/, 
          runtime->get_available_distributed_id(), runtime->address_space);
        if (launcher.predicate_false_result.get_size() > 0)
          result->set_result(launcher.predicate_false_result.get_ptr(),
                             launcher.predicate_false_result.get_size(),
                             false/*own*/);
        else
        {
          // We need to check to make sure that the task actually
          // does expect to have a void return type
          TaskImpl *impl = runtime->find_or_create_task_impl(launcher.task_id);
          if (impl->returns_value())
            REPORT_LEGION_ERROR(ERROR_PREDICATED_INDEX_TASK_LAUNCH,
              "Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "IndexTaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id())
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_EMPTY_INDEX_TASK_LAUNCH,
          "Ignoring empty index task launch in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        return Future();
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space = find_index_launch_space(launcher.launch_domain);
      IndexTask *task = runtime->get_available_index_task();
#ifdef DEBUG_LEGION
      Future result = 
        task->initialize_task(this, launcher, launch_space, redop, 
                              runtime->check_privileges);
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#else
      Future result = task->initialize_task(this, launcher, launch_space, redop,
                                            false/*check privileges*/);
#endif
      execute_task_launch(task, true/*index*/, current_trace, 
                          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::map_region(const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (IS_NO_ACCESS(launcher.requirement))
        return PhysicalRegion();
      MapOp *map_op = runtime->get_available_map_op();
#ifdef DEBUG_LEGION
      PhysicalRegion result = 
        map_op->initialize(this, launcher, runtime->check_privileges);
      log_run.debug("Registering a map operation for region "
                    "(%x,%x,%x) in task %s (ID %lld)",
                    launcher.requirement.region.index_space.id, 
                    launcher.requirement.region.field_space.id, 
                    launcher.requirement.region.tree_id, 
                    get_task_name(), get_unique_id());
#else
      PhysicalRegion result = map_op->initialize(this, launcher, 
                                                 false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;  
      const int index = 
        has_conflicting_regions(map_op, parent_conflict, inline_conflict);
      if (parent_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_INLINE_MAPPING_REGION,
          "Attempted an inline mapping of region "
                      "(%x,%x,%x) that conflicts with mapped region " 
                      "(%x,%x,%x) at index %d of parent task %s "
                      "(ID %lld) that would ultimately result in "
                      "deadlock. Instead you receive this error message.",
                      launcher.requirement.region.index_space.id,
                      launcher.requirement.region.field_space.id,
                      launcher.requirement.region.tree_id,
                      regions[index].region.index_space.id,
                      regions[index].region.field_space.id,
                      regions[index].region.tree_id,
                      index, get_task_name(), get_unique_id())
      if (inline_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_INLINE_MAPPING_REGION,
          "Attempted an inline mapping of region (%x,%x,%x) "
                      "that conflicts with previous inline mapping in "
                      "task %s (ID %lld) that would ultimately result in "
                      "deadlock.  Instead you receive this error message.",
                      launcher.requirement.region.index_space.id,
                      launcher.requirement.region.field_space.id,
                      launcher.requirement.region.tree_id,
                      get_task_name(), get_unique_id())
      register_inline_mapped_region(result);
      runtime->add_to_dependence_queue(this, executing_processor, map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.is_mapped())
        return ApEvent::NO_AP_EVENT;
      MapOp *map_op = runtime->get_available_map_op();
      map_op->initialize(this, region);
      register_inline_mapped_region(region);
      const ApEvent result = map_op->get_completion_event();
      runtime->add_to_dependence_queue(this, executing_processor, map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!region.is_mapped())
        return;
      unregister_inline_mapped_region(region);
      region.impl->unmap_region();
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FillOp *fill_op = runtime->get_available_fill_op();
#ifdef DEBUG_LEGION
      fill_op->initialize(this, launcher, runtime->check_privileges);
      log_run.debug("Registering a fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
#else
      fill_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "WARNING: Runtime is unmapping and remapping "
              "physical regions around fill_fields call in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.launch_domain.exists() && 
          (launcher.launch_domain.get_volume() == 0))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_EMPTY_INDEX_SPACE_FILL,
          "Ignoring empty index space fill in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space = find_index_launch_space(launcher.launch_domain);
      IndexFillOp *fill_op = runtime->get_available_index_fill_op();
#ifdef DEBUG_LEGION
      fill_op->initialize(this, launcher, launch_space, 
                          runtime->check_privileges);
      log_run.debug("Registering an index fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
#else
      fill_op->initialize(this, launcher, launch_space,
                          false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around fill_fields call in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_copy(const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      CopyOp *copy_op = runtime->get_available_copy_op();
#ifdef DEBUG_LEGION
      copy_op->initialize(this, launcher, runtime->check_privileges);
      log_run.debug("Registering a copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#else
      copy_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, copy_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_copy(const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_EMPTY_INDEX_SPACE_COPY,
          "Ignoring empty index space copy in task %s "
                        "(ID %lld)", get_task_name(), get_unique_id());
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space = find_index_launch_space(launcher.launch_domain);
      IndexCopyOp *copy_op = runtime->get_available_index_copy_op();
#ifdef DEBUG_LEGION
      copy_op->initialize(this, launcher, launch_space, 
                          runtime->check_privileges);
      log_run.debug("Registering an index copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#else
      copy_op->initialize(this, launcher, launch_space, 
                          false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, copy_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_acquire(const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      AcquireOp *acquire_op = runtime->get_available_acquire_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing an acquire operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      acquire_op->initialize(this, launcher, runtime->check_privileges);
#else
      acquire_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this acquire operation.
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(acquire_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around issue_acquire call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the acquire operation
      runtime->add_to_dependence_queue(this, executing_processor, acquire_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_release(const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      ReleaseOp *release_op = runtime->get_available_release_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a release operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      release_op->initialize(this, launcher, runtime->check_privileges);
#else
      release_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue the release operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(release_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around issue_release call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the release operation
      runtime->add_to_dependence_queue(this, executing_processor, release_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::attach_resource(const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      AttachOp *attach_op = runtime->get_available_attach_op();
#ifdef DEBUG_LEGION
      PhysicalRegion result = 
        attach_op->initialize(this, launcher, runtime->check_privileges);
#else
      PhysicalRegion result = 
        attach_op->initialize(this, launcher, false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;
      int index = has_conflicting_regions(attach_op, 
                                          parent_conflict, inline_conflict);
      if (parent_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_ATTACH_HDF5,
          "Attempted an attach hdf5 file operation on region "
                      "(%x,%x,%x) that conflicts with mapped region " 
                      "(%x,%x,%x) at index %d of parent task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_hdf5 on file %s",
                      launcher.handle.index_space.id, 
                      launcher.handle.field_space.id, 
                      launcher.handle.tree_id, 
                      regions[index].region.index_space.id,
                      regions[index].region.field_space.id,
                      regions[index].region.tree_id, index, 
                      get_task_name(), get_unique_id(), launcher.file_name)
      if (inline_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_ATTACH_HDF5,
          "Attempted an attach hdf5 file operation on region "
                      "(%x,%x,%x) that conflicts with previous inline "
                      "mapping in task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking attach_hdf5 on file %s",
                      launcher.handle.index_space.id, 
                      launcher.handle.field_space.id, 
                      launcher.handle.tree_id, get_task_name(), 
                      get_unique_id(), launcher.file_name)
      runtime->add_to_dependence_queue(this, executing_processor, attach_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DetachOp *detach_op = runtime->get_available_detach_op();
      Future result = detach_op->initialize_detach(this, region);
      // If the region is still mapped, then unmap it
      if (region.is_mapped())
      {
        unregister_inline_mapped_region(region);
        region.impl->unmap_region();
      }
      runtime->add_to_dependence_queue(this, executing_processor, detach_op);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_must_epoch(
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      MustEpochOp *epoch_op = runtime->get_available_epoch_op();
#ifdef DEBUG_LEGION
      log_run.debug("Executing a must epoch in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      FutureMap result = 
        epoch_op->initialize(this, launcher, runtime->check_privileges);
#else
      FutureMap result = epoch_op->initialize(this, launcher, 
                                              false/*check privileges*/);
#endif
      // Now find all the parent task regions we need to invalidate
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        epoch_op->find_conflicted_regions(unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
            "Runtime is unmapping and remapping "
              "physical regions around issue_release call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Now we can issue the must epoch
      runtime->add_to_dependence_queue(this, executing_processor, epoch_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_timing_measurement(const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a timing measurement in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      TimingOp *timing_op = runtime->get_available_timing_op();
      Future result = timing_op->initialize(this, launcher);
      runtime->add_to_dependence_queue(this, executing_processor, timing_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_mapping_fence(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FenceOp *fence_op = runtime->get_available_fence_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a mapping fence in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      fence_op->initialize(this, FenceOp::MAPPING_FENCE);
      runtime->add_to_dependence_queue(this, executing_processor, fence_op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_execution_fence(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FenceOp *fence_op = runtime->get_available_fence_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing an execution fence in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      fence_op->initialize(this, FenceOp::EXECUTION_FENCE);
      runtime->add_to_dependence_queue(this, executing_processor, fence_op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::complete_frame(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FrameOp *frame_op = runtime->get_available_frame_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a frame in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      frame_op->initialize(this);
      runtime->add_to_dependence_queue(this, executing_processor, frame_op);
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::create_predicate(const Future &f)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (f.impl == NULL)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_PREDICATE_CREATION,
          "Illegal predicate creation performed on "
                      "empty future inside of task %s (ID %lld).",
                      get_task_name(), get_unique_id())
      FuturePredOp *pred_op = runtime->get_available_future_pred_op();
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(this, f);
      runtime->add_to_dependence_queue(this, executing_processor, pred_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::predicate_not(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      NotPredOp *pred_op = runtime->get_available_not_pred_op();
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(this, p);
      runtime->add_to_dependence_queue(this, executing_processor, pred_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::create_predicate(const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.predicates.empty())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_PREDICATE_CREATION,
          "Illegal predicate creation performed on a "
                      "set of empty previous predicates in task %s (ID %lld).",
                      get_task_name(), get_unique_id())
      else if (launcher.predicates.size() == 1)
        return launcher.predicates[0];
      if (launcher.and_op)
      {
        // Check for short circuit cases
        std::vector<Predicate> actual_predicates;
        for (std::vector<Predicate>::const_iterator it = 
              launcher.predicates.begin(); it != 
              launcher.predicates.end(); it++)
        {
          if ((*it) == Predicate::FALSE_PRED)
            return Predicate::FALSE_PRED;
          else if ((*it) == Predicate::TRUE_PRED)
            continue;
          actual_predicates.push_back(*it);
        }
        if (actual_predicates.empty()) // they were all true
          return Predicate::TRUE_PRED;
        else if (actual_predicates.size() == 1)
          return actual_predicates[0];
        AndPredOp *pred_op = runtime->get_available_and_pred_op();
        // Hold a reference before initialization
        Predicate result(pred_op);
        pred_op->initialize(this, actual_predicates);
        runtime->add_to_dependence_queue(this, executing_processor, pred_op);
        return result;
      }
      else
      {
        // Check for short circuit cases
        std::vector<Predicate> actual_predicates;
        for (std::vector<Predicate>::const_iterator it = 
              launcher.predicates.begin(); it != 
              launcher.predicates.end(); it++)
        {
          if ((*it) == Predicate::TRUE_PRED)
            return Predicate::TRUE_PRED;
          else if ((*it) == Predicate::FALSE_PRED)
            continue;
          actual_predicates.push_back(*it);
        }
        if (actual_predicates.empty()) // they were all false
          return Predicate::FALSE_PRED;
        else if (actual_predicates.size() == 1)
          return actual_predicates[0];
        OrPredOp *pred_op = runtime->get_available_or_pred_op();
        // Hold a reference before initialization
        Predicate result(pred_op);
        pred_op->initialize(this, actual_predicates);
        runtime->add_to_dependence_queue(this, executing_processor, pred_op);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::get_predicate_future(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      if (p == Predicate::TRUE_PRED)
      {
        Future result = runtime->help_create_future();
        const bool value = true;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
        result.impl->complete_future();
        return result;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        Future result = runtime->help_create_future();
        const bool value = false;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
        result.impl->complete_future();
        return result;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(p.impl != NULL);
#endif
        return p.impl->get_future_result(); 
      }
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::register_new_child_operation(Operation *op,
                      const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // If we are performing a trace mark that the child has a trace
      if (current_trace != NULL)
        op->set_trace(current_trace, !current_trace->is_fixed(), dependences);
      unsigned result = total_children_count++;
      const unsigned outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,1);
      // Only need to check if we are not tracing by frames
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count > context_configuration.max_window_size))
        perform_window_wait();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_child_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id()); 
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::register_new_close_operation(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      // For now we just bump our counter
      unsigned result = total_close_count++;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_close_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned InnerContext::register_new_summary_operation(TraceSummaryOp *op)
    //--------------------------------------------------------------------------
    {
      // For now we just bump our counter
      unsigned result = total_summary_count++;
      const unsigned outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,1);
      // Only need to check if we are not tracing by frames
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count > context_configuration.max_window_size))
        perform_window_wait();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_child_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id()); 
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_window_wait(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_event;
      // Take the context lock in exclusive mode
      {
        AutoLock child_lock(child_op_lock);
        // We already hold our lock from the callsite above
        // Outstanding children count has already been incremented for the
        // operation being launched so decrement it in case we wait and then
        // re-increment it when we wake up again
        const int outstanding_count = 
          __sync_fetch_and_add(&outstanding_children_count,-1);
        // We already decided to wait, so we need to wait for any hysteresis
        // to play a role here
        if (outstanding_count >
            int((100 - context_configuration.hysteresis_percentage) *
                context_configuration.max_window_size / 100))
        {
#ifdef DEBUG_LEGION
          assert(!valid_wait_event);
#endif
          window_wait = Runtime::create_rt_user_event();
          valid_wait_event = true;
          wait_event = window_wait;
        }
      }
      begin_task_wait(false/*from runtime*/);
      wait_event.wait();
      end_task_wait();
      // Re-increment the count once we are awake again
      __sync_fetch_and_add(&outstanding_children_count,1);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_prepipeline_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      const GenerationID gen = op->get_generation();
      {
        AutoLock p_lock(prepipeline_lock);
        prepipeline_queue.push_back(std::pair<Operation*,GenerationID>(op,gen));
        // No need to have more outstanding tasks than there are processors
        if (outstanding_prepipeline < runtime->num_utility_procs)
        {
          const size_t needed_in_flight = 
            (prepipeline_queue.size() + 
             context_configuration.meta_task_vector_width - 1) / 
              context_configuration.meta_task_vector_width;
          if (outstanding_prepipeline < needed_in_flight)
          {
            outstanding_prepipeline++;
            issue_task = true;
          }
        }
      }
      if (issue_task)
      {
        PrepipelineArgs args(op, this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::process_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<Operation*,GenerationID> > to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      Operation *launch_next_op = NULL;
      {
        AutoLock p_lock(prepipeline_lock);
        for (unsigned idx = 0; idx < 
              context_configuration.meta_task_vector_width; idx++)
        {
          if (prepipeline_queue.empty())
            break;
          to_perform.push_back(prepipeline_queue.front());
          prepipeline_queue.pop_front();
        }
#ifdef DEBUG_LEGION
        assert(outstanding_prepipeline > 0);
        assert(outstanding_prepipeline <= runtime->num_utility_procs);
#endif
        if (!prepipeline_queue.empty())
        {
          const size_t needed_in_flight = 
            (prepipeline_queue.size() + 
             context_configuration.meta_task_vector_width - 1) / 
              context_configuration.meta_task_vector_width;
          if (outstanding_prepipeline <= needed_in_flight)
            launch_next_op = prepipeline_queue.back().first; 
          else
            outstanding_prepipeline--;
        }
        else
          outstanding_prepipeline--;
      }
      // Perform our prepipeline tasks
      for (std::vector<std::pair<Operation*,GenerationID> >::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        it->first->execute_prepipeline_stage(it->second, false/*need wait*/);
      if (launch_next_op != NULL)
      {
        // This could maybe give a bad op ID for profiling, but it
        // will not impact the correctness of the code
        PrepipelineArgs args(launch_next_op, this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_dependence_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY;
      // If this is tracking, add it to our data structure first
      if (op->is_tracking_parent())
      {
        AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
        assert(executing_children.find(op) == executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
        outstanding_children[op->get_ctx_index()] = op;
#endif       
        executing_children[op] = op->get_generation();
        // Bump our priority if the context is not active as it means
        // that the runtime is currently not ahead of execution
        if (!currently_active_context)
          priority = LG_THROUGHPUT_DEFERRED_PRIORITY;
      }
      bool issue_task = false;
      RtEvent precondition;
      {
        AutoLock d_lock(dependence_lock);
        if (!outstanding_dependence)
        {
#ifdef DEBUG_LEGION
          assert(dependence_queue.empty());
#endif
          issue_task = true;
          outstanding_dependence = true;
          precondition = dependence_precondition;
          dependence_precondition = RtEvent::NO_RT_EVENT;
        }
        dependence_queue.push_back(op);
      }
      if (issue_task)
      {
        DependenceArgs args(op, this);
        runtime->issue_runtime_meta_task(args, priority, precondition); 
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::process_dependence_stage(void)
    //--------------------------------------------------------------------------
    {
      std::vector<Operation*> to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      Operation *launch_next_op = NULL;
      {
        AutoLock d_lock(dependence_lock);
        for (unsigned idx = 0; idx < 
              context_configuration.meta_task_vector_width; idx++)
        {
          if (dependence_queue.empty())
            break;
          to_perform.push_back(dependence_queue.front());
          dependence_queue.pop_front();
        }
#ifdef DEBUG_LEGION
        assert(outstanding_dependence);
#endif
        if (dependence_queue.empty())
        {
          outstanding_dependence = false;
          // Guard ourselves against tasks running after us
          dependence_precondition = 
            RtEvent(Processor::get_current_finish_event());
        }
        else
          launch_next_op = dependence_queue.front();
      }
      // Perform our operations
      for (std::vector<Operation*>::const_iterator it = 
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->execute_dependence_analysis();
      // Then launch the next task if needed
      if (launch_next_op != NULL)
      {
        DependenceArgs args(launch_next_op, this);
        // Sample currently_active without the lock to try to get our priority
        runtime->issue_runtime_meta_task(args, !currently_active_context ? 
              LG_THROUGHPUT_DEFERRED_PRIORITY : LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                         const void *result, size_t size, PhysicalInstance inst)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      {
        AutoLock p_lock(post_task_lock);
        post_task_queue.push_back(PostTaskArgs(ctx, result, size,inst,wait_on));
        // No need to have more outstanding tasks than there are processors
        if (outstanding_post_task < runtime->num_utility_procs)
        {
          const size_t needed_in_flight = 
            (post_task_queue.size() + 
             context_configuration.meta_task_vector_width - 1) /
              context_configuration.meta_task_vector_width;
          if (outstanding_post_task < needed_in_flight)
          {
            outstanding_post_task++;
            issue_task = true;
          }
        }
      }
      if (issue_task)
      {
        PostEndArgs args(ctx->owner_task, this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::process_post_end_tasks(void)
    //--------------------------------------------------------------------------
    {
      std::vector<PostTaskArgs> to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      TaskOp *launch_next_op = NULL;
      {
        AutoLock p_lock(post_task_lock);
        for (std::list<PostTaskArgs>::iterator it = 
              post_task_queue.begin(); it != post_task_queue.end(); /*nothing*/)
        {
          if (!it->wait_on.exists() || it->wait_on.has_triggered())
          {
            to_perform.push_back(*it);
            it = post_task_queue.erase(it);
            if (to_perform.size() == 
                context_configuration.meta_task_vector_width)
              break;
          }
          else
            it++;
        }
#ifdef DEBUG_LEGION
        assert(outstanding_post_task > 0);
#endif
        if (!post_task_queue.empty())
        {
          const size_t needed_in_flight = 
            (post_task_queue.size() + 
             context_configuration.meta_task_vector_width - 1) /
              context_configuration.meta_task_vector_width;
          if (outstanding_post_task <= needed_in_flight)
            launch_next_op = post_task_queue.front().context->owner_task;
          else
            outstanding_post_task--;
        }
        else
          outstanding_post_task--;
      }
      if (!to_perform.empty())
      {
        if (launch_next_op == NULL)
        {
          // We're not launching the next operation, so we can just kick
          // these things off as they are without preconditions
          for (std::vector<PostTaskArgs>::iterator it =
                to_perform.begin(); it != to_perform.end(); it++)
          {
            DeferredPostTaskArgs args(*it, RtUserEvent::NO_RT_USER_EVENT);
            runtime->issue_runtime_meta_task(args, 
                  LG_THROUGHPUT_DEFERRED_PRIORITY);
          }
        }
        else
        {
          // We're going to launch the next iteration also, but don't do
          // it until all of our currently kicked off tasks have started 
          std::set<RtEvent> next_op_preconditions;
          for (std::vector<PostTaskArgs>::iterator it =
                to_perform.begin(); it != to_perform.end(); it++)
          {
            RtUserEvent start_event = Runtime::create_rt_user_event();
            DeferredPostTaskArgs args(*it, start_event);
            runtime->issue_runtime_meta_task(args, 
                  LG_THROUGHPUT_DEFERRED_PRIORITY);
            next_op_preconditions.insert(start_event);
          }
          // Now we can kick off the next iteration with the right preconditions
          PostEndArgs args(launch_next_op, this);
          runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY,
                                Runtime::merge_events(next_op_preconditions));
        }
      }
      else if (launch_next_op != NULL)
      {
        PostEndArgs args(launch_next_op, this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_executing_child(Operation *op)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
      assert(executing_children.find(op) == executing_children.end());
#endif
      executing_children[op] = op->get_generation();
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
        std::map<Operation*,GenerationID>::iterator 
          finder = executing_children.find(op);
#ifdef DEBUG_LEGION
        assert(finder != executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
#endif
        // Now put it in the list of executing operations
        // Note this doesn't change the number of active children
        // so there's no need to trigger any window waits
        executed_children.insert(*finder);
        executing_children.erase(finder);
        // Add some hysteresis here so that we have some runway for when
        // the paused task resumes it can run for a little while.
        int outstanding_count = 
          __sync_add_and_fetch(&outstanding_children_count,-1);
#ifdef DEBUG_LEGION
        assert(outstanding_count >= 0);
#endif
        if (valid_wait_event && (context_configuration.max_window_size > 0) &&
            (outstanding_count <=
             int((100 - context_configuration.hysteresis_percentage) * 
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
    void InnerContext::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock child_lock(child_op_lock);
        std::map<Operation*,GenerationID>::iterator finder = 
          executed_children.find(op);
#ifdef DEBUG_LEGION
        assert(finder != executed_children.end());
        assert(complete_children.find(op) == complete_children.end());
        assert(executing_children.find(op) == executing_children.end());
#endif
        // Put it on the list of complete children to complete
        complete_children.insert(*finder);
        executed_children.erase(finder);
        // See if we need to trigger the all children complete call
        if (task_executed && executing_children.empty() && 
            executed_children.empty() && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger && (owner_task != NULL))
        owner_task->trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock child_lock(child_op_lock);
        std::map<Operation*,GenerationID>::iterator finder = 
          complete_children.find(op);
#ifdef DEBUG_LEGION
        assert(finder != complete_children.end());
        assert(executing_children.find(op) == executing_children.end());
        assert(executed_children.find(op) == executed_children.end());
        outstanding_children.erase(op->get_ctx_index());
#endif
        complete_children.erase(finder);
        // See if we need to trigger the all children commited call
        if (task_executed && executing_children.empty() && 
            executed_children.empty() && complete_children.empty() &&
            !children_commit_invoked)
        {
          needs_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (needs_trigger && (owner_task != NULL))
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void InnerContext::unregister_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
        // Remove it from everything and then see if we need to
        // trigger the window wait event
        executing_children.erase(op);
        executed_children.erase(op);
        complete_children.erase(op);
#ifdef DEBUG_LEGION
        outstanding_children.erase(op->get_ctx_index());
#endif
        int outstanding_count = 
          __sync_add_and_fetch(&outstanding_children_count,-1);
#ifdef DEBUG_LEGION
        assert(outstanding_count >= 0);
#endif
        if (valid_wait_event && (context_configuration.max_window_size > 0) &&
            (outstanding_count <=
             int((100 - context_configuration.hysteresis_percentage) *
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
    void InnerContext::print_children(void)
    //--------------------------------------------------------------------------
    {
      // Don't both taking the lock since this is for debugging
      // and isn't actually called anywhere
      for (std::map<Operation*,GenerationID>::const_iterator it =
            executing_children.begin(); it != executing_children.end(); it++)
      {
        Operation *op = it->first;
        printf("Executing Child %p\n",op);
      }
      for (std::map<Operation*,GenerationID>::const_iterator it =
            executed_children.begin(); it != executed_children.end(); it++)
      {
        Operation *op = it->first;
        printf("Executed Child %p\n",op);
      }
      for (std::map<Operation*,GenerationID>::const_iterator it =
            complete_children.begin(); it != complete_children.end(); it++)
      {
        Operation *op = it->first;
        printf("Complete Child %p\n",op);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (current_mapping_fence != NULL)
      {
#ifdef LEGION_SPY
        // Can't prune when doing legion spy
        op->register_dependence(current_mapping_fence, mapping_fence_gen);
        unsigned num_regions = op->get_region_count();
        if (num_regions > 0)
        {
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            LegionSpy::log_mapping_dependence(
                get_unique_id(), current_fence_uid, 0,
                op->get_unique_op_id(), idx, TRUE_DEPENDENCE);
          }
        }
        else
          LegionSpy::log_mapping_dependence(
              get_unique_id(), current_fence_uid, 0,
              op->get_unique_op_id(), 0, TRUE_DEPENDENCE);
#else
        // If we can prune it then go ahead and do so
        // No need to remove the mapping reference because 
        // the fence has already been committed
        if (op->register_dependence(current_mapping_fence, mapping_fence_gen))
          current_mapping_fence = NULL;
#endif
      }
#ifdef LEGION_SPY
      previous_completion_events.insert(op->get_completion_event());
      // Periodically merge these to keep this data structure from exploding
      // when we have a long-running task, although don't do this for fence
      // operations in case we have to prune ourselves out of the set
      if ((previous_completion_events.size() >= DEFAULT_MAX_TASK_WINDOW) &&
          (op->get_operation_kind() != Operation::FENCE_OP_KIND))
      {
        ApEvent merge = Runtime::merge_events(previous_completion_events);
        previous_completion_events.clear();
        previous_completion_events.insert(merge);
      }
      // Have to record this operation in case there is a fence later
      ops_since_last_fence.push_back(op->get_unique_op_id());
      return current_execution_fence_event;
#else
      if (current_execution_fence_event.exists())
      {
        if (current_execution_fence_event.has_triggered())
          current_execution_fence_event = ApEvent::NO_AP_EVENT;
        return current_execution_fence_event;
      }
      return ApEvent::NO_AP_EVENT;
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::perform_fence_analysis(Operation *op, 
                                                 bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      std::map<Operation*,GenerationID> previous_operations;
      std::set<ApEvent> previous_events;
      // Take the lock and iterate through our current pending
      // operations and find all the ones with a context index
      // that is less than the index for the fence operation
      const unsigned next_fence_index = op->get_ctx_index();
      // We only need the list of previous operations if we are recording
      // mapping dependences for this fence
      if (!execution)
      {
        // Mapping analysis only
        AutoLock child_lock(child_op_lock,1,false/*exclusive*/);
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executing_children.begin(); it != executing_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_mapping_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_operations.insert(*it);
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_mapping_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_operations.insert(*it);
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_mapping_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_operations.insert(*it);
        }
      }
      else if (!mapping)
      {
        // Execution analysis only
        AutoLock child_lock(child_op_lock,1,false/*exclusive*/);
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executing_children.begin(); it != executing_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
      }
      else
      {
        // Both mapping and execution analysis at the same time
        AutoLock child_lock(child_op_lock,1,false/*exclusive*/);
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executing_children.begin(); it != executing_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const unsigned op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_completion_event());
        }
      }

      // Now record the dependences
      if (!previous_operations.empty())
      {
        for (std::map<Operation*,GenerationID>::const_iterator it = 
             previous_operations.begin(); it != previous_operations.end(); it++)
          op->register_dependence(it->first, it->second);
      }

#ifdef LEGION_SPY
      // Record a dependence on the previous fence
      if (mapping)
      {
        if (current_mapping_fence != NULL)
          LegionSpy::log_mapping_dependence(get_unique_id(), current_fence_uid,
              0/*index*/, op->get_unique_op_id(), 0/*index*/, TRUE_DEPENDENCE);
        for (std::deque<UniqueID>::const_iterator it = 
              ops_since_last_fence.begin(); it != 
              ops_since_last_fence.end(); it++)
        {
          // Skip ourselves if we are here
          if ((*it) == op->get_unique_op_id())
            continue;
          LegionSpy::log_mapping_dependence(get_unique_id(), *it, 0/*index*/,
              op->get_unique_op_id(), 0/*index*/, TRUE_DEPENDENCE); 
        }
      }
      // If we're doing execution record dependence on all previous operations
      if (execution)
      {
        previous_events.insert(previous_completion_events.begin(),
                               previous_completion_events.end());
        // Don't include ourselves though
        previous_events.erase(op->get_completion_event());
      }
#endif
      // Also include the current execution fence in case the operation
      // already completed and wasn't in the set, make sure to do this
      // before we update the current fence
      if (execution && current_execution_fence_event.exists())
        previous_events.insert(current_execution_fence_event);
      if (!previous_events.empty())
        return Runtime::merge_events(previous_events);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void InnerContext::update_current_fence(FenceOp *op, 
                                            bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      if (mapping)
      {
        if (current_mapping_fence != NULL)
          current_mapping_fence->remove_mapping_reference(mapping_fence_gen);
        current_mapping_fence = op;
        mapping_fence_gen = op->get_generation();
        current_mapping_fence_index = op->get_ctx_index();
        current_mapping_fence->add_mapping_reference(mapping_fence_gen);
#ifdef LEGION_SPY
        current_fence_uid = op->get_unique_op_id();
        ops_since_last_fence.clear();
#endif
      }
      if (execution)
      {
        // Only update the current fence event if we're actually an
        // execution fence, otherwise by definition we need the previous event
        current_execution_fence_event = op->get_completion_event();
        current_execution_fence_index = op->get_ctx_index();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::get_current_mapping_fence_event(void)
    //--------------------------------------------------------------------------
    {
      if (current_mapping_fence == NULL)
        return RtEvent::NO_RT_EVENT;
      RtEvent result = current_mapping_fence->get_mapped_event();
      // Check the generation
      if (current_mapping_fence->get_generation() == mapping_fence_gen)
        return result;
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::get_current_execution_fence_event(void)
    //--------------------------------------------------------------------------
    {
      return current_execution_fence_event;
    }

    //--------------------------------------------------------------------------
    void InnerContext::begin_trace(TraceID tid, bool logical_only)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing) return;

      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Beginning a trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_NESTED_TRACE,
          "Illegal nested trace with ID %d attempted in "
                       "task %s (ID %lld)", tid, get_task_name(),
                       get_unique_id())
      std::map<TraceID,DynamicTrace*>::const_iterator finder = traces.find(tid);
      DynamicTrace* dynamic_trace = NULL;
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        dynamic_trace = new DynamicTrace(tid, this, logical_only);
        dynamic_trace->add_reference();
        traces[tid] = dynamic_trace;
      }
      else
        dynamic_trace = finder->second;

#ifdef DEBUG_LEGION
      assert(dynamic_trace != NULL);
#endif
      dynamic_trace->clear_blocking_call();

      // Issue a begin op
      TraceBeginOp *begin = runtime->get_available_begin_op();
      begin->initialize_begin(this, dynamic_trace);
      runtime->add_to_dependence_queue(this, executing_processor, begin);

      if (!logical_only)
      {
        // Issue a replay op
        TraceReplayOp *replay = runtime->get_available_replay_op();
        replay->initialize_replay(this, dynamic_trace);
        runtime->add_to_dependence_queue(this, executing_processor, replay);
      }

      // Now mark that we are starting a trace
      current_trace = dynamic_trace;
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing) return;

      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Ending a trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      if (current_trace == NULL)
        REPORT_LEGION_ERROR(ERROR_UMATCHED_END_TRACE,
          "Unmatched end trace for ID %d in task %s "
                       "(ID %lld)", tid, get_task_name(),
                       get_unique_id())
      else if (!current_trace->is_dynamic_trace())
      {
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_END_TRACE_CALL,
          "Illegal end trace call on a static trace in "
                       "task %s (UID %lld)", get_task_name(), get_unique_id());
      }
      bool has_blocking_call = current_trace->has_blocking_call();
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op();
        complete_op->initialize_complete(this, has_blocking_call);
        runtime->add_to_dependence_queue(this, executing_processor, complete_op);
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(); 
        capture_op->initialize_capture(this, has_blocking_call);
        runtime->add_to_dependence_queue(this, executing_processor, capture_op);
        // Mark that the current trace is now fixed
        current_trace->as_dynamic_trace()->fix_trace();
      }
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void InnerContext::begin_static_trace(const std::set<RegionTreeID> *trees)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing) return;

      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Beginning a static trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_NESTED_STATIC_TRACE,
          "Illegal nested static trace attempted in "
                       "task %s (ID %lld)", get_task_name(), get_unique_id())
      // Issue the mapping fence into the analysis
      runtime->issue_mapping_fence(this);
      // Then we make a static trace
      current_trace = new StaticTrace(this, trees); 
      current_trace->add_reference();
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_static_trace(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing) return;

      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Ending a static trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      if (current_trace == NULL)
        REPORT_LEGION_ERROR(ERROR_UNMATCHED_END_STATIC_TRACE,
          "Unmatched end static trace in task %s "
                       "(ID %lld)", get_task_name(), get_unique_id())
      else if (current_trace->is_dynamic_trace())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_END_STATIC_TRACE,
          "Illegal end static trace call on a dynamic trace in "
                       "task %s (UID %lld)", get_task_name(), get_unique_id())
      // We're done with this trace, need a trace complete op to clean up
      // This operation takes ownership of the static trace reference
      TraceCompleteOp *complete_op = runtime->get_available_trace_op();
      complete_op->initialize_complete(this,current_trace->has_blocking_call());
      runtime->add_to_dependence_queue(this, executing_processor, complete_op);
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_previous_trace(LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      previous_trace = trace;
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_trace_cache(
                                     LegionTrace *trace, Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      if (previous_trace != NULL && previous_trace != trace)
        previous_trace->invalidate_trace_cache(invalidator);
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_blocking_call(void)
    //--------------------------------------------------------------------------
    {
      if (current_trace != NULL)
        current_trace->record_blocking_call();
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_frame(FrameOp *frame, ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      // This happens infrequently enough that we can just issue
      // a meta-task to see what we should do without holding the lock
      if (context_configuration.max_outstanding_frames > 0)
      {
        IssueFrameArgs args(owner_task, this, frame, frame_termination);
        // We know that the issuing is done in order because we block after
        // we launch this meta-task which blocks the application task
        RtEvent wait_on = runtime->issue_runtime_meta_task(args,
                                      LG_LATENCY_WORK_PRIORITY);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_frame_issue(FrameOp *frame,
                                         ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on, previous;
      {
        AutoLock child_lock(child_op_lock);
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
    void InnerContext::finish_frame(ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      // Pull off all the frame events until we reach ours
      if (context_configuration.max_outstanding_frames > 0)
      {
        AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
        assert(frame_events.front() == frame_termination);
#endif
        frame_events.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_outstanding(void)
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
        AutoLock child_lock(child_op_lock);
        if (!currently_active_context && (outstanding_subtasks == 0) && 
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
          currently_active_context = true;
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
    void InnerContext::decrement_outstanding(void)
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
        AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
        assert(outstanding_subtasks > 0);
#endif
        outstanding_subtasks--;
        if (currently_active_context && (outstanding_subtasks == 0) && 
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
          currently_active_context = false;
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
    void InnerContext::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
        pending_subtasks++;
        if (currently_active_context && (outstanding_subtasks > 0) &&
            (pending_subtasks == context_configuration.min_tasks_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
          currently_active_context = false;
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
    RtEvent InnerContext::decrement_pending(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduled by frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return RtEvent::NO_RT_EVENT;
      return decrement_pending(true/*need deferral*/);
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::decrement_pending(bool need_deferral)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
        assert(pending_subtasks > 0);
#endif
        pending_subtasks--;
        if (!currently_active_context && (outstanding_subtasks > 0) &&
            (pending_subtasks < context_configuration.min_tasks_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
          currently_active_context = true;
        }
      }
      // Do anything that we need to do
      if (to_trigger.exists())
      {
        if (need_deferral)
        {
          PostDecrementArgs post_decrement_args;
          post_decrement_args.parent_ctx = this;
          RtEvent done = runtime->issue_runtime_meta_task(post_decrement_args,
              LG_LATENCY_WORK_PRIORITY, wait_on); 
          Runtime::trigger_event(to_trigger, done);
          return to_trigger;
        }
        else
        {
          wait_on.wait();
          runtime->activate_context(this);
          Runtime::trigger_event(to_trigger);
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
        pending_frames++;
        if (currently_active_context && (outstanding_subtasks > 0) &&
            (pending_frames == context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
          currently_active_context = false;
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
    void InnerContext::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      RtEvent wait_on;
      RtUserEvent to_trigger;
      {
        AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
        assert(pending_frames > 0);
#endif
        pending_frames--;
        if (!currently_active_context && (outstanding_subtasks > 0) &&
            (pending_frames < context_configuration.min_frames_to_schedule))
        {
          wait_on = context_order_event;
          to_trigger = Runtime::create_rt_user_event();
          context_order_event = to_trigger;
          currently_active_context = true;
        }
      }
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_acquisition(AcquireOp *op,
                                       const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      if (!runtime->forest->add_acquisition(coherence_restrictions, op, req))
        // We faiiled to acquire, report the error
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_ACQUIRE_OPERATION,
          "Illegal acquire operation (ID %lld) performed in "
                      "task %s (ID %lld). Acquire was performed on a non-"
                      "restricted region.", op->get_unique_op_id(),
                      get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void InnerContext::remove_acquisition(ReleaseOp *op, 
                                          const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      if (!runtime->forest->remove_acquisition(coherence_restrictions, op, req))
        // We failed to release, report the error
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RELEASE_OPERATION,
          "Illegal release operation (ID %lld) performed in "
                      "task %s (ID %lld). Release was performed on a region "
                      "that had not previously been acquired.",
                      op->get_unique_op_id(), get_task_name(), 
                      get_unique_id())
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_restriction(AttachOp *op, InstanceManager *inst,
                                       const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      runtime->forest->add_restriction(coherence_restrictions, op, inst, req);
    }

    //--------------------------------------------------------------------------
    void InnerContext::remove_restriction(DetachOp *op, 
                                          const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      // Try to remove the restriction, it is alright if it fails
      runtime->forest->remove_restriction(coherence_restrictions, op, req);
    }

    //--------------------------------------------------------------------------
    void InnerContext::release_restrictions(void)
    //--------------------------------------------------------------------------
    {
      for (std::list<Restriction*>::const_iterator it = 
            coherence_restrictions.begin(); it != 
            coherence_restrictions.end(); it++)
        delete (*it);
      coherence_restrictions.clear();
    }

    //--------------------------------------------------------------------------
    bool InnerContext::has_restrictions(void) const
    //--------------------------------------------------------------------------
    {
      return !coherence_restrictions.empty();
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_restricted_analysis(const RegionRequirement &req,
                                                   RestrictInfo &restrict_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!coherence_restrictions.empty());
#endif
      runtime->forest->perform_restricted_analysis(
                                coherence_restrictions, req, restrict_info);
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_dynamic_collective_contribution(
                                          DynamicCollective dc, const Future &f) 
    //--------------------------------------------------------------------------
    {
      AutoLock col_lock(collective_lock);
      collective_contributions[dc.phase_barrier].push_back(f);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_collective_contributions(DynamicCollective dc, 
                                             std::vector<Future> &contributions)
    //--------------------------------------------------------------------------
    {
      // Find any future contributions and record dependences for the op
      // Contributions were made to the previous phase
      ApEvent previous = Runtime::get_previous_phase(dc.phase_barrier);
      AutoLock col_lock(collective_lock);
      std::map<ApEvent,std::vector<Future> >::iterator finder = 
          collective_contributions.find(previous);
      if (finder == collective_contributions.end())
        return;
      contributions = finder->second;
      collective_contributions.erase(finder);
    }

    //--------------------------------------------------------------------------
    TaskPriority InnerContext::get_current_priority(void) const
    //--------------------------------------------------------------------------
    {
      return current_priority;
    }

    //--------------------------------------------------------------------------
    void InnerContext::set_current_priority(TaskPriority priority)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mutable_priority);
      assert(realm_done_event.exists());
#endif
      // This can be racy but that is the mappers problem
      realm_done_event.set_operation_priority(priority);
      current_priority = priority;
    }

    //--------------------------------------------------------------------------
    void InnerContext::configure_context(MapperManager *mapper, TaskPriority p)
    //--------------------------------------------------------------------------
    {
      mapper->invoke_configure_context(owner_task, &context_configuration);
      // Do a little bit of checking on the output.  Make
      // sure that we only set one of the two cases so we
      // are counting by frames or by outstanding tasks.
      if ((context_configuration.min_tasks_to_schedule == 0) && 
          (context_configuration.min_frames_to_schedule == 0))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from call 'configure_context' "
                      "on mapper %s. One of 'min_tasks_to_schedule' and "
                      "'min_frames_to_schedule' must be non-zero for task "
                      "%s (ID %lld)", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id())
      // Hysteresis percentage is an unsigned so can't be less than 0
      if (context_configuration.hysteresis_percentage > 100)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from call 'configure_context' "
                      "on mapper %s. The 'hysteresis_percentage' %d is not "
                      "a value between 0 and 100 for task %s (ID %lld)",
                      mapper->get_mapper_name(), 
                      context_configuration.hysteresis_percentage,
                      get_task_name(), get_unique_id())
      if (context_configuration.meta_task_vector_width == 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from call 'configure context' "
                      "on mapper %s for task %s (ID %lld). The "
                      "'meta_task_vector_width' must be a non-zero value.",
                      mapper->get_mapper_name(),
                      get_task_name(), get_unique_id())

      // If we're counting by frames set min_tasks_to_schedule to zero
      if (context_configuration.min_frames_to_schedule > 0)
        context_configuration.min_tasks_to_schedule = 0;
      // otherwise we know min_frames_to_schedule is zero

      // See if we permit priority mutations from child operation mapppers
      mutable_priority = context_configuration.mutable_priority; 
      current_priority = p;
    }

    //--------------------------------------------------------------------------
    void InnerContext::initialize_region_tree_contexts(
                      const std::vector<RegionRequirement> &clone_requirements,
                      const std::vector<ApUserEvent> &unmap_events,
                      std::set<ApEvent> &preconditions,
                      std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INITIALIZE_REGION_TREE_CONTEXTS_CALL);
      // Save to cast to single task here because this will never
      // happen during inlining of index space tasks
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      SingleTask *single_task = dynamic_cast<SingleTask*>(owner_task);
      assert(single_task != NULL);
#else
      SingleTask *single_task = static_cast<SingleTask*>(owner_task); 
#endif
      const std::deque<InstanceSet> &physical_instances = 
        single_task->get_physical_instances();
      const std::vector<bool> &no_access_regions = 
        single_task->get_no_access_regions();
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
          runtime->forest->initialize_current_context(tree_context,
              clone_requirements[idx], physical_instances[idx],
              unmap_events[idx], this, idx, top_views, applied_events);
#ifdef DEBUG_LEGION
          assert(!physical_instances[idx].empty());
#endif
          // Always make reduce-only privileges restricted so that
          // we always flush data back, this will prevent us from 
          // needing a post close op later
          if (IS_REDUCE(regions[idx]))
            coherence_restrictions.push_back(
                runtime->forest->create_coherence_restriction(regions[idx],
                                                  physical_instances[idx]));
          // If we need to add restricted coherence, do that now
          // Not we only need to do this for non-virtually mapped task
          else if ((regions[idx].prop == SIMULTANEOUS) && 
                   ((regions[idx].privilege == READ_ONLY) ||
                    (regions[idx].privilege == READ_WRITE) ||
                    (regions[idx].privilege == WRITE_DISCARD)))
            coherence_restrictions.push_back(
                runtime->forest->create_coherence_restriction(regions[idx],
                                                  physical_instances[idx]));
        }
        else
        {
          runtime->forest->initialize_virtual_context(tree_context,
                                          clone_requirements[idx]);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INVALIDATE_REGION_TREE_CONTEXTS_CALL);
      // Send messages to invalidate any remote contexts
      if (!remote_instances.empty())
      {
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
          // If we have created requirements figure out what invalidations
          // we have to send to the remote context
          if (!created_requirements.empty())
          {
            std::map<unsigned,LogicalRegion> to_invalidate;
            for (unsigned idx = 0; idx < created_requirements.size(); idx++)
            {
              if (!returnable_privileges[idx] || 
                  was_created_requirement_deleted(created_requirements[idx]))
                to_invalidate[idx] = created_requirements[idx].region;
            }
            rez.serialize<size_t>(to_invalidate.size());
            for (std::map<unsigned,LogicalRegion>::const_iterator it = 
                  to_invalidate.begin(); it != to_invalidate.end(); it++)
            {
              // Add the size of the original regions to the index
              rez.serialize<unsigned>(it->first + regions.size());
              rez.serialize(it->second);
            }
          }
          else
            rez.serialize<size_t>(0);
        }
        for (std::map<AddressSpaceID,RemoteContext*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
          runtime->send_remote_context_release(it->first, rez);
      }
      // Invalidate all our region contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        runtime->forest->invalidate_current_context(tree_context,
                                                    false/*users only*/,
                                                    regions[idx].region);
        if (!virtual_mapped[idx])
          runtime->forest->invalidate_versions(tree_context, 
                                               regions[idx].region);
      }
      if (!created_requirements.empty())
      {
        TaskContext *outermost = find_outermost_local_context();
        const bool is_outermost = (outermost == this);
        RegionTreeContext outermost_ctx = outermost->get_context();
        RegionTreeContext top_ctx = find_top_context()->get_context();
        for (unsigned idx = 0; idx < created_requirements.size(); idx++)
        {
          // See if we're a returnable privilege or not
          if (returnable_privileges[idx])
          {
            // If we're the outermost context or the requirement was
            // deleted, then we can invalidate everything
            // Otherwiswe we only invalidate the users
            const bool was_deleted = 
              was_created_requirement_deleted(created_requirements[idx]);
            const bool users_only = !is_outermost && !was_deleted;
            runtime->forest->invalidate_current_context(outermost_ctx,
                        users_only, created_requirements[idx].region);
            // If it was deleted then we need to invalidate the versions
            // in the outermost context
            if (was_deleted)
              runtime->forest->invalidate_versions(top_ctx,
                                      created_requirements[idx].region);
          }
          else // Not returning so invalidate the full thing 
          {
            runtime->forest->invalidate_current_context(tree_context,
                false/*users only*/, created_requirements[idx].region);
            // Little tricky here, this is safe to invaliate the whole
            // tree even if we only had privileges on a field because
            // if we had privileges on the whole region in this context
            // it would have merged the created_requirement and we wouldn't
            // have a non returnable privilege requirement in this context
            runtime->forest->invalidate_versions(tree_context,
                                     created_requirements[idx].region);
          }
        }
      }
      // Clean up our instance top views
      if (!instance_top_views.empty())
      {
        for (std::map<PhysicalManager*,InstanceView*>::const_iterator it = 
              instance_top_views.begin(); it != instance_top_views.end(); it++)
        {
          it->first->unregister_active_context(this);
          if (it->second->remove_base_resource_ref(CONTEXT_REF))
            delete (it->second);
        }
        instance_top_views.clear();
      } 
      // Now we can free our region tree context
      runtime->free_region_tree_context(tree_context);
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_remote_tree_contexts(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Should only be called on RemoteContext
      assert(false);
    } 

    //--------------------------------------------------------------------------
    InstanceView* InnerContext::create_instance_top_view(
                        PhysicalManager *manager, AddressSpaceID request_source,
                        RtEvent *ready_event/*=NULL*/)
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
        AutoLock inst_lock(instance_view_lock);
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
        AutoLock inst_lock(instance_view_lock, 1, false/*exclusive*/);
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
      // Record the result and trigger any user event to signal that the
      // view is ready
      RtUserEvent to_trigger;
      {
        AutoLock inst_lock(instance_view_lock);
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
    void InnerContext::notify_instance_deletion(PhysicalManager *deleted)
    //--------------------------------------------------------------------------
    {
      InstanceView *removed = NULL;
      {
        AutoLock inst_lock(instance_view_lock);
        std::map<PhysicalManager*,InstanceView*>::iterator finder =  
          instance_top_views.find(deleted);
#ifdef DEBUG_LEGION
        assert(finder != instance_top_views.end());
#endif
        removed = finder->second;
        instance_top_views.erase(finder);
      }
      if (removed->remove_base_resource_ref(CONTEXT_REF))
        delete removed;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_create_top_view_request(
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
      InnerContext *context = runtime->find_context(context_uid);
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
      // Nasty deadlock case: if the request came from a different node
      // we have to defer this because we are in the view virtual channel
      // and we might invoke the update virtual channel, but we already
      // know it's possible for the update channel to block waiting on
      // the view virtual channel (paging views), so to avoid the cycle
      // we have to launch a meta-task and record when it is done
      RemoteCreateViewArgs args;
      args.proxy_this = context;
      args.manager = manager;
      args.target = target;
      args.to_trigger = to_trigger;
      args.source = source;
      runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_remote_view_creation(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemoteCreateViewArgs *rargs = (const RemoteCreateViewArgs*)args;
      
      InstanceView *result = rargs->proxy_this->create_instance_top_view(
                                                 rargs->manager, rargs->source);
      // Now we can send the response
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(result->did);
        rez.serialize(rargs->target);
        rez.serialize(rargs->to_trigger);
      }
      rargs->proxy_this->runtime->send_create_top_view_response(rargs->source, 
                                                                rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_create_top_view_response(
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
    bool InnerContext::attempt_children_complete(void)
    //--------------------------------------------------------------------------
    {
      AutoLock chil_lock(child_op_lock);
      if (task_executed && executing_children.empty() && 
          executed_children.empty() && !children_complete_invoked)
      {
        children_complete_invoked = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::attempt_children_commit(void)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
      if (task_executed && executing_children.empty() && 
          executed_children.empty() && complete_children.empty() && 
          !children_commit_invoked)
      {
        children_commit_invoked = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& InnerContext::begin_task(
                                                           Legion::Runtime *&rt)
    //--------------------------------------------------------------------------
    {
      // If we have mutable priority we need to save our realm done event
      if (mutable_priority)
        realm_done_event = ApEvent(Processor::get_current_finish_event());
      // Now do the base begin task routine
      return TaskContext::begin_task(rt);
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_task(const void *res, size_t res_size, bool owned,
                                PhysicalInstance deferred_result_instance)
    //--------------------------------------------------------------------------
    {
      if (overhead_tracker != NULL)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff = current - previous_profiling_time;
        overhead_tracker->application_time += diff;
      }
      // Safe to cast to a single task here because this will never
      // be called while inlining an index space task
#ifdef DEBUG_LEGION
      SingleTask *single_task = dynamic_cast<SingleTask*>(owner_task);
      assert(single_task != NULL);
#else
      SingleTask *single_task = static_cast<SingleTask*>(owner_task);
#endif
      // See if there are any runtime warnings to issue
      if (runtime->runtime_warnings)
      {
        if (total_children_count == 0)
        {
          // If there were no sub operations and this wasn't marked a
          // leaf task then signal a warning
          VariantImpl *impl = 
            runtime->find_variant_impl(single_task->task_id, 
                                       single_task->get_selected_variant());
          REPORT_LEGION_WARNING(LEGION_WARNING_VARIANT_TASK_NOT_MARKED,
            "Variant %s of task %s (UID %lld) was "
              "not marked as a 'leaf' variant but it didn't execute any "
              "operations. Did you forget the 'leaf' annotation?", 
              impl->get_name(), get_task_name(), get_unique_id());
        }
        else if (!single_task->is_inner())
        {
          // If this task had sub operations and wasn't marked as inner
          // and made no accessors warn about missing 'inner' annotation
          // First check for any inline accessors that were made
          bool has_accessor = has_inline_accessor;
          if (!has_accessor)
          {
            for (unsigned idx = 0; idx < physical_regions.size(); idx++)
            {
              if (!physical_regions[idx].impl->created_accessor())
                continue;
              has_accessor = true;
              break;
            }
          }
          if (!has_accessor)
          {
            VariantImpl *impl = 
              runtime->find_variant_impl(single_task->task_id, 
                                         single_task->get_selected_variant());
            REPORT_LEGION_WARNING(LEGION_WARNING_VARIANT_TASK_NOT_MARKED,
              "Variant %s of task %s (UID %lld) was "
                "not marked as an 'inner' variant but it only launched "
                "operations and did not make any accessors. Did you "
                "forget the 'inner' annotation?",
                impl->get_name(), get_task_name(), get_unique_id());
          }
        }
      }
      // Quick check to make sure the user didn't forget to end a trace
      if (current_trace != NULL)
        REPORT_LEGION_ERROR(ERROR_TASK_FAILED_END_TRACE,
          "Task %s (UID %lld) failed to end trace before exiting!",
                        get_task_name(), get_unique_id())
      // Unmap any of our mapped regions before issuing any close operations
      unmap_all_regions();
      // If we own any task local regions, we issue deletion operations on them.
      if (!local_regions.empty())
      {
        // Create a copy of this since it's about to be mutated in 
        // the function call we're about to do
        const std::vector<LogicalRegion> to_destroy(local_regions.begin(),
                                                    local_regions.end());
        for (std::vector<LogicalRegion>::const_iterator it = 
              to_destroy.begin(); it != to_destroy.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(created_regions.find(*it) != created_regions.end());
#endif
          destroy_logical_region(*it);
        }
      }
      const std::deque<InstanceSet> &physical_instances = 
        single_task->get_physical_instances();
      // Note that this loop doesn't handle create regions
      // we deal with that case below
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        // We also don't need to close up read-only instances
        // or reduction-only instances (because they are restricted)
        // so all changes have already been propagated
        if (!IS_WRITE(regions[idx]))
          continue;
        if (!virtual_mapped[idx])
        {
#ifdef DEBUG_LEGION
          assert(!physical_instances[idx].empty());
#endif
          PostCloseOp *close_op = 
            runtime->get_available_post_close_op();
          close_op->initialize(this, idx, physical_instances[idx]);
          runtime->add_to_dependence_queue(this, executing_processor, close_op);
        }
        else
        {
          // Make a virtual close op to close up the instance
          VirtualCloseOp *close_op = 
            runtime->get_available_virtual_close_op();
          close_op->initialize(this, idx, regions[idx]);
          runtime->add_to_dependence_queue(this, executing_processor, close_op);
        }
      }
      // Mark that we are done executing this operation
      // We're not actually done until we have registered our pending
      // decrement of our parent task and recorded any profiling
      if (!pending_done.has_triggered())
        owner_task->complete_execution(pending_done);
      else
        owner_task->complete_execution();
      // Grab some information before doing the next step in case it
      // results in the deletion of 'this'
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      const TaskID owner_task_id = owner_task->task_id;
#endif
      Runtime *runtime_ptr = runtime;
      // Tell the parent context that we are ready for post-end
      // Make a copy of the results if necessary
      TaskContext *parent_ctx = owner_task->get_context();
      if (deferred_result_instance.exists())
      {
        RtEvent result_ready(Processor::get_current_finish_event());
        if (last_registration.exists() && !last_registration.has_triggered())
          parent_ctx->add_to_post_task_queue(this, 
              Runtime::merge_events(result_ready, last_registration),
              res, res_size, deferred_result_instance);
        else
          parent_ctx->add_to_post_task_queue(this, result_ready,
                          res, res_size, deferred_result_instance);
      }
      else if (!owned)
      {
        void *result_copy = malloc(res_size);
        memcpy(result_copy, res, res_size);
        parent_ctx->add_to_post_task_queue(this, last_registration,
                  result_copy, res_size, PhysicalInstance::NO_INST);
      }
      else
        parent_ctx->add_to_post_task_queue(this, last_registration,
                          res, res_size, PhysicalInstance::NO_INST);
#ifdef DEBUG_LEGION
      runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                     false/*meta*/);
#else
      runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    void InnerContext::post_end_task(const void *res,size_t res_size,bool owned)
    //--------------------------------------------------------------------------
    {
      // Safe to cast to a single task here because this will never
      // be called while inlining an index space task
#ifdef DEBUG_LEGION
      SingleTask *single_task = dynamic_cast<SingleTask*>(owner_task);
      assert(single_task != NULL);
#else
      SingleTask *single_task = static_cast<SingleTask*>(owner_task);
#endif
      // Handle the future result
      single_task->handle_future(res, res_size, owned);
      // If we weren't a leaf task, compute the conditions for being mapped
      // which is that all of our children are now mapped
      // Also test for whether we need to trigger any of our child
      // complete or committed operations before marking that we
      // are done executing
      bool need_complete = false;
      bool need_commit = false;
      {
        std::set<RtEvent> preconditions;
        {
          AutoLock child_lock(child_op_lock);
          // Only need to do this for executing and executed children
          // We know that any complete children are done
          for (std::map<Operation*,GenerationID>::const_iterator it = 
                executing_children.begin(); it != 
                executing_children.end(); it++)
          {
            preconditions.insert(it->first->get_mapped_event());
          }
          for (std::map<Operation*,GenerationID>::const_iterator it = 
                executed_children.begin(); it != executed_children.end(); it++)
          {
            preconditions.insert(it->first->get_mapped_event());
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
          single_task->handle_post_mapped(Runtime::merge_events(preconditions));
        else
          single_task->handle_post_mapped();
      } 
      if (need_complete)
        owner_task->trigger_children_complete();
      if (need_commit)
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_remote_contexts(void)
    //--------------------------------------------------------------------------
    {
      UniqueID local_uid = get_unique_id();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(local_uid);
      }
      for (std::map<AddressSpaceID,RemoteContext*>::const_iterator it = 
            remote_instances.begin(); it != remote_instances.end(); it++)
      {
        runtime->send_remote_context_free(it->first, rez);
      }
      remote_instances.clear();
    }

    //--------------------------------------------------------------------------
    void InnerContext::send_remote_context(AddressSpaceID remote_instance,
                                           RemoteContext *remote_ctx)
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
      AutoLock rem_lock(remote_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_version_owner_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      UniqueID context_uid;
      derez.deserialize(context_uid);
      InnerContext *local_ctx = runtime->find_context(context_uid);
      InnerContext *remote_ctx;
      derez.deserialize(remote_ctx);
      bool is_region;
      derez.deserialize(is_region);

      Serializer rez;
      rez.serialize(remote_ctx);
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionTreeNode *node = runtime->forest->get_node(handle);

        AddressSpaceID result = local_ctx->get_version_owner(node, source);
        rez.serialize(result);
        rez.serialize<bool>(true);
        rez.serialize(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        RegionTreeNode *node = runtime->forest->get_node(handle);

        AddressSpaceID result = local_ctx->get_version_owner(node, source);
        rez.serialize(result);
        rez.serialize<bool>(false);
        rez.serialize(handle);
      }
      runtime->send_version_owner_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void InnerContext::process_version_owner_response(RegionTreeNode *node,
                                                      AddressSpaceID result)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock tree_lock(tree_owner_lock);
#ifdef DEBUG_LEGION
        assert(region_tree_owners.find(node) == region_tree_owners.end());
#endif
        region_tree_owners[node] = 
          std::pair<AddressSpaceID,bool>(result, false/*remote only*/); 
        node->register_tracking_context(this);
        // Find the event to trigger
        std::map<RegionTreeNode*,RtUserEvent>::iterator finder = 
          pending_version_owner_requests.find(node);
#ifdef DEBUG_LEGION
        assert(finder != pending_version_owner_requests.end());
#endif
        to_trigger = finder->second;
        pending_version_owner_requests.erase(finder);
      }
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_version_owner_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      InnerContext *ctx;
      derez.deserialize(ctx);
      AddressSpaceID result;
      derez.deserialize(result);
      bool is_region;
      derez.deserialize(is_region);
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionTreeNode *node = runtime->forest->get_node(handle);
        ctx->process_version_owner_response(node, result);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        RegionTreeNode *node = runtime->forest->get_node(handle);
        ctx->process_version_owner_response(node, result);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_prepipeline_stage(const void *args)
    //--------------------------------------------------------------------------
    {
      const PrepipelineArgs *pargs = (const PrepipelineArgs*)args;
      pargs->context->process_prepipeline_stage();
      if (pargs->context->remove_reference())
        delete pargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_dependence_stage(const void *args)
    //--------------------------------------------------------------------------
    {
      const DependenceArgs *dargs = (const DependenceArgs*)args;
      dargs->context->process_dependence_stage();
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_post_end_task(const void *args)
    //--------------------------------------------------------------------------
    {
      const PostEndArgs *pargs = (const PostEndArgs*)args;
      pargs->proxy_this->process_post_end_tasks();
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_deferred_post_end_task(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredPostTaskArgs *pargs = ((const DeferredPostTaskArgs*)args);
      // If we have a start event, trigger that now
      if (pargs->started.exists())
        Runtime::trigger_event(pargs->started);
      if (pargs->instance.exists())
      {
        pargs->context->post_end_task(
                                    pargs->result, pargs->size, false/*owned*/);
        // Once we've copied the data then we can destroy the instance
        pargs->instance.destroy();
      }
      else
        pargs->context->post_end_task(
                                     pargs->result, pargs->size, true/*owned*/);
    }

    //--------------------------------------------------------------------------
    void InnerContext::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INLINE_CHILD_TASK_CALL);
      // Remove this child from our context
      unregister_child_operation(child);
      // Check to see if the child is predicated
      // If it is wait for it to resolve
      if (child->is_predicated_op())
      {
        // See if the predicate speculates false, if so return false
        // and then we are done.
        if (!child->get_predicate_value(executing_processor))
          return;
      }
      // Save the state of our physical regions
      std::vector<bool> phy_regions_mapped(physical_regions.size());
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
        phy_regions_mapped[idx] = is_region_mapped(idx);
      // Inline the child task
      child->perform_inlining();
      // Now see if the mapping state of any of our
      // originally mapped regions has changed
      std::set<ApEvent> wait_events;
      for (unsigned idx = 0; idx < phy_regions_mapped.size(); idx++)
      {
        if (phy_regions_mapped[idx] && !is_region_mapped(idx))
        {
          // Need to remap
          MapOp *op = runtime->get_available_map_op();
          op->initialize(this, physical_regions[idx]);
          wait_events.insert(op->get_completion_event());
          runtime->add_to_dependence_queue(this, executing_processor, op);
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
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    VariantImpl* InnerContext::select_inline_variant(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SELECT_INLINE_VARIANT_CALL);
      Mapper::SelectVariantInput input;
      Mapper::SelectVariantOutput output;
      input.processor = executing_processor;
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
      MapperManager *child_mapper = runtime->find_mapper(executing_processor,
                                                         child->map_id);
      child_mapper->invoke_select_task_variant(child, &input, &output);
      VariantImpl *variant_impl= runtime->find_variant_impl(child->task_id,
                                  output.chosen_variant, true/*can fail*/);
      if (variant_impl == NULL)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
          "Invalid mapper output from invoction of "
                      "'select_task_variant' on mapper %s. Mapper selected "
                      "an invalidate variant ID %ld for inlining of task %s "
                      "(UID %lld).", child_mapper->get_mapper_name(),
                      output.chosen_variant, child->get_task_name(), 
                      child->get_unique_id())
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::find_index_launch_space(const Domain &domain)
    //--------------------------------------------------------------------------
    {
      std::map<Domain,IndexSpace>::const_iterator finder = 
        index_launch_spaces.find(domain);
      if (finder != index_launch_spaces.end())
        return finder->second;
      IndexSpace result = runtime->find_or_create_index_launch_space(domain);
      index_launch_spaces[domain] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::execute_task_launch(TaskOp *task, bool index,
       LegionTrace *current_trace, bool silence_warnings, bool inlining_enabled)
    //--------------------------------------------------------------------------
    {
      bool inline_task = false;
      if (inlining_enabled)
        inline_task = task->select_task_options();
      // Now check to see if we're inling the task or just performing
      // a normal asynchronous task launch
      if (inline_task)
      {
        inline_child_task(task);
        // After we're done we can deactivate it since we
        // know that it will never be used again
        task->deactivate();
      }
      else
      {
        // Normal task launch, iterate over the context task's
        // regions and see if we need to unmap any of them
        std::vector<PhysicalRegion> unmapped_regions;
        if (!runtime->unsafe_launch)
          find_conflicting_regions(task, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          if (runtime->runtime_warnings && !silence_warnings)
          {
            if (index)
            {
              REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
                "WARNING: Runtime is unmapping and remapping "
                  "physical regions around execute_index_space call in "
                  "task %s (UID %lld).", get_task_name(), get_unique_id());
            }
            else
            {
              REPORT_LEGION_WARNING(LEGION_WARNING_RUNTIME_UNMAPPING_REMAPPING,
                "WARNING: Runtime is unmapping and remapping "
                  "physical regions around execute_task call in "
                  "task %s (UID %lld).", get_task_name(), get_unique_id());
            }
          }
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
            unmapped_regions[idx].impl->unmap_region();
        }
        // Issue the task call
        runtime->add_to_dependence_queue(this, executing_processor, task);
        // Remap any unmapped regions
        if (!unmapped_regions.empty())
          remap_unmapped_regions(current_trace, unmapped_regions);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::clone_local_fields(
           std::map<FieldSpace,std::vector<LocalFieldInfo> > &child_local) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child_local.empty());
#endif
      AutoLock local_lock(local_field_lock,1,false/*exclusive*/);
      if (local_fields.empty())
        return;
      for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
            fit = local_fields.begin(); fit != local_fields.end(); fit++)
      {
        std::vector<LocalFieldInfo> &child = child_local[fit->first];
        child.resize(fit->second.size());
        for (unsigned idx = 0; idx < local_fields.size(); idx++)
        {
          LocalFieldInfo &field = child[idx];
          field = fit->second[idx];
          field.ancestor = true; // mark that this is an ancestor field
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Top Level Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(Runtime *rt, UniqueID ctx_id)
      : InnerContext(rt, NULL, false/*full inner*/, dummy_requirements, 
                     dummy_indexes, dummy_mapped, ctx_id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(const TopLevelContext &rhs)
      : InnerContext(NULL, NULL, false,
                     dummy_requirements, dummy_indexes, dummy_mapped, 0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TopLevelContext::~TopLevelContext(void)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    TopLevelContext& TopLevelContext::operator=(const TopLevelContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    int TopLevelContext::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return -1;
    }

    //--------------------------------------------------------------------------
    void TopLevelContext::pack_remote_context(Serializer &rez, 
                                              AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(true); // top level context, all we need to pack
    }

    //--------------------------------------------------------------------------
    TaskContext* TopLevelContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    void TopLevelContext::add_to_post_task_queue(TaskContext *ctx, 
        RtEvent wait_on, const void *result, size_t size, PhysicalInstance inst)
    //--------------------------------------------------------------------------
    {
      // Since we're the top-level task we should just handle this here
#ifdef DEBUG_LEGION
      assert(!inst.exists()); // This makes it safe to wait now
#endif
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      ctx->post_end_task(result, size, true/*owned*/);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID TopLevelContext::get_version_owner(RegionTreeNode *node, 
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // We're the top-level task, so we handle the request on the node
      // that made the region
      const AddressSpaceID owner_space = node->get_owner_space();
      if (owner_space == runtime->address_space)
        return InnerContext::get_version_owner(node, source);
#ifdef DEBUG_LEGION
      assert(source == runtime->address_space); // should always be local
#endif
      // See if we already have it, or we already sent a request for it
      bool send_request = false;
      RtEvent wait_on;
      {
        AutoLock tree_lock(tree_owner_lock);
        std::map<RegionTreeNode*,
                 std::pair<AddressSpaceID,bool> >::const_iterator finder =
          region_tree_owners.find(node);
        if (finder != region_tree_owners.end())
          return finder->second.first;
        // See if we already have an outstanding request
        std::map<RegionTreeNode*,RtUserEvent>::const_iterator request_finder =
          pending_version_owner_requests.find(node);
        if (request_finder == pending_version_owner_requests.end())
        {
          // We haven't sent the request yet, so do that now
          RtUserEvent request_event = Runtime::create_rt_user_event();
          pending_version_owner_requests[node] = request_event;
          wait_on = request_event;
          send_request = true;
        }
        else
          wait_on = request_finder->second;
      }
      if (send_request)
      {
        Serializer rez;
        rez.serialize(context_uid);
        rez.serialize<InnerContext*>(this);
        if (node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(node->as_partition_node()->handle);
        }
        // Send it to the owner space 
        runtime->send_version_owner_request(owner_space, rez);
      }
      wait_on.wait();
      // Retake the lock in read-only mode and get the answer
      AutoLock tree_lock(tree_owner_lock,1,false/*exclusive*/);
      std::map<RegionTreeNode*,
               std::pair<AddressSpaceID,bool> >::const_iterator finder = 
        region_tree_owners.find(node);
#ifdef DEBUG_LEGION
      assert(finder != region_tree_owners.end());
#endif
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    InnerContext* TopLevelContext::find_outermost_local_context(
                                                         InnerContext *previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;
    }

    //--------------------------------------------------------------------------
    InnerContext* TopLevelContext::find_top_context(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    VersionInfo& TopLevelContext::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *new VersionInfo();
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* TopLevelContext::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Remote Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTask::RemoteTask(RemoteContext *own)
      : owner(own), context_index(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteTask::RemoteTask(const RemoteTask &rhs)
      : owner(NULL)
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
    UniqueID RemoteTask::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_context_uid();
    }

    //--------------------------------------------------------------------------
    unsigned RemoteTask::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::set_context_index(unsigned index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }
    
    //--------------------------------------------------------------------------
    int RemoteTask::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_depth();
    }

    //--------------------------------------------------------------------------
    const char* RemoteTask::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *task_impl = owner->runtime->find_task_impl(task_id);
      return task_impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool RemoteTask::has_trace(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Remote Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(Runtime *rt, UniqueID context_uid)
      : InnerContext(rt, NULL, false/*full inner*/, remote_task.regions, 
          local_parent_req_indexes, local_virtual_mapped, 
          context_uid, true/*remote*/),
        parent_ctx(NULL), depth(-1), top_level_context(false), 
        remote_task(RemoteTask(this))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(const RemoteContext &rhs)
      : InnerContext(NULL, NULL, false, rhs.regions, local_parent_req_indexes,
          local_virtual_mapped, 0, true), remote_task(RemoteTask(this))
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteContext::~RemoteContext(void)
    //--------------------------------------------------------------------------
    {
      if (!local_fields.empty())
      {
        // If we have any local fields then tell field space that
        // we can remove them and then clear them so that the base
        // InnerContext destructor doesn't try to deallocate them
        for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
              it = local_fields.begin(); it != local_fields.end(); it++)
        {
          const std::vector<LocalFieldInfo> &infos = it->second;
          std::vector<FieldID> to_remove;
          for (unsigned idx = 0; idx < infos.size(); idx++)
          {
            if (infos[idx].ancestor)
              continue;
            to_remove.push_back(infos[idx].fid);
          }
          if (!to_remove.empty())
            runtime->forest->remove_local_fields(it->first, to_remove);
        }
        local_fields.clear();
      } 
    }

    //--------------------------------------------------------------------------
    RemoteContext& RemoteContext::operator=(const RemoteContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    int RemoteContext::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    //--------------------------------------------------------------------------
    Task* RemoteContext::get_task(void)
    //--------------------------------------------------------------------------
    {
      return &remote_task;
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_outermost_local_context(
                                                         InnerContext *previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;
    }
    
    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_top_context(void)
    //--------------------------------------------------------------------------
    {
      if (top_level_context)
        return this;
      return find_parent_context()->find_top_context();
    }
    
    //--------------------------------------------------------------------------
    VersionInfo& RemoteContext::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!top_level_context);
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const std::vector<VersionInfo>* RemoteContext::get_version_infos(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!top_level_context);
#endif
      return &version_infos;
    }

    //--------------------------------------------------------------------------
    TaskContext* RemoteContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      if (top_level_context)
        return NULL;
      // See if we already have it
      if (parent_ctx != NULL)
        return parent_ctx;
#ifdef DEBUG_LEGION
      assert(parent_context_uid != 0);
#endif
      // THIS IS ONLY SAFE BECAUSE THIS FUNCTION IS NEVER CALLED BY
      // A MESSAGE IN THE CONTEXT_VIRTUAL_CHANNEL
      parent_ctx = runtime->find_context(parent_context_uid);
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      remote_task.parent_task = parent_ctx->get_task();
      return parent_ctx;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID RemoteContext::get_version_owner(RegionTreeNode *node,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = node->get_owner_space(); 
      // If we are the top-level context then we handle the request
      // on the node that made the region
      if (top_level_context && (owner_space == runtime->address_space))
        return InnerContext::get_version_owner(node, source);
        // Otherwise we fall through and issue the request to the 
        // the node that actually made the region
#ifdef DEBUG_LEGION
      assert(source == runtime->address_space); // should always be local
#endif
      // See if we already have it, or we already sent a request for it
      bool send_request = false;
      RtEvent wait_on;
      {
        AutoLock tree_lock(tree_owner_lock);
        std::map<RegionTreeNode*,
                 std::pair<AddressSpaceID,bool> >::const_iterator finder =
          region_tree_owners.find(node);
        if (finder != region_tree_owners.end())
          return finder->second.first;
        // See if we already have an outstanding request
        std::map<RegionTreeNode*,RtUserEvent>::const_iterator request_finder =
          pending_version_owner_requests.find(node);
        if (request_finder == pending_version_owner_requests.end())
        {
          // We haven't sent the request yet, so do that now
          RtUserEvent request_event = Runtime::create_rt_user_event();
          pending_version_owner_requests[node] = request_event;
          wait_on = request_event;
          send_request = true;
        }
        else
          wait_on = request_finder->second;
      }
      if (send_request)
      {
        Serializer rez;
        rez.serialize(context_uid);
        rez.serialize<InnerContext*>(this);
        if (node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(node->as_partition_node()->handle);
        }
        // Send it to the owner space if we are the top-level context
        // otherwise we send it to the owner of the context
        const AddressSpaceID target = top_level_context ? owner_space :  
                          runtime->get_runtime_owner(context_uid);
        runtime->send_version_owner_request(target, rez);
      }
      wait_on.wait();
      // Retake the lock in read-only mode and get the answer
      AutoLock tree_lock(tree_owner_lock,1,false/*exclusive*/);
      std::map<RegionTreeNode*,
               std::pair<AddressSpaceID,bool> >::const_iterator finder = 
        region_tree_owners.find(node);
#ifdef DEBUG_LEGION
      assert(finder != region_tree_owners.end());
#endif
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    void RemoteContext::find_parent_version_info(unsigned index, unsigned depth,
                     const FieldMask &version_mask, InnerContext *context,
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(regions.size() == virtual_mapped.size()); 
#endif
      // If this isn't one of our original region requirements then 
      // we don't have any versions that the child won't discover itself
      // Same if the region was not virtually mapped
      if ((index >= virtual_mapped.size()) || !virtual_mapped[index])
        return;
#ifdef DEBUG_LEGION
      assert(index < version_infos.size());
#endif
      version_infos[index].clone_to_depth(depth, version_mask, context,
                                          version_info, ready_events);
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_parent_physical_context(unsigned index,
                                                          LogicalRegion *handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == parent_req_indexes.size());
#endif     
      const unsigned owner_size = virtual_mapped.size();
      if (index < owner_size)
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
                                            parent_req_indexes[index], handle);
        else // We mapped a physical instance so we're it
        {
          if (handle != NULL)
            *handle = regions[index].region;
          return this;
        }
      }
      else // We created it
      {
        // But we're the remote note, so we don't have updated created
        // requirements or returnable privileges so we need to see if
        // we already know the answer and if not, ask the owner context
        RtEvent wait_on;
        RtUserEvent request;
        {
          AutoLock rem_lock(remote_lock);
          std::map<unsigned,InnerContext*>::const_iterator finder = 
            physical_contexts.find(index);
          if (finder != physical_contexts.end())
          {
            if (handle != NULL)
            {
              std::map<unsigned,LogicalRegion>::const_iterator handle_finder =
                physical_handles.find(index);
#ifdef DEBUG_LEGION
              assert(handle_finder != physical_handles.end());
#endif
              *handle = handle_finder->second;
            }
            return finder->second;
          }
          std::map<unsigned,RtEvent>::const_iterator pending_finder = 
            pending_physical_contexts.find(index);
          if (pending_finder == pending_physical_contexts.end())
          {
            // Make a new request
            request = Runtime::create_rt_user_event();
            pending_physical_contexts[index] = request;
            wait_on = request;
          }
          else // Already sent it so just get the wait event
            wait_on = pending_finder->second;
        }
        if (request.exists())
        {
          // Send the request
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(context_uid);
            rez.serialize(index);
            rez.serialize(this);
            rez.serialize(request);
          }
          const AddressSpaceID target = runtime->get_runtime_owner(context_uid);
          runtime->send_remote_context_physical_request(target, rez);
        }
        // Wait for the result to come back to us
        wait_on.wait();
        // When we wake up it should be there
        AutoLock rem_lock(remote_lock, 1, false/*exclusive*/);
        // Get the handle if we need it
        if (handle != NULL)
        {
          std::map<unsigned,LogicalRegion>::const_iterator handle_finder =
            physical_handles.find(index);
#ifdef DEBUG_LEGION
          assert(handle_finder != physical_handles.end());
#endif
          *handle = handle_finder->second;
        }
#ifdef DEBUG_LEGION
        assert(physical_contexts.find(index) != physical_contexts.end());
#endif
        return physical_contexts[index]; 
      }
    }

    //--------------------------------------------------------------------------
    void RemoteContext::record_using_physical_context(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
#endif
      AutoLock rem_lock(remote_lock);
      local_physical_contexts.insert(handle);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      if (!top_level_context)
      {
#ifdef DEBUG_LEGION
        assert(regions.size() == virtual_mapped.size());
#endif
        // Deactivate any region trees that we didn't virtually map
        for (unsigned idx = 0; idx < regions.size(); idx++)
          if (!virtual_mapped[idx])
            runtime->forest->invalidate_versions(tree_context, 
                                                 regions[idx].region);
        // Also invalidate any of our local physical context regions
        for (std::set<LogicalRegion>::const_iterator it = 
              local_physical_contexts.begin(); it != 
              local_physical_contexts.end(); it++)
          runtime->forest->invalidate_versions(tree_context, *it);
      }
      else
        runtime->forest->invalidate_all_versions(tree_context);
      // Now we can free our region tree context
      runtime->free_region_tree_context(tree_context);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::invalidate_remote_tree_contexts(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_invalidates;
      derez.deserialize(num_invalidates);
      for (unsigned idx = 0; idx < num_invalidates; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        LogicalRegion handle;
        derez.deserialize(handle);
        // Check to see if we actually looked this up
        std::map<unsigned,InnerContext*>::const_iterator finder = 
          physical_contexts.find(index);
        if (finder != physical_contexts.end())
          runtime->forest->invalidate_versions(finder->second->get_context(),
                                               handle);
      }
      // Then do our normal tree invalidation
      invalidate_region_tree_contexts();
    }

    //--------------------------------------------------------------------------
    void RemoteContext::unpack_remote_context(Deserializer &derez,
                                              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, REMOTE_UNPACK_CONTEXT_CALL);
      derez.deserialize(top_level_context);
      // If we're the top-level context then we're already done
      if (top_level_context)
        return;
      derez.deserialize(depth);
      WrapperReferenceMutator mutator(preconditions);
      remote_task.unpack_external_task(derez, runtime, &mutator);
      local_parent_req_indexes.resize(remote_task.regions.size()); 
      for (unsigned idx = 0; idx < local_parent_req_indexes.size(); idx++)
        derez.deserialize(local_parent_req_indexes[idx]);
      size_t num_virtual;
      derez.deserialize(num_virtual);
      local_virtual_mapped.resize(regions.size(), false);
      for (unsigned idx = 0; idx < num_virtual; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        local_virtual_mapped[index] = true;
      }
      version_infos.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (virtual_mapped[idx])
          version_infos[idx].unpack_version_info(derez, runtime, preconditions);
        else
          version_infos[idx].unpack_version_numbers(derez, runtime->forest);
      }
      derez.deserialize(remote_completion_event);
      derez.deserialize(parent_context_uid);
      // Unpack any local fields that we have
      unpack_local_field_update(derez);
      
      // See if we can find our parent task, if not don't worry about it
      // DO NOT CHANGE THIS UNLESS YOU THINK REALLY HARD ABOUT VIRTUAL 
      // CHANNELS AND HOW CONTEXT META-DATA IS MOVED!
      parent_ctx = runtime->find_context(parent_context_uid, true/*can fail*/);
      if (parent_ctx != NULL)
        remote_task.parent_task = parent_ctx->get_task();
    }

    //--------------------------------------------------------------------------
    void RemoteContext::unpack_local_field_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_field_spaces;
      derez.deserialize(num_field_spaces);
      if (num_field_spaces == 0)
        return;
      for (unsigned fidx = 0; fidx < num_field_spaces; fidx++)
      {
        FieldSpace handle;
        derez.deserialize(handle);
        size_t num_local;
        derez.deserialize(num_local); 
        std::vector<FieldID> fields(num_local);
        std::vector<size_t> field_sizes(num_local);
        std::vector<CustomSerdezID> serdez_ids(num_local);
        std::vector<unsigned> indexes(num_local);
        {
          // Take the lock for updating this data structure
          AutoLock local_lock(local_field_lock);
          std::vector<LocalFieldInfo> &infos = local_fields[handle];
          infos.resize(num_local);
          for (unsigned idx = 0; idx < num_local; idx++)
          {
            LocalFieldInfo &info = infos[idx];
            derez.deserialize(info);
            // Update data structures for notifying the field space
            fields[idx] = info.fid;
            field_sizes[idx] = info.size;
            serdez_ids[idx] = info.serdez;
            indexes[idx] = info.index;
          }
        }
        runtime->forest->update_local_fields(handle, fields, field_sizes,
                                             serdez_ids, indexes);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::handle_local_field_update(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteContext *context;
      derez.deserialize(context);
      context->unpack_local_field_update(derez);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::handle_physical_request(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      unsigned index;
      derez.deserialize(index);
      RemoteContext *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      // Always defer this in case it blocks, we can't block the virtual channel
      RemotePhysicalRequestArgs args;
      args.context_uid = context_uid;
      args.target = target;
      args.index = index;
      args.source = source;
      args.to_trigger = to_trigger;
      args.runtime = runtime;
      runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::defer_physical_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemotePhysicalRequestArgs *rargs = 
        (const RemotePhysicalRequestArgs*)args;
      InnerContext *local = rargs->runtime->find_context(rargs->context_uid);
      LogicalRegion handle = LogicalRegion::NO_REGION;
      InnerContext *result = 
        local->find_parent_physical_context(rargs->index, &handle);
      // Also need the logical region to send back so we can tell the 
      // possibly remote context the name of the top-level region
      // so it can invalidate it when it's done using it
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(rargs->target);
        rez.serialize(rargs->index);
        rez.serialize(result->context_uid);
        rez.serialize(handle);
        rez.serialize(rargs->to_trigger);
      }
      rargs->runtime->send_remote_context_physical_response(rargs->source, rez);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::set_physical_context_result(unsigned index,
                                                    InnerContext *result,
                                                    LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoLock rem_lock(remote_lock);
#ifdef DEBUG_LEGION
      assert(physical_contexts.find(index) == physical_contexts.end());
      assert(physical_handles.find(index) == physical_handles.end());
#endif
      physical_contexts[index] = result;
      physical_handles[index] = handle;
      std::map<unsigned,RtEvent>::iterator finder = 
        pending_physical_contexts.find(index);
#ifdef DEBUG_LEGION
      assert(finder != pending_physical_contexts.end());
#endif
      pending_physical_contexts.erase(finder);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::handle_physical_response(Deserializer &derez,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteContext *target;
      derez.deserialize(target);
      unsigned index;
      derez.deserialize(index);
      UniqueID result_uid;
      derez.deserialize(result_uid);
      LogicalRegion handle;
      derez.deserialize(handle);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      InnerContext *result = runtime->find_context(result_uid, true/*weak*/);
      if (result == NULL)
      {
        // Launch a continuation in case we need to page in the context
        // We obviously can't block the virtual channel
        RemotePhysicalResponseArgs args;
        args.target = target;
        args.index = index;
        args.result_uid = result_uid;
        args.handle = handle;
        args.runtime = runtime;
        RtEvent done = 
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
        Runtime::trigger_event(to_trigger, done);
      }
      else
      {
        target->set_physical_context_result(index, result, handle);
        // Also have to tell the actual context someone is using it
        // in case it is also remote
        if (handle.exists())
          result->record_using_physical_context(handle);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::defer_physical_response(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemotePhysicalResponseArgs *rargs = 
        (const RemotePhysicalResponseArgs*)args;
      InnerContext *result = rargs->runtime->find_context(rargs->result_uid);
      rargs->target->set_physical_context_result(rargs->index, result,
          rargs->handle);
      // Also have to tell the actual context someone is using it
      // in case it is also remote
      if (rargs->handle.exists())
        result->record_using_physical_context(rargs->handle);
    }

    /////////////////////////////////////////////////////////////
    // Leaf Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LeafContext::LeafContext(Runtime *rt, TaskOp *owner)
      : TaskContext(rt, owner, owner->regions)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LeafContext::LeafContext(const LeafContext &rhs)
      : TaskContext(NULL, NULL, rhs.regions)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LeafContext::~LeafContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LeafContext& LeafContext::operator=(const LeafContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext LeafContext::get_context(void) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RegionTreeContext();
    }

    //--------------------------------------------------------------------------
    ContextID LeafContext::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::pack_remote_context(Serializer &rez,AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool LeafContext::attempt_children_complete(void)
    //--------------------------------------------------------------------------
    {
      AutoLock leaf(leaf_lock);
      if (!children_complete_invoked)
      {
        children_complete_invoked = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool LeafContext::attempt_children_commit(void)
    //--------------------------------------------------------------------------
    {
      AutoLock leaf(leaf_lock);
      if (!children_commit_invoked)
      {
        children_commit_invoked = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void LeafContext::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    VariantImpl* LeafContext::select_inline_variant(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool LeafContext::is_leaf_context(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(RegionTreeForest *forest,
                                         const void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_SPACE_CREATION,
        "Illegal index space creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::union_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_UNION_INDEX_SPACES,
        "Illegal union index spaces performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::intersect_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERSECT_INDEX_SPACES,
        "Illegal intersect index spaces performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::subtract_index_spaces(RegionTreeForest *forest,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_SUBTRACT_INDEX_SPACES,
        "Illegal subtract index spaces performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_SPACE_DELETION,
        "Illegal index space deletion performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_partition(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_PARTITION_DELETION,
        "Illegal index partition deletion performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_equal_partition(RegionTreeForest *forest,
                                             IndexSpace parent,
                                             IndexSpace color_space,
                                             size_t granularity, Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EQUAL_PARTITION_CREATION,
        "Illegal equal partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_union(
                                          RegionTreeForest *forest,
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          IndexSpace color_space,
                                          PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_UNION_PARTITION_CREATION,
        "Illegal union partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
                                                RegionTreeForest *forest,
                                                IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                IndexSpace color_space,
                                                PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERSECTION_PARTITION_CREATION,
        "Illegal intersection partition creation performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_difference(
                                                      RegionTreeForest *forest,
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      IndexSpace color_space,
                                                      PartitionKind kind,
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DIFFERENCE_PARTITION_CREATION,
        "Illegal difference partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    Color LeafContext::create_cross_product_partitions(RegionTreeForest *forest,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                   std::map<IndexSpace,IndexPartition> &handles,
                                                       PartitionKind kind,
                                                       Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_CROSS_PRODUCT_PARTITION,
        "Illegal create cross product partitions performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_association(LogicalRegion domain,
                                         LogicalRegion domain_parent,
                                         FieldID domain_fid, IndexSpace range,
                                         MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_ASSOCIATION,
        "Illegal create association performed in leaf task "
                     "%s (ID %lld)", get_task_name(),get_unique_id())
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_restricted_partition(
                                                RegionTreeForest *forest,
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
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_RESTRICTED_PARTITION,
        "Illegal create restricted partition performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_field(
                                                RegionTreeForest *forest,
                                                LogicalRegion handle,
                                                LogicalRegion parent_priv,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                Color color,
                                                MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_FIELD,
        "Illegal partition by field performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image(
                                              RegionTreeForest *forest,
                                              IndexSpace handle,
                                              LogicalPartition projection,
                                              LogicalRegion parent,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color,
                                              MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_IMAGE,
        "Illegal partition by image performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image_range(
                                              RegionTreeForest *forest,
                                              IndexSpace handle,
                                              LogicalPartition projection,
                                              LogicalRegion parent,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color,
                                              MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_IMAGE_RANGE,
        "Illegal partition by image range performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage(
                                                RegionTreeForest *forest,
                                                IndexPartition projection,
                                                LogicalRegion handle,
                                                LogicalRegion parent,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                PartitionKind part_kind,
                                                Color color,
                                                MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_PREIMAGE,
        "Illegal partition by preimage performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage_range(
                                                RegionTreeForest *forest,
                                                IndexPartition projection,
                                                LogicalRegion handle,
                                                LogicalRegion parent,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                PartitionKind part_kind,
                                                Color color,
                                                MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_PREIMAGE_RANGE,
        "Illegal partition by preimage range performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_pending_partition(
                                              RegionTreeForest *forest,
                                              IndexSpace parent,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_PENDING_PARTITION,
        "Illegal create pending partition performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_UNION,
        "Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_UNION,
        "Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_INTERSECTION,
        "Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_INTERSECTION,
        "Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_difference(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                                  IndexSpace initial,
                                          const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_DIFFERENCE,
        "Illegal create index space difference performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_FIELD_SPACE,
        "Illegal create field space performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return FieldSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_field_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DESTROY_FIELD_SPACE,
        "Illegal destroy field space performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    FieldID LeafContext::allocate_field(RegionTreeForest *forest,
                                        FieldSpace space, size_t field_size,
                                        FieldID fid, bool local,
                                        CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION,
        "Illegal non-local field allocation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_FIELD_DESTRUCTION,
        "Illegal field destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_fields(RegionTreeForest *forest,
                                      FieldSpace space,
                                      const std::vector<size_t> &sizes,
                                      std::vector<FieldID> &resuling_fields,
                                      bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION2,
        "Illegal non-local field allocation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_fields(FieldSpace space, 
                                  const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_FIELD_DESTRUCTION2,
        "Illegal field destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    LogicalRegion LeafContext::create_logical_region(RegionTreeForest *forest,
                                                     IndexSpace index_space,
                                                     FieldSpace field_space,
                                                     bool task_local)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_REGION_CREATION,
        "Illegal region creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return LogicalRegion::NO_REGION;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_REGION_DESTRUCTION,
        "Illegal region destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_DESTRUCTION,
        "Illegal partition destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    FieldAllocator LeafContext::create_field_allocator(
                                   Legion::Runtime *external, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_FIELD_ALLOCATION,
        "Illegal create field allocation requested in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FieldAllocator(handle, this, external);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_TASK_CALL,
        "Illegal execute task call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_index_space(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal execute index space call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_index_space(const IndexTaskLauncher &launcher,
                                            ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal execute index space call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::map_region(const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_MAP_REGION,
        "Illegal map_region operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return PhysicalRegion();
    }

    //--------------------------------------------------------------------------
    ApEvent LeafContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_REMAP_OPERATION,
        "Illegal remap operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_UNMAP_OPERATION,
        "Illegal unmap operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_FILL_OPERATION_CALL,
        "Illegal fill operation call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_FILL_OPERATION_CALL,
        "Illegal index fill operation call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_COPY_FILL_OPERATION_CALL,
        "Illegal copy operation call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INDEX_COPY_OPERATION,
        "Illegal index copy operation call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_acquire(const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_ACQUIRE_OPERATION,
        "Illegal acquire operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_release(const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_RELEASE_OPERATION,
        "Illegal release operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::attach_resource(const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_ATTACH_RESOURCE_OPERATION,
        "Illegal attach resource operation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return PhysicalRegion();
    }
    
    //--------------------------------------------------------------------------
    Future LeafContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_RESOURCE_OPERATION,
        "Illegal detach resource operation performed in leaf "
                      "task %s (ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_must_epoch(const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_EXECUTE_MUST_EPOCH,
        "Illegal Legion execute must epoch call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_timing_measurement(const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_TIMING_MEASUREMENT,
        "Illegal timing measurement operation in leaf task %s"
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_mapping_fence(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_MAPPING_FENCE_CALL,
        "Illegal legion mapping fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_execution_fence(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_EXECUTION_FENCE_CALL,
        "Illegal Legion execution fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::complete_frame(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_COMPLETE_FRAME_CALL,
        "Illegal Legion complete frame call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const Future &f)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PREDICATE_CREATION,
        "Illegal predicate creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Predicate();
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::predicate_not(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NOT_PREDICATE_CREATION,
        "Illegal NOT predicate creation in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Predicate();
    }
    
    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PREDICATE_CREATION,
        "Illegal predicate creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Predicate();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_predicate_future(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_GET_PREDICATE_FUTURE,
        "Illegal get predicate future performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    unsigned LeafContext::register_new_child_operation(Operation *op,
                    const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    unsigned LeafContext::register_new_close_operation(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    unsigned LeafContext::register_new_summary_operation(TraceSummaryOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_to_prepipeline_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_to_dependence_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
        const void *result, size_t size, PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::register_executing_child(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::unregister_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    ApEvent LeafContext::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent LeafContext::perform_fence_analysis(Operation *op, 
                                                bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::update_current_fence(FenceOp *op, 
                                           bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent LeafContext::get_current_mapping_fence_event(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent LeafContext::get_current_execution_fence_event(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }


    //--------------------------------------------------------------------------
    void LeafContext::begin_trace(TraceID tid, bool logical_only)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_BEGIN_TRACE,
        "Illegal Legion begin trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_END_TRACE,
        "Illegal Legion end trace call in leaf task %s (ID %lld)",
                     get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::begin_static_trace(const std::set<RegionTreeID> *managed)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_BEGIN_STATIC_TRACE,
        "Illegal Legion begin static trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_static_trace(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_BEGIN_STATIC_TRACE,
        "Illegal Legion end static trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::record_previous_trace(LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::invalidate_trace_cache(
                                     LegionTrace *trace, Operation *invalidator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::record_blocking_call(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_frame(FrameOp *frame, ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::perform_frame_issue(FrameOp *frame, ApEvent frame_term)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::finish_frame(ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::increment_outstanding(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::decrement_outstanding(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent LeafContext::decrement_pending(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent LeafContext::decrement_pending(bool need_deferral)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_parent_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_parent_physical_context(unsigned index,
                                                          LogicalRegion *handle)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void LeafContext::find_parent_version_info(unsigned index, unsigned depth,
                     const FieldMask &version_mask, InnerContext *context,
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_outermost_local_context(InnerContext *prev)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_top_context(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void LeafContext::initialize_region_tree_contexts(
            const std::vector<RegionRequirement> &clone_requirements,
            const std::vector<ApUserEvent> &unmap_events,
            std::set<ApEvent> &preconditions, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void LeafContext::send_back_created_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceView* LeafContext::create_instance_top_view(
                PhysicalManager *manager, AddressSpaceID source, RtEvent *ready)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_task(const void *res, size_t res_size, bool owned,
                               PhysicalInstance deferred_result_instance)
    //--------------------------------------------------------------------------
    {
      if (overhead_tracker != NULL)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff = current - previous_profiling_time;
        overhead_tracker->application_time += diff;
      }
      // Unmap any physical regions that we mapped
      for (std::vector<PhysicalRegion>::const_iterator it = 
            physical_regions.begin(); it != physical_regions.end(); it++)
      {
        if (it->is_mapped())
          it->impl->unmap_region();
      }
      // Mark that we are done executing this operation
      // We're not actually done until we have registered our pending
      // decrement of our parent task and recorded any profiling
      if (!pending_done.has_triggered())
        owner_task->complete_execution(pending_done);
      else
        owner_task->complete_execution();
      // Grab some information before doing the next step in case it
      // results in the deletion of 'this'
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      const TaskID owner_task_id = owner_task->task_id;
#endif
      Runtime *runtime_ptr = runtime;
      // Tell the parent context that we are ready for post-end
      // Make a copy of the results if necessary
      TaskContext *parent_ctx = owner_task->get_context();
      if (deferred_result_instance.exists())
      {
        RtEvent result_ready(Processor::get_current_finish_event());
        parent_ctx->add_to_post_task_queue(this, result_ready,
                      res, res_size, deferred_result_instance);
      }
      else if (!owned)
      {
        void *result_copy = malloc(res_size);
        memcpy(result_copy, res, res_size);
        parent_ctx->add_to_post_task_queue(this, RtEvent::NO_RT_EVENT,
                  result_copy, res_size, PhysicalInstance::NO_INST);
      }
      else
        parent_ctx->add_to_post_task_queue(this, RtEvent::NO_RT_EVENT,
                          res, res_size, PhysicalInstance::NO_INST);
#ifdef DEBUG_LEGION
      runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                     false/*meta*/);
#else
      runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    void LeafContext::post_end_task(const void *res, size_t res_size,bool owned)
    //--------------------------------------------------------------------------
    {
      // Safe to cast to a single task here because this will never
      // be called while inlining an index space task
#ifdef DEBUG_LEGION
      SingleTask *single_task = dynamic_cast<SingleTask*>(owner_task);
      assert(single_task != NULL);
#else
      SingleTask *single_task = static_cast<SingleTask*>(owner_task);
#endif
      // Handle the future result
      single_task->handle_future(res, res_size, owned);
      bool need_complete = false;
      bool need_commit = false;
      {
        AutoLock leaf(leaf_lock);
#ifdef DEBUG_LEGION
        assert(!task_executed);
#endif
        // Now that we know the last registration has taken place we
        // can mark that we are done executing
        task_executed = true;
        if (!children_complete_invoked)
        {
          need_complete = true;
          children_complete_invoked = true;
        }
        if (!children_commit_invoked)
        {
          need_commit = true;
          children_commit_invoked = true;
        }
      } 
      if (need_complete)
        owner_task->trigger_children_complete();
      if (need_commit)
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_acquisition(AcquireOp *op, 
                                      const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::remove_acquisition(ReleaseOp *op,
                                         const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_restriction(AttachOp *op, InstanceManager *instance,
                                      const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::remove_restriction(DetachOp *op,
                                         const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::release_restrictions(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool LeafContext::has_restrictions(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void LeafContext::perform_restricted_analysis(const RegionRequirement &req,
                                                  RestrictInfo &restrict_info)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::record_dynamic_collective_contribution(
                                          DynamicCollective dc, const Future &f) 
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::find_collective_contributions(DynamicCollective dc, 
                                             std::vector<Future> &contributions) 
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskPriority LeafContext::get_current_priority(void) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::set_current_priority(TaskPriority priority)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    /////////////////////////////////////////////////////////////
    // Inline Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineContext::InlineContext(Runtime *rt, TaskContext *enc, TaskOp *child)
      : TaskContext(rt, child, child->regions), 
        enclosing(enc), inline_task(child)
    //--------------------------------------------------------------------------
    {
      executing_processor = enclosing->get_executing_processor();
      physical_regions.resize(regions.size());
      parent_req_indexes.resize(regions.size());
      // Now update the parent regions so that they are valid with
      // respect to the outermost context
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        unsigned index = enclosing->find_parent_region(idx, child);
        parent_req_indexes[idx] = index;
        if (index < enclosing->regions.size())
        {
          child->regions[idx].parent = enclosing->regions[index].parent;
          physical_regions[idx] = enclosing->get_physical_region(index);
        }
        else
        {
          // This is a created requirements, so we have to make a copy
          RegionRequirement copy;
          enclosing->clone_requirement(index, copy);
          child->regions[idx].parent = copy.parent;
          // physical regions are empty becaue they are virtual
        }
      }
    }

    //--------------------------------------------------------------------------
    InlineContext::InlineContext(const InlineContext &rhs)
      : TaskContext(NULL, NULL, rhs.regions), enclosing(NULL), inline_task(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InlineContext::~InlineContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineContext& InlineContext::operator=(const InlineContext &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    RegionTreeContext InlineContext::get_context(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_context();
    }

    //--------------------------------------------------------------------------
    ContextID InlineContext::get_context_id(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_context_id();
    }

    //--------------------------------------------------------------------------
    UniqueID InlineContext::get_context_uid(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_unique_id();
    }

    //--------------------------------------------------------------------------
    int InlineContext::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_depth();
    }

    //--------------------------------------------------------------------------
    void InlineContext::pack_remote_context(Serializer &rez,
                                            AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool InlineContext::attempt_children_complete(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool InlineContext::attempt_children_commit(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void InlineContext::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      enclosing->inline_child_task(child);
    }

    //--------------------------------------------------------------------------
    VariantImpl* InlineContext::select_inline_variant(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      return enclosing->select_inline_variant(child);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space(RegionTreeForest *forest,
                                         const void *realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space(forest, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::union_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      return enclosing->union_index_spaces(forest, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::intersect_index_spaces(RegionTreeForest *forest,
                                          const std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      return enclosing->intersect_index_spaces(forest, spaces);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::subtract_index_spaces(RegionTreeForest *forest,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      return enclosing->subtract_index_spaces(forest, left, right);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      enclosing->destroy_index_space(handle);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_index_partition(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      enclosing->destroy_index_partition(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_equal_partition(
                                                RegionTreeForest *forest,
                                                IndexSpace parent,
                                                IndexSpace color_space,
                                                size_t granularity, Color color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_equal_partition(forest, parent, color_space,
                                               granularity, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_union(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      IndexSpace color_space,
                                      PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_union(forest, parent, handle1,
                                                  handle2, color_space,
                                                  kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_intersection(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      IndexSpace color_space,
                                      PartitionKind kind, Color color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_intersection(forest, parent,
                                          handle1, handle2, color_space, 
                                          kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_difference(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      IndexSpace color_space,
                                      PartitionKind kind, Color color)   
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_difference(forest, parent,
                          handle1, handle2, color_space, kind, color);
    }

    //--------------------------------------------------------------------------
    Color InlineContext::create_cross_product_partitions(
                                                       RegionTreeForest *forest,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                  std::map<IndexSpace,IndexPartition> &handles,
                                                       PartitionKind kind,
                                                       Color color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_cross_product_partitions(forest, handle1,handle2,
                                                       handles, kind, color);
    }

    //--------------------------------------------------------------------------
    void InlineContext::create_association(LogicalRegion domain,
                                           LogicalRegion domain_parent,
                                           FieldID domain_fid,
                                           IndexSpace range,
                                           MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      enclosing->create_association(domain, domain_parent, domain_fid, 
                                    range, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_restricted_partition(
                                                RegionTreeForest *forest,
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
      return enclosing->create_restricted_partition(forest, parent, color_space,
                                 transform, transform_size, extent, extent_size,
                                 part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_field(
                                                      RegionTreeForest *forest,
                                                      LogicalRegion handle,
                                                      LogicalRegion parent_priv,
                                                      FieldID fid,
                                                      IndexSpace color_space,
                                                      Color color,
                                                      MapperID id, 
                                                      MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_field(forest, handle, parent_priv,
                                            fid, color_space, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_image(
                                                    RegionTreeForest *forest,
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
      return enclosing->create_partition_by_image(forest, handle, projection,
                          parent, fid, color_space, part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_image_range(
                                                    RegionTreeForest *forest,
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
      return enclosing->create_partition_by_image_range(forest, handle, 
              projection, parent, fid, color_space, part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_preimage(
                                                    RegionTreeForest *forest,
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
      return enclosing->create_partition_by_preimage(forest, projection, handle,
                           parent, fid, color_space, part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_preimage_range(
                                                    RegionTreeForest *forest,
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
      return enclosing->create_partition_by_preimage_range(forest, projection, 
                  handle, parent, fid, color_space, part_kind, color, id, tag);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_pending_partition(
                                                    RegionTreeForest *forest,
                                                    IndexSpace parent,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_pending_partition(forest, parent, color_space,
                                                 part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_union(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_union(forest, parent, realm_color, 
                                                 type_tag, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_union(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                                  IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_union(forest, parent, realm_color, 
                                                 type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_intersection(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_intersection(forest, parent, 
                                                        realm_color, type_tag, 
                                                        handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_intersection(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                                  IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_intersection(forest, parent, 
                                                        realm_color, 
                                                        type_tag, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_difference(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                                  IndexSpace initial,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_difference(forest, parent, 
                                                      realm_color, type_tag,
                                                      initial, handles);
    }

    //--------------------------------------------------------------------------
    FieldSpace InlineContext::create_field_space(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_field_space(forest);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_field_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      enclosing->destroy_field_space(handle);
    }

    //--------------------------------------------------------------------------
    FieldID InlineContext::allocate_field(RegionTreeForest *forest,
                                          FieldSpace space, size_t field_size,
                                          FieldID fid, bool local,
                                          CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      return enclosing->allocate_field(forest, space, field_size, 
                                       fid, local, serdez_id);
    }

    //--------------------------------------------------------------------------
    void InlineContext::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      enclosing->free_field(space, fid);
    }

    //--------------------------------------------------------------------------
    void InlineContext::allocate_fields(RegionTreeForest *forest,
                                        FieldSpace space,
                                        const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      enclosing->allocate_fields(forest, space, sizes, resulting_fields,
                                 local, serdez_id);
    }

    //--------------------------------------------------------------------------
    void InlineContext::free_fields(FieldSpace space, 
                                    const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      enclosing->free_fields(space, to_free);
    }

    //--------------------------------------------------------------------------
    LogicalRegion InlineContext::create_logical_region(RegionTreeForest *forest,
                                                       IndexSpace index_space,
                                                       FieldSpace field_space,
                                                       bool task_local)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_logical_region(forest, index_space, field_space,
                                              task_local);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->destroy_logical_region(handle);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_logical_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->destroy_logical_partition(handle);
    }

    //--------------------------------------------------------------------------
    FieldAllocator InlineContext::create_field_allocator(
                                   Legion::Runtime *external, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_field_allocator(external, handle);
    }

    //--------------------------------------------------------------------------
    Future InlineContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->execute_task(launcher);
    }

    //--------------------------------------------------------------------------
    FutureMap InlineContext::execute_index_space(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->execute_index_space(launcher);
    }

    //--------------------------------------------------------------------------
    Future InlineContext::execute_index_space(const IndexTaskLauncher &launcher,
                                              ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      return enclosing->execute_index_space(launcher, redop);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InlineContext::map_region(const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->map_region(launcher);
    }

    //--------------------------------------------------------------------------
    ApEvent InlineContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      return enclosing->remap_region(region);
    }

    //--------------------------------------------------------------------------
    void InlineContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      enclosing->unmap_region(region);
    }

    //--------------------------------------------------------------------------
    void InlineContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->fill_fields(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::fill_fields(const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->fill_fields(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_copy(const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_copy(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_copy(const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_copy(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_acquire(const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_acquire(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_release(const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_release(launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InlineContext::attach_resource(
                                                 const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->attach_resource(launcher);
    }

    //--------------------------------------------------------------------------
    Future InlineContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      return enclosing->detach_resource(region);
    }

    //--------------------------------------------------------------------------
    FutureMap InlineContext::execute_must_epoch(
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->execute_must_epoch(launcher);
    }

    //--------------------------------------------------------------------------
    Future InlineContext::issue_timing_measurement(
                                                 const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->issue_timing_measurement(launcher);
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_mapping_fence(void)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_mapping_fence();
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_execution_fence(void)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_execution_fence();
    }

    //--------------------------------------------------------------------------
    void InlineContext::complete_frame(void)
    //--------------------------------------------------------------------------
    {
      enclosing->complete_frame();
    }

    //--------------------------------------------------------------------------
    Predicate InlineContext::create_predicate(const Future &f)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_predicate(f);
    }

    //--------------------------------------------------------------------------
    Predicate InlineContext::predicate_not(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      return enclosing->predicate_not(p);
    }

    //--------------------------------------------------------------------------
    Predicate InlineContext::create_predicate(const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_predicate(launcher);
    }

    //--------------------------------------------------------------------------
    Future InlineContext::get_predicate_future(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      return enclosing->get_predicate_future(p);
    }

    //--------------------------------------------------------------------------
    unsigned InlineContext::register_new_child_operation(Operation *op,
                      const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      return enclosing->register_new_child_operation(op, dependences);
    }

    //--------------------------------------------------------------------------
    unsigned InlineContext::register_new_close_operation(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      return enclosing->register_new_close_operation(op);
    }

    //--------------------------------------------------------------------------
    unsigned InlineContext::register_new_summary_operation(TraceSummaryOp *op)
    //--------------------------------------------------------------------------
    {
      return enclosing->register_new_summary_operation(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::add_to_prepipeline_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->add_to_prepipeline_queue(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::add_to_dependence_queue(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->add_to_dependence_queue(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::add_to_post_task_queue(TaskContext *ctx, 
        RtEvent wait_on, const void *result, size_t size, PhysicalInstance inst)
    //--------------------------------------------------------------------------
    {
      enclosing->add_to_post_task_queue(ctx, wait_on, result, size, inst);
    }

    //--------------------------------------------------------------------------
    void InlineContext::register_executing_child(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_executing_child(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_executed(op);
    }
    
    //--------------------------------------------------------------------------
    void InlineContext::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_complete(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_child_commit(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::unregister_child_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->unregister_child_operation(op);
    }

    //--------------------------------------------------------------------------
    ApEvent InlineContext::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      return enclosing->register_fence_dependence(op);
    }

    //--------------------------------------------------------------------------
    ApEvent InlineContext::perform_fence_analysis(Operation *op, 
                                                  bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      return enclosing->perform_fence_analysis(op, mapping, execution);
    }

    //--------------------------------------------------------------------------
    void InlineContext::update_current_fence(FenceOp *op, 
                                             bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      enclosing->update_current_fence(op, mapping, execution);
    }

    //--------------------------------------------------------------------------
    RtEvent InlineContext::get_current_mapping_fence_event(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->get_current_mapping_fence_event();
    }

    //--------------------------------------------------------------------------
    ApEvent InlineContext::get_current_execution_fence_event(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->get_current_execution_fence_event();
    }


    //--------------------------------------------------------------------------
    void InlineContext::begin_trace(TraceID tid, bool logical_only)
    //--------------------------------------------------------------------------
    {
      enclosing->begin_trace(tid, logical_only);
    }

    //--------------------------------------------------------------------------
    void InlineContext::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      enclosing->end_trace(tid);
    }

    //--------------------------------------------------------------------------
    void InlineContext::begin_static_trace(const std::set<RegionTreeID> *trees)
    //--------------------------------------------------------------------------
    {
      enclosing->begin_static_trace(trees);
    }

    //--------------------------------------------------------------------------
    void InlineContext::end_static_trace(void)
    //--------------------------------------------------------------------------
    {
      enclosing->end_static_trace();
    }

    //--------------------------------------------------------------------------
    void InlineContext::record_previous_trace(LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      enclosing->record_previous_trace(trace);
    }

    //--------------------------------------------------------------------------
    void InlineContext::invalidate_trace_cache(
                                     LegionTrace *trace, Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      enclosing->invalidate_trace_cache(trace, invalidator);
    }

    //--------------------------------------------------------------------------
    void InlineContext::record_blocking_call(void)
    //--------------------------------------------------------------------------
    {
      enclosing->record_blocking_call();
    }

    //--------------------------------------------------------------------------
    void InlineContext::issue_frame(FrameOp *frame, ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      enclosing->issue_frame(frame, frame_termination);
    }

    //--------------------------------------------------------------------------
    void InlineContext::perform_frame_issue(FrameOp *frame, ApEvent frame_term)
    //--------------------------------------------------------------------------
    {
      enclosing->perform_frame_issue(frame, frame_term);
    }

    //--------------------------------------------------------------------------
    void InlineContext::finish_frame(ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      enclosing->finish_frame(frame_termination);
    }

    //--------------------------------------------------------------------------
    void InlineContext::increment_outstanding(void)
    //--------------------------------------------------------------------------
    {
      enclosing->increment_outstanding();
    }

    //--------------------------------------------------------------------------
    void InlineContext::decrement_outstanding(void)
    //--------------------------------------------------------------------------
    {
      enclosing->decrement_outstanding();
    }

    //--------------------------------------------------------------------------
    void InlineContext::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      enclosing->increment_pending();
    }

    //--------------------------------------------------------------------------
    RtEvent InlineContext::decrement_pending(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      return enclosing->decrement_pending(child);
    }

    //--------------------------------------------------------------------------
    RtEvent InlineContext::decrement_pending(bool need_deferral)
    //--------------------------------------------------------------------------
    {
      return enclosing->decrement_pending(need_deferral);
    }

    //--------------------------------------------------------------------------
    void InlineContext::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      enclosing->increment_frame();
    }

    //--------------------------------------------------------------------------
    void InlineContext::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      enclosing->decrement_frame();
    }

    //--------------------------------------------------------------------------
    InnerContext* InlineContext::find_parent_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < parent_req_indexes.size());
#endif
      return enclosing->find_parent_logical_context(parent_req_indexes[index]);
    }

    //--------------------------------------------------------------------------
    InnerContext* InlineContext::find_parent_physical_context(unsigned index,
                                                          LogicalRegion *handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < parent_req_indexes.size());
#endif
      return enclosing->find_parent_physical_context(parent_req_indexes[index],
                                                     handle);
    }

    //--------------------------------------------------------------------------
    void InlineContext::find_parent_version_info(unsigned index, unsigned depth,
                     const FieldMask &version_mask, InnerContext *context,
                     VersionInfo &version_info, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      enclosing->find_parent_version_info(index, depth, version_mask, 
                                          context, version_info, ready_events);
    }

    //--------------------------------------------------------------------------
    InnerContext* InlineContext::find_outermost_local_context(InnerContext *pre)
    //--------------------------------------------------------------------------
    {
      return enclosing->find_outermost_local_context(pre);
    }
    
    //--------------------------------------------------------------------------
    InnerContext* InlineContext::find_top_context(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->find_top_context();
    }

    //--------------------------------------------------------------------------
    void InlineContext::initialize_region_tree_contexts(
            const std::vector<RegionRequirement> &clone_requirements,
            const std::vector<ApUserEvent> &unmap_events,
            std::set<ApEvent> &preconditions, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InlineContext::invalidate_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InlineContext::send_back_created_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceView* InlineContext::create_instance_top_view(
                PhysicalManager *manager, AddressSpaceID source, RtEvent *ready)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& InlineContext::begin_task(
                                                           Legion::Runtime *&rt)
    //--------------------------------------------------------------------------
    {
      return enclosing->begin_task(rt);
    }

    //--------------------------------------------------------------------------
    void InlineContext::end_task(const void *res, size_t res_size, bool owned,
                                 PhysicalInstance deferred_result_instance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!deferred_result_instance.exists());
#endif
      inline_task->end_inline_task(res, res_size, owned);
    }

    //--------------------------------------------------------------------------
    void InlineContext::post_end_task(const void *res, 
                                      size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InlineContext::add_acquisition(AcquireOp *op,
                                        const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      enclosing->add_acquisition(op, req);
    }
    
    //--------------------------------------------------------------------------
    void InlineContext::remove_acquisition(ReleaseOp *op,
                                           const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      enclosing->remove_acquisition(op, req);
    }

    //--------------------------------------------------------------------------
    void InlineContext::add_restriction(AttachOp *op, InstanceManager *instance,
                                        const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      enclosing->add_restriction(op, instance, req);
    }

    //--------------------------------------------------------------------------
    void InlineContext::remove_restriction(DetachOp *op,
                                           const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      enclosing->remove_restriction(op, req);
    }
    
    //--------------------------------------------------------------------------
    void InlineContext::release_restrictions(void)
    //--------------------------------------------------------------------------
    {
      enclosing->release_restrictions();
    }

    //--------------------------------------------------------------------------
    bool InlineContext::has_restrictions(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->has_restrictions();
    }

    //--------------------------------------------------------------------------
    void InlineContext::perform_restricted_analysis(
                      const RegionRequirement &req, RestrictInfo &restrict_info)
    //--------------------------------------------------------------------------
    {
      enclosing->perform_restricted_analysis(req, restrict_info);
    }

    //--------------------------------------------------------------------------
    void InlineContext::record_dynamic_collective_contribution(
                                          DynamicCollective dc, const Future &f)
    //--------------------------------------------------------------------------
    {
      enclosing->record_dynamic_collective_contribution(dc, f);
    }

    //--------------------------------------------------------------------------
    void InlineContext::find_collective_contributions(DynamicCollective dc, 
                                             std::vector<Future> &contributions)
    //--------------------------------------------------------------------------
    {
      enclosing->find_collective_contributions(dc, contributions);
    }

    //--------------------------------------------------------------------------
    TaskPriority InlineContext::get_current_priority(void) const
    //--------------------------------------------------------------------------
    {
      return enclosing->get_current_priority();
    }

    //--------------------------------------------------------------------------
    void InlineContext::set_current_priority(TaskPriority priority)
    //--------------------------------------------------------------------------
    {
      enclosing->set_current_priority(priority);
    }

  };
};

// EOF

