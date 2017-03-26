/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "runtime.h"
#include "legion_tasks.h"
#include "legion_trace.h"
#include "legion_context.h"
#include "legion_instances.h"
#include "legion_views.h"

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
        children_complete_invoked(false), children_commit_invoked(false)
    //--------------------------------------------------------------------------
    {
      context_lock = Reservation::create_reservation();
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
      context_lock.destroy_reservation();
      context_lock = Reservation::NO_RESERVATION;
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
      PhysicalRegionImpl *impl = legion_new<PhysicalRegionImpl>(req, 
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
      assert(idx < physical_regions.size());
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
    ptr_t TaskContext::perform_safe_cast(IndexSpace handle, ptr_t pointer)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
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
    DomainPoint TaskContext::perform_safe_cast(IndexSpace handle, 
                                              const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
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
    void TaskContext::add_created_region(LogicalRegion handle)
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
      returnable_privileges.push_back(true);
      // Make a new unmapped physical region if we aren't done executing yet
      if (!task_executed)
        physical_regions.push_back(PhysicalRegion(
              legion_new<PhysicalRegionImpl>(created_requirements.back(), 
                ApEvent::NO_AP_EVENT, false/*mapped*/, this, 
                owner_task->map_id, owner_task->tag, 
                is_leaf_context(), false/*virtual mapped*/, runtime)));
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
        // Skip it if there are no privilege fields
        if (created_requirements[idx].privilege_fields.empty())
          continue;
        TaskOp::log_requirement(unique_op_id, original_size + idx, 
                                created_requirements[idx]);
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
    void TaskContext::register_region_creation(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      // Create a new logical region 
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      AutoLock ctx_lock(context_lock);
#ifdef DEBUG_LEGION
      assert(created_regions.find(handle) == created_regions.end());
#endif
      created_regions.insert(handle); 
      add_created_region(handle);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_region_deletion(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      bool finalize = false;
      // Hold the operation lock when doing this since children could
      // be returning values from the utility processor
      {
        AutoLock ctx_lock(context_lock);
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
    void TaskContext::register_field_creation(FieldSpace handle, 
                                              FieldID fid, bool local)
    //--------------------------------------------------------------------------
    {
      AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
      return (created_index_spaces.find(space) != created_index_spaces.end());
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_index_space_creation(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
                                            const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      AutoLock ctx_lock(context_lock);
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
    void TaskContext::register_region_deletions(
                                            const std::set<LogicalRegion> &regs)
    //--------------------------------------------------------------------------
    {
      std::vector<LogicalRegion> to_finalize;
      {
        AutoLock ctx_lock(context_lock);
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
    void TaskContext::register_field_creations(
                     const std::map<std::pair<FieldSpace,FieldID>,bool> &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
    void TaskContext::analyze_destroy_index_space(IndexSpace handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
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
      unsigned created_index = 0;
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++, created_index++)
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
    void TaskContext::analyze_destroy_index_partition(IndexPartition handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
    void TaskContext::analyze_destroy_field_space(FieldSpace handle,
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
      unsigned created_index = 0;
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++, created_index++)
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
      unsigned created_index = 0;
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
      for (std::deque<RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++, parent_index++, created_index++)
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
          inline_regions.erase(it);
          return;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TaskContext::is_region_mapped(unsigned idx)
    //--------------------------------------------------------------------------
    {
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(idx < physical_regions.size());
#endif
      return physical_regions[idx].impl->is_mapped();
    }

    //--------------------------------------------------------------------------
    void TaskContext::clone_requirement(unsigned idx, RegionRequirement &target)
    //--------------------------------------------------------------------------
    {
      if (idx >= regions.size())
      {
        idx -= regions.size();
        AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
      const FieldSpace fs = req.parent.get_field_space(); 
      // The created region requirements have to be checked while holding
      // the lock since they are subject to mutation by the application
      // We might also mutate it so we take the lock in exclusive mode
      AutoLock ctx_lock(context_lock);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        RegionRequirement &our_req = created_requirements[idx];
        // First check that the regions match
        if (our_req.region != req.parent)
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((req.privilege & our_req.privilege) != req.privilege))
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
      // Make a new unmapped physical region if we're not done executing yet
      if (!task_executed)
        physical_regions.push_back(PhysicalRegion(
              legion_new<PhysicalRegionImpl>(created_requirements.back(),
                ApEvent::NO_AP_EVENT, false/*mapped*/, this, 
                owner_task->map_id, owner_task->tag, 
                is_leaf_context(), false/*virtual mapped*/, runtime)));
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
    unsigned TaskContext::find_parent_index_region(unsigned index,TaskOp *child)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < owner_task->indexes.size(); idx++)
      {
        if ((owner_task->indexes[idx].handle == child->indexes[idx].parent))
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
    PrivilegeMode TaskContext::find_parent_privilege_mode(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (idx < regions.size())
        return regions[idx].privilege;
      idx -= regions.size();
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
    LegionErrorType TaskContext::check_privilege(const RegionRequirement &req,
                                                 FieldID &bad_field,
                                                 bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
      if (req.flags & VERIFIED_FLAG)
        return NO_ERROR;
      // Copy privilege fields for check
      std::set<FieldID> privilege_fields(req.privilege_fields);
      // Try our original region requirements first
      for (std::vector<RegionRequirement>::const_iterator it = 
            regions.begin(); it != regions.end(); it++)
      {
        LegionErrorType et = 
          check_privilege_internal(req, *it, privilege_fields, bad_field,
                                   skip_privilege);
        // No error so we are done
        if (et == NO_ERROR)
          return et;
        // Something other than bad parent region is a real error
        if (et != ERROR_BAD_PARENT_REGION)
          return et;
        // Otherwise we just keep going
      }
      // If none of that worked, we now get to try the created requirements
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
      for (unsigned idx = 0; idx < created_requirements.size(); idx++)
      {
        LegionErrorType et = 
          check_privilege_internal(req, created_requirements[idx], 
                      privilege_fields, bad_field, skip_privilege);
        // No error so we are done
        if (et == NO_ERROR)
          return et;
        // Something other than bad parent region is a real error
        if (et != ERROR_BAD_PARENT_REGION)
          return et;
        // If we got a BAD_PARENT_REGION, see if this a returnable
        // privilege in which case we know we have privileges on all fields
        if (returnable_privileges[idx])
          return NO_ERROR;
        // Otherwise we just keep going
      }
      // Finally see if we created all the fields in which case we know
      // we have privileges on all their regions
      const FieldSpace sp = req.region.get_field_space();
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        std::pair<FieldSpace,FieldID> key(sp, *it);
        // If we don't find the field, then we are done
        if (created_fields.find(key) == created_fields.end())
          return ERROR_BAD_PARENT_REGION;
      }
      // Otherwise we have privileges on these fields for all regions
      // so we are good on privileges
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege_internal(
        const RegionRequirement &req, const RegionRequirement &our_req,
        std::set<FieldID>& privilege_fields,
        FieldID &bad_field, bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(our_req.handle_type == SINGULAR); // better be singular
#endif
      // Check to see if we found the requirement in the parent
      if (our_req.region == req.parent)
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
        for (std::set<FieldID>::const_iterator fit = 
              privilege_fields.begin(); fit != 
              privilege_fields.end(); )
        {
          if (our_req.privilege_fields.find(*fit) != 
              our_req.privilege_fields.end())
          {
            // Only need to do this check if there were overlapping fields
            if (!skip_privilege && (req.privilege & (~(our_req.privilege))))
            {
              // Handle the special case where the parent has WRITE_DISCARD
              // privilege and the sub-task wants any other kind of privilege.  
              // This case is ok because the parent could write something
              // and then hand it off to the child.
              if (our_req.privilege != WRITE_DISCARD)
              {
                if ((req.handle_type == SINGULAR) || 
                    (req.handle_type == REG_PROJECTION))
                  return ERROR_BAD_REGION_PRIVILEGES;
                else
                  return ERROR_BAD_PARTITION_PRIVILEGES;
              }
            }
            privilege_fields.erase(fit++);
          }
          else
            ++fit;
        }
      }

      if (!privilege_fields.empty()) return ERROR_BAD_PARENT_REGION;
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(index < created_requirements.size());
#endif
      return created_requirements[index].region;
    } 

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskContext::begin_task(void)
    //--------------------------------------------------------------------------
    {
      if (overhead_tracker != NULL)
        previous_profiling_time = Realm::Clock::current_time_in_nanoseconds();
      // Switch over the executing processor to the one
      // that has actually been assigned to run this task.
      executing_processor = Processor::get_executing_processor();
      if (Runtime::legion_spy_enabled)
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
      // Can't be holding the lock when we unmap in case we block
      std::vector<PhysicalRegion> unmap_regions;
      {
        AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
        for (std::vector<PhysicalRegion>::const_iterator it = 
              physical_regions.begin(); it != physical_regions.end(); it++)
        {
          if (it->impl->is_mapped())
            unmap_regions.push_back(*it);
        }
        // Also unmap any of our inline mapped physical regions
        for (LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::
              tracked::const_iterator it = inline_regions.begin();
              it != inline_regions.end(); it++)
        {
          if (it->impl->is_mapped())
            unmap_regions.push_back(*it);
        }
      }
      // Perform the unmappings after we've released the lock
      for (std::vector<PhysicalRegion>::const_iterator it = 
            unmap_regions.begin(); it != unmap_regions.end(); it++)
      {
        if (it->impl->is_mapped())
          it->impl->unmap_region();
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::execute_task_launch(TaskOp *task, bool index,
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
        if (!Runtime::unsafe_launch)
          find_conflicting_regions(task, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          if (Runtime::runtime_warnings && !silence_warnings)
          {
            if (index)
              log_run.warning("WARNING: Runtime is unmapping and remapping "
                  "physical regions around execute_index_space call in "
                  "task %s (UID %lld).", get_task_name(), get_unique_id());
            else
              log_run.warning("WARNING: Runtime is unmapping and remapping "
                  "physical regions around execute_task call in "
                  "task %s (UID %lld).", get_task_name(), get_unique_id());
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
    void TaskContext::remap_unmapped_regions(LegionTrace *trace,
                            const std::vector<PhysicalRegion> &unmapped_regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unmapped_regions.empty());
#endif
      if ((trace != NULL) && trace->is_static_trace())
      {
        log_run.error("Illegal runtime remapping in static trace inside of "
                      "task %s (UID %lld). Static traces must perfectly "
                      "manage their physical mappings with no runtime help.",
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_REMAP_IN_STATIC_TRACE);
      }
      if (unmapped_regions.size() == 1)
      {
        MapOp *op = runtime->get_available_map_op(true);
        op->initialize(this, unmapped_regions[0]);
        ApEvent mapped_event = op->get_completion_event();
        runtime->add_to_dependence_queue(this, executing_processor, op);
        if (mapped_event.has_triggered())
          return;
        begin_task_wait(true/*from runtime*/);
        mapped_event.wait();
        end_task_wait();
      }
      else
      {
        std::set<ApEvent> mapped_events;
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
        {
          MapOp *op = runtime->get_available_map_op(true);
          op->initialize(this, unmapped_regions[idx]);
          mapped_events.insert(op->get_completion_event());
          runtime->add_to_dependence_queue(this, executing_processor, op);
        }
        // Wait for all the re-mapping operations to complete
        ApEvent mapped_event = Runtime::merge_events(mapped_events);
        if (mapped_event.has_triggered())
          return;
        begin_task_wait(true/*from runtime*/);
        mapped_event.wait();
        end_task_wait();
      }
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
        total_children_count(0), total_close_count(0), 
        outstanding_children_count(0), current_trace(NULL), 
        valid_wait_event(false), outstanding_subtasks(0), pending_subtasks(0), 
        pending_frames(0), currently_active_context(false),
        current_fence(NULL), fence_gen(0) 
    //--------------------------------------------------------------------------
    {
      // Set some of the default values for a context
      context_configuration.max_window_size = 
        Runtime::initial_task_window_size;
      context_configuration.hysteresis_percentage = 
        Runtime::initial_task_window_hysteresis;
      context_configuration.max_outstanding_frames = 0;
      context_configuration.min_tasks_to_schedule = 
        Runtime::initial_tasks_to_schedule;
      context_configuration.min_frames_to_schedule = 0;
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
        invalidate_remote_contexts();
      for (std::map<TraceID,DynamicTrace*>::const_iterator it = traces.begin();
            it != traces.end(); it++)
      {
        if (it->second->remove_reference())
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
      AutoLock ctx_lock(context_lock); 
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
      return source;
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
      assert(index < returnable_privileges.size());
#endif
      if (returnable_privileges[index])
        return find_outermost_local_context();
      return this;
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_physical_context(unsigned index)
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
                                            parent_req_indexes[index]);
        else // We mapped a physical instance so we're it
          return this;
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
        AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
        if ((index >= returnable_privileges.size()) || 
            returnable_privileges[index])
          return find_top_context();
        else
          return this;
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_parent_version_info(unsigned index, unsigned depth,
                       const FieldMask &version_mask, VersionInfo &version_info)
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
      parent_info.clone_to_depth(depth, version_mask, version_info);
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
                                                size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task %s "
                      "(ID %lld) with %zd maximum elements", handle.id, 
                      get_task_name(), get_unique_id(), max_num_elmts); 
#endif
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      Realm::IndexSpace space = 
                      Realm::IndexSpace::create_index_space(max_num_elmts);
      forest->create_index_space(handle, Domain(space), 
                                 UNSTRUCTURED_KIND, MUTABLE);
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(RegionTreeForest *forest,
                                                const Domain &domain)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating dummy index space %x in task %s (ID %lld) "
                      "for domain", handle.id, get_task_name(),get_unique_id());
#endif
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      forest->create_index_space(handle, domain, DENSE_ARRAY_KIND, NO_MEMORY);
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(RegionTreeForest *forest,
                                                const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id());
      // First compute the convex hull of all the domains
      Domain hull = *(domains.begin());
#ifdef DEBUG_LEGION
      if (hull.get_dim() == 0)
      {
        log_index.error("Create index space with multiple domains "
                        "must be created with domains for non-zero "
                        "dimension in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
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
                          "%s (ID %lld)", get_task_name(), get_unique_id());
          assert(false);
          exit(ERROR_DOMAIN_DIM_MISMATCH);
        }
      }
#endif
      switch (hull.get_dim())
      {
        case 1:
          {
            LegionRuntime::Arrays::Rect<1> base = hull.get_rect<1>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              LegionRuntime::Arrays::Rect<1> next = it->get_rect<1>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<1>(base);
            break;
          }
        case 2:
          {
            LegionRuntime::Arrays::Rect<2> base = hull.get_rect<2>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              LegionRuntime::Arrays::Rect<2> next = it->get_rect<2>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<2>(base);
            break;
          }
        case 3:
          {
            LegionRuntime::Arrays::Rect<3> base = hull.get_rect<3>();
            for (std::set<Domain>::const_iterator it = domains.begin();
                  it != domains.end(); it++)
            {
              LegionRuntime::Arrays::Rect<3> next = it->get_rect<3>();
              base = base.convex_hull(next);
            }
            hull = Domain::from_rect<3>(base);
            break;
          }
        default:
          assert(false);
      }
#ifdef DEBUG_LEGION
      log_index.debug("Creating dummy index space %x in task %s (ID %lld) for "
                      "domain", handle.id, get_task_name(), get_unique_id());
#endif
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id);

      forest->create_index_space(handle, hull, domains,
                                 DENSE_ARRAY_KIND, NO_MEMORY);
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
      DeletionOp *op = runtime->get_available_deletion_op(true);
      op->initialize_index_space_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                  RegionTreeForest *forest, IndexSpace parent,
                                  const Domain &color_space,
                                  const PointColoring &coloring,
                                  PartitionKind part_kind, 
                                  int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
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
#ifdef DEBUG_LEGION
      if ((part_kind == DISJOINT_KIND) && Runtime::verify_disjointness)
        runtime->validate_unstructured_disjointness(pid, new_index_spaces);
#endif
      ColorPoint partition_color;
      // If we have a valid color, set it now
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_index_spaces, color_space, part_kind, 
                                     allocable ? MUTABLE : NO_MEMORY);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                              RegionTreeForest *forest,
                                              IndexSpace parent,
                                              const Coloring &coloring,
                                              bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no colors in "
                      "task %s (ID %lld). Index partitions must have at least "
                      "one color.", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_EMPTY_INDEX_PARTITION);
      }
      LegionRuntime::Arrays::Point<1> lower_bound(coloring.begin()->first);
      LegionRuntime::Arrays::Point<1> upper_bound(coloring.rbegin()->first);
      LegionRuntime::Arrays::Rect<1> color_range(lower_bound,upper_bound);
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
      for (LegionRuntime::Arrays::GenericPointInRectIterator<1>
	     pir(color_range); pir; pir++)
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
         LegionRuntime::Arrays::Point<1>(finder->first))] = Domain(child_space);
      }
#ifdef DEBUG_LEGION
      if (disjoint && Runtime::verify_disjointness)
        runtime->validate_unstructured_disjointness(pid, new_index_spaces);
#endif 
      ColorPoint partition_color;
      // If we have a valid color, set it now
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_index_spaces, color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     MUTABLE);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if ((part_kind == DISJOINT_KIND) && Runtime::verify_disjointness)
        runtime->validate_structured_disjointness(pid, coloring);
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     coloring, color_space, 
                                     part_kind, NO_MEMORY);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no colors in "
                      "task %s (ID %lld). Index partitions must have at least "
                      "one color.", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
            LegionRuntime::Arrays::Point<1>(it->first))] = it->second;
      }
#ifdef DEBUG_LEGION
      if (disjoint && Runtime::verify_disjointness)
        runtime->validate_structured_disjointness(pid, new_subspaces);
#endif
      forest->create_index_partition(pid, parent, partition_color, 
                                     new_subspaces, color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     NO_MEMORY);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      // Build all the convex hulls
      std::map<DomainPoint,Domain> convex_hulls;
      for (std::map<DomainPoint,std::set<Domain> >::const_iterator it = 
            coloring.begin(); it != coloring.end(); it++)
      {
        Domain hull = runtime->construct_convex_hull(it->second);
        convex_hulls[it->first] = hull;
      }
#ifdef DEBUG_LEGION
      if ((part_kind == DISJOINT_KIND) && Runtime::verify_disjointness)
        runtime->validate_multi_structured_disjointness(pid, coloring);
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     convex_hulls, coloring,
                                     color_space, part_kind, NO_MEMORY);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      if (coloring.empty())
      {
        log_run.error("Attempt to create index partition with no colors in "
                      "task %s (ID %lld). Index partitions must have at least "
                      "one color.", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
        Domain hull = runtime->construct_convex_hull(it->second);
	LegionRuntime::Arrays::Point<1> pcolor(it->first);
        DomainPoint color = DomainPoint::from_point<1>(pcolor);
        convex_hulls[color] = hull;
        color_sets[color] = it->second; 
      }
#ifdef DEBUG_LEGION
      if (disjoint && Runtime::verify_disjointness)
        runtime->validate_multi_structured_disjointness(pid, color_sets);
#endif
      ColorPoint partition_color;
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color, 
                                     convex_hulls, color_sets,
                                     color_space,
                                     disjoint ? DISJOINT_KIND : ALIASED_KIND,
                                     NO_MEMORY);
      register_index_partition_creation(pid);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_index_partition(
                                        RegionTreeForest *forest,
                                        IndexSpace parent,
                LegionRuntime::Accessor::RegionAccessor<
                 LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                        int part_color)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating index partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      // Perform the coloring
      std::map<DomainPoint,Domain> new_index_spaces;
      Domain color_space;
      // Iterate over the parent index space and make the sub-index spaces
      // for each of the different points in the space
      LegionRuntime::Accessor::RegionAccessor<
        LegionRuntime::Accessor::AccessorType::Generic,int> 
          fa_coloring = field_accessor.typeify<int>();
      {
        std::map<Color,Realm::ElementMask> child_masks;
        Domain parent_dom = forest->get_index_space_domain(parent);
        size_t parent_elmts = 
          parent_dom.get_index_space().get_valid_mask().get_num_elmts();
        for (Domain::DomainPointIterator itr(parent_dom); itr; itr++)
        {
          ptr_t cur_ptr = itr.p.get_index();
          int c;
          fa_coloring.read_untyped(cur_ptr, &c, sizeof(c));
          // Ignore all colors less than zero
          if (c >= 0)
          {
            Color color = (Color)c; 
            std::map<Color,Realm::ElementMask>::iterator finder = 
              child_masks.find(color);
            // Haven't made an index space for this color yet
            if (finder == child_masks.end())
            {
              child_masks[color] = Realm::ElementMask(parent_elmts);
              finder = child_masks.find(color);
            }
#ifdef DEBUG_LEGION
            assert(finder != child_masks.end());
#endif
            finder->second.enable(cur_ptr.value);
          }
        }
        // Now make the index spaces and their domains
        LegionRuntime::Arrays::Point<1> lower_bound(child_masks.begin()->first);
        LegionRuntime::Arrays::Point<1> upper_bound(child_masks.rbegin()->first);
        LegionRuntime::Arrays::Rect<1> color_range(lower_bound,upper_bound);
        color_space = Domain::from_rect<1>(color_range);
        // Iterate over all the colors in the range from the lower
        // bound to upper bound so we can store the color space as
        // a dense array of colors.
        for (LegionRuntime::Arrays::GenericPointInRectIterator<1>
	       pir(color_range); pir; pir++)
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
              LegionRuntime::Arrays::Point<1>(c))] = Domain(child_space);
        }
      }
      ColorPoint partition_color;
      if (part_color >= 0)
        partition_color = ColorPoint(part_color);
      forest->create_index_partition(pid, parent, partition_color,
                                     new_index_spaces, color_space,
                                     DISJOINT_KIND, MUTABLE);
      register_index_partition_creation(pid);
      return pid;
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
      DeletionOp *op = runtime->get_available_deletion_op(true);
      op->initialize_index_part_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_equal_partition(
                                                      RegionTreeForest *forest,
                                                      IndexSpace parent,
                                                      const Domain &color_space,
                                                      size_t granularity,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating equal partition %d with parent index space %x "
                      "in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_equal_partition(this, pid, granularity);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space,
                                       partition_color, DISJOINT_KIND,
                                       allocable, handle_ready, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_weighted_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const std::map<DomainPoint,int> &weights,
                                      size_t granularity, 
                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating weighted partition %d with parent index "
                      "space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      ColorPoint partition_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        partition_color = ColorPoint(color);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_weighted_partition(this, pid, granularity, weights);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, DISJOINT_KIND,
                                       allocable, handle_ready, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_union(
                                          RegionTreeForest *forest,
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          PartitionKind kind,
                                          int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating union partition %d with parent index "
                      "space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create "
                        "partition by union!", handle1.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create "
                        "partition by union!", handle2.id, parent.id);
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
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_union_partition(this, pid, handle1, handle2);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
                                              RegionTreeForest *forest,
                                              IndexSpace parent,
                                              IndexPartition handle1,
                                              IndexPartition handle2,
                                              PartitionKind kind,
                                              int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating intersection partition %d with parent "
                      "index space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create partition by "
                        "intersection!", handle1.id, parent.id);
        assert(false);
        exit(ERROR_INDEX_TREE_MISMATCH);
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        log_index.error("IndexPartition %d is not part of the same "
                        "index tree as IndexSpace %d in create partition by "
                        "intersection!", handle2.id, parent.id);
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
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_intersection_partition(this, pid, handle1, handle2);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_difference(
                                                  RegionTreeForest *forest,
                                                  IndexSpace parent,
                                                  IndexPartition handle1,
                                                  IndexPartition handle2,
                                                  PartitionKind kind,
                                                  int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating difference partition %d with parent "
                      "index space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
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
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_difference_partition(this, pid, handle1, handle2);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, parent, color_space, 
                                       partition_color, kind, allocable, 
                                       handle_ready, term_event);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_cross_product_partition(RegionTreeForest *forest,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                  std::map<DomainPoint,IndexPartition> &handles,
                                                      PartitionKind kind,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating cross product partitions in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
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
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      ApEvent handle_ready = part_op->get_handle_ready();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      std::map<DomainPoint,IndexPartition> local;
      forest->create_pending_cross_product(handle1, handle2, local, handles,
                                           kind, partition_color, allocable,
                                           handle_ready, term_event);
      part_op->initialize_cross_product(this, handle1, handle2, local);
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_field(
                                              RegionTreeForest *forest,
                                              LogicalRegion handle,
                                              LogicalRegion parent_priv,
                                              FieldID fid,
                                              const Domain &color_space,
                                              int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexSpace parent = handle.get_index_space();
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by field in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(true);
      part_op->initialize_by_field(this, pid, handle, 
                                   parent_priv, color_space, fid);
      ApEvent term_event = part_op->get_completion_event();
      ApEvent handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition 
      forest->create_pending_partition(pid, parent, color_space, part_color,
                                       DISJOINT_KIND, allocable, 
                                       handle_ready, term_event); 
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_field call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image(
                                                    RegionTreeForest *forest,
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by image in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(true);
      part_op->initialize_by_image(this, pid, projection,
                                   parent, fid, color_space);
      ApEvent term_event = part_op->get_completion_event();
      ApEvent handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, handle, color_space, part_color,
                                       part_kind, allocable, 
                                       handle_ready, term_event); 
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_image call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage(
                                                  RegionTreeForest *forest,
                                                  IndexPartition projection,
                                                  LogicalRegion handle,
                                                  LogicalRegion parent,
                                                  FieldID fid,
                                                  const Domain &color_space,
                                                  PartitionKind part_kind,
                                                  int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by preimage in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(true);
      part_op->initialize_by_preimage(this, pid, projection, handle,
                                      parent, fid, color_space);
      ApEvent term_event = part_op->get_completion_event();
      ApEvent handle_ready = part_op->get_handle_ready();
      // Tell the region tree forest about this partition
      forest->create_pending_partition(pid, handle.get_index_space(), 
                                       color_space, part_color, part_kind,
                                       allocable, handle_ready, term_event);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_preimage call "
              "in task %s (UID %lld).", get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region();
      }
      // Issue the copy operation
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_pending_partition(
                                                RegionTreeForest *forest,
                                                IndexSpace parent,
                                                const Domain &color_space, 
                                                PartitionKind part_kind,
                                                int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id());
#ifdef DEBUG_LEGION
      log_index.debug("Creating pending partition in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ColorPoint part_color;
      if (color != static_cast<int>(AUTO_GENERATE_ID))
        part_color = ColorPoint(color);
      forest->create_pending_partition(pid, parent, color_space, part_color,
                                       part_kind, allocable, 
                                       ApEvent::NO_AP_EVENT,
                                       ApEvent::NO_AP_EVENT, true/*separate*/);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_index_space_union(this, result, handles);
      Runtime::trigger_event(handle_ready, part_op->get_handle_ready());
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const DomainPoint &color,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_index_space_union(this, result, handle);
      Runtime::trigger_event(handle_ready, part_op->get_handle_ready());
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_index_space_intersection(this, result, handles);
      Runtime::trigger_event(handle_ready, part_op->get_handle_ready());
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      RegionTreeForest *forest,
                                                      IndexPartition parent,
                                                      const DomainPoint &color,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_index_space_intersection(this, result, handle);
      Runtime::trigger_event(handle_ready, part_op->get_handle_ready());
      Runtime::trigger_event(domain_ready, part_op->get_completion_event());
      // Now we can add the operation to the queue
      runtime->add_to_dependence_queue(this, executing_processor, part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_difference(
                                                    RegionTreeForest *forest,
                                                    IndexPartition parent,
                                                    const DomainPoint &color,
                                                    IndexSpace initial,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space difference in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent handle_ready, domain_ready;
      IndexSpace result = forest->find_pending_space(parent, color, 
                                                     handle_ready, 
                                                     domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op(true);
      part_op->initialize_index_space_difference(this, result, initial,handles);
      Runtime::trigger_event(handle_ready, part_op->get_handle_ready());
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
#ifdef DEBUG_LEGION
      log_field.debug("Creating field space %x in task %s (ID %lld)", 
                      space.id, get_task_name(), get_unique_id());
#endif
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_field_space(space.id);

      forest->create_field_space(space);
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
      DeletionOp *op = runtime->get_available_deletion_op(true);
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
        log_task.error("Task %s (ID %lld) attempted to allocate a field with "
                       "ID %d which exceeds the MAX_APPLICATION_FIELD_ID bound "
                       "set in legion_config.h", get_task_name(),
                       get_unique_id(), fid);
        assert(false);
      }
#endif

      if (Runtime::legion_spy_enabled)
        LegionSpy::log_field_creation(space.id, fid, field_size);

      std::set<RtEvent> done_events;
      if (local)
      {
        // See if we've exceeded our local field allocations 
        // for this field space
        std::vector<LocalFieldInfo> &infos = local_fields[space];
        if (infos.size() == Runtime::max_local_fields)
        {
          log_run.error("Exceeded maximum number of local fields in "
                        "context of task %s (UID %lld). The maximum "
                        "is currently set to %d, but can be modified "
                        "with the -lg:local flag.", get_task_name(),
                        get_unique_id(), Runtime::max_local_fields);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
        std::set<unsigned> current_indexes;
        for (std::vector<LocalFieldInfo>::const_iterator it = 
              infos.begin(); it != infos.end(); it++)
          current_indexes.insert(it->index);
        std::vector<FieldID> fields(1, fid);
        std::vector<size_t> sizes(1, field_size);
        std::vector<unsigned> new_indexes;
        if (!forest->allocate_local_fields(space, fields, sizes, serdez_id, 
                                           current_indexes, new_indexes))
        {
          log_run.error("Unable to allocate local field in context of "
                        "task %s (UID %lld) due to local field size "
                        "fragmentation. This situation can be improved "
                        "by increasing the maximum number of permitted "
                        "local fields in a context with the -lg:local "
                        "flag.", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
#ifdef DEBUG_LEGION
        assert(new_indexes.size() == 1);
#endif
        // Only need the lock here when modifying since all writes
        // to this data structure are serialized
        AutoLock ctx_lock(context_lock);
        infos.push_back(LocalFieldInfo(fid, field_size, serdez_id, 
                                       new_indexes[0], false));
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
        if (!wait_on.has_triggered())
          wait_on.wait();
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DeletionOp *op = runtime->get_available_deletion_op(true);
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
          log_task.error("Task %s (ID %lld) attempted to allocate a field with "
                         "ID %d which exceeds the MAX_APPLICATION_FIELD_ID "
                         "bound set in legion_config.h", get_task_name(),
                         get_unique_id(), resulting_fields[idx]);
          assert(false);
        }
#endif

        if (Runtime::legion_spy_enabled)
          LegionSpy::log_field_creation(space.id, 
                                        resulting_fields[idx], sizes[idx]);
      }
      std::set<RtEvent> done_events;
      if (local)
      {
        // See if we've exceeded our local field allocations 
        // for this field space
        std::vector<LocalFieldInfo> &infos = local_fields[space];
        if ((infos.size() + sizes.size()) > Runtime::max_local_fields)
        {
          log_run.error("Exceeded maximum number of local fields in "
                        "context of task %s (UID %lld). The maximum "
                        "is currently set to %d, but can be modified "
                        "with the -lg:local flag.", get_task_name(),
                        get_unique_id(), Runtime::max_local_fields);
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
        std::set<unsigned> current_indexes;
        for (std::vector<LocalFieldInfo>::const_iterator it = 
              infos.begin(); it != infos.end(); it++)
          current_indexes.insert(it->index);
        std::vector<unsigned> new_indexes;
        if (!forest->allocate_local_fields(space, resulting_fields, sizes, 
                                  serdez_id, current_indexes, new_indexes))
        {
          log_run.error("Unable to allocate local field in context of "
                        "task %s (UID %lld) due to local field size "
                        "fragmentation. This situation can be improved "
                        "by increasing the maximum number of permitted "
                        "local fields in a context with the -lg:local "
                        "flag.", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_MAX_FIELD_OVERFLOW);
        }
#ifdef DEBUG_LEGION
        assert(new_indexes.size() == resulting_fields.size());
#endif
        // Only need the lock here when writing since we know all writes
        // are serialized and we only need to worry about interfering readers
        AutoLock ctx_lock(context_lock);
        const unsigned offset = infos.size();
        for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
          infos.push_back(LocalFieldInfo(resulting_fields[idx], 
                     sizes[idx], serdez_id, new_indexes[idx], false));
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
        if (!wait_on.has_triggered())
          wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_fields(FieldSpace space, 
                                   const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DeletionOp *op = runtime->get_available_deletion_op(true);
      op->initialize_field_deletions(this, space, to_free);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::create_logical_region(RegionTreeForest *forest,
                                                      IndexSpace index_space,
                                                      FieldSpace field_space)
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
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_top_region(index_space.id, field_space.id, tid);

      forest->create_logical_region(region);
      // Register the creation of a top-level region with the context
      register_region_creation(region);
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
      DeletionOp *op = runtime->get_available_deletion_op(true);
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
      DeletionOp *op = runtime->get_available_deletion_op(true);
      op->initialize_logical_partition_deletion(this, handle);
      runtime->add_to_dependence_queue(this, executing_processor, op);
    }

    //--------------------------------------------------------------------------
    IndexAllocator InnerContext::create_index_allocator(
                                    RegionTreeForest *forest, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      return IndexAllocator(handle, forest->get_index_space_allocator(handle));
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
        FutureImpl *result = legion_new<FutureImpl>(runtime, true/*register*/,
          runtime->get_available_distributed_id(true), 
          runtime->address_space, runtime->address_space);
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
          {
            log_run.error("Predicated task launch for task %s in parent "
                          "task %s (UID %lld) has non-void return type "
                          "but no default value for its future if the task "
                          "predicate evaluates to false.  Please set either "
                          "the 'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "TaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      IndividualTask *task = runtime->get_available_individual_task(true);
#ifdef DEBUG_LEGION
      Future result = 
        task->initialize_task(this, launcher, Runtime::check_privileges);
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
        FutureMapImpl *result = legion_new<FutureMapImpl>(this, runtime);
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
            result->add_reference();
            launcher.predicate_false_future.impl->add_base_gc_ref(
                                                FUTURE_HANDLE_REF);
            Runtime::DeferredFutureMapSetArgs args;
            args.future_map = result;
            args.result = launcher.predicate_false_future.impl;
            args.domain = launcher.launch_domain;
            runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, NULL,
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
          {
            log_run.error("Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "IndexTaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
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
      IndexTask *task = runtime->get_available_index_task(true);
#ifdef DEBUG_LEGION
      FutureMap result = 
        task->initialize_task(this, launcher, Runtime::check_privileges);
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#else
      FutureMap result = task->initialize_task(this, launcher,
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
        FutureImpl *result = legion_new<FutureImpl>(runtime, true/*register*/, 
          runtime->get_available_distributed_id(true), 
          runtime->address_space, runtime->address_space);
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
          {
            log_run.error("Predicated index task launch for task %s "
                          "in parent task %s (UID %lld) has non-void "
                          "return type but no default value for its "
                          "future if the task predicate evaluates to "
                          "false.  Please set either the "
                          "'predicate_false_result' or "
                          "'predicate_false_future' fields of the "
                          "IndexTaskLauncher struct.", impl->get_name(), 
                          get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_MISSING_DEFAULT_PREDICATE_RESULT);
          }
        }
        // Now we can fix the future result
        result->complete_future();
        return Future(result);
      }
      IndexTask *task = runtime->get_available_index_task(true);
#ifdef DEBUG_LEGION
      Future result = 
        task->initialize_task(this, launcher, redop, Runtime::check_privileges);
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#else
      Future result = task->initialize_task(this, launcher, redop, 
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
      MapOp *map_op = runtime->get_available_map_op(true);
#ifdef DEBUG_LEGION
      PhysicalRegion result = 
        map_op->initialize(this, launcher, Runtime::check_privileges);
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
      {
        log_run.error("Attempted an inline mapping of region "
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
                      index, get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK);
      }
      if (inline_conflict)
      {
        log_run.error("Attempted an inline mapping of region (%x,%x,%x) "
                      "that conflicts with previous inline mapping in "
                      "task %s (ID %lld) that would ultimately result in "
                      "deadlock.  Instead you receive this error message.",
                      launcher.requirement.region.index_space.id,
                      launcher.requirement.region.field_space.id,
                      launcher.requirement.region.tree_id,
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      register_inline_mapped_region(result);
      runtime->add_to_dependence_queue(this, executing_processor, map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.impl->is_mapped())
        return;
      MapOp *map_op = runtime->get_available_map_op(true);
      map_op->initialize(this, region);
      register_inline_mapped_region(region);
      runtime->add_to_dependence_queue(this, executing_processor, map_op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if ((region.impl == NULL) || !region.impl->is_mapped())
        return;
      unregister_inline_mapped_region(region);
      region.impl->unmap_region();
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FillOp *fill_op = runtime->get_available_fill_op(true);
#ifdef DEBUG_LEGION
      fill_op->initialize(this, launcher, Runtime::check_privileges);
      log_run.debug("Registering a fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
#else
      fill_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around fill_fields call in task %s (UID %lld).",
              get_task_name(), get_unique_id());
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
      if (launcher.domain.get_volume() == 0)
      {
        log_run.warning("Ignoring empty index space fill in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        return;
      }
      IndexFillOp *fill_op = runtime->get_available_index_fill_op(true);
#ifdef DEBUG_LEGION
      fill_op->initialize(this, launcher, Runtime::check_privileges);
      log_run.debug("Registering an index fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
#else
      fill_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around fill_fields call in task %s (UID %lld).",
              get_task_name(), get_unique_id());
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
      CopyOp *copy_op = runtime->get_available_copy_op(true);
#ifdef DEBUG_LEGION
      copy_op->initialize(this, launcher, Runtime::check_privileges);
      log_run.debug("Registering a copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#else
      copy_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
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
      if (launcher.domain.get_volume() == 0)
      {
        log_run.warning("Ignoring empty index space copy in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        return;
      }
      IndexCopyOp *copy_op = runtime->get_available_index_copy_op(true);
#ifdef DEBUG_LEGION
      copy_op->initialize(this, launcher, Runtime::check_privileges);
      log_run.debug("Registering an index copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#else
      copy_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
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
      AcquireOp *acquire_op = runtime->get_available_acquire_op(true);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing an acquire operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      acquire_op->initialize(this, launcher, Runtime::check_privileges);
#else
      acquire_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this acquire operation.
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(acquire_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_acquire call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
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
      ReleaseOp *release_op = runtime->get_available_release_op(true);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a release operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      release_op->initialize(this, launcher, Runtime::check_privileges);
#else
      release_op->initialize(this, launcher, false/*check privileges*/);
#endif
      // Check to see if we need to do any unmappings and remappings
      // before we can issue the release operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        find_conflicting_regions(release_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_release call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
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
      AttachOp *attach_op = runtime->get_available_attach_op(true);
#ifdef DEBUG_LEGION
      PhysicalRegion result = 
        attach_op->initialize(this, launcher, Runtime::check_privileges);
#else
      PhysicalRegion result = 
        attach_op->initialize(this, launcher, false/*check privileges*/);
#endif
      bool parent_conflict = false, inline_conflict = false;
      int index = has_conflicting_regions(attach_op, 
                                          parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        log_run.error("Attempted an attach hdf5 file operation on region " 
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
                      get_task_name(), get_unique_id(), launcher.file_name);
#ifdef DEBUG_LEGION
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
                      launcher.handle.index_space.id, 
                      launcher.handle.field_space.id, 
                      launcher.handle.tree_id, get_task_name(), 
                      get_unique_id(), launcher.file_name);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK);
      }
      runtime->add_to_dependence_queue(this, executing_processor, attach_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DetachOp *detach_op = runtime->get_available_detach_op(true);
      detach_op->initialize_detach(this, region);
      runtime->add_to_dependence_queue(this, executing_processor, detach_op);
      // If the region is still mapped, then unmap it
      if (region.impl->is_mapped())
      {
        unregister_inline_mapped_region(region);
        region.impl->unmap_region();
      }
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_must_epoch(
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      MustEpochOp *epoch_op = runtime->get_available_epoch_op(true);
#ifdef DEBUG_LEGION
      log_run.debug("Executing a must epoch in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
      FutureMap result = 
        epoch_op->initialize(this, launcher, Runtime::check_privileges);
#else
      FutureMap result = epoch_op->initialize(this, launcher, 
                                              false/*check privileges*/);
#endif
      // Now find all the parent task regions we need to invalidate
      std::vector<PhysicalRegion> unmapped_regions;
      if (!Runtime::unsafe_launch)
        epoch_op->find_conflicted_regions(unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (Runtime::runtime_warnings && !launcher.silence_warnings)
          log_run.warning("WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_release call in "
              "task %s (UID %lld).", get_task_name(), get_unique_id());
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
      TimingOp *timing_op = runtime->get_available_timing_op(true);
      Future result = timing_op->initialize(this, launcher);
      runtime->add_to_dependence_queue(this, executing_processor, timing_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_mapping_fence(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FenceOp *fence_op = runtime->get_available_fence_op(true);
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
      FenceOp *fence_op = runtime->get_available_fence_op(true);
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
      FrameOp *frame_op = runtime->get_available_frame_op(true);
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
      {
        log_run.error("Illegal predicate creation performed on "
                      "empty future inside of task %s (ID %lld).",
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_PREDICATE_FUTURE);
      }
      FuturePredOp *pred_op = runtime->get_available_future_pred_op(true);
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
      NotPredOp *pred_op = runtime->get_available_not_pred_op(true);
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
      {
        log_run.error("Illegal predicate creation performed on a "
                      "set of empty previous predicates in task %s (ID %lld).",
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_PREDICATE_FUTURE);
      }
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
        AndPredOp *pred_op = runtime->get_available_and_pred_op(true);
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
        OrPredOp *pred_op = runtime->get_available_or_pred_op(true);
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
      unsigned outstanding_count = 
        __sync_add_and_fetch(&outstanding_children_count,1);
      // Only need to check if we are not tracing by frames
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count >= context_configuration.max_window_size))
      {
        // Try taking the lock first and see if we succeed
        RtEvent precondition = 
          Runtime::acquire_rt_reservation(context_lock, true/*exclusive*/);
        begin_task_wait(false/*from runtime*/);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // Launch a window-wait task and then wait on the event 
          WindowWaitArgs args;
          args.parent_ctx = this;  
          RtEvent wait_done = 
            runtime->issue_runtime_meta_task(args, LG_RESOURCE_PRIORITY,
                                             owner_task, precondition);
          wait_done.wait();
        }
        else // we can do the wait inline
          perform_window_wait();
        end_task_wait();
      }
      if (Runtime::legion_spy_enabled)
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
      if (Runtime::legion_spy_enabled)
        LegionSpy::log_close_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_window_wait(void)
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
      context_lock.release();
      if (wait_event.exists() && !wait_event.has_triggered())
        wait_event.wait();
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_dependence_queue(Operation *op, bool has_lock,
                                               RtEvent op_precondition)
    //--------------------------------------------------------------------------
    {
      if (!has_lock)
      {
        RtEvent lock_acquire = Runtime::acquire_rt_reservation(context_lock,
                                true/*exclusive*/, last_registration);
        if (!lock_acquire.has_triggered())
        {
          AddToDepQueueArgs args;
          args.proxy_this = this;
          args.op = op;
          args.op_pre = op_precondition;
          last_registration = 
            runtime->issue_runtime_meta_task(args, LG_RESOURCE_PRIORITY,
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
        outstanding_children[op->get_ctx_index()] = op;
#endif       
        executing_children.insert(op);
      }
      // Issue the next dependence analysis task
      DeferredDependenceArgs args;
      args.op = op;
      // If we're ahead we give extra priority to the logical analysis
      // since it is on the critical path, but if not we give it the 
      // normal priority so that we can balance doing logical analysis
      // and actually mapping and running tasks
      if (op_precondition.exists())
      {
        RtEvent pre = Runtime::merge_events(op_precondition, 
                                            dependence_precondition);
        RtEvent next = runtime->issue_runtime_meta_task(args,
                                        currently_active_context ? 
                                          LG_THROUGHPUT_PRIORITY :
                                          LG_DEFERRED_THROUGHPUT_PRIORITY,
                                        op, pre);
        dependence_precondition = next;
      }
      else
      {
        RtEvent next = runtime->issue_runtime_meta_task(args,
                                        currently_active_context ? 
                                          LG_THROUGHPUT_PRIORITY :
                                          LG_DEFERRED_THROUGHPUT_PRIORITY,
                                        op, dependence_precondition);
        dependence_precondition = next;
      }
      // Now we can release the lock
      context_lock.release();
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_executed(Operation *op)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock ctx_lock(context_lock);
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
    void InnerContext::register_child_complete(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock ctx_lock(context_lock);
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
      if (needs_trigger && (owner_task != NULL))
        owner_task->trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_child_commit(Operation *op)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock ctx_lock(context_lock);
        std::set<Operation*>::iterator finder = complete_children.find(op);
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
        AutoLock ctx_lock(context_lock);
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
    void InnerContext::print_children(void)
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
    void InnerContext::register_fence_dependence(Operation *op)
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
        if (op->register_dependence(current_fence, fence_gen))
          current_fence = NULL;
#endif
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_fence_analysis(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      RegionTreeContext ctx = get_context();
      // Do our internal regions first
      for (unsigned idx = 0; idx < regions.size(); idx++)
        runtime->forest->perform_fence_analysis(ctx, op, 
                                        regions[idx].region, true/*dominate*/);
      // Now see if we have any created regions
      // Track separately for the two possible contexts
      std::set<LogicalRegion> local_regions;
      std::set<LogicalRegion> outermost_regions;
      {
        AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
        if (created_requirements.empty())
          return;
        for (unsigned idx = 0; idx < created_requirements.size(); idx++)
        {
          const LogicalRegion &handle = created_requirements[idx].region;
          if (returnable_privileges[idx])
            outermost_regions.insert(handle);
          else
            local_regions.insert(handle);
        }
      }
      if (!local_regions.empty())
      {
        for (std::set<LogicalRegion>::const_iterator it = 
              local_regions.begin(); it != local_regions.end(); it++)
          runtime->forest->perform_fence_analysis(ctx,op,*it,true/*dominate*/);
      }
      if (!outermost_regions.empty())
      {
        // Need outermost context for these regions
        ctx = find_outermost_local_context()->get_context();
        for (std::set<LogicalRegion>::const_iterator it = 
              outermost_regions.begin(); it != outermost_regions.end(); it++)
          runtime->forest->perform_fence_analysis(ctx,op,*it,true/*dominate*/);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::update_current_fence(FenceOp *op)
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
    void InnerContext::begin_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Beginning a trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
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
      std::map<TraceID,DynamicTrace*>::const_iterator finder = traces.find(tid);
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        DynamicTrace *dynamic_trace = legion_new<DynamicTrace>(tid, this);
        dynamic_trace->add_reference();
        traces[tid] = dynamic_trace;
        current_trace = dynamic_trace;
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
    void InnerContext::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Ending a trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
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
      else if (!current_trace->is_dynamic_trace())
      {
        log_task.error("Illegal end trace call on a static trace in "
                       "task %s (UID %lld)", get_task_name(), get_unique_id());
      }
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op(true);
        complete_op->initialize_complete(this);
        runtime->add_to_dependence_queue(this, executing_processor, complete_op);
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(true); 
        capture_op->initialize_capture(this);
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
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Beginning a static trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
      {
        log_task.error("Illegal nested static trace attempted in "
                       "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_ILLEGAL_NESTED_TRACE);
      }
      // Issue the mapping fence into the analysis
      runtime->issue_mapping_fence(this);
      // Then we make a static trace
      current_trace = legion_new<StaticTrace>(this, trees); 
      current_trace->add_reference();
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_static_trace(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Ending a static trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      if (current_trace == NULL)
      {
        log_task.error("Unmatched end static trace in task %s "
                       "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNMATCHED_END_TRACE);
      }
      else if (current_trace->is_dynamic_trace())
      {
        log_task.error("Illegal end static trace call on a dynamic trace in "
                       "task %s (UID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNMATCHED_END_TRACE);
      }
      // We're done with this trace, need a trace complete op to clean up
      // This operation takes ownership of the static trace reference
      TraceCompleteOp *complete_op = runtime->get_available_trace_op(true);
      complete_op->initialize_complete(this);
      runtime->add_to_dependence_queue(this, executing_processor, complete_op);
      // We no longer have a trace that we're executing 
      current_trace = NULL;
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_frame(FrameOp *frame, ApEvent frame_termination)
    //--------------------------------------------------------------------------
    {
      // This happens infrequently enough that we can just issue
      // a meta-task to see what we should do without holding the lock
      if (context_configuration.max_outstanding_frames > 0)
      {
        IssueFrameArgs args;
        args.parent_ctx = this;
        args.frame = frame;
        args.frame_termination = frame_termination;
        // We know that the issuing is done in order because we block after
        // we launch this meta-task which blocks the application task
        RtEvent wait_on = runtime->issue_runtime_meta_task(args,
                                      LG_LATENCY_PRIORITY, owner_task);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
    RtEvent InnerContext::decrement_pending(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduled by frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return RtEvent::NO_RT_EVENT;
      // This may involve waiting, so always issue it as a meta-task 
      DecrementArgs decrement_args;
      decrement_args.parent_ctx = const_cast<InnerContext*>(this);
      RtEvent precondition = 
        Runtime::acquire_rt_reservation(context_lock, true/*exclusive*/);
      return runtime->issue_runtime_meta_task(decrement_args, 
                  LG_RESOURCE_PRIORITY, child, precondition);
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent to_trigger;
      // We already hold the lock from the dispatch site (see above)
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
      // Release the lock before doing the trigger or the wait
      context_lock.release();
      // Do anything that we need to do
      if (to_trigger.exists())
      {
        wait_on.wait();
        runtime->activate_context(this);
        Runtime::trigger_event(to_trigger);
      }
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      {
        // We faiiled to acquire, report the error
        log_run.error("Illegal acquire operation (ID %lld) performed in "
                      "task %s (ID %lld). Acquire was performed on a non-"
                      "restricted region.", op->get_unique_op_id(),
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNRESTRICTED_ACQUIRE);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::remove_acquisition(ReleaseOp *op, 
                                          const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      if (!runtime->forest->remove_acquisition(coherence_restrictions, op, req))
      {
        // We failed to release, report the error
        log_run.error("Illegal release operation (ID %lld) performed in "
                      "task %s (ID %lld). Release was performed on a region "
                      "that had not previously been acquired.",
                      op->get_unique_op_id(), get_task_name(), 
                      get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNACQUIRED_RELEASE);
      }
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
      if (!runtime->forest->remove_restriction(coherence_restrictions, op, req))
      {
        // We failed to remove the restriction
        log_run.error("Illegal detach operation (ID %lld) performed in "
                      "task %s (ID %lld). Detach was performed on an region "
                      "that had not previously been attached.",
                      op->get_unique_op_id(), get_task_name(),
                      get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_UNATTACHED_DETACH);
      }
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
      // This is a little Realm specific, but can't avoid it at the moment
      // 20 bits for the generation and everything else is ID
      const unsigned long id = dc.phase_barrier.id & 0xFFFFFFFFFFF00000UL;
      const unsigned gen = dc.phase_barrier.id & 0xFFFFF;
      AutoLock ctx(context_lock);
      collective_contributions[id][gen].push_back(f);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_collective_contributions(DynamicCollective dc, 
                                             std::vector<Future> &contributions)
    //--------------------------------------------------------------------------
    {
      // Find any future contributions and record dependences for the op
      // Contributions were made to the previous phase
      ApEvent previous = Runtime::get_previous_phase(dc.phase_barrier);
      const unsigned long id = previous.id & 0xFFFFFFFFFFF00000UL;
      const unsigned gen = previous.id & 0xFFFFF;
      AutoLock ctx(context_lock);
      std::map<unsigned long, std::map<unsigned,
        std::vector<Future> > >::iterator finder = 
          collective_contributions.find(id);
      if (finder == collective_contributions.end())
        return;
      std::map<unsigned,std::vector<Future> >::iterator it = 
        finder->second.begin();
      while ((it != finder->second.end()) && (it->first <= gen))
      {
        if (it->first == gen)
        {
          contributions = it->second;
          it++;
        }
        else
        {
          std::map<unsigned,std::vector<Future> >::iterator to_erase = it;
          it++;
          finder->second.erase(to_erase);
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::configure_context(MapperManager *mapper)
    //--------------------------------------------------------------------------
    {
      mapper->invoke_configure_context(owner_task, &context_configuration);
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
      // Invalidate all our region contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
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
        RegionTreeContext outermost_ctx = outermost->get_context();
        const bool is_outermost = (outermost == this);
        for (unsigned idx = 0; idx < created_requirements.size(); idx++)
        {
          // See if we're a returnable privilege or not
          if (returnable_privileges[idx])
          {
            // If we're the outermost context or the requirement was
            // deleted, then we can invalidate everything
            // Otherwiswe we only invalidate the users
            const bool users_only = !is_outermost && 
              !was_created_requirement_deleted(created_requirements[idx]);
            runtime->forest->invalidate_current_context(outermost_ctx,
                        users_only, created_requirements[idx].region);
          }
          else // Not returning so invalidate the full thing 
            runtime->forest->invalidate_current_context(tree_context,
                false/*users only*/, created_requirements[idx].region);
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
            LogicalView::delete_logical_view(it->second);
        }
        instance_top_views.clear();
      }
      // Before freeing our context, see if there are any version
      // state managers we need to reset
      if (!region_tree_owners.empty())
      {
        for (std::map<RegionTreeNode*,
                      std::pair<AddressSpaceID,bool> >::const_iterator it =
              region_tree_owners.begin(); it != region_tree_owners.end(); it++)
        {
          // If this is a remote only then we don't need to invalidate it
          if (!it->second.second)
            it->first->invalidate_version_state(tree_context.get_id()); 
        }
        region_tree_owners.clear();
      }
      // Now we can free our region tree context
      runtime->free_region_tree_context(tree_context);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock, 1, false/*exclusive*/);
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
        AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
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
      runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, 
                                       context->get_owner_task());
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
      AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
    void InnerContext::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      runtime->decrement_total_outstanding_tasks(owner_task->task_id, 
                                                 false/*meta*/);
#else
      runtime->decrement_total_outstanding_tasks();
#endif
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
      if (Runtime::runtime_warnings)
      {
        if (total_children_count == 0)
        {
          // If there were no sub operations and this wasn't marked a
          // leaf task then signal a warning
          VariantImpl *impl = 
            runtime->find_variant_impl(single_task->task_id, 
                                       single_task->get_selected_variant());
          log_run.warning("WARNING: Variant %s of task %s (UID %lld) was "
              "not marked as a 'leaf' variant but it didn't execute any "
              "operations. Did you forget the 'leaf' annotation?", 
              impl->get_name(), get_task_name(), get_unique_id());
        }
        else if (!single_task->is_inner())
        {
          // If this task had sub operations and wasn't marked as inner
          // and made no accessors warn about missing 'inner' annotation
          bool has_accessor = false;
          for (unsigned idx = 0; idx < physical_regions.size(); idx++)
          {
            if (!physical_regions[idx].impl->created_accessor())
              continue;
            has_accessor = true;
            break;
          }
          if (!has_accessor)
          {
            VariantImpl *impl = 
              runtime->find_variant_impl(single_task->task_id, 
                                         single_task->get_selected_variant());
            log_run.warning("WARNING: Variant %s of task %s (UID %lld) was "
                "not marked as an 'inner' variant but it only launched "
                "operations and did not make any accessors. Did you "
                "forget the 'inner' annotation?",
                impl->get_name(), get_task_name(), get_unique_id());
          }
        }
      }
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
      // Unmap any of our mapped regions before issuing any close operations
      unmap_all_regions();
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
            runtime->get_available_post_close_op(true);
          close_op->initialize(this, idx, physical_instances[idx]);
          runtime->add_to_dependence_queue(this, executing_processor, close_op);
        }
        else
        {
          // Make a virtual close op to close up the instance
          VirtualCloseOp *close_op = 
            runtime->get_available_virtual_close_op(true);
          close_op->initialize(this, idx, regions[idx]);
          runtime->add_to_dependence_queue(this, executing_processor, close_op);
        }
      }
      // See if we want to move the rest of this computation onto
      // the utility processor. We also need to be sure that we have 
      // registered all of our operations before we can do the post end task
      if (runtime->has_explicit_utility_procs || 
          !last_registration.has_triggered())
      {
        PostEndArgs post_end_args;
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
        runtime->issue_runtime_meta_task(post_end_args,
           LG_LATENCY_PRIORITY, owner_task, last_registration);
      }
      else
        post_end_task(res, res_size, owned);
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
          AutoLock ctx_lock(context_lock);
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
          // Finally unmap any of our mapped physical instances
#ifdef DEBUG_LEGION
          assert((regions.size() + 
                    created_requirements.size()) == physical_regions.size());
#endif
        }
        if (!preconditions.empty())
          single_task->handle_post_mapped(Runtime::merge_events(preconditions));
        else
          single_task->handle_post_mapped();
      }
      // Mark that we are done executing this operation
      // We're not actually done until we have registered our pending
      // decrement of our parent task and recorded any profiling
      if (!pending_done.has_triggered())
        owner_task->complete_execution(pending_done);
      else
        owner_task->complete_execution();
      if (need_complete)
        owner_task->trigger_children_complete();
      if (need_commit)
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void InnerContext::invalidate_remote_contexts(void)
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
      AutoLock ctx_lock(context_lock);
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
        AutoLock ctx_lock(context_lock);
#ifdef DEBUG_LEGION
        assert(region_tree_owners.find(node) == region_tree_owners.end());
#endif
        region_tree_owners[node] = 
          std::pair<AddressSpaceID,bool>(result, false/*remote only*/); 
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
          MapOp *op = runtime->get_available_map_op(true);
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
        if (!wait_on.has_triggered())
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
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    void InnerContext::clone_local_fields(
           std::map<FieldSpace,std::vector<LocalFieldInfo> > &child_local) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child_local.empty());
#endif
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
      // Tell the runtime that another top level task is done
      runtime->decrement_outstanding_top_level_tasks();
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
      // Invalidate our context if necessary before deactivating
      // the wrapper as it will release the context
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
      }
      else
        runtime->forest->invalidate_all_versions(tree_context);
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
        AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock,1,false/*exclusive*/);
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
                       const FieldMask &version_mask, VersionInfo &version_info)
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
      version_infos[index].clone_to_depth(depth, version_mask, version_info);
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
          AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
      AutoLock ctx_lock(context_lock);
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
                                               size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index space creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(RegionTreeForest *forest,
                                               const Domain &domain)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index space creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(RegionTreeForest *forest,
                                               const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index space creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index space deletion performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf task "
                     "%s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                              IndexSpace parent,
                                              const Coloring &coloring,
                                              bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf task "
                     "%s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                                 IndexSpace parent,
                                                 const Domain &color_space,
                                                 const DomainColoring &coloring,
                                                 bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_index_partition(RegionTreeForest *forest,
                                        IndexSpace parent,
                LegionRuntime::Accessor::RegionAccessor<
                 LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                        int part_color)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_partition(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index partition deletion performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_equal_partition(RegionTreeForest *forest,
                                             IndexSpace parent,
                                             const Domain &color_space,
                                             size_t granularity,
                                             int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal equal partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_weighted_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const std::map<DomainPoint,int> &weights,
                                      size_t granularity,
                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal weighted partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_union(
                                          RegionTreeForest *forest,
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          PartitionKind kind,
                                          int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal union partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
                                                RegionTreeForest *forest,
                                                IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                PartitionKind kind,
                                                int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal intersection partition creation performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_difference(
                                                      RegionTreeForest *forest,
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      PartitionKind kind,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal difference partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    void LeafContext::create_cross_product_partition(RegionTreeForest *forest,
                                                     IndexPartition handle1,
                                                     IndexPartition handle2,
                                  std::map<DomainPoint,IndexPartition> &handles,
                                                     PartitionKind kind,
                                                     int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create cross product partitions performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_field(
                                                RegionTreeForest *forest,
                                                LogicalRegion handle,
                                                LogicalRegion parent_priv,
                                                FieldID fid,
                                                const Domain &color_space,
                                                int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal partition by field performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image(
                                              RegionTreeForest *forest,
                                              IndexSpace handle,
                                              LogicalPartition projection,
                                              LogicalRegion parent,
                                              FieldID fid,
                                              const Domain &color_space,
                                              PartitionKind part_kind,
                                              int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal partition by image performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage(
                                                RegionTreeForest *forest,
                                                IndexPartition projection,
                                                LogicalRegion handle,
                                                LogicalRegion parent,
                                                FieldID fid,
                                                const Domain &color_space,
                                                PartitionKind part_kind,
                                                int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal partition by preimage performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_pending_partition(
                                              RegionTreeForest *forest,
                                              IndexSpace parent,
                                              const Domain &color_space,
                                              PartitionKind part_kind,
                                              int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create pending partition performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const DomainPoint &color,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles) 
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     RegionTreeForest *forest,
                                                     IndexPartition parent,
                                                     const DomainPoint &color,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_difference(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                                  IndexSpace initial,
                                          const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index space difference performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create field space performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return FieldSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_field_space(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal destroy field space performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    FieldID LeafContext::allocate_field(RegionTreeForest *forest,
                                        FieldSpace space, size_t field_size,
                                        FieldID fid, bool local,
                                        CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal non-local field allocation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_field(FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal field destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_fields(RegionTreeForest *forest,
                                      FieldSpace space,
                                      const std::vector<size_t> &sizes,
                                      std::vector<FieldID> &resuling_fields,
                                      bool local, CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal non-local field allocation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_fields(FieldSpace space, 
                                  const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal field destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    LogicalRegion LeafContext::create_logical_region(RegionTreeForest *forest,
                                                     IndexSpace index_space,
                                                     FieldSpace field_space)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal region creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return LogicalRegion::NO_REGION;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal region destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal partition destruction performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    IndexAllocator LeafContext::create_index_allocator(RegionTreeForest *forest,
                                                       IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create index allocation requested in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return IndexAllocator(handle, forest->get_index_space_allocator(handle));
    }

    //--------------------------------------------------------------------------
    FieldAllocator LeafContext::create_field_allocator(
                                   Legion::Runtime *external, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal create field allocation requested in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return FieldAllocator(handle, this, external);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal execute task call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Future();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_index_space(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal execute index space call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_index_space(const IndexTaskLauncher &launcher,
                                            ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal execute index space call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Future();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::map_region(const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal map_region operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return PhysicalRegion();
    }

    //--------------------------------------------------------------------------
    void LeafContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal remap operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal unmap operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal fill operation call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::fill_fields(const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index fill operation call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal copy operation call performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_copy(const IndexCopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal index copy operation call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_acquire(const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal acquire operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_release(const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal release operation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion LeafContext::attach_resource(const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal attach resource operation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return PhysicalRegion();
    }
    
    //--------------------------------------------------------------------------
    void LeafContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal detach resource operation performed in leaf "
                      "task %s (ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_must_epoch(const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion execute must epoch call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_timing_measurement(const TimingLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal timing measurement operation in leaf task %s"
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Future();
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_mapping_fence(void)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal legion mapping fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::issue_execution_fence(void)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion execution fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::complete_frame(void)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion complete frame call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const Future &f)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal predicate creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Predicate();
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::predicate_not(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal NOT predicate creation in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Predicate();
    }
    
    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal predicate creation performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
      return Predicate();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_predicate_future(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal get predicate future performed in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
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
    void LeafContext::add_to_dependence_queue(Operation *op, bool has_lock,
                                              RtEvent op_precondition)
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
    void LeafContext::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::perform_fence_analysis(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::update_current_fence(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::begin_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion begin trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion end trace call in leaf task %s (ID %lld)",
                     get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::begin_static_trace(const std::set<RegionTreeID> *managed)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion begin static trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_static_trace(void)
    //--------------------------------------------------------------------------
    {
      log_task.error("Illegal Legion end static trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
      assert(false);
#endif
      exit(ERROR_LEAF_TASK_VIOLATION);
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
    RtEvent LeafContext::decrement_pending(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
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
    InnerContext* LeafContext::find_parent_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void LeafContext::find_parent_version_info(unsigned index, unsigned depth,
                       const FieldMask &version_mask, VersionInfo &version_info)
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
    void LeafContext::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_task != NULL);
      runtime->decrement_total_outstanding_tasks(owner_task->task_id, 
                                                 false/*meta*/);
#else
      runtime->decrement_total_outstanding_tasks();
#endif
      if (overhead_tracker != NULL)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff = current - previous_profiling_time;
        overhead_tracker->application_time += diff;
      }
      if (runtime->has_explicit_utility_procs)
      {
        PostEndArgs post_end_args;
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
        runtime->issue_runtime_meta_task(post_end_args,
                                         LG_LATENCY_PRIORITY, owner_task);
      }
      else
        post_end_task(res, res_size, owned);
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
      std::vector<PhysicalRegion> unmap_regions;
      {
        AutoLock ctx_lock(context_lock);
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
        // Finally unmap any physical regions that we mapped
#ifdef DEBUG_LEGION
        assert((regions.size() + 
                  created_requirements.size()) == physical_regions.size());
#endif
        for (std::vector<PhysicalRegion>::const_iterator it = 
              physical_regions.begin(); it != physical_regions.end(); it++)
        {
          if (it->impl->is_mapped())
            unmap_regions.push_back(*it);
        }
      }
      // Do the unmappings while not holding the lock in case we block
      if (!unmap_regions.empty())
      {
        for (std::vector<PhysicalRegion>::const_iterator it = 
              unmap_regions.begin(); it != unmap_regions.end(); it++)
          it->impl->unmap_region();
      }
      // Mark that we are done executing this operation
      // We're not actually done until we have registered our pending
      // decrement of our parent task and recorded any profiling
      if (!pending_done.has_triggered())
        owner_task->complete_execution(pending_done);
      else
        owner_task->complete_execution();
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
                                                 size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space(forest, max_num_elmts);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space(RegionTreeForest *forest,
                                                 const Domain &domain)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space(forest, domain);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space(RegionTreeForest *forest,
                                                const std::set<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space(forest, domains);
    }

    //--------------------------------------------------------------------------
    void InlineContext::destroy_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      enclosing->destroy_index_space(handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, color_space,
                                     coloring, part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, coloring,
                                               disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, color_space,
                                               coloring, part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, color_space,
                                               coloring, disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind, int color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, color_space,
                                               coloring, part_kind, color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const MultiDomainColoring &coloring,
                                      bool disjoint, int part_color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, color_space,
                                               coloring, disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_index_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                LegionRuntime::Accessor::RegionAccessor<
                 LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                      int part_color)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_partition(forest, parent, 
                                               field_accessor, part_color);
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
                                                const Domain &color_space,
                                                size_t granularity,
                                                int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_equal_partition(forest, parent, color_space,
                                               granularity, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_weighted_partition(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      const Domain &color_space,
                                      const std::map<DomainPoint,int> &weights,
                                      size_t granularity,
                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_weighted_partition(forest, parent, color_space,
                                        weights, granularity, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_union(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      PartitionKind kind,
                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_union(forest, parent, handle1,
                                          handle2, kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_intersection(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      PartitionKind kind,
                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_intersection(forest, parent,
                                handle1, handle2, kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_difference(
                                      RegionTreeForest *forest,
                                      IndexSpace parent,
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      PartitionKind kind,
                                      int color, bool allocable)   
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_difference(forest, parent,
                                handle1, handle2, kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    void InlineContext::create_cross_product_partition(RegionTreeForest *forest,
                                                       IndexPartition handle1,
                                                       IndexPartition handle2,
                                  std::map<DomainPoint,IndexPartition> &handles,
                                                       PartitionKind kind,
                                                       int color,bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_cross_product_partition(forest, handle1, handle2,
                                               handles, kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_field(
                                                      RegionTreeForest *forest,
                                                      LogicalRegion handle,
                                                      LogicalRegion parent_priv,
                                                      FieldID fid,
                                                      const Domain &color_space,
                                                      int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_field(forest, handle, parent_priv,
                                          fid, color_space, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_image(
                                                    RegionTreeForest *forest,
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_image(forest, handle, projection,
                      parent, fid, color_space, part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_partition_by_preimage(
                                                    RegionTreeForest *forest,
                                                    IndexPartition projection,
                                                    LogicalRegion handle,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_partition_by_preimage(forest, projection, handle,
                         parent, fid, color_space, part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexPartition InlineContext::create_pending_partition(
                                                    RegionTreeForest *forest,
                                                    IndexSpace parent,
                                                    const Domain &color_space,
                                                    PartitionKind part_kind,
                                                    int color, bool allocable)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_pending_partition(forest, parent, color_space,
                                                 part_kind, color, allocable);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_union(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_union(forest, parent, color,handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_union(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                                  IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_union(forest, parent, color, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_intersection(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_intersection(forest, parent, 
                                                        color, handles);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_intersection(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                                  IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_intersection(forest, parent, 
                                                        color, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace InlineContext::create_index_space_difference(
                                                  RegionTreeForest *forest,
                                                  IndexPartition parent,
                                                  const DomainPoint &color,
                                                  IndexSpace initial,
                                        const std::vector<IndexSpace> &handles)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_space_difference(forest, parent, color,
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
                                                       FieldSpace field_space)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_logical_region(forest, index_space, field_space);
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
    IndexAllocator InlineContext::create_index_allocator(
                                    RegionTreeForest *forest, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return enclosing->create_index_allocator(forest, handle);
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
    void InlineContext::remap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      enclosing->remap_region(region);
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
    void InlineContext::detach_resource(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      enclosing->detach_resource(region);
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
    void InlineContext::add_to_dependence_queue(Operation *op, bool has_lock,
                                                RtEvent op_precondition)
    //--------------------------------------------------------------------------
    {
      enclosing->add_to_dependence_queue(op, has_lock, op_precondition);
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
    void InlineContext::register_fence_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      enclosing->register_fence_dependence(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::perform_fence_analysis(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      enclosing->perform_fence_analysis(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::update_current_fence(FenceOp *op)
    //--------------------------------------------------------------------------
    {
      enclosing->update_current_fence(op);
    }

    //--------------------------------------------------------------------------
    void InlineContext::begin_trace(TraceID tid)
    //--------------------------------------------------------------------------
    {
      enclosing->begin_trace(tid);
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
    RtEvent InlineContext::decrement_pending(TaskOp *child) const
    //--------------------------------------------------------------------------
    {
      return enclosing->decrement_pending(child);
    }

    //--------------------------------------------------------------------------
    void InlineContext::decrement_pending(void)
    //--------------------------------------------------------------------------
    {
      enclosing->decrement_pending();
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
    InnerContext* InlineContext::find_parent_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < parent_req_indexes.size());
#endif
      return enclosing->find_parent_physical_context(parent_req_indexes[index]);
    }

    //--------------------------------------------------------------------------
    void InlineContext::find_parent_version_info(unsigned index, unsigned depth,
                       const FieldMask &version_mask, VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      enclosing->find_parent_version_info(index, depth, 
                                          version_mask, version_info);
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
    const std::vector<PhysicalRegion>& InlineContext::begin_task(void)
    //--------------------------------------------------------------------------
    {
      return enclosing->get_physical_regions();
    }

    //--------------------------------------------------------------------------
    void InlineContext::end_task(const void *res, size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
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

  };
};

// EOF

