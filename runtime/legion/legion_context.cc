/* Copyright 2022 Stanford University, NVIDIA Corporation
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
#include "realm/id.h"

#define SWAP_PART_KINDS(k1, k2) \
  {                             \
    PartitionKind temp = k1;    \
    k1 = k2;                    \
    k2 = temp;                  \
  }

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Task Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(Runtime *rt, SingleTask *owner, int d,
                      const std::vector<RegionRequirement> &reqs,
                      bool inline_t, bool implicit_t)
      : runtime(rt), owner_task(owner), regions(reqs), depth(d),
        next_created_index(reqs.size()), 
        executing_processor(Processor::NO_PROC), total_tunable_count(0), 
        overhead_tracker(NULL), task_executed(false),
        has_inline_accessor(false), mutable_priority(false),
        children_complete_invoked(false), children_commit_invoked(false),
        inline_task(inline_t), implicit_task(implicit_t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(const TaskContext &rhs)
      : runtime(NULL), owner_task(NULL), regions(rhs.regions), depth(-1), 
        inline_task(false), implicit_task(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(deletion_counts.empty());
#endif
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
      if (overhead_tracker != NULL)
        delete overhead_tracker;
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

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void TaskContext::perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, const void *buffer,
                     size_t buffer_size, bool withargs, size_t dedup_tag,
                     RtEvent local_done, RtEvent global_done,
                     std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Send messages to all the other nodes to perform it
      for (AddressSpaceID space = 0; 
            space < runtime->total_address_spaces; space++)
      {
        if (space == runtime->address_space)
          continue;
        runtime->send_registration_callback(space, dso, global_done,
            preconditions, buffer, buffer_size, withargs, 
            true/*deduplicate*/, dedup_tag);
      }
    }
#endif

    //--------------------------------------------------------------------------
    VariantID TaskContext::register_variant(
            const TaskVariantRegistrar &registrar, const void *user_data,
            size_t user_data_size, const CodeDescriptor &desc, bool ret,
            VariantID vid, bool check_task_id)
    //--------------------------------------------------------------------------
    {
      return runtime->register_variant(registrar, user_data, user_data_size,
          desc, ret, vid, check_task_id, false/*check context*/);
    }

    //--------------------------------------------------------------------------
    TraceID TaskContext::generate_dynamic_trace_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_trace_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    MapperID TaskContext::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_mapper_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    ProjectionID TaskContext::generate_dynamic_projection_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_projection_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    TaskID TaskContext::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_task_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    ReductionOpID TaskContext::generate_dynamic_reduction_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_reduction_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID TaskContext::generate_dynamic_serdez_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_serdez_id(false/*check context*/);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::perform_semantic_attach(bool &global)
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskContext::post_semantic_attach(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::create_index_space(const Domain &bounds, 
                                       TypeTag type_tag, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      return create_index_space_internal(bounds, type_tag, provenance);
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::create_index_space(
                 const std::vector<DomainPoint> &points, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      switch (points[0].get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            std::vector<Realm::Point<DIM,coord_t> > \
              realm_points(points.size()); \
            for (unsigned idx = 0; idx < points.size(); idx++) \
              realm_points[idx] = Point<DIM,coord_t>(points[idx]); \
            const DomainT<DIM,coord_t> realm_is( \
                (Realm::IndexSpace<DIM,coord_t>(realm_points))); \
            const Domain bounds(realm_is); \
            return create_index_space_internal(bounds, \
                NT_TemplateHelper::encode_tag<DIM,coord_t>(), provenance); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::create_index_space(const std::vector<Domain> &rects,
                                               const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      switch (rects[0].get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            std::vector<Realm::Rect<DIM,coord_t> > realm_rects(rects.size()); \
            for (unsigned idx = 0; idx < rects.size(); idx++) \
              realm_rects[idx] = Rect<DIM,coord_t>(rects[idx]); \
            const DomainT<DIM,coord_t> realm_is( \
                (Realm::IndexSpace<DIM,coord_t>(realm_rects))); \
            const Domain bounds(realm_is); \
            return create_index_space_internal(bounds, \
                NT_TemplateHelper::encode_tag<DIM,coord_t>(), provenance); \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::create_index_space_internal(const Domain &bounds,
                                             TypeTag type_tag, const char *prov)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(runtime->get_unique_index_space_id(),
                        runtime->get_unique_index_tree_id(), type_tag);
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space %x in task%s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id()); 
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.id, runtime->address_space, prov);
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = new Provenance(prov);
      // Will take ownership of provenance if not NULL
      runtime->forest->create_index_space(handle, &bounds, did, provenance); 
      register_index_space_creation(handle);
      return handle;
    }

    //--------------------------------------------------------------------------
    void TaskContext::create_shared_ownership(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->forest->is_top_level_index_space(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_SHARED_OWNERSHIP,
            "Illegal call to create shared ownership for index space %x in " 
            "task %s (UID %lld) which is not a top-level index space. Legion "
            "only permits top-level index spaces to have shared ownership.", 
            handle.get_id(), get_task_name(), get_unique_id())
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexSpace,unsigned>::iterator finder = 
        created_index_spaces.find(handle);
      if (finder != created_index_spaces.end())
        finder->second++;
      else
        created_index_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::union_index_spaces(
                  const std::vector<IndexSpace> &spaces, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      AutoRuntimeCall call(this); 
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            spaces.begin(); it != spaces.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'union_index_spaces' "
                        "performed in task %s (UID %lld)",
                        get_task_name(), get_unique_id())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace handle(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), spaces[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->forest->create_union_space(handle, did, provenance, spaces);
      register_index_space_creation(handle);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.get_id(),
                    runtime->address_space, provenance);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::intersect_index_spaces(
                  const std::vector<IndexSpace> &spaces, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      AutoRuntimeCall call(this); 
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            spaces.begin(); it != spaces.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'intersect_index_spaces' "
                        "performed in task %s (UID %lld)",
                        get_task_name(), get_unique_id())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace handle(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), spaces[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->forest->create_intersection_space(handle,did,provenance,spaces);
      register_index_space_creation(handle);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.get_id(),
                    runtime->address_space, provenance);
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::subtract_index_spaces(
                      IndexSpace left, IndexSpace right, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      if (!left.exists())
        return IndexSpace::NO_SPACE;
      if (right.exists() && left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'create_difference_spaces' "
                        "performed in task %s (UID %lld)",
                        get_task_name(), get_unique_id())
      const IndexSpace handle(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), left.get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->forest->create_difference_space(handle, did, provenance,
                                               left, right); 
      register_index_space_creation(handle);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(handle.get_id(),
                    runtime->address_space, provenance);
      return handle;
    }

    //--------------------------------------------------------------------------
    void TaskContext::create_shared_ownership(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexPartition,unsigned>::iterator finder = 
        created_index_partitions.find(handle);
      if (finder != created_index_partitions.end())
        finder->second++;
      else
        created_index_partitions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    FieldSpace TaskContext::create_field_space(const char *provenance)
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
        LegionSpy::log_field_space(space.id, runtime->address_space,provenance);

      runtime->forest->create_field_space(space, did, provenance);
      register_field_space_creation(space);
      return space;
    }

    //--------------------------------------------------------------------------
    FieldSpace TaskContext::create_field_space(
                                         const std::vector<size_t> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         CustomSerdezID serdez_id,
                                         const char *provenance)
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
        LegionSpy::log_field_space(space.id, runtime->address_space,provenance);

      FieldSpaceNode *node =
        runtime->forest->create_field_space(space, did, provenance);
      register_field_space_creation(space);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
          REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD,
            "Task %s (ID %lld) attempted to allocate a field with "
            "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
            "bound set in legion_config.h", get_task_name(),
            get_unique_id(), resulting_fields[idx])
#endif
        if (runtime->legion_spy_enabled)
          LegionSpy::log_field_creation(space.id, resulting_fields[idx],
                                        sizes[idx], provenance);
      }
      node->initialize_fields(sizes, resulting_fields, serdez_id, provenance);
      register_all_field_creations(space, false/*local*/, resulting_fields);
      return space;
    }

    //--------------------------------------------------------------------------
    void TaskContext::create_shared_ownership(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<FieldSpace,unsigned>::iterator finder = 
        created_field_spaces.find(handle);
      if (finder != created_field_spaces.end())
        finder->second++;
      else
        created_field_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    FieldAllocatorImpl* TaskContext::create_field_allocator(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        std::map<FieldSpace,FieldAllocatorImpl*>::const_iterator finder = 
          field_allocators.find(handle);
        if (finder != field_allocators.end())
          return finder->second;
      }
      // Didn't find it, so have to make, retake the lock in exclusive mode
      FieldSpaceNode *node = runtime->forest->get_node(handle);
      AutoLock priv_lock(privilege_lock);
      // Check to see if we lost the race
      std::map<FieldSpace,FieldAllocatorImpl*>::const_iterator finder = 
        field_allocators.find(handle);
      if (finder != field_allocators.end())
        return finder->second;
      // Don't have one so make a new one
      const RtEvent ready = node->create_allocator(runtime->address_space);
      FieldAllocatorImpl *result = new FieldAllocatorImpl(node, this, ready);
      // Save it for later
      field_allocators[handle] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::destroy_field_allocator(FieldSpaceNode *node)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      const RtEvent ready = node->destroy_allocator(runtime->address_space);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      AutoLock priv_lock(privilege_lock);
      std::map<FieldSpace,FieldAllocatorImpl*>::iterator finder = 
        field_allocators.find(node->handle);
#ifdef DEBUG_LEGION
      assert(finder != field_allocators.end());
#endif
      field_allocators.erase(finder);
    }

    //--------------------------------------------------------------------------
    FieldID TaskContext::allocate_field(FieldSpace space, size_t field_size,
                                        FieldID fid, bool local,
                                        CustomSerdezID serdez_id,
                                        const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (fid == LEGION_AUTO_GENERATE_ID)
        fid = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
        REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FILED,
          "Task %s (ID %lld) attempted to allocate a field with "
          "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
          "bound set in legion_config.h", get_task_name(), get_unique_id(), fid)
#endif
      if (runtime->legion_spy_enabled)
        LegionSpy::log_field_creation(space.id, fid, field_size, provenance);

      std::set<RtEvent> done_events;
      if (local)
        allocate_local_field(space, field_size, fid, 
                             serdez_id, done_events, provenance);
      else
        runtime->forest->allocate_field(space, field_size, fid,
                                        serdez_id, provenance);
      register_field_creation(space, fid, local);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
      return fid;
    }

    //--------------------------------------------------------------------------
    void TaskContext::allocate_fields(FieldSpace space,
                                      const std::vector<size_t> &sizes,
                                      std::vector<FieldID> &resulting_fields,
                                      bool local, CustomSerdezID serdez_id,
                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
          REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD,
            "Task %s (ID %lld) attempted to allocate a field with "
            "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
            "bound set in legion_config.h", get_task_name(),
            get_unique_id(), resulting_fields[idx])
#endif
        if (runtime->legion_spy_enabled)
          LegionSpy::log_field_creation(space.id, resulting_fields[idx],
                                        sizes[idx], provenance);
      }
      std::set<RtEvent> done_events;
      if (local)
        allocate_local_fields(space, sizes, resulting_fields,
                              serdez_id, done_events, provenance);
      else
        runtime->forest->allocate_fields(space, sizes, resulting_fields,
                                         serdez_id, provenance);
      register_all_field_creations(space, local, resulting_fields);
      if (!done_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(done_events);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    LogicalRegion TaskContext::create_logical_region(IndexSpace index_space,
                                                     FieldSpace field_space,
                                                     bool task_local,
                                                     const char *provenance)
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
        LegionSpy::log_top_region(index_space.id, field_space.id,
                                  tid, runtime->address_space, provenance);

      const DistributedID did = runtime->get_available_distributed_id(); 
      runtime->forest->create_logical_region(region, did, provenance);
      // Register the creation of a top-level region with the context
      register_region_creation(region, task_local);
      return region;
    }

    //--------------------------------------------------------------------------
    void TaskContext::create_shared_ownership(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      if (!runtime->forest->is_top_level_region(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_SHARED_OWNERSHIP,
            "Illegal call to create shared ownership for logical region "
            "(%x,%x,%x in task %s (UID %lld) which is not a top-level logical "
            "region. Legion only permits top-level logical regions to have "
            "shared ownerships.", handle.index_space.id, handle.field_space.id,
            handle.tree_id, get_task_name(), get_unique_id())
      runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<LogicalRegion,unsigned>::iterator finder = 
        created_regions.find(handle);
      if (finder != created_regions.end())
        finder->second++;
      else
        created_regions[handle] = 1;
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
    void TaskContext::add_created_region(LogicalRegion handle, bool task_local)
    //--------------------------------------------------------------------------
    {
      // Already hold the lock from the caller
      RegionRequirement new_req(handle, LEGION_READ_WRITE, 
                                LEGION_EXCLUSIVE, handle);
      if (runtime->legion_spy_enabled)
        TaskOp::log_requirement(get_unique_id(), next_created_index, new_req);
      // Put a region requirement with no fields in the list of
      // created requirements, we know we can add any fields for
      // this field space in the future since we own all privileges
      created_requirements[next_created_index] = new_req;
      // Created regions always return privileges that they make
      returnable_privileges[next_created_index++] = !task_local;
    }

    //--------------------------------------------------------------------------
    void TaskContext::log_created_requirements(void)
    //--------------------------------------------------------------------------
    {
      std::vector<MappingInstance> instances(1, 
            Mapping::PhysicalInstance::get_virtual_instance());
      const UniqueID unique_op_id = get_unique_id();
      for (std::map<unsigned,RegionRequirement>::const_iterator it = 
           created_requirements.begin(); it != created_requirements.end(); it++)
      {
        // We already logged the requirement when we made it
        // Skip it if there are no privilege fields
        if (it->second.privilege_fields.empty())
          continue;
        InstanceSet instance_set;
        std::vector<PhysicalManager*> unacquired;  
        RegionTreeID bad_tree; std::vector<FieldID> missing_fields;
        runtime->forest->physical_convert_mapping(owner_task, 
            it->second, instances, instance_set, bad_tree, 
            missing_fields, NULL, unacquired, false/*do acquire_checks*/);
        runtime->forest->log_mapping_decision(unique_op_id, this,
            it->first, it->second, instance_set);
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
      assert(local_regions.find(handle) == local_regions.end());
      assert(created_regions.find(handle) == created_regions.end());
#endif
      if (task_local)
      {
        if (is_leaf_context())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_REGION_CREATION,
              "Illegal task-local region creation performed in leaf task %s "
                           "(ID %lld)", get_task_name(), get_unique_id())
        local_regions[handle] = false/*not deleted*/;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(created_regions.find(handle) == created_regions.end());
#endif
        created_regions[handle] = 1;
      }
      add_created_region(handle, task_local);
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_creation(FieldSpace handle, 
                                              FieldID fid, bool local)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      std::pair<FieldSpace,FieldID> key(handle,fid);
#ifdef DEBUG_LEGION
      assert(local_fields.find(key) == local_fields.end());
      assert(created_fields.find(key) == created_fields.end());
#endif
      if (!local)
      {
#ifdef DEBUG_LEGION
        assert(created_fields.find(key) == created_fields.end());
#endif
        created_fields.insert(key);
      }
      else
        local_fields[key] = false/*deleted*/;
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_all_field_creations(FieldSpace handle,bool local,
                                             const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (local)
      {
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          std::pair<FieldSpace,FieldID> key(handle,fields[idx]);
#ifdef DEBUG_LEGION
          assert(local_fields.find(key) == local_fields.end());
#endif
          local_fields[key] = false/*deleted*/;
        }
      }
      else
      {
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          std::pair<FieldSpace,FieldID> key(handle,fields[idx]);
#ifdef DEBUG_LEGION
          assert(created_fields.find(key) == created_fields.end());
#endif
          created_fields.insert(key);
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::register_field_space_creation(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
#ifdef DEBUG_LEGION
      assert(created_field_spaces.find(space) == created_field_spaces.end());
#endif
      created_field_spaces[space] = 1;
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
      created_index_spaces[space] = 1;
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
      created_index_partitions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void TaskContext::report_leaks_and_duplicates(
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      if (!deleted_regions.empty())
      {
        for (std::vector<DeletedRegion>::const_iterator it = 
              deleted_regions.begin(); it != deleted_regions.end(); it++)
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
              "Duplicate deletions were performed for region (%x,%x,%x) "
              "in task tree rooted by %s (provenance %s)", 
              it->region.index_space.id, it->region.field_space.id, 
              it->region.tree_id, get_task_name(), (it->provenance != NULL) ?
              it->provenance->provenance.c_str() : "unknown")
        deleted_regions.clear();
      }
      if (!deleted_fields.empty())
      {
        for (std::vector<DeletedField>::const_iterator it =
              deleted_fields.begin(); it != deleted_fields.end(); it++)
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
              "Duplicate deletions were performed on field %d of "
              "field space %x in task tree rooted by %s (provenance %s)", 
              it->fid, it->space.id, get_task_name(), 
              (it->provenance != NULL) ? it->provenance->provenance.c_str() :
              "unknown")
        deleted_fields.clear();
      }
      if (!deleted_field_spaces.empty())
      {
        for (std::vector<DeletedFieldSpace>::const_iterator it = 
              deleted_field_spaces.begin(); it != 
              deleted_field_spaces.end(); it++)
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
              "Duplicate deletions were performed on field space %x "
              "in task tree rooted by %s (provenance %s)", it->space.id,
              get_task_name(), (it->provenance != NULL) ?
              it->provenance->provenance.c_str() : "unknown")
        deleted_field_spaces.clear();
      }
      if (!deleted_index_spaces.empty())
      {
        for (std::vector<DeletedIndexSpace>::const_iterator it =
              deleted_index_spaces.begin(); it != 
              deleted_index_spaces.end(); it++)
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
              "Duplicate deletions were performed on index space %x "
              "in task tree rooted by %s (provenance %s)", it->space.id,
              get_task_name(), (it->provenance != NULL) ?
              it->provenance->provenance.c_str() : "unknown")
        deleted_index_spaces.clear();
      }
      if (!deleted_index_partitions.empty())
      {
        for (std::vector<DeletedPartition>::const_iterator it =
              deleted_index_partitions.begin(); it !=
              deleted_index_partitions.end(); it++)
          REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
              "Duplicate deletions were performed on index partition %x "
              "in task tree rooted by %s (provenance %s)", it->partition.id,
              get_task_name(), (it->provenance != NULL) ?
              it->provenance->provenance.c_str() : "unknown")
        deleted_index_partitions.clear();
      }
      // Now we go through and delete anything that the user leaked
      if (!created_regions.empty())
      {
        for (std::map<LogicalRegion,unsigned>::const_iterator it = 
              created_regions.begin(); it != created_regions.end(); it++)
        {
          if (runtime->report_leaks)
            REPORT_LEGION_WARNING(LEGION_WARNING_LEAKED_RESOURCE,
                "Logical region (%x,%x,%x) was leaked out of task tree rooted "
                "by task %s", it->first.index_space.id, 
                it->first.field_space.id, it->first.tree_id, get_task_name())
          runtime->forest->destroy_logical_region(it->first, preconditions);
        }
        created_regions.clear();
      }
      if (!created_fields.empty())
      {
        std::map<FieldSpace,FieldAllocatorImpl*> leak_allocators;
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator 
              it = created_fields.begin(); it != created_fields.end(); it++)
        {
          if (runtime->report_leaks)
            REPORT_LEGION_WARNING(LEGION_WARNING_LEAKED_RESOURCE,
                "Field %d of field space %x was leaked out of task tree rooted "
                "by task %s", it->second, it->first.id, get_task_name())
          std::map<FieldSpace,FieldAllocatorImpl*>::const_iterator finder =
              leak_allocators.find(it->first);
          if (finder == leak_allocators.end())
          {
            FieldAllocatorImpl *allocator = create_field_allocator(it->first);
            allocator->add_reference();
            leak_allocators[it->first] = allocator;
            allocator->ready_event.wait();
          }
          else
            finder->second->ready_event.wait();
          runtime->forest->free_field(it->first, it->second, preconditions);
        }
        for (std::map<FieldSpace,FieldAllocatorImpl*>::const_iterator it =
              leak_allocators.begin(); it != leak_allocators.end(); it++)
          if (it->second->remove_reference())
            delete it->second;
        created_fields.clear();
      }
      if (!created_field_spaces.empty())
      {
        for (std::map<FieldSpace,unsigned>::const_iterator it = 
              created_field_spaces.begin(); it != 
              created_field_spaces.end(); it++)
        {
          if (runtime->report_leaks)
            REPORT_LEGION_WARNING(LEGION_WARNING_LEAKED_RESOURCE,
                "Field space %x was leaked out of task tree rooted by task %s",
                it->first.id, get_task_name())
          runtime->forest->destroy_field_space(it->first, preconditions);
        }
        created_field_spaces.clear();
      }
      if (!created_index_partitions.empty())
      {
        for (std::map<IndexPartition,unsigned>::const_iterator it =
              created_index_partitions.begin(); it != 
              created_index_partitions.end(); it++)
        {
          if (runtime->report_leaks)
            REPORT_LEGION_WARNING(LEGION_WARNING_LEAKED_RESOURCE,
                "Index partition %x was leaked out of task tree rooted by "
                "task %s", it->first.id, get_task_name())
          runtime->forest->destroy_index_partition(it->first, preconditions);
        }
        created_index_partitions.clear();
      }
      if (!created_index_spaces.empty())
      {
        for (std::map<IndexSpace,unsigned>::const_iterator it = 
              created_index_spaces.begin(); it !=
              created_index_spaces.end(); it++)
        {
          if (runtime->report_leaks)
            REPORT_LEGION_WARNING(LEGION_WARNING_LEAKED_RESOURCE,
                "Index space %x was leaked out of task tree rooted by task %s",
                it->first.id, get_task_name())
          runtime->forest->destroy_index_space(it->first, 
                  runtime->address_space, preconditions);
        }
        created_index_spaces.clear();
      } 
    }

    //--------------------------------------------------------------------------
    void TaskContext::analyze_destroy_fields(FieldSpace handle,
                                             const std::set<FieldID> &to_delete,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes,
                                    std::vector<FieldID> &global_to_free,
                                    std::vector<FieldID> &local_to_free,
                                    std::vector<unsigned> &local_field_indexes,
                                    std::vector<unsigned> &deletion_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
#endif
      {
        // We can't destroy any fields from our original regions because we
        // were not the ones that made them.
        AutoLock priv_lock(privilege_lock);
        // We can actually remove the fields from the data structure now 
        for (std::set<FieldID>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          const std::pair<FieldSpace,FieldID> key(handle, *it);
          std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
            created_fields.find(key);
          if (finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace,FieldID>,bool>::iterator 
              local_finder = local_fields.find(key);
#ifdef DEBUG_LEGION
            assert(local_finder != local_fields.end());
            assert(local_finder->second);
#endif
            local_fields.erase(local_finder);
            local_to_free.push_back(*it);
          }
          else
          {
            created_fields.erase(finder);
            global_to_free.push_back(*it);
          }
        }
        // Now figure out which region requirements can be destroyed
        for (std::map<unsigned,RegionRequirement>::iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); it++)
        {
          if (it->second.region.get_field_space() != handle)
            continue;
          std::set<FieldID> overlapping_fields;
          for (std::set<FieldID>::const_iterator fit = to_delete.begin();
                fit != to_delete.end(); fit++)
          {
            std::set<FieldID>::const_iterator finder = 
              it->second.privilege_fields.find(*fit);
            if (finder != it->second.privilege_fields.end())
              overlapping_fields.insert(*fit);
          }
          if (overlapping_fields.empty())
            continue;
          delete_reqs.resize(delete_reqs.size()+1);
          RegionRequirement &req = delete_reqs.back();
          req.region = it->second.region;
          req.parent = it->second.region;
          req.privilege = LEGION_READ_WRITE;
          req.prop = LEGION_EXCLUSIVE;
          req.privilege_fields = overlapping_fields;
          req.handle_type = LEGION_SINGULAR_PROJECTION;
          parent_req_indexes.push_back(it->first);
          std::map<unsigned,unsigned>::iterator deletion_finder =
            deletion_counts.find(it->first);
          if (deletion_finder != deletion_counts.end())
          {
            deletion_finder->second++;
            deletion_indexes.push_back(it->first);
          }
          // We need some extra logging for legion spy
          if (runtime->legion_spy_enabled)
          {
            LegionSpy::log_requirement_fields(get_unique_id(),
                                              it->first, overlapping_fields);
            std::vector<MappingInstance> instances(1, 
                          Mapping::PhysicalInstance::get_virtual_instance());
            InstanceSet instance_set;
            std::vector<PhysicalManager*> unacquired;  
            RegionTreeID bad_tree; std::vector<FieldID> missing_fields;
            runtime->forest->physical_convert_mapping(owner_task, 
                req, instances, instance_set, bad_tree, 
                missing_fields, NULL, unacquired, false/*do acquire_checks*/);
            runtime->forest->log_mapping_decision(get_unique_id(), this,
                it->first, req, instance_set);
          }
        }
      }
      if (!local_to_free.empty())
        analyze_free_local_fields(handle, local_to_free, local_field_indexes);
    }

    //--------------------------------------------------------------------------
    void TaskContext::analyze_free_local_fields(FieldSpace handle,
                                     const std::vector<FieldID> &local_to_free,
                                     std::vector<unsigned> &local_field_indexes)
    //--------------------------------------------------------------------------
    {
      // Should only be performed on derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void TaskContext::analyze_destroy_logical_region(LogicalRegion handle,
                                    std::vector<RegionRequirement> &delete_reqs,
                                    std::vector<unsigned> &parent_req_indexes,
                                    std::vector<bool> &returnable)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_leaf_context());
#endif
      // If we're deleting a field space then we can't be deleting any of the 
      // original requirements, only requirements that we created
      if (runtime->legion_spy_enabled)
      {
        // We need some extra logging for legion spy
        std::vector<MappingInstance> instances(1, 
              Mapping::PhysicalInstance::get_virtual_instance());
        const UniqueID unique_op_id = get_unique_id();
        AutoLock priv_lock(privilege_lock);
        for (std::map<unsigned,RegionRequirement>::iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); /*nothing*/)
        {
          // Has to match precisely
          if (handle.get_tree_id() == it->second.region.get_tree_id())
          {
#ifdef DEBUG_LEGION
            // Should be the same region
            assert(handle == it->second.region);
            assert(returnable_privileges.find(it->first) !=
                    returnable_privileges.end());
#endif
            // Only need to record this if there are privilege fields
            if (!it->second.privilege_fields.empty())
            {
              // Do extra logging for legion spy
              InstanceSet instance_set;
              std::vector<PhysicalManager*> unacquired;  
              RegionTreeID bad_tree; std::vector<FieldID> missing_fields;
              runtime->forest->physical_convert_mapping(owner_task, 
                  it->second, instances, instance_set, bad_tree, 
                  missing_fields, NULL, unacquired, false/*do acquire_checks*/);
              runtime->forest->log_mapping_decision(unique_op_id, this,
                  it->first, it->second, instance_set);
              // Then do the result of the normal operations
              delete_reqs.resize(delete_reqs.size()+1);
              RegionRequirement &req = delete_reqs.back();
              req.region = it->second.region;
              req.parent = it->second.region;
              req.privilege = LEGION_READ_WRITE;
              req.prop = LEGION_EXCLUSIVE;
              req.privilege_fields = it->second.privilege_fields;
              req.handle_type = LEGION_SINGULAR_PROJECTION;
              req.flags = it->second.flags;
              parent_req_indexes.push_back(it->first);
              returnable.push_back(returnable_privileges[it->first]);
              // Always put a deletion index on here to mark that 
              // the requirement is going to be deleted
              std::map<unsigned,unsigned>::iterator deletion_finder =
                deletion_counts.find(it->first);
              if (deletion_finder == deletion_counts.end())
                deletion_counts[it->first] = 1;
              else
                deletion_finder->second++;
              // Can't delete this yet since other deletions might 
              // need to find it until it is finally applied
              it++;
            }
            else // Can erase the returnable privileges now
            {
              returnable_privileges.erase(it->first);
              // Remove the requirement from the created set 
              std::map<unsigned,RegionRequirement>::iterator to_delete = it++;
              created_requirements.erase(to_delete);
            }
          }
          else
            it++;
        }
        // Remove the region from the created set
        {
          std::map<LogicalRegion,unsigned>::iterator finder = 
            created_regions.find(handle);
          if (finder == created_regions.end())
          {
            std::map<LogicalRegion,bool>::iterator local_finder = 
              local_regions.find(handle);
#ifdef DEBUG_LEGION
            assert(local_finder != local_regions.end());
            assert(local_finder->second);
#endif
            local_regions.erase(local_finder);
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(finder->second == 0);
#endif
            created_regions.erase(finder);
          }
        }
        // Check to see if we have any latent field spaces to clean up
        if (!latent_field_spaces.empty())
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder =
            latent_field_spaces.find(handle.get_field_space());
          if (finder != latent_field_spaces.end())
          {
            std::set<LogicalRegion>::iterator region_finder = 
              finder->second.find(handle);
#ifdef DEBUG_LEGION
            assert(region_finder != finder->second.end());
#endif
            finder->second.erase(region_finder);
            if (finder->second.empty())
            {
              // Now that all the regions using this field space have
              // been deleted we can clean up all the created_fields
              for (std::set<std::pair<FieldSpace,FieldID> >::iterator it =
                    created_fields.begin(); it != 
                    created_fields.end(); /*nothing*/)
              {
                if (it->first == finder->first)
                {
                  std::set<std::pair<FieldSpace,FieldID> >::iterator 
                    to_delete = it++;
                  created_fields.erase(to_delete);
                }
                else
                  it++;
              }
              latent_field_spaces.erase(finder);
            }
          }
        }
      }
      else
      {
        AutoLock priv_lock(privilege_lock);
        for (std::map<unsigned,RegionRequirement>::iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); /*nothing*/)
        {
          // Has to match precisely
          if (handle.get_tree_id() == it->second.region.get_tree_id())
          {
#ifdef DEBUG_LEGION
            // Should be the same region
            assert(handle == it->second.region);
            assert(returnable_privileges.find(it->first) !=
                    returnable_privileges.end());
#endif
            // Only need to record this if there are privilege fields
            if (!it->second.privilege_fields.empty())
            {
              delete_reqs.resize(delete_reqs.size()+1);
              RegionRequirement &req = delete_reqs.back();
              req.region = it->second.region;
              req.parent = it->second.region;
              req.privilege = LEGION_READ_WRITE;
              req.prop = LEGION_EXCLUSIVE;
              req.privilege_fields = it->second.privilege_fields;
              req.handle_type = LEGION_SINGULAR_PROJECTION;
              parent_req_indexes.push_back(it->first);
              returnable.push_back(returnable_privileges[it->first]);
              // Always put a deletion index on here to mark that 
              // the requirement is going to be deleted
              std::map<unsigned,unsigned>::iterator deletion_finder =
                deletion_counts.find(it->first);
              if (deletion_finder == deletion_counts.end())
                deletion_counts[it->first] = 1;
              else
                deletion_finder->second++;
              // Can't delete this yet until it's actually performed
              // because other deletions might need to depend on it
              it++;
            }
            else // Can erase the returnable privileges now
            {
              returnable_privileges.erase(it->first);
              // Remove the requirement from the created set 
              std::map<unsigned,RegionRequirement>::iterator to_delete = it++;
              created_requirements.erase(to_delete);
            }
          }
          else
            it++;
        }
        // Remove the region from the created set
        {
          std::map<LogicalRegion,unsigned>::iterator finder = 
            created_regions.find(handle);
          if (finder == created_regions.end())
          {
            std::map<LogicalRegion,bool>::iterator local_finder = 
              local_regions.find(handle);
#ifdef DEBUG_LEGION
            assert(local_finder != local_regions.end());
            assert(local_finder->second);
#endif
            local_regions.erase(local_finder);
          }
          else
          {
#ifdef DEBUG_LEGION
            assert(finder->second == 0);
#endif
            created_regions.erase(finder);
          }
        }
        // Check to see if we have any latent field spaces to clean up
        if (!latent_field_spaces.empty())
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder =
            latent_field_spaces.find(handle.get_field_space());
          if (finder != latent_field_spaces.end())
          {
            std::set<LogicalRegion>::iterator region_finder = 
              finder->second.find(handle);
#ifdef DEBUG_LEGION
            assert(region_finder != finder->second.end());
#endif
            finder->second.erase(region_finder);
            if (finder->second.empty())
            {
              // Now that all the regions using this field space have
              // been deleted we can clean up all the created_fields
              for (std::set<std::pair<FieldSpace,FieldID> >::iterator it =
                    created_fields.begin(); it != 
                    created_fields.end(); /*nothing*/)
              {
                if (it->first == finder->first)
                {
                  std::set<std::pair<FieldSpace,FieldID> >::iterator 
                    to_delete = it++;
                  created_fields.erase(to_delete);
                }
                else
                  it++;
              }
              latent_field_spaces.erase(finder);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::remove_deleted_requirements(
                                          const std::vector<unsigned> &indexes,
                                          std::vector<LogicalRegion> &to_delete)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      for (std::vector<unsigned>::const_iterator it = 
            indexes.begin(); it != indexes.end(); it++) 
      {
        std::map<unsigned,unsigned>::iterator finder = 
          deletion_counts.find(*it);
#ifdef DEBUG_LEGION
        assert(finder != deletion_counts.end());
        assert(finder->second > 0);
#endif
        finder->second--;
        // Check to see if we're the last deletion with this region requirement
        if (finder->second > 0)
          continue;
        deletion_counts.erase(finder); 
        std::map<unsigned,RegionRequirement>::iterator req_finder = 
          created_requirements.find(*it);
#ifdef DEBUG_LEGION
        assert(req_finder != created_requirements.end());
        assert(returnable_privileges.find(*it) != returnable_privileges.end());
#endif
        to_delete.push_back(req_finder->second.parent);
        created_requirements.erase(req_finder);
        returnable_privileges.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::remove_deleted_fields(const std::set<FieldID> &to_free,
                                           const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      for (std::vector<unsigned>::const_iterator it = 
            indexes.begin(); it != indexes.end(); it++) 
      {
        std::map<unsigned,RegionRequirement>::iterator req_finder = 
          created_requirements.find(*it);
#ifdef DEBUG_LEGION
        assert(req_finder != created_requirements.end());
#endif
        std::set<FieldID> &priv_fields = req_finder->second.privilege_fields;
        if (priv_fields.empty())
          continue;
        for (std::set<FieldID>::const_iterator fit = 
              to_free.begin(); fit != to_free.end(); fit++)
          priv_fields.erase(*fit);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::remove_deleted_local_fields(FieldSpace space,
                                          const std::vector<FieldID> &to_remove)
    //--------------------------------------------------------------------------
    {
      // Should only be implemented by derived classes
      assert(false);
    } 

    //--------------------------------------------------------------------------
    void TaskContext::raise_poison_exception(void)
    //--------------------------------------------------------------------------
    {
      // TODO: handle poisoned task
      assert(false);
    }

    //--------------------------------------------------------------------------
    void TaskContext::raise_region_exception(PhysicalRegion region,bool nuclear)
    //--------------------------------------------------------------------------
    {
      // TODO: handle region exception
      assert(false);
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
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        std::map<unsigned,RegionRequirement>::const_iterator finder = 
          created_requirements.find(idx);
#ifdef DEBUG_LEGION
        assert(finder != created_requirements.end());
#endif
        target = finder->second;
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
      for (std::map<unsigned,RegionRequirement>::iterator it = 
           created_requirements.begin(); it != created_requirements.end(); it++)
      {
        RegionRequirement &our_req = it->second;
        // First check that the regions match
        if (our_req.region != req.parent)
          continue;
        // Next check the privileges
        if (check_privilege && 
            ((PRIV_ONLY(req) & our_req.privilege) != PRIV_ONLY(req)))
          continue;
#ifdef DEBUG_LEGION
        assert(returnable_privileges.find(it->first) != 
                returnable_privileges.end());
#endif
        // If this is a returnable privilege requiremnt that means
        // that we made this region so we always have privileges
        // on any fields for that region, just add them and be done
        if (returnable_privileges[it->first])
        {
          our_req.privilege_fields.insert(req.privilege_fields.begin(),
                                          req.privilege_fields.end());
          return it->first;
        }
        // Finally check that all the fields are contained
        bool dominated = true;
        for (std::set<FieldID>::const_iterator fit = 
              req.privilege_fields.begin(); fit !=
              req.privilege_fields.end(); fit++)
        {
          if (our_req.privilege_fields.find(*fit) ==
              our_req.privilege_fields.end())
          {
            // Check to see if this is a field we made
            // and haven't destroyed yet
            std::pair<FieldSpace,FieldID> key(fs, *fit);
            if (created_fields.find(key) != created_fields.end())
            {
              // We made it so we can add it to the requirement
              // and continue on our way
              our_req.privilege_fields.insert(*fit);
              continue;
            }
            if (local_fields.find(key) != local_fields.end())
            {
              // We made it so we can add it to the requirement
              // and continue on our way
              our_req.privilege_fields.insert(*fit);
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
        return it->first;
      }
      // Method of last resort, check to see if we made all the fields
      // if we did, then we can make a new requirement for all the fields
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        std::pair<FieldSpace,FieldID> key(fs, *it);
        // Didn't make it so we don't have privileges anywhere
        if ((created_fields.find(key) == created_fields.end()) &&
            (local_fields.find(key) == local_fields.end()))
          return -1;
      }
      // If we get here then we can make a new requirement
      // which has non-returnable privileges
      // Get the top level region for the region tree
      RegionNode *top = runtime->forest->get_tree(req.parent.get_tree_id());
      const unsigned index = next_created_index++;
      RegionRequirement &new_req = created_requirements[index];
      new_req = RegionRequirement(top->handle, LEGION_READ_WRITE, 
                                  LEGION_EXCLUSIVE, top->handle);
      if (runtime->legion_spy_enabled)
        TaskOp::log_requirement(get_unique_id(), index, new_req);
      // Add our fields
      new_req.privilege_fields.insert(
          req.privilege_fields.begin(), req.privilege_fields.end());
      // This is not a returnable privilege requirement
      returnable_privileges[index] = false;
      return index;
    }

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege(
                                         const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_PRIVILEGE_CALL);
      if (req.verified)
        return LEGION_NO_ERROR;
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
          return LEGION_NO_ERROR;
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
        return LEGION_NO_ERROR;
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
      if (req.flags & LEGION_VERIFIED_FLAG)
        return LEGION_NO_ERROR;
      // Copy privilege fields for check
      std::set<FieldID> privilege_fields(req.privilege_fields);
      // Try our original region requirements first
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        LegionErrorType et = 
          check_privilege_internal(req, regions[idx], privilege_fields, 
                                   bad_field, idx, bad_index, skip_privilege);
        // No error so we are done
        if (et == LEGION_NO_ERROR)
          return et;
        // Something other than bad parent region is a real error
        if (et != ERROR_BAD_PARENT_REGION)
          return et;
        // Otherwise we just keep going
      }
      // If none of that worked, we now get to try the created requirements
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
      for (std::map<unsigned,RegionRequirement>::const_iterator it = 
            created_requirements.begin(); it != 
            created_requirements.end(); it++)
      {
        const RegionRequirement &created_req = it->second;
        LegionErrorType et = 
          check_privilege_internal(req, created_req, privilege_fields, 
                      bad_field, it->first, bad_index, skip_privilege);
        // No error so we are done
        if (et == LEGION_NO_ERROR)
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
            return LEGION_NO_ERROR;
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
        if ((created_fields.find(key) == created_fields.end()) &&
            (local_fields.find(key) == local_fields.end()))
          return ERROR_BAD_PARENT_REGION;
      }
      // Check that the parent is the root of the tree, if not it is bad
      RegionNode *parent_region = runtime->forest->get_node(req.parent);
      if (parent_region->parent != NULL)
        return ERROR_BAD_PARENT_REGION;
      // Otherwise we have privileges on these fields for all regions
      // so we are good on privileges
      return LEGION_NO_ERROR;
    } 

    //--------------------------------------------------------------------------
    LegionErrorType TaskContext::check_privilege_internal(
        const RegionRequirement &req, const RegionRequirement &our_req,
        std::set<FieldID>& privilege_fields, FieldID &bad_field, 
        int local_index, int &bad_index, bool skip_privilege) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
      // Check to see if we found the requirement in the parent
      if (our_req.region == req.parent)
      {
        // If we make it in here then we know we have at least found
        // the parent name so we can set the bad index
        bad_index = local_index;
        bad_field = LEGION_AUTO_GENERATE_ID; // set it to an invalid field
        if ((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
            (req.handle_type == LEGION_REGION_PROJECTION))
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
        for (std::set<FieldID>::iterator fit = privilege_fields.begin();
              fit != privilege_fields.end(); /*nothing*/)
        {
          if (our_req.privilege_fields.find(*fit) != 
              our_req.privilege_fields.end())
          {
            // Only need to do this check if there were overlapping fields
            if (!skip_privilege && (PRIV_ONLY(req) & (~(our_req.privilege))))
            {
              if ((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
                  (req.handle_type == LEGION_REGION_PROJECTION))
                return ERROR_BAD_REGION_PRIVILEGES;
              else
                return ERROR_BAD_PARTITION_PRIVILEGES;
            }
            std::set<FieldID>::iterator to_delete = fit++;
            privilege_fields.erase(to_delete);
          }
          else
            fit++;
        }
      }

      if (!privilege_fields.empty()) 
      {
        bad_field = *(privilege_fields.begin());
        return ERROR_BAD_PARENT_REGION;
      }
        // If we make it here then we are good
      return LEGION_NO_ERROR;
    }

    //--------------------------------------------------------------------------
    bool TaskContext::check_region_dependence(RegionTreeID our_tid,
                                              IndexSpace our_space,
                                              const RegionRequirement &our_req,
                                              const RegionUsage &our_usage,
                                              const RegionRequirement &req,
                                              bool check_privileges) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CHECK_REGION_DEPENDENCE_CALL);
      if ((req.handle_type == LEGION_SINGULAR_PROJECTION) || 
          (req.handle_type == LEGION_REGION_PROJECTION))
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
      // If we aren't supposed to check privileges then we're done
      if (!check_privileges)
        return true;
      // Finally if everything has overlapped, do a dependence analysis
      // on the privileges and coherence
      RegionUsage usage(req);
      switch (check_dependence_type<true>(our_usage,usage))
      {
        // Only allow no-dependence, or simultaneous dependence through
        case LEGION_NO_DEPENDENCE:
        case LEGION_SIMULTANEOUS_DEPENDENCE:
          {
            return false;
          }
        default:
          break;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    LogicalRegion TaskContext::find_logical_region(unsigned index)
    //--------------------------------------------------------------------------
    {
      if (index < regions.size())
        return regions[index].region;
      AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
      std::map<unsigned,RegionRequirement>::const_iterator finder = 
        created_requirements.find(index);
#ifdef DEBUG_LEGION
      assert(finder != created_requirements.end());
#endif
      return finder->second.region;
    } 

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& TaskContext::begin_task(
                                                           Legion::Runtime *&rt)
    //--------------------------------------------------------------------------
    {
      implicit_context = this;
      implicit_runtime = this->runtime;
      rt = this->runtime->external;
      implicit_provenance = owner_task->get_unique_op_id();
      if (overhead_tracker != NULL)
        previous_profiling_time = Realm::Clock::current_time_in_nanoseconds();
      // Switch over the executing processor to the one
      // that has actually been assigned to run this task.
      executing_processor = Processor::get_executing_processor();
      owner_task->current_proc = executing_processor;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_task_processor(get_unique_id(), executing_processor.id);
#ifdef DEBUG_LEGION
      log_task.debug("Task %s (ID %lld) starting on processor " IDFMT "",
                    get_task_name(), get_unique_id(), executing_processor.id);
      assert(regions.size() == physical_regions.size());
#endif
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      if (!inline_task)
        find_parent_context()->decrement_pending(owner_task);
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance TaskContext::create_task_local_instance(Memory memory, 
                                           Realm::InstanceLayoutGeneric *layout)
    //--------------------------------------------------------------------------
    {
      PhysicalInstance instance;
      Realm::ProfilingRequestSet no_requests;
#ifdef LEGION_MALLOC_INSTANCES
      uintptr_t ptr = runtime->allocate_deferred_instance(memory, 
                              layout->bytes_used, false/*free*/); 
      const RtEvent wait_on(Realm::RegionInstance::create_external(instance,
                                          memory, ptr, layout, no_requests));
      task_local_instances.push_back(std::make_pair(instance, ptr));
#else
      const RtEvent wait_on(Realm::RegionInstance::create_instance(instance, 
                                              memory, layout, no_requests));
      if (!instance.exists())
      {
        const char *mem_names[] = {
#define MEM_NAMES(name, desc) desc,
          REALM_MEMORY_KINDS(MEM_NAMES) 
#undef MEM_NAMES
        };
        REPORT_LEGION_ERROR(ERROR_DEFERRED_ALLOCATION_FAILURE,
            "Failed to allocate DeferredBuffer/Value/Reductionin task %s "
            "(UID %lld) because %s memory " IDFMT " is full.", get_task_name(),
            get_unique_id(), mem_names[memory.kind()], memory.id) 
      }
      task_local_instances.push_back(instance);
#endif
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      return instance;
    }

    //--------------------------------------------------------------------------
    void TaskContext::begin_misspeculation(void)
    //--------------------------------------------------------------------------
    {
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      owner_task->get_context()->decrement_pending(owner_task);
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
      post_end_task(result, result_size, false/*owner*/, NULL/*functor*/);
#ifdef DEBUG_LEGION
      runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                     false/*meta*/);
#else
      runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    Lock TaskContext::create_lock(void)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      return Lock(Reservation::create_reservation());
    }

    //--------------------------------------------------------------------------
    PhaseBarrier TaskContext::create_phase_barrier(unsigned arrivals)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      return PhaseBarrier(ApBarrier(Realm::Barrier::create_barrier(arrivals)));
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
    void TaskContext::remap_unmapped_regions(LegionTrace *trace,
                            const std::vector<PhysicalRegion> &unmapped_regions,
                            const char *provenance)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unmapped_regions.empty());
#endif
      if (trace != NULL)
      {
        if (trace->is_static_trace())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_RUNTIME_REMAPPING,
                      "Illegal runtime remapping in static trace inside of "
                      "task %s (UID %lld). Static traces must perfectly "
                      "manage their physical mappings with no runtime help.",
                      get_task_name(), get_unique_id())
        else
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_RUNTIME_REMAPPING,
                      "Illegal runtime remapping in dynamic trace %d inside of "
                      "task %s (UID %lld). Dynamic traces must perfectly "
                      "manage their physical mappings with no runtime help.",
                      trace->tid, get_task_name(), get_unique_id())
      }
      std::set<ApEvent> mapped_events;
      for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
      {
        const ApEvent ready = remap_region(unmapped_regions[idx], provenance);
        if (ready.exists())
          mapped_events.insert(ready);
      }
      // Wait for all the re-mapping operations to complete
      const ApEvent mapped_event = Runtime::merge_events(NULL, mapped_events);
      bool poisoned = false;
      if (mapped_event.has_triggered_faultaware(poisoned))
      {
        if (poisoned)
          raise_poison_exception();
        return;
      }
      begin_task_wait(true/*from runtime*/);
      mapped_event.wait_faultaware(poisoned);
      if (poisoned)
        raise_poison_exception();
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

    //--------------------------------------------------------------------------
    void TaskContext::yield(void)
    //--------------------------------------------------------------------------
    {
      YieldArgs args(owner_task->get_unique_id());
      // Run this task with minimum priority to allow other things to run
      const RtEvent wait_for = 
        runtime->issue_runtime_meta_task(args, LG_MIN_PRIORITY);
      begin_task_wait(false/*from runtime*/);
      wait_for.wait();
      end_task_wait();
    }

    //--------------------------------------------------------------------------
    void TaskContext::release_task_local_instances(PhysicalInstance return_inst)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < task_local_instances.size(); idx++)
      {
#ifdef LEGION_MALLOC_INSTANCES
        std::pair<PhysicalInstance,uintptr_t> inst = task_local_instances[idx];
        if (inst.first == return_inst)
          task_local_instances.front() = inst; // save this for clean up
        else
          inst.first.destroy(Processor::get_current_finish_event());
#else
        PhysicalInstance inst = task_local_instances[idx];
        // Don't delete the return inst for now, the cleanup code will do that
        if (inst != return_inst)
          inst.destroy(Processor::get_current_finish_event());
#endif
      }
#ifdef LEGION_MALLOC_INSTANCES
      if (return_inst.exists())
      {
        task_local_instances.resize(1);
#ifdef DEBUG_LEGION
        assert(task_local_instances.front().first == return_inst);
#endif
      }
      else
#endif
      task_local_instances.clear();
    }

#ifdef LEGION_MALLOC_INSTANCES
    //--------------------------------------------------------------------------
    void TaskContext::release_future_local_instance(PhysicalInstance instance)
    //--------------------------------------------------------------------------
    {
      // Get the pointer and free it
      MemoryManager *manager = 
        runtime->find_memory_manager(instance.get_location());
#ifdef DEBUG_LEGION
      assert(task_local_instances.size() == 1);
#endif
      const std::pair<PhysicalInstance,uintptr_t> &inst = 
        task_local_instances.back();
#ifdef DEBUG
      assert(inst.first == instance);
#endif
      manager->free_legion_instance(RtEvent::NO_RT_EVENT, inst.second);
    }
#endif

    //--------------------------------------------------------------------------
    Future TaskContext::predicate_task_false(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (launcher.predicate_false_future.impl != NULL)
        return launcher.predicate_false_future;
      // Otherwise check to see if we have a value
      FutureImpl *result = new FutureImpl(this, runtime, true/*register*/,
        runtime->get_available_distributed_id(), 
        runtime->address_space, ApEvent::NO_AP_EVENT);
      if (launcher.predicate_false_result.get_size() > 0)
        result->set_result(launcher.predicate_false_result.get_ptr(),
                           launcher.predicate_false_result.get_size(),
                           false/*own*/);
      else
        result->set_result(NULL, 0, false/*own*/);
      return Future(result);
    }

    //--------------------------------------------------------------------------
    FutureMap TaskContext::predicate_index_task_false(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      FutureMapImpl *result = new FutureMapImpl(this, runtime,
          runtime->get_available_distributed_id(),
          runtime->address_space, RtEvent::NO_RT_EVENT);
      if (launcher.predicate_false_future.impl != NULL)
      {
        ApEvent ready_event = 
          launcher.predicate_false_future.impl->get_ready_event(); 
        if (ready_event.has_triggered_faultignorant())
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
        // Just initialize all the futures
        for (Domain::DomainPointIterator itr(launcher.launch_domain); 
              itr; itr++)
        {
          Future f = result->get_future(itr.p);
          f.impl->set_result(NULL, 0, false/*own*/);
        }
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
      return FutureMap(result);
    }

    //--------------------------------------------------------------------------
    Future TaskContext::predicate_index_task_reduce_false(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (launcher.predicate_false_future.impl != NULL)
        return launcher.predicate_false_future;
      // Otherwise check to see if we have a value
      FutureImpl *result = new FutureImpl(this, runtime, true/*register*/, 
        runtime->get_available_distributed_id(), 
        runtime->address_space, ApEvent::NO_AP_EVENT);
      if (launcher.predicate_false_result.get_size() > 0)
        result->set_result(launcher.predicate_false_result.get_ptr(),
                           launcher.predicate_false_result.get_size(),
                           false/*own*/);
      else
        result->set_result(NULL, 0, false/*own*/);
      return Future(result);
    }

    //--------------------------------------------------------------------------
    IndexSpace TaskContext::find_index_launch_space(const Domain &domain,
                                                    const std::string &prov)
    //--------------------------------------------------------------------------
    {
      std::map<Domain,IndexSpace>::const_iterator finder =
        index_launch_spaces.find(domain);
      if (finder != index_launch_spaces.end())
        return finder->second;
      IndexSpace result;
      switch (domain.get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            result = TaskContext::create_index_space(domain, \
              NT_TemplateHelper::encode_tag<DIM,coord_t>(), \
              (prov.length() > 0) ? prov.c_str() : NULL); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      index_launch_spaces[domain] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    VariantImpl* TaskContext::select_inline_variant(TaskOp *child,
                              const std::vector<PhysicalRegion> &parent_regions,
                              std::deque<InstanceSet> &physical_instances)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SELECT_INLINE_VARIANT_CALL);
      Mapper::SelectVariantInput input;
      Mapper::SelectVariantOutput output;
      input.processor = executing_processor;
      input.chosen_instances.resize(parent_regions.size());
      // Extract the specific field instances for each region requirement
      for (unsigned idx1 = 0; idx1 < parent_regions.size(); idx1++)
      {
        const RegionRequirement &child_req = child->regions[idx1];
        FieldSpaceNode *space = 
          runtime->forest->get_node(child_req.parent.get_field_space());
        FieldMask mask = space->get_field_mask(child_req.privilege_fields);
        InstanceSet instances;
        parent_regions[idx1].impl->get_references(instances);
        for (unsigned idx2 = 0; idx2 < instances.size(); idx2++)
        {
          const InstanceRef &ref = instances[idx2];
          const FieldMask overlap = mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          physical_instances[idx1].add_instance(
              InstanceRef(ref.get_manager(), overlap, ref.get_ready_event()));
          input.chosen_instances[idx1].push_back(
              MappingInstance(ref.get_manager()));
          mask -= overlap;
          if (!mask)
            break;
        }
#ifdef DEBUG_LEGION
        assert(!mask);
#endif
      }
      output.chosen_variant = 0;
      // Always do this with the child mapper
      MapperManager *child_mapper = 
        runtime->find_mapper(executing_processor, child->map_id);
      child_mapper->invoke_select_task_variant(child, &input, &output);
      VariantImpl *variant_impl = runtime->find_variant_impl(child->task_id,
                                   output.chosen_variant, true/*can fail*/);
      if (variant_impl == NULL)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invoction of "
                      "'select_task_variant' on mapper %s. Mapper selected "
                      "an invalid variant ID %d for inlining of task %s "
                      "(UID %lld).", child_mapper->get_mapper_name(),
                      output.chosen_variant, child->get_task_name(), 
                      child->get_unique_id())
      if (!runtime->unsafe_mapper)
        child->validate_variant_selection(child_mapper, variant_impl,
         executing_processor.kind(), physical_instances, "select_task_variant");
      return variant_impl;
    }

    /////////////////////////////////////////////////////////////
    // Inner Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InnerContext::InnerContext(Runtime *rt, SingleTask *owner,int d,bool finner,
                               const std::vector<RegionRequirement> &reqs,
                               const std::vector<unsigned> &parent_indexes,
                               const std::vector<bool> &virt_mapped,
                               UniqueID uid, ApEvent exec_fence, bool remote,
                               bool inline_task, bool implicit_task)
      : TaskContext(rt, owner, d, reqs, inline_task, implicit_task),
        tree_context(rt->allocate_region_tree_context()), context_uid(uid), 
        remote_context(remote), full_inner_context(finner),
        finished_execution(false), parent_req_indexes(parent_indexes),
        virtual_mapped(virt_mapped), total_children_count(0),
        total_close_count(0), total_summary_count(0),
        outstanding_children_count(0), outstanding_prepipeline(0),
        outstanding_dependence(false),
        ready_comp_queue(CompletionQueue::NO_QUEUE),
        enqueue_task_comp_queue(CompletionQueue::NO_QUEUE),
        distribute_task_comp_queue(CompletionQueue::NO_QUEUE),
        launch_task_comp_queue(CompletionQueue::NO_QUEUE),
        resolution_comp_queue(CompletionQueue::NO_QUEUE),
        trigger_execution_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_execution_comp_queue(CompletionQueue::NO_QUEUE),
        trigger_completion_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_completion_comp_queue(CompletionQueue::NO_QUEUE),
        trigger_commit_comp_queue(CompletionQueue::NO_QUEUE),
        deferred_commit_comp_queue(CompletionQueue::NO_QUEUE),
        post_task_comp_queue(CompletionQueue::NO_QUEUE), 
        current_trace(NULL), previous_trace(NULL),
        physical_trace_replay_status(0), valid_wait_event(false), 
        outstanding_subtasks(0), pending_subtasks(0), pending_frames(0),
        currently_active_context(false), current_mapping_fence(NULL),
        mapping_fence_gen(0), current_mapping_fence_index(0), 
        current_execution_fence_event(exec_fence),
        current_execution_fence_index(0), last_implicit(NULL),
        last_implicit_gen(0)
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
      context_configuration.max_templates_per_trace =
        LEGION_DEFAULT_MAX_TEMPLATES_PER_TRACE;
      context_configuration.mutable_priority = false;
#ifdef DEBUG_LEGION
      assert(tree_context.exists());
      runtime->forest->check_context_state(tree_context);
#endif
      // If we have an owner, clone our local fields from its context
      // and also compute the coordinates for this context in the task tree
      if (owner != NULL)
      {
        TaskContext *owner_ctx = owner_task->get_context();
#ifdef DEBUG_LEGION
        InnerContext *parent_ctx = dynamic_cast<InnerContext*>(owner_ctx);
        assert(parent_ctx != NULL);
#else
        InnerContext *parent_ctx = static_cast<InnerContext*>(owner_ctx);
#endif
        parent_ctx->clone_local_fields(local_field_infos);
        // Get the coordinates for the parent task
        parent_ctx->compute_task_tree_coordinates(context_coordinates);
        // Then add our coordinates for our task
        context_coordinates.push_back(std::make_pair(
              owner_task->get_context_index(), owner_task->index_point));
      }
      if (!remote_context)
        runtime->register_local_context(context_uid, this);
    }

    //--------------------------------------------------------------------------
    InnerContext::InnerContext(const InnerContext &rhs)
      : TaskContext(NULL, NULL, 0, rhs.regions, false), 
        tree_context(rhs.tree_context), context_uid(0), remote_context(false), 
        full_inner_context(false), parent_req_indexes(rhs.parent_req_indexes), 
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
      if (ready_comp_queue.exists())
        ready_comp_queue.destroy();
      if (enqueue_task_comp_queue.exists())
        enqueue_task_comp_queue.destroy();
      if (distribute_task_comp_queue.exists())
        distribute_task_comp_queue.destroy();
      if (launch_task_comp_queue.exists())
        launch_task_comp_queue.destroy();
      if (resolution_comp_queue.exists())
        resolution_comp_queue.destroy();
      if (trigger_execution_comp_queue.exists())
        trigger_execution_comp_queue.destroy();
      if (deferred_execution_comp_queue.exists())
        deferred_execution_comp_queue.destroy();
      if (trigger_completion_comp_queue.exists())
        trigger_completion_comp_queue.destroy();
      if (deferred_completion_comp_queue.exists())
        deferred_completion_comp_queue.destroy();
      if (trigger_commit_comp_queue.exists())
        trigger_commit_comp_queue.destroy();
      if (deferred_commit_comp_queue.exists())
        deferred_commit_comp_queue.destroy();
      if (post_task_comp_queue.exists())
        post_task_comp_queue.destroy();
      for (std::map<TraceID,LegionTrace*>::const_iterator it = 
            traces.begin(); it != traces.end(); it++)
        if (it->second->remove_reference())
          delete (it->second);
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
      if (!local_field_infos.empty())
        local_field_infos.clear();
      // Unregister ourselves from any tracking contexts that we might have
      if (!tree_equivalence_sets.empty())
      {
        for (std::map<RegionTreeID,EquivalenceSet*>::const_iterator it = 
              tree_equivalence_sets.begin(); it != 
              tree_equivalence_sets.end(); it++)
          if (it->second->remove_base_resource_ref(CONTEXT_REF))
            delete it->second;
        tree_equivalence_sets.clear();
      }
      if (!empty_equivalence_sets.empty())
      {
        for (std::map<std::pair<RegionTreeID,IndexSpaceExprID>,
                                EquivalenceSet*>::const_iterator it = 
              empty_equivalence_sets.begin(); it != 
              empty_equivalence_sets.end(); it++)
          if (it->second->remove_base_resource_ref(CONTEXT_REF))
            delete it->second;
        empty_equivalence_sets.clear();
      }
      if (!fill_view_cache.empty())
      {
        for (std::list<FillView*>::const_iterator it = 
              fill_view_cache.begin(); it != fill_view_cache.end(); it++)
          if ((*it)->remove_base_valid_ref(CONTEXT_REF))
            delete (*it);
        fill_view_cache.clear();
      }
      if (!attach_functions.empty())
      {
        for (std::map<IndexTreeNode*,
                std::vector<AttachProjectionFunctor*> >::const_iterator fit =
              attach_functions.begin(); fit != attach_functions.end(); fit++)
        {
          for (std::vector<AttachProjectionFunctor*>::const_iterator it =
                fit->second.begin(); it != fit->second.end(); it++)
          {
            // Unregister it with the runtime if it is not the identity
            // The runtime will delete the functor for us
            if ((*it)->pid > 0)
              runtime->unregister_projection_functor((*it)->pid);
            else // This is the identity so we can just delete it ourself
              delete (*it);
          }
        }
        attach_functions.clear();
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
    void InnerContext::receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regs,
              std::vector<DeletedRegion> &deleted_regs,
              std::set<std::pair<FieldSpace,FieldID> > &created_fids,
              std::vector<DeletedField> &deleted_fids,
              std::map<FieldSpace,unsigned> &created_fs,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_fs,
              std::vector<DeletedFieldSpace> &deleted_fs,
              std::map<IndexSpace,unsigned> &created_is,
              std::vector<DeletedIndexSpace> &deleted_is,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      bool need_deletion_dependences = true;
      ApEvent precondition;
      std::map<Operation*,GenerationID> dependences;
      if (!created_regs.empty())
        register_region_creations(created_regs);
      if (!deleted_regs.empty())
      {
        precondition = 
          compute_return_deletion_dependences(return_index, dependences);
        need_deletion_dependences = false;
        register_region_deletions(precondition, dependences, 
                                  deleted_regs, preconditions);
      }
      if (!created_fids.empty())
        register_field_creations(created_fids);
      if (!deleted_fids.empty())
      {
        if (need_deletion_dependences)
        {
          precondition = 
            compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_deletions(precondition, dependences, 
                                 deleted_fids, preconditions);
      }
      if (!created_fs.empty())
        register_field_space_creations(created_fs);
      if (!latent_fs.empty())
        register_latent_field_spaces(latent_fs);
      if (!deleted_fs.empty())
      {
        if (need_deletion_dependences)
        {
          precondition = 
            compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_space_deletions(precondition, dependences,
                                       deleted_fs, preconditions);
      }
      if (!created_is.empty())
        register_index_space_creations(created_is);
      if (!deleted_is.empty())
      {
        if (need_deletion_dependences)
        {
          precondition = 
            compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_space_deletions(precondition, dependences,
                                       deleted_is, preconditions);
      }
      if (!created_partitions.empty())
        register_index_partition_creations(created_partitions);
      if (!deleted_partitions.empty())
      {
        if (need_deletion_dependences)
        {
          precondition = 
            compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_partition_deletions(precondition, dependences,
                                           deleted_partitions, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_region_creations(
                                      std::map<LogicalRegion,unsigned> &regions)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!latent_field_spaces.empty())
      {
        for (std::map<LogicalRegion,unsigned>::const_iterator it = 
              regions.begin(); it != regions.end(); it++)
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder =
            latent_field_spaces.find(it->first.get_field_space());
          if (finder != latent_field_spaces.end())
            finder->second.insert(it->first);
        }
      }
      if (!created_regions.empty())
      {
        for (std::map<LogicalRegion,unsigned>::const_iterator it = 
              regions.begin(); it != regions.end(); it++)
        {
          std::map<LogicalRegion,unsigned>::iterator finder = 
            created_regions.find(it->first);
          if (finder == created_regions.end())
          {
            created_regions.insert(*it);
            add_created_region(it->first, false/*task local*/);
          }
          else
            finder->second += it->second;
        }
      }
      else
      {
        created_regions.swap(regions);
        for (std::map<LogicalRegion,unsigned>::const_iterator it = 
              created_regions.begin(); it != created_regions.end(); it++)
          add_created_region(it->first, false/*task local*/);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_region_deletions(ApEvent precondition,
                           const std::map<Operation*,GenerationID> &dependences,
                                            std::vector<DeletedRegion> &regions,
                                            std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedRegion> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::vector<DeletedRegion>::const_iterator rit =
              regions.begin(); rit != regions.end(); rit++)
        {
          std::map<LogicalRegion,unsigned>::iterator region_finder = 
            created_regions.find(rit->region);
          if (region_finder == created_regions.end())
          {
            if (local_regions.find(rit->region) != local_regions.end())
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
                  "Local logical region (%x,%x,%x) in task %s (UID %lld) was "
                  "not deleted by this task. Local regions can only be deleted "
                  "by the task that made them.", rit->region.index_space.id,
                  rit->region.field_space.id, rit->region.tree_id, 
                  get_task_name(), get_unique_id())
            // Deletion keeps going up
            deleted_regions.push_back(*rit);
          }
          else
          {
            // One of ours to delete
#ifdef DEBUG_LEGION
            assert(region_finder->second > 0);
#endif
            if (--region_finder->second == 0)
            {
              created_regions.erase(region_finder);
              // Check to see if we have any latent field spaces to clean up
              if (!latent_field_spaces.empty())
              {
                std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder =
                  latent_field_spaces.find(rit->region.get_field_space());
                if (finder != latent_field_spaces.end())
                {
                  std::set<LogicalRegion>::iterator latent_finder = 
                    finder->second.find(rit->region);
#ifdef DEBUG_LEGION
                  assert(latent_finder != finder->second.end());
#endif
                  finder->second.erase(latent_finder);
                  if (finder->second.empty())
                  {
                    // Now that all the regions using this field space have
                    // been deleted we can clean up all the created_fields
                    for (std::set<std::pair<FieldSpace,FieldID> >::iterator it =
                          created_fields.begin(); it != 
                          created_fields.end(); /*nothing*/)
                    {
                      if (it->first == finder->first)
                      {
                        std::set<std::pair<FieldSpace,FieldID> >::iterator 
                          to_delete = it++;
                        created_fields.erase(to_delete);
                      }
                      else
                        it++;
                    }
                    latent_field_spaces.erase(finder);
                  }
                }
              }
              delete_now.emplace_back(*rit);
            }
          }
        }
      }
      if (!delete_now.empty())
      {
        for (std::vector<DeletedRegion>::const_iterator it = 
              delete_now.begin(); it != delete_now.end(); it++)
        {
          DeletionOp *op = runtime->get_available_deletion_op();
          op->initialize_logical_region_deletion(this, it->region, 
              true/*unordered*/, it->provenance);
          op->set_deletion_preconditions(precondition, dependences);
          if (!add_to_dependence_queue(op, true/*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(
                Runtime::protect_event(op->get_completion_event()));
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_creations(
                               std::set<std::pair<FieldSpace,FieldID> > &fields)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_fields.empty())
      {
#ifdef DEBUG_LEGION
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
              fields.begin(); it != fields.end(); it++)
        {
          assert(created_fields.find(*it) == created_fields.end());
          created_fields.insert(*it);
        }
#else
        created_fields.insert(fields.begin(), fields.end());
#endif
      }
      else
        created_fields.swap(fields);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_deletions(ApEvent precondition,
                           const std::map<Operation*,GenerationID> &dependences,
                           std::vector<DeletedField> &fields,
                           std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      std::map<std::pair<FieldSpace,Provenance*>,std::set<FieldID> > delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::vector<DeletedField>::const_iterator fit =
              fields.begin(); fit != fields.end(); fit++)
        {
          const std::pair<FieldSpace,FieldID> key(fit->space, fit->fid);
          std::set<std::pair<FieldSpace,FieldID> >::const_iterator 
            field_finder = created_fields.find(key);
          if (field_finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace,FieldID>,bool>::iterator 
              local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
              REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
                  "Local field %d in field space %x in task %s (UID %lld) was "
                  "not deleted by this task. Local fields can only be deleted "
                  "by the task that made them.", fit->fid, fit->space.id,
                  get_task_name(), get_unique_id())
            deleted_fields.emplace_back(*fit);
          }
          else
          {
            // One of ours to delete
            const std::pair<FieldSpace,Provenance*> 
              now_key(fit->space, fit->provenance);
            delete_now[now_key].insert(fit->fid);
            created_fields.erase(field_finder);
          }
        }
      }
      if (!delete_now.empty())
      {
        for (std::map<std::pair<FieldSpace,Provenance*>,
                      std::set<FieldID> >::const_iterator it = 
              delete_now.begin(); it != delete_now.end(); it++)
        {
          DeletionOp *op = runtime->get_available_deletion_op();
          FieldAllocatorImpl *allocator = 
            create_field_allocator(it->first.first);
          op->initialize_field_deletions(this, it->first.first, it->second, 
                           true/*unordered*/, allocator, it->first.second);
          op->set_deletion_preconditions(precondition, dependences);
          if (!add_to_dependence_queue(op, true/*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(
                Runtime::protect_event(op->get_completion_event()));
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_space_creations(
                                          std::map<FieldSpace,unsigned> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!latent_field_spaces.empty())
      {
        // Remove any latent field spaces we have ownership for
        for (std::map<FieldSpace,unsigned>::const_iterator it =
              spaces.begin(); it != spaces.end(); it++)
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
            latent_field_spaces.find(it->first);
          if (finder != latent_field_spaces.end())
            latent_field_spaces.erase(finder);
        }
      }
      if (!created_field_spaces.empty())
      {
        for (std::map<FieldSpace,unsigned>::const_iterator it = 
              spaces.begin(); it != spaces.end(); it++)
        {
          std::map<FieldSpace,unsigned>::iterator finder = 
            created_field_spaces.find(it->first);
          if (finder == created_field_spaces.end())
            created_field_spaces.insert(*it);
          else
            finder->second += it->second;
        }
      }
      else
        created_field_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_latent_field_spaces(
                          std::map<FieldSpace,std::set<LogicalRegion> > &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(privilege_lock);
      if (!created_field_spaces.empty())
      {
        // Remote any latent field spaces we already have ownership on
        for (std::map<FieldSpace,std::set<LogicalRegion> >::iterator it =
              spaces.begin(); it != spaces.end(); /*nothing*/)
        {
          if (created_field_spaces.find(it->first) != 
                created_field_spaces.end())
          {
            std::map<FieldSpace,std::set<LogicalRegion> >::iterator 
              to_delete = it++;
            spaces.erase(to_delete);
          }
          else
            it++;
        }
        if (spaces.empty())
          return;
      }
      if (!created_regions.empty())
      {
        // See if any of these regions are copies of our latent spaces
        for (std::map<LogicalRegion,unsigned>::const_iterator it = 
              created_regions.begin(); it != created_regions.end(); it++)
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
            spaces.find(it->first.get_field_space());
          if (finder != spaces.end())
            finder->second.insert(it->first);
        }
      }
      // Now we can do the merge
      if (!latent_field_spaces.empty())
      {
        for (std::map<FieldSpace,std::set<LogicalRegion> >::const_iterator it =
              spaces.begin(); it != spaces.end(); it++)
        {
          std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
            latent_field_spaces.find(it->first);
          if (finder != latent_field_spaces.end())
            finder->second.insert(it->second.begin(), it->second.end());
          else
            latent_field_spaces.insert(*it);
        }
      }
      else
        latent_field_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_field_space_deletions(ApEvent precondition,
                           const std::map<Operation*,GenerationID> &dependences,
                                         std::vector<DeletedFieldSpace> &spaces,
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedFieldSpace> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::vector<DeletedFieldSpace>::const_iterator fit = 
              spaces.begin(); fit != spaces.end(); fit++)
        {
          std::map<FieldSpace,unsigned>::iterator finder = 
            created_field_spaces.find(fit->space);
          if (finder != created_field_spaces.end())
          {
#ifdef DEBUG_LEGION
            assert(finder->second > 0);
#endif
            if (--finder->second == 0)
            {
              delete_now.emplace_back(*fit);
              created_field_spaces.erase(finder);
              // Count how many regions are still using this field space
              // that still need to be deleted before we can remove the
              // list of created fields
              std::set<LogicalRegion> remaining_regions;
              for (std::map<LogicalRegion,unsigned>::const_iterator it = 
                    created_regions.begin(); it != created_regions.end(); it++)
                if (it->first.get_field_space() == fit->space)
                  remaining_regions.insert(it->first);
              for (std::map<LogicalRegion,bool>::const_iterator it = 
                    local_regions.begin(); it != local_regions.end(); it++)
                if (it->first.get_field_space() == fit->space)
                  remaining_regions.insert(it->first);
              if (remaining_regions.empty())
              {
                // No remaining regions so we can remove any created fields now
                for (std::set<std::pair<FieldSpace,FieldID> >::iterator it = 
                      created_fields.begin(); it != 
                      created_fields.end(); /*nothing*/)
                {
                  if (it->first == fit->space)
                  {
                    std::set<std::pair<FieldSpace,FieldID> >::iterator 
                      to_delete = it++;
                    created_fields.erase(to_delete);
                  }
                  else
                    it++;
                }
              }
              else
                latent_field_spaces[fit->space] = remaining_regions;
            }
          }
          else
            // If we didn't make this field space, record the deletion
            // and keep going. It will be handled by the context that
            // made the field space
            deleted_field_spaces.emplace_back(*fit);
        }
      }
      if (!delete_now.empty())
      {
        for (std::vector<DeletedFieldSpace>::const_iterator it = 
              delete_now.begin(); it != delete_now.end(); it++)
        {
          DeletionOp *op = runtime->get_available_deletion_op();
          op->initialize_field_space_deletion(this, it->space,
                            true/*unordered*/, it->provenance);
          op->set_deletion_preconditions(precondition, dependences);
          if (!add_to_dependence_queue(op, true/*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(
                Runtime::protect_event(op->get_completion_event()));
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_space_creations(
                                          std::map<IndexSpace,unsigned> &spaces)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_index_spaces.empty())
      {
        for (std::map<IndexSpace,unsigned>::const_iterator it = 
              spaces.begin(); it != spaces.end(); it++)
        {
          std::map<IndexSpace,unsigned>::iterator finder = 
            created_index_spaces.find(it->first);
          if (finder == created_index_spaces.end())
            created_index_spaces.insert(*it);
          else
            finder->second += it->second;
        }
      }
      else
        created_index_spaces.swap(spaces);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_space_deletions(ApEvent precondition,
                           const std::map<Operation*,GenerationID> &dependences,
                                         std::vector<DeletedIndexSpace> &spaces,
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedIndexSpace> delete_now;
      std::vector<std::vector<IndexPartition> > sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::vector<DeletedIndexSpace>::const_iterator sit =
              spaces.begin(); sit != spaces.end(); sit++)
        {
          std::map<IndexSpace,unsigned>::iterator finder = 
            created_index_spaces.find(sit->space);
          if (finder != created_index_spaces.end())
          {
#ifdef DEBUG_LEGION
            assert(finder->second > 0);
#endif
            if (--finder->second == 0)
            {
              delete_now.emplace_back(*sit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_spaces.erase(finder);
              if (sit->recurse)
              {
                std::vector<IndexPartition> &subs = sub_partitions.back();
                // Also remove any index partitions for this index space tree
                for (std::map<IndexPartition,unsigned>::iterator it = 
                      created_index_partitions.begin(); it !=
                      created_index_partitions.end(); /*nothing*/)
                {
                  if (it->first.get_tree_id() == sit->space.get_tree_id()) 
                  {
#ifdef DEBUG_LEGION
                    assert(it->second > 0);
#endif
                    if (--it->second == 0)
                    {
                      subs.push_back(it->first);
                      std::map<IndexPartition,unsigned>::iterator 
                        to_delete = it++;
                      created_index_partitions.erase(to_delete);
                    }
                    else
                      it++;
                  }
                  else
                    it++;
                }
              }
            }
          }
          else
            // If we didn't make the index space in this context, just
            // record it and keep going, it will get handled later
            deleted_index_spaces.emplace_back(*sit);
        }
      }
      if (!delete_now.empty())
      {
#ifdef DEBUG_LEGION
        assert(delete_now.size() == sub_partitions.size());
#endif
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          DeletionOp *op = runtime->get_available_deletion_op();
          op->initialize_index_space_deletion(this, delete_now[idx].space,
            sub_partitions[idx], true/*unordered*/, delete_now[idx].provenance);
          op->set_deletion_preconditions(precondition, dependences);
          if (!add_to_dependence_queue(op, true/*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(
                Runtime::protect_event(op->get_completion_event()));
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_partition_creations(
                                       std::map<IndexPartition,unsigned> &parts)
    //--------------------------------------------------------------------------
    {
      AutoLock priv_lock(privilege_lock);
      if (!created_index_partitions.empty())
      {
        for (std::map<IndexPartition,unsigned>::const_iterator it = 
              parts.begin(); it != parts.end(); it++)
        {
          std::map<IndexPartition,unsigned>::iterator finder = 
            created_index_partitions.find(it->first);
          if (finder == created_index_partitions.end())
            created_index_partitions.insert(*it);
          else
            finder->second += it->second;
        }
      }
      else
        created_index_partitions.swap(parts);
    }

    //--------------------------------------------------------------------------
    void InnerContext::register_index_partition_deletions(ApEvent precondition,
                           const std::map<Operation*,GenerationID> &dependences,
                                           std::vector<DeletedPartition> &parts, 
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedPartition> delete_now;
      std::vector<std::vector<IndexPartition> > sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::vector<DeletedPartition>::const_iterator pit =
              parts.begin(); pit != parts.end(); pit++)
        {
          std::map<IndexPartition,unsigned>::iterator finder = 
            created_index_partitions.find(pit->partition);
          if (finder != created_index_partitions.end())
          {
#ifdef DEBUG_LEGION
            assert(finder->second > 0);
#endif
            if (--finder->second == 0)
            {
              delete_now.emplace_back(*pit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_partitions.erase(finder);
              if (pit->recurse)
              {
                std::vector<IndexPartition> &subs = sub_partitions.back();
                // Remove any other partitions that this partition dominates
                for (std::map<IndexPartition,unsigned>::iterator it = 
                      created_index_partitions.begin(); it !=
                      created_index_partitions.end(); /*nothing*/)
                {
                  if ((pit->partition.get_tree_id() == it->first.get_tree_id()) 
                        && runtime->forest->is_dominated_tree_only(it->first, 
                                                                pit->partition))
                  {
#ifdef DEBUG_LEGION
                    assert(it->second > 0);
#endif
                    if (--it->second == 0)
                    {
                      subs.push_back(it->first);
                      std::map<IndexPartition,unsigned>::iterator 
                        to_delete = it++;
                      created_index_partitions.erase(to_delete);
                    }
                    else
                      it++;
                  }
                  else
                    it++;
                }
              }
            }
          }
          else
            // If we didn't make the partition, record it and keep going
            deleted_index_partitions.emplace_back(*pit);
        }
      }
      if (!delete_now.empty())
      {
#ifdef DEBUG_LEGION
        assert(delete_now.size() == sub_partitions.size());
#endif
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          DeletionOp *op = runtime->get_available_deletion_op();
          op->initialize_index_part_deletion(this, delete_now[idx].partition,
            sub_partitions[idx], true/*unordered*/, delete_now[idx].provenance);
          op->set_deletion_preconditions(precondition, dependences);
          if (!add_to_dependence_queue(op, true/*unordered*/))
          {
            // We're past the execution of the parent task so we need
            // to run this manually and capture its effects ourselves
            preconditions.insert(
                Runtime::protect_event(op->get_completion_event()));
            op->execute_dependence_analysis();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::compute_return_deletion_dependences(
            size_t return_index, std::map<Operation*,GenerationID> &dependences)
    //--------------------------------------------------------------------------
    {
      // This is a mixed mapping and execution fence analysis 
      std::set<ApEvent> previous_events;
      {
        AutoLock child_lock(child_op_lock,1,false/*exclusive*/); 
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executing_children.begin(); it != executing_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our deletion we don't care
          if (op_index >= return_index)
            continue;
          dependences.insert(*it);
          previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our deletion we don't care
          if (op_index >= return_index)
            continue;
          dependences.insert(*it);
          previous_events.insert(it->first->get_completion_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our deletion we don't care
          if (op_index >= return_index)
            continue;
          dependences.insert(*it);
          previous_events.insert(it->first->get_completion_event());
        }
      }
      // Do not check the current execution fence as it may have come after us
      if (!previous_events.empty())
        return Runtime::merge_events(NULL, previous_events);
      return ApEvent::NO_AP_EVENT;
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
    bool InnerContext::is_inner_context(void) const
    //--------------------------------------------------------------------------
    {
      return full_inner_context;
    }

    //--------------------------------------------------------------------------
    RtEvent InnerContext::compute_equivalence_sets(VersionManager *manager,
                              RegionTreeID tree_id, IndexSpace handle,
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(handle.exists());
#endif
      EquivalenceSet *root = NULL;
      if (!expr->is_empty())
      {
        AutoLock tree_lock(tree_set_lock,1,false/*exclusive*/);
        std::map<RegionTreeID,EquivalenceSet*>::const_iterator finder = 
          tree_equivalence_sets.find(tree_id);
        if (finder != tree_equivalence_sets.end())
          root = finder->second;
      }
      else
        return RtEvent::NO_RT_EVENT;
      if (root == NULL)
      {
        RegionNode *root_node = runtime->forest->get_tree(tree_id);
        IndexSpaceExpression *root_expr = 
          root_node->get_index_space_expression();
        AutoLock tree_lock(tree_set_lock);
        // See if we lost the race
        std::map<RegionTreeID,EquivalenceSet*>::const_iterator finder = 
          tree_equivalence_sets.find(tree_id);
        if (finder == tree_equivalence_sets.end())
        {
          // Didn't loose the race so we have to make the top-level
          // equivalence set for this region tree
          const AddressSpaceID local_space = runtime->address_space;
          root = new EquivalenceSet(runtime, 
              runtime->get_available_distributed_id(), local_space, local_space,
              root_expr, root_node->row_source, true/*register now*/); 
          tree_equivalence_sets[tree_id] = root;
          root->add_base_resource_ref(CONTEXT_REF);
        }
        else
          root = finder->second;
      }
#ifdef DEBUG_LEGION
      assert(root != NULL);
#endif
      RtUserEvent ready = Runtime::create_rt_user_event();
      root->ray_trace_equivalence_sets(manager, expr, mask,
                                       handle, source, ready);
      return ready;
    } 

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_logical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      // If this is one of our original region requirements then
      // we can do the analysis in our original context
      if (index < regions.size())
        return this;
      // Otherwise we need to see if this going to be one of our
      // region requirements that returns privileges or not. If
      // it is then we do the analysis in the outermost context
      // otherwise we do it locally in our own context. We need
      // to hold the operation lock to look at this data structure.
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
#ifdef DEBUG_LEGION
        assert(returnable_privileges.find(index) != 
                returnable_privileges.end());
#endif
        if (!returnable_privileges[index])
          return this;
      }
      // Fall through and return the outermost conext
      return find_outermost_local_context();
    }

    //--------------------------------------------------------------------------
    InnerContext* InnerContext::find_parent_physical_context(unsigned index,
                                                           LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == parent_req_indexes.size());
#endif     
      if (index < virtual_mapped.size())
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
                                    parent_req_indexes[index], parent);
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
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        std::map<unsigned,bool>::const_iterator finder = 
          returnable_privileges.find(index);
        if ((finder != returnable_privileges.end()) && !finder->second)
          return this;
      }
      // All through and return the top context
      return find_top_context();
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
    InnerContext* InnerContext::find_top_context(InnerContext *previous)
    //--------------------------------------------------------------------------
    {
      TaskContext *parent = find_parent_context();
      if (parent != NULL)
        return parent->find_top_context(this);
#ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;
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
      rez.serialize(find_parent_context()->get_context_uid());
      rez.serialize<size_t>(context_coordinates.size());
      for (std::vector<std::pair<size_t,DomainPoint> >::const_iterator it =
            context_coordinates.begin(); it != context_coordinates.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      // Finally pack the local field infos
      AutoLock local_lock(local_field_lock,1,false/*exclusive*/);
      rez.serialize<size_t>(local_field_infos.size());
      for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator 
            it = local_field_infos.begin(); 
            it != local_field_infos.end(); it++)
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
    void InnerContext::compute_task_tree_coordinates(
                       std::vector<std::pair<size_t,DomainPoint> > &coordinates)
    //--------------------------------------------------------------------------
    {
      coordinates = context_coordinates;
    }

    //--------------------------------------------------------------------------
    void InnerContext::send_back_created_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (created_requirements.empty())
        return;
      UniqueID target_context_uid = find_parent_context()->get_context_uid();
      for (std::map<unsigned,RegionRequirement>::const_iterator it = 
           created_requirements.begin(); it != created_requirements.end(); it++)
      {
        const RegionRequirement &req = it->second;
#ifdef DEBUG_LEGION
        assert(returnable_privileges.find(it->first) != 
                returnable_privileges.end());
#endif
        // Skip anything that doesn't have returnable privileges
        if (!returnable_privileges[it->first])
          continue;
        runtime->forest->send_back_logical_state(tree_context, 
                        target_context_uid, req, target);
      }
    } 

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space(const Future &future, 
                                       TypeTag type_tag, const char *provenance)
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
        LegionSpy::log_top_index_space(handle.id,
              runtime->address_space, provenance);
      // Get a new creation operation
      CreationOp *creator_op = runtime->get_available_creation_op();
      const ApEvent ready = creator_op->get_completion_event();
      IndexSpaceNode *node = runtime->forest->create_index_space(handle, 
          NULL, did, creator_op->get_provenance(), ready);
      creator_op->initialize_index_space(this, node, future, provenance);
      register_index_space_creation(handle);
      add_to_dependence_queue(creator_op);
      return handle;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_space(IndexSpace handle, 
               const bool unordered, const bool recurse, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->forest->is_top_level_index_space(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy index space %x in task %s (UID %lld) "
            "which is not a top-level index space. Legion only permits "
            "top-level index spaces to be destroyed.", handle.get_id(),
            get_task_name(), get_unique_id())
      Provenance *prov = NULL;
      // Check to see if this is one that we should be allowed to destory
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexSpace,unsigned>::iterator finder = 
          created_index_spaces.find(handle);
        if (finder == created_index_spaces.end())
        {
          if (provenance != NULL)
            prov = new Provenance(provenance);
          // If we didn't make the index space in this context, just
          // record it and keep going, it will get handled later
          deleted_index_spaces.emplace_back(
              DeletedIndexSpace(handle, recurse, prov));
          return;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_index_spaces.erase(finder);
          else
            return;
        }
        if (recurse)
        {
          // Also remove any index partitions for this index space tree
          for (std::map<IndexPartition,unsigned>::iterator it = 
                created_index_partitions.begin(); it !=
                created_index_partitions.end(); /*nothing*/)
          {
            if (it->first.get_tree_id() == handle.get_tree_id()) 
            {
              sub_partitions.push_back(it->first);
#ifdef DEBUG_LEGION
              assert(it->second > 0);
#endif
              if (--it->second == 0)
              {
                std::map<IndexPartition,unsigned>::iterator to_delete = it++;
                created_index_partitions.erase(to_delete);
              }
              else
                it++;
            }
            else
              it++;
          }
        }
      }
      if (provenance != NULL)
        prov = new Provenance(provenance);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_index_space_deletion(this, handle, sub_partitions,
                                          unordered, prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered index space deletion performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_index_partition(IndexPartition handle,
               const bool unordered, const bool recurse, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index partition %x in task %s (ID %lld)",
                      handle.id, get_task_name(), get_unique_id());
#endif
      Provenance *prov = NULL;
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexPartition,unsigned>::iterator finder = 
          created_index_partitions.find(handle);
        if (finder != created_index_partitions.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_index_partitions.erase(finder);
          else
            return;
          if (recurse)
          {
            // Remove any other partitions that this partition dominates
            for (std::map<IndexPartition,unsigned>::iterator it = 
                  created_index_partitions.begin(); it !=
                  created_index_partitions.end(); /*nothing*/)
            {
              if ((handle.get_tree_id() == it->first.get_tree_id()) &&
                  runtime->forest->is_dominated_tree_only(it->first, handle))
              {
                sub_partitions.push_back(it->first);
#ifdef DEBUG_LEGION
                assert(it->second > 0);
#endif
                if (--it->second == 0)
                {
                  std::map<IndexPartition,unsigned>::iterator to_delete = it++;
                  created_index_partitions.erase(to_delete);
                }
                else
                  it++;
              }
              else
                it++;
            }
          }
        }
        else
        {
          if (provenance != NULL)
            prov = new Provenance(provenance);
          // If we didn't make the partition, record it and keep going
          deleted_index_partitions.push_back(
              DeletedPartition(handle, recurse, prov));
          return;
        }
      }
      if (provenance != NULL)
        prov = new Provenance(provenance);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_index_part_deletion(this, handle, 
                                         sub_partitions, unordered, prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered index partition deletion performed after task %s"
            " (UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_equal_partition(
                                                      IndexSpace parent,
                                                      IndexSpace color_space,
                                                      size_t granularity,
                                                      Color color,
                                                      const char *provenance)
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
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_equal_partition(this, pid, granularity, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this,pid,parent,
                    color_space, partition_color, LEGION_DISJOINT_COMPLETE_KIND,
                    did, part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_weights(IndexSpace parent,
                                                const FutureMap &weights, 
                                                IndexSpace color_space,
                                                size_t granularity, Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);  
      const IndexPartition pid(runtime->get_unique_index_partition_id(), 
                               parent.get_tree_id(), parent.get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition %d by weights with parent index "
                      "space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_weight_partition(this, pid, weights, 
                                           granularity, provenance);
      const ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RegionTreeForest *forest = runtime->forest;
      const RtEvent safe = forest->create_pending_partition(this, pid, parent,
                  color_space, partition_color, LEGION_DISJOINT_COMPLETE_KIND,
                  did, part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_union(
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          IndexSpace color_space,
                                          PartitionKind kind, Color color,
                                          const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind)
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
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_union_partition(this, pid, handle1, 
                                          handle2, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // If either partition is aliased the result is aliased
      if ((kind == LEGION_COMPUTE_KIND) || 
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // If one of these partitions is aliased then the result is aliased
        IndexPartNode *p1 = runtime->forest->get_node(handle1);
        if (p1->is_disjoint(true/*from app*/))
        {
          IndexPartNode *p2 = runtime->forest->get_node(handle2);
          if (!p2->is_disjoint(true/*from app*/))
          {
            if (kind == LEGION_COMPUTE_KIND)
              kind = LEGION_ALIASED_KIND;
            else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
              kind = LEGION_ALIASED_COMPLETE_KIND;
            else
              kind = LEGION_ALIASED_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_ALIASED_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_ALIASED_COMPLETE_KIND;
          else
            kind = LEGION_ALIASED_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            parent, color_space, partition_color, kind, did,
            part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__); 
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
                                              IndexSpace parent,
                                              IndexPartition handle1,
                                              IndexPartition handle2,
                                              IndexSpace color_space,
                                              PartitionKind kind, Color color,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind)
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
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_intersection_partition(this, pid, handle1, 
                                                 handle2, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // If either partition is disjoint then the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) || 
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p1 = runtime->forest->get_node(handle1);
        if (!p1->is_disjoint(true/*from app*/))
        {
          IndexPartNode *p2 = runtime->forest->get_node(handle2);
          if (p2->is_disjoint(true/*from app*/))
          {
            if (kind == LEGION_COMPUTE_KIND)
              kind = LEGION_DISJOINT_KIND;
            else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
              kind = LEGION_DISJOINT_COMPLETE_KIND;
            else
              kind = LEGION_DISJOINT_INCOMPLETE_KIND;
          }
        }
        else
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            parent, color_space, partition_color, kind, did,
            part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_intersection(
                                              IndexSpace parent,
                                              IndexPartition partition,
                                              PartitionKind kind, Color color,
                                              bool dominates,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating intersection partition %d with parent "
                      "index space %x in task %s (ID %lld)", pid.id, parent.id,
                      get_task_name(), get_unique_id());
      if (parent.get_type_tag() != partition.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_INDEXPARTITION_NOT_SAME_INDEX_TREE,
            "IndexPartition %d does not have the same type as the "
            "parent index space %x in task %s (UID %lld)", partition.id,
            parent.id, get_task_name(), get_unique_id())
#endif
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_intersection_partition(this, pid, partition,
                                                 dominates, provenance);
      ApEvent term_event = part_op->get_completion_event();
      IndexPartNode *part_node = runtime->forest->get_node(partition);
      // See if we can determine disjointness if we weren't told
      if ((kind == LEGION_COMPUTE_KIND) || 
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (part_node->is_disjoint(true/*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid,parent,
                     part_node->color_space->handle, partition_color, kind, did,
                     part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_difference(
                                                  IndexSpace parent,
                                                  IndexPartition handle1,
                                                  IndexPartition handle2,
                                                  IndexSpace color_space,
                                                  PartitionKind kind, 
                                                  Color color,
                                                  const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind)
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
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_difference_partition(this, pid, handle1, 
                                               handle2, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // If the left-hand-side is disjoint the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) || 
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p1 = runtime->forest->get_node(handle1);
        if (p1->is_disjoint(true/*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
                         parent, color_space, partition_color, kind, did,
                         part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    Color InnerContext::create_cross_product_partitions(
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                   std::map<IndexSpace,IndexPartition> &handles,
                                                      PartitionKind kind,
                                                      Color color,
                                                      const char *provenance)
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
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind)
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      std::set<RtEvent> safe_events;
      runtime->forest->create_pending_cross_product(this, handle1, handle2, 
                                  handles, kind, part_op->get_provenance(),
                                  partition_color, term_event, safe_events);
      part_op->initialize_cross_product(this, handle1, handle2,
                                        partition_color, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (!safe_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(safe_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      if (runtime->verify_partitions)
      {
        Domain color_space = runtime->get_index_partition_color_space(handle1);
        // This code will only work if the color space has type coord_t
        TypeTag type_tag;
        switch (color_space.get_dim())
        {
#define DIMFUNC(DIM) \
          case DIM: \
            { \
              type_tag = NT_TemplateHelper::encode_tag<DIM,coord_t>(); \
              break; \
            }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            assert(false);
        }
        for (Domain::DomainPointIterator itr(color_space); itr; itr++)
        {
          IndexSpace subspace;
          switch (color_space.get_dim())
          {
#define DIMFUNC(DIM) \
            case DIM: \
              { \
                const Point<DIM,coord_t> p(itr.p); \
                subspace = runtime->get_index_subspace(handle1, &p, type_tag); \
                break; \
              }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              assert(false);
          }
          IndexPartition part = 
            runtime->get_index_partition(subspace, partition_color);
          verify_partition(part, verify_kind, __func__);
        }
      }
      return partition_color;
    }

    //--------------------------------------------------------------------------
    void InnerContext::create_association(LogicalRegion domain,
                                          LogicalRegion domain_parent,
                                          FieldID domain_fid,
                                          IndexSpace range,
                                          MapperID id, MappingTagID tag,
                                          const UntypedBuffer &marg,
                                          const char *prov)
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
                              domain_fid, range, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_restricted_partition(
                                              IndexSpace parent,
                                              IndexSpace color_space,
                                              const void *transform,
                                              size_t transform_size,
                                              const void *extent,
                                              size_t extent_size,
                                              PartitionKind part_kind,
                                              Color color,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating restricted partition in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color; 
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_restricted_partition(this, pid, transform, 
                    transform_size, extent, extent_size, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            parent, color_space, part_color, part_kind, did,
            part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_domain(
                                                IndexSpace parent,
                                                const FutureMap &domains,
                                                IndexSpace color_space,
                                                bool perform_intersections,
                                                PartitionKind part_kind,
                                                Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by domain in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color; 
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_by_domain(this, pid, domains, 
                      perform_intersections, provenance);
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            parent, color_space, part_color, part_kind, did, 
            part_op->get_provenance(), term_event);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_field(
                                              LogicalRegion handle,
                                              LogicalRegion parent_priv,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              Color color,
                                              MapperID id, MappingTagID tag,
                                              PartitionKind part_kind,
                                              const UntypedBuffer &marg,
                                              const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Partition by field is disjoint by construction
      PartitionKind verify_kind = LEGION_DISJOINT_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexSpace parent = handle.get_index_space(); 
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by field in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition 
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            parent, color_space, part_color, part_kind, did,
            part_op->get_provenance(), term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_field(this, pid, handle, parent_priv, 
                                   fid, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image(
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color,
                                                    MapperID id, 
                                                    MappingTagID tag,
                                                    const UntypedBuffer &marg,
                                                    const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_tree_id(), handle.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by image in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event(); 
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            handle, color_space, part_color, part_kind, did,
            part_op->get_provenance(), term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image(this, pid, projection, parent,
                                   fid, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_image_range(
                                                    IndexSpace handle,
                                                    LogicalPartition projection,
                                                    LogicalRegion parent,
                                                    FieldID fid,
                                                    IndexSpace color_space,
                                                    PartitionKind part_kind,
                                                    Color color,
                                                    MapperID id, 
                                                    MappingTagID tag,
                                                    const UntypedBuffer &marg,
                                                    const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_tree_id(), handle.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by image range in task %s (ID %lld)",
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op();
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
            handle, color_space, part_color, part_kind, did,
            part_op->get_provenance(), term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_image_range(this, pid, projection, parent, 
                                         fid, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage(
                                                  IndexPartition projection,
                                                  LogicalRegion handle,
                                                  LogicalRegion parent,
                                                  FieldID fid,
                                                  IndexSpace color_space,
                                                  PartitionKind part_kind,
                                                  Color color,
                                                  MapperID id, MappingTagID tag,
                                                  const UntypedBuffer &marg,
                                                  const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id(),
                         parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by preimage in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(); 
      ApEvent term_event = part_op->get_completion_event();
      // If the source of the preimage is disjoint then the result is disjoint
      // Note this only applies here and not to range
      if ((part_kind == LEGION_COMPUTE_KIND) || 
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode *p = runtime->forest->get_node(projection);
        if (p->is_disjoint(true/*from app*/))
        {
          if (part_kind == LEGION_COMPUTE_KIND)
            part_kind = LEGION_DISJOINT_KIND;
          else if (part_kind == LEGION_COMPUTE_COMPLETE_KIND)
            part_kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            part_kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
                                       handle.get_index_space(), color_space, 
                                       part_color, part_kind, did,
                                       part_op->get_provenance(), term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage(this, pid, projection, handle, 
                                      parent, fid, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_partition_by_preimage_range(
                                                  IndexPartition projection,
                                                  LogicalRegion handle,
                                                  LogicalRegion parent,
                                                  FieldID fid,
                                                  IndexSpace color_space,
                                                  PartitionKind part_kind,
                                                  Color color,
                                                  MapperID id, MappingTagID tag,
                                                  const UntypedBuffer &marg,
                                                  const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         handle.get_index_space().get_tree_id(),
                         parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating partition by preimage range in task %s "
                      "(ID %lld)", get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      // Allocate the partition operation
      DependentPartitionOp *part_op = 
        runtime->get_available_dependent_partition_op(); 
      ApEvent term_event = part_op->get_completion_event();
      // Tell the region tree forest about this partition
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
                                       handle.get_index_space(), color_space, 
                                       part_color, part_kind, did,
                                       part_op->get_provenance(), term_event);
      // Do this after creating the pending partition so the node exists
      // in case we need to look at it during initialization
      part_op->initialize_by_preimage_range(this, pid, projection, handle,
                                            parent, fid, id, tag, marg, prov);
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
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, prov);
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition InnerContext::create_pending_partition(
                                                IndexSpace parent,
                                                IndexSpace color_space, 
                                                PartitionKind part_kind,
                                                Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind)
      IndexPartition pid(runtime->get_unique_index_partition_id(), 
                         parent.get_tree_id(), parent.get_type_tag());
      DistributedID did = runtime->get_available_distributed_id();
#ifdef DEBUG_LEGION
      log_index.debug("Creating pending partition in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      LegionColor part_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      Provenance *provenance = NULL;
      if (prov != NULL)
      {
        provenance = new Provenance(prov);
        provenance->add_reference();
      }
      const ApUserEvent partition_ready = Runtime::create_ap_user_event(NULL);
      RtEvent safe = runtime->forest->create_pending_partition(this, pid, 
                            parent, color_space, part_color, part_kind, 
                            did, provenance, partition_ready, partition_ready);
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      // Wait for any notifications to occur before returning
      if (safe.exists())
        safe.wait();
      if (runtime->verify_partitions)
      {
        // We can't block to check this here because the user needs 
        // control back in order to fill in the pieces of the partitions
        // so just launch a meta-task to check it when we can
        VerifyPartitionArgs args(this, pid, verify_kind, __func__);
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, 
            Runtime::protect_event(partition_ready));
      }
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles,
                                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = runtime->forest->find_pending_space(parent, 
                                realm_color, type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_union(this, result, handles, provenance);
      Runtime::trigger_event(NULL,domain_ready,part_op->get_completion_event());
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_union(IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                                      IndexPartition handle,
                                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space union in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = runtime->forest->find_pending_space(parent, 
                                realm_color, type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_union(this, result, handle, provenance);
      Runtime::trigger_event(NULL,domain_ready,part_op->get_completion_event());
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles,
                                                      const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = runtime->forest->find_pending_space(parent, 
                                realm_color, type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(this, result, handles, prov);
      Runtime::trigger_event(NULL,domain_ready,part_op->get_completion_event());
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_intersection(
                                                      IndexPartition parent,
                                                      const void *realm_color,
                                                      TypeTag type_tag,
                                                      IndexPartition handle,
                                                      const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space intersection in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = runtime->forest->find_pending_space(parent, 
                                realm_color, type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_intersection(this, result, handle, prov);
      Runtime::trigger_event(NULL,domain_ready,part_op->get_completion_event());
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace InnerContext::create_index_space_difference(
                                                    IndexPartition parent,
                                                    const void *realm_color,
                                                    TypeTag type_tag,
                                                    IndexSpace initial,
                                        const std::vector<IndexSpace> &handles,
                                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_index.debug("Creating index space difference in task %s (ID %lld)", 
                      get_task_name(), get_unique_id());
#endif
      ApUserEvent domain_ready;
      IndexSpace result = runtime->forest->find_pending_space(parent, 
                                realm_color, type_tag, domain_ready);
      PendingPartitionOp *part_op = 
        runtime->get_available_pending_partition_op();
      part_op->initialize_index_space_difference(this, result, initial,
                                                 handles, provenance);
      Runtime::trigger_event(NULL,domain_ready,part_op->get_completion_event());
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    } 

    //--------------------------------------------------------------------------
    void InnerContext::verify_partition(IndexPartition pid, PartitionKind kind,
                                        const char *function_name)
    //--------------------------------------------------------------------------
    {
      IndexPartNode *node = runtime->forest->get_node(pid);
      // Check containment first because our implementation of the algorithms
      // for disjointnss and completeness rely upon it.
      if (node->total_children == node->max_linearized_color)
      {
        for (LegionColor color = 0; color < node->total_children; color++)
        {
          IndexSpaceNode *child_node = node->get_child(color);
          IndexSpaceExpression *diff = 
            runtime->forest->subtract_index_spaces(child_node, node->parent);
          if (!diff->is_empty())
          {
            const DomainPoint bad = 
              node->color_space->delinearize_color_to_point(color);
            switch (bad.get_dim())
            {
              case 1:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0])
              case 2:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1])
              case 3:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2])
              case 4:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3])
              case 5:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4])
              case 6:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5])
              case 7:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6])
              case 8:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6],
                    bad[7])
              case 9:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6],
                    bad[7], bad[8])
              default:
                assert(false);
            }
          }
        }
      }
      else
      {
        ColorSpaceIterator *itr =
          node->color_space->create_color_space_iterator();
        while (itr->is_valid())
        {
          const LegionColor color = itr->yield_color();
          IndexSpaceNode *child_node = node->get_child(color);
          IndexSpaceExpression *diff = 
            runtime->forest->subtract_index_spaces(child_node, node->parent);
          if (!diff->is_empty())
          {
            const DomainPoint bad = 
              node->color_space->delinearize_color_to_point(color);
            switch (bad.get_dim())
            {
              case 1:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0])
              case 2:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1])
              case 3:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2])
              case 4:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3])
              case 5:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4])
              case 6:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5])
              case 7:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6])
              case 8:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6],
                    bad[7])
              case 9:
                REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
                    "Call to partition function %s in %s (UID %lld) has "
                    "non-dominated child sub-region at color (%lld,%lld,"
                    "%lld,%lld,%lld,%lld,%lld,%lld,%lld).",
                    function_name, get_task_name(), get_unique_id(),
                    bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6],
                    bad[7], bad[8])
              default:
                assert(false);
            }
          }
        }
        delete itr;
      }
      // Check disjointness
      if ((kind == LEGION_DISJOINT_KIND) || 
          (kind == LEGION_DISJOINT_COMPLETE_KIND) ||
          (kind == LEGION_DISJOINT_INCOMPLETE_KIND))
      {
        if (!node->is_disjoint(true/*from application*/))
          REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
              "Call to partitioning function %s in %s (UID %lld) specified "
              "partition was %s but the partition is aliased.",
              function_name, get_task_name(), get_unique_id(),
              (kind == LEGION_DISJOINT_KIND) ? "DISJOINT_KIND" :
              (kind == LEGION_DISJOINT_COMPLETE_KIND) ? "DISJOINT_COMPLETE_KIND"
              : "DISJOINT_INCOMPLETE_KIND")
      }
      else if ((kind == LEGION_ALIASED_KIND) || 
               (kind == LEGION_ALIASED_COMPLETE_KIND) ||
               (kind == LEGION_ALIASED_INCOMPLETE_KIND))
      {
        if (node->is_disjoint(true/*from application*/))
          REPORT_LEGION_WARNING(LEGION_WARNING_PARTITION_VERIFICATION,
              "Call to partitioning function %s in %s (UID %lld) specified "
              "partition was %s but the partition is disjoint. This could "
              "lead to a performance bug.",
              function_name, get_task_name(), get_unique_id(),
              (kind == LEGION_ALIASED_KIND) ? "ALIASED_KIND" :
              (kind == LEGION_ALIASED_COMPLETE_KIND) ? "ALIASED_COMPLETE_KIND" :
              "ALIASED_INCOMPLETE_KIND")
      }
      // Check completeness
      if ((kind == LEGION_DISJOINT_COMPLETE_KIND) || 
          (kind == LEGION_ALIASED_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND))
      {
        if (!node->is_complete(true/*from application*/))
          REPORT_LEGION_ERROR(ERROR_PARTITION_VERIFICATION,
              "Call to partitioning function %s in %s (UID %lld) specified "
              "partition was %s but the partition is incomplete.",
              function_name, get_task_name(), get_unique_id(),
              (kind == LEGION_DISJOINT_COMPLETE_KIND) ? "DISJOINT_COMPLETE_KIND" 
            : (kind == LEGION_ALIASED_COMPLETE_KIND) ? "ALIASED_COMPLETE_KIND" :
              "COMPUTE_COMPLETE_KIND")
      }
      else if ((kind == LEGION_DISJOINT_INCOMPLETE_KIND) || 
               (kind == LEGION_ALIASED_INCOMPLETE_KIND) || 
               (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (node->is_complete(true/*from application*/))
          REPORT_LEGION_WARNING(LEGION_WARNING_PARTITION_VERIFICATION,
              "Call to partitioning function %s in %s (UID %lld) specified "
              "partition was %s but the partition is complete. This could "
              "lead to a performance bug.",
              function_name, get_task_name(), get_unique_id(),
              (kind == LEGION_DISJOINT_INCOMPLETE_KIND) ? 
                "DISJOINT_INCOMPLETE_KIND" :
              (kind == LEGION_ALIASED_INCOMPLETE_KIND) ? 
              "ALIASED_INCOMPLETE_KIND" : "COMPUTE_INCOMPLETE_KIND")
      }
    }

    //--------------------------------------------------------------------------
    /*static*/void InnerContext::handle_partition_verification(const void *args)
    //--------------------------------------------------------------------------
    {
      const VerifyPartitionArgs *vargs = (const VerifyPartitionArgs*)args;
      vargs->proxy_this->verify_partition(vargs->pid, vargs->kind, vargs->func);
    }

    //--------------------------------------------------------------------------
    FieldSpace InnerContext::create_field_space(
                                         const std::vector<Future> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         CustomSerdezID serdez_id,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      const FieldSpace space = TaskContext::create_field_space(provenance);
      AutoRuntimeCall call(this);
      FieldSpaceNode *node = runtime->forest->get_node(space);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
          REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD,
            "Task %s (ID %lld) attempted to allocate a field with "
            "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
            "bound set in legion_config.h", get_task_name(),
            get_unique_id(), resulting_fields[idx])
#endif
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
        if (sizes[idx].impl == NULL)
          REPORT_LEGION_ERROR(ERROR_REQUEST_FOR_EMPTY_FUTURE,
              "Invalid empty future passed to field allocation for field %d "
              "in task %s (UID %lld)", resulting_fields[idx],
              get_task_name(), get_unique_id())
      // Get a new creation operation
      CreationOp *creator_op = runtime->get_available_creation_op();  
      const ApEvent ready = creator_op->get_completion_event();
      node->initialize_fields(ready, resulting_fields, serdez_id,
                              creator_op->get_provenance());
      creator_op->initialize_fields(this, node, resulting_fields,
                                    sizes, provenance);
      register_all_field_creations(space, false/*local*/, resulting_fields);
      add_to_dependence_queue(creator_op);
      return space;
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_field_space(FieldSpace handle,
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
#ifdef DEBUG_LEGION
      log_field.debug("Destroying field space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      Provenance *prov = NULL;
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock);
        std::map<FieldSpace,unsigned>::iterator finder = 
          created_field_spaces.find(handle);
        if (finder != created_field_spaces.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_field_spaces.erase(finder);
          else
            return;
          // Count how many regions are still using this field space
          // that still need to be deleted before we can remove the
          // list of created fields
          std::set<LogicalRegion> latent_regions;
          for (std::map<LogicalRegion,unsigned>::const_iterator it = 
                created_regions.begin(); it != created_regions.end(); it++)
            if (it->first.get_field_space() == handle)
              latent_regions.insert(it->first);
          for (std::map<LogicalRegion,bool>::const_iterator it = 
                local_regions.begin(); it != local_regions.end(); it++)
            if (it->first.get_field_space() == handle)
              latent_regions.insert(it->first);
          if (latent_regions.empty())
          {
            // No remaining regions so we can remove any created fields now
            for (std::set<std::pair<FieldSpace,FieldID> >::iterator it = 
                  created_fields.begin(); it != 
                  created_fields.end(); /*nothing*/)
            {
              if (it->first == handle)
              {
                std::set<std::pair<FieldSpace,FieldID> >::iterator 
                  to_delete = it++;
                created_fields.erase(to_delete);
              }
              else
                it++;
            }
          }
          else
            latent_field_spaces[handle] = latent_regions;
        }
        else
        {
          if (provenance != NULL)
            prov = new Provenance(provenance);
          // If we didn't make this field space, record the deletion
          // and keep going. It will be handled by the context that
          // made the field space
          deleted_field_spaces.emplace_back(DeletedFieldSpace(handle, prov));
          return;
        }
      }
      if (provenance != NULL)
        prov = new Provenance(provenance);
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_field_space_deletion(this, handle, unordered, prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered field space deletion performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    } 

    //--------------------------------------------------------------------------
    FieldID InnerContext::allocate_field(FieldSpace space, 
                                         const Future &field_size,
                                         FieldID fid, bool local,
                                         CustomSerdezID serdez_id,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (local)
        REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
            "Local fields do no support allocation with future sizes yet.")
      if (fid == LEGION_AUTO_GENERATE_ID)
        fid = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
        REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FILED,
          "Task %s (ID %lld) attempted to allocate a field with "
          "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
          "bound set in legion_config.h", get_task_name(), get_unique_id(), fid)
#endif
      if (field_size.impl == NULL)
        REPORT_LEGION_ERROR(ERROR_REQUEST_FOR_EMPTY_FUTURE,
            "Invalid empty future passed to field allocation for field %d "
            "in task %s (UID %lld)", fid, get_task_name(), get_unique_id())
      // Get a new creation operation
      CreationOp *creator_op = runtime->get_available_creation_op();  
      const ApEvent ready = creator_op->get_completion_event();
      // Tell the node that we're allocating a field of size zero
      // which will indicate that we'll fill in the size later
      FieldSpaceNode *node = 
        runtime->forest->allocate_field(space, ready, fid,serdez_id,provenance);
      creator_op->initialize_field(this, node, fid, field_size, provenance); 
      register_field_creation(space, fid, local);
      add_to_dependence_queue(creator_op);
      return fid;
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_local_field(FieldSpace space, size_t field_size,
                                          FieldID fid, CustomSerdezID serdez_id,
                                          std::set<RtEvent> &done_events,
                                          const char *provenance)
    //--------------------------------------------------------------------------
    {
      // See if we've exceeded our local field allocations 
      // for this field space
      AutoLock local_lock(local_field_lock);
      std::vector<LocalFieldInfo> &infos = local_field_infos[space];
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
      if (!runtime->forest->allocate_local_fields(space, fields, sizes, 
                  serdez_id, current_indexes, new_indexes, provenance))
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
      const size_t prov_size = (provenance != NULL) ? strlen(provenance) : 0;
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
          rez.serialize(prov_size);
          if (prov_size > 0)
            rez.serialize(provenance, prov_size);
          rez.serialize<size_t>(1); // field count
          rez.serialize(infos.back());
          rez.serialize(done_event);
        }
        runtime->send_local_field_update(it->first, rez);
        done_events.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_fields(FieldSpace space,
                                       const std::vector<Future> &sizes,
                                       std::vector<FieldID> &resulting_fields,
                                       bool local, CustomSerdezID serdez_id,
                                       const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (local)
        REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
            "Local fields do no support allocation with future sizes yet.") 
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
          resulting_fields[idx] = runtime->get_unique_field_id();
#ifdef DEBUG_LEGION
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
          REPORT_LEGION_ERROR(ERROR_TASK_ATTEMPTED_ALLOCATE_FIELD,
            "Task %s (ID %lld) attempted to allocate a field with "
            "ID %d which exceeds the LEGION_MAX_APPLICATION_FIELD_ID "
            "bound set in legion_config.h", get_task_name(),
            get_unique_id(), resulting_fields[idx])
#endif
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
        if (sizes[idx].impl == NULL)
          REPORT_LEGION_ERROR(ERROR_REQUEST_FOR_EMPTY_FUTURE,
              "Invalid empty future passed to field allocation for field %d "
              "in task %s (UID %lld)", resulting_fields[idx],
              get_task_name(), get_unique_id())
      // Get a new creation operation
      CreationOp *creator_op = runtime->get_available_creation_op();  
      const ApEvent ready = creator_op->get_completion_event();
      // Tell the node that we're allocating a field of size zero
      // which will indicate that we'll fill in the size later
      FieldSpaceNode *node = runtime->forest->allocate_fields(space, ready,
                                  resulting_fields, serdez_id, provenance);
      creator_op->initialize_fields(this, node, resulting_fields, 
                                    sizes, provenance);
      register_all_field_creations(space, local, resulting_fields);
      add_to_dependence_queue(creator_op);
    }

    //--------------------------------------------------------------------------
    void InnerContext::allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resulting_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events,
                                   const char *provenance)
    //--------------------------------------------------------------------------
    {
      // See if we've exceeded our local field allocations 
      // for this field space
      AutoLock local_lock(local_field_lock);
      std::vector<LocalFieldInfo> &infos = local_field_infos[space];
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
      if (!runtime->forest->allocate_local_fields(space, resulting_fields, 
              sizes, serdez_id, current_indexes, new_indexes, provenance))
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
      const size_t prov_size = (provenance != NULL) ? strlen(provenance) : 0;
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
          rez.serialize(prov_size);
          if (prov_size > 0)
            rez.serialize(provenance, prov_size);
          rez.serialize<size_t>(resulting_fields.size()); // field count
          for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
            rez.serialize(infos[offset+idx]);
          rez.serialize(done_event);
        }
        runtime->send_local_field_update(it->first, rez);
        done_events.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::free_field(FieldAllocatorImpl *allocator, 
    FieldSpace space, FieldID fid, const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        const std::pair<FieldSpace,FieldID> key(space, fid);
        // This field will actually be removed in analyze_destroy_fields
        std::set<std::pair<FieldSpace,FieldID> >::const_iterator 
          finder = created_fields.find(key);
        if (finder == created_fields.end())
        {
          std::map<std::pair<FieldSpace,FieldID>,bool>::iterator 
            local_finder = local_fields.find(key);
          if (local_finder == local_fields.end())
          {
            // If we didn't make this field, record the deletion and
            // then have a later context handle it
            deleted_fields.emplace_back(DeletedField(space, fid, prov));
            return;
          }
          else
            local_finder->second = true;
        }
        // Don't remove anything from created fields yet, we still might
        // need it as part of the logical dependence analysis for earlier ops
      }
      // Launch off the deletion operation
      DeletionOp *op = runtime->get_available_deletion_op(); 
      op->initialize_field_deletion(this, space, fid, unordered,allocator,prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered field free performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    } 

    //--------------------------------------------------------------------------
    void InnerContext::free_fields(FieldAllocatorImpl *allocator, 
                                   FieldSpace space,
                                   const std::set<FieldID> &to_free,
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);
      std::set<FieldID> free_now;
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        // These fields will actually be removed in analyze_destroy_fields
        for (std::set<FieldID>::const_iterator it = 
              to_free.begin(); it != to_free.end(); it++)
        {
          const std::pair<FieldSpace,FieldID> key(space, *it);
          std::set<std::pair<FieldSpace,FieldID> >::const_iterator 
            finder = created_fields.find(key);
          if (finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace,FieldID>,bool>::iterator 
              local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
            {
              local_finder->second = true;
              free_now.insert(*it);
            }
            else
              deleted_fields.emplace_back(DeletedField(space, *it, prov));
          }
          else
          {
            // Don't remove anything from created fields yet, 
            // we still might need need it as part of the logical 
            // dependence analysis for earlier ops
            free_now.insert(*it);
          }
        }
      }
      if (free_now.empty())
        return;
      DeletionOp *op = runtime->get_available_deletion_op();
      op->initialize_field_deletions(this, space, free_now, unordered,
                                     allocator, prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered free fields performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_logical_region(LogicalRegion handle,
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      Provenance *prov = NULL;
      
#ifdef DEBUG_LEGION
      log_region.debug("Deleting logical region (%x,%x) in task %s (ID %lld)",
                       handle.index_space.id, handle.field_space.id, 
                       get_task_name(), get_unique_id());
#endif
      // Check to see if this is a top-level logical region, if not then
      // we shouldn't even be destroying it
      if (!runtime->forest->is_top_level_region(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy logical region (%x,%x,%x in task %s "
            "(UID %lld) which is not a top-level logical region. Legion only "
            "permits top-level logical regions to be destroyed.", 
            handle.index_space.id, handle.field_space.id, handle.tree_id,
            get_task_name(), get_unique_id())
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        std::map<LogicalRegion,unsigned>::iterator finder = 
          created_regions.find(handle);
        if (finder == created_regions.end())
        {
          // Check to see if it is a local region
          std::map<LogicalRegion,bool>::iterator local_finder = 
            local_regions.find(handle);
          // Mark that this region is deleted, safe even though this
          // is a read-only lock because we're not changing the structure
          // of the map
          if (local_finder == local_regions.end())
          {
            if (provenance != NULL)
              prov = new Provenance(provenance);
            // Record the deletion for later and propagate it up
            deleted_regions.emplace_back(DeletedRegion(handle, prov));
            return;
          }
          else
            local_finder->second = true;
        }
        else
        {
          if (finder->second == 0)
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_DUPLICATE_DELETION,
                "Duplicate deletions were performed for region (%x,%x,%x) "
                "in task tree rooted by %s", handle.index_space.id, 
                handle.field_space.id, handle.tree_id, get_task_name())
            return;
          }
          if (--finder->second > 0)
            return;
          // Don't remove anything from created regions yet, we still might
          // need it as part of the logical dependence analysis for earlier
          // operations, but the reference count is zero so we're protected
        }
      }
      if (provenance != NULL)
        prov = new Provenance(provenance);
      DeletionOp *op = runtime->get_available_deletion_op(); 
      op->initialize_logical_region_deletion(this, handle, unordered, prov);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered logical region deletion performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::get_local_field_set(const FieldSpace handle,
                                           const std::set<unsigned> &indexes,
                                           std::set<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      AutoLock lf_lock(local_field_lock, 1, false/*exclusive*/);
      std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
        finder = local_field_infos.find(handle);
#ifdef DEBUG_LEGION
      assert(finder != local_field_infos.end());
      unsigned found = 0;
#endif
      for (std::vector<LocalFieldInfo>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
        if (indexes.find(it->index) != indexes.end())
        {
#ifdef DEBUG_LEGION
          found++;
#endif
          to_set.insert(it->fid);
        }
      }
#ifdef DEBUG_LEGION
      assert(found == indexes.size());
#endif
    }

    //--------------------------------------------------------------------------
    void InnerContext::get_local_field_set(const FieldSpace handle,
                                           const std::set<unsigned> &indexes,
                                           std::vector<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      AutoLock lf_lock(local_field_lock, 1, false/*exclusive*/);
      std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
        finder = local_field_infos.find(handle);
#ifdef DEBUG_LEGION
      assert(finder != local_field_infos.end());
      unsigned found = 0;
#endif
      for (std::vector<LocalFieldInfo>::const_iterator it = 
            finder->second.begin(); it != finder->second.end(); it++)
      {
        if (indexes.find(it->index) != indexes.end())
        {
#ifdef DEBUG_LEGION
          found++;
#endif
          to_set.push_back(it->fid);
        }
      }
#ifdef DEBUG_LEGION
      assert(found == indexes.size());
#endif
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_physical_region(const RegionRequirement &req,
          bool mapped, MapperID mid, MappingTagID tag, ApUserEvent &unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unmap_event.exists());
#endif
      if (!virtual_mapped)
        unmap_event = Runtime::create_ap_user_event(NULL);
      PhysicalRegionImpl *impl = new PhysicalRegionImpl(req,
          RtEvent::NO_RT_EVENT, ApEvent::NO_AP_EVENT,
          mapped ? unmap_event : ApUserEvent::NO_AP_USER_EVENT, mapped, this,
          mid, tag, false/*leaf region*/, virtual_mapped, runtime);
      physical_regions.push_back(PhysicalRegion(impl));
      if (!virtual_mapped)
        impl->set_references(physical_instances, true/*safe*/); 
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_task_false(launcher);
      IndividualTask *task = runtime->get_available_individual_task();
      Future result = task->initialize_task(this, launcher);
#ifdef DEBUG_LEGION
      log_task.debug("Registering new single task with unique id %lld "
                      "and task %s (ID %lld) with high level runtime in "
                      "addresss space %d",
                      task->get_unique_id(), task->get_task_name(), 
                      task->get_unique_id(), runtime->address_space);
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
        return predicate_index_task_false(launcher);
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
        launch_space = find_index_launch_space(launcher.launch_domain,
                                               launcher.provenance);
      IndexTask *task = runtime->get_available_index_task();
      FutureMap result = task->initialize_task(this, launcher, launch_space);
#ifdef DEBUG_LEGION
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#endif
      execute_task_launch(task, true/*index*/, current_trace, 
                          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::execute_index_space(const IndexTaskLauncher &launcher,
                                        ReductionOpID redop, bool deterministic)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        FutureMap result = execute_must_epoch(epoch_launcher);
        return reduce_future_map(result, redop, deterministic,
                                 launcher.provenance.c_str());
      }
      AutoRuntimeCall call(this);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_index_task_reduce_false(launcher);
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_EMPTY_INDEX_TASK_LAUNCH,
          "Ignoring empty index task launch in task %s (ID %lld)",
                        get_task_name(), get_unique_id());
        const ReductionOp *reduction_op = runtime->get_reduction(redop);
        FutureImpl *result = new FutureImpl(this, runtime, true/*register*/,
          runtime->get_available_distributed_id(),
          runtime->address_space, ApEvent::NO_AP_EVENT);
        result->set_result(reduction_op->identity,
                           reduction_op->sizeof_rhs, false/*own*/);
        return Future(result);
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space = find_index_launch_space(launcher.launch_domain,
                                               launcher.provenance);
      IndexTask *task = runtime->get_available_index_task();
      Future result = task->initialize_task(this, launcher, launch_space, 
                                            redop, deterministic);
#ifdef DEBUG_LEGION
      log_task.debug("Registering new index space task with unique id "
                     "%lld and task %s (ID %lld) with high level runtime in "
                     "address space %d",
                     task->get_unique_id(), task->get_task_name(), 
                     task->get_unique_id(), runtime->address_space);
#endif
      execute_task_launch(task, true/*index*/, current_trace, 
                          launcher.silence_warnings, launcher.enable_inlining);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::reduce_future_map(const FutureMap &future_map,
                      ReductionOpID redop, bool deterministic, const char *prov)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this); 
      if (future_map.impl == NULL)
      {
        const ReductionOp *reduction_op = runtime->get_reduction(redop);
        FutureImpl *result = new FutureImpl(this, runtime, true/*register*/,
          runtime->get_available_distributed_id(),
          runtime->address_space, ApEvent::NO_AP_EVENT);
        result->set_result(reduction_op->identity,
                           reduction_op->sizeof_rhs, false/*own*/);
        return Future(result);
      }
      AllReduceOp *all_reduce_op = runtime->get_available_all_reduce_op();
      Future result = 
        all_reduce_op->initialize(this, future_map, redop, deterministic, prov);
      add_to_dependence_queue(all_reduce_op);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(IndexSpace space,
                                const std::map<DomainPoint,UntypedBuffer> &data,
                                bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      Domain domain;
      runtime->forest->find_launch_space_domain(space, domain);
      return construct_future_map(domain, data, collective, sid, implicit);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(const Domain &domain,
                                const std::map<DomainPoint,UntypedBuffer> &data,
                                bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (data.size() != domain.get_volume())
        REPORT_LEGION_ERROR(ERROR_FUTURE_MAP_COUNT_MISMATCH,
          "The number of buffers passed into a future map construction (%zd) "
          "does not match the volume of the domain (%zd) for the future map "
          "in task %s (UID %lld)", data.size(), domain.get_volume(),
          get_task_name(), get_unique_id())
      const DistributedID did = runtime->get_available_distributed_id();
      FutureMapImpl *impl = new FutureMapImpl(this, runtime, did,
                  runtime->address_space, RtEvent::NO_RT_EVENT);
      LocalReferenceMutator mutator;
      for (std::map<DomainPoint,UntypedBuffer>::const_iterator it =
            data.begin(); it != data.end(); it++)
      {
        if (!domain.contains(it->first))
          REPORT_LEGION_ERROR(ERROR_FUTURE_MAP_COUNT_MISMATCH,
            "Point passed into future map construction is not contained "
            "within the bounds of the domain in task %s (UID %lld)",
            get_task_name(), get_unique_id())
        FutureImpl *future = new FutureImpl(this, runtime, true/*register*/,
            runtime->get_available_distributed_id(), runtime->address_space,
            ApEvent::NO_AP_EVENT);
        future->set_result(it->second.get_ptr(), it->second.get_size(), false);
        impl->set_future(it->first, future, &mutator);
      }
      return FutureMap(impl);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(IndexSpace space,
                                    const std::map<DomainPoint,Future> &futures,
                                    bool internal, bool collective, 
                                    ShardingID sid, bool implicit,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      Domain domain;
      runtime->forest->find_launch_space_domain(space, domain);
      return construct_future_map(domain, futures, internal, collective, sid,
                                  provenance);
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::construct_future_map(const Domain &domain,
                                    const std::map<DomainPoint,Future> &futures,
                                    bool internal, bool collective,
                                    ShardingID sid, bool implicit,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (!internal)
      {
        AutoRuntimeCall call(this);
        if (futures.size() != domain.get_volume())
          REPORT_LEGION_ERROR(ERROR_FUTURE_MAP_COUNT_MISMATCH,
            "The number of futures passed into a future map construction (%zd) "
            "does not match the volume of the domain (%zd) for the future map "
            "in task %s (UID %lld)", futures.size(), domain.get_volume(),
            get_task_name(), get_unique_id())
        return construct_future_map(domain, futures, true/*internal*/,
                                    collective, sid, provenance);
      }
      CreationOp *creation_op = runtime->get_available_creation_op();
      creation_op->initialize_map(this, provenance, futures);
      const DistributedID did = runtime->get_available_distributed_id();
      FutureMapImpl *impl = new FutureMapImpl(this, creation_op, 
          RtEvent::NO_RT_EVENT, runtime, did, runtime->address_space);
      add_to_dependence_queue(creation_op);
      impl->set_all_futures(futures);
      return FutureMap(impl);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::map_region(const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (IS_NO_ACCESS(launcher.requirement))
        return PhysicalRegion();
      MapOp *map_op = runtime->get_available_map_op();
      PhysicalRegion result = map_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Registering a map operation for region "
                    "(%x,%x,%x) in task %s (ID %lld)",
                    launcher.requirement.region.index_space.id, 
                    launcher.requirement.region.field_space.id, 
                    launcher.requirement.region.tree_id, 
                    get_task_name(), get_unique_id());
#endif
      if (current_trace != NULL)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_INLINE_MAPPING_REGION,
                      "Attempted an inline mapping of region "
                      "(%x,%x,%x) inside of trace %d of parent task %s "
                      "(ID %lld). It is illegal to perform inline mapping "
                      "operations inside of traces.",
                      launcher.requirement.region.index_space.id, 
                      launcher.requirement.region.field_space.id, 
                      launcher.requirement.region.tree_id, 
                      current_trace->tid, get_task_name(), get_unique_id())
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
      add_to_dependence_queue(map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent InnerContext::remap_region(const PhysicalRegion &region,
                                       const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.is_mapped())
        return ApEvent::NO_AP_EVENT;
      if (current_trace != NULL)
      {
        const RegionRequirement &req = region.impl->get_requirement();
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_INLINE_MAPPING_REGION,
                      "Attempted an inline mapping of region "
                      "(%x,%x,%x) inside of trace %d of parent task %s "
                      "(ID %lld). It is illegal to perform inline mapping "
                      "operations inside of traces.", req.region.index_space.id,
                      req.region.field_space.id, req.region.tree_id, 
                      current_trace->tid, get_task_name(), get_unique_id())
      }
      MapOp *map_op = runtime->get_available_map_op();
      map_op->initialize(this, region, provenance);
      register_inline_mapped_region(region);
      const ApEvent result = map_op->get_program_order_event();
      add_to_dependence_queue(map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!region.is_mapped())
        return;
      region.impl->unmap_region();
      unregister_inline_mapped_region(region);
    }

    //--------------------------------------------------------------------------
    void InnerContext::unmap_all_regions(bool external)
    //--------------------------------------------------------------------------
    {
      if (external)
      {
        AutoRuntimeCall call(this);
        unmap_all_regions(false);
        return;
      }
      for (std::vector<PhysicalRegion>::const_iterator it = 
            physical_regions.begin(); it != physical_regions.end(); it++)
      {
        if (it->is_mapped())
          it->impl->unmap_region();
      }
      // Also unmap any of our inline mapped physical regions
      AutoLock i_lock(inline_lock);
      for (LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::const_iterator
            it = inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (it->is_mapped())
          it->impl->unmap_region();
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(const FillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.fields.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_EMPTY_FILL_FIELDS,
            "Ignoring fill request with no fields in ask %s (UID %lld)",
            get_task_name(), get_unique_id())
        return;
      }
      FillOp *fill_op = runtime->get_available_fill_op();
      fill_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Registering a fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
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
      add_to_dependence_queue(fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
    }

    //--------------------------------------------------------------------------
    void InnerContext::fill_fields(const IndexFillLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.fields.empty())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_EMPTY_FILL_FIELDS,
            "Ignoring index fill request with no fields in ask %s (UID %lld)",
            get_task_name(), get_unique_id())
        return;
      }
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
        launch_space = find_index_launch_space(launcher.launch_domain,
                                               launcher.provenance);
      IndexFillOp *fill_op = runtime->get_available_index_fill_op();
      fill_op->initialize(this, launcher, launch_space); 
#ifdef DEBUG_LEGION
      log_run.debug("Registering an index fill operation in task %s (ID %lld)",
                     get_task_name(), get_unique_id());
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
      add_to_dependence_queue(fill_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_copy(const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      CopyOp *copy_op = runtime->get_available_copy_op();
      copy_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Registering a copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
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
      add_to_dependence_queue(copy_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
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
        launch_space = find_index_launch_space(launcher.launch_domain,
                                               launcher.provenance);
      IndexCopyOp *copy_op = runtime->get_available_index_copy_op();
      copy_op->initialize(this, launcher, launch_space); 
#ifdef DEBUG_LEGION
      log_run.debug("Registering an index copy operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
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
      add_to_dependence_queue(copy_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_acquire(const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      AcquireOp *acquire_op = runtime->get_available_acquire_op();
      acquire_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing an acquire operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
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
      add_to_dependence_queue(acquire_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
    }

    //--------------------------------------------------------------------------
    void InnerContext::issue_release(const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      ReleaseOp *release_op = runtime->get_available_release_op();
      release_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a release operation in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
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
      add_to_dependence_queue(release_op);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
    }

    //--------------------------------------------------------------------------
    PhysicalRegion InnerContext::attach_resource(const AttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      AttachOp *attach_op = runtime->get_available_attach_op();
      PhysicalRegion result = attach_op->initialize(this, launcher);
      bool parent_conflict = false, inline_conflict = false;
      int index = has_conflicting_regions(attach_op, 
                                          parent_conflict, inline_conflict);
      if (parent_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                      "Attempted an external attach operation on region "
                      "(%x,%x,%x) that conflicts with mapped region " 
                      "(%x,%x,%x) at index %d of parent task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking 'attach_external_resource'.",
                      launcher.handle.index_space.id, 
                      launcher.handle.field_space.id, 
                      launcher.handle.tree_id, 
                      regions[index].region.index_space.id,
                      regions[index].region.field_space.id,
                      regions[index].region.tree_id, index, 
                      get_task_name(), get_unique_id())
      if (inline_conflict)
        REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                      "Attempted an external attach operation on region "
                      "(%x,%x,%x) that conflicts with previous inline "
                      "mapping in task %s (ID %lld) "
                      "that would ultimately result in deadlock. Instead you "
                      "receive this error message. Try unmapping the region "
                      "before invoking 'attach_external_resource'.",
                      launcher.handle.index_space.id, 
                      launcher.handle.field_space.id, launcher.handle.tree_id,
                      get_task_name(), get_unique_id())
      // Add this region to the list of inline mapped regions if it is mapped
      if (result.is_mapped())
        register_inline_mapped_region(result);
      add_to_dependence_queue(attach_op);
      return result;
    }

    //--------------------------------------------------------------------------
    ExternalResources InnerContext::attach_resources(
                                            const IndexAttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (launcher.handles.empty())
        return ExternalResources();
      // This is not control replicated so no need to deduplicate anything
      std::vector<unsigned> indexes(launcher.handles.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        indexes[idx] = idx;
      // Compute the upper bound partition node from this launcher
      RegionTreeNode *node = compute_index_attach_upper_bound(launcher,indexes);
      IndexSpaceNode *launch_space = runtime->forest->get_node(
       find_index_launch_space(Domain(Point<1>(0),Point<1>(indexes.size()-1)),
                               launcher.provenance));
      IndexAttachOp *attach_op = runtime->get_available_index_attach_op();
      ExternalResources result = 
        attach_op->initialize(this, node, launch_space, launcher, indexes);
      const RegionRequirement &req = attach_op->get_requirement();
      bool parent_conflict = false, inline_conflict = false;
      int index = has_conflicting_internal(req,parent_conflict,inline_conflict);
      if (parent_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
          REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                        "Attempted an index attach operation with upper bound "
                        "partition (%x,%x,%x) that conflicts with mapped region"
                        " (%x,%x,%x) at index %d of parent task %s (ID %lld) "
                        "that would ultimately result in deadlock. Instead you "
                        "receive this error message. Try unmapping the region "
                        "before invoking 'attach_external_resources'.",
                        req.partition.index_partition.id, 
                        req.partition.field_space.id, 
                        req.partition.tree_id, 
                        regions[index].region.index_space.id,
                        regions[index].region.field_space.id,
                        regions[index].region.tree_id, index, 
                        get_task_name(), get_unique_id())
        else
          REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                        "Attempted an index attach operation with upper bound "
                        "region (%x,%x,%x) that conflicts with mapped region "
                        "(%x,%x,%x) at index %d of parent task %s (ID %lld) "
                        "that would ultimately result in deadlock. Instead you "
                        "receive this error message. Try unmapping the region "
                        "before invoking 'attach_external_resources'.",
                        req.region.index_space.id, 
                        req.region.field_space.id, 
                        req.region.tree_id, 
                        regions[index].region.index_space.id,
                        regions[index].region.field_space.id,
                        regions[index].region.tree_id, index, 
                        get_task_name(), get_unique_id())
      }
      if (inline_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
          REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                        "Attempted an index attach operation with upper bound "
                        "partition (%x,%x,%x) that conflicts with previous "
                        "inline mapping in task %s (ID %lld) "
                        "that would ultimately result in deadlock. Instead you "
                        "receive this error message. Try unmapping the region "
                        "before invoking 'attach_external_resources'.",
                        req.partition.index_partition.id, 
                        req.partition.field_space.id, req.partition.tree_id,
                        get_task_name(), get_unique_id())
        else
          REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
                        "Attempted an index attach operation with upper bound "
                        "region (%x,%x,%x) that conflicts with previous inline "
                        "mapping in task %s (ID %lld) "
                        "that would ultimately result in deadlock. Instead you "
                        "receive this error message. Try unmapping the region "
                        "before invoking 'attach_external_resources'.",
                        req.region.index_space.id, 
                        req.region.field_space.id, req.region.tree_id,
                        get_task_name(), get_unique_id())
      }
      add_to_dependence_queue(attach_op);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* InnerContext::compute_index_attach_upper_bound(
      const IndexAttachLauncher &launcher, const std::vector<unsigned> &indexes)
    //--------------------------------------------------------------------------
    {
      std::vector<RegionTreeNode*> previous_nodes(indexes.size());
      std::vector<unsigned> depths(indexes.size());
      unsigned max_depth = 0;
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        const unsigned index = indexes[idx];
        LogicalRegion handle = launcher.handles[index];
        if (handle.get_tree_id() != launcher.parent.get_tree_id())
          REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
              "Handle (%d,%d,%d) of index attach operation in parent task %s "
              "(UID %lld) does not come from the same region tree as the "
              "parent region (%d,%d,%d). All regions for an index space "
              "attach must be from the same tree.", handle.index_space.id,
              handle.field_space.id, handle.tree_id, get_task_name(),
              get_unique_id(), launcher.parent.index_space.id,
              launcher.parent.field_space.id, launcher.parent.tree_id)
        previous_nodes[idx] = runtime->forest->get_node(handle);
        depths[idx] = previous_nodes[idx]->get_depth();
        if (max_depth < depths[idx])
          max_depth = depths[idx];
      }
      // Walk all the nodes up from the bottom until they arrive at a 
      // common ancestor, along the way check to make sure that any nodes
      // that arrive at a common join point from two different paths do
      // so at a disjoint partition
      std::vector<RegionTreeNode*> next_nodes(indexes.size());
      while (max_depth > 0)
      {
        std::map<RegionTreeNode*,std::vector<unsigned> > next_to_previous;
        bool all_same = true;
        for (unsigned idx = 0; idx < indexes.size(); idx++)
        {
          if (depths[idx] == max_depth)
          {
            depths[idx]--;
            next_nodes[idx] = previous_nodes[idx]->get_parent();
            next_to_previous[next_nodes[idx]].push_back(idx);
            if (all_same && (idx > 0) && (next_nodes[idx-1] != next_nodes[idx]))
              all_same = false;
          }
          else
          {
            next_nodes[idx] = previous_nodes[idx];
            all_same = false;
          }
        }
        // check to see if all the next to previous cases play by the rules
        for (std::map<RegionTreeNode*,std::vector<unsigned> >::const_iterator
              it = next_to_previous.begin(); it != next_to_previous.end(); it++)
        {
          if (it->second.size() == 1)
            continue;
          // Can skip any disjoint partitions since it doesn't matter where
          // their children came from
          if (!it->first->is_region() &&
              it->first->as_partition_node()->row_source->is_disjoint())
            continue;
          // Otherwise check to see that they all came from the same child
          // If they didn't, then we can't prove tree disjointness
          RegionTreeNode *previous = previous_nodes[it->second.front()];
          for (unsigned idx = 1; idx < it->second.size(); idx++)
          {
            if (previous == previous_nodes[it->second[idx]])
              continue;
            const LogicalRegion h1 = launcher.handles[it->second.front()];
            const LogicalRegion h2 = launcher.handles[it->second[idx]];
            REPORT_LEGION_ERROR(ERROR_ATTEMPTED_EXTERNAL_ATTACH,
              "Logical region handle (%d,%d,%d) from index %d of index attach "
              "operation in parent task %s (UID %lld) is not region-tree "
              "disjoint with logical region handle (%d,%d,%d) from index %d. "
              "All regions in index space attach operations must be "
              "region-tree disjoint.", h1.index_space.id,
              h1.field_space.id, h1.tree_id, it->second.front(),
              get_task_name(), get_unique_id(), h2.index_space.id,
              h2.field_space.id, h2.tree_id, it->second[idx])
          }
        }
        previous_nodes.swap(next_nodes);
        if (all_same)
          break;
        max_depth--;
      }
      // At this point all the previous nodes should be the same
      return previous_nodes.back();
    }

    //--------------------------------------------------------------------------
    ProjectionID InnerContext::compute_index_attach_projection(
                    IndexTreeNode *upper_bound, std::vector<IndexSpace> &spaces)
    //--------------------------------------------------------------------------
    {
      
      std::map<IndexTreeNode*,std::vector<AttachProjectionFunctor*> >::iterator
        finder = attach_functions.find(upper_bound);
      if (finder != attach_functions.end())
      {
        for (std::vector<AttachProjectionFunctor*>::const_iterator it =
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if ((*it)->handles.size() != spaces.size())
            continue;
          bool equal = true;
          for (unsigned idx = 0; idx < spaces.size(); idx++)
          {
            if ((*it)->handles[idx] == spaces[idx])
              continue;
            equal = false;
            break;
          }
          if (equal)
            return (*it)->pid;
        }
      }
      else // instantiate the entry in the map
        finder = attach_functions.insert(std::make_pair(upper_bound,
              std::vector<AttachProjectionFunctor*>())).first;
      // If the upper bound is a partition, do a quick check to see if
      // all the spaces are immediate children of the upper bound, if
      // so then we can use projection function 0
      if (!upper_bound->is_index_space_node())
      {
        bool all_children = true;
        IndexPartNode *parent = upper_bound->as_index_part_node();
        for (std::vector<IndexSpace>::const_iterator it =
              spaces.begin(); it != spaces.end(); it++)
        {
          IndexSpaceNode *child = runtime->forest->get_node(*it);
          if (child->parent == parent)
            continue;
          all_children = false;
          break;
        }
        if (all_children)
        {
          // We can use the identity projection in this case
          // so just make it, but no need to register it with the runtime
          finder->second.push_back(
              new AttachProjectionFunctor(0/*identity*/, std::move(spaces)));
          return 0;
        }
      }
      // If we get here then we need to make it
      // Generate a fresh dynamic ID and store it
      const ProjectionID result =
        runtime->generate_dynamic_projection_id(false/*check context*/);
      AttachProjectionFunctor *functor =
        new AttachProjectionFunctor(result, std::move(spaces));
      runtime->register_projection_functor(result, functor, false/*check*/,
                                           true/*silence warnings*/);
      finder->second.push_back(functor);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_projection_function(result, functor->get_depth(),
                                           functor->is_invertible());
      return result;
    }

    //--------------------------------------------------------------------------
    InnerContext::AttachProjectionFunctor::AttachProjectionFunctor(
                               ProjectionID p, std::vector<IndexSpace> &&spaces)
      : handles(spaces), pid(p)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::AttachProjectionFunctor::project(
      LogicalRegion upper_bound, const DomainPoint &point, const Domain &launch)
    //--------------------------------------------------------------------------
    {
      const Point<1> p = point;
      return runtime->get_logical_subregion_by_tree(handles[p[0]],
          upper_bound.get_field_space(), upper_bound.get_tree_id());
    }

    //--------------------------------------------------------------------------
    LogicalRegion InnerContext::AttachProjectionFunctor::project(
         LogicalPartition upper, const DomainPoint &point, const Domain &launch)
    //--------------------------------------------------------------------------
    {
      const Point<1> p = point;
      return runtime->get_logical_subregion_by_tree(handles[p[0]],
                    upper.get_field_space(), upper.get_tree_id());
    }

    //--------------------------------------------------------------------------
    Future InnerContext::detach_resource(PhysicalRegion region,
                 const bool flush, const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Unmap the region here so that it is safe for re-use
      if (region.impl->is_mapped())
      {
        region.impl->unmap_region();
        // Remove this region from the list of unmapped regions if it is mapped
        unregister_inline_mapped_region(region);
      }
      DetachOp *op = runtime->get_available_detach_op();
      Future result =
        op->initialize_detach(this, region, flush, unordered, provenance);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered detach operation performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::detach_resources(ExternalResources resources,
                 const bool flush, const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (resources.impl == NULL)
        return Future();
      IndexDetachOp *op = runtime->get_available_index_detach_op();
      Future result =
        resources.impl->detach(this, op, flush, unordered, provenance);
      if (!add_to_dependence_queue(op, unordered))
      {
#ifdef DEBUG_LEGION
        assert(unordered);
#endif
        REPORT_LEGION_ERROR(ERROR_POST_EXECUTION_UNORDERED_OPERATION,
            "Illegal unordered index detach operation performed after task %s "
            "(UID %lld) has finished executing. All unordered operations must "
            "be performed before the end of the execution of the parent task.",
            get_task_name(), get_unique_id())
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::progress_unordered_operations(void)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      RtEvent precondition;
      Operation *op = NULL;
      {
        AutoLock d_lock(dependence_lock);
        // If we have any unordered ops and we're not in the middle of
        // a trace then add them into the queue
        if (!unordered_ops.empty() && (current_trace == NULL))
          insert_unordered_ops(d_lock);
        if (dependence_queue.empty())
          return;
        if (!outstanding_dependence)
        {
          issue_task = true;
          outstanding_dependence = true;
          precondition = dependence_precondition;
          dependence_precondition = RtEvent::NO_RT_EVENT;
          op = dependence_queue.front();
        }
      }
      if (issue_task)
      {
        DependenceArgs args(op, this);
        const LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY;
        runtime->issue_runtime_meta_task(args, priority, precondition); 
      }
    }

    //--------------------------------------------------------------------------
    FutureMap InnerContext::execute_must_epoch(
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
#ifdef SAFE_MUST_EPOCH_LAUNCHES
      // Must epoch launches can sometimes block on external resources which
      // Realm does not know about. In theory this can lead to deadlock, so
      // we provide this mechanism for ordering must epoch launches. By 
      // inserting an execution fence before every must epoch launche we
      // guarantee that it is ordered with respect to every other one. This
      // is heavy-handed for sure, but it is effective and sound and gives
      // us the property that we want until someone comes up with a use
      // case that proves that they need something better.
      // See github issue #659
      issue_execution_fence();
#endif
      AutoRuntimeCall call(this);
      MustEpochOp *epoch_op = runtime->get_available_epoch_op();
      FutureMap result = epoch_op->initialize(this, launcher);
#ifdef DEBUG_LEGION
      log_run.debug("Executing a must epoch in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
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
      add_to_dependence_queue(epoch_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions,
                               launcher.provenance.c_str());
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
      add_to_dependence_queue(timing_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::select_tunable_value(const TunableLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a tunable request in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      TunableOp *tunable_op = runtime->get_available_tunable_op();
      Future result = tunable_op->initialize(this, launcher);
      add_to_dependence_queue(tunable_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_mapping_fence(const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FenceOp *fence_op = runtime->get_available_fence_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a mapping fence in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      Future f = fence_op->initialize(this, FenceOp::MAPPING_FENCE,
                                      true/*return future*/, provenance);
      add_to_dependence_queue(fence_op);
      return f;
    }

    //--------------------------------------------------------------------------
    Future InnerContext::issue_execution_fence(const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FenceOp *fence_op = runtime->get_available_fence_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing an execution fence in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      Future f = fence_op->initialize(this, FenceOp::EXECUTION_FENCE,
                                      true/*return future*/, provenance);
      add_to_dependence_queue(fence_op);
      return f; 
    }

    //--------------------------------------------------------------------------
    void InnerContext::complete_frame(const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      FrameOp *frame_op = runtime->get_available_frame_op();
#ifdef DEBUG_LEGION
      log_run.debug("Issuing a frame in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      frame_op->initialize(this, provenance);
      add_to_dependence_queue(frame_op);
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::create_predicate(const Future &f,
                                             const char *provenance)
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
      pred_op->initialize(this, f, provenance);
      add_to_dependence_queue(pred_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Predicate InnerContext::predicate_not(const Predicate &p,
                                          const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      NotPredOp *pred_op = runtime->get_available_not_pred_op();
      // Hold a reference before initialization
      Predicate result(pred_op);
      pred_op->initialize(this, p, provenance);
      add_to_dependence_queue(pred_op);
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
        pred_op->initialize(this, actual_predicates, launcher.provenance);
        add_to_dependence_queue(pred_op);
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
        pred_op->initialize(this, actual_predicates, launcher.provenance);
        add_to_dependence_queue(pred_op);
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
        Future result = runtime->help_create_future(this, ApEvent::NO_AP_EVENT);
        const bool value = true;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
        return result;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        Future result = runtime->help_create_future(this, ApEvent::NO_AP_EVENT);
        const bool value = false;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
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
    size_t InnerContext::register_new_child_operation(Operation *op,
                      const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      // If we are performing a trace mark that the child has a trace
      if (current_trace != NULL)
        op->set_trace(current_trace, dependences);
      size_t result = total_children_count++;
      const size_t outstanding_count = 
        outstanding_children_count.fetch_add(1) + 1;
      // Need to check if we are not tracing by frames
      // Also, do not perform window waits if we are in the middle of a 
      // physical trace because we might deadlock if the trace is bigger
      // than the size of our window.
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count > context_configuration.max_window_size) &&
            !is_replaying_physical_trace())
        perform_window_wait();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_child_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id()); 
      return result;
    }

    //--------------------------------------------------------------------------
    void InnerContext::insert_unordered_ops(AutoLock &d_lock)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unordered_ops.empty());
      assert(current_trace == NULL);
#endif
      // We need the child op lock here so we can add these to this
      // list of executing children as well
      AutoLock child_lock(child_op_lock);
      for (std::vector<Operation*>::const_iterator it = 
            unordered_ops.begin(); it != unordered_ops.end(); it++)
      {
        (*it)->set_tracking_parent(total_children_count++);
#ifdef DEBUG_LEGION
        assert(executing_children.find(*it) == executing_children.end());
        assert(executed_children.find(*it) == executed_children.end());
        assert(complete_children.find(*it) == complete_children.end());
        outstanding_children[(*it)->get_ctx_index()] = (*it);
#endif       
        executing_children[*it] = (*it)->get_generation();
        dependence_queue.push_back(*it);
      }
      outstanding_children_count.fetch_add(unordered_ops.size());
      unordered_ops.clear();
    }

    //--------------------------------------------------------------------------
    size_t InnerContext::register_new_close_operation(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      // For now we just bump our counter
      size_t result = total_close_count++;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_close_operation_index(get_context_uid(), result, 
                                             op->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    size_t InnerContext::register_new_summary_operation(TraceSummaryOp *op)
    //--------------------------------------------------------------------------
    {
      // For now we just bump our counter
      size_t result = total_summary_count++;
      const size_t outstanding_count = 
        outstanding_children_count.fetch_add(1) + 1;
      // Need to check if we are not tracing by frames
      // Also, do not perform window waits if we are in the middle of a 
      // physical trace because we might deadlock if the trace is bigger
      // than the size of our window.
      if ((context_configuration.min_frames_to_schedule == 0) && 
          (context_configuration.max_window_size > 0) && 
            (outstanding_count > context_configuration.max_window_size) &&
            !is_replaying_physical_trace())
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
        const int outstanding_count = outstanding_children_count.fetch_sub(1);
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
      outstanding_children_count.fetch_add(1);
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
        add_reference();
        PrepipelineArgs args(op, this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY);
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_prepipeline_stage(void)
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
        // Reference keeps flowing with the continuation
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::add_to_dependence_queue(Operation *op, bool unordered)
    //--------------------------------------------------------------------------
    {
      // Launch the task to perform the prepipeline stage for the operation
      if (op->has_prepipeline_stage())
        add_to_prepipeline_queue(op);
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
      ApEvent term_event;
      // We disable program order execution when we are replaying a
      // physical trace since it might not be sound to block
      if (runtime->program_order_execution && !unordered && 
          !is_replaying_physical_trace())
        term_event = op->get_program_order_event();
      {
        AutoLock d_lock(dependence_lock);
        if (unordered)
        {
          if (finished_execution)
            return false;
          // If this is unordered, stick it on the list of 
          // unordered ops to be added later and then we're done
          unordered_ops.push_back(op);
          return true;
        }
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
        // If we have any unordered ops and we're not in the middle of
        // a trace then add them into the queue
        if (!unordered_ops.empty() && (current_trace == NULL))
          insert_unordered_ops(d_lock);
      }
      if (issue_task)
      {
        DependenceArgs args(op, this);
        runtime->issue_runtime_meta_task(args, priority, precondition); 
      }
      if (term_event.exists()) 
      {
        begin_task_wait(true/*from runtime*/);
        bool poisoned = false;
        term_event.wait_faultaware(poisoned);
        if (poisoned)
          raise_poison_exception();
        end_task_wait();
      }
      return true;
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
    template<typename T, typename ARGS, bool HAS_BOUND>
    void InnerContext::add_to_queue(QueueEntry<T> entry, LocalLock &lock,
                  std::list<QueueEntry<T> > &queue, CompletionQueue &comp_queue)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(entry.ready.exists());
#endif
      bool issue_task = false;
      RtEvent precondition;
      {
        AutoLock l_lock(lock);
        // Issue a task if there isn't one running right now
        if (queue.empty())
        {
          issue_task = true;
          // Add a reference to the context the first time we defer this
          add_reference();
          // Make the queue the first time if necessary
          if (!comp_queue.exists())
            comp_queue = CompletionQueue::create_completion_queue(
                HAS_BOUND ? context_configuration.max_window_size : 0);
        }
        queue.push_back(entry);
        comp_queue.add_event(entry.ready);
        if (issue_task)
          precondition = RtEvent(comp_queue.get_nonempty_event());
      }
      if (issue_task)
      {
        ARGS args(entry.op, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_ready_queue(Operation *op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,TriggerReadyArgs,true/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), ready_lock,
          ready_queue, ready_comp_queue);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    T InnerContext::process_queue(LocalLock &lock, RtEvent &next_ready, 
                                  std::list<QueueEntry<T> > &queue,
                                  CompletionQueue &comp_queue,
                                  std::vector<T> &to_perform) const
    //--------------------------------------------------------------------------
    {
      T next{};
      const size_t vector_width = context_configuration.meta_task_vector_width;
      to_perform.reserve(vector_width);
      AutoLock l_lock(lock);
      std::vector<RtEvent> ready_events(vector_width);
      size_t num_ready =
        comp_queue.pop_events(&ready_events.front(), vector_width);
      // Realm permits spurious wake-ups sometimes on completion queues where
      // no events are actually ready. The number of times this can happen is
      // bounded by the number of events that are added into the queue so we
      // don't need to worry about indefinite starvation.
      if (num_ready > 0)
      {
        ready_events.resize(num_ready);
        std::sort(ready_events.begin(), ready_events.end());
        // Find the entries
        for (typename std::list<QueueEntry<T> >::iterator it =
              queue.begin(); it != queue.end(); /*nothing*/)
        {
          std::vector<RtEvent>::iterator finder = 
            std::lower_bound(ready_events.begin(),ready_events.end(),it->ready);
          if ((finder != ready_events.end()) && (*finder == it->ready))
          {
            to_perform.push_back(it->op);
            it = queue.erase(it);
            ready_events.erase(finder);
            if (ready_events.empty())
              break;
          }
          else
            it++;
        }
#ifdef DEBUG_LEGION
        assert(ready_events.empty());
#endif
      }
      if (!queue.empty())
      {
        next_ready = RtEvent(comp_queue.get_nonempty_event());
        next = queue.front().op;
      }
      return next;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_ready_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(ready_lock, precondition,
                                 ready_queue, ready_comp_queue, to_perform);
      if (next != NULL)
      {
        TriggerReadyArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->trigger_ready();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_task_queue(TaskOp *task, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<TaskOp*,DeferredEnqueueTaskArgs,false/*has bounds*/>(
          QueueEntry<TaskOp*>(task, ready), enqueue_task_lock,
          enqueue_task_queue, enqueue_task_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_enqueue_task_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<TaskOp*> to_perform;
      TaskOp *next = process_queue<TaskOp*>(enqueue_task_lock, precondition,
                    enqueue_task_queue, enqueue_task_comp_queue, to_perform);
      if (next != NULL)
      {
        DeferredEnqueueTaskArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<TaskOp*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->enqueue_ready_task(false/*use target*/);
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_distribute_task_queue(TaskOp *task, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<TaskOp*,DeferredDistributeTaskArgs,false/*has bounds*/>(
          QueueEntry<TaskOp*>(task, ready), distribute_task_lock,
          distribute_task_queue, distribute_task_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_distribute_task_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<TaskOp*> to_perform;
      TaskOp *next = process_queue<TaskOp*>(distribute_task_lock, precondition,
                distribute_task_queue, distribute_task_comp_queue, to_perform);
      if (next != NULL)
      {
        DeferredDistributeTaskArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<TaskOp*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        if ((*it)->distribute_task())
          (*it)->launch_task();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_launch_task_queue(TaskOp *task, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<TaskOp*,DeferredLaunchTaskArgs,false/*has bounds*/>(
          QueueEntry<TaskOp*>(task, ready), launch_task_lock,
          launch_task_queue, launch_task_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_launch_task_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<TaskOp*> to_perform;
      TaskOp *next = process_queue<TaskOp*>(launch_task_lock, precondition,
                    launch_task_queue, launch_task_comp_queue, to_perform);
      if (next != NULL)
      {
        DeferredLaunchTaskArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<TaskOp*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->launch_task();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_resolution_queue(Operation *op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,TriggerResolutionArgs,true/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), resolution_lock,
          resolution_queue, resolution_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_resolution_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(resolution_lock, precondition,
                           resolution_queue, resolution_comp_queue, to_perform);
      if (next != NULL)
      {
        TriggerResolutionArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->trigger_resolution();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_trigger_execution_queue(Operation *op, 
                                                      RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,TriggerExecutionArgs,false/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), trigger_execution_lock,
          trigger_execution_queue, trigger_execution_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_trigger_execution_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(trigger_execution_lock,
          precondition, trigger_execution_queue, 
          trigger_execution_comp_queue, to_perform);
      if (next != NULL)
      {
        TriggerExecutionArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->trigger_execution();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_execution_queue(Operation *op,
                                                       RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,DeferredExecutionArgs,false/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), deferred_execution_lock,
          deferred_execution_queue, deferred_execution_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_execution_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(deferred_execution_lock,
          precondition, deferred_execution_queue, 
          deferred_execution_comp_queue, to_perform);
      if (next != NULL)
      {
        DeferredExecutionArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->complete_execution();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_trigger_completion_queue(Operation *op,
                                                       RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,TriggerCompletionArgs,false/*has bounds*/>(
          QueueEntry<Operation*>(op, ready),
          trigger_completion_lock, trigger_completion_queue,
          trigger_completion_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_trigger_completion_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(trigger_completion_lock,
          precondition, trigger_completion_queue,
          trigger_completion_comp_queue, to_perform);
      if (next != NULL)
      {
        TriggerCompletionArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->trigger_complete();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_completion_queue(Operation *op,
                                                        RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,DeferredCompletionArgs,false/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), deferred_completion_lock,
          deferred_completion_queue, deferred_completion_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_completion_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(deferred_completion_lock,
          precondition, deferred_completion_queue,
          deferred_completion_comp_queue, to_perform);
      if (next != NULL)
      {
        DeferredCompletionArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->complete_operation();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_trigger_commit_queue(Operation *op, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      add_to_queue<Operation*,TriggerCommitArgs,false/*has bounds*/>(
          QueueEntry<Operation*>(op, ready), trigger_commit_lock,
          trigger_commit_queue, trigger_commit_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_trigger_commit_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<Operation*> to_perform;
      Operation *next = process_queue<Operation*>(trigger_commit_lock,
          precondition, trigger_commit_queue,
          trigger_commit_comp_queue, to_perform);
      if (next != NULL)
      {
        TriggerCommitArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<Operation*>::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        (*it)->trigger_commit();
      return (next == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_deferred_commit_queue(Operation *op,
                                                 RtEvent ready, bool deactivate)
    //--------------------------------------------------------------------------
    {
      add_to_queue<std::pair<Operation*,bool>,
          DeferredCommitArgs,false/*has bounds*/>(
            QueueEntry<std::pair<Operation*,bool> >(std::pair<Operation*,bool>(
              op, deactivate), ready), deferred_commit_lock,
          deferred_commit_queue, deferred_commit_comp_queue);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_deferred_commit_queue(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      std::vector<std::pair<Operation*,bool> > to_perform;
      std::pair<Operation*,bool> next =
        process_queue<std::pair<Operation*,bool> >(deferred_commit_lock,
          precondition, deferred_commit_queue, 
          deferred_commit_comp_queue, to_perform);
      if (next.first != NULL)
      {
        DeferredCommitArgs args(next, this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_WORK_PRIORITY, precondition);
      }
      for (std::vector<std::pair<Operation*,bool> >::const_iterator it =
            to_perform.begin(); it != to_perform.end(); it++)
        it->first->commit_operation(it->second);
      return (next.first == NULL);
    }

    //--------------------------------------------------------------------------
    void InnerContext::add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                              const void *result, size_t size, 
                                              PhysicalInstance inst,
                                              FutureFunctor *callback_functor,
                                              bool own_functor)
    //--------------------------------------------------------------------------
    {
      bool issue_task = false;
      RtEvent precondition;
      const size_t task_index = ctx->get_owner_task()->get_context_index();
      {
        AutoLock p_lock(post_task_lock);
        // Issue a task if there isn't one running right now
        if (post_task_queue.empty())
        {
          issue_task = true;
          // Add a reference to the context the first time we defer this
          add_reference();
        }
        post_task_queue.push_back(PostTaskArgs(ctx, task_index, result, size, 
              inst, wait_on, callback_functor, own_functor));
        // If we've already got a completion queue then use it
        if (post_task_comp_queue.exists())
          post_task_comp_queue.add_event(wait_on);
        if (issue_task)
        {
          if (!post_task_comp_queue.exists())
          {
            // Find the one with the minimum index
            size_t min_index = 0;
            for (std::list<PostTaskArgs>::const_iterator it = 
                  post_task_queue.begin(); it != post_task_queue.end(); it++)
            {
              if (!precondition.exists() || (it->index < min_index))
              {
                precondition = it->wait_on;
                min_index = it->index;
              }
            }
          }
          else
            precondition = RtEvent(post_task_comp_queue.get_nonempty_event());
        }
      }
      if (issue_task)
      {
        // Other things could be added to the queue by the time we're here
        PostEndArgs args(ctx->owner_task, this);
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    bool InnerContext::process_post_end_tasks(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      TaskContext *next_ctx = NULL;
      std::vector<PostTaskArgs> to_perform;
      to_perform.reserve(context_configuration.meta_task_vector_width);
      {
        std::vector<RtEvent> ready_events(
                          context_configuration.meta_task_vector_width);
        AutoLock p_lock(post_task_lock);
        // Ask the completion queue for the ready events
        size_t num_ready = 0;
        if (!post_task_comp_queue.exists())
        {
          // No completion queue so go through and do this manually
          for (std::list<PostTaskArgs>::const_iterator it = 
                post_task_queue.begin(); it != post_task_queue.end(); it++)
          {
            if (it->wait_on.has_triggered())
            {
              ready_events[num_ready++] = it->wait_on;
              if (num_ready == ready_events.size())
                break;
            }
          }
        }
        else // We can just use the comp queue to get the ready events
          num_ready = post_task_comp_queue.pop_events(
            &ready_events.front(), ready_events.size());
        // Realm permits spurious wake-ups sometimes on completion queues where
        // no events are actually ready. The number of times this can happen is
        // bounded by the number of events that are added into the queue so we
        // don't need to worry about indefinite starvation.
        if (num_ready > 0)
        {
          // Find all the entries for all the ready events
          for (std::list<PostTaskArgs>::iterator it = post_task_queue.begin();
                it != post_task_queue.end(); /*nothing*/)
          {
            bool found = false;
            for (unsigned idx = 0; idx < num_ready; idx++)
            {
              if (it->wait_on == ready_events[idx])  
              {
                found = true;
                break;
              }
            }
            if (found)
            {
              to_perform.push_back(*it);
              it = post_task_queue.erase(it);
              // Check to see if we're done early
              if (to_perform.size() == num_ready)
                break;
            }
            else
              it++;
          }
        }
        if (!post_task_queue.empty())
        {
          if (!post_task_comp_queue.exists())
          {
            // See if we want to switch over to using a completion queue
            if (post_task_queue.size() < 
                  context_configuration.meta_task_vector_width)
            {
              // Find the one with the minimum index
              size_t min_index = 0;
              for (std::list<PostTaskArgs>::const_iterator it = 
                    post_task_queue.begin(); it != post_task_queue.end(); it++)
              {
                if (!precondition.exists() || (it->index < min_index))
                {
                  precondition = it->wait_on;
                  min_index = it->index;
                  next_ctx = it->context;
                }
              }
            }
            else
            {
              // Switch over to using a completion queue
              post_task_comp_queue =
                CompletionQueue::create_completion_queue(0);
              // Fill in the completion queue with events
              for (std::list<PostTaskArgs>::const_iterator it = 
                    post_task_queue.begin(); it != post_task_queue.end(); it++)
                post_task_comp_queue.add_event(it->wait_on);
              precondition = RtEvent(post_task_comp_queue.get_nonempty_event());
              next_ctx = post_task_queue.front().context;
            }
          }
          else
          {
            precondition = RtEvent(post_task_comp_queue.get_nonempty_event());
            next_ctx = post_task_queue.front().context;
          }
#ifdef DEBUG_LEGION
          assert(next_ctx != NULL);
#endif
        }
      }
      // Launch this first to get it in flight so it can run when ready
      if (next_ctx != NULL)
      {
        PostEndArgs args(next_ctx->owner_task, this);
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      }
      // Now perform our operations
      if (!to_perform.empty())
      {
        // Sort these into order by their index before we perform them 
        // so we do them in order or we could risk a hang
        std::sort(to_perform.begin(), to_perform.end());
        for (std::vector<PostTaskArgs>::const_iterator it = 
              to_perform.begin(); it != to_perform.end(); it++)
        {
          if (it->instance.exists())
          {
#ifdef LEGION_MALLOC_INSTANCES
            // Need to keep the context alive until we release the instance
            it->context->add_reference();
#endif
            it->context->post_end_task(it->result,it->size,false/*owned*/,NULL);
#ifdef LEGION_MALLOC_INSTANCES
            it->context->release_future_local_instance(it->instance); 
            if (it->context->remove_reference())
              delete it->context;
#endif
            // Once we've copied the data then we can destroy the instance
            it->instance.destroy();
          }
          else if (it->functor != NULL)
            it->context->post_end_task(NULL, 0, it->owned, it->functor);
          else
            it->context->post_end_task(it->result,it->size,true/*owned*/,NULL);
        }
      }
      // If we didn't launch a next op, then we can remove the reference
      return (next_ctx == NULL);
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
        int outstanding_count = outstanding_children_count.fetch_sub(1) - 1;
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
      std::set<ApEvent> child_completion_events;
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
        if (task_executed && (owner_task != NULL) && executing_children.empty()
            && executed_children.empty() && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
          for (LegionMap<Operation*,GenerationID,
                COMPLETE_CHILD_ALLOC>::const_iterator it =
                complete_children.begin(); it != complete_children.end(); it++)
            child_completion_events.insert(it->first->get_completion_event());
        }
      }
      if (needs_trigger)
      {
        if (!child_completion_events.empty())
          owner_task->trigger_children_complete(
            Runtime::merge_events(NULL, child_completion_events));
        else
          owner_task->trigger_children_complete(ApEvent::NO_AP_EVENT);
      }
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
    int InnerContext::has_conflicting_regions(MapOp *op, bool &parent_conflict,
                                              bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = op->get_requirement(); 
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int InnerContext::has_conflicting_regions(AttachOp *attach,
                                              bool &parent_conflict,
                                              bool &inline_conflict)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = attach->get_requirement();
      return has_conflicting_internal(req, parent_conflict, inline_conflict);
    }

    //--------------------------------------------------------------------------
    int InnerContext::has_conflicting_internal(const RegionRequirement &req,
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
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock,1,false/*exclusive*/);
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_LEGION
        // This better be true for a single task
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
    void InnerContext::find_conflicting_regions(TaskOp *task,
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
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock,1,false/*exclusive*/);
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_LEGION
        // This better be true for a single task
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
    void InnerContext::find_conflicting_regions(CopyOp *copy,
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
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->src_indirect_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->src_indirect_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->dst_indirect_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->dst_indirect_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        if (has_conflict)
          conflicting.push_back(physical_regions[our_idx]);
      }
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock,1,false/*exclusive*/);
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_LEGION
        // This better be true for a single task
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
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
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->src_indirect_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->src_indirect_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        for (unsigned idx = 0; !has_conflict &&
              (idx < copy->dst_indirect_requirements.size()); idx++)
        {
          const RegionRequirement &req = copy->dst_indirect_requirements[idx];
          if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
            has_conflict = true;
        }
        if (has_conflict)
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(AcquireOp *acquire,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = acquire->get_requirement();
      find_conflicting_internal(req, conflicting); 
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(ReleaseOp *release,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = release->get_requirement();
      find_conflicting_internal(req, conflicting);      
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(DependentPartitionOp *partition,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = partition->get_requirement();
      find_conflicting_internal(req, conflicting);
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_internal(const RegionRequirement &req,
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
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(physical_regions[our_idx]);
      }
      // Need lock here because of unordered detach operations
      AutoLock i_lock(inline_lock,1,false/*exclusive*/);
      for (std::list<PhysicalRegion>::const_iterator it = 
            inline_regions.begin(); it != inline_regions.end(); it++)
      {
        if (!it->is_mapped())
          continue;
        const RegionRequirement &our_req = it->impl->get_requirement();
#ifdef DEBUG_LEGION
        // This better be true for a single task
        assert(our_req.handle_type == LEGION_SINGULAR_PROJECTION);
#endif
        RegionTreeID our_tid = our_req.region.get_tree_id();
        IndexSpace our_space = our_req.region.get_index_space();
        RegionUsage our_usage(our_req);
        if (check_region_dependence(our_tid,our_space,our_req,our_usage,req))
          conflicting.push_back(*it);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::find_conflicting_regions(FillOp *fill,
                                       std::vector<PhysicalRegion> &conflicting)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FIND_CONFLICTING_CALL);
      const RegionRequirement &req = fill->get_requirement();
      find_conflicting_internal(req, conflicting);
    } 

    //--------------------------------------------------------------------------
    void InnerContext::register_inline_mapped_region(
                                                   const PhysicalRegion &region)
    //--------------------------------------------------------------------------
    {
      // Because of 'remap_region', this method can be called
      // both for inline regions as well as regions which were
      // initally mapped for the task.  Do a quick check to see
      // if it was an original region.  If it was then we're done.
      for (unsigned idx = 0; idx < physical_regions.size(); idx++)
      {
        if (physical_regions[idx].impl == region.impl)
          return;
      }
      // Need lock because of unordered detach operations
      AutoLock i_lock(inline_lock);
      inline_regions.push_back(region);
    }

    //--------------------------------------------------------------------------
    void InnerContext::unregister_inline_mapped_region(
                                                   const PhysicalRegion &region)
    //--------------------------------------------------------------------------
    {
      // Need lock because of unordered detach operations
      AutoLock i_lock(inline_lock);
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
    ApEvent InnerContext::register_implicit_dependences(Operation *op)
    //--------------------------------------------------------------------------
    {
      // If there are any outstanding unmapped dependent partition operations
      // outstanding then we might have an implicit dependence on its execution
      // so we always record a dependence on it
      if (last_implicit != NULL)
      {
#ifdef LEGION_SPY
        // Can't prune when doing legion spy
        op->register_dependence(last_implicit, last_implicit_gen);
#else
        if (op->register_dependence(last_implicit, last_implicit_gen))
          last_implicit = NULL;
#endif
      }
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
      previous_completion_events.insert(op->get_program_order_event());
      // Periodically merge these to keep this data structure from exploding
      // when we have a long-running task, although don't do this for fence
      // operations in case we have to prune ourselves out of the set
      if (previous_completion_events.size() >= LEGION_DEFAULT_MAX_TASK_WINDOW)
      {
        const Operation::OpKind op_kind = op->get_operation_kind(); 
        if ((op_kind != Operation::FENCE_OP_KIND) &&
            (op_kind != Operation::FRAME_OP_KIND) &&
            (op_kind != Operation::DELETION_OP_KIND) &&
            (op_kind != Operation::TRACE_BEGIN_OP_KIND) && 
            (op_kind != Operation::TRACE_COMPLETE_OP_KIND) &&
            (op_kind != Operation::TRACE_CAPTURE_OP_KIND) &&
            (op_kind != Operation::TRACE_REPLAY_OP_KIND) &&
            (op_kind != Operation::TRACE_SUMMARY_OP_KIND))
        {
          const ApEvent merge = 
            Runtime::merge_events(NULL, previous_completion_events);
          previous_completion_events.clear();
          previous_completion_events.insert(merge);
        }
      }
      // Have to record this operation in case there is a fence later
      ops_since_last_fence.push_back(op->get_unique_op_id());
      return current_execution_fence_event;
#else
      if (current_execution_fence_event.exists())
      {
        // We can't have internal operations pruning out fences
        // because we can't test if they are memoizing or not
        // Their 'get_memoizable' method will always return NULL
        bool poisoned = false;
        if (current_execution_fence_event.has_triggered_faultaware(poisoned))
        {
          if (poisoned)
            raise_poison_exception();
          if (!op->is_internal_op())
          {
            // We can only do this optimization safely if we're not 
            // recording a physical trace, otherwise the physical
            // trace needs to see this dependence
            Memoizable *memo = op->get_memoizable();
            if ((memo == NULL) || !memo->is_recording())
              current_execution_fence_event = ApEvent::NO_AP_EVENT;
          }
        }
        return current_execution_fence_event;
      }
      return ApEvent::NO_AP_EVENT;
#endif
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_fence_analysis(Operation *op, 
               std::set<ApEvent> &previous_events, bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      {
        const Operation::OpKind op_kind = op->get_operation_kind();
        // It's alright if you hit this assertion for a new operation kind
        // Just add the new operation kind here and then update the check
        // in register_implicit_dependences that looks for all these kinds too
        // so that we do not run into trouble when running with Legion Spy.
        assert((op_kind == Operation::FENCE_OP_KIND) || 
               (op_kind == Operation::FRAME_OP_KIND) || 
               (op_kind == Operation::DELETION_OP_KIND) ||
               (op_kind == Operation::TRACE_BEGIN_OP_KIND) ||
               (op_kind == Operation::TRACE_COMPLETE_OP_KIND) ||
               (op_kind == Operation::TRACE_CAPTURE_OP_KIND) ||
               (op_kind == Operation::TRACE_REPLAY_OP_KIND) ||
               (op_kind == Operation::TRACE_SUMMARY_OP_KIND));
      }
#endif
      std::map<Operation*,GenerationID> previous_operations;
      // Take the lock and iterate through our current pending
      // operations and find all the ones with a context index
      // that is less than the index for the fence operation
      const size_t next_fence_index = op->get_ctx_index();
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
          const size_t op_index = it->first->get_ctx_index();
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
          const size_t op_index = it->first->get_ctx_index();
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
          const size_t op_index = it->first->get_ctx_index();
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
          const size_t op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_program_order_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_program_order_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's older than the previous fence then we don't care
          if (op_index < current_execution_fence_index)
            continue;
          // Record a dependence if it didn't come after our fence
          if (op_index < next_fence_index)
            previous_events.insert(it->first->get_program_order_event());
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
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_program_order_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              executed_children.begin(); it != executed_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_program_order_event());
        }
        for (std::map<Operation*,GenerationID>::const_iterator it = 
              complete_children.begin(); it != complete_children.end(); it++)
        {
          if (it->first->get_generation() != it->second)
            continue;
          const size_t op_index = it->first->get_ctx_index();
          // If it's younger than our fence we don't care
          if (op_index >= next_fence_index)
            continue;
          if (op_index >= current_mapping_fence_index)
            previous_operations.insert(*it);
          if (op_index >= current_execution_fence_index)
            previous_events.insert(it->first->get_program_order_event());
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
        previous_events.erase(op->get_program_order_event());
      }
#endif
      // Also include the current execution fence in case the operation
      // already completed and wasn't in the set, make sure to do this
      // before we update the current fence
      if (execution && current_execution_fence_event.exists())
        previous_events.insert(current_execution_fence_event);
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
    void InnerContext::update_current_implicit(Operation *op)
    //--------------------------------------------------------------------------
    {
      // Just overwrite since we know we already recorded a dependence
      // between this operation and the previous last deppart op
      last_implicit = op;
      last_implicit_gen = op->get_generation();
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
    void InnerContext::begin_trace(TraceID tid, bool logical_only,
        bool static_trace, const std::set<RegionTreeID> *trees,
        bool deprecated, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (runtime->no_tracing) return;
      if (runtime->no_physical_tracing) logical_only = true;

      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Beginning a trace in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != NULL)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_NESTED_TRACE,
          "Illegal nested trace with ID %d attempted in task %s (ID %lld)", 
          tid, get_task_name(), get_unique_id())

      Provenance *prov = NULL;
      if (provenance != NULL)
        prov = new Provenance(provenance);

      std::map<TraceID,LegionTrace*>::const_iterator finder = traces.find(tid);
      LegionTrace *trace = NULL;
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        if (static_trace)
          trace = new StaticTrace(tid, this, logical_only, prov, trees);
        else
          trace = new DynamicTrace(tid, this, logical_only, prov);
        if (!deprecated)
          traces[tid] = trace;
        trace->add_reference();
      }
      else
        trace = finder->second;

#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      trace->clear_blocking_call();

      // Issue a begin op
      TraceBeginOp *begin = runtime->get_available_begin_op();
      begin->initialize_begin(this, trace, prov);
      add_to_dependence_queue(begin);

      if (!logical_only)
      {
        // Issue a replay op
        TraceReplayOp *replay = runtime->get_available_replay_op();
        replay->initialize_replay(this, trace, prov);
        // Record the event for when the trace replay is ready
        physical_trace_replay_status.exchange(replay->get_mapped_event().id);
        add_to_dependence_queue(replay);
      }

      // Now mark that we are starting a trace
      current_trace = trace;
    }

    //--------------------------------------------------------------------------
    void InnerContext::record_physical_trace_replay(RtEvent ready, bool replay)
    //--------------------------------------------------------------------------
    {
      physical_trace_replay_status.compare_exchange_strong(ready.id, 
                                                           replay ? 1 : 0);
    }

    //--------------------------------------------------------------------------
    bool InnerContext::is_replaying_physical_trace(void)
    //--------------------------------------------------------------------------
    {
      if (current_trace == NULL)
        return false;
      if (!current_trace->is_fixed())
        return false;
      realm_id_t status = physical_trace_replay_status.load();
      if (status > 1)
      {
        // Result is not ready yet so wait until it is
        RtEvent ready;
        ready.id = status;
        if (!ready.has_triggered())
          ready.wait();
        status = physical_trace_replay_status.load();
        // No need to spin again because there won't be anymore outstanding
        // trace capture ops to be setting this
#ifdef DEBUG_LEGION
        assert((status == 0) || (status == 1));
#endif
      }
      return (status == 1);
    }

    //--------------------------------------------------------------------------
    void InnerContext::end_trace(TraceID tid, bool deprecated,
                                 const char *provenance)
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
      else if (!deprecated && (current_trace->tid != tid))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_END_TRACE_CALL,
          "Illegal end trace call on trace ID %d that does not match "
          "the current trace ID %d in task %s (UID %lld)", tid,
          current_trace->tid, get_task_name(), get_unique_id())
      bool has_blocking_call = current_trace->has_blocking_call();
      if (current_trace->is_fixed())
      {
        // Already fixed, dump a complete trace op into the stream
        TraceCompleteOp *complete_op = runtime->get_available_trace_op();
        complete_op->initialize_complete(this, has_blocking_call, provenance);
        // Remove the current trace now so we block at the end of the
        // trace in the case of program order execution
        current_trace = NULL;
        add_to_dependence_queue(complete_op);
      }
      else
      {
        // Not fixed yet, dump a capture trace op into the stream
        TraceCaptureOp *capture_op = runtime->get_available_capture_op(); 
        capture_op->initialize_capture(this, has_blocking_call,
                                       deprecated, provenance);
        // Mark that the current trace is now fixed
        current_trace->fix_trace(capture_op->get_provenance());
        // Remove the current trace now so we block at the end of the
        // trace in the case of program order execution
        current_trace = NULL;
        add_to_dependence_queue(capture_op);
      }
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
      if ((previous_trace != NULL) && (previous_trace != trace))
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
      bool poisoned = false;
      if (!wait_on.has_triggered_faultaware(poisoned))
        wait_on.wait_faultaware(poisoned);
      if (poisoned)
        raise_poison_exception();
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
      AutoLock child_lock(child_op_lock);
      if (!currently_active_context && (outstanding_subtasks == 0) && 
          (((context_configuration.min_tasks_to_schedule > 0) && 
            (pending_subtasks < 
             context_configuration.min_tasks_to_schedule)) ||
           ((context_configuration.min_frames_to_schedule > 0) &&
            (pending_frames < 
             context_configuration.min_frames_to_schedule))))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
      outstanding_subtasks++;
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
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_pending(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped frames
      if (context_configuration.min_tasks_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
      pending_subtasks++;
      if (currently_active_context && (outstanding_subtasks > 0) &&
          (pending_subtasks == context_configuration.min_tasks_to_schedule))
      {
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_pending(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduled by frames
      if (context_configuration.min_tasks_to_schedule > 0)
        decrement_pending(true/*need deferral*/);
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_pending(bool need_deferral)
    //--------------------------------------------------------------------------
    {
      AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
      assert(pending_subtasks > 0);
#endif
      pending_subtasks--;
      if (!currently_active_context && (outstanding_subtasks > 0) &&
          (pending_subtasks < context_configuration.min_tasks_to_schedule))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::increment_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
      pending_frames++;
      if (currently_active_context && (outstanding_subtasks > 0) &&
          (pending_frames == context_configuration.min_frames_to_schedule))
      {
        currently_active_context = false;
        runtime->deactivate_context(this);
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::decrement_frame(void)
    //--------------------------------------------------------------------------
    {
      // Don't need to do this if we are scheduling based on mapped tasks
      if (context_configuration.min_frames_to_schedule == 0)
        return;
      AutoLock child_lock(child_op_lock);
#ifdef DEBUG_LEGION
      assert(pending_frames > 0);
#endif
      pending_frames--;
      if (!currently_active_context && (outstanding_subtasks > 0) &&
          (pending_frames < context_configuration.min_frames_to_schedule))
      {
        currently_active_context = true;
        runtime->activate_context(this);
      }
    } 

    //--------------------------------------------------------------------------
    void InnerContext::destroy_lock(Lock l)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Can only be called from user land so no need to hold the lock
      context_locks.push_back(l.reservation_lock);
    }

    //--------------------------------------------------------------------------
    Grant InnerContext::acquire_grant(const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
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
      return Grant(new GrantImpl(unpack_requests));
    }

    //--------------------------------------------------------------------------
    void InnerContext::release_grant(Grant grant)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      grant.impl->release_grant();
    } 

    //--------------------------------------------------------------------------
    void InnerContext::destroy_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Can only be called from user land so no need to hold the lock
      context_barriers.push_back(pb.phase_barrier);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier InnerContext::advance_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      PhaseBarrier result = pb;
      Runtime::advance_barrier(result);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(pb.phase_barrier, result.phase_barrier);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective InnerContext::create_dynamic_collective(
                                       unsigned arrivals, ReductionOpID redop,
                                       const void *init_value, size_t init_size)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      return DynamicCollective(
          ApBarrier(Realm::Barrier::create_barrier(arrivals, redop, 
                                    init_value, init_size)), redop);
    }

    //--------------------------------------------------------------------------
    void InnerContext::destroy_dynamic_collective(DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Can only be called from user land so no need to hold the lock
      context_barriers.push_back(dc.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void InnerContext::arrive_dynamic_collective(DynamicCollective dc,
                                const void *buffer, size_t size, unsigned count)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      Runtime::phase_barrier_arrive(dc,count,ApEvent::NO_AP_EVENT,buffer,size);
    }

    //--------------------------------------------------------------------------
    void InnerContext::defer_dynamic_collective_arrival(DynamicCollective dc,
                                           const Future &future, unsigned count)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      future.impl->contribute_to_collective(dc, count);
      // No need to register anything if this future is an application future
      // or it was made in a context above this in the region tree
      if ((future.impl->producer_op == NULL) ||
          (future.impl->producer_depth < get_depth()))
        return;
      // Record this future as a contribution to the collective
      // for future dependence analysis
      const size_t barrier_gen = 
        Realm::ID(dc.phase_barrier.id).event_generation();
      const size_t barrier_name = dc.phase_barrier.id - barrier_gen;
      AutoLock pb_lock(phase_barrier_lock);
      barrier_contributions[barrier_name].push_back(
          BarrierContribution(future.impl->producer_op, future.impl->op_gen,
#ifdef LEGION_SPY
            future.impl->producer_uid,
#else
            0/*no uid*/,
#endif
            0/*no muid*/, barrier_gen));
    }

    //--------------------------------------------------------------------------
    void InnerContext::perform_barrier_dependence_analysis(Operation *op,
                            const std::vector<PhaseBarrier> &wait_barriers,
                            const std::vector<PhaseBarrier> &arrive_barriers,
                            MustEpochOp *must_epoch)
    //--------------------------------------------------------------------------
    {
      AutoLock pb_lock(phase_barrier_lock);
      if (!wait_barriers.empty())
        analyze_barrier_dependences(op, wait_barriers, must_epoch, true);
      if (!arrive_barriers.empty())
        analyze_barrier_dependences(op, arrive_barriers, must_epoch, false);
    }

    //--------------------------------------------------------------------------
    void InnerContext::analyze_barrier_dependences(Operation *op,
                              const std::vector<PhaseBarrier> &barriers,
                              MustEpochOp *must_epoch_op, bool previous_gen)
    //--------------------------------------------------------------------------
    {
      const UniqueID uid = op->get_unique_op_id();
      const GenerationID gen = op->get_generation();
      const UniqueID muid = (must_epoch_op == NULL) ? 0 :
        must_epoch_op->get_unique_op_id();
      // Record our barriers for future uses
      for (std::vector<PhaseBarrier>::const_iterator ait =
            barriers.begin(); ait != barriers.end(); ait++)
      {
        // Figure out the generic barrier ID
        const ApBarrier barrier = previous_gen ? ait->phase_barrier :
          Runtime::get_previous_phase(ait->phase_barrier);
        const size_t barrier_gen = Realm::ID(barrier.id).event_generation();
        const size_t barrier_name = barrier.id - barrier_gen;
        std::list<BarrierContribution> &previous =
          barrier_contributions[barrier_name];
        for (std::list<BarrierContribution>::iterator it =
              previous.begin(); it != previous.end(); /*nothing*/)
        {
          // skip anything with a larger barrier generation
          if (it->bargen >= barrier_gen)
          {
            it++;
            continue;
          }
          // If must epoch and same uid then skip it
          if ((muid > 0) && (muid == it->muid))
          {
            it++;
            continue;
          }
#ifdef LEGION_SPY
          // No pruning for Legion Spy
          op->register_dependence(it->op, it->gen);
          LegionSpy::log_mapping_dependence(get_unique_id(), 
              it->uid, 0, uid, 0, TRUE_DEPENDENCE);
          it++;
#else
          if (op->register_dependence(it->op, it->gen))
            it++;
          else
            it = previous.erase(it);
#endif        
        }
        previous.push_back(BarrierContribution(op,gen,uid,muid,barrier_gen));
      }
    }

    //--------------------------------------------------------------------------
    Future InnerContext::get_dynamic_collective_result(DynamicCollective dc,
                                                       const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
#ifdef DEBUG_LEGION
      log_run.debug("Get dynamic collective result in task %s (ID %lld)",
                    get_task_name(), get_unique_id());
#endif
      DynamicCollectiveOp *collective = 
        runtime->get_available_dynamic_collective_op();
      Future result = collective->initialize(this, dc, provenance);
      add_to_dependence_queue(collective);
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective InnerContext::advance_dynamic_collective(
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      DynamicCollective result = dc;
      Runtime::advance_barrier(result);
#ifdef LEGION_SPY
      LegionSpy::log_event_dependence(dc.phase_barrier, result.phase_barrier);
#endif
      return result;
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
      if (context_configuration.max_templates_per_trace == 0)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from call 'configure context' "
                      "on mapper %s for task %s (ID %lld). The "
                      "'max_templates_per_trace' must be a non-zero value.",
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
        assert(regions[idx].handle_type == LEGION_SINGULAR_PROJECTION);
#endif
        // If this is a NO_ACCESS or had no privilege fields we can skip this
        if (no_access_regions[idx])
          continue;
        // Only need to initialize the context if this is
        // not a leaf and it wasn't virtual mapped
        if (!virtual_mapped[idx])
        {
          // The parent region requirement is restricted if it is
          // simultaneous or it is reduce-only. Simultaneous is 
          // restricted because of normal Legion coherence semantics.
          // Reduce-only is restricted because we don't issue close
          // operations at the end of a context for reduce-only cases
          // right now so by making it restricted things are eagerly
          // flushed out to the parent task's instance.
          const bool restricted = 
            IS_SIMULT(regions[idx]) || IS_REDUCE(regions[idx]);
          runtime->forest->initialize_current_context(tree_context,
              clone_requirements[idx], restricted, physical_instances[idx],
              unmap_events[idx], this, idx, top_views, applied_events);
#ifdef DEBUG_LEGION
          assert(!physical_instances[idx].empty());
#endif
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
            for (std::map<unsigned,RegionRequirement>::const_iterator it = 
                  created_requirements.begin(); it !=
                  created_requirements.end(); it++)
            {
#ifdef DEBUG_LEGION
              assert(returnable_privileges.find(it->first) !=
                      returnable_privileges.end());
#endif
              if (!returnable_privileges[it->first])
                to_invalidate[it->first] = it->second.region;
            }
            rez.serialize<size_t>(to_invalidate.size());
            for (std::map<unsigned,LogicalRegion>::const_iterator it = 
                  to_invalidate.begin(); it != to_invalidate.end(); it++)
            {
              // Add the size of the original regions to the index
              rez.serialize<unsigned>(it->first);
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
        for (std::map<unsigned,RegionRequirement>::const_iterator it = 
              created_requirements.begin(); it != 
              created_requirements.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(returnable_privileges.find(it->first) !=
                  returnable_privileges.end());
#endif
          // See if we're a returnable privilege or not
          if (returnable_privileges[it->first])
          {
            // If we're the outermost context or the requirement was
            // deleted, then we can invalidate everything
            // Otherwiswe we only invalidate the users
            const bool users_only = !is_outermost;
            runtime->forest->invalidate_current_context(outermost_ctx,
                                        users_only, it->second.region);
          }
          else // Not returning so invalidate the full thing 
          {
            runtime->forest->invalidate_current_context(tree_context,
                              false/*users only*/, it->second.region);
            // Little tricky here, this is safe to invaliate the whole
            // tree even if we only had privileges on a field because
            // if we had privileges on the whole region in this context
            // it would have merged the created_requirement and we wouldn't
            // have a non returnable privilege requirement in this context
            runtime->forest->invalidate_versions(tree_context, 
                                                 it->second.region);
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
    void InnerContext::convert_target_views(const InstanceSet &targets,
                                       std::vector<InstanceView*> &target_views)
    //--------------------------------------------------------------------------
    {
      target_views.resize(targets.size());
      std::vector<unsigned> still_needed;
      {
        AutoLock inst_lock(instance_view_lock,1,false/*exclusive*/); 
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          // See if we can find it
          PhysicalManager *manager = targets[idx].get_physical_manager();
          std::map<PhysicalManager*,InstanceView*>::const_iterator finder = 
            instance_top_views.find(manager);     
          if (finder != instance_top_views.end())
            target_views[idx] = finder->second;
          else
            still_needed.push_back(idx);
        }
      }
      if (!still_needed.empty())
      {
        std::set<RtEvent> ready_events;
        const AddressSpaceID local_space = runtime->address_space;
        for (std::vector<unsigned>::const_iterator it = 
              still_needed.begin(); it != still_needed.end(); it++)
        {
          PhysicalManager *manager = targets[*it].get_physical_manager();
          RtEvent ready;
          target_views[*it] = 
            create_instance_top_view(manager, local_space, &ready);
          if (ready.exists())
            ready_events.insert(ready);
        }
        if (!ready_events.empty())
        {
          RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists())
            wait_on.wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::convert_target_views(const InstanceSet &targets,
                                   std::vector<MaterializedView*> &target_views)
    //--------------------------------------------------------------------------
    {
      std::vector<InstanceView*> inst_views(targets.size());
      convert_target_views(targets, inst_views);
      target_views.resize(inst_views.size());
      for (unsigned idx = 0; idx < inst_views.size(); idx++)
        target_views[idx] = inst_views[idx]->as_materialized_view();
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
    FillView* InnerContext::find_or_create_fill_view(FillOp *op, 
                                     std::set<RtEvent> &map_applied_events,
                                     const void *value, const size_t value_size,
                                     bool &took_ownership)
    //--------------------------------------------------------------------------
    {
      // Two versions of this method depending on whether we are doing 
      // Legion Spy or not, Legion Spy wants to know exactly which op
      // made each fill view so we can't cache them
      WrapperReferenceMutator mutator(map_applied_events);
#ifndef LEGION_SPY
      // See if we can find this in the cache first
      AutoLock f_lock(fill_view_lock);
      for (std::list<FillView*>::iterator it = 
            fill_view_cache.begin(); it != fill_view_cache.end(); it++)
      {
        if (!(*it)->value->matches(value, value_size))
          continue;
        // Record a reference on it and then return
        FillView *result = (*it);
        // Move it back to the front of the list
        fill_view_cache.erase(it);
        fill_view_cache.push_front(result);
        result->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
        took_ownership = false;
        return result;
      }
      // At this point we have to make it since we couldn't find it
#endif
      DistributedID did = runtime->get_available_distributed_id();
      FillView::FillViewValue *fill_value = 
        new FillView::FillViewValue(value, value_size);
      FillView *fill_view = 
        new FillView(runtime->forest, did, runtime->address_space,
                     fill_value, true/*register now*/
#ifdef LEGION_SPY
                     , op->get_unique_op_id()

#endif
                     );
      fill_view->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
#ifndef LEGION_SPY
      // Add it to the cache since we're not doing Legion Spy
      fill_view->add_base_valid_ref(CONTEXT_REF, &mutator);
      fill_view_cache.push_front(fill_view);
      if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
      {
        FillView *oldest = fill_view_cache.back();
        fill_view_cache.pop_back();
        if (oldest->remove_base_valid_ref(CONTEXT_REF))
          delete oldest;
      }
#endif
      took_ownership = true;
      return fill_view;
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
      RtEvent ctx_ready;
      InnerContext *context = 
        runtime->find_context(context_uid, false, &ctx_ready);
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
      RemoteCreateViewArgs args(context, manager, target, to_trigger, source);
      runtime->issue_runtime_meta_task(args, 
          LG_LATENCY_DEFERRED_PRIORITY, ctx_ready);
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
     PhysicalInstance deferred_result_instance, FutureFunctor *callback_functor)
    //--------------------------------------------------------------------------
    {
      // See if we have any local regions or fields that need to be deallocated
      std::vector<LogicalRegion> local_regions_to_delete;
      std::map<FieldSpace,std::set<FieldID> > local_fields_to_delete;
      {
        AutoLock priv_lock(privilege_lock,1,false/*exclusive*/);
        for (std::map<LogicalRegion,bool>::const_iterator it = 
              local_regions.begin(); it != local_regions.end(); it++)
          if (!it->second)
            local_regions_to_delete.push_back(it->first);
        for (std::map<std::pair<FieldSpace,FieldID>,bool>::const_iterator it =
              local_fields.begin(); it != local_fields.end(); it++)
          if (!it->second)
            local_fields_to_delete[it->first.first].insert(it->first.second);
      }
      if (!local_regions_to_delete.empty())
      {
        for (std::vector<LogicalRegion>::const_iterator it = 
              local_regions_to_delete.begin(); it != 
              local_regions_to_delete.end(); it++)
          destroy_logical_region(*it, false/*unordered*/, NULL/*provenace*/);
      }
      if (!local_fields_to_delete.empty())
      {
        for (std::map<FieldSpace,std::set<FieldID> >::const_iterator it = 
              local_fields_to_delete.begin(); it !=
              local_fields_to_delete.end(); it++)
        {
          FieldAllocatorImpl *allocator = create_field_allocator(it->first);
          free_fields(allocator, it->first, it->second, 
                  false/*unordered*/, NULL/*provenace*/);
        }
      }
      if (!index_launch_spaces.empty())
      {
        for (std::map<Domain,IndexSpace>::const_iterator it = 
              index_launch_spaces.begin(); it != 
              index_launch_spaces.end(); it++)
          destroy_index_space(it->second, false/*unordered*/, 
              true/*recurse*/, NULL/*provenance*/);
      }
      if (overhead_tracker != NULL)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff = current - previous_profiling_time;
        overhead_tracker->application_time += diff;
      }
      if (!task_local_instances.empty())
        release_task_local_instances(deferred_result_instance);
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
      unmap_all_regions(false/*external*/);
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
          add_to_dependence_queue(close_op);
        }
        else
        {
          // Make a virtual close op to close up the instance
          VirtualCloseOp *close_op = 
            runtime->get_available_virtual_close_op();
          close_op->initialize(this, idx, regions[idx]);
          add_to_dependence_queue(close_op);
        }
      }
      // Check to see if we have any unordered operations that we need to inject
      {
        AutoLock d_lock(dependence_lock);
#ifdef DEBUG_LEGION
        assert(!finished_execution);
#endif
        if (!unordered_ops.empty())
          insert_unordered_ops(d_lock);
        finished_execution = true;
      }
      // Mark that we are done executing this operation
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
      const bool internal_task = Processor::get_executing_processor().exists();
      RtEvent effects_done(internal_task && !inline_task ? 
          Processor::get_current_finish_event() : Realm::Event::NO_EVENT); 
      if (last_registration.exists() && !last_registration.has_triggered())
        effects_done = Runtime::merge_events(effects_done, last_registration);
      if (deferred_result_instance.exists())
        parent_ctx->add_to_post_task_queue(this, effects_done,
                                           res, res_size, 
                                           deferred_result_instance);
      else if (callback_functor != NULL)
      {
        if (owner_task->is_reducing_future())
        {
          // If we're reducing this future value then just do the callback
          // now since there is no point in deferring it to later
          const size_t callback_size = 
            callback_functor->callback_get_future_size();
          void *buffer = malloc(callback_size);
          callback_functor->callback_pack_future(buffer, callback_size);
          callback_functor->callback_release_future();
          if (owned)
            delete callback_functor;
          parent_ctx->add_to_post_task_queue(this, effects_done, 
                                             buffer, callback_size);
        }
        else
          parent_ctx->add_to_post_task_queue(this, effects_done, res, res_size,
                            deferred_result_instance, callback_functor, owned);
      }
      else if (!owned)
      {
        if (res_size > 0)
        {
          void *result_copy = malloc(res_size);
          memcpy(result_copy, res, res_size);
          parent_ctx->add_to_post_task_queue(this, effects_done,
                                             result_copy, res_size);
        }
        else
          parent_ctx->add_to_post_task_queue(this, effects_done,
                                             res, res_size);
      }
      else
        parent_ctx->add_to_post_task_queue(this, effects_done, res, res_size);
      if (!inline_task)
#ifdef DEBUG_LEGION
        runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                       false/*meta*/);
#else
        runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    void InnerContext::post_end_task(const void *res,size_t res_size,
                                     bool owned,FutureFunctor *callback_functor)
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
      single_task->handle_future(res, res_size, owned, 
                                 callback_functor, executing_processor);
      // If we weren't a leaf task, compute the conditions for being mapped
      // which is that all of our children are now mapped
      // Also test for whether we need to trigger any of our child
      // complete or committed operations before marking that we
      // are done executing
      bool need_complete = false;
      bool need_commit = false;
      std::set<RtEvent> preconditions;
      std::set<ApEvent> child_completion_events;
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
            for (LegionMap<Operation*,GenerationID,
                  COMPLETE_CHILD_ALLOC>::const_iterator it =
                 complete_children.begin(); it != complete_children.end(); it++)
              child_completion_events.insert(it->first->get_completion_event());
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
      if (need_complete)
      {
        if (!child_completion_events.empty())
          owner_task->trigger_children_complete(
              Runtime::merge_events(NULL, child_completion_events));
        else
          owner_task->trigger_children_complete(ApEvent::NO_AP_EVENT);
      }
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
    /*static*/ void InnerContext::handle_compute_equivalence_sets_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      // This should always be coming back to the owner node so there's no
      // need to defer this is at should always be here
      InnerContext *local_ctx = runtime->find_context(context_uid);
      VersionManager *target_manager;
      derez.deserialize(target_manager);
      RegionTreeID tree_id;
      derez.deserialize(tree_id);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      FieldMask mask;
      derez.deserialize(mask);
      IndexSpace handle;
      derez.deserialize(handle);
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);

      const RtEvent done = local_ctx->compute_equivalence_sets(target_manager, 
                                           tree_id, handle, expr, mask, origin);
      Runtime::trigger_event(ready_event, done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_prepipeline_stage(const void *args)
    //--------------------------------------------------------------------------
    {
      const PrepipelineArgs *pargs = (const PrepipelineArgs*)args;
      if (pargs->context->process_prepipeline_stage() &&
          pargs->context->remove_reference())
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
    /*static*/ void InnerContext::handle_ready_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const TriggerReadyArgs *targs = (const TriggerReadyArgs*)args;
      if (targs->context->process_ready_queue() &&
          targs->context->remove_reference())
        delete targs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_enqueue_task_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredEnqueueTaskArgs *dargs = 
        (const DeferredEnqueueTaskArgs*)args;
      if (dargs->context->process_enqueue_task_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_distribute_task_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredDistributeTaskArgs *dargs = 
        (const DeferredDistributeTaskArgs*)args;
      if (dargs->context->process_distribute_task_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_launch_task_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredLaunchTaskArgs *dargs = 
        (const DeferredLaunchTaskArgs*)args;
      if (dargs->context->process_launch_task_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_resolution_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const TriggerResolutionArgs *targs = (const TriggerResolutionArgs*)args;
      if (targs->context->process_resolution_queue() &&
          targs->context->remove_reference())
        delete targs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_trigger_execution_queue(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const TriggerExecutionArgs *targs = (const TriggerExecutionArgs*)args;
      if (targs->context->process_trigger_execution_queue() &&
          targs->context->remove_reference())
        delete targs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_deferred_execution_queue(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredExecutionArgs *dargs = (const DeferredExecutionArgs*)args;
      if (dargs->context->process_deferred_execution_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_trigger_completion_queue(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const TriggerCompletionArgs *targs = (const TriggerCompletionArgs*)args;
      if (targs->context->process_trigger_completion_queue() &&
          targs->context->remove_reference())
        delete targs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_deferred_completion_queue(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredCompletionArgs *dargs = (const DeferredCompletionArgs*)args;
      if (dargs->context->process_deferred_completion_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_trigger_commit_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const TriggerCommitArgs *targs = (const TriggerCommitArgs*)args;
      if (targs->context->process_trigger_commit_queue() &&
          targs->context->remove_reference())
        delete targs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_deferred_commit_queue(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferredCommitArgs *dargs = (const DeferredCommitArgs*)args;
      if (dargs->context->process_deferred_commit_queue() &&
          dargs->context->remove_reference())
        delete dargs->context;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InnerContext::handle_post_end_task(const void *args)
    //--------------------------------------------------------------------------
    {
      const PostEndArgs *pargs = (const PostEndArgs*)args;
      if (pargs->proxy_this->process_post_end_tasks() && 
          pargs->proxy_this->remove_reference())
        delete pargs->proxy_this;
    }

    //--------------------------------------------------------------------------
    bool InnerContext::inline_child_task(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INLINE_CHILD_TASK_CALL);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_inline_task(child->get_unique_id());
      // Check to see if the child is predicated
      // If it is wait for it to resolve
      if (child->is_predicated_op())
      {
        // See if the predicate speculates false, if so return false
        // and then we are done.
        if (!child->get_predicate_value(executing_processor))
          return true;
      }
      // Find the mapped physical regions associated with each of the
      // child task's region requirements. If we don't have one then
      // it's not legal to inline the child task
      std::vector<PhysicalRegion> child_regions(child->regions.size());
      for (unsigned childidx = 0; childidx < child_regions.size(); childidx++)
      {
        const RegionRequirement &child_req = child->regions[childidx]; 
        bool found = false;
        for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
        {
          if (!physical_regions[our_idx].is_mapped())
            continue;
          const RegionRequirement &our_req = regions[our_idx];
          const RegionTreeID our_tid = our_req.region.get_tree_id();
          const IndexSpace our_space = our_req.region.get_index_space();
          const RegionUsage our_usage(our_req);
          if (!check_region_dependence(our_tid, our_space, our_req,
                  our_usage, child_req, false/*ignore privileges*/))
            continue;
          child_regions[childidx] = physical_regions[our_idx];
          found = true;
          break;
        }
        if (found)
          continue;
        // Need the lock here because of unordered detach operations
        AutoLock i_lock(inline_lock,1,false/*exclusive*/);
        for (std::list<PhysicalRegion>::const_iterator it =
              inline_regions.begin(); it != inline_regions.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->is_mapped());
#endif
          const RegionRequirement &our_req = it->impl->get_requirement();
          const RegionTreeID our_tid = our_req.region.get_tree_id();
          const IndexSpace our_space = our_req.region.get_index_space();
          const RegionUsage our_usage(our_req);
          if (!check_region_dependence(our_tid, our_space, our_req,
                  our_usage, child_req, false/*ignore privileges*/))
            continue;
          child_regions[childidx] = *it;
          found = true;
          break;
        }
        // If we didn't find any physical region then report the warning
        // and return because we couldn't find a mapped physical region
        if (!found)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_FAILED_INLINING,
              "Failed to inline task %s (UID %lld) into parent task "
              "%s (UID %lld) because there was no mapped region for "
              "region requirement %d to use. Currently all regions "
              "must be mapped in the parent task in order to allow "
              "for inlining. If you believe you have a compelling use "
              "case for inline a task with virtually mapped regions "
              "then please contact the Legion developers.", 
              child->get_task_name(), child->get_unique_id(), 
              owner_task->get_task_name(), owner_task->get_unique_id(),childidx)
          return false;
        }
      }
      register_executing_child(child);
      const ApEvent child_done = child->get_completion_event();
      // Now select the variant for task based on the regions 
      std::deque<InstanceSet> physical_instances(child_regions.size());
      VariantImpl *variant = 
        select_inline_variant(child, child_regions, physical_instances); 
      child->perform_inlining(variant, physical_instances);
      // Then wait for the child operation to be finished
      bool poisoned = false;
      if (!child_done.has_triggered_faultaware(poisoned))
        child_done.wait_faultaware(poisoned);
      if (poisoned)
        raise_poison_exception();
      return true;
    } 

    //--------------------------------------------------------------------------
    void InnerContext::analyze_free_local_fields(FieldSpace handle,
                                     const std::vector<FieldID> &local_to_free,
                                     std::vector<unsigned> &local_field_indexes)
    //--------------------------------------------------------------------------
    {
      AutoLock local_lock(local_field_lock,1,false/*exclusive*/);
      std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator 
        finder = local_field_infos.find(handle);
#ifdef DEBUG_LEGION
      assert(finder != local_field_infos.end());
#endif
      for (unsigned idx = 0; idx < local_to_free.size(); idx++)
      {
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (std::vector<LocalFieldInfo>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if (it->fid == local_to_free[idx])
          {
            // Can't remove it yet
            local_field_indexes.push_back(it->index);
#ifdef DEBUG_LEGION
            found = true;
#endif
            break;
          }
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void InnerContext::remove_deleted_local_fields(FieldSpace space,
                                          const std::vector<FieldID> &to_remove)
    //--------------------------------------------------------------------------
    {
      AutoLock local_lock(local_field_lock);
      std::map<FieldSpace,std::vector<LocalFieldInfo> >::iterator 
        finder = local_field_infos.find(space);
#ifdef DEBUG_LEGION
      assert(finder != local_field_infos.end());
#endif
      for (unsigned idx = 0; idx < to_remove.size(); idx++)
      {
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (std::vector<LocalFieldInfo>::iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
        {
          if (it->fid == to_remove[idx])
          {
            finder->second.erase(it);
#ifdef DEBUG_LEGION
            found = true;
#endif
            break;
          }
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      if (finder->second.empty())
        local_field_infos.erase(finder);
    } 

    //--------------------------------------------------------------------------
    void InnerContext::execute_task_launch(TaskOp *task, bool index,
       LegionTrace *current_trace, bool silence_warnings, bool inlining_enabled)
    //--------------------------------------------------------------------------
    {
      bool inline_task = false;
      if (inlining_enabled)
        inline_task = task->select_task_options(true/*prioritize*/);
      // Now check to see if we're inling the task or just performing
      // a normal asynchronous task launch
      if (!inline_task || !inline_child_task(task))
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
        add_to_dependence_queue(task);
        // Remap any unmapped regions
        if (!unmapped_regions.empty())
        {
          Provenance *prov = task->get_provenance();
          remap_unmapped_regions(current_trace, unmapped_regions,
              (prov == NULL) ? (const char*)NULL : prov->provenance.c_str());
        }
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
      if (local_field_infos.empty())
        return;
      for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
            fit = local_field_infos.begin(); 
            fit != local_field_infos.end(); fit++)
      {
        std::vector<LocalFieldInfo> &child = child_local[fit->first];
        child.resize(fit->second.size());
        for (unsigned idx = 0; idx < fit->second.size(); idx++)
        {
          LocalFieldInfo &field = child[idx];
          field = fit->second[idx];
          field.ancestor = true; // mark that this is an ancestor field
        }
      }
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    Operation* InnerContext::get_earliest(void) const
    //--------------------------------------------------------------------------
    {
      Operation *result = NULL;
      unsigned index = 0;
      for (std::map<Operation*,GenerationID>::const_iterator it = 
            executing_children.begin(); it != executing_children.end(); it++)
      {
        if (result == NULL)
        {
          result = it->first;
          index = result->get_ctx_index();
        }
        else if (it->first->get_ctx_index() < index)
        {
          result = it->first;
          index = result->get_ctx_index();
        }
      }
      return result;
    }
#endif

#ifdef LEGION_SPY
    //--------------------------------------------------------------------------
    void InnerContext::register_implicit_replay_dependence(Operation *op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_mapping_dependence(get_unique_id(), 
          current_fence_uid, 0/*idx*/, op->get_unique_op_id(),
          0/*idx*/, LEGION_TRUE_DEPENDENCE);
    }
#endif

    /////////////////////////////////////////////////////////////
    // Top Level Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(Runtime *rt, UniqueID ctx_id)
      : InnerContext(rt, NULL, -1, false/*full inner*/, dummy_requirements, 
                     dummy_indexes, dummy_mapped, ctx_id, ApEvent::NO_AP_EVENT)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TopLevelContext::TopLevelContext(const TopLevelContext &rhs)
      : InnerContext(NULL, NULL, -1, false, dummy_requirements, dummy_indexes, 
                     dummy_mapped, 0, ApEvent::NO_AP_EVENT)
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
    void TopLevelContext::pack_remote_context(Serializer &rez, 
                                              AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(depth);
    }

    //--------------------------------------------------------------------------
    TaskContext* TopLevelContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    RtEvent TopLevelContext::compute_equivalence_sets(VersionManager *manager,
                              RegionTreeID tree_id, IndexSpace handle, 
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
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
    InnerContext* TopLevelContext::find_top_context(InnerContext *previous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;
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
    int RemoteTask::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_depth();
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteTask::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return owner->get_context_uid();
    }

    //--------------------------------------------------------------------------
    Domain RemoteTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    size_t RemoteTask::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTask::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    bool RemoteTask::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteTask::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if ((parent_task == NULL) && has_parent_task())
        parent_task = owner->get_parent_task();
      return parent_task;
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
      : InnerContext(rt, NULL, -1, false/*full inner*/, remote_task.regions, 
          local_parent_req_indexes, local_virtual_mapped, 
          context_uid, ApEvent::NO_AP_EVENT, true/*remote*/),
        parent_ctx(NULL), top_level_context(false), 
        remote_task(RemoteTask(this))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteContext::RemoteContext(const RemoteContext &rhs)
      : InnerContext(NULL, NULL, 0, false, rhs.regions,local_parent_req_indexes,
          local_virtual_mapped, 0, ApEvent::NO_AP_EVENT, true), 
        remote_task(RemoteTask(this))
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteContext::~RemoteContext(void)
    //--------------------------------------------------------------------------
    {
      if (!local_field_infos.empty())
      {
        // If we have any local fields then tell field space that
        // we can remove them and then clear them 
        for (std::map<FieldSpace,std::vector<LocalFieldInfo> >::const_iterator
              it = local_field_infos.begin(); 
              it != local_field_infos.end(); it++)
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
        local_field_infos.clear();
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
    InnerContext* RemoteContext::find_top_context(InnerContext *previous)
    //--------------------------------------------------------------------------
    {
      if (!top_level_context)
        return find_parent_context()->find_top_context(this);
 #ifdef DEBUG_LEGION
      assert(previous != NULL);
#endif
      return previous;     
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
    RtEvent RemoteContext::compute_equivalence_sets(VersionManager *manager,
                              RegionTreeID tree_id, IndexSpace handle,
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!top_level_context);
      assert(source == runtime->address_space); // should always be local
#endif
      // Send it to the owner space if we are the top-level context
      // otherwise we send it to the owner of the context
      const AddressSpaceID target = runtime->get_runtime_owner(context_uid);
      RtUserEvent ready_event = Runtime::create_rt_user_event();
      // Send off a request to the owner node to handle it
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(context_uid);
        rez.serialize(manager);
        rez.serialize(tree_id);
        expr->pack_expression(rez, target);
        rez.serialize(mask);
        rez.serialize(handle);
        rez.serialize(source);
        rez.serialize(ready_event);
      }
      // Send it to the owner space 
      runtime->send_compute_equivalence_sets_request(target, rez);
      return ready_event;
    }

    //--------------------------------------------------------------------------
    InnerContext* RemoteContext::find_parent_physical_context(unsigned index,
                                                           LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(regions.size() == virtual_mapped.size());
      assert(regions.size() == parent_req_indexes.size());
#endif     
      if (index < virtual_mapped.size())
      {
        // See if it is virtual mapped
        if (virtual_mapped[index])
          return find_parent_context()->find_parent_physical_context(
                                            parent_req_indexes[index], parent);
        else // We mapped a physical instance so we're it
          return this;
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
            return finder->second;
          std::map<unsigned,RtEvent>::const_iterator pending_finder = 
            pending_physical_contexts.find(index);
          if (pending_finder == pending_physical_contexts.end())
          {
            // Make a new request
            request = Runtime::create_rt_user_event();
            pending_physical_contexts[index] = request;
            wait_on = request;
            // Record that someone on the local node is using this context
            local_physical_contexts.insert(parent);
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
            rez.serialize(parent);
            rez.serialize(request);
          }
          const AddressSpaceID target = runtime->get_runtime_owner(context_uid);
          runtime->send_remote_context_physical_request(target, rez);
        }
        // Wait for the result to come back to us
        wait_on.wait();
        // When we wake up it should be there
        AutoLock rem_lock(remote_lock, 1, false/*exclusive*/);
#ifdef DEBUG_LEGION
        assert(physical_contexts.find(index) != physical_contexts.end());
#endif
        return physical_contexts[index]; 
      }
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
      derez.deserialize(depth);
      top_level_context = (depth < 0);
      // If we're the top-level context then we're already done
      if (top_level_context)
        return;
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
      derez.deserialize(parent_context_uid);
      size_t num_coordinates;
      derez.deserialize(num_coordinates);
      context_coordinates.resize(num_coordinates);
      for (unsigned idx = 0; idx < num_coordinates; idx++)
      {
        std::pair<size_t,DomainPoint> &coordinate = context_coordinates[idx];
        derez.deserialize(coordinate.first);
        derez.deserialize(coordinate.second);
      }
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
    const Task* RemoteContext::get_parent_task(void)
    //--------------------------------------------------------------------------
    {
      // Note that it safe to actually perform the find_context call here
      // because we are no longer in the virtual channel for unpacking
      // remote contexts therefore we can page in the context
      if (parent_ctx == NULL)
        parent_ctx = runtime->find_context(parent_context_uid);
      return parent_ctx->get_task();
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
        size_t prov_size;
        derez.deserialize(prov_size);
        Provenance *provenance = NULL;
        if (prov_size > 0)
        {
          provenance = new Provenance((const char*)derez.get_current_pointer());
          derez.advance_pointer(prov_size);
        }
        size_t num_local;
        derez.deserialize(num_local); 
        std::vector<FieldID> fields(num_local);
        std::vector<size_t> field_sizes(num_local);
        std::vector<CustomSerdezID> serdez_ids(num_local);
        std::vector<unsigned> indexes(num_local);
        {
          // Take the lock for updating this data structure
          AutoLock local_lock(local_field_lock);
          std::vector<LocalFieldInfo> &infos = local_field_infos[handle];
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
                                             serdez_ids, indexes, provenance);
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
      LogicalRegion parent;
      derez.deserialize(parent);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtEvent ctx_ready;
      InnerContext *local = 
        runtime->find_context(context_uid, false/*can fail*/, &ctx_ready);

      // Always defer this in case it blocks, we can't block the virtual channel
      RemotePhysicalRequestArgs args(context_uid, target, local, 
                                     index, source, to_trigger, parent);
      runtime->issue_runtime_meta_task(args, 
          LG_LATENCY_DEFERRED_PRIORITY, ctx_ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::defer_physical_request(const void *args,
                                                          Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const RemotePhysicalRequestArgs *rargs = 
        (const RemotePhysicalRequestArgs*)args;
      InnerContext *result = 
        rargs->local->find_parent_physical_context(rargs->index, rargs->parent);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(rargs->target);
        rez.serialize(rargs->index);
        rez.serialize(result->context_uid);
        rez.serialize(rargs->to_trigger);
      }
      runtime->send_remote_context_physical_response(rargs->source, rez);
    }

    //--------------------------------------------------------------------------
    void RemoteContext::set_physical_context_result(unsigned index,
                                                    InnerContext *result)
    //--------------------------------------------------------------------------
    {
      AutoLock rem_lock(remote_lock);
#ifdef DEBUG_LEGION
      assert(physical_contexts.find(index) == physical_contexts.end());
#endif
      physical_contexts[index] = result;
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
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      RtEvent ctx_ready;
      InnerContext *result = 
        runtime->find_context(result_uid, false/*weak*/, &ctx_ready);
      if (ctx_ready.exists())
      {
        // Launch a continuation in case we need to page in the context
        // We obviously can't block the virtual channel
        RemotePhysicalResponseArgs args(target, result, index);
        RtEvent done = 
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
        Runtime::trigger_event(to_trigger, done);
      }
      else
      {
        target->set_physical_context_result(index, result);
        Runtime::trigger_event(to_trigger);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteContext::defer_physical_response(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemotePhysicalResponseArgs *rargs = 
        (const RemotePhysicalResponseArgs*)args;
      rargs->target->set_physical_context_result(rargs->index, rargs->result);
    }

    /////////////////////////////////////////////////////////////
    // Leaf Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LeafContext::LeafContext(Runtime *rt, SingleTask *owner, bool inline_task)
      : TaskContext(rt, owner, owner->get_depth(), owner->regions, inline_task)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LeafContext::LeafContext(const LeafContext &rhs)
      : TaskContext(NULL, NULL, 0, rhs.regions, false)
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
    void LeafContext::receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regs,
              std::vector<DeletedRegion> &deleted_regs,
              std::set<std::pair<FieldSpace,FieldID> > &created_fids,
              std::vector<DeletedField> &deleted_fids,
              std::map<FieldSpace,unsigned> &created_fs,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_fs,
              std::vector<DeletedFieldSpace> &deleted_fs,
              std::map<IndexSpace,unsigned> &created_is,
              std::vector<DeletedIndexSpace> &deleted_is,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
    void LeafContext::compute_task_tree_coordinates(
                       std::vector<std::pair<size_t,DomainPoint> > &coordinates)
    //--------------------------------------------------------------------------
    {
      TaskContext *owner_ctx = owner_task->get_context();
#ifdef DEBUG_LEGION
      InnerContext *parent_ctx = dynamic_cast<InnerContext*>(owner_ctx);
      assert(parent_ctx != NULL);
#else
      InnerContext *parent_ctx = static_cast<InnerContext*>(owner_ctx);
#endif
      parent_ctx->compute_task_tree_coordinates(coordinates);
      coordinates.push_back(std::make_pair(
            owner_task->get_context_index(), owner_task->index_point));
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
      if (runtime->check_privileges)
        child->perform_privilege_checks();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_inline_task(child->get_unique_id());
      // Find the mapped physical regions associated with each of the
      // child task's region requirements. If they aren't mapped then
      // we need a mapping fence to ensure that all the mappings are
      // done before we attempt to run this task. If they are all mapped
      // though then we can run this right away.
      std::vector<PhysicalRegion> child_regions(child->regions.size());
      for (unsigned childidx = 0; childidx < child_regions.size(); childidx++)
      {
        const RegionRequirement &child_req = child->regions[childidx];
#ifdef DEBUG_LEGION
        bool found = false;
#endif
        for (unsigned our_idx = 0; our_idx < physical_regions.size(); our_idx++)
        {
          if (!physical_regions[our_idx].is_mapped())
            continue;
          const RegionRequirement &our_req = regions[our_idx];
          const RegionTreeID our_tid = our_req.region.get_tree_id();
          const IndexSpace our_space = our_req.region.get_index_space();
          const RegionUsage our_usage(our_req);
          if (!check_region_dependence(our_tid, our_space, our_req,
                  our_usage, child_req, false/*ignore privileges*/))
            continue;
          child_regions[childidx] = physical_regions[our_idx];
#ifdef DEBUG_LEGION
          found = true;
#endif
          break;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      // Now select the variant for task based on the regions 
      std::deque<InstanceSet> physical_instances(child_regions.size());
      VariantImpl *variant = 
        select_inline_variant(child, child_regions, physical_instances); 
      child->perform_inlining(variant, physical_instances);
      // No need to wait here, we know everything are leaves all the way
      // down from here so there is no way for there to be effects
    }

    //--------------------------------------------------------------------------
    VariantImpl* LeafContext::select_inline_variant(TaskOp *child,
                              const std::vector<PhysicalRegion> &parent_regions,
                              std::deque<InstanceSet> &physical_instances)
    //--------------------------------------------------------------------------
    {
      VariantImpl *variant_impl = TaskContext::select_inline_variant(child,
                                        parent_regions, physical_instances);
      if (!variant_impl->is_leaf())
      {
        MapperManager *child_mapper = 
          runtime->find_mapper(executing_processor, child->map_id);
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invoction of "
                      "'select_task_variant' on mapper %s. Mapper selected "
                      "an invalid variant ID %d for inlining of task %s "
                      "(UID %lld). Parent task %s (UID %lld) is a leaf task "
                      "but mapper selected non-leaf variant %d for task %s.",
                      child_mapper->get_mapper_name(),
                      variant_impl->vid, child->get_task_name(), 
                      child->get_unique_id(), owner_task->get_task_name(),
                      owner_task->get_unique_id(), variant_impl->vid,
                      child->get_task_name())
      }
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    bool LeafContext::is_leaf_context(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space(const Future &f, TypeTag tag,
                                               const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
        "Illegal index space from future creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    } 

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_space(IndexSpace handle, 
               const bool unordered, const bool recurse, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->forest->is_top_level_index_space(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy index space %x in task %s (UID %lld) "
            "which is not a top-level index space. Legion only permits "
            "top-level index spaces to be destroyed.", handle.get_id(),
            get_task_name(), get_unique_id())
      // Check to see if this is one that we should be allowed to destory
      bool has_created = true;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexSpace,unsigned>::iterator finder = 
          created_index_spaces.find(handle);
        if (finder != created_index_spaces.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_index_spaces.erase(finder);
          else
            return;
        }
        else
          has_created = false;
      }
      if (!has_created)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy index space %x in task %s (UID %lld) "
            "which is not the task that made the index space or one of its "
            "ancestor tasks. Index space deletions must be lexicographically "
            "scoped by the task tree.", handle.get_id(), 
            get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      std::set<RtEvent> preconditions;
      runtime->forest->destroy_index_space(handle,
            runtime->address_space, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    } 

    //--------------------------------------------------------------------------
    void LeafContext::destroy_index_partition(IndexPartition handle,
               const bool unordered, const bool recurse, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Check to see if this is one that we should be allowed to destory
      bool has_created = true;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexPartition,unsigned>::iterator finder = 
          created_index_partitions.find(handle);
        if (finder != created_index_partitions.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
          {
            created_index_partitions.erase(finder);
            if (recurse)
            {
              // Remove any other partitions that this partition dominates
              for (std::map<IndexPartition,unsigned>::iterator it = 
                    created_index_partitions.begin(); it !=
                    created_index_partitions.end(); /*nothing*/)
              {
                if ((handle.get_tree_id() == it->first.get_tree_id()) &&
                    runtime->forest->is_dominated_tree_only(it->first, handle))
                {
#ifdef DEBUG_LEGION
                  assert(it->second > 0);
#endif
                  if (--it->second == 0)
                  {
                    std::map<IndexPartition,unsigned>::iterator 
                      to_delete = it++;
                    created_index_partitions.erase(to_delete);
                  }
                  else
                    it++;
                }
                else
                  it++;
              }
            }
          }
          else
            return;
        }
        else
          has_created = false;
      }
      if (!has_created)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy index partition %x in task %s (UID %lld) "
            "which is not the task that made the index space or one of its "
            "ancestor tasks. Index space deletions must be lexicographically "
            "scoped by the task tree.", handle.get_id(), 
            get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
      log_index.debug("Destroying index partition %x in task %s (ID %lld)",
                      handle.id, get_task_name(), get_unique_id());
#endif
      std::set<RtEvent> preconditions;
      runtime->forest->destroy_index_partition(handle, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_equal_partition(
                                             IndexSpace parent,
                                             IndexSpace color_space,
                                             size_t granularity, Color color,
                                             const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EQUAL_PARTITION_CREATION,
        "Illegal equal partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_weights(IndexSpace parent,
                                                const FutureMap &weights,
                                                IndexSpace color_space,
                                                size_t granularity, Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EQUAL_PARTITION_CREATION,
        "Illegal create partition by weights performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_union(
                                          IndexSpace parent,
                                          IndexPartition handle1,
                                          IndexPartition handle2,
                                          IndexSpace color_space,
                                          PartitionKind kind, Color color,
                                          const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_UNION_PARTITION_CREATION,
        "Illegal union partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
                                                IndexSpace parent,
                                                IndexPartition handle1,
                                                IndexPartition handle2,
                                                IndexSpace color_space,
                                                PartitionKind kind, Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERSECTION_PARTITION_CREATION,
        "Illegal intersection partition creation performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_intersection(
                                                IndexSpace parent,
                                                IndexPartition partition,
                                                PartitionKind kind, Color color,
                                                bool dominates,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_INTERSECTION_PARTITION_CREATION,
        "Illegal intersection partition creation performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_difference(
                                                      IndexSpace parent,
                                                      IndexPartition handle1,
                                                      IndexPartition handle2,
                                                      IndexSpace color_space,
                                                      PartitionKind kind,
                                                      Color color,
                                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DIFFERENCE_PARTITION_CREATION,
        "Illegal difference partition creation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    Color LeafContext::create_cross_product_partitions(IndexPartition handle1,
                                                       IndexPartition handle2,
                                   std::map<IndexSpace,IndexPartition> &handles,
                                                       PartitionKind kind,
                                                       Color color,
                                                       const char *provenance)
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
                                         MapperID id, MappingTagID tag,
                                         const UntypedBuffer &marg,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_ASSOCIATION,
        "Illegal create association performed in leaf task "
                     "%s (ID %lld)", get_task_name(),get_unique_id())
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_restricted_partition(
                                                IndexSpace parent,
                                                IndexSpace color_space,
                                                const void *transform,
                                                size_t transform_size,
                                                const void *extent,
                                                size_t extent_size,
                                                PartitionKind part_kind,
                                                Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_RESTRICTED_PARTITION,
        "Illegal create restricted partition performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_domain(
                                                IndexSpace parent,
                                                const FutureMap &domains,
                                                IndexSpace color_space,
                                                bool perform_intersections,
                                                PartitionKind part_kind,
                                                Color color,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_BY_DOMAIN,
          "Illegal create partition by domain performed in leaf "
          "task %s (UID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_field(
                                                LogicalRegion handle,
                                                LogicalRegion parent_priv,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                Color color,
                                                MapperID id, MappingTagID tag,
                                                PartitionKind part_kind,
                                                const UntypedBuffer &marg,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_FIELD,
        "Illegal partition by field performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image(
                                              IndexSpace handle,
                                              LogicalPartition projection,
                                              LogicalRegion parent,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color,
                                              MapperID id, MappingTagID tag,
                                              const UntypedBuffer &marg,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_IMAGE,
        "Illegal partition by image performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_image_range(
                                              IndexSpace handle,
                                              LogicalPartition projection,
                                              LogicalRegion parent,
                                              FieldID fid,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color,
                                              MapperID id, MappingTagID tag,
                                              const UntypedBuffer &marg,
                                              const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_IMAGE_RANGE,
        "Illegal partition by image range performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage(
                                                IndexPartition projection,
                                                LogicalRegion handle,
                                                LogicalRegion parent,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                PartitionKind part_kind,
                                                Color color,
                                                MapperID id, MappingTagID tag,
                                                const UntypedBuffer &marg,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_PREIMAGE,
        "Illegal partition by preimage performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_partition_by_preimage_range(
                                                IndexPartition projection,
                                                LogicalRegion handle,
                                                LogicalRegion parent,
                                                FieldID fid,
                                                IndexSpace color_space,
                                                PartitionKind part_kind,
                                                Color color,
                                                MapperID id, MappingTagID tag,
                                                const UntypedBuffer &marg,
                                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_PARTITION_PREIMAGE_RANGE,
        "Illegal partition by preimage range performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexPartition LeafContext::create_pending_partition(
                                              IndexSpace parent,
                                              IndexSpace color_space,
                                              PartitionKind part_kind,
                                              Color color, const char *prov)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_PENDING_PARTITION,
        "Illegal create pending partition performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexPartition::NO_PART;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles,
                                                     const char *provenance) 
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_UNION,
        "Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_union(IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                                     IndexPartition handle,
                                                     const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_UNION,
        "Illegal create index space union performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                        const std::vector<IndexSpace> &handles,
                                                      const char *provenance) 
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_INTERSECTION,
        "Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_intersection(
                                                     IndexPartition parent,
                                                     const void *realm_color,
                                                     TypeTag type_tag,
                                                     IndexPartition handle,
                                                     const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_INTERSECTION,
        "Illegal create index space intersection performed in "
                     "leaf task %s (ID %lld)", get_task_name(),get_unique_id())
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace LeafContext::create_index_space_difference(
                                                  IndexPartition parent,
                                                  const void *realm_color,
                                                  TypeTag type_tag,
                                                  IndexSpace initial,
                                          const std::vector<IndexSpace> &handles,
                                                  const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CREATE_INDEX_SPACE_DIFFERENCE,
        "Illegal create index space difference performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return IndexSpace::NO_SPACE;
    } 

    //--------------------------------------------------------------------------
    FieldSpace LeafContext::create_field_space(const std::vector<Future> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         CustomSerdezID serdez_id,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION2,
       "Illegal deferred field allocations performed in leaf task %s (ID %lld)",
       get_task_name(), get_unique_id())
      return FieldSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_field_space(FieldSpace handle, 
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      // Check to see if this is one that we should be allowed to destory
      bool has_created = true;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<FieldSpace,unsigned>::iterator finder = 
          created_field_spaces.find(handle);
        if (finder != created_field_spaces.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_field_spaces.erase(finder);
          else
            return;
        }
        else
          has_created = false;
      }
      if (!has_created)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy field space %x in task %s (UID %lld) "
            "which is not the task that made the field space or one of its "
            "ancestor tasks. Field space deletions must be lexicographically "
            "scoped by the task tree.", handle.get_id(), 
            get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
      log_field.debug("Destroying field space %x in task %s (ID %lld)", 
                      handle.id, get_task_name(), get_unique_id());
#endif
      std::set<RtEvent> preconditions;
      runtime->forest->destroy_field_space(handle, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_field(FieldAllocatorImpl *allocator,FieldSpace space, 
                                 FieldID fid, const bool unordered,
                                 const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      bool has_created = true;
      {
        AutoLock priv_lock(privilege_lock);
        const std::pair<FieldSpace,FieldID> key(space, fid);
        std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
          created_fields.find(key);
        if (finder != created_fields.end())
          created_fields.erase(finder);
        else // No need to check for local fields since we can't make them
          has_created = false;
      }
      if (!has_created)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to deallocate field %d in field space %x in task %s "
            "(UID %lld) which is not the task that allocated the field "
            "or one of its ancestor tasks. Field deallocations must be " 
            "lexicographically scoped by the task tree.", fid, space.id,
            get_task_name(), get_unique_id())
      // If the allocator is not ready we need to wait for it here
      if (allocator->ready_event.exists() && 
          !allocator->ready_event.has_triggered())
        allocator->ready_event.wait();
      // Free the indexes first and immediately
      std::vector<FieldID> to_free(1,fid);
      runtime->forest->free_field_indexes(space, to_free, RtEvent::NO_RT_EVENT);
      // We can free this field immediately
      std::set<RtEvent> preconditions;
      runtime->forest->free_field(space, fid, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    void LeafContext::free_fields(FieldAllocatorImpl *allocator, 
                                  FieldSpace space, 
                                  const std::set<FieldID> &to_free,
                                  const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      long bad_fid = -1;
      {
        AutoLock priv_lock(privilege_lock);
        for (std::set<FieldID>::const_iterator it = 
              to_free.begin(); it != to_free.end(); it++)
        {
          const std::pair<FieldSpace,FieldID> key(space, *it);
          std::set<std::pair<FieldSpace,FieldID> >::iterator finder = 
            created_fields.find(key);
          if (finder == created_fields.end())
          {
            // No need to check for local fields since we know
            // that leaf tasks are not allowed to make them
            bad_fid = *it;
            break;
          }
          else
            created_fields.erase(finder);
        }
      }
      if (bad_fid != -1)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to deallocate field %ld in field space %x in task %s "
            "(UID %lld) which is not the task that allocated the field "
            "or one of its ancestor tasks. Field deallocations must be " 
            "lexicographically scoped by the task tree.", bad_fid, space.id,
            get_task_name(), get_unique_id())
      // If the allocator is not ready we need to wait for it here
      if (allocator->ready_event.exists() && 
          !allocator->ready_event.has_triggered())
        allocator->ready_event.wait();
      // Free the indexes first and immediately
      const std::vector<FieldID> field_vec(to_free.begin(), to_free.end());
      runtime->forest->free_field_indexes(space,field_vec,RtEvent::NO_RT_EVENT);
      // We can free these fields immediately
      std::set<RtEvent> preconditions;
      runtime->forest->free_fields(space, field_vec, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    FieldID LeafContext::allocate_field(FieldSpace space, 
                                        const Future &field_size,
                                        FieldID fid, bool local,
                                        CustomSerdezID serdez_id,
                                        const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION,
        "Illegal deferred field allocation performed in leaf task %s (ID %lld)",
        get_task_name(), get_unique_id())
      return 0;
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events,
                                     const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION,
          "Illegal local field allocation performed in leaf task %s (ID %lld)",
          get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_fields(FieldSpace space,
                                      const std::vector<Future> &sizes,
                                      std::vector<FieldID> &resuling_fields,
                                      bool local, CustomSerdezID serdez_id,
                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION2,
       "Illegal deferred field allocations performed in leaf task %s (ID %lld)",
       get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events,
                                   const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_NONLOCAL_FIELD_ALLOCATION2,
          "Illegal local field allocations performed in leaf task %s (ID %lld)",
          get_task_name(), get_unique_id())
    } 

    //--------------------------------------------------------------------------
    void LeafContext::destroy_logical_region(LogicalRegion handle,
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      AutoRuntimeCall call(this);
      if (!handle.exists())
        return;
      // Check to see if this is a top-level logical region, if not then
      // we shouldn't even be destroying it
      if (!runtime->forest->is_top_level_region(handle))
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy logical region (%x,%x,%x in task %s "
            "(UID %lld) which is not a top-level logical region. Legion only "
            "permits top-level logical regions to be destroyed.", 
            handle.index_space.id, handle.field_space.id, handle.tree_id,
            get_task_name(), get_unique_id())
      // Check to see if this is one that we should be allowed to destory
      bool has_created = true;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<LogicalRegion,unsigned>::iterator finder = 
          created_regions.find(handle);
        if (finder != created_regions.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second > 0);
#endif
          if (--finder->second == 0)
            created_regions.erase(finder);
          else
            return;
        }
        else
          has_created = false;
      }
      if (!has_created)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_RESOURCE_DESTRUCTION,
            "Illegal call to destroy logical region (%x,%x,%x) in task %s "
            "(UID %lld) which is not the task that made the logical region "
            "or one of its ancestor tasks. Logical region deletions must be " 
            "lexicographically scoped by the task tree.", handle.index_space.id,
            handle.field_space.id, handle.tree_id,
            get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
      log_region.debug("Deleting logical region (%x,%x) in task %s (ID %lld)",
                       handle.index_space.id, handle.field_space.id, 
                       get_task_name(), get_unique_id());
#endif
      std::set<RtEvent> preconditions;
      runtime->forest->destroy_logical_region(handle, preconditions);
      if (!preconditions.empty())
      {
        AutoLock l_lock(leaf_lock);
        execution_events.insert(preconditions.begin(), preconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    void LeafContext::get_local_field_set(const FieldSpace handle,
                                          const std::set<unsigned> &indexes,
                                          std::set<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::get_local_field_set(const FieldSpace handle,
                                          const std::set<unsigned> &indexes,
                                          std::vector<FieldID> &to_set) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_physical_region(const RegionRequirement &req,
          bool mapped, MapperID mid, MappingTagID tag, ApUserEvent &unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!unmap_event.exists());
#endif
      PhysicalRegionImpl *impl = new PhysicalRegionImpl(req, 
          RtEvent::NO_RT_EVENT, ApEvent::NO_AP_EVENT, 
          ApUserEvent::NO_AP_USER_EVENT, mapped, this, mid, tag, 
          true/*leaf region*/, virtual_mapped, runtime);
      physical_regions.push_back(PhysicalRegion(impl));
      if (mapped)
        impl->set_references(physical_instances, true/*safe*/);
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_task(const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (launcher.enable_inlining)
      {
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_task_false(launcher);

        IndividualTask *task = runtime->get_available_individual_task(); 
        InnerContext *parent = owner_task->get_context();
        Future result = task->initialize_task(parent, launcher, false/*track*/);
        inline_child_task(task);
        return result;
      }
      else
      {
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_TASK_CALL,
          "Illegal execute task call performed in leaf task %s "
                       "(ID %lld)", get_task_name(), get_unique_id())
        return Future();
      }
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::execute_index_space(
                                              const IndexTaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      if (!launcher.must_parallelism && launcher.enable_inlining)
      {
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_index_task_false(launcher); 
        IndexTask *task = runtime->get_available_index_task();
        InnerContext *parent = owner_task->get_context();
        IndexSpace launch_space = launcher.launch_space;
        if (!launch_space.exists())
          launch_space = find_index_launch_space(launcher.launch_domain,
                                                 launcher.provenance);
        FutureMap result = task->initialize_task(parent, launcher, 
                                                 launch_space, false/*track*/);
        inline_child_task(task);
        return result;
      }
      else
      {
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
          "Illegal execute index space call performed in leaf "
                       "task %s (ID %lld)", get_task_name(), get_unique_id())
        return FutureMap();
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::execute_index_space(const IndexTaskLauncher &launcher, 
                                        ReductionOpID redop, bool deterministic)
    //--------------------------------------------------------------------------
    {
      if (!launcher.must_parallelism && launcher.enable_inlining)
      {
        if (launcher.predicate == Predicate::FALSE_PRED)
          return predicate_index_task_reduce_false(launcher);
        IndexTask *task = runtime->get_available_index_task();
        InnerContext *parent = owner_task->get_context();
        IndexSpace launch_space = launcher.launch_space;
        if (!launch_space.exists())
          launch_space = find_index_launch_space(launcher.launch_domain,
                                                 launcher.provenance);
        Future result = task->initialize_task(parent, launcher, launch_space, 
                                        redop, deterministic, false/*track*/);
        inline_child_task(task);
        return result;
      }
      else
      {
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
          "Illegal execute index space call performed in leaf "
                       "task %s (ID %lld)", get_task_name(), get_unique_id())
        return Future();
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::reduce_future_map(const FutureMap &future_map,
                ReductionOpID redop, bool deterministic, const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal reduce future map call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(IndexSpace domain,
                                const std::map<DomainPoint,UntypedBuffer> &data,
                                bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal construct future map call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(const Domain &domain,
                                const std::map<DomainPoint,UntypedBuffer> &data,
                                bool collective, ShardingID sid, bool implicit)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal construct future map call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(IndexSpace domain,
                                    const std::map<DomainPoint,Future> &futures,
                                    bool internal, bool collective,
                                    ShardingID sid, bool implicit,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal construct future map call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
    }

    //--------------------------------------------------------------------------
    FutureMap LeafContext::construct_future_map(const Domain &domain,
                                    const std::map<DomainPoint,Future> &futures,
                                    bool internal, bool collective,
                                    ShardingID sid, bool implicit,
                                    const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_EXECUTE_INDEX_SPACE,
        "Illegal construct future map call performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return FutureMap();
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
    ApEvent LeafContext::remap_region(const PhysicalRegion &region,
                                      const char *provenance)
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
    void LeafContext::unmap_all_regions(bool external)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_UNMAP_OPERATION,
        "Illegal unmap_all_regions call performed in leaf task %s "
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
    ExternalResources LeafContext::attach_resources(
                                            const IndexAttachLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_ATTACH_RESOURCE_OPERATION,
        "Illegal attach resources operation performed in leaf "
                     "task %s (ID %lld)", get_task_name(), get_unique_id())
      return ExternalResources();
    }
    
    //--------------------------------------------------------------------------
    Future LeafContext::detach_resource(PhysicalRegion region, const bool flush,
                                   const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_RESOURCE_OPERATION,
        "Illegal detach resource operation performed in leaf "
                      "task %s (ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::detach_resources(ExternalResources resources,
                 const bool flush, const bool unordered, const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_RESOURCE_OPERATION,
        "Illegal index detach resource operation performed in leaf "
                      "task %s (ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    void LeafContext::progress_unordered_operations(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_DETACH_RESOURCE_OPERATION,
        "Illegal progress unordered operations performed in leaf "
                      "task %s (ID %lld)", get_task_name(), get_unique_id())
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
    Future LeafContext::select_tunable_value(const TunableLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
        "Illegal tunable value operation request in leaf task %s (ID %lld)",
        get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_mapping_fence(const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_MAPPING_FENCE_CALL,
        "Illegal legion mapping fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    Future LeafContext::issue_execution_fence(const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_EXECUTION_FENCE_CALL,
        "Illegal Legion execution fence call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    void LeafContext::complete_frame(const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_COMPLETE_FRAME_CALL,
        "Illegal Legion complete frame call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const Future &f,
                                            const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (f.impl == NULL)
        return Predicate::FALSE_PRED;
      // Always eagerly evaluate predicates in leaf contexts
      bool valid = false;
      const bool value = f.impl->get_boolean_value(valid);
#ifdef DEBUG_LEGION
      assert(valid); // all futures should be ready
#endif
      if (value)
        return Predicate::TRUE_PRED;
      else
        return Predicate::FALSE_PRED;
    }

    //--------------------------------------------------------------------------
    Predicate LeafContext::predicate_not(const Predicate &p,
                                         const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
        return Predicate::FALSE_PRED;
      else if (p == Predicate::FALSE_PRED)
        return Predicate::TRUE_PRED;
      else // should never get here, all predicates should be eagerly evaluated
        assert(false);  
      return Predicate::TRUE_PRED;
    }
    
    //--------------------------------------------------------------------------
    Predicate LeafContext::create_predicate(const PredicateLauncher &launcher)
    //--------------------------------------------------------------------------
    {
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
        for (std::vector<Predicate>::const_iterator it = 
              launcher.predicates.begin(); it != 
              launcher.predicates.end(); it++)
        {
          if ((*it) == Predicate::FALSE_PRED)
            return Predicate::FALSE_PRED;
          else if ((*it) == Predicate::TRUE_PRED)
            continue;
          else // should never get here, 
            // all predicates should be eagerly evaluated
            assert(false);
        }
        return Predicate::TRUE_PRED;
      }
      else
      {
        // Check for short circuit cases
        for (std::vector<Predicate>::const_iterator it = 
              launcher.predicates.begin(); it != 
              launcher.predicates.end(); it++)
        {
          if ((*it) == Predicate::TRUE_PRED)
            return Predicate::TRUE_PRED;
          else if ((*it) == Predicate::FALSE_PRED)
            continue;
          else // should never get here, 
            // all predicates should be eagerly evaluated
            assert(false);
        }
        return Predicate::FALSE_PRED;
      }
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_predicate_future(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      if (p == Predicate::TRUE_PRED)
      {
        Future result = runtime->help_create_future(this, ApEvent::NO_AP_EVENT);
        const bool value = true;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
        return result;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        Future result = runtime->help_create_future(this, ApEvent::NO_AP_EVENT);
        const bool value = false;
        result.impl->set_result(&value, sizeof(value), false/*owned*/);
        return result;
      }
      else // should never get here, all predicates should be eagerly evaluated
        assert(false);
      return Future();
    }

    //--------------------------------------------------------------------------
    size_t LeafContext::register_new_child_operation(Operation *op,
                    const std::vector<StaticDependence> *dependences)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    size_t LeafContext::register_new_close_operation(CloseOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    size_t LeafContext::register_new_summary_operation(TraceSummaryOp *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    bool LeafContext::add_to_dependence_queue(Operation *op, bool unordered)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void LeafContext::add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                             const void *result, size_t size, 
                                             PhysicalInstance instance,
                                             FutureFunctor *callback_functor,
                                             bool own_functor)
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
    ApEvent LeafContext::register_implicit_dependences(Operation *op)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void LeafContext::perform_fence_analysis(Operation *op,
                 std::set<ApEvent> &preconditions, bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::update_current_fence(FenceOp *op, 
                                           bool mapping, bool execution)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::update_current_implicit(Operation *op) 
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
    void LeafContext::begin_trace(TraceID tid, bool logical_only,
        bool static_trace, const std::set<RegionTreeID> *trees,
        bool deprecated, const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_BEGIN_TRACE,
        "Illegal Legion begin trace call in leaf task %s "
                     "(ID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::end_trace(TraceID tid, bool deprecated,
                                const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_LEGION_END_TRACE,
        "Illegal Legion end trace call in leaf task %s (ID %lld)",
                     get_task_name(), get_unique_id())
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
    void LeafContext::decrement_pending(TaskOp *child)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void LeafContext::decrement_pending(bool need_deferral)
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
    InnerContext* LeafContext::find_parent_physical_context(unsigned index,
                                                           LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_outermost_local_context(InnerContext *prev)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    InnerContext* LeafContext::find_top_context(InnerContext *previous)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void LeafContext::initialize_region_tree_contexts(
                       const std::vector<RegionRequirement> &clone_requirements,
                       const std::vector<ApUserEvent> &unmap_events,
                       std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      for (unsigned idx = 0; idx < unmap_events.size(); idx++)
        assert(!unmap_events[idx].exists());
#endif
#endif
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
     PhysicalInstance deferred_result_instance, FutureFunctor *callback_functor)
    //--------------------------------------------------------------------------
    {
      // No local regions or fields permitted in leaf tasks
      if (overhead_tracker != NULL)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff = current - previous_profiling_time;
        overhead_tracker->application_time += diff;
      }
      if (!task_local_instances.empty())
        release_task_local_instances(deferred_result_instance);
      if (!index_launch_spaces.empty())
      {
        for (std::map<Domain,IndexSpace>::const_iterator it = 
              index_launch_spaces.begin(); it != 
              index_launch_spaces.end(); it++)
          destroy_index_space(it->second, false/*unordered*/,
                              true/*recurse*/, NULL/*provenance*/);
      }
      // No need to unmap the physical regions, they never had events
      if (!execution_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(execution_events);
        wait_on.wait();
      }
      // Mark that we are done executing this operation
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
      const RtEvent effects_done(!inline_task ?
          Processor::get_current_finish_event() : Realm::Event::NO_EVENT);
      if (deferred_result_instance.exists())
        parent_ctx->add_to_post_task_queue(this, effects_done,
                                           res, res_size, 
                                           deferred_result_instance);
      else if (callback_functor != NULL)
      {
        if (owner_task->is_reducing_future())
        {
          // If we're reducing this future value then just do the callback
          // now since there is no point in deferring it
          const size_t callback_size = 
            callback_functor->callback_get_future_size();
          void *buffer = malloc(callback_size);
          callback_functor->callback_pack_future(buffer, callback_size);
          callback_functor->callback_release_future();
          if (owned)
            delete callback_functor;
          parent_ctx->add_to_post_task_queue(this, effects_done, 
                                             buffer, callback_size);
        }
        else
          parent_ctx->add_to_post_task_queue(this, effects_done, res, res_size,
                            deferred_result_instance, callback_functor, owned);
      }
      else if (!owned)
      {
        if (res_size > 0)
        {
          void *result_copy = malloc(res_size);
          memcpy(result_copy, res, res_size);
          parent_ctx->add_to_post_task_queue(this, effects_done,
                                             result_copy, res_size); 
        }
        else
          parent_ctx->add_to_post_task_queue(this, effects_done,
                                             res, res_size);
      }
      else
        parent_ctx->add_to_post_task_queue(this, effects_done, res, res_size);
      if (!inline_task)
#ifdef DEBUG_LEGION
        runtime_ptr->decrement_total_outstanding_tasks(owner_task_id, 
                                                       false/*meta*/);
#else
        runtime_ptr->decrement_total_outstanding_tasks();
#endif
    }

    //--------------------------------------------------------------------------
    void LeafContext::post_end_task(const void *res, size_t res_size,
                                    bool owned, FutureFunctor *callback_functor)
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
      single_task->handle_future(res, res_size, owned, 
                                 callback_functor, executing_processor);
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
        owner_task->trigger_children_complete(ApEvent::NO_AP_EVENT);
      if (need_commit)
        owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_lock(Lock l)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal destroy lock performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    Grant LeafContext::acquire_grant(const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal acquire grant performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
      return Grant();
    }

    //--------------------------------------------------------------------------
    void LeafContext::release_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal release grant performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal destroy phase barrier performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    PhaseBarrier LeafContext::advance_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal advance phase barrier performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
      return PhaseBarrier();
    }

    //--------------------------------------------------------------------------
    DynamicCollective LeafContext::create_dynamic_collective(
                                       unsigned arrivals, ReductionOpID redop,
                                       const void *init_value, size_t init_size)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal create dynamic collective performed in leaf task %s "
          "(UID %lld)", get_task_name(), get_unique_id())
      return DynamicCollective();
    }

    //--------------------------------------------------------------------------
    void LeafContext::destroy_dynamic_collective(DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal destroy dynamic collective performed in leaf task %s "
          "(UID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::arrive_dynamic_collective(DynamicCollective dc,
                                const void *buffer, size_t size, unsigned count)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal arrive dynamic collective performed in leaf task %s "
          "(UID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    void LeafContext::defer_dynamic_collective_arrival(DynamicCollective dc,
                                           const Future &future, unsigned count)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal defer dynamic collective performed in leaf task %s "
          "(UID %lld)", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    Future LeafContext::get_dynamic_collective_result(DynamicCollective dc,
                                                      const char *provenance)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal get dynamic collective performed in leaf task %s (UID %lld)",
          get_task_name(), get_unique_id())
      return Future();
    }

    //--------------------------------------------------------------------------
    DynamicCollective LeafContext::advance_dynamic_collective(
                                                           DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_LEAF_TASK_VIOLATION,
          "Illegal advance dynamic collective performed in leaf task %s "
          "(UID %lld)", get_task_name(), get_unique_id())
      return DynamicCollective();
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

  };
};

// EOF

