/* Copyright 2012 Stanford University
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


#include "legion_ops.h"
#include "region_tree.h"
#include "legion_logging.h"
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
    // Generalized Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GeneralizedOperation::GeneralizedOperation(HighLevelRuntime *rt)
      : Lockable(), Mappable(), active(false), context_owner(false), forest_ctx(NULL), 
        generation(0), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    GeneralizedOperation::~GeneralizedOperation(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool GeneralizedOperation::activate_base(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool result = !active;
      // Check to see if we can activate this operation 
      if (result)
      {
        active = true;
        if (parent != NULL)
        {
          context_owner = false;
          forest_ctx = parent->forest_ctx;
        }
        else
        {
          context_owner = true;
          forest_ctx = runtime->create_region_forest();
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(forest_ctx != NULL);
#endif
        unique_id = runtime->get_unique_op_id();
        outstanding_dependences = 0;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::deactivate_base(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
#endif
      // Need to be holidng the lock to update generation
      lock();
      // Whenever we deactivate, up the generation count
      generation++;
      unlock();
      if (context_owner)
      {
        runtime->destroy_region_forest(forest_ctx);
      }
      forest_ctx = NULL;
      context_owner = false;
      mapper = NULL;
#ifdef LOW_LEVEL_LOCKS
      mapper_lock = Lock::NO_LOCK;
#endif
      active = false;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::lock_context(bool exclusive /*= true*/) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->lock_context(exclusive);
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::unlock_context(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->unlock_context();
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void GeneralizedOperation::assert_context_locked(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->assert_locked();
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::assert_context_not_locked(void) const
    //--------------------------------------------------------------------------
    {
      forest_ctx->assert_not_locked();
    }
#endif

    //--------------------------------------------------------------------------
    UniqueID GeneralizedOperation::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_id;
    }

    //--------------------------------------------------------------------------
    bool GeneralizedOperation::is_ready(void)
    //--------------------------------------------------------------------------
    {
      lock();
      bool ready = (outstanding_dependences == 0);
      unlock();
      return ready;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::notify(void) 
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
      assert(outstanding_dependences > 0);
#endif
      outstanding_dependences--;
      bool ready = (outstanding_dependences == 0);
      unlock();
      if (ready)
      {
        trigger();
      }
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::clone_generalized_operation_from(GeneralizedOperation *rhs)
    //--------------------------------------------------------------------------
    {
      this->context_owner = false;
      this->forest_ctx = rhs->forest_ctx;
      this->unique_id = rhs->unique_id;
    }

    //--------------------------------------------------------------------------
    LegionErrorType GeneralizedOperation::verify_requirement(const RegionRequirement &req, 
                                                              FieldID &bad_field, size_t &bad_size, unsigned &bad_idx)
    //--------------------------------------------------------------------------
    {
      // First make sure that all the privilege fields are valid for the given
      // fields space of the region or partition
      lock_context();
      FieldSpace sp = (req.handle_type == SINGULAR) ? req.region.field_space : req.partition.field_space;
      for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
            it != req.privilege_fields.end(); it++)
      {
        if (!forest_ctx->has_field(sp, *it))
        {
          unlock_context();
          bad_field = *it;
          return ERROR_FIELD_SPACE_FIELD_MISMATCH;
        }
      }

      // Make sure that the requested node is a valid request
      if (req.handle_type == SINGULAR)
      {
        if (!forest_ctx->has_node(req.region, false/*strict*/))
        {
          unlock_context();
          return ERROR_INVALID_REGION_HANDLE;
        }
      }
      else
      {
        if (!forest_ctx->has_node(req.partition, false/*strict*/))
        {
          unlock_context();
          return ERROR_INVALID_PARTITION_HANDLE;
        }
      }
      unlock_context();

      // Then check that any instance fields are included in the privilege fields
      // Make sure that there are no duplicates in the instance fields
      std::set<FieldID> inst_duplicates;
      for (std::vector<FieldID>::const_iterator it = req.instance_fields.begin();
            it != req.instance_fields.end(); it++)
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
      if ((req.handle_type == PROJECTION) && 
          (IS_WRITE(req)))
      {
        lock_context();
        if (!forest_ctx->is_disjoint(req.partition))
        {
          unlock_context();
          return ERROR_NON_DISJOINT_PARTITION;
        }
        unlock_context();
      }
      
      // Finally check that the type matches the instance fields
      // Only do this if the user requested it
      if (req.inst_type != 0)
      {
        const TypeTable &tt = HighLevelRuntime::get_type_table();  
        TypeTable::const_iterator tt_it = tt.find(req.inst_type);
        if (tt_it == tt.end())
          return ERROR_INVALID_TYPE_HANDLE;
        const Structure &st = tt_it->second;
        if (st.field_sizes.size() != req.instance_fields.size())
          return ERROR_TYPE_INST_MISSIZE;
        lock_context();
        for (unsigned idx = 0; idx < st.field_sizes.size(); idx++)
        {
          if (st.field_sizes[idx] != forest_ctx->get_field_size(sp, req.instance_fields[idx]))
          {
            bad_size = forest_ctx->get_field_size(sp, req.instance_fields[idx]);
            unlock_context();
            bad_field = req.instance_fields[idx];
            bad_idx = idx;
            return ERROR_TYPE_INST_MISMATCH;
          }
        }
        unlock_context();
      }
      return NO_ERROR;
    }

    //--------------------------------------------------------------------------
    size_t GeneralizedOperation::compute_operation_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(UniqueID);
      return result;
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::pack_operation(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<UniqueID>(unique_id);
    }

    //--------------------------------------------------------------------------
    void GeneralizedOperation::unpack_operation(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<UniqueID>(unique_id);
    }

    /////////////////////////////////////////////////////////////
    // Mapping Operaiton 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingOperation::MappingOperation(HighLevelRuntime *rt)
      : GeneralizedOperation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MappingOperation::~MappingOperation(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MappingOperation::initialize(Context ctx, const RegionRequirement &req, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
      assert(ctx != NULL);
#endif
      parent_ctx = ctx;
      requirement = req;
      mapped_event = UserEvent::create_user_event();
      ready_event = Event::NO_EVENT;
      unmapped_event = UserEvent::create_user_event();
      tag = t;
      // Check privileges for the region requirement
      check_privilege();
#ifdef LEGION_SPY
      LegionSpy::log_mapping_operation(unique_id, parent_ctx->get_unique_id(), parent_ctx->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
      LegionSpy::log_logical_requirement(unique_id,0,true,req.region.index_space.id, req.region.field_space.id,
                                        req.region.tree_id, req.privilege, req.prop, req.redop);
      LegionSpy::log_requirement_fields(unique_id, 0, req.privilege_fields);
#endif
      parent_ctx->register_child_map(this);
    }
    
    //--------------------------------------------------------------------------
    void MappingOperation::initialize(Context ctx, unsigned idx, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(active);
      assert(ctx != NULL);
#endif
      parent_ctx = ctx;
      requirement = parent_ctx->get_region_requirement(idx);
      mapped_event = UserEvent::create_user_event();
      ready_event = Event::NO_EVENT;
      unmapped_event = UserEvent::create_user_event();
      tag = t;
      // Check privileges for the region requirement
      check_privilege();
#ifdef LEGION_SPY
      LegionSpy::log_mapping_operation(unique_id, parent_ctx->get_unique_id(), parent_ctx->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
      LegionSpy::log_logical_requirement(unique_id,0,true,requirement.region.index_space.id, requirement.region.field_space.id,
                                        requirement.region.tree_id, requirement.privilege, requirement.prop, requirement.redop);
      LegionSpy::log_requirement_fields(unique_id, 0, requirement.privilege_fields);
#endif
      parent_ctx->register_child_map(this, idx);
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::is_valid(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      if (mapped_event.has_triggered())
        return ready_event.has_triggered();
      return false;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::wait_until_valid(GenerationID gen_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      // Make sure to wait until we've mapped
      mapped_event.wait();
      // Then wait until we're ready
      ready_event.wait();
    }

    //--------------------------------------------------------------------------
    LogicalRegion MappingOperation::get_logical_region(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
#endif
      return requirement.region;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance MappingOperation::get_physical_instance(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
      assert(mapped_event.has_triggered());
#endif
      return physical_instance.get_instance();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion MappingOperation::get_physical_region(void)
    //--------------------------------------------------------------------------
    {
      return PhysicalRegion(this, generation);
    }
    
    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
      MappingOperation::get_accessor(GenerationID gen_id) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
      assert(mapped_event.has_triggered());
#endif
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> result = 
        physical_instance.get_manager()->get_accessor();
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(requirement.get_accessor_privilege());
#endif
      return result;
    }
    
    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> 
      MappingOperation::get_field_accessor(GenerationID gen_id, FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (gen_id != generation)
      {
        log_region(LEVEL_ERROR,"Accessing stale inline mapping operation that has been invalided");
        exit(ERROR_STALE_INLINE_MAPPING_ACCESS);
      }
      assert(mapped_event.has_triggered());
#endif
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> result = 
        physical_instance.get_manager()->get_field_accessor(fid);
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(requirement.get_accessor_privilege());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Event MappingOperation::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      return activate_base(parent);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // Need to unmap this operation before we can deactivate it
      unmapped_event.trigger();
      // Now we can go about removing our references
      parent_ctx->lock_context();
      physical_instance.remove_reference(unique_id, false/*strict*/); 
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        source_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
      }
      parent_ctx->unlock_context();
      physical_instance = InstanceRef(); // virtual ref
      source_copy_instances.clear();

      deactivate_base();
#ifndef INORDER_EXECUTION
      Context parent = parent_ctx;
#endif
      parent_ctx = NULL;
      tag = 0;
      map_dependent_waiters.clear();
#ifndef INORDER_EXECUTION
      runtime->notify_operation_complete(parent);
#endif
      runtime->free_mapping(this);
    }

    //--------------------------------------------------------------------------
    void MappingOperation::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_dependence(parent_ctx->get_unique_id(), parent_ctx->ctx_id, 
                                        runtime->utility_proc.id, parent_ctx->get_gen(), prev.op->get_unique_id(),
                                        prev.idx, get_unique_id(), idx, dtype);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // Need to hold the lock to avoid destroying data during deactivation
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation); // make sure the generations make sense
      assert(idx == 0);
#endif
      bool result;
      do {
        if (gen < generation)
        {
          result = false; // this mapping operation has already been recycled
          break;
        }
        // Check to see if we've already been mapped
        if (mapped_event.has_triggered())
        {
          result = false;
          break;
        }
        // Make sure we don't add it twice
        std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
          map_dependent_waiters.insert(waiter);
        result = added.second;
      } while (false);
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    { 
      lock_context();
      {
        RegionAnalyzer az(parent_ctx->ctx_id, this, 0/*idx*/, requirement);
        // Compute the path to the right place
        forest_ctx->compute_index_path(requirement.parent.index_space, requirement.region.index_space, az.path);
        forest_ctx->analyze_region(az);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    bool MappingOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;  
      forest_ctx->lock_context();
      ContextID phy_ctx = parent_ctx->find_enclosing_physical_context(requirement.parent);
      RegionMapper reg_mapper(parent_ctx, unique_id, phy_ctx, 0/*idx*/, requirement, mapper, mapper_lock, parent_ctx->get_executing_processor(), 
                              unmapped_event, unmapped_event, tag, false/*sanitizing*/,
                              true/*inline mapping*/, source_copy_instances);
      // Compute the path 
#ifdef DEBUG_HIGH_LEVEL
      bool result = 
#endif
      forest_ctx->compute_index_path(requirement.parent.index_space, requirement.region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
      assert(result);
#endif
      forest_ctx->map_region(reg_mapper, requirement.parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(reg_mapper.path.empty());
#endif
      physical_instance = reg_mapper.result;
      forest_ctx->unlock_context();

      if (!physical_instance.is_virtual_ref())
      {
        Event start_event = physical_instance.get_ready_event();
#ifdef LEGION_SPY
        LegionSpy::log_mapping_user(this->get_unique_id(), physical_instance.get_manager()->get_unique_id());
        if (!start_event.exists())
        {
          UserEvent new_start = UserEvent::create_user_event();
          new_start.trigger();
          start_event = new_start;
        }
#endif
        // Mapping successful  
        // Set the ready event and the physical instance
        if (physical_instance.has_required_lock())
        {
          // Issue lock acquire on ready event, issue unlock on unmap event
          Lock required_lock = physical_instance.get_required_lock();
          this->ready_event = required_lock.lock(0,true/*exclusive*/,start_event);
#ifdef LEGION_SPY
          if (!ready_event.exists())
          {
            UserEvent new_ready = UserEvent::create_user_event();
            new_ready.trigger();
            this->ready_event = new_ready;
          }
          LegionSpy::log_event_dependence(start_event, this->ready_event);
#endif
          required_lock.unlock(unmapped_event);
        }
        else
        {
          this->ready_event = start_event;
        }
#ifdef LEGION_SPY
        LegionSpy::log_map_events(get_unique_id(), ready_event, unmapped_event);
        LegionSpy::log_mapping_user(get_unique_id(), physical_instance.get_manager()->get_unique_id());
#endif
        // finally we can trigger the event saying that we're mapped
        mapped_event.trigger();
        // Notify all our waiters that we're mapped
        for (std::set<GeneralizedOperation*>::const_iterator it = map_dependent_waiters.begin();
              it != map_dependent_waiters.end(); it++)
        {
          (*it)->notify();
        }
        map_dependent_waiters.clear();
#ifdef INORDER_EXECUTION
        runtime->notify_operation_complete(parent_ctx);
#endif
      }
      else
      {
        // Mapping failed
        map_success = false;
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        mapper->notify_failed_mapping(parent_ctx, parent_ctx->get_executing_processor(),
                                      requirement, 0/*index*/, true/*inline mapping*/);
      }
      
      return map_success;
    }

    //--------------------------------------------------------------------------
    void MappingOperation::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Enqueue this operation with the runtime
      runtime->add_to_ready_queue(parent_ctx->get_executing_processor(), this);
    }

    //--------------------------------------------------------------------------
    MapperID MappingOperation::get_mapper_id(void) const
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_mapper_id();
    }

    //--------------------------------------------------------------------------
    void MappingOperation::check_privilege(void)
    //--------------------------------------------------------------------------
    {
      FieldID bad_field;
      size_t bad_size;
      unsigned bad_idx;
      if (requirement.handle_type == PROJECTION)
      {
        log_region(LEVEL_ERROR,"Projection region requirements are not permitted for inline mappings (in task %s)",
                                parent_ctx->variants->name);
        exit(ERROR_BAD_PROJECTION_USE);
      }
      LegionErrorType et = verify_requirement(requirement, bad_field, bad_size, bad_idx);
      // If that worked, then check the privileges with the parent context
      if (et == NO_ERROR)
        et = parent_ctx->check_privilege(requirement, bad_field);
      switch (et)
      {
        case NO_ERROR:
          break;
        case ERROR_INVALID_REGION_HANDLE:
          {
            log_region(LEVEL_ERROR,"Requirest for invalid region handle (%x,%d,%d) for inline mapping (ID %d)",
                requirement.region.index_space.id, requirement.region.field_space.id, requirement.region.tree_id, get_unique_id());
            exit (ERROR_INVALID_REGION_HANDLE);
          }
        case ERROR_FIELD_SPACE_FIELD_MISMATCH:
          {
            FieldSpace sp = (requirement.handle_type == SINGULAR) ? requirement.region.field_space : requirement.partition.field_space;
            log_region(LEVEL_ERROR,"Field %d is not a valid field of field space %d for inline mapping (ID %d)",
                                    bad_field, sp.id, get_unique_id());
            exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
          }
        case ERROR_INVALID_INSTANCE_FIELD:
          {
            log_region(LEVEL_ERROR,"Instance field %d is not one of the privilege fields for inline mapping (ID %d)",
                                    bad_field, get_unique_id());
            exit(ERROR_INVALID_INSTANCE_FIELD);
          }
        case ERROR_DUPLICATE_INSTANCE_FIELD:
          {
            log_region(LEVEL_ERROR, "Instance field %d is a duplicate for inline mapping (ID %d)",
                                  bad_field, get_unique_id());
            exit(ERROR_DUPLICATE_INSTANCE_FIELD);
          }
        case ERROR_INVALID_TYPE_HANDLE:
          {
            log_region(LEVEL_ERROR, "Type handle %d does not name a valid registered structure type for inline mapping (ID %d)",
                                    requirement.inst_type, get_unique_id());
            exit(ERROR_INVALID_TYPE_HANDLE);
          }
        case ERROR_TYPE_INST_MISSIZE:
          {
            TypeTable &tt = HighLevelRuntime::get_type_table();
            const Structure &st = tt[requirement.inst_type];
            log_region(LEVEL_ERROR, "Type %s had %ld fields, but there are %ld instance fields for inline mapping (ID %d)",
                                    st.name, st.field_sizes.size(), requirement.instance_fields.size(), get_unique_id());
            exit(ERROR_TYPE_INST_MISSIZE);
          }
        case ERROR_TYPE_INST_MISMATCH:
          {
            TypeTable &tt = HighLevelRuntime::get_type_table();
            const Structure &st = tt[requirement.inst_type]; 
            log_region(LEVEL_ERROR, "Type %s has field %s with size %ld for field %d but requirement for inline mapping (ID %d) has size %ld",
                                    st.name, st.field_names[bad_idx], st.field_sizes[bad_idx], bad_idx,
                                    get_unique_id(), bad_size);
            exit(ERROR_TYPE_INST_MISMATCH);
          }
        case ERROR_BAD_PARENT_REGION:
          {
            log_region(LEVEL_ERROR,"Parent task %s (ID %d) of inline mapping (ID %d) does not have a region requirement "
                                    "for region REG_PAT as a parent of region requirement",
                                    parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                    get_unique_id());
            exit(ERROR_BAD_PARENT_REGION);
          }
        case ERROR_BAD_REGION_PATH:
          {
            log_region(LEVEL_ERROR,"Region (%x,%x,%x) is not a sub-region of parent region (%x,%x,%x) for "
                                    "region requirement of inline mapping (ID %d)",
                                    requirement.region.index_space.id,requirement.region.field_space.id, requirement.region.tree_id,
                                    PRINT_REG(requirement.parent), get_unique_id());
            exit(ERROR_BAD_REGION_PATH);
          }
        case ERROR_BAD_REGION_TYPE:
          {
            log_region(LEVEL_ERROR,"Region requirement of inline mapping (ID %d) cannot find privileges for field %d in parent task",
                                    get_unique_id(), bad_field);
            exit(ERROR_BAD_REGION_TYPE);
          }
        case ERROR_BAD_REGION_PRIVILEGES:
          {
            log_region(LEVEL_ERROR,"Privileges %x for region (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                   "region requirement of inline mapping (ID %d)",
                                   requirement.privilege, requirement.region.index_space.id,requirement.region.field_space.id, 
                                   requirement.region.tree_id, get_unique_id());
            exit(ERROR_BAD_REGION_PRIVILEGES);
          }
        case ERROR_NON_DISJOINT_PARTITION: // this should never happen with an inline mapping
        default:
          assert(false); // Should never happen
      }
    }

    /////////////////////////////////////////////////////////////
    // Deletion Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionOperation::DeletionOperation(HighLevelRuntime *rt)
      : GeneralizedOperation(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionOperation::~DeletionOperation(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_index_space_deletion(Context parent, IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      index.space = space;
      handle_tag = DESTROY_INDEX_SPACE;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                  parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_index_partition_deletion(Context parent, IndexPartition part)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      index.partition = part;
      handle_tag = DESTROY_INDEX_PARTITION;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                            parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_field_space_deletion(Context parent, FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      field_space = space;
      handle_tag = DESTROY_FIELD_SPACE;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                            parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_field_deletion(Context parent, FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      field_space = space;
      handle_tag = DESTROY_FIELD;
      free_fields = to_free;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                          parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_region_deletion(Context parent, LogicalRegion reg)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      region = reg;
      handle_tag = DESTROY_REGION;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                          parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::initialize_partition_deletion(Context parent, LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      parent_ctx = parent;
      partition = handle;
      handle_tag = DESTROY_PARTITION;
      performed = false;
      parent->register_child_deletion(this);
#ifdef LEGION_SPY
      LegionSpy::log_deletion_operation(get_unique_id(), parent->get_unique_id(), 
                          parent->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen());
#endif
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      termination_event = UserEvent::create_user_event();
      return activate_base(parent);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(performed);
#endif
      // Trigger the termination event
      termination_event.trigger();
      deactivate_base();
      parent_ctx = NULL;
      runtime->free_deletion(this);
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx == 0);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // This should never be called for deletion operations since they
      // should never ever be registered in the logical region tree
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock();
      // Since deletions can be flushed, they can sometimes be performed before
      // this dependence analysis is performed, so check to make sure we haven't already been performed
      if (performed)
      {
        unlock();
        trigger();
        return;
      }
      unlock();
      lock_context();
      switch (handle_tag)
      {
        case DESTROY_INDEX_SPACE:
          {
            forest_ctx->analyze_index_space_deletion(parent_ctx->ctx_id, index.space, this);
            break;
          }
        case DESTROY_INDEX_PARTITION:
          {
            forest_ctx->analyze_index_part_deletion(parent_ctx->ctx_id, index.partition, this);
            break;
          }
        case DESTROY_FIELD_SPACE:
          {
            forest_ctx->analyze_field_space_deletion(parent_ctx->ctx_id, field_space, this);
            break;
          }
        case DESTROY_FIELD:
          {
            forest_ctx->analyze_field_deletion(parent_ctx->ctx_id, field_space, free_fields, this);
            break;
          }
        case DESTROY_REGION:
          {
            forest_ctx->analyze_region_deletion(parent_ctx->ctx_id, region, this);
            break;
          }
        case DESTROY_PARTITION:
          {
            forest_ctx->analyze_partition_deletion(parent_ctx->ctx_id, partition, this);
            break;
          }
        default:
          assert(false);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    bool DeletionOperation::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      // Lock to test if the operation has been performed yet 
      lock();
      Context parent = parent_ctx; // copy this while holding the lock
      if (!performed)
      {
        perform_internal();
        // Mark that this has been performed and unlock
        performed = true;
        unlock();
        unlock_context();
        runtime->notify_operation_complete(parent);
      }
      else
      {
        unlock();
        unlock_context();
#ifdef INORDER_EXECUTION
        runtime->notify_operation_complete(parent);
#endif
        // The deletion was already performed, so we can now deactivate the operation 
        deactivate();
      }
      // Deletion operations should never fail
      return true;
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Enqueue this operation with the runtime
      runtime->add_to_ready_queue(parent_ctx->get_executing_processor(), this);
    }

    //--------------------------------------------------------------------------
    MapperID DeletionOperation::get_mapper_id(void) const
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_mapper_id();
    }

    //--------------------------------------------------------------------------
    Event DeletionOperation::flush(void)
    //--------------------------------------------------------------------------
    {
      // Looks very similar to the one above, except in the INORDER_EXECUTION case
      // we don't notify that the operation is complete
      lock_context();
      // Lock to test if the operation has been performed yet 
      lock();
      Event result = termination_event;
      if (!performed)
      {
        perform_internal();
#ifndef INORDER_EXECUTION
        Context parent = parent_ctx; // copy this while holding the lock
#endif
        // Mark that this has been performed and unlock
        performed = true;
        unlock();
        unlock_context();
#ifndef INORDER_EXECUTION
        runtime->notify_operation_complete(parent);
#endif
      }
      else
      {
        unlock();
        unlock_context();
        // The deletion was already performed, so we can now deactivate the operation 
        deactivate();
      }
      // Deletion operations never fail
      return result;
    }

    //--------------------------------------------------------------------------
    void DeletionOperation::perform_internal(void)
    //--------------------------------------------------------------------------
    {
      // Should be holding the lock
      switch (handle_tag)
      {
        case DESTROY_INDEX_SPACE:
          {
            parent_ctx->destroy_index_space(index.space);
            break;
          }
        case DESTROY_INDEX_PARTITION:
          {
            parent_ctx->destroy_index_partition(index.partition);
            break;
          }
        case DESTROY_FIELD_SPACE:
          {
            parent_ctx->destroy_field_space(field_space);
            break;
          }
        case DESTROY_FIELD:
          {
            parent_ctx->free_fields(field_space, free_fields);
            break;
          }
        case DESTROY_REGION:
          {
            parent_ctx->destroy_region(region);
            break;
          }
        case DESTROY_PARTITION:
          {
            parent_ctx->destroy_partition(partition);
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    /////////////////////////////////////////////////////////////
    // Task Context 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(HighLevelRuntime *rt, ContextID id)
      : Task(), GeneralizedOperation(rt), ctx_id(id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool TaskContext::activate_task(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_base(parent);
      if (activated)
      {
        parent_ctx = NULL;
        task_pred = Predicate::TRUE_PRED;
        task_id = 0;
        args = NULL;
        arglen = 0;
        map_id = 0;
        tag = 0;
        orig_proc = Processor::NO_PROC;
        steal_count = 0;
        must_parallelism = false;
        is_index_space = false;
        index_space = IndexSpace::NO_SPACE;
        index_point = NULL;
        index_element_size = 0;
        index_dimensions = 0;
        variants = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void TaskContext::deactivate_task(void)
    //--------------------------------------------------------------------------
    {
      indexes.clear();
      fields.clear();
      regions.clear();
      created_index_spaces.clear();
      created_field_spaces.clear();
      created_regions.clear();
      created_fields.clear();
      deleted_regions.clear();
      deleted_partitions.clear();
      deleted_fields.clear();
      launch_preconditions.clear();
      mapped_preconditions.clear();
      map_dependent_waiters.clear();
      if (args != NULL)
      {
        free(args);
        args = NULL;
      }
      if (index_point != NULL)
      {
        free(index_point);
        index_point = NULL;
      }
      // This will remove a reference to any other predicate
      task_pred = Predicate::FALSE_PRED;
      deactivate_base();
    }

    //--------------------------------------------------------------------------
    void TaskContext::initialize_task(Context parent, Processor::TaskFuncID tid,
                                      void *a, size_t len, bool index_space,
                                      const Predicate &predicate,
                                      MapperID mid, MappingTagID t)
    //--------------------------------------------------------------------------
    {
      task_id = tid;
      arglen = len;
      is_index_space = index_space;
      if (arglen > 0)
      {
        args = malloc(arglen);
        memcpy(args, a, arglen);
      }
      // otherwise user_args better be NULL
#ifdef DEBUG_HIGH_LEVEL
      else
      {
        assert(args == NULL);
      }
#endif
      variants = HighLevelRuntime::find_collection(task_id);
      parent_ctx = parent;
      task_pred = predicate;
      map_id = mid;
      tag = t;
      // Initialize remaining fields in the Task as well
      if (parent != NULL)
        orig_proc = parent->get_executing_processor();
#ifdef LEGION_SPY
      if (parent_ctx != NULL)
        LegionSpy::log_task_operation(get_unique_id(), tid, parent_ctx->get_unique_id(), 
          parent_ctx->ctx_id, runtime->utility_proc.id, parent_ctx->get_gen(), this->is_index_space);
#endif
      // Intialize fields in any sub-types
      initialize_subtype_fields();
      // Register with the parent task, only NULL if initializing top-level task
      if (parent != NULL)
        parent->register_child_task(this);
    }

    //--------------------------------------------------------------------------
    void TaskContext::set_requirements(const std::vector<IndexSpaceRequirement> &index_reqs,
                                       const std::vector<FieldSpaceRequirement> &field_reqs,
                                       const std::vector<RegionRequirement> &region_reqs, bool perform_checks)
    //--------------------------------------------------------------------------
    {
      indexes = index_reqs;
      fields  = field_reqs;
      regions = region_reqs;
#ifdef LEGION_SPY
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        LegionSpy::log_logical_requirement(get_unique_id(), idx, (regions[idx].handle_type == SINGULAR),
              (regions[idx].handle_type == SINGULAR) ? regions[idx].region.index_space.id : regions[idx].partition.index_partition,
              (regions[idx].handle_type == SINGULAR) ? regions[idx].region.field_space.id : regions[idx].partition.field_space.id,
              (regions[idx].handle_type == SINGULAR) ? regions[idx].region.tree_id : regions[idx].partition.tree_id,
              regions[idx].privilege, regions[idx].prop, regions[idx].redop);
        LegionSpy::log_requirement_fields(get_unique_id(), idx, regions[idx].privilege_fields);
      }
#endif
      map_dependent_waiters.resize(regions.size());
      if (perform_checks)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(parent_ctx != NULL);
#endif
        // Check all the privileges
        for (unsigned idx = 0; idx < indexes.size(); idx++)
        {
          LegionErrorType et = parent_ctx->check_privilege(indexes[idx]);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_BAD_PARENT_INDEX:
              {
                log_index(LEVEL_ERROR,"Parent task %s (ID %d) of task %s (ID %d) does not have an index requirement "
                                      "for index space %x as a parent of child task's index requirement index %d",
                                      parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                      this->variants->name, get_unique_id(), indexes[idx].parent.id, idx);
                exit(ERROR_BAD_PARENT_INDEX);
              }
            case ERROR_BAD_INDEX_PATH:
              {
                log_index(LEVEL_ERROR,"Index space %x is not a sub-space of parent index space %x for index requirement %d of task %s (ID %d)",
                                      indexes[idx].handle.id, indexes[idx].parent.id, idx,
                                      this->variants->name, get_unique_id());
                exit(ERROR_BAD_INDEX_PATH);
              }
            case ERROR_BAD_INDEX_PRIVILEGES:
              {
                log_index(LEVEL_ERROR,"Privileges %x for index space %x are not a subset of privileges of parent task's privileges for "
                                      "index space requirement %d of task %s (ID %d)",
                                      indexes[idx].privilege, indexes[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_INDEX_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
        for (unsigned idx = 0; idx < fields.size(); idx++)
        {
          LegionErrorType et = parent_ctx->check_privilege(fields[idx]);
          switch (et)
          {
            case NO_ERROR:
              break;
            case ERROR_BAD_FIELD:
              {
                log_field(LEVEL_ERROR,"Parent task %s (ID %d) does not have privileges for field space %x "
                                      "from field space requirement %d of child task %s (ID %d)",
                                      parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                      fields[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_FIELD);
              }
            case ERROR_BAD_FIELD_PRIVILEGES:
              {
                log_field(LEVEL_ERROR,"Privileges %x for field space %x are not a subset of privileges of parent task's privileges "
                                      "for field space requirement %d of task %s (ID %d)",
                                      fields[idx].privilege, fields[idx].handle.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_FIELD_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Verify that the requirement is self-consistent
          FieldID bad_field;
          size_t bad_size;
          unsigned bad_idx;
          LegionErrorType et = verify_requirement(regions[idx], bad_field, bad_size, bad_idx);
          if ((et == NO_ERROR) && !is_index_space && (regions[idx].handle_type == PROJECTION))
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
                log_region(LEVEL_ERROR, "Invalid region handle (%x,%d,%d) for region requirement %d of task %s (ID %d)",
                    regions[idx].region.index_space.id, regions[idx].region.field_space.id, regions[idx].region.tree_id,
                    idx, variants->name, get_unique_id());
                exit(ERROR_INVALID_REGION_HANDLE);
              }
            case ERROR_INVALID_PARTITION_HANDLE:
              {
                log_region(LEVEL_ERROR, "Invalid partition handle (%x,%d,%d) for partition requirement %d of task %s (ID %d)",
                    regions[idx].partition.index_partition, regions[idx].partition.field_space.id, regions[idx].partition.tree_id,
                    idx, variants->name, get_unique_id());
                exit(ERROR_INVALID_PARTITION_HANDLE);
              }
            case ERROR_BAD_PROJECTION_USE:
              {
                log_region(LEVEL_ERROR,"Projection region requirement %d used in non-index space task %s",
                                        idx, this->variants->name);
                exit(ERROR_BAD_PROJECTION_USE);
              }
            case ERROR_NON_DISJOINT_PARTITION:
              {
                log_region(LEVEL_ERROR,"Non disjoint partition selected for writing region requirement %d of task %s.  All projection partitions "
                                        "which are not read-only and not reduce must be disjoint", idx, this->variants->name);
                exit(ERROR_NON_DISJOINT_PARTITION);
              }
            case ERROR_FIELD_SPACE_FIELD_MISMATCH:
              {
                FieldSpace sp = (regions[idx].handle_type == SINGULAR) ? regions[idx].region.field_space : regions[idx].partition.field_space;
                log_region(LEVEL_ERROR,"Field %d is not a valid field of field space %d for region %d of task %s (ID %d)",
                                        bad_field, sp.id, idx, this->variants->name, get_unique_id());
                exit(ERROR_FIELD_SPACE_FIELD_MISMATCH);
              }
            case ERROR_INVALID_INSTANCE_FIELD:
              {
                log_region(LEVEL_ERROR,"Instance field %d is not one of the privilege fields for region %d of task %s (ID %d)",
                                        bad_field, idx, this->variants->name, get_unique_id());
                exit(ERROR_INVALID_INSTANCE_FIELD);
              }
            case ERROR_DUPLICATE_INSTANCE_FIELD:
              {
                log_region(LEVEL_ERROR, "Instance field %d is a duplicate for region %d of task %s (ID %d)",
                                      bad_field, idx, this->variants->name, get_unique_id());
                exit(ERROR_DUPLICATE_INSTANCE_FIELD);
              }
            case ERROR_INVALID_TYPE_HANDLE:
              {
                log_region(LEVEL_ERROR, "Type handle %d does not name a valid registered structure type for region %d of task %s (ID %d)",
                                        regions[idx].inst_type, idx, this->variants->name, get_unique_id());
                exit(ERROR_INVALID_TYPE_HANDLE);
              }
            case ERROR_TYPE_INST_MISSIZE:
              {
                TypeTable &tt = HighLevelRuntime::get_type_table();
                const Structure &st = tt[regions[idx].inst_type];
                log_region(LEVEL_ERROR, "Type %s had %ld fields, but there are %ld instance fields for region %d of task %s (ID %d)",
                                        st.name, st.field_sizes.size(), regions[idx].instance_fields.size(), 
                                        idx, this->variants->name, get_unique_id());
                exit(ERROR_TYPE_INST_MISSIZE);
              }
            case ERROR_TYPE_INST_MISMATCH:
              {
                TypeTable &tt = HighLevelRuntime::get_type_table();
                const Structure &st = tt[regions[idx].inst_type]; 
                log_region(LEVEL_ERROR, "Type %s has field %s with size %ld for field %d but requirement for region %d of "
                                        "task %s (ID %d) has size %ld",
                                        st.name, st.field_names[bad_idx], st.field_sizes[bad_idx], bad_idx,
                                        idx, this->variants->name, get_unique_id(), bad_size);
                exit(ERROR_TYPE_INST_MISMATCH);
              }
            case ERROR_BAD_PARENT_REGION:
              {
                log_region(LEVEL_ERROR,"Parent task %s (ID %d) of task %s (ID %d) does not have a region requirement "
                                        "for region REG_PAT as a parent of child task's region requirement index %d",
                                        parent_ctx->variants->name, parent_ctx->get_unique_id(),
                                        this->variants->name, get_unique_id(), idx);
                exit(ERROR_BAD_PARENT_REGION);
              }
            case ERROR_BAD_REGION_PATH:
              {
                log_region(LEVEL_ERROR,"Region (%x,%x,%x) is not a sub-region of parent region (%x,%x,%x) for "
                                        "region requirement %d of task %s (ID %d)",
                                        regions[idx].region.index_space.id,regions[idx].region.field_space.id, regions[idx].region.tree_id,
                                        PRINT_REG(regions[idx].parent), idx,this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PATH);
              }
            case ERROR_BAD_PARTITION_PATH:
              {
                log_region(LEVEL_ERROR,"Partition (%x,%x,%x) is not a sub-partition of parent region (%x,%x,%x) for "
                    "                   region requirement %d of task %s (ID %d)",
                                        regions[idx].partition.index_partition, regions[idx].partition.field_space.id, 
                                        regions[idx].partition.tree_id, PRINT_REG(regions[idx].parent), idx,
                                        this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PATH);
              }
            case ERROR_BAD_REGION_TYPE:
              {
                log_region(LEVEL_ERROR,"Region requirement %d of task %s (ID %d) cannot find privileges for field %d in parent task",
                                        idx, this->variants->name, get_unique_id(), bad_field);
                exit(ERROR_BAD_REGION_TYPE);
              }
            case ERROR_BAD_REGION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for region (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].region.index_space.id,regions[idx].region.field_space.id, 
                                       regions[idx].region.tree_id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_REGION_PRIVILEGES);
              }
            case ERROR_BAD_PARTITION_PRIVILEGES:
              {
                log_region(LEVEL_ERROR,"Privileges %x for partition (%x,%x,%x) are not a subset of privileges of parent task's privileges for "
                                       "region requirement %d of task %s (ID %d)",
                                       regions[idx].privilege, regions[idx].partition.index_partition, regions[idx].partition.field_space.id, 
                                       regions[idx].partition.tree_id, idx, this->variants->name, get_unique_id());
                exit(ERROR_BAD_PARTITION_PRIVILEGES);
              }
            default:
              assert(false); // Should never happen
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::add_mapping_dependence(unsigned idx, const LogicalUser &prev, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (this == prev.op)
      {
        log_task(LEVEL_ERROR,"Illegal dependence between two region requirements with indexes %d and %d in task %s (ID %d)",
                              prev.idx, idx, this->variants->name, get_unique_id());
        exit(ERROR_ALIASED_INTRA_TASK_REGIONS);
      }
#endif
#ifdef LEGION_SPY
      LegionSpy::log_mapping_dependence(parent_ctx->get_unique_id(), parent_ctx->ctx_id, 
                                        runtime->utility_proc.id, parent_ctx->get_gen(), prev.op->get_unique_id(),
                                        prev.idx, get_unique_id(), idx, dtype);
#endif
      if (prev.op->add_waiting_dependence(this, prev.idx, prev.gen))
      {
        outstanding_dependences++;
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::return_privileges(const std::set<IndexSpace> &new_indexes,
                                        const std::set<FieldSpace> &new_fields,
                                        const std::set<LogicalRegion> &new_regions,
                                        const std::map<FieldID,FieldSpace> &new_field_ids)
    //--------------------------------------------------------------------------
    {
      lock();
      created_index_spaces.insert(new_indexes.begin(),new_indexes.end());
      created_field_spaces.insert(new_fields.begin(),new_fields.end());
      created_regions.insert(new_regions.begin(),new_regions.end());
      // For the return fields, if they are in a field space we created
      // we no longer need to track them as uniquely created fields
      for (std::map<FieldID,FieldSpace>::const_iterator it = new_field_ids.begin();
            it != new_field_ids.end(); it++)
      {
        if (created_field_spaces.find(it->second) == created_field_spaces.end())
          created_fields.insert(*it);
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool TaskContext::has_created_index_space(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      return (created_index_spaces.find(handle) != created_index_spaces.end());
    }

    //--------------------------------------------------------------------------
    bool TaskContext::has_created_field_space(FieldSpace handle) const
    //--------------------------------------------------------------------------
    {
      return (created_field_spaces.find(handle) != created_field_spaces.end());
    }

    //--------------------------------------------------------------------------
    bool TaskContext::has_created_region(LogicalRegion handle) const
    //--------------------------------------------------------------------------
    {
      return (created_regions.find(handle) != created_regions.end());
    }

    //--------------------------------------------------------------------------
    bool TaskContext::has_created_field(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (created_fields.find(fid) != created_fields.end());
    }

    //--------------------------------------------------------------------------
    size_t TaskContext::compute_task_context_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_user_task_size();
      result += compute_operation_size();
      result += 2*sizeof(size_t); // size of preconditions sets
      result += ((launch_preconditions.size() + mapped_preconditions.size()) * sizeof(Event));
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_task_context(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_user_task(rez);
      pack_operation(rez);
      rez.serialize<size_t>(launch_preconditions.size());
      for (std::set<Event>::const_iterator it = launch_preconditions.begin();
            it != launch_preconditions.end(); it++)
      {
        rez.serialize<Event>(*it);
      }
      rez.serialize<size_t>(mapped_preconditions.size());
      for (std::set<Event>::const_iterator it = mapped_preconditions.begin();
            it != mapped_preconditions.end(); it++)
      {
        rez.serialize<Event>(*it);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::unpack_task_context(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_user_task(derez);
      unpack_operation(derez);
      size_t num_events;
      derez.deserialize<size_t>(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        Event e;
        derez.deserialize<Event>(e);
        launch_preconditions.insert(e);
      }
      derez.deserialize<size_t>(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        Event e;
        derez.deserialize<Event>(e);
        mapped_preconditions.insert(e);
      }
    }

    //--------------------------------------------------------------------------
    bool TaskContext::invoke_mapper_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->map_task_locally(this); 
    }

    //--------------------------------------------------------------------------
    bool TaskContext::invoke_mapper_stealable(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->spawn_task(this);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::invoke_mapper_map_region_virtual(unsigned idx, Processor target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->map_region_virtually(this, target, regions[idx], idx);
    }

    //--------------------------------------------------------------------------
    Processor TaskContext::invoke_mapper_select_target_proc(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      Processor result = mapper->select_target_processor(this);
#ifdef DEBUG_HIGH_LEVEL
      if (!result.exists())
      {
        log_task(LEVEL_ERROR,"Mapper selected invalid NO_PROC for target processor for task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_INVALID_PROCESSOR_SELECTION);
      }
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Processor::TaskFuncID TaskContext::invoke_mapper_select_variant(Processor target)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      VariantID vid = mapper->select_task_variant(this, target);
#ifdef DEBUG_HIGH_LEVEL
      if (!variants->has_variant(vid))
      {
        log_task(LEVEL_ERROR,"Mapper selected invalid variant ID %ld for task %s (ID %d)",
                              vid, this->variants->name, get_unique_id());
        exit(ERROR_INVALID_VARIANT_SELECTION);
      }
#endif
      return variants->get_variant(vid).low_id;
    }

    //--------------------------------------------------------------------------
    void TaskContext::invoke_mapper_failed_mapping(unsigned idx, Processor target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      AutoLock m_lock(mapper_lock);
      DetailedTimer::ScopedPush sp(TIME_MAPPER);
      return mapper->notify_failed_mapping(this, target, regions[idx], idx, false/*inline mapping*/);
    }

    //--------------------------------------------------------------------------
    void TaskContext::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Analyze everything in the parent contexts logical scope
        RegionAnalyzer az(parent_ctx->ctx_id, this, idx, regions[idx]);
        // Compute the path to path to the destination
        if (regions[idx].handle_type == SINGULAR)
          forest_ctx->compute_index_path(regions[idx].parent.index_space, 
                                          regions[idx].region.index_space, az.path);
        else
          forest_ctx->compute_partition_path(regions[idx].parent.index_space, 
                                              regions[idx].partition.index_partition, az.path);
        forest_ctx->analyze_region(az);
      }
      bool ready = is_ready();
      unlock_context();
      if (ready)
        trigger();
    }

    //--------------------------------------------------------------------------
    MapperID TaskContext::get_mapper_id(void) const
    //--------------------------------------------------------------------------
    {
      return map_id;
    }

    //--------------------------------------------------------------------------
    size_t TaskContext::compute_privileges_return_size(void)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here since we know the task is done
      size_t result = 4*sizeof(size_t);
      result += (created_index_spaces.size() * sizeof(IndexSpace));
      result += (created_field_spaces.size() * sizeof(FieldSpace));
      result += (created_regions.size() * sizeof(LogicalRegion));
      result += (created_fields.size() * (sizeof(FieldID) + sizeof(FieldSpace)));
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_privileges_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock here since we know the task is done
      rez.serialize<size_t>(created_index_spaces.size());
      for (std::set<IndexSpace>::const_iterator it = created_index_spaces.begin();
            it != created_index_spaces.end(); it++)
      {
        rez.serialize<IndexSpace>(*it);
      }
      rez.serialize<size_t>(created_field_spaces.size());
      for (std::set<FieldSpace>::const_iterator it = created_field_spaces.begin();
            it != created_field_spaces.end(); it++)
      {
        rez.serialize<FieldSpace>(*it);
      }
      rez.serialize<size_t>(created_regions.size());
      for (std::set<LogicalRegion>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        rez.serialize<LogicalRegion>(*it);
      }
      rez.serialize(created_fields.size());
      for (std::map<FieldID,FieldSpace>::const_iterator it = created_fields.begin();
            it != created_fields.end(); it++)
      {
        rez.serialize<FieldID>(it->first);
        rez.serialize<FieldSpace>(it->second);
      }
    }

    //--------------------------------------------------------------------------
    size_t TaskContext::unpack_privileges_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Need the lock here since it's possible that a child task is
      // returning while the task itself is still running
      lock();
      size_t num_elmts;
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        IndexSpace space;
        derez.deserialize<IndexSpace>(space);
        created_index_spaces.insert(space);
      }
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        FieldSpace space;
        derez.deserialize<FieldSpace>(space);
        created_field_spaces.insert(space);
      }
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        LogicalRegion region;
        derez.deserialize<LogicalRegion>(region);
        created_regions.insert(region);
      }
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        FieldID fid;
        derez.deserialize<FieldID>(fid);
        FieldSpace handle;
        derez.deserialize<FieldSpace>(handle);
        created_fields[fid] = handle;
      }
      unlock();
      // Return the number of new regions
      return num_elmts;
    }

    //--------------------------------------------------------------------------
    size_t TaskContext::compute_deletions_return_size(void)
    //--------------------------------------------------------------------------
    {
      // No need to hold the lock since we know the task is done 
      size_t result = 0;
      result += sizeof(size_t); // number of regions
      result += sizeof(size_t); // number of partitions
      result += sizeof(size_t); // number of fields
      result += (deleted_regions.size() * sizeof(LogicalRegion));
      result += (deleted_partitions.size() * sizeof(LogicalPartition));
      result += (deleted_fields.size() * (sizeof(FieldID) + sizeof(FieldSpace)));
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::pack_deletions_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(deleted_regions.size());
      for (std::vector<LogicalRegion>::const_iterator it = deleted_regions.begin();
            it != deleted_regions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_partitions.size());
      for (std::vector<LogicalPartition>::const_iterator it = deleted_partitions.begin();
            it != deleted_partitions.end(); it++)
      {
        rez.serialize(*it);
      }
      rez.serialize(deleted_fields.size());
      for (std::map<FieldID,FieldSpace>::const_iterator it = deleted_fields.begin();
            it != deleted_fields.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::unpack_deletions_return(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // hold the lock to make sure there is no interference from still
      // executing children
      lock();
      size_t num_regions;
      derez.deserialize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        std::set<LogicalRegion>::iterator finder = created_regions.find(handle);
        // Check to see if we created it, if so just remove it from this list
        // of created regions
        if (finder != created_regions.end())
          created_regions.erase(finder);
        else
          deleted_regions.push_back(handle);
      }
      size_t num_partitions;
      derez.deserialize(num_partitions);
      for (unsigned idx = 0; idx < num_partitions; idx++)
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        deleted_partitions.push_back(handle);
      }
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        FieldSpace space;
        derez.deserialize(space);
        std::map<FieldID,FieldSpace>::iterator finder = deleted_fields.find(fid);
        if (finder != deleted_fields.end())
          deleted_fields.erase(finder);
        else
          deleted_fields[fid] = space;
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void TaskContext::clone_task_context_from(TaskContext *rhs)
    //--------------------------------------------------------------------------
    {
      clone_task_from(rhs);
      clone_generalized_operation_from(rhs);
      this->parent_ctx = rhs->parent_ctx;
      this->task_pred = rhs->task_pred;
      this->mapper= rhs->mapper;
      this->mapper_lock = rhs->mapper_lock;
      this->launch_preconditions = rhs->launch_preconditions;
      this->mapped_preconditions = rhs->mapped_preconditions;
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(HighLevelRuntime *rt, ContextID id)
      : TaskContext(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    bool SingleTask::activate_single(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_task(parent);
      if (activated)
      {
        executing_processor = Processor::NO_PROC;
        low_id = 0;
        unmapped = 0;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate_single(void)
    //--------------------------------------------------------------------------
    {
      non_virtual_mapped_region.clear();
      physical_instances.clear();
      clone_instances.clear();
      source_copy_instances.clear();
      close_copy_instances.clear();
      physical_contexts.clear();
      physical_region_impls.clear();
      child_tasks.clear();
      child_maps.clear();
      child_deletions.clear();
      deactivate_task();
    }

    //--------------------------------------------------------------------------
    bool SingleTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        if (!is_distributed() && !is_stolen())
        {
          // This task is still on the processor
          // where it originated, so we have to do the mapping now
          if (perform_mapping())
          {
            if (distribute_task())
            {
              // Still local so launch the task
              launch_task();
            }
            // otherwise it was sent away and we're done
          }
          else // mapping failed
          {
            success = false;
          }
        }
        else
        {
          // If it was stolen and hasn't been distributed yet
          // we have to try distributing it first
          if (!is_distributed())
          {
            if (distribute_task())
            {
              launch_task();
            }
          }
          else
          {
            // This was task was already distributed 
            // so just run it here regardless of whether
            // it was stolen or not
            launch_task();
          }
        }
      }
      else // not locally mapped
      {
        if (!is_distributed())
        {
          // Don't need to do sanitization if we were already stolen
          // since that means we're remote and were already sanitized
          if (is_stolen() || sanitize_region_forest())
          {
            if (distribute_task())
            {
              if (perform_mapping())
                launch_task();
              else
                success = false;
            }
            // otherwise it was sent away and we're done
          }
          else
            success = false;
        }
        else // already been distributed
        {
          if (perform_mapping())
            launch_task();
          else
            success = false;
        }
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        // If task is locally mapped it shouldn't have even been on the
        // list of tasks to steal see HighLevelRuntime::process_steal
        assert(false);
        success = false;
      }
      else
      {
        // If it hasn't been distributed and it hasn't been
        // stolen then we have to be able to sanitize it to
        // be able to steal it
        if (!is_distributed() && !is_stolen())
          success = sanitize_region_forest();
      }
      if (success)
        steal_count++;
      return success;
    }

    //--------------------------------------------------------------------------
    ContextID SingleTask::find_enclosing_physical_context(LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_contexts.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].region == parent)
        {
          return physical_contexts[idx];
        }
      }
      // Otherwise check the created regions
      if (created_regions.find(parent) != created_regions.end())
      {
        return find_outermost_physical_context();
      }
      // otherwise this is really bad and indicates a runtime error
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    ContextID SingleTask::find_outermost_physical_context(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_ctx != NULL)
        return parent_ctx->find_outermost_physical_context();
      // Otherwise we're the outermost physical context
      return ctx_id;
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_task(TaskContext *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal child task launch performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_tasks.push_back(child);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_map(MappingOperation *child, int idx /*= -1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal inline mapping performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_maps.push_back(child);
      // Check to make sure that this region still isn't mapped
      if (idx > -1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(unsigned(idx) < clone_instances.size());
#endif
        // Check this on the cloned_instances since this will be where
        // we unmap regions that the task has previously mapped
        if (!clone_instances[idx].is_virtual_ref())
        {
          log_task(LEVEL_ERROR,"Illegal inline mapping for originally mapped region at index %d."
                                " Region is still mapped!",idx);
          exit(ERROR_INVALID_DUPLICATE_MAPPING);
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_child_deletion(DeletionOperation *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal deletion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock();
      child_deletions.push_back(child);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index space creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_index_space(space);
      unlock_context();
      // Also add it the set of index spaces that we have privileges for
      lock();
      created_index_spaces.insert(space);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_index_space(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      // Note we don't need to defer anything here since that has already
      // been handled by a DeletionOperation.  No need to hold the context
      // lock either since the enclosing deletion already has it.
      std::vector<LogicalRegion> new_deletions;
      // Have to call this before destroy index space
      forest_ctx->get_destroyed_regions(space,new_deletions);
      forest_ctx->destroy_index_space(space);
      // Check to see if it is in the list of spaces that we created
      // and if it is then delete it
      lock();
      created_index_spaces.erase(space);
      unlock();
      // Register the deletions locally
      for (std::vector<LogicalRegion>::const_iterator it = new_deletions.begin();
            it != new_deletions.end(); it++)
      {
        register_deletion(*it);
      }
    }

    //--------------------------------------------------------------------------
    Color SingleTask::create_index_partition(IndexPartition pid, IndexSpace parent, 
                                            bool disjoint, int color,
                                            const std::map<Color,IndexSpace> &coloring) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal index partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      Color result = forest_ctx->create_index_partition(pid, parent, disjoint, color, coloring);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_index_partition(IndexPartition pid)
    //--------------------------------------------------------------------------
    {
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation.  The DeletionOperation already has
      // taken the context lock too.
      std::vector<LogicalPartition> new_deletions;
      // Have to call this before destroy_index_partition
      forest_ctx->get_destroyed_partitions(pid,new_deletions);
      forest_ctx->destroy_index_partition(pid);
      for (std::vector<LogicalPartition>::const_iterator it = new_deletions.begin();
            it != new_deletions.end(); it++)
      {
        register_deletion(*it);
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition SingleTask::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      IndexPartition result = forest_ctx->get_index_partition(parent, color);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace SingleTask::get_index_subspace(IndexPartition pid, Color color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get index subspace performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      IndexSpace result = forest_ctx->get_index_subspace(pid, color);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal create field space performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_field_space(space);
      unlock_context();
      // Also add this to the list of field spaces for which we have privileges
      lock();
      created_field_spaces.insert(space);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_field_space(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation.  The DeletionOperation has also taken
      // the context lock.
      std::vector<LogicalRegion> new_deletions;
      // Have to call this before destroy_field_space
      forest_ctx->get_destroyed_regions(space, new_deletions);
      forest_ctx->destroy_field_space(space);
      // Also check to see if this is one of the field spaces we created
      lock();
      created_field_spaces.erase(space);
      // Also go through and see if there were any created fields that we had
      // for this field space that we can now also destroy
      {
        std::vector<FieldID> to_delete;
        for (std::map<FieldID,FieldSpace>::const_iterator it = created_fields.begin();
              it != created_fields.end(); it++)
        {
          if (it->second == space)
            to_delete.push_back(it->first);
        }
        for (std::vector<FieldID>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
        {
          created_fields.erase(*it);
        }
      }
      unlock();
      for (std::vector<LogicalRegion>::const_iterator it = new_deletions.begin();
            it != new_deletions.end(); it++)
      {
        register_deletion(*it);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::allocate_fields(FieldSpace space, const std::map<FieldID,size_t> &field_allocations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal field allocation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->allocate_fields(space, field_allocations);
      unlock_context();
      // Also add this to the list of fields we created
      lock();
      for (std::map<FieldID,size_t>::const_iterator it = field_allocations.begin();
            it != field_allocations.end(); it++)
      {
        created_fields[it->first] = space;
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::free_fields(FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal field deallocation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation
      lock_context();
      forest_ctx->free_fields(space, to_free);
      unlock_context();
      std::map<FieldID,FieldSpace> free_fields;
      for (std::set<FieldID,FieldSpace>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
        free_fields[*it] = space;
      }
      register_deletion(free_fields);
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      forest_ctx->create_region(handle, find_outermost_physical_context());
      unlock_context();
      // Add this to the list of our created regions
      lock();
      created_regions.insert(handle);
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // No need to worry about deferring this, it's already been done
      // by the DeletionOperation.  DeletionOperation has context lock too.
      forest_ctx->destroy_region(handle);
      // Also check to see if it is one of created regions so we can delete it
      lock();
      created_regions.erase(handle);
      unlock();
      register_deletion(handle);
    }

    //--------------------------------------------------------------------------
    void SingleTask::destroy_partition(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal region creation performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      // The DeletionOperation already took the context lock and has
      // deffered this operation to observe any dependences.
      forest_ctx->destroy_partition(handle);
      register_deletion(handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_region_partition(LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get region partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalPartition result = forest_ctx->get_region_partition(parent, handle);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_partition_subregion(LogicalPartition pid, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get partition subregion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalRegion result = forest_ctx->get_partition_subregion(pid, handle);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_region_subcolor(LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get region partition performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalPartition result = forest_ctx->get_region_subcolor(parent, c);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_partition_subcolor(LogicalPartition pid, Color c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get partition subregion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalRegion result = forest_ctx->get_partition_subcolor(pid, c);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition SingleTask::get_partition_subtree(IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get partition subregion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalPartition result = forest_ctx->get_partition_subtree(handle, space, tid);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion SingleTask::get_region_subtree(IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        log_task(LEVEL_ERROR,"Illegal get partition subregion performed in leaf task %s (ID %d)",
                              this->variants->name, get_unique_id());
        exit(ERROR_LEAF_TASK_VIOLATION);
      }
#endif
      lock_context();
      LogicalRegion result = forest_ctx->get_region_subtree(handle, space, tid);
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::unmap_physical_region(PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      if (region.is_impl)
      {
        unsigned idx = region.op.impl->idx;
        lock();
        if (idx >= regions.size())
        {
          log_task(LEVEL_ERROR,"Unmap operation for task argument region %d is out of range",idx);
          exit(ERROR_INVALID_REGION_ARGUMENT_INDEX);
        }
        // Check to see if this region was actually mapped
        // If it wasn't then this is still ok since we want to allow mapping
        // agnostic code, which means programs should still work regardless
        // of whether regions were virtually mapped or not
        if (!clone_instances[idx].is_virtual_ref())
        {
          physical_region_impls[idx]->invalidate();
          clone_instances[idx].remove_reference(unique_id, false/*strict*/);
          clone_instances[idx] = InstanceRef(); // make it a virtual ref now
        }
        unlock();
      }
      else
      {
        // Go through the list of mapping operations, remove it, and deactivate it
#ifdef DEBUG_HIGH_LEVEL
        bool found = false;
#endif
        lock();
        for (std::list<MappingOperation*>::iterator it = child_maps.begin();
              it != child_maps.end(); it++)
        {
          if ((*it) == region.op.map)
          {
            child_maps.erase(it);
#ifdef DEBUG_HIGH_LEVEL
            found = true;
#endif
            break;
          }
        }
        unlock();
#ifdef DEBUG_HIGH_LEVEL
        if (!found)
        {
          log_task(LEVEL_ERROR,"Invalid unmap operation on inline mapping");
          exit(ERROR_INVALID_UNMAP_OP);
        }
#endif
        region.op.map->deactivate();
      }
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const IndexSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      // Find the parent index space
      for (std::vector<IndexSpaceRequirement>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
      {
        // Check to see if we found the requirement in the parent 
        if (it->handle == req.parent)
        {
          // Check that there is a path between the parent and the child
          {
            std::vector<unsigned> path;
            lock_context();
            if (!forest_ctx->compute_index_path(req.parent, req.handle, path))
            {
              unlock_context();
              return ERROR_BAD_INDEX_PATH;
            }
            unlock_context();
          }
          // Now check that the privileges are less than or equal
          if (req.privilege & (~(it->privilege)))
          {
            return ERROR_BAD_INDEX_PRIVILEGES;  
          }
          return NO_ERROR;
        }
      }
      // If we didn't find it here, we have to check the added index spaces that we have
      if (created_index_spaces.find(req.parent) != created_index_spaces.end())
      {
        // Still need to check that there is a path between the two
        std::vector<unsigned> path;
        lock_context();
        if (!forest_ctx->compute_index_path(req.parent, req.handle, path))
        {
          unlock_context();
          return ERROR_BAD_INDEX_PATH;
        }
        unlock_context();
        // No need to check privileges here since it is a created space
        // which means that the parent has all privileges.
        return NO_ERROR;
      }
      return ERROR_BAD_PARENT_INDEX;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const FieldSpaceRequirement &req) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      for (std::vector<FieldSpaceRequirement>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        // Check to see if they match
        if (it->handle == req.handle)
        {
          // Check that the privileges are less than or equal
          if (req.privilege & (~(it->privilege)))
          {
            return ERROR_BAD_FIELD_PRIVILEGES;
          }
          return NO_ERROR;
        }
      }
      // If we didn't find it here, we also need to check the added field spaces
      if (created_field_spaces.find(req.handle) != created_field_spaces.end())
      {
        // No need to check the privileges since by definition of
        // a created field space the parent has all privileges.
        return NO_ERROR;
      }
      return ERROR_BAD_FIELD;
    }

    //--------------------------------------------------------------------------
    LegionErrorType SingleTask::check_privilege(const RegionRequirement &req, FieldID &bad_field) const
    //--------------------------------------------------------------------------
    {
      if (req.verified)
        return NO_ERROR;
      for (std::vector<RegionRequirement>::const_iterator it = regions.begin();
            it != regions.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(it->handle_type == SINGULAR); // better be singular
#endif
        // Check to see if we found the requirement in the parent
        if (it->region == req.parent)
        {
          // Check that there is a path between the parent and the child
          lock_context();
          if (req.handle_type == SINGULAR)
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_index_path(req.parent.index_space, req.region.index_space, path))
            {
              unlock_context();
              return ERROR_BAD_REGION_PATH;
            }
          }
          else
          {
            std::vector<unsigned> path;
            if (!forest_ctx->compute_partition_path(req.parent.index_space, req.partition.index_partition, path))
            {
              unlock_context();
              return ERROR_BAD_PARTITION_PATH;
            }
          }
          unlock_context();
          // Now check that the types are subset of the fields
          // Note we can use the parent since all the regions/partitions
          // in the same region tree have the same field space
          for (std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
                fit != req.privilege_fields.end(); fit++)
          {
            if ((it->privilege_fields.find(*fit) == it->privilege_fields.end()) &&
                (!has_created_field(*fit)))
            {
              bad_field = *fit;
              return ERROR_BAD_REGION_TYPE;
            }
          }
          if (req.privilege & (~(it->privilege)))
          {
            if (req.handle_type == SINGULAR)
              return ERROR_BAD_REGION_PRIVILEGES;
            else
              return ERROR_BAD_PARTITION_PRIVILEGES;
          }
          return NO_ERROR;
        }
      }
      // Also check to see if it was a created region
      if (created_regions.find(req.parent) != created_regions.end())
      {
        // Check that there is a path between the parent and the child
        lock_context();
        if (req.handle_type == SINGULAR)
        {
          std::vector<unsigned> path;
          if (!forest_ctx->compute_index_path(req.parent.index_space, req.region.index_space, path))
          {
            unlock_context();
            return ERROR_BAD_REGION_PATH;
          }
        }
        else
        {
          std::vector<unsigned> path;
          if (!forest_ctx->compute_partition_path(req.parent.index_space, req.partition.index_partition, path))
          {
            unlock_context();
            return ERROR_BAD_PARTITION_PATH;
          }
        }
        unlock_context();
        // No need to check the field privileges since we should have them all

        // No need to check the privileges since we know we have them all
        return NO_ERROR;
      }
      return ERROR_BAD_PARENT_REGION;
    }

    //--------------------------------------------------------------------------
    void SingleTask::start_task(std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Task %s (ID %d) starting on processor %x",
                this->variants->name, get_unique_id(), executing_processor.id);
      assert(regions.size() == physical_instances.size());
#endif
      physical_regions.resize(regions.size());
      physical_region_impls.resize(regions.size());
#ifdef DEBUG_HIGH_LEVEL
      assert(physical_instances.size() == regions.size());
#endif
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        physical_region_impls[idx] = new PhysicalRegionImpl(idx, regions[idx].region, 
            (physical_instances[idx].is_virtual_ref() ? NULL : physical_instances[idx].get_manager()), &(regions[idx]));
        physical_regions[idx] = PhysicalRegion(physical_region_impls[idx]);
#ifdef LEGION_SPY
        if (!physical_instances[idx].is_virtual_ref())
          LegionSpy::log_task_user(this->get_unique_id(), this->ctx_id, this->get_gen(), runtime->utility_proc.id, idx,
                                    physical_instances[idx].get_manager()->get_unique_id());
#endif
      }
      // If we're not remote, then we can release all the source copy instances
      if (!is_remote())
      {
        lock_context();
        release_source_copy_instances();
        unlock_context();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::complete_task(const void *result, size_t result_size, std::vector<PhysicalRegion> &physical_regions)
    //--------------------------------------------------------------------------
    {
      // Clean up some of our stuff from the task execution
      for (unsigned idx = 0; idx < physical_region_impls.size(); idx++)
      {
        delete physical_region_impls[idx];
      }
      physical_region_impls.clear();
      // Handle the future result
      handle_future(result, result_size, get_termination_event());
      
      if (is_leaf())
      {
        // Invoke the function for when we're done 
        finish_task();
      }
      else
      {
        // Otherwise go through all the children tasks and get their mapping events
        std::set<Event> map_events = mapped_preconditions;
        lock();
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          map_events.insert((*it)->get_map_event());
        }
        // Do this for the mapping operations as well, deletions have a different path
        for (std::list<MappingOperation*>::const_iterator it = child_maps.begin();
              it != child_maps.end(); it++)
        {
          map_events.insert((*it)->get_map_event());
        }
        unlock();
        Event wait_on_event = Event::merge_events(map_events);
        if (!wait_on_event.exists())
        {
          // All the children are mapped, just do the next thing
          children_mapped();
        }
        else
        {
          // Otherwise launch a task to be run once all the children
          // have been mapped
          size_t buffer_size = sizeof(Processor) + sizeof(Context);
          Serializer rez(buffer_size);
          rez.serialize<Processor>(runtime->utility_proc);
          rez.serialize<Context>(this);
          // Launch the task on the utility processor
          Processor utility = runtime->utility_proc;
          utility.spawn(CHILDREN_MAPPED_ID,rez.get_buffer(),buffer_size,wait_on_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    const RegionRequirement& SingleTask::get_region_requirement(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(idx < regions.size());
#endif
      return regions[idx];
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_deletion(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      lock();
      // To cut down on over approximation of the set of deleted regions
      // if we find we were the ones that created the region and it hasn't
      // been returned yet, then we no longer have to tell others to delete it
      std::set<LogicalRegion>::iterator finder = created_regions.find(handle);
      if (finder != created_regions.end())
        created_regions.erase(finder);
      else
        deleted_regions.push_back(handle);
      unlock();
      // We should already hold the context lock here

      // Invalidate the state in the this context.  Note if we were virtually
      // mapped then this won't matter, but it would take too much to figure
      // out whether we should have done it or not, so just do it.
      forest_ctx->invalidate_physical_context(handle, ctx_id);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_deletion(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      lock();
      deleted_partitions.push_back(handle);
      unlock();
      // We should already hold the context lock here

      // Invalidate the state in the this context.  Note if we were virtually
      // mapped then this won't matter, but it would take too much to figure
      // out whether we should have done it or not, so just do it.
      forest_ctx->invalidate_physical_context(handle, ctx_id);
    }

    //--------------------------------------------------------------------------
    void SingleTask::register_deletion(const std::map<FieldID,FieldSpace> &fields)
    //--------------------------------------------------------------------------
    {
      // For each of our trees see if we have any deleted fields to handle
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        invalidate_matches(regions[idx].region, fields); 
      }
      for (std::set<LogicalRegion>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        invalidate_matches(*it, fields);
      }
      // Now see if we can add the fields, or if we were the ones to create it
      lock();
      for (std::map<FieldID,FieldSpace>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        std::map<FieldID,FieldSpace>::iterator finder = created_fields.find(it->first);
        if (finder != created_fields.end())
          created_fields.erase(finder);
        else
          deleted_fields.insert(*it);
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    void SingleTask::return_deletions(const std::vector<LogicalRegion> &handles)
    //--------------------------------------------------------------------------
    {
      lock_context();
      for (std::vector<LogicalRegion>::const_iterator it = handles.begin();
            it != handles.end(); it++)
      {
        register_deletion(*it);
      }
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::return_deletions(const std::vector<LogicalPartition> &handles)
    //--------------------------------------------------------------------------
    {
      lock_context();
      for (std::vector<LogicalPartition>::const_iterator it = handles.begin();
            it != handles.end(); it++)
      {
        register_deletion(*it);
      }
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::return_deletions(const std::map<FieldID,FieldSpace> &fields)
    //--------------------------------------------------------------------------
    {
      register_deletion(fields);
    }

    //--------------------------------------------------------------------------
    void SingleTask::invalidate_matches(LogicalRegion handle, const std::map<FieldID,FieldSpace> &fields)
    //--------------------------------------------------------------------------
    {
      FieldSpace space = handle.field_space;
      std::vector<FieldID> matched_fields;
      for (std::map<FieldID,FieldSpace>::const_iterator it = fields.begin();
            it != fields.end(); it++)
      {
        if (it->second == space)
          matched_fields.push_back(it->first);
      }
      if (!matched_fields.empty())
      {
        lock_context();
        forest_ctx->invalidate_physical_context(handle, ctx_id, matched_fields);
        unlock_context();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::return_created_field_contexts(SingleTask *enclosing)
    //--------------------------------------------------------------------------
    {
      // go through all of our pre-existing regions and find ones which have newly
      // created field states
      std::vector<FieldID> new_fields;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif
        // Only need to do this if it was not virtual mapped
        if (non_virtual_mapped_region[idx])
        {
          FieldSpace space = regions[idx].region.field_space;
          for (std::map<FieldID,FieldSpace>::const_iterator it = created_fields.begin();
                it != created_fields.end(); it++)
          {
            if (it->second == space)
              new_fields.push_back(it->first);
          }
          // See if we have anything to return
          if (!new_fields.empty())
          {
            lock_context();
            enclosing->return_field_context(regions[idx].region, ctx_id/*our context*/,
                                            new_fields);
            // Now we can invalidate the new fields in our context
            forest_ctx->invalidate_physical_context(regions[idx], new_fields, 
                                                    physical_contexts[idx], true/*added only*/);
            unlock_context();
            new_fields.clear();
          }
        }
      }
    }
    
    //--------------------------------------------------------------------------
    void SingleTask::return_field_context(LogicalRegion handle, ContextID inner_ctx,
                                          const std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      // Go through our pre-existing regions, either ones we had or ones we
      // created and unpack the field state into the right context, note
      // the context better already exist.  There's no reason it shouldn't
      // other than a bug.
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Just need to find a region in the same tree for which we know the context
        if (regions[idx].region.tree_id == handle.tree_id)
        {
          forest_ctx->merge_field_context(handle, physical_contexts[idx], inner_ctx, fields); 
          return;
        }
      }
      // Otherwise just assume that it is a created region and return it
      ContextID phy_ctx = find_outermost_physical_context();
      forest_ctx->merge_field_context(handle, phy_ctx, inner_ctx, fields);
    }

    //--------------------------------------------------------------------------
    size_t SingleTask::compute_return_created_contexts(void)
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      // Go through the pre-existing trees and see if they have any fields
      // that need to be packed up.  Note we need to do this regardless
      // of whether it was virtual mapped or not since even the virtual
      // mapping ones need to be passed back.
      result += (2*sizeof(size_t)); // number of returning tree instances + number of returning new contexts
      std::vector<FieldID> new_fields;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif
        FieldSpace space = regions[idx].region.field_space; 
        for (std::map<FieldID,FieldSpace>::const_iterator it = created_fields.begin();
              it != created_fields.end(); it++)
        {
          if (it->second == space)
            new_fields.push_back(it->first);
        }
        if (!new_fields.empty())
        {
          need_pack_created_fields[idx] = new_fields;
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_contexts[idx] == ctx_id);
#endif
          result += sizeof(regions[idx].region);
          result += forest_ctx->compute_created_field_state_return(regions[idx].region,
                               new_fields, ctx_id/*should always be our context*/ 
#ifdef DEBUG_HIGH_LEVEL
                                , this->variants->name, this->get_unique_id()
#endif
                                );
        }
      }
      // Now pack up any of the created region tree contexts
      for (std::set<LogicalRegion>::const_iterator it = created_regions.begin();
            it != created_regions.end(); it++)
      {
        result += sizeof(*it); 
        result += forest_ctx->compute_created_state_return(*it, ctx_id
#ifdef DEBUG_HIGH_LEVEL
                                    , this->variants->name, this->get_unique_id()
#endif
                                    );
      }
      return result;
    }
    
    //--------------------------------------------------------------------------
    void SingleTask::pack_return_created_contexts(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(need_pack_created_fields.size());
      rez.serialize(created_regions.size());
      if (!need_pack_created_fields.empty())
      {
        for (std::map<unsigned, std::vector<FieldID> >::const_iterator it = need_pack_created_fields.begin();
              it != need_pack_created_fields.end(); it++)
        {
          rez.serialize(regions[it->first].region);
          // Recall that the act of packing return will automatically invalidate
          // these instances so we don't need to do it manually.
          forest_ctx->pack_created_field_state_return(regions[it->first].region,
                                              it->second, ctx_id, rez);
        }
        for (std::set<LogicalRegion>::const_iterator it = created_regions.begin();
              it != created_regions.end(); it++)
        {
          rez.serialize(*it);
          forest_ctx->pack_created_state_return(*it, ctx_id, rez);
        }
        need_pack_created_fields.clear();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_return_created_contexts(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_returning_field_contexts; 
      derez.deserialize(num_returning_field_contexts);
      size_t num_returning_new_contexts;
      derez.deserialize(num_returning_new_contexts);
      if ((num_returning_field_contexts > 0) || (num_returning_new_contexts > 0))
      {
        ContextID outermost_ctx = find_outermost_physical_context();
        for (unsigned idx = 0; idx < num_returning_field_contexts; idx++)
        {
          // Unpack the logical region handle and find on of our trees
          // with the same tree id so we can get the context
          LogicalRegion handle;
          derez.deserialize(handle);
          bool found = false;
          for (unsigned ridx = 0; ridx < regions.size(); ridx++)
          {
            if (regions[ridx].region.tree_id == handle.tree_id)
            {
              // Get the right context
              found = true;
              ContextID phy_ctx = physical_contexts[ridx];
              forest_ctx->unpack_created_field_state_return(handle, phy_ctx, derez
#ifdef DEBUG_HIGH_LEVEL
                                                            , this->variants->name
                                                            , this->get_unique_id()
#endif
                                                            );
              break;
            }
          }
          if (found)
            continue;
          // Not in a pre-existing tree, try the created trees
          for (std::set<LogicalRegion>::const_iterator it = created_regions.begin();
                it != created_regions.end(); it++)
          {
            if (it->tree_id == handle.tree_id)
            {
              found = true;
              forest_ctx->unpack_created_field_state_return(handle, outermost_ctx, derez
#ifdef DEBUG_HIGH_LEVEL
                                                            , this->variants->name
                                                            , this->get_unique_id()
#endif
                                                            );
              break;
            }
          }
          if (!found)
            assert(false); // should never happen
        }
        for (unsigned idx = 0; idx < num_returning_new_contexts; idx++)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          forest_ctx->unpack_created_state_return(handle, outermost_ctx, derez
#ifdef DEBUG_HIGH_LEVEL
                                                  , this->variants->name
                                                  , this->get_unique_id()
#endif
                                                  );
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t SingleTask::compute_source_copy_instances_return(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = sizeof(size_t); // number of returning instances
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        result += forest_ctx->compute_reference_size_return(source_copy_instances[idx]);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_source_copy_instances_return(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      rez.serialize<size_t>(source_copy_instances.size());
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        forest_ctx->pack_reference_return(source_copy_instances[idx], rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::unpack_source_copy_instances_return(Deserializer &derez, RegionTreeForest *forest, UniqueID uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      forest->assert_locked();
#endif
      size_t num_refs;
      derez.deserialize<size_t>(num_refs);
      for (unsigned idx = 0; idx < num_refs; idx++)
      {
        forest->unpack_and_remove_reference(derez, uid);
      }
    }

    //--------------------------------------------------------------------------
    size_t SingleTask::compute_single_task_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_task_context_size();
      result += sizeof(bool); // regions mapped
      if (!non_virtual_mapped_region.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(non_virtual_mapped_region.size() == regions.size());
#endif
        result += (regions.size() * sizeof(bool));
        result += sizeof(executing_processor);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_task_context(rez);
      bool has_mapped = !non_virtual_mapped_region.empty();
      rez.serialize<bool>(has_mapped);
      if (!non_virtual_mapped_region.empty())
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          bool non_virt = non_virtual_mapped_region[idx];
          rez.serialize<bool>(non_virt);
        }
        rez.serialize<Processor>(executing_processor);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_task_context(derez);
      bool has_mapped;
      derez.deserialize<bool>(has_mapped);
      if (has_mapped)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          bool non_virt;
          derez.deserialize<bool>(non_virt);
          non_virtual_mapped_region.push_back(non_virt);
        }
        derez.deserialize(executing_processor);
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(Processor target, Event single_term, Event multi_term)
    //--------------------------------------------------------------------------
    {
      bool map_success = true;
      // First select the variant for this task
      low_id = invoke_mapper_select_variant(target);
      // Do the mapping for all the regions
      // Hold the context lock when doing this
      forest_ctx->lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // See if this region was pre-mapped
        InstanceRef premapped = find_premapped_region(idx);
        // If it was premapped or it has no instance fields to map, continue
        if (!premapped.is_virtual_ref() ||
            (regions[idx].instance_fields.empty()))
        {
          lock();
          non_virtual_mapped_region.push_back(true);
          physical_instances.push_back(premapped);
          physical_contexts.push_back(ctx_id);
          unlock();
          continue;
        }
        // Otherwise see if we're going to do an actual mapping
        ContextID phy_ctx = get_enclosing_physical_context(idx);
        // First check to see if we want to map the given region  
        if (invoke_mapper_map_region_virtual(idx, target))
        {
          // Can't be a leaf task
          if (is_leaf())
          {
            log_region(LEVEL_ERROR,"Illegal request for a virtual mapping of region (%x,%d,%d) index %d "
                                  "in LEAF task %s (ID %d)", regions[idx].region.index_space.id,
                                  regions[idx].region.field_space.id, regions[idx].region.tree_id, idx,
                                  this->variants->name, get_unique_id());
            exit(ERROR_VIRTUAL_MAP_IN_LEAF_TASK);
          }
          // Want a virtual mapping
          lock();
          unmapped++;
          non_virtual_mapped_region.push_back(false);
          physical_instances.push_back(InstanceRef());
          physical_contexts.push_back(phy_ctx); // use same context as parent for all child mappings
          unlock();
        }
        else
        {
          // Otherwise we want to do an actual physical mapping
          RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock, target, 
                                  single_term, multi_term, tag, false/*sanitizing*/,
                                  false/*inline mapping*/, source_copy_instances);
          // Compute the path 
          // If the region was sanitized, we only need to do the path from the region itself
          if (regions[idx].sanitized)
          {
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            forest_ctx->compute_index_path(regions[idx].region.index_space, regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
            assert(result);
#endif
            forest_ctx->map_region(reg_mapper, regions[idx].region);
          }
          else
          {
            // Not sanitized so map from the parent
#ifdef DEBUG_HIGH_LEVEL
            bool result = 
#endif
            forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
            assert(result);
            assert(!reg_mapper.success);
#endif
            forest_ctx->map_region(reg_mapper, regions[idx].parent);
          }
          lock();
          physical_instances.push_back(reg_mapper.result);
          // Check to make sure that the result isn't virtual, if it is then the mapping failed
          if (physical_instances[idx].is_virtual_ref())
          {
            unlock();
            assert(!reg_mapper.success);
            // Mapping failed
            invoke_mapper_failed_mapping(idx, target);
            map_success = false;
            physical_instances.pop_back();
            break;
          }
          non_virtual_mapped_region.push_back(true);
          physical_contexts.push_back(ctx_id); // use our context for all child mappings
          unlock();
        }
      }
      forest_ctx->unlock_context();

      if (map_success)
      {
        // Since the mapping was a success we now know that we're going to run on this processor
        this->executing_processor = target;
      }
      else
      {
        // Mapping failed so undo everything that was done
        forest_ctx->lock_context();
        lock();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          physical_instances[idx].remove_reference(unique_id, true/*strict*/);
        }
        forest_ctx->unlock_context();
        physical_instances.clear();
        physical_contexts.clear();
        non_virtual_mapped_region.clear();
        unmapped = 0;
        unlock();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(is_remote());
#endif
        finish_task_unpack();
      }
      initialize_region_tree_contexts();

      std::set<Event> wait_on_events = launch_preconditions;
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
#endif
      // Sort the lock order requests by instance manager id
      // to remove any possibility of deadlock
      std::map<UniqueManagerID,unsigned/*idx*/> needed_locks;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (!physical_instances[idx].is_virtual_ref())
        {
          Event precondition = physical_instances[idx].get_ready_event();
          // Do we need to acquire a lock for this region
          if (physical_instances[idx].has_required_lock())
          {
            needed_locks[physical_instances[idx].get_manager()->get_unique_id()] = idx;
            Lock atomic_lock = physical_instances[idx].get_required_lock();
            Event atomic_pre = atomic_lock.lock(0,true/*exclusive*/,precondition);
            wait_on_events.insert(atomic_pre);
          }
          else
          {
            // No need for a lock here
            wait_on_events.insert(precondition);
          }
        }
      }

      if (needed_locks.empty())
      {
        for (std::map<UniqueManagerID,unsigned>::const_iterator it = needed_locks.begin();
              it != needed_locks.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!physical_instances[it->second].is_virtual_ref());
          assert(physical_instances[it->second].has_required_lock());
#endif
          Lock atomic_lock = physical_instances[it->second].get_required_lock();
#ifdef DEBUG_HIGH_LEVEL
          assert(atomic_lock.exists());
#endif
          Event precondition = physical_instances[it->second].get_ready_event();
          Event atomic_pre = atomic_lock.lock(0,true/*exclusive*/,precondition);
          wait_on_events.insert(atomic_pre);
#ifdef LEGION_SPY
          if (precondition.exists() && atomic_pre.exists())
          {
            LegionSpy::log_event_dependence(precondition, atomic_pre);
          }
#endif
        }
      }

      Event start_condition = Event::merge_events(wait_on_events);
#ifdef LEGION_SPY
      LegionSpy::log_task_name(this->get_unique_id(), this->variants->name);
      if (!start_condition.exists())
      {
        UserEvent new_start = UserEvent::create_user_event();
        new_start.trigger();
        start_condition = new_start;
      }
      // Record the dependences
      LegionSpy::log_event_dependences(wait_on_events, start_condition);
      LegionSpy::log_task_events(runtime->utility_proc.id, this->get_gen(), this->ctx_id, get_unique_id(), is_index_space, 
                                  (is_index_space ? *((unsigned*)index_point) : 0), 
                                  start_condition, get_termination_event());
#endif
      // Launch the task, passing the pointer to this Context as the argument
      SingleTask *this_ptr = this; // dumb c++
#ifdef DEBUG_HIGH_LEVEL
      assert(this->executing_processor.exists());
      assert(low_id > 0);
#endif
      Event task_launch_event = this->executing_processor.spawn(low_id,&this_ptr,sizeof(SingleTask*),start_condition);

      // After we launched the task, see if we had any atomic locks to release
      if (!needed_locks.empty())
      {
        for (std::map<UniqueManagerID,unsigned>::const_iterator it = needed_locks.begin();
              it != needed_locks.end(); it++)
        {
          Lock atomic_lock = physical_instances[it->second].get_required_lock();
          // Release the lock once the task is done
          atomic_lock.unlock(task_launch_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_region_tree_contexts(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
#endif
      lock_context();
      // For all of the regions we need to initialize the logical contexts
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR); // this better be true for single tasks
#endif
        forest_ctx->initialize_logical_context(regions[idx].region, ctx_id);
      }
      // For all of the physical contexts that were mapped, initialize them
      // with a specified reference, otherwise make them a virtual reference
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        clone_instances.push_back(forest_ctx->initialize_physical_context(regions[idx],
                                                    physical_instances[idx], unique_id, ctx_id));
      }
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void SingleTask::release_source_copy_instances(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
      {
        source_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
      }
      source_copy_instances.clear();
    }

    //--------------------------------------------------------------------------
    void SingleTask::flush_deletions(std::set<Event> &cleanup_events)
    //--------------------------------------------------------------------------
    {
      for (std::list<DeletionOperation*>::const_iterator it = child_deletions.begin();
            it != child_deletions.end(); it++)
      {
        cleanup_events.insert((*it)->flush());
      }
      child_deletions.clear();
    }

    //--------------------------------------------------------------------------
    void SingleTask::issue_restoring_copies(std::set<Event> &wait_on_events, 
                                          Event single_event, Event multi_event)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (!physical_instances[idx].is_virtual_ref())
        {
          // Only need to do the close if there is a possiblity of dirty data
          if (IS_WRITE(regions[idx]))
          {
            // This should be our physical context since we mapped the region
            RegionMapper rm(this, unique_id, ctx_id, idx, regions[idx], NULL/*shouldn't need it*/, mapper_lock,
                            Processor::NO_PROC, single_event, multi_event,
                            tag, false/*sanitizing*/, false/*inline mapping*/, source_copy_instances);
            // First remove our reference so we don't accidentally end up waiting on ourself
            InstanceRef clone_ref = clone_instances[idx];
            clone_instances[idx].remove_reference(unique_id, true/*strict*/);
            Event close_event = forest_ctx->close_to_instance(clone_ref, rm);
#ifdef DEBUG_HIGH_LEVEL
            assert(close_event != single_event);
            assert(close_event != multi_event);
#endif
            wait_on_events.insert(close_event);
          }
          else if (IS_REDUCE(regions[idx]))
          {
            RegionMapper rm(this, unique_id, ctx_id, idx, regions[idx], NULL/*shouldn't need it*/, mapper_lock,
                            Processor::NO_PROC, single_event, multi_event,
                            tag, false/*sanitizing*/, false/*inline mapping*/, source_copy_instances);
            // First remove our reference so we don't accidentally end up waiting on ourself
            InstanceRef clone_ref = clone_instances[idx];
            clone_instances[idx].remove_reference(unique_id, true/*strict*/);
            // Close to the reduction instance
            Event close_event = forest_ctx->close_to_reduction(clone_ref, rm);
#ifdef DEBUG_HIGH_LEVEL
            assert(close_event != single_event);
            assert(close_event != multi_event);
#endif
            wait_on_events.insert(close_event);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invalidate_owned_contexts(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == physical_contexts.size());
#endif
      // Only invalidate the contexts which were pre-existing and are not
      // going to be sent back.  Any created state will get invalidated later.
      std::vector<FieldID> added_fields;
      lock_context();
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
 #ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif       
        if (non_virtual_mapped_region[idx])
        {
          forest_ctx->invalidate_physical_context(regions[idx], added_fields, physical_contexts[idx], false/*added only*/);
        }
      }
      unlock_context();
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(HighLevelRuntime *rt, ContextID id)
      : TaskContext(rt, id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool MultiTask::activate_multi(GeneralizedOperation *parent)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_task(parent);
      if (activated)
      {
        index_space = IndexSpace::NO_SPACE;
        sliced = false;
        has_reduction = false;
        redop_id = 0;
        reduction_state = NULL;
        reduction_state_size = 0;
        arg_map_impl = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate_multi(void)
    //--------------------------------------------------------------------------
    {
      if (reduction_state != NULL)
      {
        free(reduction_state);
        reduction_state = NULL;
      }
      if (arg_map_impl != NULL)
      {
        if (arg_map_impl->remove_reference())
        {
          delete arg_map_impl;
        }
      }
      slices.clear();
      deactivate_task();
    }

    //--------------------------------------------------------------------------
    bool MultiTask::perform_operation(void)
    //--------------------------------------------------------------------------
    {
      bool success = true;
      if (is_locally_mapped())
      {
        // Slice first, then map, finally distribute 
        if (is_sliced())
        {
          if (!is_distributed() && !is_stolen())
          {
            // Task is still on the originating processor
            // so we have to do the mapping now
            if (perform_mapping())
            {
              if (distribute_task())
              {
                launch_task();
              }
            }
            else
            {
              success = false;
            }
          }
          else
          {
            if (!is_distributed())
            {
              if (distribute_task())
              {
                launch_task();
              }
            }
            else
            {
              // Already been distributed, so we can launch it now
              launch_task();
            }
          }
        }
        else
        {
          // Will recursively invoke perform_operation
          // on the new slice tasks
          success = slice_index_space();
        }
      }
      else // Not locally mapped
      {
        // Check to make sure that our region trees
        // have been sanitized
        if (is_stolen() || sanitize_region_forest())
        {
          // Distribute first, then slice, finally map
          // Check if we need to sanitize
          if (!is_distributed())
          {
            // Try distributing, if still local
            // then go about slicing
            if (distribute_task())
            {
              if (is_sliced())
              {
                // This task has been sliced and is local
                // so map it and launch it
                success = map_and_launch();
              }
              else
              {
                // Task to be sliced on this processor
                // Will recursively invoke perform_operation
                // on the new slice tasks
                success = slice_index_space();
              }
            }
          }
          else // Already been distributed
          {
            if (is_sliced())
            {
              success = map_and_launch();
            }
            else
            {
              success = slice_index_space();
            }
          }
        }
        else
          success = false; // sanitization failed
      }
      return success;
    }

    //--------------------------------------------------------------------------
    void MultiTask::return_deletions(const std::vector<LogicalRegion> &region_handles,
                                     const std::vector<LogicalPartition> &partition_handles,
                                     const std::map<FieldID,FieldSpace> &fields)
    //--------------------------------------------------------------------------
    {
      deleted_regions.insert(deleted_regions.end(),region_handles.begin(),region_handles.end());
      deleted_partitions.insert(deleted_partitions.end(),partition_handles.begin(),partition_handles.end());
      deleted_fields.insert(fields.begin(),fields.end());
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void)
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
      std::vector<Mapper::IndexSplit> splits;
      {
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);
        mapper->slice_index_space(this, index_space, splits);
#ifdef DEBUG_HIGH_LEVEL
        assert(!splits.empty());
#endif
      }
      // TODO: add a check here that the split index spaces
      // are a total of the original index space.
#ifdef DEBUG_HIGH_LEVEL
      assert(!splits.empty());
#endif
      // Make sure we can pre-slice this task
      if (!pre_slice())
        return false;

      for (unsigned idx = 0; idx < splits.size(); idx++)
      {
        SliceTask *slice = this->clone_as_slice_task(splits[idx].space,
                                                     splits[idx].proc,
                                                     splits[idx].recurse,
                                                     splits[idx].stealable);
        slices.push_back(slice);
      }

      // If we're doing must parallelism, increase the barrier count
      // by the number of new slices created.  We can subtract one
      // because we were already anticipating one arrival for this slice
      // that will now no longer happen.
      if (must_parallelism)
      {
        must_barrier.alter_arrival_count(slices.size()-1);
      }

      // This will tell each of the slices what their denominator should be
      // and will return whether or not to deactivate the current slice
      // because it no longer contains any parts of the index space.
      bool reclaim = post_slice();

      bool success = true;
      // Now invoke perform_operation on all of the slices, keep around
      // any that aren't successfully performed
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        bool slice_success = (*it)->perform_operation();
        if (!slice_success)
        {
          success = false;
          it++;
        }
        else
        {
          // Remove it from the list since we're done
          it = slices.erase(it);
        }
      }

      // Reclaim if we should and everything was a success
      if (reclaim && success)
        this->deactivate();
      return success;
    }

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(MultiTask *rhs, IndexSpace new_space, bool recurse)
    //--------------------------------------------------------------------------
    {
      this->clone_task_context_from(rhs);
      this->index_space = new_space;
      this->sliced = !recurse;
      this->has_reduction = rhs->has_reduction;
      if (has_reduction)
      {
        this->redop_id = rhs->redop_id;
        this->reduction_state = malloc(rhs->reduction_state_size);
        memcpy(this->reduction_state,rhs->reduction_state,rhs->reduction_state_size);
        this->reduction_state_size = rhs->reduction_state_size;
      }
      if (must_parallelism)
      {
        this->must_barrier = rhs->must_barrier;
      }
      this->arg_map_impl = rhs->arg_map_impl;
      this->arg_map_impl->add_reference();
      this->premapped_regions = rhs->premapped_regions;
    }

    //--------------------------------------------------------------------------
    size_t MultiTask::compute_multi_task_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = compute_task_context_size();
      result += sizeof(sliced);
      result += sizeof(has_reduction);
      if (has_reduction)
      {
        result += sizeof(redop_id);
        result += sizeof(reduction_state_size);
        result += reduction_state_size;
      }
      if (must_parallelism)
        result += sizeof(must_barrier);
      // ArgumentMap handled by sub-types since it is packed in
      // some cases but not others
      return result;
    }

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      pack_task_context(rez);
      rez.serialize<bool>(sliced);
      rez.serialize<bool>(has_reduction);
      if (has_reduction)
      {
        rez.serialize<ReductionOpID>(redop_id);
        rez.serialize<size_t>(reduction_state_size);
        rez.serialize(reduction_state,reduction_state_size);
      }
      if (must_parallelism)
        rez.serialize<Barrier>(must_barrier);
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_task_context(derez);
      derez.deserialize<bool>(sliced);
      derez.deserialize<bool>(has_reduction);
      if (has_reduction)
      {
        derez.deserialize<ReductionOpID>(redop_id);
        derez.deserialize<size_t>(reduction_state_size);
#ifdef DEBUG_HIGH_LEVEL
        assert(reduction_state == NULL);
#endif
        reduction_state = malloc(reduction_state_size);
        derez.deserialize(reduction_state,reduction_state_size);
      }
      if (must_parallelism)
        derez.deserialize<Barrier>(must_barrier);
    }

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(HighLevelRuntime *rt, ContextID id)
      : SingleTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_single(parent);
      if (activated)
      {
        current_proc = Processor::NO_PROC;
        distributed = false;
        locally_set = false;
        locally_mapped = false;
        stealable_set = false;
        stealable = false;
        remote = false;
        top_level_task = false;
        future = NULL;
        remote_future = NULL;
        remote_future_len = 0;
        orig_ctx = this;
        remote_start_event = Event::NO_EVENT;
        remote_mapped_event = Event::NO_EVENT;
        partially_unpacked = false;
        remaining_buffer = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      lock();
      if (future != NULL)
      {
        if (future->remove_reference())
        {
          delete future;
        }
        future = NULL;
      }
      if (remote_future != NULL)
      {
        free(remote_future);
        remote_future = NULL;
        remote_future_len = 0;
      }
      if (remaining_buffer != NULL)
      {
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
      }
      unlock();
      deactivate_single();
      // Free this back up to the runtime
      runtime->free_individual_task(this);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation);
#endif
      bool result;
      do {
        if (gen < generation)
        {
          result = false; // This task has already been recycled
          break;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < map_dependent_waiters.size());
#endif
        // Check to see if everything has been mapped
        if ((idx >= non_virtual_mapped_region.size()) ||
            !non_virtual_mapped_region[idx])
        {
          // hasn't been mapped yet, try adding it
          std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
            map_dependent_waiters[idx].insert(waiter);
          result = added.second;
        }
        else
        {
          // It's already been mapped
          result = false;
        }
      } while (false);
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
#endif
      lock();
      if (task_pred == Predicate::TRUE_PRED)
      {
        // Task evaluated should be run, put it on the ready queue
        // for the processor that it originated on
        unlock();
        runtime->add_to_ready_queue(orig_proc, this,false/*remote*/);
      }
      else if (task_pred == Predicate::FALSE_PRED)
      {
        unlock();
      }
      else
      {
        // TODO: handle predication
        assert(false); 
      }
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      return distributed;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if we've already evaluated it
      if (!locally_set)
      {
        locally_mapped = invoke_mapper_locally_mapped();
        locally_set = true;
        // Locally mapped tasks are not stealable
        if (locally_mapped)
        {
          stealable = false;
          stealable_set = true;
        }
      }
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      if (!stealable_set)
      {
        // Check to make sure locally mapped is set first so
        // we only ask about stealing if we're not locally mapped
        if (!is_locally_mapped())
        {
          stealable = invoke_mapper_stealable();
          stealable_set = true;
        }
      }
      return stealable;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      return remote;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      return partially_unpacked;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!distributed);
      assert(current_proc.exists());
#endif
      // Allow this to be re-entrant in case sanitization fails
      Processor target_proc = invoke_mapper_select_target_proc();
#ifdef DEBUG_HIGH_LEVEL
      if (!target_proc.exists())
      {
        log_run(LEVEL_ERROR,"Invalid mapper selection of NO_PROC for target processor of task %s (ID %d)",
                          variants->name, get_unique_id());
        exit(ERROR_INVALID_PROCESSOR_SELECTION);
      }
#endif
      distributed = true;
      // If the target processor isn't us we have to
      // send our task away
      if (target_proc != current_proc)
      {
        // Update the current proc and send it
        current_proc = target_proc;
        return runtime->send_task(target_proc, this);
      }
      // Definitely still local
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      bool map_success = map_all_regions(current_proc, termination_event, termination_event); 
      if (map_success)
      {
        // Mark that we're no longer stealable now that we've been mapped
        stealable = false;
        stealable_set = true;
        // If we're remote, send back our mapping information
        if (remote)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!locally_mapped); // we shouldn't be here if we were locally mapped
#endif
          size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx);
          buffer_size += (regions.size()*sizeof(bool)); // mapped or not for each region
          lock_context();
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx], idx, ctx_id, 
                                              IS_WRITE(regions[idx]), RegionTreeForest::PHYSICAL 
#ifdef DEBUG_HIGH_LEVEL
                                              , variants->name
                                              , this->get_unique_id()
#endif
                                              );
            }
          }
          buffer_size += forest_ctx->post_compute_region_tree_state_return(false/*created only*/);
          // Now pack everything up and send it back
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            rez.serialize<bool>(non_virtual_mapped_region[idx]);
          }
          forest_ctx->begin_pack_region_tree_state_return(rez, false/*created only*/);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              forest_ctx->pack_region_tree_state_return(regions[idx], idx, ctx_id, 
                IS_WRITE(regions[idx]), RegionTreeForest::PHYSICAL, rez);
            }
          }
          forest_ctx->end_pack_region_tree_state_return(rez, false/*created only*/);
          unlock_context();
          // Now send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_start_event = utility.spawn(NOTIFY_START_ID,rez.get_buffer(),buffer_size);
        }
        else
        {
          // Hold the lock to prevent new waiters from registering
          lock_context();
          lock();
          // notify any tasks that we have waiting on us
#ifdef DEBUG_HIGH_LEVEL
          assert(map_dependent_waiters.size() == regions.size());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mapped_region[idx])
            {
              std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
              for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                    it != waiters.end(); it++)
              {
                (*it)->notify();
              }
              waiters.clear();
            }
          }
          unlock();
          unlock_context();
          if (unmapped == 0)
          {
            // If everything has been mapped, then trigger the mapped event
            mapped_event.trigger();
          }
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!remote);
#endif
      // For each of our regions perform a walk on the physical tree to 
      // destination region, but without doing any mapping.  Then update
      // the parent region in the region requirements so that we only have
      // to walk from the target region.

      bool result = true;
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(regions[idx].handle_type == SINGULAR);
#endif
        // Check to see if this region is already sanitized
        if (regions[idx].sanitized)
          continue;
        ContextID phy_ctx = get_enclosing_physical_context(idx);
        // Create a sanitizing region mapper and map it
        RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock,
                                parent_ctx->get_executing_processor(), termination_event, termination_event,
                                tag, true/*sanitizing*/, false/*inline mapping*/,
                                source_copy_instances);
#ifdef DEBUG_HIGH_LEVEL
        bool result = 
#endif
        forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
        assert(result); // better have been able to compute the path
#endif
        // Now do the sanitizing walk 
        forest_ctx->map_region(reg_mapper, regions[idx].parent);
#ifdef DEBUG_HIGH_LEVEL
        assert(reg_mapper.path.empty());
#endif
        if (reg_mapper.success)
        {
          regions[idx].sanitized = true; 
        }
        else
        {
          // Couldn't sanitize the tree
          result = false;
          break;
        }
      }
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      mapped_event = UserEvent::create_user_event();
      termination_event = UserEvent::create_user_event();
      current_proc = this->orig_proc;
    }

    //--------------------------------------------------------------------------
    Event IndividualTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    Event IndividualTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------
    ContextID IndividualTask::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // If we're remote, then everything is already in our own context ID
      if (remote)
        return ctx_id;
      else
        return parent_ctx->find_enclosing_physical_context(regions[idx].parent);
    }

    //--------------------------------------------------------------------------
    size_t IndividualTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = compute_single_task_size();
      result += sizeof(distributed);
      result += sizeof(locally_mapped);
      result += sizeof(stealable);
      result += sizeof(termination_event);
      result += sizeof(Context);
      result += sizeof(current_proc);
      if (partially_unpacked)
      {
        result += remaining_bytes;
      }
      else
      {
        if (locally_mapped)
        {
          if (is_leaf())
          {
            // Don't need to pack the region trees, but still
            // need to pack the instances
#ifdef DEBUG_HIGH_LEVEL
            assert(regions.size() == physical_instances.size());
#endif
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              result += forest_ctx->compute_reference_size(physical_instances[idx]);
            }
          }
          else
          {
            // Need to pack the region trees and the instances
            // or the states if they were virtually mapped
            result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (physical_instances[idx].is_virtual_ref())
              {
                // Virtual mapping, pack the state
                result += forest_ctx->compute_region_tree_state_size(regions[idx], 
                                        get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE);
              }
              else
              {
                result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                        get_enclosing_physical_context(idx), RegionTreeForest::DIFF);
                result += forest_ctx->compute_reference_size(physical_instances[idx]);
              }
            }
          }
        }
        else
        {
          // Need to pack the region trees and states
          result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                    get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE);
          }
        }
        result += forest_ctx->post_compute_region_tree_state_size();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      pack_single_task(rez);
      rez.serialize<bool>(distributed);
      rez.serialize<bool>(locally_mapped);
      rez.serialize<bool>(stealable);
      rez.serialize<UserEvent>(termination_event);
      rez.serialize<Context>(orig_ctx);
      rez.serialize<Processor>(current_proc);
      if (partially_unpacked)
      {
        rez.serialize(remaining_buffer,remaining_bytes);
      }
      else
      {
        if (locally_mapped)
        {
          if (is_leaf())
          {
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              forest_ctx->pack_reference(physical_instances[idx], rez);
            }
          }
          else
          {
            forest_ctx->pack_region_forest_shape(rez); 
            forest_ctx->begin_pack_region_tree_state(rez);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (physical_instances[idx].is_virtual_ref())
              {
                forest_ctx->pack_region_tree_state(regions[idx],
                              get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE, rez
#ifdef DEBUG_HIGH_LEVEL
                              , idx, variants->name
                              , this->get_unique_id()
#endif
                              );
              }
              else
              {
                forest_ctx->pack_region_tree_state(regions[idx],
                              get_enclosing_physical_context(idx), RegionTreeForest::DIFF, rez
#ifdef DEBUG_HIGH_LEVEL
                              , idx, variants->name
                              , this->get_unique_id()
#endif
                              );
                forest_ctx->pack_reference(physical_instances[idx], rez);
              }
            }
          }
        }
        else
        {
          forest_ctx->pack_region_forest_shape(rez);
          forest_ctx->begin_pack_region_tree_state(rez);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            forest_ctx->pack_region_tree_state(regions[idx], 
                            get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE, rez
#ifdef DEBUG_HIGH_LEVEL
                            , idx, variants->name
                            , this->get_unique_id()
#endif
                            );
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_single_task(derez);
      derez.deserialize<bool>(distributed);
      derez.deserialize<bool>(locally_mapped);
      locally_set = true;
      derez.deserialize<bool>(stealable);
      stealable_set = true;
      remote = true;
      derez.deserialize<UserEvent>(termination_event);
      derez.deserialize<Context>(orig_ctx);
      derez.deserialize<Processor>(current_proc);
      remaining_bytes = derez.get_remaining_bytes();
      remaining_buffer = malloc(remaining_bytes);
      derez.deserialize(remaining_buffer,remaining_bytes);
      partially_unpacked = true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partially_unpacked);
#endif
      Deserializer derez(remaining_buffer,remaining_bytes);
      lock_context();
      if (locally_mapped)
      {
        if (is_leaf())
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(physical_instances.empty());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            physical_instances.push_back(forest_ctx->unpack_reference(derez));
          }
        }
        else
        {
          forest_ctx->unpack_region_forest_shape(derez);
          forest_ctx->begin_unpack_region_tree_state(derez, ctx_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(non_virtual_mapped_region.size() == regions.size());
#endif
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              // Unpack the state in our context
              forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::PRIVILEGE, derez
#ifdef DEBUG_HIGH_LEVEL
                  , idx, variants->name, this->get_unique_id()
#endif
                  ); 
              physical_instances.push_back(InstanceRef()); // virtual instance
            }
            else
            {
              forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::DIFF, derez
#ifdef DEBUG_HIGH_LEVEL
                  , idx, variants->name, this->get_unique_id()
#endif
                  );
              physical_instances.push_back(forest_ctx->unpack_reference(derez)); 
            }
          }
        }
      }
      else
      {
        forest_ctx->unpack_region_forest_shape(derez);
        forest_ctx->begin_unpack_region_tree_state(derez, ctx_id);
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          // Unpack the state in our context
          forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::PRIVILEGE, derez
#ifdef DEBUG_HIGH_LEVEL
              , idx, variants->name, this->get_unique_id()
#endif
              );
        }
      }
      unlock_context();
      free(remaining_buffer);
      remaining_buffer = NULL;
      remaining_bytes = 0;
      partially_unpacked = false;
    }

    //--------------------------------------------------------------------------
    InstanceRef IndividualTask::find_premapped_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // Never has pre-mapped regions
      return InstanceRef();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf()); // shouldn't be here if we're a leaf task
#endif
      std::set<Event> cleanup_events;
      // Make sure all the deletion operations for this task have been performed
      // to ensure that the region tree is in a good state either to be sent back
      // or for other users to begin using it.
      flush_deletions(cleanup_events);

      lock_context();
      // Get the termination events for all of the tasks
      {
        lock(); // need lock to touch child tasks
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          cleanup_events.insert((*it)->get_termination_event());
        }
        unlock();
      }
      // Issue the restoring copies for this task
      issue_restoring_copies(cleanup_events, termination_event, termination_event);
      unlock_context();

      if (remote)
      {
        // Only need to send things back if we had unmapped regions
        // and this isn't a leaf task.  Note virtual mappings on leaf
        // tasks are pretty worthless.
        if ((unmapped > 0) && !is_leaf())
        {
          size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx);
          lock_context();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          // Figure out which states we need to send back
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx], idx, ctx_id, 
                                                IS_WRITE(regions[idx]), RegionTreeForest::PRIVILEGE
#ifdef DEBUG_HIGH_LEVEL
                                                , variants->name, this->get_unique_id()
#endif
                                                );
            }
            else
            {
              // Physical was already sent back, send back the other fields
              buffer_size += forest_ctx->compute_region_tree_state_return(regions[idx], idx, ctx_id, 
                                                IS_WRITE(regions[idx]), RegionTreeForest::DIFF
#ifdef DEBUG_HIGH_LEVEL
                                                , variants->name, this->get_unique_id()
#endif
                                                );
            }
          }
          buffer_size += forest_ctx->post_compute_region_tree_state_return(false/*created only*/);
          // Finally pack up our source copy instances to send back
          buffer_size += compute_source_copy_instances_return();
          // Now pack it all up
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<Context>(orig_ctx);
          forest_ctx->pack_region_tree_updates_return(rez);
          forest_ctx->begin_pack_region_tree_state_return(rez, false/*created only*/);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (!non_virtual_mapped_region[idx])
            {
              forest_ctx->pack_region_tree_state_return(regions[idx], idx, ctx_id, 
                IS_WRITE(regions[idx]), RegionTreeForest::PRIVILEGE, rez);
            }
            else
            {
              forest_ctx->pack_region_tree_state_return(regions[idx], idx, ctx_id, 
                IS_WRITE(regions[idx]), RegionTreeForest::DIFF, rez);
            }
          }
          forest_ctx->end_pack_region_tree_state_return(rez, false/*created only*/);
          // Pack up the source copy instances
          pack_source_copy_instances_return(rez);
          unlock_context();
          // Send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_mapped_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,this->remote_start_event);
        }
      }
      else
      {
        // Otherwise notify all the waiters on virtual mapped regions
#ifdef DEBUG_HIGH_LEVEL
        assert(map_dependent_waiters.size() == regions.size());
        assert(non_virtual_mapped_region.size() == regions.size());
#endif
        // Hold the lock to prevent new waiters from registering
        lock();
        for (unsigned idx = 0; idx < map_dependent_waiters.size(); idx++)
        {
          if (!non_virtual_mapped_region[idx])
          {
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
        unlock();

        // If we haven't triggered it yet, trigger the mapped event
        if (unmapped > 0)
        {
          mapped_event.trigger();
          unmapped = 0;
        }
      }

      // Now can invalidate any valid physical instances our context for
      // all the regions which we had as region requirements.  We can't invalidate
      // virtual mapped regions since that information has escaped our context.
      // Note we don't need to worry about doing this for leaf tasks since there
      // are no mappings performed in a leaf task context.  We also don't have to
      // do this for any created regions since either they are explicitly deleted
      // or their state is passed back in finish_task to the enclosing context.
      invalidate_owned_contexts();

      // Figure out whether we need to wait to launch the finish task
      Event wait_on_event = Event::merge_events(cleanup_events);
      if (!wait_on_event.exists())
      {
        finish_task();
      }
      else
      {
        size_t buffer_size = sizeof(Processor)+sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(runtime->utility_proc);
        rez.serialize<Context>(this);
        // Launch the task on the utility processor
        Processor utility = runtime->utility_proc;
        utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,wait_on_event);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::finish_task(void)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        size_t buffer_size = sizeof(orig_proc) + sizeof(orig_ctx);
        // Only need to send this stuff back if we're not a leaf task
        lock_context();
        if (!is_leaf())
        {
          buffer_size += compute_privileges_return_size();
          buffer_size += compute_deletions_return_size();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          buffer_size += compute_return_created_contexts();
          buffer_size += forest_ctx->post_compute_region_tree_state_return(true/*create only*/);
        }
        // Always need to send back the leaked references
        buffer_size += forest_ctx->compute_leaked_return_size();
        buffer_size += sizeof(remote_future_len);
        buffer_size += remote_future_len;
        // Now pack everything up
        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<Context>(orig_ctx);
        if (!is_leaf())
        {
          pack_privileges_return(rez);
          pack_deletions_return(rez);
          forest_ctx->pack_region_tree_updates_return(rez);
          forest_ctx->begin_pack_region_tree_state_return(rez, true/*created only*/);
          pack_return_created_contexts(rez);
          forest_ctx->end_pack_region_tree_state_return(rez, true/*created only*/);
        }
        forest_ctx->pack_leaked_return(rez);
        unlock_context();
        rez.serialize<size_t>(remote_future_len);
        rez.serialize(remote_future,remote_future_len);
        // Send this back to the utility processor.  The event we wait on
        // depends on whether this is a leaf task or not
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(remote_start_event,remote_mapped_event));
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
#endif
        // Remove the mapped references, note in the remote version
        // this will happen via the leaked references mechanism
        lock_context();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (!physical_instances[idx].is_virtual_ref())
          {
            physical_instances[idx].remove_reference(unique_id, false/*strict*/);
          }
        }
        // We also remove the source copy instances that got generated by
        // any close copies that were performed
        for (unsigned idx = 0; idx < close_copy_instances.size(); idx++)
        {
          close_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
        }
        unlock_context();
        physical_instances.clear();
        close_copy_instances.clear();
        // Send back privileges for any added operations
        // Note parent_ctx is only NULL if this is the top level task
        if (parent_ctx != NULL)
        {
          parent_ctx->return_privileges(created_index_spaces,created_field_spaces,
                                        created_regions,created_fields);
          if (!deleted_regions.empty())
            parent_ctx->return_deletions(deleted_regions);
          if (!deleted_partitions.empty())
            parent_ctx->return_deletions(deleted_partitions);
          if (!deleted_fields.empty())
            parent_ctx->return_deletions(deleted_fields);
          return_created_field_contexts(parent_ctx);
        }
        if (parent_ctx != NULL)
          runtime->notify_operation_complete(parent_ctx);
      }

      // Hold the local context lock before triggering the
      // termination event since it could result in us
      // begin deactivated before we're done deactivating children.
      lock();
      if (!remote)
      {
        // Trigger the termination event
        termination_event.trigger();
      }
#ifdef DEBUG_HIGH_LEVEL
      if (is_leaf())
      {
        assert(child_tasks.empty());
        assert(child_maps.empty());
        assert(child_deletions.empty());
      }
#endif
      std::vector<TaskContext*> deactivated_children(child_tasks.begin(),child_tasks.end());
      std::vector<MappingOperation*> deactivated_maps(child_maps.begin(),child_maps.end());
      unlock();
      // Don't want to be holding a lock while deactivating children
      // Deactivate all of our child tasks 
      for (std::vector<TaskContext*>::const_iterator it = deactivated_children.begin();
            it != deactivated_children.end(); it++)
      {
        (*it)->deactivate();
      }
      // Deactivate all of our child inline mapping operations
      // Deletions will take care of themselves
      for (std::vector<MappingOperation*>::const_iterator it = deactivated_maps.begin();
            it != deactivated_maps.end(); it++)
      {
        (*it)->deactivate();
      }

      // If we're remote or the top level task, deactivate ourself
      if (remote || top_level_task)
        this->deactivate();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!locally_mapped); // shouldn't be here if we were locally mapped
#endif
      Deserializer derez(args,arglen); 
      non_virtual_mapped_region.resize(regions.size());
      unmapped = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(map_dependent_waiters.size() == regions.size());
#endif
      lock();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool next = non_virtual_mapped_region[idx];
        derez.deserialize<bool>(next);
        if (non_virtual_mapped_region[idx])
        {
          // hold the lock to prevent others from waiting
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
        }
        else
        {
          unmapped++;
        }
      }
      unlock();
      lock_context();
      forest_ctx->begin_unpack_region_tree_state_return(derez, false/*created only*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
        {
          ContextID phy_ctx = get_enclosing_physical_context(idx);
          forest_ctx->unpack_region_tree_state_return(regions[idx], phy_ctx, IS_WRITE(regions[idx]), 
                                                                  RegionTreeForest::PHYSICAL, derez
#ifdef DEBUG_HIGH_LEVEL
                                                                  , idx, variants->name
                                                                  , this->get_unique_id()
#endif
                                                                  ); 
        }
      }
      forest_ctx->end_unpack_region_tree_state_return(derez, false/*created only*/);
      unlock_context();
      if (unmapped == 0)
      {
        // Everybody mapped, so trigger the mapped event
        mapped_event.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf());
      assert(unmapped > 0);
#endif
      Deserializer derez(args,arglen);
      lock_context();
      forest_ctx->unpack_region_tree_updates_return(derez);
      forest_ctx->begin_unpack_region_tree_state_return(derez, false/*created only*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        ContextID phy_ctx = get_enclosing_physical_context(idx);
        if (!non_virtual_mapped_region[idx])
        {
          forest_ctx->unpack_region_tree_state_return(regions[idx], phy_ctx, IS_WRITE(regions[idx]), 
                                                                  RegionTreeForest::PRIVILEGE, derez
#ifdef DEBUG_HIGH_LEVEL
                                                                  , idx, variants->name
                                                                  , this->get_unique_id()
#endif
                                                                  );
        }
        else
        {
          forest_ctx->unpack_region_tree_state_return(regions[idx], phy_ctx, IS_WRITE(regions[idx]), 
                                                                        RegionTreeForest::DIFF, derez
#ifdef DEBUG_HIGH_LEVEL
                                                                        , idx, variants->name
                                                                        , this->get_unique_id()
#endif
                                                                        );
        }
      }
      forest_ctx->end_unpack_region_tree_state_return(derez, false/*created only*/);
      unpack_source_copy_instances_return(derez,forest_ctx,unique_id);
      // We can also release all the source copy waiters
      release_source_copy_instances();
      unlock_context();
      // Notify all the waiters
      bool needs_trigger = (unmapped > 0);
      lock();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(unmapped > 0);
#endif
          unmapped--;
          std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
          for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                it != waiters.end(); it++)
          {
            (*it)->notify();
          }
          waiters.clear();
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(unmapped == 0);
#endif
      unlock();
      if (needs_trigger)
        mapped_event.trigger();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      if (!is_leaf())
      {
        unpack_privileges_return(derez);
        unpack_deletions_return(derez);
        lock_context();
        forest_ctx->unpack_region_tree_updates_return(derez);
        forest_ctx->begin_unpack_region_tree_state_return(derez, true/*created only*/);
        if (parent_ctx != NULL)
          parent_ctx->unpack_return_created_contexts(derez);
        else
          unpack_return_created_contexts(derez); // only happens if top-level task runs remotely
        forest_ctx->end_unpack_region_tree_state_return(derez, true/*created only*/);
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
        if (parent_ctx != NULL)
        {
          parent_ctx->return_privileges(created_index_spaces, created_field_spaces,
                                        created_regions, created_fields);
          if (!deleted_regions.empty())
            parent_ctx->return_deletions(deleted_regions);
          if (!deleted_partitions.empty())
            parent_ctx->return_deletions(deleted_partitions);
          if (!deleted_fields.empty())
            parent_ctx->return_deletions(deleted_fields);
        }
      }
      else
      {
        lock_context();
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
      }
      // Now set the future result and trigger the termination event
#ifdef DEBUG_HIGH_LEVEL
      assert(this->future != NULL);
#endif
      future->set_result(derez);
      lock();
      termination_event.trigger();
      // We can now remove our reference to the future for garbage collection
      if (future->remove_reference())
      {
        delete future;
      }
      future = NULL;
      if (parent_ctx != NULL)
        runtime->notify_operation_complete(parent_ctx);
      unlock();
    }

    //--------------------------------------------------------------------------
    const void* IndividualTask::get_local_args(void *point, size_t point_size, size_t &local_size)
    //--------------------------------------------------------------------------
    {
      // Should never be called for an individual task
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future(const void *result, size_t result_size, Event ready_event)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        // Save the future locally
#ifdef DEBUG_HIGH_LEVEL
        assert(remote_future == NULL);
        assert(remote_future_len == 0);
#endif
        if (result_size > 0)
        {
          remote_future_len = result_size;
          remote_future = malloc(result_size);
          memcpy(remote_future, result, result_size); 
        }
      }
      else
      {
        // Otherwise we can set the future result and remove our reference
        // which will allow the future to be garbage collected naturally.
        future->set_result(result, result_size);
        if (future->remove_reference())
        {
          delete future;
        }
        future = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::get_future(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(future == NULL); // better be NULL before this
#endif
      future = new FutureImpl(this->termination_event);
      // Reference from this task context
      future->add_reference();
      return Future(future);
    }

    /////////////////////////////////////////////////////////////
    // Point Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointTask::PointTask(HighLevelRuntime *rt, ContextID id)
      : SingleTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PointTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_single(parent);
      if (activated)
      {
        slice_owner = NULL;
        local_point_argument = NULL;
        local_point_argument_len = 0;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      // We never own our local point arguments since
      // they were pulled out of an ArgumentMap so just
      // set them back to their default settings
      local_point_argument = NULL;
      local_point_argument_len = 0;
      deactivate_single();
      runtime->free_point_task(this);
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      return slice_owner->remote;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      // Never partially unpacked
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      bool success = map_all_regions(slice_owner->current_proc, point_termination_event,
                                      slice_owner->get_termination_event());
      if (success && (unmapped == 0))
      {
        slice_owner->point_task_mapped(this);
      }
      return success;
    }

    //--------------------------------------------------------------------------
    bool PointTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    Event PointTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    Event PointTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return point_termination_event;
    }

    //--------------------------------------------------------------------------
    ContextID PointTask::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_enclosing_physical_context(idx);
    }

    //--------------------------------------------------------------------------
    size_t PointTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
      // Here we won't invoke the SingleTask methods for packing since
      // we really only need to pack up our information.
      size_t result = 0;
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point != NULL);
#endif
      result += sizeof(index_element_size);
      result += sizeof(index_dimensions);
      result += (index_element_size*index_dimensions);
      result += sizeof(point_termination_event);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_point_argument != NULL);
#endif
      result += sizeof(local_point_argument_len);
      result += local_point_argument_len;

#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped_region.size() == regions.size());
      assert(physical_instances.size() == regions.size());
#endif
      result += (non_virtual_mapped_region.size() * sizeof(bool));
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          result += forest_ctx->compute_reference_size(physical_instances[idx]);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void PointTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(index_element_size);
      rez.serialize<unsigned>(index_dimensions);
      rez.serialize(index_point,index_element_size*index_dimensions);
      rez.serialize<UserEvent>(point_termination_event);
      rez.serialize<size_t>(local_point_argument_len);
      rez.serialize(local_point_argument,local_point_argument_len);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool non_virt = non_virtual_mapped_region[idx];
        rez.serialize<bool>(non_virt);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          forest_ctx->pack_reference(physical_instances[idx], rez);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<size_t>(index_element_size);
      derez.deserialize<unsigned>(index_dimensions);
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point == NULL);
#endif
      index_point = malloc(index_element_size*index_dimensions);
      derez.deserialize(index_point,index_element_size*index_dimensions);
      derez.deserialize<UserEvent>(point_termination_event);
      derez.deserialize<size_t>(local_point_argument_len);
#ifdef DEBUG_HIGH_LEVEL
      assert(local_point_argument == NULL);
#endif
      local_point_argument = malloc(local_point_argument_len);
      derez.deserialize(local_point_argument,local_point_argument_len);
      non_virtual_mapped_region.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        bool non_virt;
        derez.deserialize<bool>(non_virt);
        non_virtual_mapped_region[idx] = non_virt;
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          physical_instances.push_back(forest_ctx->unpack_reference(derez));
        else
          physical_instances.push_back(InstanceRef()/*virtual ref*/);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InstanceRef PointTask::find_premapped_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // Check to see if the slice owner has a premapped region
      std::map<unsigned,InstanceRef>::const_iterator it = 
                      slice_owner->premapped_regions.find(idx);
      if (it != slice_owner->premapped_regions.end())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!it->second.is_virtual_ref());
#endif
        return it->second;
      }
      return InstanceRef();
    }

    //--------------------------------------------------------------------------
    void PointTask::children_mapped(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_leaf());
#endif

      std::set<Event> cleanup_events;
      // Make sure that all the deletion operations for this task have been performed
      flush_deletions(cleanup_events);

      lock_context();
      // Get the termination events for all of the tasks
      {
        lock(); // need lock to touch child tasks
        for (std::list<TaskContext*>::const_iterator it = child_tasks.begin();
              it != child_tasks.end(); it++)
        {
          cleanup_events.insert((*it)->get_termination_event());
        }
        unlock();
      }
      // Issue the restoring copies for this task
      issue_restoring_copies(cleanup_events, point_termination_event, slice_owner->termination_event);
      unlock_context();
      // If it hadn't been mapped already, notify it that it is now
      if (unmapped > 0)
      {
        slice_owner->point_task_mapped(this);
        unmapped = 0;
      }

      // Now can invalidate any valid physical instances our context for
      // all the regions which we had as region requirements.  We can't invalidate
      // virtual mapped regions since that information has escaped our context.
      // Note we don't need to worry about doing this for leaf tasks since there
      // are no mappings performed in a leaf task context.  We also don't have to
      // do this for any created regions since either they are explicitly deleted
      // or their state is passed back in finish_task to the enclosing context.
      invalidate_owned_contexts();

      // Now figure out whether we need to wait to launch the finish task
      Event wait_on_event = Event::merge_events(cleanup_events);
      if (!wait_on_event.exists())
      {
        finish_task();
      }
      else
      {
        size_t buffer_size = sizeof(Processor)+sizeof(Context);
        Serializer rez(buffer_size);
        rez.serialize<Processor>(runtime->utility_proc);
        rez.serialize<Context>(this);
        // Launch the task on the utility processor
        Processor utility = runtime->utility_proc;
        utility.spawn(FINISH_ID,rez.get_buffer(),buffer_size,wait_on_event);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::finish_task(void)
    //--------------------------------------------------------------------------
    {
      // Return privileges to the slice owner
      slice_owner->return_privileges(created_index_spaces,created_field_spaces,
                                      created_regions,created_fields);
      slice_owner->return_deletions(deleted_regions, deleted_partitions, deleted_fields);
      // If not remote, remove our physical instance usages
      if (!slice_owner->remote)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(physical_instances.size() == regions.size());
#endif
        lock_context();
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
        {
          if (!physical_instances[idx].is_virtual_ref())
          {
            physical_instances[idx].remove_reference(unique_id, false/*strict*/);
          }
        }
        for (unsigned idx = 0; idx < close_copy_instances.size(); idx++)
        {
          close_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
        }
        unlock_context();
        physical_instances.clear();
        close_copy_instances.clear();
      }
      // Indicate that this point has terminated
      point_termination_event.trigger();
      // notify the slice owner that this task has finished
      slice_owner->point_task_finished(this);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    const void* PointTask::get_local_args(void *point, size_t point_size, size_t &local_size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(point_size == (index_element_size*index_dimensions));
#endif
      // Copy the point value into the point size
      memcpy(point, index_point, point_size);
      // Set the local size and return the pointer to the local size argument
      local_size = local_point_argument_len;
      return local_point_argument;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future(const void *result, size_t result_size, Event ready_event)
    //--------------------------------------------------------------------------
    {
      AnyPoint local_point(index_point,index_element_size,index_dimensions);
      slice_owner->handle_future(local_point,result, result_size, ready_event); 
    }

    //--------------------------------------------------------------------------
    void PointTask::unmap_all_regions(void)
    //--------------------------------------------------------------------------
    {
      // Go through all our regions and if they were mapped, release the reference
#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped_region.size() == physical_instances.size());
#endif
      // Move any non-virtual mapped references to the source copy references.
      // We can't just remove the references because they might cause the instance
      // to get deleted before the copy completes.
      // TODO: how do we handle the pending copy events for these copies?
      // When do we know that is safe to remove the references because our
      // task will no longer depend on the event for when the copy is done.
      for (unsigned idx = 0; idx < physical_instances.size(); idx++)
      {
        if (non_virtual_mapped_region[idx])
          source_copy_instances.push_back(physical_instances[idx]);
      }
      physical_instances.clear();
      non_virtual_mapped_region.clear();
    }

    //--------------------------------------------------------------------------
    void PointTask::update_requirements(const std::vector<RegionRequirement> &reqs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point != NULL);
      assert(regions.empty());
#endif
      // go through the region requirements for the slice task and compute
      // the ones for this individual point
      for (std::vector<RegionRequirement>::const_iterator it = reqs.begin();
            it != reqs.end(); it++)
      {
        if (it->handle_type == SINGULAR)
        {
          // Singlular so we can just copy it directly
          regions.push_back(*it);
        }
        else
        {
          Color subregion_color;
          // We need to compute the projected function based on this point
          if (it->projection == 0)
          {
            // Handle the built-in case of isomorphic translation for 1-D points-only
            if ((index_element_size != sizeof(Color)) ||
                (index_dimensions != 1))
            {
              log_task(LEVEL_ERROR,"Projection ID 0 is invalid for tasks whose points"
                                  " are not one dimensional unsigned integers.  Points for "
                                  "task %s have elements of %ld bytes and %d dimensions",
                                  this->variants->name, index_element_size, index_dimensions);
              exit(ERROR_INVALID_IDENTITY_PROJECTION_USE);
            }
            subregion_color = *((Color*)index_point);  
          }
          else
          {
            ProjectionFnptr projfn = HighLevelRuntime::find_projection_function(it->projection);
            // Compute the color for the subregion that we want
            subregion_color = (*projfn)(index_point, index_element_size, index_dimensions);
          }
          lock_context();
          LogicalRegion region_handle = forest_ctx->get_partition_subcolor(it->partition, subregion_color);
          LogicalRegion parent_handle = forest_ctx->get_partition_parent(it->partition);
          unlock_context();
          // We use different constructors for reductions
          if (it->redop == 0)
          {
            regions.push_back(RegionRequirement(region_handle, it->privilege_fields,
                      it->instance_fields, it->privilege, it->prop, parent_handle,
                      it->tag, true/*verified*/, it->inst_type));
          }
          else
          {
            regions.push_back(RegionRequirement(region_handle, it->privilege_fields,
                      it->instance_fields, it->redop, it->prop, parent_handle,
                      it->tag, true/*verified*/, it->inst_type));
          }
        }
      }
#ifdef LEGION_SPY
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        LegionSpy::log_task_instance_requirement(this->get_unique_id(), this->ctx_id, this->get_gen(), runtime->utility_proc.id, idx, regions[idx].region.get_index_space().id); 
      }
#endif
    }

    //--------------------------------------------------------------------------
    void PointTask::update_argument(const ArgumentMapImpl *impl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(index_point != NULL);
      assert(impl != NULL);
#endif
      AnyPoint point(index_point, index_element_size, index_dimensions); 
      TaskArgument local_arg = impl->find_point(point);
      // Note we don't need to clone the data since we know the ArgumentMapImpl
      // owns it and the ArgumentMapImpl won't be reclaimed until after the
      // task is done at the earliest
      this->local_point_argument = local_arg.get_ptr();
      this->local_point_argument_len = local_arg.get_size();
    }

    /////////////////////////////////////////////////////////////
    // Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(HighLevelRuntime *rt, ContextID id)
      : MultiTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_multi(parent);
      if (activated)
      {
        locally_set = false;
        locally_mapped = false;
        frac_index_space = std::pair<unsigned long,unsigned long>(0,1);
        num_total_points = 0;
        num_finished_points = 0;
        unmapped = 0;
        future_map = NULL;
        reduction_future = NULL;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      lock();
      mapped_points.clear();
      if (future_map != NULL)
      {
        if (future_map->remove_reference())
        {
          delete future_map;
        }
        future_map = NULL;
      }
      if (reduction_future != NULL)
      {
        if (reduction_future->remove_reference())
        {
          delete reduction_future;
        }
        reduction_future = NULL;
      }
      source_copy_instances.clear();
#ifdef DEBUG_HIGH_LEVEL
      slice_overlap.clear();
#endif
      unlock();
      deactivate_multi();
      runtime->free_index_task(this);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      lock();
      if (task_pred == Predicate::TRUE_PRED)
      {
        // Task evaluated should be run, put it on the ready queue
        unlock();
        runtime->add_to_ready_queue(orig_proc, this);
      }
      else if (task_pred == Predicate::FALSE_PRED)
      {
        unlock();
      }
      else
      {
        // TODO: handle predication
        assert(false); 
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(gen <= generation);
#endif
      bool result;
      do {
        if (gen < generation) // already been recycled
        {
          result = false;
          break;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(idx < map_dependent_waiters.size());
#endif
        // Check to see if it has been mapped by everybody and we've seen the
        // whole index space 
        if ((frac_index_space.first == frac_index_space.second) &&
            (mapped_points[idx] == num_total_points))
        {
          // Already been mapped by everyone
          result = false;
        }
        else
        {
          std::pair<std::set<GeneralizedOperation*>::iterator,bool> added = 
            map_dependent_waiters[idx].insert(waiter);
          result = added.second;
        }
      } while (false);
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      // IndexTasks are already where they are supposed to be
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      if (!locally_set)
      {
        locally_mapped = invoke_mapper_locally_mapped();
        locally_set = true;
      }
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask are not stealable, only their slices are
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      // IndexTasks are never remote
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      // Never partially unpacked
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // This will only get called if we had slices that couldn't map, but
      // they have now all mapped
#ifdef DEBUG_HIGH_LEVEL
      assert(slices.empty());
#endif
      // We're never actually here
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(void)
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
        bool slice_success = (*it)->perform_operation();
        if (!slice_success)
        {
          map_success = false;
          it++;
        }
        else
        {
          // Remove it from the list since we're done
          it = slices.erase(it);
        }
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be stealable
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Check to see if this region has been sanitized or has been premapped
        if (regions[idx].sanitized || (premapped_regions.find(idx) != premapped_regions.end()))
          continue;
        ContextID phy_ctx = get_enclosing_physical_context(idx); 
        // Create a sanitizing region mapper and map it
        RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock,
                                parent_ctx->get_executing_processor(), termination_event, termination_event,
                                tag, true/*sanitizing*/, false/*inline mapping*/,
                                source_copy_instances);
        if (regions[idx].handle_type == SINGULAR)
        {
#ifdef DEBUG_HIGH_LEVEL
          bool result = 
#endif
          forest_ctx->compute_index_path(regions[idx].parent.index_space,regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
          assert(result);
#endif
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          bool result = 
#endif
          forest_ctx->compute_partition_path(regions[idx].parent.index_space,regions[idx].partition.index_partition, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
          assert(result);
#endif
        }
        // Now do the sanitizing walk
        forest_ctx->map_region(reg_mapper, regions[idx].parent);
        if (reg_mapper.success)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(reg_mapper.path.empty());
#endif
          regions[idx].sanitized = true;
        }
        else
        {
          result = false;
          break;
        }
      }
      unlock_context();
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      mapped_event = UserEvent::create_user_event();
      termination_event = UserEvent::create_user_event();
#ifdef LEGION_SPY
      LegionSpy::log_index_task_termination(get_unique_id(), termination_event);
#endif
      if (must_parallelism)
      {
        must_barrier = Barrier::create_barrier(1/*expected arrivals*/);
        launch_preconditions.insert(must_barrier);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // IndexTask should never be launched
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    Event IndexTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    Event IndexTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------
    ContextID IndexTask::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_enclosing_physical_context(regions[idx].parent); 
    }

    //--------------------------------------------------------------------------
    size_t IndexTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void IndexTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      unsigned long denominator;
      derez.deserialize<unsigned long>(denominator);
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      std::vector<unsigned> non_virtual_mappings(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        derez.deserialize<unsigned>(non_virtual_mappings[idx]);
      }
      lock_context();
      // Unpack any trees that were sent back because they were fully mapped
      forest_ctx->begin_unpack_region_tree_state_return(derez, false/*created only*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (non_virtual_mappings[idx] == num_points)
        {
          ContextID phy_ctx = get_enclosing_physical_context(idx);
          unpack_tree_state_return(idx, phy_ctx, RegionTreeForest::PHYSICAL, derez);
        }
      }
      forest_ctx->end_unpack_region_tree_state_return(derez, false/*created only*/);
      unlock_context();
      slice_start(denominator, num_points, non_virtual_mappings); 
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      std::vector<unsigned> virtual_mappings(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        derez.deserialize<unsigned>(virtual_mappings[idx]);
      }
      lock_context();
      forest_ctx->unpack_region_tree_updates_return(derez);
      forest_ctx->begin_unpack_region_tree_state_return(derez, false/*created only*/);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        ContextID phy_ctx = get_enclosing_physical_context(idx);
        if (virtual_mappings[idx] > 0)
        {
          unpack_tree_state_return(idx, phy_ctx, RegionTreeForest::PRIVILEGE, derez);
        }
        else
        {
          unpack_tree_state_return(idx, phy_ctx, RegionTreeForest::DIFF, derez);
        }
      }
      forest_ctx->end_unpack_region_tree_state_return(derez, false/*created only*/);
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        SingleTask::unpack_source_copy_instances_return(derez,forest_ctx,unique_id);
      } 
      unlock_context();
      slice_mapped(virtual_mappings);
    }

    //--------------------------------------------------------------------------
    void IndexTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      if (!is_leaf())
      {
        unpack_privileges_return(derez);
        unpack_deletions_return(derez);
        lock_context();
        forest_ctx->unpack_region_tree_updates_return(derez);
        forest_ctx->begin_unpack_region_tree_state_return(derez, true/*created only*/);
        for (unsigned idx = 0; idx < num_points; idx++)
        {
          parent_ctx->unpack_return_created_contexts(derez);
        }
        forest_ctx->end_unpack_region_tree_state_return(derez, true/*created only*/);
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
      }
      else
      {
        lock_context();
        forest_ctx->unpack_leaked_return(derez);
        unlock_context();
      }
      
      // Unpack the future(s)
      if (has_reduction)
      {
        const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
        // Create a fake AnyPoint 
        AnyPoint no_point(NULL,0,0);
        const void *ptr = derez.get_pointer();
        derez.advance_pointer(redop->sizeof_rhs);
        handle_future(no_point, const_cast<void*>(ptr), redop->sizeof_rhs, 
                        get_termination_event()/*won't matter*/);
      }
      else
      {
        size_t result_size;
        derez.deserialize<size_t>(result_size);
        derez.deserialize<size_t>(this->index_element_size);
        derez.deserialize<unsigned>(this->index_dimensions);
        size_t point_size = index_element_size * index_dimensions;
        for (unsigned idx = 0; idx < num_points; idx++)
        {
          AnyPoint point(const_cast<void*>(derez.get_pointer()),index_element_size,index_dimensions); 
          derez.advance_pointer(point_size);
          const void *ptr = derez.get_pointer();
          derez.advance_pointer(result_size);
          Event ready_event;
          derez.deserialize(ready_event);
          handle_future(point, ptr, result_size, ready_event);
        }
      }
      
      slice_finished(num_points);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pre_slice(void)
    //--------------------------------------------------------------------------
    {
      // Go through and see if we have any regions that need to be pre-mapped
      // prior to slicing up this index space
      bool success = true;
      lock_context();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if ((regions[idx].handle_type == SINGULAR) &&
            (IS_WRITE(regions[idx])))
        {
          // See if we already pre-mapped this region
          if (premapped_regions.find(idx) != premapped_regions.end())
            continue;
#ifdef DEBUG_HIGH_LEVEL
          assert(!regions[idx].sanitized);
          assert(parent_ctx != NULL);
#endif
          // If it is a singular write, then the index space must premap the region
          ContextID phy_ctx = parent_ctx->find_enclosing_physical_context(regions[idx].parent);
          RegionMapper reg_mapper(this, unique_id, phy_ctx, idx, regions[idx], mapper, mapper_lock,
              Processor::NO_PROC, termination_event, termination_event, tag, false/*sanitizing*/,
              false/*inline mapping*/, source_copy_instances);
#ifdef DEBUG_HIGH_LEVEL
          bool result = 
#endif
          forest_ctx->compute_index_path(regions[idx].parent.index_space, regions[idx].region.index_space, reg_mapper.path);
#ifdef DEBUG_HIGH_LEVEL
          assert(result);
#endif
          forest_ctx->map_region(reg_mapper, regions[idx].parent);
          if (reg_mapper.result.is_virtual_ref())
          {
            // Failed the mapping
            success = false;
            break;
          }
          // Otherwise mapping was successful, save it and continue
          premapped_regions[idx] = reg_mapper.result;
        }
      }
      unlock_context();
      return success;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::post_slice(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      // Update all the slices with their new denominator
      for (std::list<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(slices.size(), slices.size());
      }
      // No need to reclaim this since it is referenced by the calling context
      return false;
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(IndexSpace new_space, Processor target,
                                              bool recurse, bool steal)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(parent_ctx);
      result->clone_multi_from(this,new_space,recurse); 
      result->distributed = false;
      result->locally_mapped = is_locally_mapped();
      result->stealable = steal;
      result->remote = false;
      result->termination_event = this->termination_event;
      result->current_proc = target;
      result->orig_proc = this->orig_proc;
      result->index_owner = this;
      // denominator gets set by post_slice
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_future(const AnyPoint &point, const void *result, size_t result_size, Event ready_event)
    //--------------------------------------------------------------------------
    {
      if (has_reduction)
      {
        const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id); 
#ifdef DEBUG_HIGH_LEVEL
        assert(reduction_state != NULL);
        assert(reduction_state_size == redop->sizeof_lhs);
        assert(result_size == redop->sizeof_rhs);
#endif
        lock();
        redop->apply(reduction_state, result, 1/*num elements*/);
        unlock();
      }
      else
      {
        // Put it in the future map
#ifdef DEBUG_HIGH_LEVEL
        assert(future_map != NULL);
#endif
        // No need to hold the lock, the future map has its own lock
        future_map->set_result(point, result, result_size, ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::set_index_space(IndexSpace space, const ArgumentMap &map, size_t num_regions, bool must)
    //--------------------------------------------------------------------------
    {
      this->is_index_space = true;
      this->index_space = space;
      this->must_parallelism = must;
#ifdef DEBUG_HIGH_LEVEL
      assert(arg_map_impl == NULL);
#endif
      // Freeze the current impl so we can use it
      arg_map_impl = map.impl->freeze();
      arg_map_impl->add_reference();
      mapped_points.resize(num_regions);
      for (unsigned idx = 0; idx < num_regions; idx++)
        mapped_points[idx] = 0;
    }

    //--------------------------------------------------------------------------
    void IndexTask::set_reduction_args(ReductionOpID id, const TaskArgument &initial_value)
    //--------------------------------------------------------------------------
    {
      has_reduction = true; 
      redop_id = id;
      const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
#ifdef DEBUG_HIGH_LEVEL
      if (initial_value.get_size() != redop->sizeof_lhs)
      {
        log_task(LEVEL_ERROR,"Initial value for reduction for task %s (ID %d) is %ld bytes "
                              "but ReductionOpID %d requires left-hand size arguments of %ld bytes",
                              this->variants->name, get_unique_id(), initial_value.get_size(),
                              redop_id, redop->sizeof_lhs);
        exit(ERROR_REDUCTION_INITIAL_VALUE_MISMATCH);
      }
      assert(reduction_state == NULL); // this better be NULL
#endif
      reduction_state_size = redop->sizeof_lhs;
      reduction_state = malloc(reduction_state_size);
      memcpy(reduction_state,initial_value.get_ptr(),initial_value.get_size());
    }

    //--------------------------------------------------------------------------
    Future IndexTask::get_future(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reduction_future == NULL); // better be NULL before this
#endif
      reduction_future = new FutureImpl(termination_event);
      // Add a reference so it doesn't get deleted
      reduction_future->add_reference(); 
      return Future(reduction_future);
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::get_future_map(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(future_map == NULL); // better be NULL before this
#endif
      future_map = new FutureMapImpl(termination_event);
      // Add a reference so it doesn't get deleted
      future_map->add_reference();
      return FutureMap(future_map);
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_start(unsigned long denominator, size_t points,
                                const std::vector<unsigned> &non_virtual_mapped)
    //--------------------------------------------------------------------------
    {
      lock_context();
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(points > 0);
#endif
      num_total_points += points;
#ifdef DEBUG_HIGH_LEVEL
      assert(non_virtual_mapped.size() == mapped_points.size());
#endif
      for (unsigned idx = 0; idx < mapped_points.size(); idx++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(non_virtual_mapped[idx] <= points);
#endif
        mapped_points[idx] += non_virtual_mapped[idx];
      }
      // Now update the fraction of the index space that we've seen
      // Check to see if the denominators are the same
      if (frac_index_space.second == denominator)
      {
        // Easy add one to our numerator
        frac_index_space.first++;
      }
      else
      {
        // Denominators are different, make them the same
        // Check if one denominator is divisible by another
        if ((frac_index_space.second % denominator) == 0)
        {
          frac_index_space.first += (frac_index_space.second / denominator);
        }
        else if ((denominator % frac_index_space.second) == 0)
        {
          frac_index_space.first = (frac_index_space.first * (denominator / frac_index_space.second)) + 1;
          frac_index_space.second = denominator;
        }
        else
        {
          // One denominator is not divisilbe by the other, compute a common denominator
          unsigned new_denom = frac_index_space.second * denominator;
          unsigned other_num = frac_index_space.second; // *1
          unsigned local_num = frac_index_space.first * denominator;
          frac_index_space.first = local_num + other_num;
          frac_index_space.second = new_denom;
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(frac_index_space.first <= frac_index_space.second); // should be a fraction <= 1
#endif
      // Check to see if this index space has been fully enumerated
      if (frac_index_space.first == frac_index_space.second)
      {
        // If we've fully enumerated, let's see if we've mapped regions for all the points
        unmapped = 0;
#ifdef DEBUG_HIGH_LEVEL
        assert(mapped_points.size() == regions.size());
#endif
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (mapped_points[idx] < num_total_points)
          {
            // Not all points in the index space mapped the region so it is unmapped
            unmapped++;
          }
          else
          {
            // It's been mapped so notify all it's waiting dependences
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
        // Check to see if we're fully mapped, if so trigger the mapped event
        if (unmapped == 0)
        {
          mapped_event.trigger();
        }
      }
      unlock(); 
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_mapped(const std::vector<unsigned> &virtual_mapped)
    //--------------------------------------------------------------------------
    {
      lock_context();
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(virtual_mapped.size() == mapped_points.size());
#endif
      unsigned newly_mapped = 0;
      for (unsigned idx = 0; idx < virtual_mapped.size(); idx++)
      {
        if (virtual_mapped[idx] > 0)
        {
          mapped_points[idx] += virtual_mapped[idx];
#ifdef DEBUG_HIGH_LEVEL
          assert(mapped_points[idx] <= num_total_points);
#endif
          // Check to see if we should notify all the waiters, points have to
          // equal and the index space must be fully enumerated
          if ((mapped_points[idx] == num_total_points) &&
              (frac_index_space.first == frac_index_space.second))
          {
            newly_mapped++;
            std::set<GeneralizedOperation*> &waiters = map_dependent_waiters[idx];
            for (std::set<GeneralizedOperation*>::const_iterator it = waiters.begin();
                  it != waiters.end(); it++)
            {
              (*it)->notify();
            }
            waiters.clear();
          }
        }
      }
      // Update the number of unmapped regions and trigger the mapped_event if we're done
      if (newly_mapped > 0)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(newly_mapped <= unmapped);
#endif
        unmapped -= newly_mapped;
        if (unmapped == 0)
        {
          mapped_event.trigger();
        }
      }
      // Once a slice comes back then we know that all the sanitization preconditions
      // were met and so we can release all the source copy instances
      if (!source_copy_instances.empty())
      {
        for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
        {
          source_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
        }
        source_copy_instances.clear();
      }
      unlock();
      unlock_context();
    }

    //--------------------------------------------------------------------------
    void IndexTask::slice_finished(size_t points)
    //--------------------------------------------------------------------------
    {
      // Need to take locks in order
      lock_context();
      // Hold the lock when testing the num_finished_points
      // and frac_index_space
      lock();
      // Remove any outstanding references for copies that we have
      if (!source_copy_instances.empty())
      {
        for (unsigned idx = 0; idx < source_copy_instances.size(); idx++)
        {
          source_copy_instances[idx].remove_reference(unique_id, false/*strict*/);
        }
        source_copy_instances.clear();
      }
      unlock_context();
      num_finished_points += points;
#ifdef DEBUG_HIGH_LEVEL
      assert(num_finished_points <= num_total_points);
#endif
      // Check to see if we've seen all our points and if
      // the index space has been fully enumerated
      if ((num_finished_points == num_total_points) &&
          (frac_index_space.first == frac_index_space.second))
      {
        unlock();
        // Handle the future or future map
        if (has_reduction)
        {
          // Set the future 
#ifdef DEBUG_HIGH_LEVEL
          assert(reduction_future != NULL);
#endif
          reduction_future->set_result(reduction_state,reduction_state_size);
        }
        // Otherwise we have a reduction map and we're already set everything

#ifdef DEBUG_HIGH_LEVEL
        assert(parent_ctx != NULL);
#endif
        parent_ctx->return_privileges(created_index_spaces,created_field_spaces,
                                      created_regions,created_fields);
        if (!deleted_regions.empty())
          parent_ctx->return_deletions(deleted_regions);
        if (!deleted_partitions.empty())
          parent_ctx->return_deletions(deleted_partitions);
        if (!deleted_fields.empty())
          parent_ctx->return_deletions(deleted_fields);
        // Reclaim the lock now that we're done with any 
        // calls that may end up taking the context lock
        lock();
        // We're done, trigger the termination event
        termination_event.trigger();
        runtime->notify_operation_complete(parent_ctx);
        // Remove our reference since we're done
        if (has_reduction)
        {
          if (reduction_future->remove_reference())
          {
            delete reduction_future;
          }
          reduction_future = NULL;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(future_map != NULL);
#endif
          if (future_map->remove_reference())
          {
            delete future_map;
          }
          future_map = NULL;
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(future_map == NULL);
        assert(reduction_future == NULL);
#endif
      }
      unlock();
      // No need to deactivate our slices since they will deactivate themsevles
      // We also don't need to deactivate ourself since our enclosing parent
      // task will take care of that
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void IndexTask::check_overlapping_slices(unsigned idx, const std::set<LogicalRegion> &touched_regions)
    //--------------------------------------------------------------------------
    {
      lock();
      if (slice_overlap.find(idx) == slice_overlap.end())
      {
        slice_overlap[idx] = touched_regions;
      }
      else
      {
        std::set<LogicalRegion> &already_touched = slice_overlap[idx];
        std::vector<LogicalRegion> overlap;
        std::vector<LogicalRegion>::iterator end_it;
        end_it = std::set_intersection(already_touched.begin(), already_touched.end(),
                                       touched_regions.begin(), touched_regions.end(),
                                       overlap.begin());
        if (end_it != overlap.begin())
        {
          log_task(LEVEL_ERROR,"Violation of the independent slices rule for projection region requirement %d "
                                " of task %s.  The following regions where used in multiple slices:",
                                idx, this->variants->name);
          for (std::vector<LogicalRegion>::iterator it = overlap.begin();
                it != end_it; it++)
          {
            log_task(LEVEL_ERROR,"Logical Region (%x,%x,%x)",
                      it->tree_id, it->index_space.id, it->field_space.id); 
          }
          exit(ERROR_INDEPENDENT_SLICES_VIOLATION);
        }
        already_touched.insert(touched_regions.begin(), touched_regions.end());
      }
      unlock();
    }
#endif
    
    //--------------------------------------------------------------------------
    void IndexTask::unpack_tree_state_return(unsigned idx, ContextID ctx, 
              RegionTreeForest::SendingMode mode, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // If mode is a DIFF, make sure that the difference isn't empty
      if (mode == RegionTreeForest::DIFF)
      {
        std::set<FieldID> packing_fields;
        for (std::vector<FieldID>::const_iterator it = regions[idx].instance_fields.begin();
              it != regions[idx].instance_fields.end(); it++)
        {
          packing_fields.erase(*it);
        }
        if (packing_fields.empty())
          return;
      }
      if (regions[idx].handle_type == SINGULAR)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!IS_WRITE(regions[idx])); // If this was the case it should have been premapped
#endif
        forest_ctx->unpack_region_tree_state_return(regions[idx], ctx, false/*overwrite*/, mode, derez
#ifdef DEBUG_HIGH_LEVEL
            , idx, variants->name, this->get_unique_id()
#endif
            );
      }
      else
      {
        if (IS_WRITE(regions[idx]))
        {
          size_t num_regions;
          derez.deserialize(num_regions);
          std::set<LogicalRegion> touched_regions;
          // Pretend this is a single region coming back
          regions[idx].handle_type = SINGULAR;
          for (unsigned cnt = 0; cnt < num_regions; cnt++)
          {
            LogicalRegion touched;
            derez.deserialize(touched);
            touched_regions.insert(touched);
            regions[idx].region = touched;
            forest_ctx->unpack_region_tree_state_return(regions[idx], ctx, true/*overwrite*/,
                                                        mode, derez
#ifdef DEBUG_HIGH_LEVEL
                                                        , idx, variants->name
                                                        , this->get_unique_id()
#endif
                                                        );
          }
          // Set the handle type back
          regions[idx].handle_type = PROJECTION;
          // If this is not a DIFF, then check to make sure we didn't overlap
          // slices at any point.  If it is a DIFF then we already did this check
          // when having the PHYSICAL parts come back
#ifdef DEBUG_HIGH_LEVEL
          if (mode != RegionTreeForest::DIFF)
          {
            check_overlapping_slices(idx, touched_regions);  
          }
#endif
        }
        else
        {
          forest_ctx->unpack_region_tree_state_return(regions[idx], ctx, false/*overwrite*/,
                                                      mode, derez
#ifdef DEBUG_HIGH_LEVEL
                                                      , idx, variants->name
                                                      , this->get_unique_id()
#endif
                                                      );
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Slice Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(HighLevelRuntime *rt, ContextID id)
      : MultiTask(rt,id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool SliceTask::activate(GeneralizedOperation *parent /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      bool activated = activate_multi(parent);
      if (activated)
      {
        distributed = false;
        locally_mapped = false;
        stealable = false;
        remote = false;
        termination_event = Event::NO_EVENT;
        current_proc = Processor::NO_PROC;
        index_owner = NULL;
        remote_start_event = Event::NO_EVENT;
        remote_mapped_event = Event::NO_EVENT;
        partially_unpacked = false;
        remaining_buffer = NULL;
        remaining_bytes = 0;
        denominator = 1;
        split_factor = 1;
        enumerating = false;
        enumerator = NULL;
        remaining_enumerated = 0;
        num_unmapped_points = 0;
        num_unfinished_points = 0;
      }
      return activated;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      points.clear();
      future_results.clear();
      non_virtual_mappings.clear();
      if (remaining_buffer != NULL)
      {
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
      }
      if (enumerator != NULL)
      {
        delete enumerator;
        enumerator = NULL;
      }
      deactivate_multi();
      runtime->free_slice_task(this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::add_waiting_dependence(GeneralizedOperation *waiter, unsigned idx, GenerationID gen)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_distributed(void)
    //--------------------------------------------------------------------------
    {
      return distributed; 
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_locally_mapped(void)
    //--------------------------------------------------------------------------
    {
      return locally_mapped;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void)
    //--------------------------------------------------------------------------
    {
      return stealable; 
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_remote(void)
    //--------------------------------------------------------------------------
    {
      return remote;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_partially_unpacked(void)
    //--------------------------------------------------------------------------
    {
      return partially_unpacked;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!distributed);
#endif
      distributed = true;
      // Check to see if the target processor is the local one
      if (!runtime->is_local_processor(current_proc))
      {
#ifdef DEBUG_HIGH_LEVEL
        bool is_local = 
#endif
        runtime->send_task(current_proc,this);
#ifdef DEBUG_HIGH_LEVEL
        assert(!is_local);
#endif
        // Now we can deactivate this task since it's been distributed
        this->deactivate();
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::perform_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      bool map_success = true;
      // No longer stealable since we're being mapped
      stealable = false;
      // This is a leaf slice so do the normal thing
      if (slices.empty())
      {
        // only need to do this part if we didn't enumnerate before
        if (points.empty())
        {
          lock();
          enumerating = true;
          LowLevel::ElementMask::Enumerator *enumerator = 
                      index_space.get_valid_mask().enumerate_enabled();
          int value, length;
          while (enumerator->get_next(value,length))
          {
            for (int idx = 0; idx < length; idx++)
            {
              PointTask *next_point = clone_as_point_task(true/*new point*/);
              next_point->set_index_point(&value, sizeof(int), 1);
              // Update the region requirements for this point
              next_point->update_requirements(regions);
              next_point->update_argument(arg_map_impl);
              points.push_back(next_point); 
              value++;
            }
          }
          delete enumerator;

          num_unmapped_points = points.size();
          num_unfinished_points = points.size();
          enumerating = false;
          unlock();
        }
        
        for (unsigned idx = 0; idx < points.size(); idx++)
        {
          bool point_success = points[idx]->perform_mapping();
          if (!point_success)
          {
            // Unmap all the points up to this point 
            for (unsigned i = 0; i < idx; i++)
              points[i]->unmap_all_regions();
            map_success = false;
            break;
          }
        }

        // No need to hold the lock here since none of
        // the point tasks have begun running yet
        if (map_success)
        {
          // If we're doing must parallelism, register
          // that all our tasks have been mapped and will 
          // be scheduled on their target processor.
          if (must_parallelism)
          {
            must_barrier.arrive();
          }
          post_slice_start();
        }
      }
      else
      {
        // This case only occurs if this is an intermediate slice
        // and its subslices failed to map, so try to remap them
        for (std::list<SliceTask*>::iterator it = slices.begin();
              it != slices.end(); /*nothing*/)
        {
          bool slice_success = (*it)->perform_operation();
          if (!slice_success)
          {
            map_success = false;
            it++;
          }
          else
          {
            // Remove it from the list since we're done
            it = slices.erase(it);
          }
        }
        // If we mapped all our sub-slices, we're done
        if (map_success)
          this->deactivate();
      }

      return map_success;
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        points[idx]->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    bool SliceTask::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_locally_mapped());
#endif
      // No need to do anything here since the region trees were sanitized
      // prior to slicing the index space task
      steal_count++;
      return true;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::sanitize_region_forest(void)
    //--------------------------------------------------------------------------
    {
      // Do nothing.  Region trees for slices were already sanitized by their IndexTask
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::initialize_subtype_fields(void)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      if (is_partially_unpacked())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remote);
#endif
        finish_task_unpack();
      }
      lock();
      stealable = false; // no longer stealable
      enumerating = true;
      num_unmapped_points = 0;
      num_unfinished_points = 0;

      bool map_success = true;
      if (enumerator == NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(remaining_enumerated == 0);
#endif
        enumerator = index_space.get_valid_mask().enumerate_enabled();
      }
      do
      {
        // Handle mapping any previous points
        while (remaining_enumerated > 0)
        {
          unlock();
#ifdef DEBUG_HIGH_LEVEL
          assert(int(points.size()) >= remaining_enumerated);
#endif
          PointTask *next_point = points[points.size()-remaining_enumerated];
          bool point_success = next_point->perform_mapping();     
          if (!point_success)
          {
            map_success = false;
            lock();
            break;
          }
          else
          {
            next_point->launch_task(); 
          }
          lock();
          remaining_enumerated--;
        }
        // If we didn't succeed in mapping all the points, break out
        if (!map_success)
          break;
        int value;
        // Make new points for everything, we'll map them the next
        // time around the loop
        if (enumerator->get_next(value, remaining_enumerated))
        {
          // Make points for all of them 
          for (int idx = 0; idx < remaining_enumerated; idx++)
          {
            PointTask *next_point = clone_as_point_task(true/*new point*/);
            next_point->set_index_point(&value, sizeof(int), 1);
            // This must be called after the point is set
            next_point->update_requirements(regions);
            next_point->update_argument(arg_map_impl);
            points.push_back(next_point);
            num_unmapped_points++;
            num_unfinished_points++;
            value++;
          }
        }
        else
        {
          // Our enumerator is done
          break;
        }
      }
      while (remaining_enumerated > 0);
      unlock();
      // No need to hold the lock when doing the post-slice-start since
      // we know that all the points have been enumerated at this point
      if (map_success)
      {
        // Can clean up our enumerator
        delete enumerator;
        enumerator = NULL;
        // Call post slice start
        post_slice_start(); 
      }
      lock();
      // Handle the case where all the children have called point_task_mapped
      // before we made it here.  The fine-grained locking here is necessary
      // to allow many children to run while others are still being instantiated
      // and mapped.
      bool all_mapped = (num_unmapped_points==0);
      bool all_finished = (num_unfinished_points == 0);
      if (map_success)
        enumerating = false; // need this here to make sure post_slice_mapped gets called after post_slice_start
      unlock();
      // If we need to do the post-mapped part, do that now too
      if (map_success && all_mapped)
      {
        post_slice_mapped();
      }
      if (map_success && all_finished)
      {
        post_slice_finished();
      }
      return map_success;
    }

    //--------------------------------------------------------------------------
    Event SliceTask::get_map_event(void) const
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------
    Event SliceTask::get_termination_event(void) const
    //--------------------------------------------------------------------------
    {
      return termination_event;
    }

    //--------------------------------------------------------------------------
    ContextID SliceTask::get_enclosing_physical_context(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (remote)
        return ctx_id;
      else
        return index_owner->get_enclosing_physical_context(idx);
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::compute_task_size(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      size_t result = compute_multi_task_size();
      result += sizeof(distributed);
      result += sizeof(locally_mapped);
      result += sizeof(stealable);
      result += sizeof(termination_event);
      result += sizeof(index_owner);
      result += sizeof(denominator);
      result += sizeof(current_proc);
      if (partially_unpacked)
      {
        result += remaining_bytes;
      }
      else
      {
        if (locally_mapped)
        {
          result += sizeof(size_t); // number of points
          if (!is_leaf())
          {
            // Need to pack the region trees and the instances or
            // the state if they were virtually mapped
            result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
#ifdef DEBUG_HIGH_LEVEL
            assert(regions.size() == non_virtual_mappings.size());
#endif
            result += (regions.size() * sizeof(unsigned)); // number of non-virtual mappings
            // Figure out which region states we need to send
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(non_virtual_mappings[idx] <= points.size());
#endif
              if (non_virtual_mappings[idx] < points.size())
              {
                result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                          get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE);
              }
              else
              {
                result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                          get_enclosing_physical_context(idx), RegionTreeForest::DIFF);
              }
            }
          }
          // Then we need to pack the mappings for all of the points
          for (unsigned idx = 0; idx < points.size(); idx++)
          {
            result += points[idx]->compute_task_size();
          }
        }
        else
        {
          // Need to pack the region trees and the states  
          result += forest_ctx->compute_region_forest_shape_size(indexes, fields, regions);
          // Need to pack any pre-mapped regions
          result += sizeof(size_t); // number of premapped regions
          for (std::map<unsigned,InstanceRef>::const_iterator it = premapped_regions.begin();
                it != premapped_regions.end(); it++)
          {
            result += sizeof(it->first);
            result += forest_ctx->compute_reference_size(it->second);
          }
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // If it was premapped only need to send the difference
            if (premapped_regions.find(idx) != premapped_regions.end())
            {
              result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                        get_enclosing_physical_context(idx), RegionTreeForest::DIFF);
            }
            else
            {
              result += forest_ctx->compute_region_tree_state_size(regions[idx],
                                        get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE);
            }
          }
          // since nothing has been enumerated, we need to pack the argument map
          result += sizeof(bool); // has argument map
          if (arg_map_impl != NULL)
          {
            result += arg_map_impl->compute_arg_map_size();
          }
        }
        result += forest_ctx->post_compute_region_tree_state_size();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      // If we're going to reset the split factor, do so before packing
      unsigned long temp_split_factor = this->split_factor;
      if (!partially_unpacked)
        this->split_factor = 1;
      pack_multi_task(rez);
      rez.serialize<bool>(distributed);
      rez.serialize<bool>(locally_mapped);
      rez.serialize<bool>(stealable);
      rez.serialize<Event>(termination_event);
      rez.serialize<IndexTask*>(index_owner);
      rez.serialize<unsigned long>(denominator);
      rez.serialize<Processor>(current_proc);
      if (partially_unpacked)
      {
        rez.serialize(remaining_buffer, remaining_bytes);
        free(remaining_buffer);
        remaining_buffer = NULL;
        remaining_bytes = 0;
        partially_unpacked = false;
      }
      else
      {
        if (locally_mapped)
        {
          rez.serialize<size_t>(points.size());
          if (!is_leaf())
          {
            forest_ctx->pack_region_forest_shape(rez);
            forest_ctx->begin_pack_region_tree_state(rez, temp_split_factor);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              rez.serialize<unsigned>(non_virtual_mappings[idx]);
            }
            // Now pack up the region states we need to send
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (non_virtual_mappings[idx] < points.size())
              {
                forest_ctx->pack_region_tree_state(regions[idx],
                                get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE, rez
#ifdef DEBUG_HIGH_LEVEL
                                , idx, variants->name, this->get_unique_id()
#endif
                                );
              }
              else
              {
                forest_ctx->pack_region_tree_state(regions[idx],
                                get_enclosing_physical_context(idx), RegionTreeForest::DIFF, rez
#ifdef DEBUG_HIGH_LEVEL
                                , idx, variants->name, this->get_unique_id()
#endif
                                );
              }
            }
          }
          // Now pack each of the point mappings
          for (unsigned idx = 0; idx < points.size(); idx++)
          {
            points[idx]->pack_task(rez);
          }
        }
        else
        {
          forest_ctx->pack_region_forest_shape(rez);
          forest_ctx->begin_pack_region_tree_state(rez, temp_split_factor);
          // Pack any premapped regions
          rez.serialize(premapped_regions.size());
          for (std::map<unsigned,InstanceRef>::iterator it = premapped_regions.begin();
                it != premapped_regions.end(); it++)
          {
            rez.serialize(it->first);
            forest_ctx->pack_reference(it->second, rez);
          }
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (premapped_regions.find(idx) != premapped_regions.end())
            {
              forest_ctx->pack_region_tree_state(regions[idx],
                              get_enclosing_physical_context(idx), RegionTreeForest::DIFF, rez
#ifdef DEBUG_HIGH_LEVEL
                              , idx, variants->name, this->get_unique_id()
#endif
                              );
            }
            else
            {
              forest_ctx->pack_region_tree_state(regions[idx],
                              get_enclosing_physical_context(idx), RegionTreeForest::PRIVILEGE, rez
#ifdef DEBUG_HIGH_LEVEL
                              , idx, variants->name, this->get_unique_id()
#endif
                              );
            }
          }
          // Now we need to pack the argument map
          bool has_arg_map = (arg_map_impl != NULL);
          rez.serialize<bool>(has_arg_map);
          if (has_arg_map)
          {
            arg_map_impl->pack_arg_map(rez); 
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::unpack_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert_context_locked();
#endif
      unpack_multi_task(derez);
      derez.deserialize<bool>(distributed);
      derez.deserialize<bool>(locally_mapped);
      derez.deserialize<bool>(stealable);
      remote = true;
      derez.deserialize<Event>(termination_event);
      derez.deserialize<IndexTask*>(index_owner);
      derez.deserialize<unsigned long>(denominator);
      derez.deserialize<Processor>(current_proc);
      remaining_bytes = derez.get_remaining_bytes();
      remaining_buffer = malloc(remaining_bytes);
      derez.deserialize(remaining_buffer,remaining_bytes);
      partially_unpacked = true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::finish_task_unpack(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(partially_unpacked);
#endif
      Deserializer derez(remaining_buffer,remaining_bytes);
      lock_context();
      if (locally_mapped)
      {
        size_t num_points;
        derez.deserialize<size_t>(num_points);
        if (!is_leaf())
        {
          forest_ctx->unpack_region_forest_shape(derez);
          forest_ctx->begin_unpack_region_tree_state(derez, split_factor);
          non_virtual_mappings.resize(regions.size());  
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            derez.deserialize<unsigned>(non_virtual_mappings[idx]);
          }
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mappings[idx] < num_points)
            {
              // Unpack the physical state in our context
              forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::PRIVILEGE, derez
#ifdef DEBUG_HIGH_LEVEL
                  , idx, variants->name, this->get_unique_id()
#endif
                  );
            }
            else
            {
              forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::DIFF, derez
#ifdef DEBUG_HIGH_LEVEL
                  , idx, variants->name, this->get_unique_id()
#endif
                  );
            }
          }
        }
        for (unsigned idx = 0; idx < num_points; idx++)
        {
          // Clone this as a point task, then unpack it
          PointTask *next_point = clone_as_point_task(false/*new point*/);
          next_point->unpack_task(derez);
          points.push_back(next_point);
        }
      }
      else
      {
        forest_ctx->unpack_region_forest_shape(derez);
        forest_ctx->begin_unpack_region_tree_state(derez, split_factor);
        // Unpack any premapped regions
        size_t num_premapped;
        derez.deserialize(num_premapped);
        for (unsigned idx = 0; idx < num_premapped; idx++)
        {
          unsigned region_idx;
          derez.deserialize(region_idx);
          premapped_regions[region_idx] = forest_ctx->unpack_reference(derez);
        }
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (premapped_regions.find(idx) != premapped_regions.end())
          {
            forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::DIFF, derez
#ifdef DEBUG_HIGH_LEVEL
                , idx, variants->name, this->get_unique_id()
#endif
                );
          }
          else
          {
            forest_ctx->unpack_region_tree_state(regions[idx], ctx_id, RegionTreeForest::PRIVILEGE, derez
#ifdef DEBUG_HIGH_LEVEL
                , idx, variants->name, this->get_unique_id()
#endif
                );
          }
        }
        
        // Unpack the argument map
        bool has_arg_map;
        derez.deserialize<bool>(has_arg_map);
        if (has_arg_map)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(arg_map_impl == NULL);
#endif
          arg_map_impl = new ArgumentMapImpl(new ArgumentMapStore());
          arg_map_impl->add_reference();
          arg_map_impl->unpack_arg_map(derez);
        }
      }
      unlock_context();
      free(remaining_buffer);
      remaining_buffer = NULL;
      remaining_bytes = 0;
      partially_unpacked = false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_start(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_children_mapped(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::remote_finish(const void *args, size_t arglen)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::pre_slice(void)
    //--------------------------------------------------------------------------
    {
      // Intentionally do nothing
      return true; // success
    }

    //--------------------------------------------------------------------------
    bool SliceTask::post_slice(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!slices.empty());
#endif
      for (std::list<SliceTask*>::const_iterator it = slices.begin();
            it != slices.end(); it++)
      {
        (*it)->set_denominator(denominator*slices.size(), split_factor*slices.size());
      }

      // Deactivate this context when done since we've split it into sub-slices
      return true;
    }

    //--------------------------------------------------------------------------
    SliceTask* SliceTask::clone_as_slice_task(IndexSpace new_space, Processor target,
                                              bool recurse, bool steal)
    //--------------------------------------------------------------------------
    {
      SliceTask *result = runtime->get_available_slice_task(this/*use this as the parent in case remote*/);
      result->clone_multi_from(this, new_space, recurse);
      result->distributed = false;
      result->locally_mapped = this->locally_mapped;
      result->stealable = steal;
      result->remote = this->remote;
      result->termination_event = this->termination_event;
      result->current_proc = target;
      result->index_owner = this->index_owner;
      result->partially_unpacked = this->partially_unpacked;
      if (partially_unpacked)
      {
        result->remaining_buffer = malloc(this->remaining_bytes);
        memcpy(result->remaining_buffer, this->remaining_buffer, this->remaining_bytes);
        result->remaining_bytes = this->remaining_bytes;
      }
      // denominator gets set by post_slice
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const AnyPoint &point, const void *result, size_t result_size, Event ready_event)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        if (has_reduction)
        {
          // Get the reduction op 
          const ReductionOp *redop = HighLevelRuntime::get_reduction_op(redop_id);
#ifdef DEBUG_HIGH_LEVEL
          assert(reduction_state != NULL);
          assert(reduction_state_size == redop->sizeof_rhs);
          assert(result_size == redop->sizeof_rhs);
#endif
          lock();
          // Fold the value
          redop->fold(reduction_state, result, 1/*num elements*/);
          unlock();
        }
        else
        {
          // We need to store the value locally
          // Copy the value over
          void *future_copy = malloc(result_size);
          memcpy(future_copy, result, result_size);
          lock();
#ifdef DEBUG_HIGH_LEVEL
          assert(future_results.find(point) == future_results.end());
#endif
          future_results[point] = FutureResult(future_copy,result_size,ready_event); 
          unlock();
        }
      }
      else
      {
        index_owner->handle_future(point,result,result_size,ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::set_denominator(unsigned long denom, unsigned long split)
    //--------------------------------------------------------------------------
    {
      this->denominator = denom;
      this->split_factor = denom;
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(bool new_point)
    //--------------------------------------------------------------------------
    {
      PointTask *result = runtime->get_available_point_task(this);
      result->clone_task_context_from(this);
      // Clear out the region requirements since we don't actually want that clone
      result->regions.clear();
      result->slice_owner = this;
      if (new_point)
      {
        result->point_termination_event = UserEvent::create_user_event();
#ifdef LEGION_SPY
        // Log a dependence between the point event and the final termination event
        LegionSpy::log_event_dependence(result->point_termination_event, this->termination_event);
#endif
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_mapped(PointTask *point)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(num_unmapped_points > 0);
#endif
      // Decrement the count of the number of unmapped children
      num_unmapped_points--;
      if (!enumerating && (num_unmapped_points == 0))
      {
        unlock();
        post_slice_mapped();
      }
      else
      {
        unlock();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::point_task_finished(PointTask *point)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(num_unfinished_points > 0);
#endif
      num_unfinished_points--;
      if (!enumerating && (num_unfinished_points == 0))
      {
        unlock();
        post_slice_finished();
      }
      else
      {
        unlock();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_start(void)
    //--------------------------------------------------------------------------
    {
      // Figure out if we're a leaf, will be the same for all points
      // since they will all select the same variant (at least for now)
#ifdef DEBUG_HIGH_LEVEL
      assert(!points.empty());
#endif
      // Initialize the non_virtual_mappings vector
      non_virtual_mappings.resize(regions.size());
      for (unsigned idx = 0; idx < non_virtual_mappings.size(); idx++)
      {
        non_virtual_mappings[idx] = 0;
      }
      // Go through and figure out how many non-virtual mappings there have been
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->non_virtual_mapped_region.size() == regions.size());
#endif
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if ((*it)->non_virtual_mapped_region[idx])
            non_virtual_mappings[idx]++; 
        }
      }
      if (remote)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!locally_mapped); // shouldn't be here if we were locally mapped
#endif
        // Otherwise we have to pack stuff up and send it back
        size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner);
        buffer_size += sizeof(denominator);
        buffer_size += sizeof(size_t); // number of points
        buffer_size += (regions.size() * sizeof(unsigned));
        lock_context();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(non_virtual_mappings[idx] <= points.size());
#endif
          // Send back the physically mapped parts if they are all mapped
          // and it wasn't premapped
          if ((non_virtual_mappings[idx] == points.size()) &&
              (premapped_regions.find(idx) == premapped_regions.end()))
          {
            // Everybody mapped a region in this tree, it is fully mapped
            // so send it back
            buffer_size += compute_state_return_size(idx, ctx_id, RegionTreeForest::PHYSICAL);
          }
        }
        buffer_size += forest_ctx->post_compute_region_tree_state_return(false/*created only*/);
        // Now pack everything up
        Serializer rez(buffer_size);
        rez.serialize(orig_proc);
        rez.serialize<IndexTask*>(index_owner);
        rez.serialize(denominator);
        rez.serialize(points.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          rez.serialize<unsigned>(non_virtual_mappings[idx]);
        }
        forest_ctx->begin_pack_region_tree_state_return(rez, false/*created only*/);
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if ((non_virtual_mappings[idx] == points.size()) &&
              (premapped_regions.find(idx) == premapped_regions.end()))
          {
            pack_tree_state_return(idx, ctx_id, RegionTreeForest::PHYSICAL, rez);
          }
        }
        forest_ctx->end_pack_region_tree_state_return(rez, false/*created only*/);
        unlock_context();
        // Now send it back to the utility processor
        Processor utility = orig_proc.get_utility_processor();
        this->remote_start_event = utility.spawn(NOTIFY_START_ID,rez.get_buffer(),buffer_size);
      }
      else
      {
        // If we're not remote we can just tell our index space context directly
        index_owner->slice_start(denominator, points.size(), non_virtual_mappings);
#ifdef DEBUG_HIGH_LEVEL
        // Also check for overlapping slices
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if ((regions[idx].handle_type == PROJECTION) &&
              (IS_WRITE(regions[idx])))
          {
            std::set<LogicalRegion> touched_regions;
            for (std::vector<PointTask*>::const_iterator it = points.begin();
                  it != points.end(); it++)
            {
              touched_regions.insert((*it)->regions[idx].region);
            }
            index_owner->check_overlapping_slices(idx, touched_regions);
          }
        }
#endif
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_mapped(void)
    //--------------------------------------------------------------------------
    {
      // Do a quick check to see if we mapped everything in the first phase
      // in which case we don't have to send this message
      {
        bool all_mapped = true;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (non_virtual_mappings[idx] < points.size())
          {
            all_mapped = false;
            break;
          }
        }
        if (all_mapped)
          return; // We're done
      }
      if (remote)
      {
        // Only send stuff back if we're not a leaf.
        if (!is_leaf())
        {
          // Need to send back the results to the enclosing context
          size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner); 
          lock_context();
          buffer_size += (regions.size() * sizeof(unsigned));
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          // Figure out which states we need to send back
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            // If we didn't send it back before, we need to send it back now
            if (non_virtual_mappings[idx] < points.size())
            {
              buffer_size += compute_state_return_size(idx, ctx_id, RegionTreeForest::PRIVILEGE);
            }
            else
            {
              // Send back the ones that we have privileges on, but didn't make a physical instance
              buffer_size += compute_state_return_size(idx, ctx_id, RegionTreeForest::DIFF);
            }
          }
          buffer_size += forest_ctx->post_compute_region_tree_state_return(false/*created only*/);
          buffer_size += sizeof(size_t);
          // Also send back any source copy instances to be released
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            buffer_size += (*it)->compute_source_copy_instances_return();
          }
          // Now pack it all up
          Serializer rez(buffer_size);
          rez.serialize<Processor>(orig_proc);
          rez.serialize<IndexTask*>(index_owner);
          {
            unsigned num_points = points.size();
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              rez.serialize<unsigned>(num_points-non_virtual_mappings[idx]);
            }
          }
          forest_ctx->pack_region_tree_updates_return(rez);
          forest_ctx->begin_pack_region_tree_state_return(rez, false/*created only*/);
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            if (non_virtual_mappings[idx] < points.size())
            {
              pack_tree_state_return(idx, ctx_id, RegionTreeForest::PRIVILEGE, rez);
            }
            else
            {
              pack_tree_state_return(idx, ctx_id, RegionTreeForest::DIFF, rez);
            }
          }
          forest_ctx->end_pack_region_tree_state_return(rez, false/*created only*/);
          rez.serialize<size_t>(points.size());
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            (*it)->pack_source_copy_instances_return(rez);
          }
          unlock_context();
          // Send it back on the utility processor
          Processor utility = orig_proc.get_utility_processor();
          this->remote_mapped_event = utility.spawn(NOTIFY_MAPPED_ID,rez.get_buffer(),buffer_size,this->remote_start_event);
        }
      }
      else
      {
        // Otherwise we're local so just tell our enclosing context that all our remaining points are mapped
        std::vector<unsigned> virtual_mapped(regions.size());
        unsigned num_points = points.size();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          virtual_mapped[idx] = num_points - non_virtual_mappings[idx];
        }
        index_owner->slice_mapped(virtual_mapped);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::post_slice_finished(void)
    //--------------------------------------------------------------------------
    {
      if (remote)
      {
        size_t result_size = 0;
        // Need to send back the results to the enclosing context
        size_t buffer_size = sizeof(orig_proc) + sizeof(index_owner);
        buffer_size += sizeof(size_t); // number of points
        // Need to send back the tasks for which we have privileges
        lock_context();
        if (!is_leaf())
        {
          buffer_size += compute_privileges_return_size();
          buffer_size += compute_deletions_return_size();
          buffer_size += forest_ctx->compute_region_tree_updates_return();
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            buffer_size += (*it)->compute_return_created_contexts();
          }
          buffer_size += forest_ctx->post_compute_region_tree_state_return(true/*created only*/);
        }
        // Always need to send back the leaked return state
        buffer_size += forest_ctx->compute_leaked_return_size();
        if (has_reduction)
        {
          buffer_size += sizeof(reduction_state_size); 
          buffer_size += reduction_state_size;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(future_results.size() == points.size());
#endif
          // Get the result size
          std::map<AnyPoint,FutureResult>::const_iterator first_point = future_results.begin();
          this->index_element_size = first_point->first.elmt_size;
          this->index_dimensions = first_point->first.dim;
          result_size = first_point->second.buffer_size;

          buffer_size += sizeof(size_t); // number of future results
          buffer_size += sizeof(this->index_element_size);
          buffer_size += sizeof(this->index_dimensions);
          buffer_size += (future_results.size() * (index_dimensions*index_element_size + result_size + sizeof(Event)));
        }

        Serializer rez(buffer_size);
        rez.serialize<Processor>(orig_proc);
        rez.serialize<IndexTask*>(index_owner);
        rez.serialize<size_t>(points.size());
        if (!is_leaf())
        {
          pack_privileges_return(rez);
          pack_deletions_return(rez);
          forest_ctx->pack_region_tree_updates_return(rez);
          forest_ctx->begin_pack_region_tree_state_return(rez, true/*created only*/);
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            (*it)->pack_return_created_contexts(rez);
          }
          forest_ctx->end_pack_region_tree_state_return(rez, true/*created only*/);
        }
        forest_ctx->pack_leaked_return(rez);
        unlock_context();
        // Pack up the future(s)
        if (has_reduction)
        {
          rez.serialize<size_t>(reduction_state_size);
          rez.serialize(reduction_state,reduction_state_size);
        }
        else
        {
          rez.serialize<size_t>(result_size); 
          rez.serialize<size_t>(this->index_element_size);
          rez.serialize<unsigned>(this->index_dimensions);
          for (std::map<AnyPoint,FutureResult>::const_iterator it = future_results.begin();
                it != future_results.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(it->first.elmt_size == index_element_size);
            assert(it->first.dim == index_dimensions);
#endif
            rez.serialize(it->first.buffer,(it->first.elmt_size) * (it->first.dim));
#ifdef DEBUG_HIGH_LEVEL
            assert(it->second.buffer_size == result_size);
#endif
            rez.serialize(it->second.buffer,result_size);
            rez.serialize(it->second.ready_event);
          }
        }
        // Send it back on the utility processor
        Processor utility = orig_proc.get_utility_processor();
        utility.spawn(NOTIFY_FINISH_ID,rez.get_buffer(),buffer_size,Event::merge_events(remote_start_event,remote_mapped_event));
      }
      else
      {
        // Otherwise we're done, so pass back our privileges and then tell the owner
        index_owner->return_privileges(created_index_spaces,created_field_spaces,
                                        created_regions,created_fields);
        index_owner->return_deletions(deleted_regions, deleted_partitions, deleted_fields);
        for (std::vector<PointTask*>::const_iterator it = points.begin();
              it != points.end(); it++)
        {
          (*it)->return_created_field_contexts(index_owner->parent_ctx);
        }
        // Created field spaces have already been sent back
        index_owner->slice_finished(points.size());
      }

      // Once we're done doing this we need to deactivate any point tasks we have
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        (*it)->deactivate();
      }

      // Finally deactivate ourselves.  Note we do this regardless of whether we're remote
      // or not since all Slice Tasks are responsible for deactivating themselves
      this->deactivate();
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::compute_state_return_size(unsigned idx, ContextID ctx, RegionTreeForest::SendingMode mode)
    //--------------------------------------------------------------------------
    {
      // If mode is a DIFF, make sure that the difference isn't empty
      if (mode == RegionTreeForest::DIFF)
      {
        std::set<FieldID> packing_fields;
        for (std::vector<FieldID>::const_iterator it = regions[idx].instance_fields.begin();
              it != regions[idx].instance_fields.end(); it++)
        {
          packing_fields.erase(*it);
        }
        if (packing_fields.empty())
          return 0;
      }
      size_t result = 0;
      if (regions[idx].handle_type == SINGULAR)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!IS_WRITE(regions[idx])); // if this was the case it should have been premapped
#endif
        result += forest_ctx->compute_region_tree_state_return(regions[idx], idx, ctx, 
                                                    false/*overwrite*/, mode
#ifdef DEBUG_HIGH_LEVEL
                                                    , variants->name
                                                    , this->get_unique_id()
#endif
                                                    );
      }
      else
      {
        if (IS_WRITE(regions[idx]))
        {
          // Send back each of the points individually
          std::set<LogicalRegion> sending_set;
          result += sizeof(size_t); // number of regions touched
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            if (sending_set.find((*it)->regions[idx].region) == sending_set.end())
            {
#ifdef DEBUG_HIGH_LEVEL
              assert((*it)->regions[idx].handle_type == SINGULAR);
#endif
              sending_set.insert((*it)->regions[idx].region);
              result += forest_ctx->compute_region_tree_state_return((*it)->regions[idx], idx, ctx,
                                                      true/*overwrite*/, mode
#ifdef DEBUG_HIGH_LEVEL
                                                      , variants->name
                                                      , this->get_unique_id()
#endif
                                                      );
            }
          }
          forest_ctx->post_partition_state_return(regions[idx], ctx, mode);
          result += (sending_set.size() * sizeof(LogicalRegion));
        }
        else
        {
          // We this was a read-only or reduce requirement so we
          // only need to send back the diff
          result += forest_ctx->compute_region_tree_state_return(regions[idx], idx, ctx,
                                                      false/*overwrite*/, mode
#ifdef DEBUG_HIGH_LEVEL
                                                      , variants->name
                                                      , this->get_unique_id()
#endif
                                                      );
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_tree_state_return(unsigned idx, ContextID ctx, 
                RegionTreeForest::SendingMode mode, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // If mode is a DIFF, make sure that the difference isn't empty
      if (mode == RegionTreeForest::DIFF)
      {
        std::set<FieldID> packing_fields;
        for (std::vector<FieldID>::const_iterator it = regions[idx].instance_fields.begin();
              it != regions[idx].instance_fields.end(); it++)
        {
          packing_fields.erase(*it);
        }
        if (packing_fields.empty())
          return;
      }
      if (regions[idx].handle_type == SINGULAR)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(!IS_WRITE(regions[idx])); // if this was the case it should have been premapped
#endif
        forest_ctx->pack_region_tree_state_return(regions[idx], idx, ctx, false/*overwrite*/, 
                                                  mode, rez);
      }
      else
      {
        if (IS_WRITE(regions[idx]))
        {
          std::map<LogicalRegion,PointTask*> sending_set;
          for (std::vector<PointTask*>::const_iterator it = points.begin();
                it != points.end(); it++)
          {
            sending_set[(*it)->regions[idx].region] = *it;
          }
          rez.serialize(sending_set.size());
          for (std::map<LogicalRegion,PointTask*>::const_iterator it = sending_set.begin();
                it != sending_set.end(); it++)
          {
            rez.serialize(it->first);
            forest_ctx->pack_region_tree_state_return(it->second->regions[idx], idx, ctx,
                                          true/*overwrite*/, mode, rez);
          }
          forest_ctx->post_partition_pack_return(regions[idx], ctx, mode);
        }
        else
        {
          forest_ctx->pack_region_tree_state_return(regions[idx], idx, ctx, 
                                false/*overwrite*/, mode, rez);
        }
      }
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime 

// EOF

