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
#include "legion_logging.h"
#include "legion_profiling.h"

namespace LegionRuntime {
  namespace HighLevel {

    Logger::Category log_run("runtime");
    Logger::Category log_task("tasks");
    Logger::Category log_index("index_spaces");
    Logger::Category log_field("field_spaces");
    Logger::Category log_region("regions");
    Logger::Category log_inst("instances");
    Logger::Category log_garbage("gc");
    Logger::Category log_leak("leaks");
    Logger::Category log_variant("variants");
#ifdef LEGION_SPY
    namespace LegionSpy {
      Logger::Category log_spy("legion_spy");
    };
#endif

#ifdef LEGION_LOGGING
    namespace LegionLogging {
      Logger::Category log_logging("legion_logging");
      std::list<ProcessorProfiler *> processor_profilers;
      pthread_key_t pthread_profiler_key;
      pthread_mutex_t profiler_mutex = PTHREAD_MUTEX_INITIALIZER;
      std::deque<LogMsgProcessor> msgs_processor;
      std::deque<LogMsgMemory> msgs_memory;
      std::deque<LogMsgProcMemAffinity> msgs_proc_mem_affinity;
      std::deque<LogMsgMemMemAffinity> msgs_mem_mem_affinity;
      std::deque<LogMsgTaskCollection> msgs_task_collection;
      std::deque<LogMsgTaskVariant> msgs_task_variant;
      std::deque<LogMsgTopLevelTask> msgs_top_level_task;
      unsigned long long init_time;
      AddressSpaceID address_space;
    };
#endif

#ifdef LEGION_PROF
    namespace LegionProf {
      Logger::Category log_prof("legion_prof");
      ProcessorProfiler *legion_prof_table = 
        new ProcessorProfiler[MAX_NUM_PROCS + 1];
      bool profiling_enabled = true;
    };
#endif

    const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
    const LogicalPartition LogicalPartition::NO_PART = LogicalPartition(); 

    /////////////////////////////////////////////////////////////
    // Mappable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mappable::Mappable(void)
      : map_id(0)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Task::Task(void)
      : Mappable(), args(NULL), arglen(0), local_args(NULL), local_arglen(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Task::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return depth;
    }

    /////////////////////////////////////////////////////////////
    // Copy 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Copy::Copy(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Copy::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Inline 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Inline::Inline(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Inline::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Acquire 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Acquire::Acquire(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Acquire::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // Release 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Release::Release(void)
      : Mappable(), parent_task(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    unsigned Release::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_task->depth+1);
    }

    /////////////////////////////////////////////////////////////
    // FieldSpace 
    /////////////////////////////////////////////////////////////

    /*static*/ const FieldSpace FieldSpace::NO_SPACE = FieldSpace(0);

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(unsigned _id)
      : id(_id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(void)
      : id(0)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    FieldSpace::FieldSpace(const FieldSpace &rhs)
      : id(rhs.id)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Region  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(RegionTreeID tid, IndexSpace index, 
                                 FieldSpace field)
      : tree_id(tid), index_space(index), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(void)
      : tree_id(0), index_space(IndexSpace::NO_SPACE), 
        field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(const LogicalRegion &rhs)
      : tree_id(rhs.tree_id), index_space(rhs.index_space), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Partition 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(RegionTreeID tid, IndexPartition pid, 
                                       FieldSpace field)
      : tree_id(tid), index_partition(pid), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(void)
      : tree_id(0), index_partition(0), field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(const LogicalPartition &rhs)
      : tree_id(rhs.tree_id), index_partition(rhs.index_partition), 
        field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Index Allocator 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(void)
      : index_space(IndexSpace::NO_SPACE), allocator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(const IndexAllocator &rhs)
      : index_space(rhs.index_space), allocator(rhs.allocator)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(IndexSpace space, IndexSpaceAllocator *a)
      : index_space(space), allocator(a) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::~IndexAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator& IndexAllocator::operator=(const IndexAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      index_space = rhs.index_space;
      allocator = rhs.allocator;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(void)
      : field_space(FieldSpace::NO_SPACE), parent(NULL), runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(const FieldAllocator &allocator)
      : field_space(allocator.field_space), parent(allocator.parent), 
        runtime(allocator.runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(FieldSpace f, Context p, 
                                   HighLevelRuntime *rt)
      : field_space(f), parent(p), runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::~FieldAllocator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator& FieldAllocator::operator=(const FieldAllocator &rhs)
    //--------------------------------------------------------------------------
    {
      field_space = rhs.field_space;
      parent = rhs.parent;
      runtime = rhs.runtime;
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      impl = new ArgumentMap::Impl();
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(const ArgumentMap &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    ArgumentMap::~ArgumentMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        // Remove our reference and if we were the
        // last reference holder, then delete it
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }
    
    //--------------------------------------------------------------------------
    ArgumentMap& ArgumentMap::operator=(const ArgumentMap &rhs)
    //--------------------------------------------------------------------------
    {
      // Check to see if our current impl is not NULL,
      // if so remove our reference
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      impl = rhs.impl;
      // Add our reference to the new impl
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::has_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->has_point(point);
    }

    //--------------------------------------------------------------------------
    void ArgumentMap::set_point(const DomainPoint &point, 
                                const TaskArgument &arg, bool replace/*= true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->set_point(point, arg, replace);
    }

    //--------------------------------------------------------------------------
    bool ArgumentMap::remove_point(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->remove_point(point);
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMap::get_point(const DomainPoint &point) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_point(point);
    }

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////

    const Predicate Predicate::TRUE_PRED = Predicate(true);
    const Predicate Predicate::FALSE_PRED = Predicate(false);

    //--------------------------------------------------------------------------
    Predicate::Predicate(void)
      : impl(NULL), const_value(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(bool value)
      : impl(NULL), const_value(value)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(Predicate::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::~Predicate(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->remove_reference();
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Predicate& Predicate::operator=(const Predicate &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->remove_reference();
      }
      const_value = rhs.const_value;
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Lock 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Lock::Lock(void)
      : reservation_lock(Reservation::NO_RESERVATION)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Lock::Lock(Reservation r)
      : reservation_lock(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool Lock::operator<(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock < rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    bool Lock::operator==(const Lock &rhs) const
    //--------------------------------------------------------------------------
    {
      return (reservation_lock == rhs.reservation_lock);
    }

    //--------------------------------------------------------------------------
    void Lock::acquire(unsigned mode /*=0*/, bool exclusive /*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reservation_lock.exists());
#endif
      Event lock_event = reservation_lock.acquire(mode,exclusive);
      if (!lock_event.has_triggered())
      {
        Processor proc = Machine::get_executing_processor();
        Runtime *rt = Runtime::get_runtime(proc);
        rt->pre_wait(proc);
        lock_event.wait();
        rt->post_wait(proc);
      }
    }

    //--------------------------------------------------------------------------
    void Lock::release(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(reservation_lock.exists());
#endif
      reservation_lock.release();
    }

    /////////////////////////////////////////////////////////////
    // Grant 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Grant::Grant(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Grant::Grant(Grant::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::Grant(const Grant &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Grant::~Grant(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Grant& Grant::operator=(const Grant &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Phase Barrier 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhaseBarrier::PhaseBarrier(Barrier b, unsigned parts)
      : phase_barrier(b), participants(parts)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator<(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier < rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    bool PhaseBarrier::operator==(const PhaseBarrier &rhs) const
    //--------------------------------------------------------------------------
    {
      return (phase_barrier == rhs.phase_barrier);
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::arrive(unsigned count /*=0*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(phase_barrier.exists());
#endif
      // This is a no-op since we would just end up
      // altering the arrival count and then decrementing
      // it by the same amount.
    }

    //--------------------------------------------------------------------------
    void PhaseBarrier::wait(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(phase_barrier.exists());
#endif
      if (!phase_barrier.has_triggered())
      {
        Processor proc = Machine::get_executing_processor();
        Runtime *rt = Runtime::get_runtime(proc);
        rt->pre_wait(proc);
        phase_barrier.wait();
        rt->post_wait(proc);
      }
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                        const std::set<FieldID> &priv_fields,
                                        const std::vector<FieldID> &inst_fields,
                                        PrivilegeMode _priv, 
                                        CoherenceProperty _prop, 
                                        LogicalRegion _parent,
				        MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR), projection(0)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                ProjectionID _proj, 
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                ProjectionID _proj,
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                    const std::set<FieldID> &priv_fields,
                                    const std::vector<FieldID> &inst_fields,
                                    ReductionOpID op, CoherenceProperty _prop, 
                                    LogicalRegion _parent, MappingTagID _tag, 
                                    bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                        ProjectionID _proj,  
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                        ProjectionID _proj,
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, 
                        bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG),
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent,
					 MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj, 
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         PrivilegeMode _priv, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement "
                                "constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop, 
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(SINGULAR)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, 
                                         ProjectionID _proj,  
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(PART_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                         ProjectionID _proj,
                                         ReductionOpID op, 
                                         CoherenceProperty _prop,
                                         LogicalRegion _parent, 
                                         MappingTagID _tag, 
                                         bool _verified)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), flags(_verified ? VERIFIED_FLAG : NO_FLAG), 
        handle_type(REG_PROJECTION), projection(_proj)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (redop == 0)
      {
        log_region(LEVEL_ERROR,"Zero is not a valid ReductionOpID");
        assert(false);
        exit(ERROR_RESERVED_REDOP_ID);
      }
#endif
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator==(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if ((handle_type == rhs.handle_type) && (privilege == rhs.privilege) &&
          (prop == rhs.prop) && (parent == rhs.parent) && (redop == rhs.redop)
          && (tag == rhs.tag) && (flags == rhs.flags))
      {
        if (((handle_type == SINGULAR) && (region == rhs.region)) ||
            ((handle_type == PART_PROJECTION) && (partition == rhs.partition) && 
             (projection == rhs.projection)) ||
            ((handle_type == REG_PROJECTION) && (region == rhs.region)))
        {
          if ((privilege_fields.size() == rhs.privilege_fields.size()) &&
              (instance_fields.size() == rhs.instance_fields.size()))
          {
            return ((privilege_fields == rhs.privilege_fields) 
                && (instance_fields == rhs.instance_fields));
          }
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool RegionRequirement::operator<(const RegionRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle_type < rhs.handle_type)
        return true;
      else if (handle_type > rhs.handle_type)
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (prop < rhs.prop)
            return true;
          else if (prop > rhs.prop)
            return false;
          else
          {
            if (parent < rhs.parent)
              return true;
            else if (!(parent == rhs.parent)) // therefore greater than
              return false;
            else
            {
              if (redop < rhs.redop)
                return true;
              else if (redop > rhs.redop)
                return false;
              else
              {
                if (tag < rhs.tag)
                  return true;
                else if (tag > rhs.tag)
                  return false;
                else
                {
                  if (flags < rhs.flags)
                    return true;
                  else if (flags > rhs.flags)
                    return false;
                  else
                  {
                    if (privilege_fields < rhs.privilege_fields)
                      return true;
                    else if (privilege_fields > rhs.privilege_fields)
                      return false;
                    else
                    {
                      if (instance_fields < rhs.instance_fields)
                        return true;
                      else if (instance_fields > rhs.instance_fields)
                        return false;
                      else
                      {
                        if (handle_type == SINGULAR)
                          return (region < rhs.region);
                        else if (handle_type == PART_PROJECTION)
                        {
                          if (partition < rhs.partition)
                            return true;
                          // therefore greater than
                          else if (partition != rhs.partition) 
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                        else
                        {
                          if (region < rhs.region)
                            return true;
                          else if (region != rhs.region)
                            return false;
                          else
                            return (projection < rhs.projection);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

#ifdef PRIVILEGE_CHECKS
    //--------------------------------------------------------------------------
    AccessorPrivilege RegionRequirement::get_accessor_privilege(void) const
    //--------------------------------------------------------------------------
    {
      switch (privilege)
      {
        case NO_ACCESS:
          return ACCESSOR_NONE;
        case READ_ONLY:
          return ACCESSOR_READ;
        case READ_WRITE:
        case WRITE_DISCARD:
          return ACCESSOR_ALL;
        case REDUCE:
          return ACCESSOR_REDUCE;
        default:
          assert(false);
      }
      return ACCESSOR_NONE;
    }
#endif

    //--------------------------------------------------------------------------
    bool RegionRequirement::has_field_privilege(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      return (privilege_fields.find(fid) != privilege_fields.end());
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::copy_without_mapping_info(
                                                  const RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      region = rhs.region;
      partition = rhs.partition;
      privilege_fields = rhs.privilege_fields;
      instance_fields = rhs.instance_fields;
      privilege = rhs.privilege;
      prop = rhs.prop;
      parent = rhs.parent;
      redop = rhs.redop;
      tag = rhs.tag;
      flags = rhs.flags;
      handle_type = rhs.handle_type;
      projection = rhs.projection;
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::initialize_mapping_fields(void)
    //--------------------------------------------------------------------------
    {
      premapped = false;
      must_early_map = false;
      restricted = false;
      max_blocking_factor = 1;
      current_instances.clear();
      virtual_map = false;
      early_map = false;
      enable_WAR_optimization = false;
      reduction_list = false;
      make_persistent = false;
      blocking_factor = 1;
      target_ranking.clear();
      additional_fields.clear();
      mapping_failed = false;
      selected_memory = Memory::NO_MEMORY;
    }

    /////////////////////////////////////////////////////////////
    // Index Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(void)
      : handle(IndexSpace::NO_SPACE), privilege(NO_MEMORY), 
        parent(IndexSpace::NO_SPACE), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpaceRequirement::IndexSpaceRequirement(IndexSpace _handle, 
                                                 AllocateMode _priv,
                                                 IndexSpace _parent, 
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator<(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (handle != rhs.handle) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
        {
          if (parent < rhs.parent)
            return true;
          else if (parent != rhs.parent) // therefore greater than
            return false;
          else
            return verified < rhs.verified;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator==(
                                        const IndexSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) &&
             (parent == rhs.parent) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Field Space Requirement 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(void)
      : handle(FieldSpace::NO_SPACE), privilege(NO_MEMORY), verified(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldSpaceRequirement::FieldSpaceRequirement(FieldSpace _handle, 
                                                 AllocateMode _priv,
                                                 bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator<(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      if (handle < rhs.handle)
        return true;
      else if (!(handle == rhs.handle)) // therefore greater than
        return false;
      else
      {
        if (privilege < rhs.privilege)
          return true;
        else if (privilege > rhs.privilege)
          return false;
        else
          return verified < rhs.verified;
      }
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator==(
                                        const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && 
              (privilege == rhs.privilege) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // TaskLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(void)
      : task_id(0), argument(TaskArgument()), predicate(Predicate::TRUE_PRED),
        map_id(0), tag(0), point(DomainPoint())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskLauncher::TaskLauncher(Processor::TaskFuncID tid, TaskArgument arg,
                               Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : task_id(tid), argument(arg), predicate(pred), 
        map_id(mid), tag(t), point(DomainPoint())
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // IndexLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexLauncher::IndexLauncher(void)
      : task_id(0), launch_domain(Domain::NO_DOMAIN), 
        global_arg(TaskArgument()), argument_map(ArgumentMap()), 
        predicate(Predicate::TRUE_PRED), must_parallelism(false), 
        map_id(0), tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexLauncher::IndexLauncher(Processor::TaskFuncID tid, Domain dom,
                                 TaskArgument global,
                                 ArgumentMap map,
                                 Predicate pred /*= Predicate::TRUE_PRED*/,
                                 bool must /*=false*/, MapperID mid /*=0*/,
                                 MappingTagID t /*=0*/)
      : task_id(tid), launch_domain(dom), global_arg(global), 
        argument_map(map), predicate(pred), must_parallelism(must),
        map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // InlineLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(void)
      : map_id(0), tag(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InlineLauncher::InlineLauncher(const RegionRequirement &req,
                                   MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : requirement(req), map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // CopyLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyLauncher::CopyLauncher(Predicate pred /*= Predicate::TRUE_PRED*/,
                               MapperID mid /*=0*/, MappingTagID t /*=0*/)
      : predicate(pred), map_id(mid), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // AcquireLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireLauncher::AcquireLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ReleaseLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseLauncher::ReleaseLauncher(LogicalRegion reg, LogicalRegion par,
                                     PhysicalRegion phy,
                                     Predicate pred /*= Predicate::TRUE_PRED*/,
                                     MapperID id /*=0*/, MappingTagID t /*=0*/)
      : logical_region(reg), parent_region(par), physical_region(phy), 
        predicate(pred), map_id(id), tag(t)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // MustEpochLauncher 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochLauncher::MustEpochLauncher(MapperID id /*= 0*/,   
                                         MappingTagID tag/*= 0*/)
      : map_id(id), mapping_tag(tag)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Future 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Future::Future(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Future::Future(const Future &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    Future::~Future(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_gc_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future::Future(Future::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_gc_reference();
    }

    //--------------------------------------------------------------------------
    Future& Future::operator=(const Future &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_gc_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_gc_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void Future::get_void_result(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result();
    }

    //--------------------------------------------------------------------------
    bool Future::is_empty(bool block /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->is_empty(block);
      return true;
    }

    //--------------------------------------------------------------------------
    void* Future::get_untyped_result(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        return impl->get_untyped_result();
      else
        return NULL;
    }

    /////////////////////////////////////////////////////////////
    // Future Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(const FutureMap &map)
      : impl(map.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(FutureMap::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap& FutureMap::operator=(const FutureMap &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    Future FutureMap::get_future(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_future(point);
    }

    //--------------------------------------------------------------------------
    void FutureMap::get_void_result(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->get_void_result(point);
    }

    //--------------------------------------------------------------------------
    void FutureMap::wait_all_results(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->wait_all_results();
    }

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(void)
      : impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(const PhysicalRegion &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(PhysicalRegion::Impl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
        impl->add_reference();
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::~PhysicalRegion(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
          delete impl;
      }
      impl = rhs.impl;
      if (impl != NULL)
        impl->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->wait_until_valid();
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_valid(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->is_valid();
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_logical_region();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_accessor();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic>
      PhysicalRegion::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      return impl->get_field_accessor(fid);
    }

    /////////////////////////////////////////////////////////////
    // Index Iterator  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(IndexSpace space)
      : enumerator(space.get_valid_mask().enumerate_enabled())
    //--------------------------------------------------------------------------
    {
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(LogicalRegion handle)
      : enumerator(handle.get_index_space().get_valid_mask().enumerate_enabled())
    //--------------------------------------------------------------------------
    {
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::IndexIterator(const IndexIterator &rhs)
      : enumerator(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexIterator::~IndexIterator(void)
    //--------------------------------------------------------------------------
    {
      delete enumerator;
    }

    //--------------------------------------------------------------------------
    IndexIterator& IndexIterator::operator=(const IndexIterator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Task Variant Collection
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(Processor::Kind kind, 
                                            bool single,
                                            bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    VariantID TaskVariantCollection::get_variant(Processor::Kind kind, 
                                                 bool single,
                                                 bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            ((it->second.single_task <= single) || 
            (it->second.index_space <= index_space)))
        {
          return it->first;
        }
      }
      log_variant(LEVEL_ERROR,"User task %s (ID %d) has no registered variants "
                              "for processors of kind %d and index space %d",
                              name, user_id, kind, index_space);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_UNREGISTERED_VARIANT);
      return 0;
    }

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      return (variants.find(vid) != variants.end());
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::get_variant(
                                                                  VariantID vid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(variants.find(vid) != variants.end());
#endif
      return variants[vid];
    }

    //--------------------------------------------------------------------------
    void TaskVariantCollection::add_variant(Processor::TaskFuncID low_id, 
                                            Processor::Kind kind, 
                                            bool single, bool index,
                                            bool inner, bool leaf,
                                            VariantID vid)
    //--------------------------------------------------------------------------
    {
      if (vid == AUTO_GENERATE_ID)
      {
        for (unsigned idx = 0; idx < AUTO_GENERATE_ID; idx++)
        {
          if (variants.find(idx) == variants.end())
          {
            vid = idx;
            break;
          }
        }
      }
      variants[vid] = Variant(low_id, kind, single, index, inner, leaf, vid);
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::select_variant(
                                  bool single, bool index, Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && 
            (it->second.single_task <= single) &&
            (it->second.index_space <= index))
        {
          return it->second;
        }
      }
      log_variant(LEVEL_ERROR,"User task %s (ID %d) has no registered variants "
                              "for processors of kind %d and index space %d",
                              name, user_id, kind, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_UNREGISTERED_VARIANT);
      return variants[0];
    }

    /////////////////////////////////////////////////////////////
    // Task Config Options 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskConfigOptions::TaskConfigOptions(bool l /*=false*/,
                                         bool in /*=false*/,
                                         bool idem /*=false*/)
      : leaf(l), inner(in), idempotent(idem)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColoringSerializer::ColoringSerializer(const Coloring &c)
      : coloring(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        result += sizeof(Color);
        result += 2*sizeof(size_t); // number of each kind of pointer
        result += (it->second.points.size() * sizeof(ptr_t));
        result += (it->second.ranges.size() * 2 * sizeof(ptr_t));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer; 
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (Coloring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first;
        target += sizeof(it->first);
        *((size_t*)target) = it->second.points.size();
        target += sizeof(size_t);
        for (std::set<ptr_t>::const_iterator ptr_it = it->second.points.begin();
              ptr_it != it->second.points.end(); ptr_it++)
        {
          *((ptr_t*)target) = *ptr_it;
          target += sizeof(ptr_t);
        }
        *((size_t*)target) = it->second.ranges.size();
        target += sizeof(size_t);
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator range_it = 
              it->second.ranges.begin(); range_it != it->second.ranges.end();
              range_it++)
        {
          *((ptr_t*)target) = range_it->first;
          target += sizeof(range_it->first);
          *((ptr_t*)target) = range_it->second;
          target += sizeof(range_it->second);
        }
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t ColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_colors = *((const size_t*)source);
      source += sizeof(num_colors);
      for (unsigned idx = 0; idx < num_colors; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        coloring[c]; // Force coloring to exist even if empty.
        size_t num_points = *((const size_t*)source);
        source += sizeof(num_points);
        for (unsigned p = 0; p < num_points; p++)
        {
          ptr_t ptr = *((const ptr_t*)source);
          source += sizeof(ptr);
          coloring[c].points.insert(ptr);
        }
        size_t num_ranges = *((const size_t*)source);
        source += sizeof(num_ranges);
        for (unsigned r = 0; r < num_ranges; r++)
        {
          ptr_t start = *((const ptr_t*)source);
          source += sizeof(start);
          ptr_t stop = *((const ptr_t*)source);
          source += sizeof(stop);
          coloring[c].ranges.insert(std::pair<ptr_t,ptr_t>(start,stop));
        }
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // Domain Coloring Serializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DomainColoringSerializer::DomainColoringSerializer(const DomainColoring &d)
      : coloring(d)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_buffer_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of elements
      result += (coloring.size() * (sizeof(Color) + sizeof(Domain)));
      return result;
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_serialize(void *buffer) const
    //--------------------------------------------------------------------------
    {
      char *target = (char*)buffer;
      *((size_t*)target) = coloring.size();
      target += sizeof(size_t);
      for (DomainColoring::const_iterator it = coloring.begin();
            it != coloring.end(); it++)
      {
        *((Color*)target) = it->first; 
        target += sizeof(it->first);
        *((Domain*)target) = it->second;
        target += sizeof(it->second);
      }
      return (size_t(target) - size_t(buffer));
    }

    //--------------------------------------------------------------------------
    size_t DomainColoringSerializer::legion_deserialize(const void *buffer)
    //--------------------------------------------------------------------------
    {
      const char *source = (const char*)buffer;
      size_t num_elements = *((const size_t*)source);
      source += sizeof(size_t);
      for (unsigned idx = 0; idx < num_elements; idx++)
      {
        Color c = *((const Color*)source);
        source += sizeof(c);
        Domain d = *((const Domain*)source);
        source += sizeof(d);
        coloring[c] = d;
      }
      // Return the number of bytes consumed
      return (size_t(source) - size_t(buffer));
    }

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(Runtime *rt)
      : runtime(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::create_index_space(Context ctx,
                                                    size_t max_num_elmts)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, max_num_elmts);
    }

    //--------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_space(ctx, domain);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_space(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          const Coloring &coloring,
                                          bool disjoint,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, coloring, disjoint,
                                             part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(
                                          Context ctx, IndexSpace parent,
                                          Domain color_space,
                                          const DomainColoring &coloring,
                                          bool disjoint, 
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, color_space, coloring,
                                             disjoint, part_color);
    }

    //--------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(
                                          Context ctx, IndexSpace parent,
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> field_accessor,
                                          int part_color)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_partition(ctx, parent, field_accessor, 
                                             part_color);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_partition(Context ctx, 
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_index_partition(Context ctx, 
                                                IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition(ctx, parent, color);
    }

    //--------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::get_index_subspace(Context ctx, 
                                                  IndexPartition p, Color color)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_subspace(ctx, p, color);
    }

    //--------------------------------------------------------------------------
    Domain HighLevelRuntime::get_index_space_domain(Context ctx, 
                                                    IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_domain(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Domain HighLevelRuntime::get_index_partition_color_space(Context ctx, 
                                                             IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color_space(ctx, p);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::get_index_space_partition_colors(Context ctx, 
                                                            IndexSpace sp,
                                                        std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      runtime->get_index_space_partition_colors(ctx, sp, colors);
    }

    //--------------------------------------------------------------------------
    bool HighLevelRuntime::is_index_partition_disjoint(Context ctx, 
                                                       IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return runtime->is_index_partition_disjoint(ctx, p);
    }

    //--------------------------------------------------------------------------
    Color HighLevelRuntime::get_index_space_color(Context ctx, 
                                                  IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_space_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color HighLevelRuntime::get_index_partition_color(Context ctx,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_index_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::get_parent_index_space(Context ctx,
                                                        IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool HighLevelRuntime::has_parent_index_partition(Context ctx,
                                                      IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_parent_index_partition(Context ctx,
                                                              IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_index_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    ptr_t HighLevelRuntime::safe_cast(Context ctx, ptr_t pointer, 
                                      LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      return runtime->safe_cast(ctx, pointer, region);
    }

    //--------------------------------------------------------------------------
    DomainPoint HighLevelRuntime::safe_cast(Context ctx, DomainPoint point, 
                                            LogicalRegion region)
    //--------------------------------------------------------------------------
    {
      return runtime->safe_cast(ctx, point, region);
    }

    //--------------------------------------------------------------------------
    FieldSpace HighLevelRuntime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_space(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_field_space(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_field_space(ctx, handle);
    }

    //--------------------------------------------------------------------------
    size_t HighLevelRuntime::get_field_size(Context ctx, FieldSpace handle,
                                            FieldID fid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_field_size(ctx, handle, fid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::create_logical_region(Context ctx, 
                                            IndexSpace index, FieldSpace fields)
    //--------------------------------------------------------------------------
    {
      return runtime->create_logical_region(ctx, index, fields);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, 
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_partition(Context ctx, 
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition(Context ctx, 
                                    LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition_by_color(
                                    Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition_by_tree(
                                            Context ctx, IndexPartition handle, 
                                            FieldSpace fspace, RegionTreeID tid) 
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion(Context ctx, 
                                    LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion(ctx, parent, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion_by_color(Context ctx, 
                                             LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(ctx, parent, c);
    }

    //--------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion_by_tree(Context ctx, 
                        IndexSpace handle, FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_tree(ctx, handle, fspace, tid);
    }

    //--------------------------------------------------------------------------
    Color HighLevelRuntime::get_logical_region_color(Context ctx,
                                                     LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_region_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    Color HighLevelRuntime::get_logical_partition_color(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_partition_color(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_parent_logical_region(Context ctx,
                                                        LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_region(ctx, handle);
    }

    //--------------------------------------------------------------------------
    bool HighLevelRuntime::has_parent_logical_partition(Context ctx,
                                                        LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->has_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_parent_logical_partition(Context ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return runtime->get_parent_logical_partition(ctx, handle);
    }

    //--------------------------------------------------------------------------
    IndexAllocator HighLevelRuntime::create_index_allocator(Context ctx, 
                                                            IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_index_allocator(ctx, handle);
    }

    //--------------------------------------------------------------------------
    FieldAllocator HighLevelRuntime::create_field_allocator(Context ctx, 
                                                            FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      return runtime->create_field_allocator(ctx, handle);
    }

    //--------------------------------------------------------------------------
    ArgumentMap HighLevelRuntime::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_argument_map(ctx);
    }

    //--------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, 
                                          const TaskLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_task(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    FutureMap HighLevelRuntime::execute_index_space(Context ctx, 
                                                  const IndexLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Future HighLevelRuntime::execute_index_space(Context ctx, 
                            const IndexLauncher &launcher, ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_index_space(ctx, launcher, redop);
    }

    //--------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, 
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
      return runtime->execute_task(ctx, task_id, indexes, fields, regions,
                                   arg, predicate, id, tag);
    }

    //--------------------------------------------------------------------------
    FutureMap HighLevelRuntime::execute_index_space(Context ctx, 
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
      return runtime->execute_index_space(ctx, task_id, domain, indexes,
                                          fields, regions, global_arg,
                                          arg_map, predicate,
                                          must_parallelism, id, tag);
    }


    //--------------------------------------------------------------------------
    Future HighLevelRuntime::execute_index_space(Context ctx, 
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
      return runtime->execute_index_space(ctx, task_id, domain, indexes,
                                          fields, regions, global_arg, arg_map,
                                          reduction, initial_value, predicate,
                                          must_parallelism, id, tag);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, 
                                                const InlineLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, 
                    const RegionRequirement &req, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, req, id, tag);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, unsigned idx, 
                                                  MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->map_region(ctx, idx, id, tag);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::remap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->remap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_region(ctx, region);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::map_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->map_all_regions(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::unmap_all_regions(Context ctx)
    //--------------------------------------------------------------------------
    {
      runtime->unmap_all_regions(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::issue_copy_operation(Context ctx, 
                                                const CopyLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_copy_operation(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    Predicate HighLevelRuntime::create_predicate(Context ctx, const Future &f)
    //--------------------------------------------------------------------------
    {
      return runtime->create_predicate(ctx, f);
    }

    //--------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_not(Context ctx, const Predicate &p) 
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_not(ctx, p);
    }

    //--------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_and(Context ctx, 
                                       const Predicate &p1, const Predicate &p2) 
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_and(ctx, p1, p2);
    }

    //--------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_or(Context ctx,
                                       const Predicate &p1, const Predicate &p2)  
    //--------------------------------------------------------------------------
    {
      return runtime->predicate_or(ctx, p1, p2);
    }

    //--------------------------------------------------------------------------
    Lock HighLevelRuntime::create_lock(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->create_lock(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_lock(Context ctx, Lock l)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_lock(ctx, l);
    }

    //--------------------------------------------------------------------------
    Grant HighLevelRuntime::acquire_grant(Context ctx,
                                      const std::vector<LockRequest> &requests)
    //--------------------------------------------------------------------------
    {
      return runtime->acquire_grant(ctx, requests);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::release_grant(Context ctx, Grant grant)
    //--------------------------------------------------------------------------
    {
      runtime->release_grant(ctx, grant);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier HighLevelRuntime::create_phase_barrier(Context ctx, 
                                                        unsigned participants)
    //--------------------------------------------------------------------------
    {
      return runtime->create_phase_barrier(ctx, participants);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::destroy_phase_barrier(Context ctx, PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      runtime->destroy_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier HighLevelRuntime::advance_phase_barrier(Context ctx, 
                                                         PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      return runtime->advance_phase_barrier(ctx, pb);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::issue_acquire(Context ctx,
                                         const AcquireLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_acquire(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::issue_release(Context ctx,
                                         const ReleaseLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      runtime->issue_release(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::issue_mapping_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_mapping_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::issue_execution_fence(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->issue_execution_fence(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::begin_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      runtime->begin_trace(ctx, tid);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::end_trace(Context ctx, TraceID tid)
    //--------------------------------------------------------------------------
    {
      runtime->end_trace(ctx, tid);
    }

    //--------------------------------------------------------------------------
    FutureMap HighLevelRuntime::execute_must_epoch(Context ctx,
                                              const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
      return runtime->execute_must_epoch(ctx, launcher);
    }

    //--------------------------------------------------------------------------
    int HighLevelRuntime::get_tunable_value(Context ctx, TunableID tid,
                                            MapperID mid, MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      return runtime->get_tunable_value(ctx, tid, mid, tag);
    }

    //--------------------------------------------------------------------------
    Mapper* HighLevelRuntime::get_mapper(Context ctx, MapperID id)
    //--------------------------------------------------------------------------
    {
      return runtime->get_mapper(ctx, id);
    }

    //--------------------------------------------------------------------------
    Processor HighLevelRuntime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->get_executing_processor(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::raise_region_exception(Context ctx, 
                                                  PhysicalRegion region,
                                                  bool nuclear)
    //--------------------------------------------------------------------------
    {
      runtime->raise_region_exception(ctx, region, nuclear);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID map_id, Mapper *mapper, 
                                      Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->add_mapper(map_id, mapper, proc);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::replace_default_mapper(Mapper *mapper, 
                                                  Processor proc)
    //--------------------------------------------------------------------------
    {
      runtime->replace_default_mapper(mapper, proc);
    }

    //--------------------------------------------------------------------------
    FieldID HighLevelRuntime::allocate_field(Context ctx, FieldSpace space,
                                             size_t field_size, FieldID fid,
                                             bool local)
    //--------------------------------------------------------------------------
    {
      return runtime->allocate_field(ctx, space, field_size, fid, local);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::free_field(Context ctx, FieldSpace sp, FieldID fid)
    //--------------------------------------------------------------------------
    {
      runtime->free_field(ctx, sp, fid);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::allocate_fields(Context ctx, FieldSpace space,
                                           const std::vector<size_t> &sizes,
                                         std::vector<FieldID> &resulting_fields,
                                         bool local)
    //--------------------------------------------------------------------------
    {
      runtime->allocate_fields(ctx, space, sizes, resulting_fields, local);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::free_fields(Context ctx, FieldSpace space,
                                       const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------
    {
      runtime->free_fields(ctx, space, to_free);
    }

    //--------------------------------------------------------------------------
    const std::vector<PhysicalRegion>& HighLevelRuntime::begin_task(Context ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->begin_task(ctx);
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::end_task(Context ctx, const void *result, 
                                    size_t result_size, bool owned /*= false*/)
    //--------------------------------------------------------------------------
    {
      runtime->end_task(ctx, result, result_size, owned);
    }

    //--------------------------------------------------------------------------
    const void* HighLevelRuntime::get_local_args(Context ctx, 
                                         DomainPoint &point, size_t &local_size)
    //--------------------------------------------------------------------------
    {
      return runtime->get_local_args(ctx, point, local_size); 
    }
    
    //--------------------------------------------------------------------------
    /*static*/ int HighLevelRuntime::start(int argc, char **argv, 
                                           bool background)
    //--------------------------------------------------------------------------
    {
      return Runtime::start(argc, argv, background);
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      Runtime::wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_top_level_task_id(
                                                  Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      Runtime::set_top_level_task_id(top_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* HighLevelRuntime::get_reduction_op(
                                                        ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      return Runtime::get_reduction_op(redop_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_registration_callback(
                                            RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      Runtime::set_registration_callback(callback);
    }

    //--------------------------------------------------------------------------
    /*static*/ const InputArgs& HighLevelRuntime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::get_input_args();
    }

    //--------------------------------------------------------------------------
    /*static*/ HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
      return Runtime::get_runtime(p)->high_level;
    }

    //--------------------------------------------------------------------------
    /*static*/ LowLevel::ReductionOpTable& HighLevelRuntime::
                                                      get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::get_reduction_table();
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID HighLevelRuntime::
      register_region_projection_function(ProjectionID handle, void *func_ptr)
    //--------------------------------------------------------------------------
    {
      return Runtime::register_region_projection_function(handle, 
                                                                func_ptr);
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID HighLevelRuntime::
      register_partition_projection_function(ProjectionID handle, 
                                             void *func_ptr)
    //--------------------------------------------------------------------------
    {
      return Runtime::register_partition_projection_function(handle,
                                                                   func_ptr);
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID HighLevelRuntime::update_collection_table(
        LowLevelFnptr low_level_ptr, InlineFnptr inline_ptr, TaskID uid,
        Processor::Kind proc_kind, bool single_task, bool index_space_task,
        VariantID vid, size_t return_size, 
        const TaskConfigOptions &options, const char *name)
    //--------------------------------------------------------------------------
    {
      return Runtime::update_collection_table(low_level_ptr,inline_ptr,
                                                    uid,proc_kind,single_task, 
                                                    index_space_task,vid,
                                                    return_size,options,name);
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::enable_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::enable_profiling();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::disable_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::disable_profiling();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::dump_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::dump_profiling();
#endif
    }

    /////////////////////////////////////////////////////////////
    // Mapper 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void Mapper::send_message(Processor target, 
                              const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      runtime->runtime->handle_mapper_send_message(this, target, 
                                                   message, length);
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

