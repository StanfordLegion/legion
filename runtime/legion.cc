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
#include "legion_utilities.h"
#include "legion_logging.h"
#include "legion_profiling.h"
#include "legion_ops.h"
#include "region_tree.h"
#include "default_mapper.h"

// The maximum number of proces on a node
#define MAX_NUM_PROCS           1024
#define DEFAULT_MAPPER_SLOTS    8
#define DEFAULT_OPS             4
#define MAX_TASK_WINDOW         1024
#define MIN_TASKS_TO_PERFORM_SCHEDULING 1
#define MAX_FILTER_SIZE         (16*MIN_TASKS_TO_PERFORM_SCHEDULING) 

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

#ifdef LEGION_PROF
    namespace LegionProf {
      Logger::Category log_prof("legion_prof");
      ProcessorProfiler *legion_prof_table = 
        (ProcessorProfiler*)malloc((MAX_NUM_PROCS+1)*sizeof(ProcessorProfiler));
    };
#endif

    const LogicalRegion LogicalRegion::NO_REGION = LogicalRegion();
    const LogicalPartition LogicalPartition::NO_PART = LogicalPartition();

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
    LogicalRegion::LogicalRegion(RegionTreeID tid, IndexSpace index, FieldSpace field)
      : tree_id(tid), index_space(index), field_space(field)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(void)
      : tree_id(0), index_space(IndexSpace::NO_SPACE), field_space(FieldSpace::NO_SPACE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegion::LogicalRegion(const LogicalRegion &rhs)
      : tree_id(rhs.tree_id), index_space(rhs.index_space), field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Logical Partition 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPartition::LogicalPartition(RegionTreeID tid, IndexPartition pid, FieldSpace field)
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
      : tree_id(rhs.tree_id), index_partition(rhs.index_partition), field_space(rhs.field_space)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Task::Task(void)
      : task_id(0), args(NULL), arglen(0), map_id(0), tag(0),
        orig_proc(Processor::NO_PROC), steal_count(0), depth(0), 
        must_parallelism(false), is_index_space(false), 
        index_domain(Domain::NO_DOMAIN), variants(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Task::clone_task_from(Task *rhs)
    //--------------------------------------------------------------------------
    {
      this->task_id = rhs->task_id;
      this->indexes = rhs->indexes;
      this->fields = rhs->fields;
      this->regions = rhs->regions;
      if (rhs->args != NULL)
      {
        this->args = malloc(rhs->arglen);
        memcpy(this->args,rhs->args,rhs->arglen);
        this->arglen = rhs->arglen;
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      this->orig_proc = rhs->orig_proc;
      this->steal_count = rhs->steal_count;
      this->depth = rhs->depth;
      this->must_parallelism = rhs->must_parallelism;
      this->is_index_space = rhs->is_index_space;
      this->index_domain = rhs->index_domain;
      this->index_point = rhs->index_point;
      this->variants = rhs->variants;
    }

    //--------------------------------------------------------------------------
    size_t Task::compute_user_task_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = 0;  
      result += sizeof(task_id);
      result += 3*sizeof(size_t); // sizes of indexes, fields, and regions 
      result += (indexes.size() * sizeof(IndexSpaceRequirement));
      result += (fields.size() * sizeof(FieldSpaceRequirement));
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        result += regions[idx].compute_size();
      }
      result += sizeof(arglen);
      result += arglen;
      result += sizeof(map_id);
      result += sizeof(tag);
      result += sizeof(orig_proc);
      result += sizeof(steal_count);
      result += sizeof(depth);
      result += sizeof(is_index_space);
      if (is_index_space)
      {
        result += sizeof(must_parallelism);
        result += sizeof(index_domain);
        result += sizeof(index_point);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void Task::pack_user_task(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<Processor::TaskFuncID>(task_id); 
      rez.serialize<size_t>(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        rez.serialize<IndexSpaceRequirement>(indexes[idx]);
      }
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        rez.serialize<FieldSpaceRequirement>(fields[idx]);
      }
      rez.serialize<size_t>(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        regions[idx].pack_requirement(rez);
      }
      rez.serialize<size_t>(arglen);
      rez.serialize(args,arglen);
      rez.serialize<MapperID>(map_id);
      rez.serialize<MappingTagID>(tag);
      rez.serialize<Processor>(orig_proc);
      rez.serialize<unsigned>(steal_count);
      rez.serialize<unsigned>(depth);
      rez.serialize<bool>(is_index_space);
      if (is_index_space)
      {
        rez.serialize<bool>(must_parallelism);
        rez.serialize<Domain>(index_domain);
        rez.serialize<DomainPoint>(index_point);
      }
    }
    
    //--------------------------------------------------------------------------
    void Task::unpack_user_task(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize<Processor::TaskFuncID>(task_id);
      size_t num_items;
      derez.deserialize<size_t>(num_items);
      indexes.resize(num_items);
      for (unsigned idx = 0; idx < num_items; idx++)
      {
        derez.deserialize<IndexSpaceRequirement>(indexes[idx]);
      }
      derez.deserialize<size_t>(num_items);
      fields.resize(num_items);
      for (unsigned idx = 0; idx < num_items; idx++)
      {
        derez.deserialize<FieldSpaceRequirement>(fields[idx]);
      }
      derez.deserialize<size_t>(num_items);
      regions.resize(num_items);
      for (unsigned idx = 0; idx < num_items; idx++)
      {
        regions[idx].unpack_requirement(derez);
      }
      derez.deserialize<size_t>(arglen);
      if (arglen > 0)
      {
        args = malloc(arglen);
        derez.deserialize(args,arglen);
      }
      derez.deserialize<MapperID>(map_id);
      derez.deserialize<MappingTagID>(tag);
      derez.deserialize<Processor>(orig_proc);
      derez.deserialize<unsigned>(steal_count);
      derez.deserialize<unsigned>(depth);
      derez.deserialize<bool>(is_index_space);
      if (is_index_space)
      {
        derez.deserialize<bool>(must_parallelism);
        derez.deserialize<Domain>(index_domain);
        derez.deserialize<DomainPoint>(index_point);
      }
      variants = HighLevelRuntime::find_collection(task_id);
    }

    /////////////////////////////////////////////////////////////
    // Task Variant Collection
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool TaskVariantCollection::has_variant(Processor::Kind kind, bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && (it->second.index_space == index_space))
        {
          return true;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    VariantID TaskVariantCollection::get_variant(Processor::Kind kind, bool index_space)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && (it->second.index_space == index_space))
        {
          return it->first;
        }
      }
      log_variant(LEVEL_ERROR,"User task %s (ID %d) has no registered variants for "
          "processors of kind %d and index space %d",name, user_id, kind, index_space);
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
    const TaskVariantCollection::Variant& TaskVariantCollection::get_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(variants.find(vid) != variants.end());
#endif
      return variants[vid];
    }

    //--------------------------------------------------------------------------
    void TaskVariantCollection::add_variant(Processor::TaskFuncID low_id, Processor::Kind kind, bool index)
    //--------------------------------------------------------------------------
    {
      // for right now we'll make up our own VariantID
      VariantID vid = variants.size();
      variants[vid] = Variant(low_id, kind, index);
    }

    //--------------------------------------------------------------------------
    const TaskVariantCollection::Variant& TaskVariantCollection::select_variant(bool index, Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      for (std::map<VariantID,Variant>::const_iterator it = variants.begin();
            it != variants.end(); it++)
      {
        if ((it->second.proc_kind == kind) && (it->second.index_space == index))
        {
          return it->second;
        }
      }
      log_variant(LEVEL_ERROR,"User task %s (ID %d) has no registered variants for "
          "processors of kind %d and index space %d",name, user_id, kind, index);
#ifdef DEBUG_HIGH_LEVEL
      assert(false);
#endif
      exit(ERROR_UNREGISTERED_VARIANT);
      return variants[0];
    }

    /////////////////////////////////////////////////////////////
    // Argument Map 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMap::ArgumentMap(ArgumentMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
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
      {
        impl->add_reference();
      }
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
    Future::Future(const Future &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    Future::Future(FutureImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Future::~Future()
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Future& Future::operator=(const Future &f)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      this->impl = f.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
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
    FutureMap::FutureMap(const FutureMap &f)
      : impl(f.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    FutureMap::FutureMap(FutureMapImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    FutureMap::~FutureMap()
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap& FutureMap::operator=(const FutureMap &f)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      this->impl = f.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
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
    IndexSpaceRequirement::IndexSpaceRequirement(IndexSpace _handle, AllocateMode _priv,
                                                 IndexSpace _parent, bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), parent(_parent), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool IndexSpaceRequirement::operator<(const IndexSpaceRequirement &rhs) const
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
    bool IndexSpaceRequirement::operator==(const IndexSpaceRequirement &rhs) const
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
    FieldSpaceRequirement::FieldSpaceRequirement(FieldSpace _handle, AllocateMode _priv,
                                                  bool _verified /*=false*/)
      : handle(_handle), privilege(_priv), verified(_verified)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool FieldSpaceRequirement::operator<(const FieldSpaceRequirement &rhs) const
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
    bool FieldSpaceRequirement::operator==(const FieldSpaceRequirement &rhs) const
    //--------------------------------------------------------------------------
    {
      return (handle == rhs.handle) && (privilege == rhs.privilege) && (verified == rhs.verified);
    }

    /////////////////////////////////////////////////////////////
    // Region Requirement 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, 
                                        const std::set<FieldID> &priv_fields,
                                        const std::vector<FieldID> &inst_fields,
                                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
					 MappingTagID _tag, bool _verified, TypeHandle _inst)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(SINGULAR), projection(0), inst_type(_inst)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified,
                TypeHandle _inst)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(PART_PROJECTION), projection(_proj), inst_type(_inst)
    //--------------------------------------------------------------------------
    { 
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                const std::set<FieldID> &priv_fields,
                const std::vector<FieldID> &inst_fields,
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified,
                TypeHandle _inst)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false),
        handle_type(REG_PROJECTION), projection(_proj), inst_type(_inst)
    //--------------------------------------------------------------------------
    {
      privilege_fields = priv_fields;
      instance_fields = inst_fields;
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
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
                                    LogicalRegion _parent, MappingTagID _tag, bool _verified,
                                    TypeHandle _inst)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(SINGULAR), inst_type(_inst)
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
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj,  
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, bool _verified,
                        TypeHandle _inst)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(PART_PROJECTION), projection(_proj), inst_type(_inst)
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
    RegionRequirement::RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        const std::set<FieldID> &priv_fields,
                        const std::vector<FieldID> &inst_fields,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, bool _verified,
                        TypeHandle _inst)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false),
        handle_type(REG_PROJECTION), projection(_proj), inst_type(_inst)
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
                                        PrivilegeMode _priv, CoherenceProperty _prop, LogicalRegion _parent,
					 MappingTagID _tag, bool _verified, TypeHandle _inst)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(SINGULAR), inst_type(_inst)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this)) // Shouldn't use this constructor for reductions
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj, 
                PrivilegeMode _priv, CoherenceProperty _prop,
                LogicalRegion _parent, MappingTagID _tag, bool _verified,
                TypeHandle _inst)
      : partition(pid), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false), 
        handle_type(PART_PROJECTION), projection(_proj), inst_type(_inst)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                  PrivilegeMode _priv, CoherenceProperty _prop,
                  LogicalRegion _parent, MappingTagID _tag, bool _verified,
                  TypeHandle _inst)
      : region(_handle), privilege(_priv), prop(_prop), parent(_parent),
        redop(0), tag(_tag), verified(_verified), sanitized(false),
        handle_type(REG_PROJECTION), projection(_proj), inst_type(_inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(*this))
      {
        log_region(LEVEL_ERROR,"ERROR: Use different RegionRequirement constructor for reductions");
        assert(false);
        exit(ERROR_USE_REDUCTION_REGION_REQ);
      }
#endif
    }

    //--------------------------------------------------------------------------
    RegionRequirement::RegionRequirement(LogicalRegion _handle,  
                                    ReductionOpID op, CoherenceProperty _prop, 
                                    LogicalRegion _parent, MappingTagID _tag, bool _verified,
                                    TypeHandle _inst)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false),
        handle_type(SINGULAR), inst_type(_inst)
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
    RegionRequirement::RegionRequirement(LogicalPartition pid, ProjectionID _proj,  
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, bool _verified,
                        TypeHandle _inst)
      : partition(pid), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false),
        handle_type(PART_PROJECTION), projection(_proj), inst_type(_inst)
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
    RegionRequirement::RegionRequirement(LogicalRegion _handle, ProjectionID _proj,
                        ReductionOpID op, CoherenceProperty _prop,
                        LogicalRegion _parent, MappingTagID _tag, bool _verified,
                        TypeHandle _inst)
      : region(_handle), privilege(REDUCE), prop(_prop), parent(_parent),
        redop(op), tag(_tag), verified(_verified), sanitized(false),
        handle_type(REG_PROJECTION), projection(_proj), inst_type(_inst)
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
          (prop == rhs.prop) && (parent == rhs.parent) && (redop == rhs.redop) &&
          (tag == rhs.tag) && (verified == rhs.verified) && 
          (sanitized == rhs.sanitized) && (inst_type == rhs.inst_type))
      {
        if (((handle_type == SINGULAR) && (region == rhs.region)) ||
            ((handle_type == PART_PROJECTION) && (partition == rhs.partition) && (projection == rhs.projection)) ||
            ((handle_type == REG_PROJECTION) && (region == rhs.region)))
        {
          if ((privilege_fields.size() == rhs.privilege_fields.size()) &&
              (instance_fields.size() == rhs.instance_fields.size()))
          {
            return ((privilege_fields == rhs.privilege_fields) && (instance_fields == rhs.instance_fields));
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
                  if (verified < rhs.verified)
                    return true;
                  else if (verified > rhs.verified)
                    return false;
                  else
                  {
                    if (sanitized < rhs.sanitized)
                      return true;
                    else if (sanitized > rhs.sanitized)
                      return false;
                    else
                    {
                      if (inst_type < rhs.inst_type)
                        return true;
                      else if (inst_type > rhs.inst_type)
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
                              else if (partition != rhs.partition) // therefore greater than
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
      }
    }

    //--------------------------------------------------------------------------
    RegionRequirement& RegionRequirement::operator=(const RegionRequirement &rhs)
    //--------------------------------------------------------------------------
    {
      if ((rhs.handle_type == SINGULAR) || (rhs.handle_type == REG_PROJECTION))
        region = rhs.region;
      else
        partition = rhs.partition;
      privilege_fields = rhs.privilege_fields;
      instance_fields = rhs.instance_fields;
      privilege = rhs.privilege;
      prop = rhs.prop;
      parent = rhs.parent;
      redop = rhs.redop;
      tag = rhs.tag;
      verified = rhs.verified;
      sanitized = rhs.sanitized;
      handle_type = rhs.handle_type;
      projection = rhs.projection;
      inst_type = rhs.inst_type;
      return *this;
    }

    //--------------------------------------------------------------------------
    size_t RegionRequirement::compute_size(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      if ((handle_type == SINGULAR) || (handle_type == REG_PROJECTION))
        result += sizeof(this->region);
      else
        result += sizeof(this->partition);
      result += 2*sizeof(size_t); // size of privilege and instance field vectors
      result += ((privilege_fields.size() + instance_fields.size()) * sizeof(FieldID));
      result += sizeof(this->privilege);
      result += sizeof(this->prop);
      result += sizeof(this->parent);
      result += sizeof(this->redop);
      result += sizeof(this->tag);
      result += sizeof(this->verified);
      result += sizeof(this->sanitized);
      result += sizeof(this->handle_type);
      result += sizeof(this->projection);
      result += sizeof(this->inst_type);
      return result;
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::pack_requirement(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(this->handle_type);
      if ((handle_type == SINGULAR) || (handle_type == REG_PROJECTION))
        rez.serialize(this->region);
      else
        rez.serialize(this->partition);
      rez.serialize<size_t>(privilege_fields.size());
      for (std::set<FieldID>::const_iterator it = privilege_fields.begin();
            it != privilege_fields.end(); it++)
      {
        rez.serialize<FieldID>(*it);
      }
      rez.serialize<size_t>(instance_fields.size());
      for (std::vector<FieldID>::const_iterator it = instance_fields.begin();
            it != instance_fields.end(); it++)
      {
        rez.serialize<FieldID>(*it);
      }
      rez.serialize(this->privilege);
      rez.serialize(this->prop);
      rez.serialize(this->parent);
      rez.serialize(this->redop);
      rez.serialize(this->tag);
      rez.serialize(this->verified);
      rez.serialize(this->sanitized);
      rez.serialize(this->projection);
      rez.serialize(this->inst_type);
    }

    //--------------------------------------------------------------------------
    void RegionRequirement::unpack_requirement(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(this->handle_type);
      if ((handle_type == SINGULAR) || (handle_type == REG_PROJECTION))
        derez.deserialize(this->region);
      else
        derez.deserialize(this->partition);
      size_t num_elmts;
      derez.deserialize<size_t>(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        FieldID temp;
        derez.deserialize<FieldID>(temp);
        privilege_fields.insert(temp);
      }
      derez.deserialize<size_t>(num_elmts);
      instance_fields.resize(num_elmts);
      for (unsigned idx = 0; idx < num_elmts; idx++)
      {
        derez.deserialize<FieldID>(instance_fields[idx]);
      }
      derez.deserialize(this->privilege);
      derez.deserialize(this->prop);
      derez.deserialize(this->parent);
      derez.deserialize(this->redop);
      derez.deserialize(this->tag);
      derez.deserialize(this->verified);
      derez.deserialize(this->sanitized);
      derez.deserialize(this->projection);
      derez.deserialize(this->inst_type);
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
          return ACCESSOR_ALL;
        case WRITE_ONLY:
          return ACCESSOR_WRITE;
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

    /////////////////////////////////////////////////////////////
    // Physical Region 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(void)
      : is_impl(false), map_set(false), accessor_map(0), gen_id(0) // note this is an invalid configuration
    //--------------------------------------------------------------------------
    {
      op.map = NULL;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(PhysicalRegionImpl *i)
      : is_impl(true), map_set(false), accessor_map(0), gen_id(0)
    //--------------------------------------------------------------------------
    {
      op.impl = i;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion::PhysicalRegion(MappingOperation *map_op, GenerationID id)
      : is_impl(false), map_set(false), accessor_map(0), gen_id(id)
    //--------------------------------------------------------------------------
    {
      op.map = map_op;
    }

    //-------------------------------------------------------------------------- 
    PhysicalRegion::PhysicalRegion(const PhysicalRegion &rhs)
      : is_impl(rhs.is_impl), map_set(rhs.map_set), accessor_map(rhs.accessor_map)
    //--------------------------------------------------------------------------
    {
      if (is_impl)
        op.impl = rhs.op.impl;
      else
      {
        op.map = rhs.op.map;
        gen_id = rhs.gen_id;
      }
    }

    //--------------------------------------------------------------------------
    PhysicalRegion& PhysicalRegion::operator=(const PhysicalRegion &rhs)
    //--------------------------------------------------------------------------
    {
      this->is_impl = rhs.is_impl;
      if (this->is_impl)
        op.impl = rhs.op.impl;
      else
      {
        op.map = rhs.op.map;
        gen_id = rhs.gen_id;
      }
      this->map_set = rhs.map_set;
      this->accessor_map = rhs.accessor_map;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::operator==(const PhysicalRegion &reg) const
    //--------------------------------------------------------------------------
    {
      if (is_impl != reg.is_impl)
        return false;
      if (is_impl)
        return (op.impl == reg.op.impl);
      else
        return ((op.map == reg.op.map) && (gen_id == reg.gen_id));
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::operator<(const PhysicalRegion &reg) const
    //--------------------------------------------------------------------------
    {
      if (is_impl < reg.is_impl)
        return true;
      else if (is_impl > reg.is_impl)
        return false;
      else
      {
        if (is_impl)
          return (op.impl < reg.op.impl);
        else
        {
          if (op.map < reg.op.map)
            return true;
          else if (op.map > reg.op.map)
            return false;
          else
            return (gen_id < reg.gen_id);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalRegion::wait_until_valid(void)
    //--------------------------------------------------------------------------
    {
      if (!is_impl)
        op.map->wait_until_valid(gen_id);
      // else it's a physical region from a task and is already valid
    }

    //--------------------------------------------------------------------------
    bool PhysicalRegion::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (!is_impl)
        return op.map->is_valid(gen_id);
      return true; // else it's a task in which case it's already valid
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegion::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
      if (is_impl)
        return op.impl->get_logical_region();
      else
        return op.map->get_logical_region(gen_id);
    }

#ifdef OLD_STUFF
    //--------------------------------------------------------------------------
    bool PhysicalRegion::has_accessor(AccessorType at) const
    //--------------------------------------------------------------------------
    {
      // if we haven't computed the map yet, do it
      if (!map_set)
      {
        PhysicalInstance inst = PhysicalInstance::NO_INST;
        if (is_impl)
          inst = op.impl->get_physical_instance();
        else
          inst = op.map->get_physical_instance(gen_id);
#ifdef DEBUG_HIGH_LEVEL
        assert(inst.exists());
#endif
        LowLevel::RegionAccessor<LowLevel::AccessorGeneric> generic = 
          inst.get_accessor();
#define SET_MASK(AT) accessor_map |= (generic.can_convert<LowLevel::AT>() ? AT : 0)
        SET_MASK(AccessorGeneric);
        //SET_MASK(AccessorArray);
        //SET_MASK(AccessorArrayReductionFold);
        // FIXME(Elliott): Shared low-level doesn't have these
        //SET_MASK(AccessorGPU);
        //SET_MASK(AccessorGPUReductionFold);
        //SET_MASK(AccessorReductionList);
#undef SET_MASK
        map_set = true;
      }
      return ((at & accessor_map) != 0);
    }
#endif

    Accessor::RegionAccessor<Accessor::AccessorType::Generic> PhysicalRegion::get_accessor(void) const
    {
      if (is_impl)
        return op.impl->get_accessor();
      else
        return op.map->get_accessor(gen_id);
    }

#ifdef DEBUG_HIGH_LEVEL
#define GET_ACCESSOR_IMPL(AT)                                                     \
    template<>                                                                    \
    Accessor::RegionAccessor<Accessor::AccessorType::AT> PhysicalRegion::get_accessor<AT>(void) const \
    {                                                                             \
      bool has_access = has_accessor(AT);                                         \
      if (!has_access)                                                            \
      {                                                                           \
        log_run(LEVEL_ERROR,"Physical region does not have an accessor of type %d\n",AT); \
        exit(ERROR_INVALID_ACCESSOR_REQUESTED);                                   \
      }                                                                           \
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> generic;                \
      if (is_impl)                                                                \
        generic = op.impl->get_accessor();                                        \
      else                                                                        \
        generic = op.map->get_accessor(gen_id);                                   \
      return generic.convert<LowLevel::AT>();                                     \
    }
#else // DEBUG_HIGH_LEVEL
#define GET_ACCESSOR_IMPL(AT)                                                     \
    template<>                                                                    \
    LowLevel::RegionAccessor<LowLevel::AT> PhysicalRegion::get_accessor<AT>(void) const \
    {                                                                             \
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> generic;                \
      if (is_impl)                                                                \
        generic = op.impl->get_accessor();                                        \
      else                                                                        \
        generic = op.map->get_accessor(gen_id);                                   \
      return generic.convert<LowLevel::AT>();                                     \
    }
#endif
    // GET_ACCESSOR_IMPL(AccessorGeneric)
    // GET_ACCESSOR_IMPL(AccessorArray)
    // GET_ACCESSOR_IMPL(AccessorArrayReductionFold)
    // GET_ACCESSOR_IMPL(AccessorGPU)
    // GET_ACCESSOR_IMPL(AccessorGPUReductionFold)
    // GET_ACCESSOR_IMPL(AccessorReductionList)
#undef GET_ACCESSOR_IMPL

    Accessor::RegionAccessor<Accessor::AccessorType::Generic> PhysicalRegion::get_field_accessor(FieldID fid) const
    {
      if (is_impl)
        return op.impl->get_field_accessor(fid);
      else
        return op.map->get_field_accessor(gen_id, fid);
    }

#ifdef DEBUG_HIGH_LEVEL
#define GET_FIELD_ACCESSOR_IMPL(AT)                                                       \
    template<>                                                                            \
    LowLevel::RegionAccessor<LowLevel::AT> PhysicalRegion::get_accessor<AT>(FieldID fid) const  \
    {                                                                                     \
      bool has_access = has_accessor(AT);                                                 \
      if (!has_access)                                                                    \
      {                                                                                   \
        log_run(LEVEL_ERROR,"Physical region does not have an accessor of type %d\n",AT); \
        exit(ERROR_INVALID_ACCESSOR_REQUESTED);                                           \
      }                                                                                   \
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> generic;                        \
      if (is_impl)                                                                        \
        generic = op.impl->get_field_accessor(fid);                                       \
      else                                                                                \
        generic = op.map->get_field_accessor(gen_id, fid);                                \
      return generic.convert<LowLevel::AT>();                                             \
    }
#else
#define GET_FIELD_ACCESSOR_IMPL(AT)                                                       \
    template<>                                                                            \
    LowLevel::RegionAccessor<LowLevel::AT> PhysicalRegion::get_accessor<AT>(FieldID fid) const \
    {                                                                                     \
      LowLevel::RegionAccessor<LowLevel::AccessorGeneric> generic;                        \
      if (is_impl)                                                                        \
        generic = op.impl->get_field_accessor(fid);                                       \
      else                                                                                \
        generic = op.map->get_field_accessor(gen_id, fid);                                \
      return generic.convert<LowLevel::AT>();                                             \
    }
#endif
    //GET_FIELD_ACCESSOR_IMPL(AccessorGeneric)
    //GET_FIELD_DACCESSOR_IMPL(AccessorArray)
    //GET_FIELD_ACCESSOR_IMPL(AccessorArrayReductionFold)
    //GET_FIELD_ACCESSOR_IMPL(AccessorGPU)
    //GET_FIELD_ACCESSOR_IMPL(AccessorGPUReductionFold)
    //GET_FIELD_ACCESSOR_IMPL(AccessorReductionList)
#undef GET_FIELD_ACCESSOR_IMPL

    /////////////////////////////////////////////////////////////
    // Index Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(void)
      : impl (NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(const IndexAllocator &rhs)
      : impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    IndexAllocator::IndexAllocator(IndexAllocatorImpl *i)
      : impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    IndexAllocator::~IndexAllocator(void)
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
    IndexAllocator& IndexAllocator::operator=(const IndexAllocator &rhs)
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
      // Add our reference tot he new impl if it isn't NULL
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Field Allocator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(void)
      : space(FieldSpace::NO_SPACE), parent(NULL), runtime(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(const FieldAllocator &allocator)
      : space(allocator.space), parent(allocator.parent), runtime(allocator.runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldAllocator::FieldAllocator(FieldSpace f, Context p, HighLevelRuntime *rt)
      : space(f), parent(p), runtime(rt)
    //--------------------------------------------------------------------------
    {
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
    IndexIterator::IndexIterator(const LogicalRegion &handle)
      : enumerator(handle.get_index_space().get_valid_mask().enumerate_enabled())
    //--------------------------------------------------------------------------
    {
      finished = !(enumerator->get_next(current_pointer,remaining_elmts));
    }

    //--------------------------------------------------------------------------
    IndexIterator::~IndexIterator(void)
    //--------------------------------------------------------------------------
    {
      delete enumerator;
    }

    /////////////////////////////////////////////////////////////
    // ColoringSerializer 
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
    // DomainColoringSerializer 
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
    // Lockable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Lockable::Lockable(void)
#ifdef LOW_LEVEL_LOCKS
      : base_lock(Lock::create_lock())
#else
      : base_lock(ImmovableLock(true/*initialize*/))
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Lockable::~Lockable(void)
    //--------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      base_lock.destroy_lock();
#else
      base_lock.destroy();
#endif
    }


    /////////////////////////////////////////////////////////////
    // Collectable  
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Collectable::Collectable(unsigned init /*= 0*/)
      : references(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void Collectable::add_reference(unsigned cnt /*= 1*/, bool need_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
        lock();
      references += cnt;
      if (need_lock)
        unlock();
    }

    //--------------------------------------------------------------------------
    bool Collectable::remove_reference(unsigned cnt /*= 1*/, bool need_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
        lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(references >= cnt);
#endif
      references -= cnt;
      bool result = (references == 0);
      if (need_lock)
        unlock();
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Mappable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Mappable::Mappable(void)
      : mapper(NULL)
    //--------------------------------------------------------------------------
    {
    }
    
#ifdef LOW_LEVEL_LOCKS
    //--------------------------------------------------------------------------
    void Mappable::initialize_mappable(Mapper *map, Lock m_lock)
    //--------------------------------------------------------------------------
    {
      mapper = map;
      mapper_lock = m_lock;
    }
#else
    //--------------------------------------------------------------------------
    void Mappable::initialize_mappable(Mapper *map, ImmovableLock m_lock)
    //--------------------------------------------------------------------------
    {
      mapper = map;
      mapper_lock = m_lock;
    }
#endif

    /////////////////////////////////////////////////////////////
    // Physical Region Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(void)
      : valid(false), idx(0), handle(LogicalRegion::NO_REGION), 
        manager(NULL), req(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(unsigned id, LogicalRegion h,
                                            PhysicalManager *man,
                                            const RegionRequirement *r)
      : valid(true), idx(id), handle(h), manager(man), req(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl::PhysicalRegionImpl(const PhysicalRegionImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalRegionImpl& PhysicalRegionImpl::operator=(const PhysicalRegionImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LogicalRegion PhysicalRegionImpl::get_logical_region(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!valid)
      {
        log_region(LEVEL_ERROR,"Accessing invalidated mapping for task region %d",idx);
        assert(false);
        exit(ERROR_INVALID_MAPPING_ACCESS);
      }
#endif
      return handle;
    }

    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalRegionImpl::get_physical_instance(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      if (!valid)
      {
        log_region(LEVEL_ERROR,"Accessing invalidated mapping for task region %d",idx);
        assert(false);
        exit(ERROR_INVALID_MAPPING_ACCESS);
      }
#endif
      return manager->get_instance();
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> PhysicalRegionImpl::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> result = manager->get_accessor();
#ifdef PRIVILEGE_CHECKS
      result.set_privileges_untyped(req->get_accessor_privilege());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Accessor::RegionAccessor<Accessor::AccessorType::Generic> PhysicalRegionImpl::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      Accessor::RegionAccessor<Accessor::AccessorType::Generic> result = manager->get_field_accessor(fid);
#ifdef DEBUG_HIGH_LEVEL
      if (!req->has_field_privilege(fid))
      {
        log_run(LEVEL_ERROR,"Invalid request for field accessor without privileges for field %d", fid);
        assert(false);
        exit(ERROR_INVALID_FIELD_ACCESSOR_PRIVILEGES);
      }
#endif
#ifdef PRIVILEGE_CHECKS
      assert(req->has_field_privilege(fid));
      result.set_privileges_untyped(req->get_accessor_privilege());
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalRegionImpl::invalidate(void)
    //--------------------------------------------------------------------------
    {
      valid = false;
    }

    /////////////////////////////////////////////////////////////
    // Index Allocator Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexAllocatorImpl::IndexAllocatorImpl(IndexSpaceAllocator alloc)
      : allocator(alloc)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexAllocatorImpl::~IndexAllocatorImpl(void)
    //--------------------------------------------------------------------------
    {
      // Need a copy because of dumb C++
      IndexSpaceAllocator copy = allocator;
      copy.destroy();
    }

    /////////////////////////////////////////////////////////////
    // Argument Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(ArgumentMapStore *st)
      : next(NULL), store(st), frozen(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(const ArgumentMapImpl &rhs)
      : next(NULL), store(NULL), frozen(false)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::~ArgumentMapImpl(void)
    //--------------------------------------------------------------------------
    {
      // Remove our reference to the next thing in the list
      // and delete it if we're done with it
      if (next != NULL)
      {
        if (next->remove_reference())
        {
          delete next;
        }
      }
      else
      {
        // We're the last one in the list being deleted, so 
        // delete the store as well
        delete store;
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl& ArgumentMapImpl::operator=(const ArgumentMapImpl &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMapImpl::find_point(const AnyPoint &point) const
    //--------------------------------------------------------------------------
    {
      // Good property to have here: if we don't find the point, we just
      // return an empty task argument which is the right answer.
      TaskArgument result;
      for (std::map<AnyPoint,TaskArgument>::const_iterator it = arguments.begin();
            it != arguments.end(); it++)
      {
        if (point.equals(it->first))
        {
          result = it->second;
          break;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl* ArgumentMapImpl::freeze(void)
    //--------------------------------------------------------------------------
    {
      if (next == NULL)
      {
        frozen = true;
        return this;
      }
      else
      {
        return next->freeze();
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl* ArgumentMapImpl::clone(void) const
    //--------------------------------------------------------------------------
    {
      ArgumentMapImpl *new_impl = new ArgumentMapImpl(store);
      new_impl->arguments = this->arguments; 
      return new_impl;
    }

    //--------------------------------------------------------------------------
    size_t ArgumentMapImpl::compute_arg_map_size(void)
    //--------------------------------------------------------------------------
    {
      size_t result = sizeof(size_t); // number of arguments 
      // Element and dimension sizes for the any points and the buffer size for the argument
      result += (arguments.size() * (2*sizeof(size_t) + sizeof(unsigned)));
      for (std::map<AnyPoint,TaskArgument>::const_iterator it = arguments.begin();
            it != arguments.end(); it++)
      {
        const AnyPoint &point = it->first;
        result += (point.elmt_size * point.dim);
        const TaskArgument &arg = it->second;
        result += (arg.get_size());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::pack_arg_map(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(arguments.size());
      for (std::map<AnyPoint,TaskArgument>::const_iterator it = arguments.begin();
            it != arguments.end(); it++)
      {
        const AnyPoint &point = it->first;
        rez.serialize<size_t>(point.elmt_size);
        rez.serialize<unsigned>(point.dim);
        rez.serialize(point.buffer,point.elmt_size*point.dim);
        const TaskArgument &arg = it->second;
        rez.serialize<size_t>(arg.get_size());
        rez.serialize(arg.get_ptr(),arg.get_size());
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::unpack_arg_map(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_points;
      derez.deserialize<size_t>(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        AnyPoint point = store->add_point(derez);
        arguments[point] = store->add_arg(derez);
      }
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
      // Go through and delete all the memory that we own
      for (std::set<AnyPoint>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        free(const_cast<void*>(it->buffer));
      }
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
    AnyPoint ArgumentMapStore::add_point(size_t elmt_size, unsigned dim, const void *buffer)
    //--------------------------------------------------------------------------
    {
      // Don't bother de-duplicating points here, the ArgumentMapImpls will
      // do a good job of that to begin with and its really just not worth
      // the extra computation overhead since its not that much memory anyway
      void *new_buffer = malloc(elmt_size*dim);
      memcpy(new_buffer, buffer, elmt_size*dim);
      AnyPoint new_point(new_buffer, elmt_size, dim);
      points.insert(new_point);
      return new_point;
    }

    //--------------------------------------------------------------------------
    AnyPoint ArgumentMapStore::add_point(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t elmt_size;
      derez.deserialize<size_t>(elmt_size);
      unsigned dim;
      derez.deserialize<unsigned>(dim);
      void *buffer = malloc(elmt_size * dim);
      derez.deserialize(buffer, elmt_size * dim);
      AnyPoint new_point(buffer, elmt_size, dim);
      points.insert(new_point);
      return new_point;
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

    //--------------------------------------------------------------------------
    TaskArgument ArgumentMapStore::add_arg(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t buf_size;
      derez.deserialize<size_t>(buf_size);
      void *buffer = malloc(buf_size);
      derez.deserialize(buffer,buf_size);
      TaskArgument new_arg(buffer,buf_size);
      values.insert(new_arg);
      return new_arg;
    }

    /////////////////////////////////////////////////////////////
    // Any Point 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool AnyPoint::equals(const AnyPoint &other) const
    //--------------------------------------------------------------------------
    {
      if (buffer == other.buffer)
        return true;
      if ((elmt_size != other.elmt_size) || (dim != other.dim))
        return false;
      return (memcmp(buffer,other.buffer,elmt_size*dim)==0);
    }

    /////////////////////////////////////////////////////////////
    // Future Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureImpl::FutureImpl(HighLevelRuntime *rt, Processor owner,
                           Event set_e /*= Event::NO_EVENT*/)
      : Collectable(), set_event(set_e), result(NULL), is_set(false),
        runtime(rt), owner_proc(owner)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureImpl::~FutureImpl(void)
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
      {
        free(result);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(waiters.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *res, size_t result_size, bool owner)
    //--------------------------------------------------------------------------
    {
      lock();
      if (!owner)
      {
        result = malloc(result_size); 
#ifdef DEBUG_HIGH_LEVEL
        if (result_size > 0)
        {
          assert(res != NULL);
          assert(result != NULL);
        }
#endif
        memcpy(result, res, result_size);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        if (result_size > 0)
          assert(res != NULL);
#endif
        result = const_cast<void*>(res);
      }
      is_set = true;
      unlock();
      notify_all_waiters();
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(const void *res, size_t result_size, Event ready_event, bool owner)
    //--------------------------------------------------------------------------
    {
      set_event = ready_event;
      set_result(res, result_size, owner);
    }

    //--------------------------------------------------------------------------
    void FutureImpl::set_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      lock();
      size_t result_size;
      derez.deserialize<size_t>(result_size);
      result = malloc(result_size);
#ifdef DEBUG_HIGH_LEVEL
      assert(!set_event.has_triggered());
      if (result_size > 0)
      {
        assert(result != NULL);
      }
#endif
      derez.deserialize(result,result_size);
      is_set = true;
      unlock();
      notify_all_waiters();
    }

    //--------------------------------------------------------------------------
    bool FutureImpl::register_waiter(Notifiable *waiter, bool &value)
    //--------------------------------------------------------------------------
    {
      lock();
      bool result = !is_set;
      if (!is_set)
      {
        waiters.push_back(waiter);
      }
      else
      {
        value = *((bool*)result);
      }
      unlock();
      return result;
    }
    
    //--------------------------------------------------------------------------
    void FutureImpl::notify_all_waiters(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_set);
#endif
      for (std::vector<Notifiable*>::iterator it = waiters.begin();
            it != waiters.end(); it++)
      {
        bool value = *((bool*)result);
        if ((*it)->notify(value))
        {
          delete *it;
        }
      }
      waiters.clear();
    }

    /////////////////////////////////////////////////////////////
    // Future Map Impl 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureMapImpl::FutureMapImpl(HighLevelRuntime *rt, Processor owner, 
                                 Event set_e /*= Event::NO_EVENT*/)
      : all_set_event(set_e), runtime(rt), owner_proc(owner)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureMapImpl::~FutureMapImpl(void)
    //--------------------------------------------------------------------------
    {
      // We only need to clear out the valid results since we
      // know that we don't own the points in the outstanding waits
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        free(const_cast<void*>(it->first.buffer));
        // Release the reference on the future impl and see if we're the last reference
        if (it->second->remove_reference())
        {
          // Delete the future impl
          delete it->second;
        }
      }
      futures.clear();
      waiter_events.clear();
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(AnyPoint point, const void *res, size_t result_size, 
                                   Event point_finish, bool owner)
    //--------------------------------------------------------------------------
    {
      // Go through and see if we can find the future
      lock();
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(point))
        {
          // Match get the impl and set the result
          FutureImpl *impl = it->second;
          // See if there is an event we need to trigger or not
          UserEvent ready_event;
          bool has_ready_event = false;
          std::map<FutureImpl*,UserEvent>::iterator finder = waiter_events.find(impl);
          if (finder != waiter_events.end())
          {
            has_ready_event = true;
            ready_event = finder->second;
#ifdef DEBUG_HIGH_LEVEL
            assert(ready_event.exists());
#endif
            // Remove it from the map
            waiter_events.erase(finder);
          }
          // Don't need to be holding the lock when doing this
          unlock();
          impl->set_result(res, result_size, point_finish, owner);
          // Then trigger the waiting event 
          if (has_ready_event)
            ready_event.trigger();
          // Now we're done
          return;
        }
      }
      // Otherwise it wasn't here yet, so make a new point
      // and a new FutureImpl and set everything up
      // Copy the point buffer
      void * point_buffer = malloc(point.elmt_size * point.dim);
      memcpy(point_buffer,point.buffer,point.elmt_size * point.dim);
      AnyPoint p(point_buffer,point.elmt_size,point.dim);
      FutureImpl *impl = new FutureImpl(runtime, owner_proc, point_finish);
      impl->add_reference();
      impl->set_result(res, result_size, point_finish, owner);
      futures[p] = impl;
      // Unlock since we're done now
      unlock();
    }

    //--------------------------------------------------------------------------
    void FutureMapImpl::set_result(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Unpack the point
      size_t elmt_size;
      derez.deserialize<size_t>(elmt_size);
      unsigned dim;
      derez.deserialize<unsigned>(dim);
      void * point_buffer = malloc(elmt_size * dim);
      derez.deserialize(point_buffer,elmt_size * dim);
      Event ready_event;
      derez.deserialize(ready_event);
      AnyPoint point(point_buffer,elmt_size,dim);
      // Go through and see if we can find the future
      lock();
      for (std::map<AnyPoint,FutureImpl*>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        if (it->first.equals(point))
        {
          // Match get the impl and set the result
          FutureImpl *impl = it->second;
          // There better have been a user event too
#ifdef DEBUG_HIGH_LEVEL
          assert(waiter_events.find(impl) != waiter_events.end());
#endif
          UserEvent ready_event = waiter_events[impl];
          // Remove it from the map
          waiter_events.erase(impl);
          // Don't need to be holding the lock when doing this
          unlock();
          impl->set_result(point_buffer,elmt_size*dim,ready_event);
          // Then trigger the waiting event 
          ready_event.trigger();
          // We can also free the point since we didn't need it
          free(point_buffer);
          return;
        }
      }
      // Otherwise it didn't exist yet, so make it
      FutureImpl *impl = new FutureImpl(runtime, owner_proc);
      impl->add_reference();
      impl->set_result(point_buffer,elmt_size*dim,ready_event);
      futures[point] = impl;
      // Unlock since we're done now
      unlock();
    }

    /////////////////////////////////////////////////////////////
    // Predicate 
    /////////////////////////////////////////////////////////////

    const Predicate Predicate::TRUE_PRED = Predicate(true);
    const Predicate Predicate::FALSE_PRED = Predicate(false);

    //--------------------------------------------------------------------------
    Predicate::Predicate(void)
      : const_value(false), impl(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(bool value)
      : const_value(value), impl(NULL)
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
      {
        impl->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    Predicate::Predicate(PredicateImpl *i)
      : const_value(false), impl(i)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(impl != NULL);
#endif
      impl->add_reference();
    }

    //--------------------------------------------------------------------------
    Predicate::~Predicate(void)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
        impl = NULL;
      }
    }

    //--------------------------------------------------------------------------
    Predicate& Predicate::operator=(const Predicate &p)
    //--------------------------------------------------------------------------
    {
      if (impl != NULL)
      {
        if (impl->remove_reference())
        {
          delete impl;
        }
      }
      const_value = p.const_value;
      impl = p.impl;
      if (impl != NULL)
      {
        impl->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    bool Predicate::operator==(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      return (const_value == p.const_value) && (impl == p.impl);
    }

    //--------------------------------------------------------------------------
    bool Predicate::operator<(const Predicate &p) const
    //--------------------------------------------------------------------------
    {
      if (const_value < p.const_value)
        return true;
      else if (const_value > p.const_value)
        return false;
      else
        return (impl < p.impl);
    }

    /////////////////////////////////////////////////////////////
    // Predicate Implementation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicateImpl::PredicateImpl(MappingTagID t)
      : Mappable(), mapper_invoked(false), tag(t), evaluated(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::get_predicate_value(void)
    //--------------------------------------------------------------------------
    {
      bool result;
      lock();
      if (evaluated)
      {
        result = value;
      }
      else
      {
        if (!mapper_invoked)
        {
          invoke_mapper(); 
        }
        if (speculate)
        {
          result = speculative_value;
        }
        else
        {
          wait_for_evaluation();
#ifdef DEBUG_HIGH_LEVEL
          assert(evaluated);
#endif
          result = value;
        }
      }
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::register_waiter(Notifiable *waiter, bool &valid, bool &val)
    //--------------------------------------------------------------------------
    {
      bool result;
      lock();
      if (evaluated)
      {
        result = false;
        valid = true;
        val = value; 
      }
      else
      {
        result = true;
        waiters.push_back(waiter);
        if (!mapper_invoked)
        {
          invoke_mapper();
        }
        if (speculate)
        {
          valid = true;
          val = speculative_value;
        }
        else
        {
          valid = false;
        }
      }
      unlock();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!mapper_invoked);
      assert(!evaluated);
#endif
      {
        AutoLock m_lock(mapper_lock);
        DetailedTimer::ScopedPush sp(TIME_MAPPER);  
        speculate = mapper->speculate_on_predicate(tag, speculative_value);
      }
      mapper_invoked = true;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::notify_all_waiters(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(evaluated);
#endif
      for (std::vector<Notifiable*>::const_iterator it = waiters.begin();
            it != waiters.end(); it++)
      {
        // Remember to check to see if we should delete it
        if ((*it)->notify(value))
          delete *it;
      }
      waiters.clear();
    }

    /////////////////////////////////////////////////////////////
    // PredicateAnd 
    /////////////////////////////////////////////////////////////

    class PredicateAnd : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateAnd(Predicate p1, Predicate p2,
                   MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      bool first_set;
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateAnd::PredicateAnd(Predicate p1, Predicate p2, 
                               MappingTagID tag)
      : PredicateImpl(tag), first_set(false), 
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p1.register_waiter(this, valid, val))
      {
        // Registered a waiter
        // Increment the reference count 
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        first_set = true;
        // Short-circuit on false, automatically making the predicate false
        if (!val)
        {
          this->value = val;
          this->evaluated = true;
          // Trigger the set_event
          set_event.trigger();
        }
      }
      if (!evaluated && p2.register_waiter(this, valid, val))
      {
        // Register a waiter
        // Increment the reference count
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        if (first_set)
        {
          this->value = val;
          this->evaluated = true;
          set_event.trigger();
        }
        else
        {
          first_set = true;
          // Short-circuit on false, automatically making predicate false
          if (!val)
          {
            this->value = val;
            this->evaluated = true;
            set_event.trigger();
          }
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateAnd::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // Track whether the evaluation was finalized here
      bool local_eval = false;
      lock();
      // Make sure we didn't short circuit, if we did just ignore
      // the incoming value
      if (!evaluated)
      {
        if (first_set)
        {
          this->value = val; 
          this->evaluated = true;
          set_event.trigger();
          local_eval = true;
        }
        else
        {
          this->first_set = true;
          // Short circuit on false
          if (!val)
          {
            this->value = val; 
            this->evaluated = true;
            set_event.trigger();
            local_eval = true;
          }
        }
      }
      // Remove the reference
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      // If we were set here, notify all our waiters
      // before potentially deleting ourselves
      if (local_eval)
        notify_all_waiters();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateAnd::wait_for_evaluation(void) 
    //--------------------------------------------------------------------------
    {
      // TODO: tell the runtime about waiting here
      assert(false);
      set_event.wait();
    }

    /////////////////////////////////////////////////////////////
    // PredicateOr 
    /////////////////////////////////////////////////////////////

    class PredicateOr : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateOr(Predicate p1, Predicate p2,
                  MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      bool first_set;
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateOr::PredicateOr(Predicate p1, Predicate p2, 
                             MappingTagID tag)
      : PredicateImpl(tag), first_set(false), 
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p1.register_waiter(this, valid, val))
      {
        // Registered a waiter
        // Increment the reference count 
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        first_set = true;
        // Short-circuit on true, automatically making the predicate true 
        if (val)
        {
          this->value = val;
          this->evaluated = true;
          // Trigger the set_event
          set_event.trigger();
        }
      }
      if (!evaluated && p2.register_waiter(this, valid, val))
      {
        // Register a waiter
        // Increment the reference count
        add_reference(1/*ref count*/, false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // this better be an evaluated value
#endif
        if (first_set)
        {
          this->value = val;
          this->evaluated = true;
          set_event.trigger();
        }
        else
        {
          first_set = true;
          // Short-circuit on true, automatically making predicate true 
          if (val)
          {
            this->value = val;
            this->evaluated = true;
            set_event.trigger();
          }
        }
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateOr::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // Track whether the evaluation was finalized here
      bool local_eval = false;
      lock();
      // Make sure we didn't short circuit, if we did just ignore
      // the incoming value
      if (!evaluated)
      {
        if (first_set)
        {
          this->value = val; 
          this->evaluated = true;
          set_event.trigger();
          local_eval = true;
        }
        else
        {
          this->first_set = true;
          // Short circuit on true 
          if (val)
          {
            this->value = val; 
            this->evaluated = true;
            set_event.trigger();
            local_eval = true;
          }
        }
      }
      // Remove the reference
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      // If we were set here, notify all our waiters
      // before potentially deleting ourselves
      if (local_eval)
        notify_all_waiters();
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateOr::wait_for_evaluation(void) 
    //--------------------------------------------------------------------------
    {
      // TODO: tell the runtime about waiting here
      assert(false);
      set_event.wait();
    }
    
    /////////////////////////////////////////////////////////////
    // PredicateNot 
    /////////////////////////////////////////////////////////////

    class PredicateNot : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateNot(Predicate p,
                   MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      UserEvent set_event;
    };

    //--------------------------------------------------------------------------
    PredicateNot::PredicateNot(Predicate p, 
                             MappingTagID tag)
      : PredicateImpl(tag),  
        set_event(UserEvent::create_user_event())
    //--------------------------------------------------------------------------
    {
      // Hold the lock so nobody can notify us while doing this
      lock();
      bool valid, val;
      if (p.register_waiter(this, valid, val))
      {
        add_reference(1/*ref count*/,false/*need lock*/);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(valid); // If we're here it better be valid
#endif
        this->value = !val;
        this->evaluated = true;
        set_event.trigger();
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateNot::notify(bool val)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = !val;
      this->evaluated = true;
      set_event.trigger();
      unlock();
      notify_all_waiters();
      return true;
    }

    //--------------------------------------------------------------------------
    void PredicateNot::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      // TODO: tell the runtime about waiting here
      assert(false);
      set_event.wait();
    }
    
    /////////////////////////////////////////////////////////////
    // PredicateFuture 
    /////////////////////////////////////////////////////////////

    class PredicateFuture : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateFuture(Future f, 
                      MappingTagID tag);
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    private:
      // Note that this is just an event like any other
      // corresponding to when the future is set
      Event set_event;
      Future future;
    };

    //--------------------------------------------------------------------------
    PredicateFuture::PredicateFuture(Future f, 
                                     MappingTagID tag)
      : PredicateImpl(tag),  
        set_event(f.impl->set_event), future(f)
    //--------------------------------------------------------------------------
    {
      // Try registering ourselves with the future
      lock();
      if (f.impl->register_waiter(this,this->value))
      {
        // Add a reference
        add_reference(1/*ref count*/,false/*need lock*/);
      }
      else
      {
        this->evaluated = true;
      }
      unlock();
    }

    //--------------------------------------------------------------------------
    bool PredicateFuture::notify(bool val/*dummy value here*/)
    //--------------------------------------------------------------------------
    {
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = val;
      this->evaluated = true;
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      notify_all_waiters(); 
      return result;
    }

    //--------------------------------------------------------------------------
    void PredicateFuture::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      // TODO: tell the runtime about waiting here
      assert(false);
      set_event.wait(); 
    }

    /////////////////////////////////////////////////////////////
    // PredicateCustom 
    /////////////////////////////////////////////////////////////

    class PredicateCustom : public PredicateImpl {
    protected:
      friend class HighLevelRuntime;
      PredicateCustom(PredicateFnptr func, const std::vector<Future> &futures,
                      const TaskArgument &arg, Processor local_proc, 
                      MappingTagID tag);
      virtual ~PredicateCustom(void) { } // Make the warnings go away
    public:
      virtual bool notify(bool value);
      virtual void wait_for_evaluation(void);
    public:
      bool evaluate(void);
    private:
      Event set_event;
      PredicateFnptr custom_func;
      void *custom_arg;
      size_t custom_arg_size;
      std::vector<Future> custom_futures;
    };

    //--------------------------------------------------------------------------  
    PredicateCustom::PredicateCustom(PredicateFnptr func, const std::vector<Future> &futures,
                                     const TaskArgument &arg, Processor local_proc, 
                                     MappingTagID tag)
      : PredicateImpl(tag),
        custom_func(func), custom_futures(futures)
    //--------------------------------------------------------------------------
    {
      lock();
      // Copy in the argument
      custom_arg_size = arg.get_size();
      custom_arg = malloc(custom_arg_size);
      memcpy(custom_arg,arg.get_ptr(),custom_arg_size);
      // Get the set of events corresponding to when futures are ready
      std::set<Event> future_events;
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
      {
        future_events.insert(it->impl->set_event);
      }
      // Add a reference so it doesn't get reclaimed
      add_reference();
      // Get the precondition event
      Event precondition = Event::merge_events(future_events);
      // Launch a task on the local processor to evaluate the predicate once
      // all of the futures are ready
      PredicateCustom *pred = this;
      this->set_event = local_proc.spawn(CUSTOM_PREDICATE_ID,&pred,sizeof(PredicateCustom*),precondition);
      unlock(); 
    }

    //--------------------------------------------------------------------------
    bool PredicateCustom::notify(bool val)
    //--------------------------------------------------------------------------
    {
      // This should never be called for custom predicates
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void PredicateCustom::wait_for_evaluation(void)
    //--------------------------------------------------------------------------
    {
      // TODO: tell the runtime about waiting here
      assert(false);
      set_event.wait();
    }

    //--------------------------------------------------------------------------
    bool PredicateCustom::evaluate(void)
    //--------------------------------------------------------------------------
    {
      // Evaluate the function
      bool pred_value = (*custom_func)(custom_arg,custom_arg_size,custom_futures);
      // Set the value
      lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(!evaluated);
#endif
      this->value = pred_value;
      this->evaluated = true;
      // Remove the reference and see if we're done
      bool result = remove_reference(1/*ref count*/,false/*need lock*/);
      unlock();
      notify_all_waiters();
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Processor Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    HighLevelRuntime::ProcessorManager::ProcessorManager(Processor proc, Processor::Kind kind, HighLevelRuntime *rt, 
                                        unsigned min_out, unsigned def_mappers, unsigned max_steals)
      : runtime(rt), local_proc(proc), proc_kind(kind),
        min_outstanding(min_out), current_outstanding(0),
        idle_task_enabled(false), // to be safe always set it to false
        ready_queues(std::vector<std::list<TaskContext*> >(def_mappers)),
        mapper_objects(std::vector<Mapper*>(def_mappers)),
#ifdef LOW_LEVEL_LOCKS
        mapper_locks(std::vector<Lock>(def_mappers)),
#else
        mapper_locks(std::vector<ImmovableLock>(def_mappers)),
#endif
        max_outstanding_steals(max_steals)
    //--------------------------------------------------------------------------
    {
      // Set up default mapper and locks
      for (unsigned int i=0; i<mapper_objects.size(); i++)
      {
        mapper_objects[i] = NULL;
#ifdef LOW_LEVEL_LOCKS
        mapper_locks[i] = Lock::NO_LOCK;
#else
        mapper_locks[i].clear();
#endif
        ready_queues[i].clear();
        outstanding_steals[i] = std::set<Processor>();
      }
      this->dependence_lock = Lock::create_lock();
      this->idle_lock = Lock::create_lock();
#ifdef LOW_LEVEL_LOCKS
      mapper_locks[0] = Lock::create_lock();
      this->mapping_lock = Lock::create_lock();
      this->queue_lock = Lock::create_lock();
      this->stealing_lock = Lock::create_lock();
      this->thieving_lock = Lock::create_lock();
#ifdef DEBUG_HIGH_LEVEL
      assert(mapping_lock.exists() && queue_lock.exists() &&
              stealing_lock.exists() && thieving_lock.exists()); 
#endif
#else
      mapper_locks[0].init();
      this->mapping_lock.init();
      this->queue_lock.init();
      this->stealing_lock.init();
      this->thieving_lock.init();
#endif
    }

    //--------------------------------------------------------------------------
    HighLevelRuntime::ProcessorManager::~ProcessorManager(void)
    //--------------------------------------------------------------------------
    {
      // Clean up mapper objects and all the low-level locks that we own
#ifdef DEBUG_HIGH_LEVEL
      assert(mapper_objects.size() == mapper_locks.size());
#endif
      for (unsigned i=0; i < mapper_objects.size(); i++)
      {
        if (mapper_objects[i] != NULL)
        {
          delete mapper_objects[i];
          mapper_objects[i] = NULL;
#ifdef DEBUG_HIGH_LEVEL
#ifdef LOW_LEVEL_LOCKS
          assert(mapper_locks[i].exists());
#endif
#endif
#ifdef LOW_LEVEL_LOCKS
          mapper_locks[i].destroy_lock();
#else
          mapper_locks[i].destroy();
#endif
        }
      }
      mapper_objects.clear();
      mapper_locks.clear();
      ready_queues.clear();
      dependence_lock.destroy_lock();
      idle_lock.destroy_lock();
#ifdef LOW_LEVEL_LOCKS
      mapping_lock.destroy_lock();
      queue_lock.destroy_lock();
      stealing_lock.destroy_lock();
      thieving_lock.destroy_lock();
#else
      mapping_lock.destroy();
      queue_lock.destroy();
      stealing_lock.destroy();
      thieving_lock.destroy();
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::add_mapper(MapperID id, Mapper *m, bool check)
    //--------------------------------------------------------------------------------------------
    {
      log_run(LEVEL_SPEW,"Adding mapper %d on processor %x",id,local_proc.id);
#ifdef DEBUG_HIGH_LEVEL
      if (check && (id == 0))
      {
        log_run(LEVEL_ERROR,"Invalid mapping ID.  ID 0 is reserved.");
        assert(false);
        exit(ERROR_RESERVED_MAPPING_ID);
      }
#endif
      AutoLock map_lock(mapping_lock);
      // Increase the size of the mapper vector if necessary
      if (id >= mapper_objects.size())
      {
        int old_size = mapper_objects.size();
        mapper_objects.resize(id+1);
        mapper_locks.resize(id+1);
        ready_queues.resize(id+1);
        for (unsigned int i=old_size; i<(id+1); i++)
        {
          mapper_objects[i] = NULL;
#ifdef LOW_LEVEL_LOCKS
          mapper_locks[i] = Lock::NO_LOCK;
#else
          mapper_locks[i].clear();
#endif
          ready_queues[i].clear();
          outstanding_steals[i] = std::set<Processor>();
        }
      } 
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
      assert(mapper_objects[id] == NULL);
#ifdef LOW_LEVEL_LOCKS
      assert(!mapper_locks[id].exists());
#endif
#endif
#ifdef LOW_LEVEL_LOCKS
      mapper_locks[id] = Lock::create_lock();
#else
      mapper_locks[id].init();
#endif
      AutoLock mapper_lock(mapper_locks[id]);
      mapper_objects[id] = m;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::replace_default_mapper(Mapper *m)
    //--------------------------------------------------------------------------------------------
    {
      // Take an exclusive lock on the mapper data structure
      AutoLock map_lock(mapping_lock);
      AutoLock mapper_lock(mapper_locks[0]);
      delete mapper_objects[0];
      mapper_objects[0] = m;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::initialize_mappable(Mappable *mappable, MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      AutoLock map_lock(mapping_lock,false/*exclusive*/);
#else
      AutoLock map_lock(mapping_lock);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(map_id < mapper_objects.size());
      assert(mapper_objects[map_id] != NULL);
#ifdef LOW_LEVEL_LOCKS
      assert(mapper_locks[map_id].exists());
#endif
#endif
      mappable->initialize_mappable(mapper_objects[map_id], mapper_locks[map_id]);
    }

    //--------------------------------------------------------------------------------------------
    Mapper* HighLevelRuntime::ProcessorManager::find_mapper(MapperID id) 
    //--------------------------------------------------------------------------------------------
    {
#ifdef LOW_LEVEL_LOCKS
      AutoLock map_lock(mapping_lock,false/*exclusive*/);
#else
      AutoLock map_lock(mapping_lock);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(id < mapper_objects.size());
#endif
      return mapper_objects[id];
    }

#ifdef INORDER_EXECUTION
    /////////////////////////////////////////////////////////////
    // Inorder Queue 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    HighLevelRuntime::InorderQueue::InorderQueue(void)
      : eligible(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool HighLevelRuntime::InorderQueue::has_ready(void) const
    //--------------------------------------------------------------------------
    {
      if (!eligible)
        return false;
      if (order_queue.empty())
        return false;
      return order_queue.front().first->is_ready();
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::schedule_next(Context key, std::map<Context,TaskContext*> &tasks_to_map,
                                                       std::map<Context,GeneralizedOperation *> &ops_to_map)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(has_ready());
#endif
      std::pair<GeneralizedOperation*,bool> next = order_queue.front();
      order_queue.pop_front();
      if (next.second)
      {
        tasks_to_map[key] = static_cast<TaskContext*>(next.first);
      }
      else
      {
        ops_to_map[key] = next.first;
      }
      // Mark this queue as ineligible since it now has something executing
      eligible = false;
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::notify_eligible(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!eligible);
#endif
      eligible = true;
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::enqueue_op(GeneralizedOperation *op)
    //--------------------------------------------------------------------------
    {
      order_queue.push_back(std::pair<GeneralizedOperation*,bool>(op,false));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::enqueue_task(TaskContext *task)
    //--------------------------------------------------------------------------
    {
      order_queue.push_back(std::pair<GeneralizedOperation*,bool>(task,true));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::requeue_op(GeneralizedOperation *op)
    //--------------------------------------------------------------------------
    {
      order_queue.push_front(std::pair<GeneralizedOperation*,bool>(op,false));
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::InorderQueue::requeue_task(TaskContext *task)
    //--------------------------------------------------------------------------
    {
      order_queue.push_front(std::pair<GeneralizedOperation*,bool>(task,true));
    }

#ifdef DEBUG_HIGH_LEVEL
    //--------------------------------------------------------------------------
    void HighLevelRuntime::dump_inorder_queues(void)
    //--------------------------------------------------------------------------
    {
      printf("Inorder Queues: %ld\n",inorder_queues.size());
      for (std::map<Context,InorderQueue*>::const_iterator it = inorder_queues.begin();
            it != inorder_queues.end(); it++)
      {
        printf("Queue for task %s has state %d and size %ld\n",
                it->first->variants->name, it->second->eligible, it->second->order_queue.size());
      }
    }
#endif
#endif // INORDER_EXECUTION

    /////////////////////////////////////////////////////////////
    // High Level Runtime 
    /////////////////////////////////////////////////////////////

    // The high level runtime map 
    HighLevelRuntime **HighLevelRuntime::runtime_map = 
      (HighLevelRuntime**)malloc((MAX_NUM_PROCS+1/*+1 since 0 proc is NO_PROC*/)*sizeof(HighLevelRuntime*));

    //--------------------------------------------------------------------------
    HighLevelRuntime::HighLevelRuntime(LowLevel::Machine *m, Processor local)
      : utility_proc(local.get_utility_processor()), 
        explicit_utility_proc(m->has_explicit_utility_processors()), 
        local_procs(m->get_local_processors(local)),
        machine(m),
        unique_stride(m->get_all_processors().size())
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(utility_proc == local); // should only be making high level runtimes for utility processors
#endif
      log_run(LEVEL_DEBUG,"Initializing high-level runtime on processor %x",utility_proc.id);
      // Disable for the idle task for the utility processor.  Note that if this is a real
      // processor that is its own utility processor, it will have the idle task enabled
      // when the processor manager for it is created.
      //local.disable_idle_task();

      // Mark that we are the valid high-level runtime instance for all of our local processors
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        HighLevelRuntime::runtime_map[(it->id & 0xffff)] = this;
#ifdef LEGION_PROF
        Processor::Kind kind = machine->get_processor_kind(*it);
        LegionProf::initialize_processor(*it, false/*util*/, kind);
#endif
      }
#ifdef LEGION_PROF
      if (explicit_utility_proc)
      {
        Processor::Kind kind = machine->get_processor_kind(utility_proc);
        LegionProf::initialize_processor(utility_proc, true/*util*/, kind);
      }
#endif
      {
        // Compute our location in the list of processors
        unsigned idx = 1;
        bool found = false;
        const std::set<Processor>& all_procs = m->get_all_processors();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          idx++;
          if (it->get_utility_processor() == utility_proc)
          {
            found = true;
            break; 
          }
        }
#ifdef DEBUG_HIGH_LEVEL
        assert(found);
#endif
        // Initialize our next id values and strides
        next_partition_id       = idx;
        next_op_id              = idx;
        next_instance_id        = idx;
        next_region_tree_id     = idx;
        next_field_space_id     = idx;
        next_field_id           = idx;
        next_manager_id         = idx;
        start_color             = idx;
      }

      // Set up processor managers for all of our processors
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        ProcessorManager *manager = new ProcessorManager(*it, machine->get_processor_kind(*it), this, min_tasks_to_schedule, 
                                                    DEFAULT_MAPPER_SLOTS, machine->get_all_processors().size()-1);
        proc_managers[*it] = manager;
        // Add the default mapper
        manager->add_mapper(0, new DefaultMapper(machine, this, *it), false/*check*/);
      }
      
#ifdef LOW_LEVEL_LOCKS
      // Initialize our locks
      this->unique_lock = Lock::create_lock();
      this->available_lock= Lock::create_lock();
      this->forest_lock = Lock::create_lock();
#else
      this->unique_lock.init();
      this->available_lock.init();
      this->forest_lock.init();
#endif
#ifdef DEBUG_HIGH_LEVEL
#ifdef LOW_LEVEL_LOCKS
      assert(unique_lock.exists() && 
              available_lock.exists() &&  
               forest_lock.exists());
#endif
#endif
      // Make some default contexts
      this->total_contexts = 0;
      for (unsigned idx = 0; idx < DEFAULT_OPS; idx++)
      {
        available_indiv_tasks.push_back(new IndividualTask(this, this->total_contexts++));
        available_index_tasks.push_back(new IndexTask(this, this->total_contexts++));
        available_slice_tasks.push_back(new SliceTask(this, this->total_contexts++));
        available_point_tasks.push_back(new PointTask(this, this->total_contexts++));
        // Map and deletion ops don't get their own contexts
        available_maps.push_back(new MappingOperation(this));
        available_deletions.push_back(new DeletionOperation(this));
      }

      // Now initialize any mappers that we have
      if (registration_callback != NULL)
        (*registration_callback)(m, this, local_procs);

      // If this is the first processor, launch the legion main task on this processor
      const std::set<Processor> &all_procs = machine->get_all_processors();
      std::set<Processor>::const_iterator first_cpu = all_procs.begin();
      while(machine->get_processor_kind(*first_cpu) != Processor::LOC_PROC)
	first_cpu++;
#ifdef DEBUG_HIGH_LEVEL
      assert(first_cpu->exists());
#endif
      if (local_procs.find(*first_cpu) != local_procs.end())
      {
        log_run(LEVEL_SPEW,"Issuing Legion main task on processor %x",first_cpu->id);
        IndividualTask *top = get_available_individual_task(NULL/*no parent*/);
        // Initialize the task, copying arguments
        top->initialize_task(NULL/*no parent*/, HighLevelRuntime::legion_main_id,
                              &HighLevelRuntime::get_input_args(), sizeof(InputArgs), false/*is index space*/,
                              Predicate::TRUE_PRED, 0/*map id*/, 0/*mapping tag*/); 
        // Mark the top level task so it knows to reclaim itself
        top->top_level_task = true;
        top->orig_proc = *first_cpu;
        top->current_proc = *first_cpu;
        top->executing_processor = *first_cpu;
        find_manager(*first_cpu)->initialize_mappable(top, 0);
#ifdef LEGION_SPY
        LegionSpy::log_top_level_task(utility_proc.id, top->get_gen(), top->get_unique_task_id(), top->ctx_id, HighLevelRuntime::legion_main_id);
        // Also log all the low-level information for the shape of the machine
        {
          std::set<Processor> utility_procs;
          const std::set<Processor> &all_procs = m->get_all_processors();
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
            Processor::Kind k = m->get_processor_kind(*it);
            LegionSpy::log_processor(it->id, it->get_utility_processor().id, k); 
          }
          // Log memories
          const std::set<Memory> &all_mems = m->get_all_memories();
          for (std::set<Memory>::const_iterator it = all_mems.begin();
                it != all_mems.end(); it++)
            LegionSpy::log_memory(it->id, m->get_memory_size(*it));
          // Log Proc-Mem Affinity
          for (std::set<Processor>::const_iterator pit = all_procs.begin();
                pit != all_procs.end(); pit++)
          {
            std::vector<ProcessorMemoryAffinity> affinities;
            m->get_proc_mem_affinity(affinities, *pit);
            for (std::vector<ProcessorMemoryAffinity>::const_iterator it = affinities.begin();
                  it != affinities.end(); it++)
            {
              LegionSpy::log_proc_mem_affinity(pit->id, it->m.id, it->bandwidth, it->latency);
            }
          }
          // Log Mem-Mem Affinity
          for (std::set<Memory>::const_iterator mit = all_mems.begin();
                mit != all_mems.begin(); mit++)
          {
            std::vector<MemoryMemoryAffinity> affinities;
            m->get_mem_mem_affinity(affinities, *mit);
            for (std::vector<MemoryMemoryAffinity>::const_iterator it = affinities.begin();
                  it != affinities.end(); it++)
            {
              LegionSpy::log_mem_mem_affinity(it->m1.id, it->m2.id, it->bandwidth, it->latency);
            }
          }
        }
#endif
#ifdef LEGION_PROF
        {
          // Tell the profiler about all the memories and their kinds
          const std::set<Memory> &all_mems = machine->get_all_memories();
          for (std::set<Memory>::const_iterator it = all_mems.begin();
                it != all_mems.end(); it++)
          {
            Memory::Kind kind = machine->get_memory_kind(*it);
            LegionProf::initialize_memory(*it, kind);
          }
        }
#endif
        // Pack up the event for when the future will trigger so we can deactivate the task
        // context when we're done.  This will make sure that everything gets
        // cleanedup and will help capture any leaks.
        Future f = top->get_future();
        Serializer rez(sizeof(Event));
        rez.serialize<Event>(f.impl->set_event);
        first_cpu->spawn(TERMINATION_ID,rez.get_buffer(),sizeof(Event));
        // Now we can launch the task on the actual processor that we're running on
        top->perform_mapping();
        top->launch_task();
      }
#ifdef DEBUG_HIGH_LEVEL
      tree_state_logger = NULL;
      if (logging_region_tree_state)
      {
        tree_state_logger = new TreeStateLogger(utility_proc);
        assert(tree_state_logger != NULL);
      }
#endif
    }

    //--------------------------------------------------------------------------
    HighLevelRuntime::~HighLevelRuntime(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      for (std::set<Processor>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        LegionProf::finalize_processor(*it);
      }
      if (explicit_utility_proc && (local_procs.find(utility_proc) == local_procs.end()))
        LegionProf::finalize_processor(utility_proc);
#endif
      log_run(LEVEL_SPEW,"Shutting down high-level runtime on processor %x",utility_proc.id);
      {
        AutoLock ctx_lock(available_lock);
#define DELETE_ALL_OPS(listing,type)                                    \
        for (std::vector<type*>::iterator it = listing.begin();         \
              it != listing.end(); it++)                                \
        {                                                               \
          delete *it;                                                   \
        }                                                               \
        listing.clear();
        DELETE_ALL_OPS(available_indiv_tasks,IndividualTask);
        DELETE_ALL_OPS(available_index_tasks,IndexTask);
        DELETE_ALL_OPS(available_slice_tasks,SliceTask);
        DELETE_ALL_OPS(available_point_tasks,PointTask);
        DELETE_ALL_OPS(available_maps,MappingOperation);
        DELETE_ALL_OPS(available_deletions,DeletionOperation);
#undef DELETE_ALL_OPS
      }

      for (std::map<Processor,ProcessorManager*>::iterator it = proc_managers.begin();
            it != proc_managers.end(); it++)
      {
        delete it->second;
      }
      proc_managers.clear();

#ifdef LOW_LEVEL_LOCKS
      available_lock.destroy_lock();
      unique_lock.destroy_lock();
      forest_lock.destroy_lock();
#else
      available_lock.destroy();
      unique_lock.destroy();
      forest_lock.destroy();
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (logging_region_tree_state)
      {
        assert(tree_state_logger != NULL);
        delete tree_state_logger;
        tree_state_logger = NULL;
      }
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskIDTable& HighLevelRuntime::get_task_table(bool add_runtime_tasks)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskIDTable table;
      if (add_runtime_tasks)
      {
        HighLevelRuntime::register_runtime_tasks(table);
      }
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ LowLevel::ReductionOpTable& HighLevelRuntime::get_reduction_table(void)
    //--------------------------------------------------------------------------
    {
      static LowLevel::ReductionOpTable table;
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<Processor::TaskFuncID,TaskVariantCollection*>& HighLevelRuntime::get_collection_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<Processor::TaskFuncID,TaskVariantCollection*> collection_table;
      return collection_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ TypeTable& HighLevelRuntime::get_type_table(void)
    //--------------------------------------------------------------------------
    {
      static TypeTable type_table;
      return type_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionProjectionTable& HighLevelRuntime::get_region_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static RegionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionTable& HighLevelRuntime::get_partition_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static PartitionProjectionTable proj_table;
      return proj_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor::TaskFuncID HighLevelRuntime::get_next_available_id(void)
    //--------------------------------------------------------------------------
    {
      static Processor::TaskFuncID available = TASK_ID_AVAILABLE;
      return available++;
    }

    //--------------------------------------------------------------------------
    /*static*/ InputArgs& HighLevelRuntime::get_input_args(void)
    //--------------------------------------------------------------------------
    {
      static InputArgs inputs = { NULL, 0 };
      return inputs;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID HighLevelRuntime::update_collection_table(void (*low_level_ptr)(const void*,size_t,Processor),
                                                    TaskID uid, const char *name, bool index_space,
                                                    Processor::Kind proc_kind, bool leaf)
    //--------------------------------------------------------------------------
    {
      std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = HighLevelRuntime::get_collection_table();
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
      Processor::TaskFuncID low_id = HighLevelRuntime::get_next_available_id();
      // Add it to the low level table
      HighLevelRuntime::get_task_table(false)[low_id] = low_level_ptr;
      // Now see if an entry already exists in the attribute table for this uid
      if (table.find(uid) == table.end())
      {
        TaskVariantCollection *collec = new TaskVariantCollection(uid, name, leaf);
#ifdef DEBUG_HIGH_LEVEL
        assert(collec != NULL);
#endif
        table[uid] = collec;
        collec->add_variant(low_id, proc_kind, index_space);
      }
      else
      {
        if (table[uid]->leaf != leaf)
        {
          log_run(LEVEL_ERROR,"Tasks of variant %s different leaf statements.  All tasks of the "
                              "same variant must all be leaves, or all be not leaves.", table[uid]->name);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_LEAF_MISMATCH);
        }
        // Update the variants for the attribute
        table[uid]->add_variant(low_id, proc_kind, index_space);
      }
      return uid;
    }

    //--------------------------------------------------------------------------
    /*static*/ RegionProjectionFnptr HighLevelRuntime::find_region_projection_function(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      const RegionProjectionTable &table = get_region_projection_table();
      RegionProjectionTable::const_iterator finder = table.find(pid);
      if (finder == table.end())
      {
        log_run(LEVEL_ERROR,"Unable to find registered region projection ID %d", pid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PROJECTION_ID);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ PartitionProjectionFnptr HighLevelRuntime::find_partition_projection_function(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      const PartitionProjectionTable &table = get_partition_projection_table();
      PartitionProjectionTable::const_iterator finder = table.find(pid);
      if (finder == table.end())
      {
        log_run(LEVEL_ERROR,"Unable to find registered partition projection ID %d", pid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_PROJECTION_ID);
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void HighLevelRuntime::register_runtime_tasks(Processor::TaskIDTable &table)
    //--------------------------------------------------------------------------
    {
      // Check to make sure that nobody has registered any tasks here
      for (unsigned idx = 0; idx < TASK_ID_AVAILABLE; idx++)
      {
        if (table.find(idx) != table.end())
        {
          log_run(LEVEL_ERROR,"Task ID %d is reserved for high-level runtime tasks",idx);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_RESERVED_TASK_ID);
        }
      }
      table[INIT_FUNC_ID]       = HighLevelRuntime::initialize_runtime;
      table[SHUTDOWN_FUNC_ID]   = HighLevelRuntime::shutdown_runtime;
      table[SCHEDULER_ID]       = HighLevelRuntime::schedule;
      table[ENQUEUE_TASK_ID]    = HighLevelRuntime::enqueue_tasks;
      table[STEAL_TASK_ID]      = HighLevelRuntime::steal_request;
      table[CHILDREN_MAPPED_ID] = HighLevelRuntime::children_mapped;
      table[FINISH_ID]          = HighLevelRuntime::finish_task;
      table[NOTIFY_START_ID]    = HighLevelRuntime::notify_start;
      table[NOTIFY_MAPPED_ID]   = HighLevelRuntime::notify_children_mapped;
      table[NOTIFY_FINISH_ID]   = HighLevelRuntime::notify_finish;
      table[ADVERTISEMENT_ID]   = HighLevelRuntime::advertise_work;
      table[FINALIZE_DEL_ID]    = HighLevelRuntime::finalize_deletion;
      table[TERMINATION_ID]     = HighLevelRuntime::detect_termination;
      table[CUSTOM_PREDICATE_ID]= HighLevelRuntime::custom_predicate_eval;
    }

    //--------------------------------------------------------------------------
    TaskVariantCollection* HighLevelRuntime::find_collection(Processor::TaskFuncID tid)
    //--------------------------------------------------------------------------
    {
      std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = HighLevelRuntime::get_collection_table();
      if (table.find(tid) == table.end())
      {
        log_run(LEVEL_ERROR, "Unable to find task variant collection for tasks with ID %d", tid);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_MISSING_TASK_COLLECTION);
      }
      return table[tid];
    }

    /*static*/ volatile RegistrationCallbackFnptr HighLevelRuntime::registration_callback = NULL;
    /*static*/ Processor::TaskFuncID HighLevelRuntime::legion_main_id = 0;
    /*static*/ const long long HighLevelRuntime::init_time = TimeStamp::get_current_time_in_micros();
    /*static*/ unsigned HighLevelRuntime::max_task_window_per_context = MAX_TASK_WINDOW;
    /*static*/ unsigned HighLevelRuntime::min_tasks_to_schedule = MIN_TASKS_TO_PERFORM_SCHEDULING;
    /*static*/ unsigned HighLevelRuntime::max_filter_size = MAX_FILTER_SIZE;
#ifdef INORDER_EXECUTION
    /*static*/ bool HighLevelRuntime::program_order_execution = false;
#endif
#ifdef DYNAMIC_TESTS
    /*static*/ bool HighLevelRuntime::dynamic_independence_tests = false;
#endif
#ifdef DEBUG_HIGH_LEVEL
    /*static*/ bool HighLevelRuntime::logging_region_tree_state = false;
    /*static*/ bool HighLevelRuntime::check_privileges = true;
    /*static*/ bool HighLevelRuntime::verify_disjointness = false;
#endif

    //--------------------------------------------------------------------------
    /*static*/ bool HighLevelRuntime::is_subtype(TypeHandle parent, TypeHandle child)
    //--------------------------------------------------------------------------
    {
      TypeTable &type_table = HighLevelRuntime::get_type_table();
#ifdef DEBUG_HIGH_LEVEL
      if (type_table.find(parent) == type_table.end())
      {
        log_field(LEVEL_ERROR,"Invalid type handle %d", parent);
        assert(false);
        exit(ERROR_INVALID_TYPE_HANDLE);
      }
      if (type_table.find(child) == type_table.end())
      {
        log_field(LEVEL_ERROR,"Invalid type handle %d", child);
        assert(false);
        exit(ERROR_INVALID_TYPE_HANDLE);
      }
#endif
      // Handle the easy case
      if (parent == child)
        return true;
      Structure &current = type_table[child];
      while (current.parent != 0)
      {
        if (current.parent == parent)
          return true;
#ifdef DEBUG_HIGH_LEVEL
        assert(type_table.find(current.parent) != type_table.end());
#endif
        current = type_table[current.parent];
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_registration_callback(RegistrationCallbackFnptr callback)
    //--------------------------------------------------------------------------
    {
      registration_callback = callback;
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* HighLevelRuntime::get_reduction_op(ReductionOpID redop_id)
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
      LowLevel::ReductionOpTable &red_table = HighLevelRuntime::get_reduction_table();
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
    /*static*/ int HighLevelRuntime::start(int argc, char **argv, bool background /*= false*/)
    //--------------------------------------------------------------------------
    {
      // Need to pass argc and argv to low-level runtime before we can record their values
      // as they might be changed by GASNet or MPI or whatever.
      // Note that the logger isn't initialized until after this call returns which means
      // any logging that occurs before this has undefined behavior.
      Machine *m = new Machine(&argc, &argv, HighLevelRuntime::get_task_table(true/*add runtime tasks*/), 
			       HighLevelRuntime::get_reduction_table(), false/*cps style*/);
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

        max_task_window_per_context = MAX_TASK_WINDOW;
        min_tasks_to_schedule = MIN_TASKS_TO_PERFORM_SCHEDULING;
        max_filter_size = MAX_FILTER_SIZE;
        for (int i = 1; i < argc; i++)
        {
#ifdef INORDER_EXECUTION
          BOOL_ARG("-hl:inorder",program_order_execution);
#else
          if (!strcmp(argv[i],"-hl:inorder"))
          {
            log_run(LEVEL_WARNING,"WARNING: Inorder execution is disabled.  To enable inorder execution compile with "
                            "the -DINORDER_EXECUTION flag.");
          }
#endif
          INT_ARG("-hl:window", max_task_window_per_context);
          INT_ARG("-hl:sched", min_tasks_to_schedule);
          INT_ARG("-hl:maxfilter", max_filter_size);
#ifdef DYNAMIC_TESTS
          BOOL_ARG("-hl:dynamic",dynamic_independence_tests); 
#else
          if (!strcmp(argv[i],"-hl:dynamic"))
          {
            log_run(LEVEL_WARNING,"WARNING: Dynamic independence tests are disabled.  To enable dynamic independence tests "
                              "compile with the -DDYNAMIC_TESTS flag.");
          }
#endif
#ifdef DEBUG_HIGH_LEVEL
          BOOL_ARG("-hl:tree",logging_region_tree_state);
          BOOL_ARG("-hl:disjointness",verify_disjointness);
#else
          if (!strcmp(argv[i],"-hl:tree"))
          {
            log_run(LEVEL_WARNING,"WARNING: Region tree state logging is disabled.  To enable region tree state logging "
                              "compile in debug mode.");
          }
          if (!strcmp(argv[i],"-hl:disjointness"))
          {
            log_run(LEVEL_WARNING,"WARNING: Disjointness verification for partition creation is disabled.  To enable dynamic "
                              "disjointness testing compile in debug mode.");
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
      // Do some logging here since the logger was initialized earlier
      {
        const std::map<Processor::TaskFuncID,TaskVariantCollection*>& table = HighLevelRuntime::get_collection_table();
        for (std::map<Processor::TaskFuncID,TaskVariantCollection*>::const_iterator it = table.begin();
              it != table.end(); it++)
        {
          LegionProf::register_task_variant(it->first, it->second->leaf, it->second->name);
        }
      }
#endif
      // Now we can set out input args
      HighLevelRuntime::get_input_args().argv = argv;
      HighLevelRuntime::get_input_args().argc = argc;
      // Kick off the low-level machine
      m->run(0, Machine::ONE_TASK_ONLY, 0, 0, background);
      // We should only make it here if the machine thread is backgrounded
      assert(background);
      return -1;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      Machine *machine = Machine::get_machine();
      machine->wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void HighLevelRuntime::set_top_level_task_id(Processor::TaskFuncID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ HighLevelRuntime* HighLevelRuntime::get_runtime(Processor p)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((p.id & 0xffff) < MAX_NUM_PROCS);
#endif
      return runtime_map[(p.id & 0xffff)];
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_runtime(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // Only do this for utility processors
      if (p.get_utility_processor() == p)
        runtime_map[(p.id & 0xffff)] = new HighLevelRuntime(Machine::get_machine(), p);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::shutdown_runtime(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // Only do this for utility processors
      if (p.get_utility_processor() == p)
        delete get_runtime(p);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::schedule(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      HighLevelRuntime::get_runtime(p)->process_schedule_request(p);
    }

#define UNPACK_ORIGINAL_PROCESSOR(input,output,set_proc)    \
        const char *output = (const char*)input;            \
        Processor set_proc = *((const Processor*)output);   \
        output += sizeof(Processor);
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::enqueue_tasks(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ENQUEUE_TASKS);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_tasks(buffer,arglen-sizeof(Processor),proc);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::steal_request(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_STEAL_REQUEST);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_steal(buffer,arglen-sizeof(Processor),proc);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::children_mapped(const void *result, size_t result_size, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CHILDREN_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(result,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_mapped(buffer,result_size-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finish_task(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FINISH_TASK);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_finish(buffer, arglen-sizeof(Processor));
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_start(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_START);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_start(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_children_mapped(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_MAPPED);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_children_mapped(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_finish(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_NOTIFY_FINISH);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_notify_finish(buffer, arglen-sizeof(Processor));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::advertise_work(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      UNPACK_ORIGINAL_PROCESSOR(args,buffer,proc);
      HighLevelRuntime::get_runtime(proc)->process_advertisement(buffer, arglen-sizeof(Processor));
    }
#undef UNPACK_ORIGINAL_PROCESSOR

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::finalize_deletion(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DeletionOperation *del_op = *((DeletionOperation**)args);
      del_op->finalize();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::custom_predicate_eval(const void *args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      // This will just be normal application level code to evaluate the predicate
      PredicateCustom *pred = (PredicateCustom*)args;
      bool reclaim = pred->evaluate();
      if (reclaim)
        delete pred;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::detect_termination(const void * args, size_t arglen, Processor p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL);
      HighLevelRuntime::get_runtime(p)->process_termination(args, arglen);
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_task(Context ctx, Processor::TaskFuncID task_id,
                                          const std::vector<IndexSpaceRequirement> &indexes,
                                          const std::vector<FieldSpaceRequirement> &fields,
                                          const std::vector<RegionRequirement> &regions,
                                          const TaskArgument &arg, const Predicate &predicate,
                                          MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndividualTask *task = get_available_individual_task(ctx);
      task->initialize_task(ctx, task_id, arg.get_ptr(), arg.get_size(), false/*is index space*/,
                            predicate, id, tag); 
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new single task with unique id %d and task %s (ID %d) with high level runtime on processor %x",
                task->get_unique_id(), task->variants->name, task_id, utility_proc.id);
#endif
#ifdef DEBUG_HIGH_LEVEL
      task->set_requirements(indexes, fields, regions, (check_privileges && !explicit_utility_proc)/*perform checks*/);
#else
      task->set_requirements(indexes, fields, regions, false/*preform checks*/);
#endif

      // Need to get this before we put things on the queue to execute
      Future result = task->get_future();

      // If its not ready it's registered in the logical tree and someone will
      // notify it and it will add itself to the ready queue
      add_to_dependence_queue(ctx->get_executing_processor(), task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      return result;
    }

    //--------------------------------------------------------------------------------------------
    FutureMap HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id, const Domain domain,
                                                    const std::vector<IndexSpaceRequirement> &indexes,
                                                    const std::vector<FieldSpaceRequirement> &fields,
                                                    const std::vector<RegionRequirement> &regions,
                                                    const TaskArgument &global_arg, const ArgumentMap &arg_map,
                                                    const Predicate &predicate, bool must, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndexTask *task = get_available_index_task(ctx);
      task->initialize_task(ctx, task_id, global_arg.get_ptr(), global_arg.get_size(), true/*is index space*/,
                            predicate, id, tag);
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task %s (ID %d) with "
                            "high level runtime on processor %x", task->get_unique_id(), task->variants->name, task_id, utility_proc.id);
#endif
      task->set_index_domain(domain, arg_map, regions.size(), must);
#ifdef DEBUG_HIGH_LEVEL
      task->set_requirements(indexes, fields, regions, (check_privileges && !explicit_utility_proc));
#else
      task->set_requirements(indexes, fields, regions, false/*perform checks*/);
#endif

      // Need to get the future map prior to putting this on the queue to execute
      FutureMap result = task->get_future_map();

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      return result;
    }

    //--------------------------------------------------------------------------------------------
    Future HighLevelRuntime::execute_index_space(Context ctx, Processor::TaskFuncID task_id, Domain domain,
                                                 const std::vector<IndexSpaceRequirement> &indexes,
                                                 const std::vector<FieldSpaceRequirement> &fields,
                                                 const std::vector<RegionRequirement> &regions,
                                                 const TaskArgument &global_arg, const ArgumentMap &arg_map,
                                                 ReductionOpID reduction, const TaskArgument &initial_value,
                                                 const Predicate &predicate, bool must, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_EXECUTE_TASK);
      IndexTask *task = get_available_index_task(ctx);
      task->initialize_task(ctx, task_id, global_arg.get_ptr(), global_arg.get_size(), true/*is index space*/,
                            predicate, id, tag); 
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Registering new index space task with unique id %d and task %s (ID %d) with "
                            "high level runtime on processor %x", task->get_unique_id(), task->variants->name, task_id, utility_proc.id);
#endif
      task->set_index_domain(domain, arg_map, regions.size(), must);
#ifdef DEBUG_HIGH_LEVEL
      task->set_requirements(indexes, fields, regions, (check_privileges && !explicit_utility_proc));
#else
      task->set_requirements(indexes, fields, regions, false/*perform checks*/);
#endif
      task->set_reduction_args(reduction, initial_value);

      // Need to get this before putting it on the queue to execute
      Future result = task->get_future();

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), task);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, task);
      }
#endif

      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::create_index_space(Context ctx, size_t max_num_elmts)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_SPACE);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_INDEX_SPACE> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      IndexSpace space = IndexSpace::create_index_space(max_num_elmts);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Creating index space %x in task %s (ID %d) with %ld maximum elements", space.id,
                              ctx->variants->name,ctx->get_unique_id(), max_num_elmts);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(space.id);
#endif
      // If we have an explicit utilty processor deffer this operation
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_index_space_creation(ctx, Domain(space));
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // Otherwise just do the operation now
        ctx->create_index_space(Domain(space));
      }
      return space;
    }

    //--------------------------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::create_index_space(Context ctx, Domain domain)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_SPACE);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_INDEX_SPACE> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      // Make a dummy index space that will be associated with the domain
      IndexSpace space = 
      domain.get_index_space(true/*create if needed*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(domain.exists());
      log_index(LEVEL_DEBUG, "Creating dummy index space %x in task %s (ID %d) for domain", space.id,
                              ctx->variants->name, ctx->get_unique_id());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_index_space(space.id);
#endif
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_index_space_creation(ctx, domain);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // Otherwise just do the operation now
        ctx->create_index_space(domain);
      }
      return space;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_space(Context ctx, IndexSpace space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Destroying index space %x in task %s (ID %d)", space.id,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_index_space_deletion(ctx, space);

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(Context ctx, IndexSpace parent,
                                  const Coloring &coloring, bool disjoint, int part_color /*= -1*/)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_INDEX_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(pid > 0);
      log_region(LEVEL_DEBUG, "Creating index partition %d with parent index space %x in task %s (ID %d)",
                              pid, parent.id, ctx->variants->name, ctx->get_unique_id());
#endif
      // Perform the coloring
      std::map<Color,Domain> new_index_spaces; 
      for (std::map<Color,ColoredPoints<ptr_t> >::const_iterator cit = 
            coloring.begin(); cit != coloring.end(); cit++)
      {
        LowLevel::ElementMask child_mask(parent.get_valid_mask().get_num_elmts());
        const ColoredPoints<ptr_t> &coloring = cit->second;
        for (std::set<ptr_t>::const_iterator it = coloring.points.begin();
              it != coloring.points.end(); it++)
        {
          child_mask.enable(*it,1);
        }
        for (std::set<std::pair<ptr_t,ptr_t> >::const_iterator it = 
              coloring.ranges.begin(); it != coloring.ranges.end(); it++)
        {
          child_mask.enable(it->first.value, it->second-it->first+1);
        }
        IndexSpace child_space = IndexSpace::create_index_space(parent, child_mask);
        new_index_spaces[cit->first] = Domain(child_space);
      }
#ifdef DEBUG_HIGH_LEVEL
      if (disjoint && verify_disjointness)
      {
        std::set<Color> current_colors;
        for (std::map<Color,Domain>::const_iterator it1 = new_index_spaces.begin();
              it1 != new_index_spaces.end(); it1++)
        {
          current_colors.insert(it1->first);
          for (std::map<Color,Domain>::const_iterator it2 = new_index_spaces.begin();
                it2 != new_index_spaces.end(); it2++)
          {
            // Skip pairs that we already checked
            if (current_colors.find(it2->first) != current_colors.end())
              continue;
            // Otherwise perform the check
            const LowLevel::ElementMask &em1 = it1->second.get_index_space().get_valid_mask();
            const LowLevel::ElementMask &em2 = it2->second.get_index_space().get_valid_mask();
            LowLevel::ElementMask::OverlapResult result = em1.overlaps_with(em2, 1/*effort level*/);
            if (result == LowLevel::ElementMask::OVERLAP_YES)
            {
              log_run(LEVEL_ERROR, "ERROR: colors %d and %d of partition %d are not disjoint when they were claimed to be!",
                                  it1->first, it2->first, pid);
              assert(false);
              exit(ERROR_DISJOINTNESS_TEST_FAILURE);
            }
            else if (result == LowLevel::ElementMask::OVERLAP_MAYBE)
            {
              log_run(LEVEL_WARNING, "WARNING: colors %d and %d of partition %d may not be disjoint when they were claimed to be!"
                                    "(At least according to the low-level runtime.  You might also try telling the the low-level runtime "
                                    "to stop being a lazy bum and try harder.)", it1->first, it2->first, pid);
            }
          }
        }
      }
#endif 
      // Create the new partition
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx); 
        creation->initialize_index_partition_creation(ctx, pid, parent, disjoint, part_color, new_index_spaces);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // Otherwise we can just do the creation here
        ctx->create_index_partition(pid, parent, disjoint, part_color, new_index_spaces);
      }
      return pid;
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::create_index_partition(Context ctx, IndexSpace parent,
                                  Domain color_space, const DomainColoring &coloring,
                                  bool disjoint, int part_color /*= -1*/)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_INDEX_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      IndexPartition pid = get_unique_partition_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(pid > 0);
      log_region(LEVEL_DEBUG, "Creating index partition %d with parent index space %x in task %s (ID %d)",
                              pid, parent.id, ctx->variants->name, ctx->get_unique_id());
#endif
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_index_partition_creation(ctx, pid, parent, disjoint, part_color, coloring, color_space);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // otherwise we can just do the creation here 
        ctx->create_index_partition(pid, parent, disjoint, part_color, coloring, color_space);
      }
      return pid;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_index_partition(Context ctx, IndexPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION);
#ifdef DEBUG_HIGH_LEVEL
      log_index(LEVEL_DEBUG, "Destroying index partition %x in task %s (ID %d)", handle,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_index_partition_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_index_partition(Context ctx, IndexSpace parent, Color color)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_INDEX_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      // Tricky thing here: since we can deffer the creation of things like partitions, they might
      // still be sitting in the dependence queue waiting to be performed, so if we might have
      // defferred it we have to do the following to find it
      // 1. lock the dependence queue to prevent it from escaping
      // 2. try to find what we're looking for in the context
      // 3. if it fails then check the dependence queue
      // 4. if that still doesn't work then something is wrong so raise an error
      IndexPartition result = 0;
      if (explicit_utility_proc)
      {
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();
        result = ctx->get_index_partition(parent, color, false/*can create*/);
        // If we still haven't found it, check the dependence queue
        if (result == 0)
        {
          // Failed to find it, go backwards through the queue looking for matches
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            // dynamic cast shouldn't be too expensive since we already have the runtime
            // type information around implicitly for doing the virtual function dispatches
            // on GeneralizedOperation and its subtypes.
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it); 
            // See if it is a creation op and it is the one we're looking for, if
            // it is then it returns true and we're done
            if ((op != NULL) && op->get_index_partition(ctx, parent, color, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
      }
      else
      {
        result = ctx->get_index_partition(parent, color, true/*can create*/); 
      }
      if (result == 0)
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index partitions", color);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_SPACE_COLOR);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexSpace HighLevelRuntime::get_index_subspace(Context ctx, IndexPartition parent, Color color)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_INDEX_SUBSPACE> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      IndexSpace result = IndexSpace::NO_SPACE;
      if (explicit_utility_proc)
      {
        // See comments in get_index_partition about what is going on here
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();
        result = ctx->get_index_subspace(parent, color, false/*can create*/);
        if (!result.exists())
        {
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it);
            if ((op != NULL) && op->get_index_subspace(ctx, parent, color, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
      }
      else
      {
        result = ctx->get_index_subspace(parent, color, true/*can create*/);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid color %d for get index subspace", color);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_PART_COLOR); 
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    Domain HighLevelRuntime::get_index_space_domain(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_INDEX_DOMAIN> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      Domain result = Domain::NO_DOMAIN;
      if (explicit_utility_proc)
      {
        // See comments in get_index_partition about what is going on here
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();
        result = ctx->get_index_space_domain(handle, false/*can create*/);
        if (!result.exists())
        {
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it);
            if ((op != NULL) && op->get_index_space_domain(ctx, handle, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
      }
      else
      {
        result = ctx->get_index_space_domain(handle, true/*can create*/);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid handle %d for get index space domain", handle.id);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_DOMAIN);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    Domain HighLevelRuntime::get_index_partition_color_space(Context ctx, IndexPartition p)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_INDEX_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_INDEX_PARTITION_COLOR_SPACE> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      Domain result = Domain::NO_DOMAIN;
      if (explicit_utility_proc)
      {
        // See comments in get_index_partition about what is going on here
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();
        result = ctx->get_index_partition_color_space(p, false/*can create*/);
        if (!result.exists())
        {
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it);
            if ((op != NULL) && op->get_index_partition_color_space(ctx, p, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
      }
      else
      {
        result = ctx->get_index_partition_color_space(p, true/*can create*/);
      }
      if (!result.exists())
      {
        log_index(LEVEL_ERROR, "Invalid partition handle %d for get index partition color space", p);
#ifdef DEBUG_HIGH_LEVEL
        assert(false);
#endif
        exit(ERROR_INVALID_INDEX_PART_DOMAIN);
      }
      return result;
    }

    //--------------------------------------------------------------------------------------------
    ptr_t HighLevelRuntime::safe_cast(Context ctx, ptr_t pointer, LogicalRegion region)
    //--------------------------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_SAFE_CAST> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
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

    //--------------------------------------------------------------------------------------------
    DomainPoint HighLevelRuntime::safe_cast(Context ctx, DomainPoint point, LogicalRegion region)
    //--------------------------------------------------------------------------------------------
    {
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_SAFE_CAST> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (point.is_null())
        return point;
      Domain domain = get_index_space_domain(ctx, region.get_index_space());
      if (domain.contains(point))
        return point;
      else
        return DomainPoint::nil();
    }

    //--------------------------------------------------------------------------------------------
    FieldSpace HighLevelRuntime::create_field_space(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_FIELD_SPACE);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_FIELD_SPACE> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      FieldSpace space(get_unique_field_space_id());
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Creating field space %x in task %s (ID %d)", space.id,
                              ctx->variants->name,ctx->get_unique_id());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_field_space(space.id);
#endif
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_field_space_creation(ctx, space);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // No utility processor, so just make it now
        ctx->create_field_space(space);
      }
      return space;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_field_space(Context ctx, FieldSpace space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE);
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG, "Destroying field space %x in task %s (ID %d)", space.id,
                              ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_field_space_deletion(ctx, space);

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    FieldID HighLevelRuntime::allocate_field(Context ctx, FieldSpace space, size_t field_size, FieldID desired_fieldid)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ALLOCATE_FIELD);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_ALLOCATE_FIELDS> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (desired_fieldid == FIELDID_DYNAMIC)
        desired_fieldid = get_unique_field_id();
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG,"Allocating new field %d of size %ld for field space %d in task %s (ID %d)",
                              desired_fieldid, field_size, space.id, ctx->variants->name, ctx->get_unique_id());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_field_creation(space.id, desired_fieldid);
#endif
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_field_creation(ctx, space, desired_fieldid, field_size);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        std::map<FieldID,size_t> field_allocations;
        field_allocations[desired_fieldid] = field_size;
        ctx->allocate_fields(space, field_allocations);
      }
      return desired_fieldid;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_field(Context ctx, FieldSpace space, FieldID fid)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FREE_FIELD);
#ifdef DEBUG_HIGH_LEVEL
      log_field(LEVEL_DEBUG,"Registering a deletion of field %d for field space %d in task %s (ID %d)",
                              fid, space.id, ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      {
        std::set<FieldID> to_free;
        to_free.insert(fid);
        deletion->initialize_field_deletion(ctx, space, to_free);
      }
      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::allocate_fields(Context ctx, FieldSpace space, const std::vector<size_t> &field_sizes,
                                            std::vector<FieldID> &resulting_fields)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_ALLOCATE_FIELD);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_ALLOCATE_FIELDS> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      resulting_fields.clear();
      std::map<FieldID,size_t> field_allocations;
      unsigned field_idx = 0;
      for (std::vector<size_t>::const_iterator it = field_sizes.begin();
            it != field_sizes.end(); it++, field_idx++)
      {
        // check to see if the user has already specified a field id to use
        FieldID new_field;
        if ((field_idx >= resulting_fields.size()) || 
            (resulting_fields[field_idx] == FIELDID_DYNAMIC))
        {
          new_field = get_unique_field_id();
        }
        else
        {
          new_field = resulting_fields[field_idx];
        }
#ifdef DEBUG_HIGH_LEVEL
        log_field(LEVEL_DEBUG,"Allocating new field %d of size %ld for field space %d in task %s (ID %d)",
                              new_field, *it, space.id, ctx->variants->name, ctx->get_unique_id());
#endif
#ifdef LEGION_SPY
        LegionSpy::log_field_creation(space.id, new_field);
#endif
        field_allocations[new_field] = *it; 
        // If it's already been allocated just write the value in
        // otherwise add it to the back
        if (field_idx < resulting_fields.size())
          resulting_fields[field_idx] = new_field;
        else
          resulting_fields.push_back(new_field);
      }
      if (explicit_utility_proc)
      {
        // Defer the creation
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_field_creation(ctx, space, field_allocations);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // Otherwise just do it now
        ctx->allocate_fields(space, field_allocations);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_fields(Context ctx, FieldSpace space, const std::set<FieldID> &to_free)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_FREE_FIELD);
#ifdef DEBUG_HIGH_LEVEL
      for (std::set<FieldID>::const_iterator it = to_free.begin();
            it != to_free.end(); it++)
      {
        log_field(LEVEL_DEBUG,"Registering a deletion of field %d for field space %d in task %s (ID %d)",
                              *it, space.id, ctx->variants->name, ctx->get_unique_id());
      }
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_field_deletion(ctx, space, to_free);
      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::create_logical_region(Context ctx, IndexSpace index_space, FieldSpace field_space)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_CREATE_REGION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_CREATE_REGION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      RegionTreeID tid = get_unique_tree_id();
      LogicalRegion region(tid, index_space, field_space);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Creating logical region in task %s (ID %d) with index space %x and field space %x in new tree %d",
                              ctx->variants->name,ctx->get_unique_id(), index_space.id, field_space.id, tid);
#endif
#ifdef LEGION_SPY
      LegionSpy::log_top_region(index_space.id, field_space.id, tid);
#endif
      if (explicit_utility_proc)
      {
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_region_creation(ctx, region);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
      }
      else
      {
        // Just make it now
        ctx->create_region(region);
      }

      return region;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_region(Context ctx, LogicalRegion handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_REGION);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Deleting logical region (%x,%x) in task %s (ID %d)",
                              handle.index_space.id, handle.field_space.id, ctx->variants->name,ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_region_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_logical_partition(Context ctx, LogicalPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_DESTROY_PARTITION);
#ifdef DEBUG_HIGH_LEVEL
      log_region(LEVEL_DEBUG, "Deleting logical partition (%x,%x) in task %s (ID %d)",
                              handle.index_partition, handle.field_space.id, ctx->variants->name, ctx->get_unique_id());
#endif
      DeletionOperation *deletion = get_available_deletion(ctx);
      deletion->initialize_partition_deletion(ctx, handle);

      // Perform the dependence analysis
      add_to_dependence_queue(ctx->get_executing_processor(), deletion);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, deletion);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition(Context ctx, LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // We already know the answer here so make the right answer
        // and then insert an op that will update the region tree correctly
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_partition(ctx, parent, handle);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return LogicalPartition(parent.tree_id, handle, parent.field_space);
      }
      else
        return ctx->get_region_partition(parent, handle);
    }

    //--------------------------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition_by_color(Context ctx, LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // Need to do this one for real since we can't infer the answer 
        // See get_index_partition for comments about what is going on here
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();
        LogicalPartition result = ctx->get_region_subcolor(parent, c, false/*can create*/);
        if (result == LogicalPartition::NO_PART)
        {
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it);
            if ((op != NULL) && op->get_logical_partition_by_color(ctx, parent, c, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
        if (result == LogicalPartition::NO_PART)
        {
          log_region(LEVEL_ERROR, "Invalid color %d for get logical partition", c);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_INDEX_PART_COLOR);
        }
        // No matter what we have to inject the operation to touch the nodes since we couldn't
        // be sure if we made them this time or not
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_partition_by_color(ctx, parent, c);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return result;
      }
      else
        return ctx->get_region_subcolor(parent, c, true/*can create*/);
    }

    //--------------------------------------------------------------------------------------------
    LogicalPartition HighLevelRuntime::get_logical_partition_by_tree(Context ctx, IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_PARTITION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // We know the answer here, so we can make it, still need op to touch the node
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_partition_by_tree(ctx, handle, space, tid);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return LogicalPartition(tid, handle, space);
      }
      else
        return ctx->get_partition_subtree(handle, space, tid);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion(Context ctx, LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_SUBREGION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // We know the answer here, so we can make it, still need to touch the node
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_subregion(ctx, parent, handle);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return LogicalRegion(parent.tree_id, handle, parent.field_space);
      }
      else
        return ctx->get_partition_subregion(parent, handle);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion_by_color(Context ctx, LogicalPartition parent, Color c)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_SUBREGION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // Don't know the answer here, so we need to go get it
        // See get_index_partition for comments about what is going on here
        ProcessorManager *manager = find_manager(ctx->get_executing_processor());
        manager->lock_dependence_queue();    
        LogicalRegion result = ctx->get_partition_subcolor(parent, c, false/*can create*/);
        if (result == LogicalRegion::NO_REGION)
        {
          const std::list<GeneralizedOperation*> &dependence_queue = manager->get_dependence_queue();
          for (std::list<GeneralizedOperation*>::const_reverse_iterator it = dependence_queue.rbegin();
                it != dependence_queue.rend(); it++)
          {
            CreationOperation *op = dynamic_cast<CreationOperation*>(*it);
            if ((op != NULL) && op->get_logical_subregion_by_color(ctx, parent, c, result))
              break;
          }
        }
        manager->unlock_dependence_queue();
        if (result == LogicalRegion::NO_REGION)
        {
          log_region(LEVEL_ERROR, "Invalid color %d for get logical region", c);
#ifdef DEBUG_HIGH_LEVEL
          assert(false);
#endif
          exit(ERROR_INVALID_INDEX_SPACE_COLOR);
        }
        // No matter what we still need to push the operation to touch the node
        // since we were unable to be sure if we touched it before
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_subregion_by_color(ctx, parent, c);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return result;
      }
      else
        return ctx->get_partition_subcolor(parent, c, true/*can create*/);
    }

    //--------------------------------------------------------------------------------------------
    LogicalRegion HighLevelRuntime::get_logical_subregion_by_tree(Context ctx, IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_GET_LOGICAL_SUBREGION> rec(ctx->task_id, ctx->get_unique_task_id(), ctx->index_point);
#endif
      if (explicit_utility_proc)
      {
        // We know the right answer here
        CreationOperation *creation = get_available_creation(ctx);
        creation->initialize_get_logical_subregion_by_tree(ctx, handle, space, tid);
        add_to_dependence_queue(ctx->get_executing_processor(), creation);
        return LogicalRegion(tid, handle, space);
      }
      else
        return ctx->get_region_subtree(handle, space, tid);
    }

    //--------------------------------------------------------------------------------------------
    ArgumentMap HighLevelRuntime::create_argument_map(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      // Create a new argument map and put it in the list of active maps
      ArgumentMapImpl *arg_map = new ArgumentMapImpl(new ArgumentMapStore());
#ifdef DEBUG_HIGH_LEVEL
      assert(arg_map != NULL);
#endif
      active_argument_maps.insert(arg_map);
      return ArgumentMap(arg_map);
    }

    //--------------------------------------------------------------------------------------------
    IndexAllocator HighLevelRuntime::create_index_allocator(Context ctx, IndexSpace handle)
    //--------------------------------------------------------------------------------------------
    {
      IndexAllocator result(new IndexAllocatorImpl(handle.create_allocator()));
      return result;
    }

    //--------------------------------------------------------------------------------------------
    FieldAllocator HighLevelRuntime::create_field_allocator(Context ctx, FieldSpace handle)
    //--------------------------------------------------------------------------------------------
    {
      return FieldAllocator(handle, ctx, this);
    }

    //--------------------------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, const RegionRequirement &req, 
                                                MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP);
      MappingOperation *map_op = get_available_mapping(ctx);
#ifdef DEBUG_HIGH_LEVEL
      map_op->initialize(ctx, req, id, tag, (check_privileges && !explicit_utility_proc));
#else
      map_op->initialize(ctx, req, id, tag, false/*check privilege*/);
#endif
      log_run(LEVEL_DEBUG, "Registering a map operation for region (%x,%x,%x) in task %s (ID %d)",
                           req.region.index_space.id, req.region.field_space.id, req.region.tree_id,
                           ctx->variants->name, ctx->get_unique_id());
      add_to_dependence_queue(ctx->get_executing_processor(), map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, map_op);
      }
#endif
      return map_op->get_physical_region(); 
    }

    //--------------------------------------------------------------------------------------------
    PhysicalRegion HighLevelRuntime::map_region(Context ctx, unsigned idx, MapperID id, MappingTagID tag)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP);
      MappingOperation *map_op = get_available_mapping(ctx);
      // Don't need to check privileges for these inline mappings since they are original
      // region requirements for our task in which case we already know we have the privileges
      map_op->initialize(ctx, idx, id, tag, false);

      log_run(LEVEL_DEBUG, "Registering a map operation for region index %d in task %s (ID %d)",
                           idx, ctx->variants->name, ctx->get_unique_id());
      add_to_dependence_queue(ctx->get_executing_processor(), map_op);
#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        add_to_inorder_queue(ctx, map_op);
      }
#endif
      return map_op->get_physical_region();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::unmap_region(Context ctx, PhysicalRegion region)
    //--------------------------------------------------------------------------------------------
    {
      DetailedTimer::ScopedPush sp(TIME_HIGH_LEVEL_INLINE_MAP); 
      if (explicit_utility_proc)
      {
        UnmapOperation *unmap = get_available_unmap(ctx);
        unmap->initialize(ctx, region);
        add_to_dependence_queue(ctx->get_executing_processor(), unmap);
      }
      else
        ctx->unmap_physical_region(region);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::create_predicate(Future f, Processor proc, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PredicateFuture *pred = new PredicateFuture(f, tag);
      initialize_mappable(pred, proc, id);
      return Predicate(pred);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::create_predicate(PredicateFnptr function, const std::vector<Future> &futures,
                                                 const TaskArgument &arg, Processor proc, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PredicateCustom *pred = new PredicateCustom(function, futures, arg, utility_proc, tag);
      initialize_mappable(pred, proc, id);
      return Predicate(pred);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_not(Predicate p, Processor proc, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PredicateNot *pred = new PredicateNot(p, tag);
      initialize_mappable(pred, proc, id);
      return Predicate(pred);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_and(Predicate p1, Predicate p2, Processor proc, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PredicateAnd *pred = new PredicateAnd(p1, p2, tag);
      initialize_mappable(pred, proc, id);
      return Predicate(pred);
    }

    //--------------------------------------------------------------------------------------------
    Predicate HighLevelRuntime::predicate_or(Predicate p1, Predicate p2, Processor proc, MapperID id /*=0*/, MappingTagID tag /*=0*/)
    //--------------------------------------------------------------------------------------------
    {
      PredicateOr *pred = new PredicateOr(p1, p2, tag);
      initialize_mappable(pred, proc, id);
      return Predicate(pred);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_mapper(MapperID map_id, Mapper *mapper, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
      find_manager(proc)->add_mapper(map_id, mapper, true/*check*/);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::replace_default_mapper(Mapper *mapper, Processor proc)
    //--------------------------------------------------------------------------------------------
    {
      find_manager(proc)->replace_default_mapper(mapper);
    }
    
    //--------------------------------------------------------------------------------------------
    const std::vector<RegionRequirement>& HighLevelRuntime::begin_task(Context ctx, 
                                             std::vector<PhysicalRegion> &physical_regions, 
                                             const void *&argptr, size_t &arglen)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Beginning task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name,ctx->task_id,ctx->get_unique_id(),utility_proc.id);
#endif
      if (ctx->needs_pre_start())
      {
        // We do this to defer any calls that may involve touching
        // the RegionTreeForest so that we don't block waiting on any of its locks 
        if (explicit_utility_proc)
        {
          // Make a start operation and put it on the dependence queue
          StartOperation *start = get_available_start(ctx);
          start->initialize(ctx);
          add_to_dependence_queue(ctx->get_executing_processor(), start);
        }
        else
        {
          // No utility processor so just do it here
          ctx->pre_start();
        }
      }
      ctx->start_task(physical_regions);
      // Set the argument length and return the pointer to the arguments buffer for the task
      arglen = ctx->arglen;
      argptr = ctx->args;
      return ctx->regions;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::end_task(Context ctx, const void *result, size_t result_size,
                                    std::vector<PhysicalRegion> &physical_regions, bool owned)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Ending task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name, ctx->task_id,ctx->get_unique_id(),utility_proc.id);
#endif
      // Tell the processor manager that this task has finished running on the low-level runtime
      ctx->complete_task(result,result_size,physical_regions, owned);
      // Then figure out how to do the post-complete part of the task
      // We do this to defer any calls taht may involve touching
      // the RegionTreeForest so that we don't block waiting on any of its locks
      if (explicit_utility_proc)
      {
        CompleteOperation *complete = get_available_complete(ctx);
        complete->initialize(ctx);
        add_to_dependence_queue(ctx->get_executing_processor(), complete);
      }
      else
      {
        // no utility proc so just do it now
        ctx->post_complete_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    const void* HighLevelRuntime::get_local_args(Context ctx, DomainPoint &point, size_t &local_size)
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_local_args(point, local_size);
    }

    //--------------------------------------------------------------------------------------------
    Mapper* HighLevelRuntime::get_mapper(Context ctx, MapperID id)
    //--------------------------------------------------------------------------------------------
    {
      ProcessorManager *manager = find_manager(ctx->get_executing_processor());
      return manager->find_mapper(id);
    }

    //--------------------------------------------------------------------------------------------
    Processor HighLevelRuntime::get_executing_processor(Context ctx)
    //--------------------------------------------------------------------------------------------
    {
      return ctx->get_executing_processor();
    }

    //--------------------------------------------------------------------------------------------
    IndividualTask* HighLevelRuntime::get_available_individual_task(Context parent)
    //--------------------------------------------------------------------------------------------
    {
      IndividualTask *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_indiv_tasks.empty())
        {
          result = available_indiv_tasks.back();
          available_indiv_tasks.pop_back();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new IndividualTask(this,id);
        }
        // Update the window before releasing the lock
        if (parent != NULL)
        {
          window_wait = increment_task_window(parent);  
        }
      }
      // Now that we've released the lock, check to see if we need to wait
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        // Tell the runtime that this task is now waiting to run
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        // Once we return, tell the runtime that this task is running again
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexTask* HighLevelRuntime::get_available_index_task(Context parent)
    //--------------------------------------------------------------------------------------------
    {
      IndexTask *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_index_tasks.empty())
        {
          result = available_index_tasks.back();
          available_index_tasks.pop_back();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new IndexTask(this,id);
        }
        // Update the window before releasing the lock
        if (parent != NULL)
        {
          window_wait = increment_task_window(parent);
        }
      }
      // Now that we've released the lock, check to see if we need to wait
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    SliceTask* HighLevelRuntime::get_available_slice_task(TaskContext *parent)
    //--------------------------------------------------------------------------------------------
    {
      SliceTask *result = NULL;
      {
        AutoLock av_lock(available_lock);
        if (!available_slice_tasks.empty())
        {
          result = available_slice_tasks.back();
          available_slice_tasks.pop_back();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new SliceTask(this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    PointTask* HighLevelRuntime::get_available_point_task(TaskContext *parent)
    //--------------------------------------------------------------------------------------------
    {
      PointTask *result = NULL;
      {
        AutoLock av_lock(available_lock);
        if (!available_point_tasks.empty())
        {
          result = available_point_tasks.back();
          available_point_tasks.pop_back();
        }
        else
        {
          ContextID id = total_contexts++;
          result = new PointTask(this,id);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    MappingOperation* HighLevelRuntime::get_available_mapping(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      MappingOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_maps.empty())
        {
          result = available_maps.back();
          available_maps.pop_back();
        }
        else
        {
          result = new MappingOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    UnmapOperation* HighLevelRuntime::get_available_unmap(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      UnmapOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_unmaps.empty())
        {
          result = available_unmaps.back();
          available_unmaps.pop_back();
        }
        else
        {
          result = new UnmapOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    DeletionOperation* HighLevelRuntime::get_available_deletion(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      DeletionOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_deletions.empty())
        {
          result = available_deletions.back();
          available_deletions.pop_back();
        }
        else
        {
          result = new DeletionOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    CreationOperation* HighLevelRuntime::get_available_creation(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      CreationOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_creations.empty())
        {
          result = available_creations.back();
          available_creations.pop_back();
        }
        else
        {
          result = new CreationOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    EpochOperation* HighLevelRuntime::get_available_epoch(void)
    //--------------------------------------------------------------------------------------------
    {
      EpochOperation *result = NULL;
      {
        AutoLock av_lock(available_lock);
        if (!available_epochs.empty())
        {
          result = available_epochs.back();
          available_epochs.pop_back();
        }
        else
        {
          result = new EpochOperation(this);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(NULL/*no parent*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    StartOperation* HighLevelRuntime::get_available_start(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      StartOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_starts.empty())
        {
          result = available_starts.back();
          available_starts.pop_back();
        }
        else
        {
          result = new StartOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    CompleteOperation* HighLevelRuntime::get_available_complete(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      CompleteOperation *result = NULL;
      Event window_wait = Event::NO_EVENT;
      {
        AutoLock av_lock(available_lock);
        if (!available_completes.empty())
        {
          result = available_completes.back();
          available_completes.pop_back();
        }
        else
        {
          result = new CompleteOperation(this);
        }
        window_wait = increment_task_window(parent);
      }
      if (window_wait.exists())
      {
        Processor executing_proc = parent->get_executing_processor();
        decrement_processor_executing(executing_proc);
        window_wait.wait();
        increment_processor_executing(executing_proc);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(result != NULL);
      bool activated = 
#endif
      result->activate(parent);
#ifdef DEBUG_HIGH_LEVEL
      assert(activated);
#endif
      return result;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_individual_task(IndividualTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_indiv_tasks.push_back(task);
    }

   //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_index_task(IndexTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_index_tasks.push_back(task);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_slice_task(SliceTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_slice_tasks.push_back(task);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_point_task(PointTask *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_point_tasks.push_back(task);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_mapping(MappingOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_maps.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_unmap(UnmapOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_unmaps.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_deletion(DeletionOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_deletions.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_creation(CreationOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_creations.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_epoch(EpochOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_epochs.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_start(StartOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_starts.push_back(op);
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::free_complete(CompleteOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock av_lock(available_lock);
      available_completes.push_back(op);
    }

    //--------------------------------------------------------------------------------------------
    RegionTreeForest* HighLevelRuntime::create_region_forest(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock);
      RegionTreeForest *result = new RegionTreeForest(this);
      active_forests.insert(result);
      return result;
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::destroy_region_forest(RegionTreeForest *forest)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock);
      std::set<RegionTreeForest*>::iterator finder = active_forests.find(forest);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != active_forests.end());
#endif
      active_forests.erase(finder);
      // Free up the memory
      delete forest;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::notify_operation_complete(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif

#ifdef INORDER_EXECUTION
      if (program_order_execution)
      {
        // If we're doing inorder execution notify a queue whenever an operation
        // finishes so we can do the next one
        AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(inorder_queues.find(parent) != inorder_queues.end());
#endif
        inorder_queues[parent]->notify_eligible();
        if (!idle_task_enabled)
        {
          idle_task_enabled = true;
          UtilityProcessor copy = utility_proc;
          copy.enable_idle_task();
        }
      }
#endif
      // Always do this when an operation completes
      {
        AutoLock av_lock(available_lock);
        decrement_task_window(parent);
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::is_local_processor(Processor target) const
    //--------------------------------------------------------------------------------------------
    {
      return (local_procs.find(target) != local_procs.end());
    }

    //--------------------------------------------------------------------------------------------
    HighLevelRuntime::ProcessorManager* HighLevelRuntime::find_manager(Processor proc) const
    //--------------------------------------------------------------------------------------------
    {
      std::map<Processor,ProcessorManager*>::const_iterator finder = proc_managers.find(proc);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != proc_managers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::increment_processor_executing(Processor proc)
    //--------------------------------------------------------------------------------------------
    {
      ProcessorManager *manager = find_manager(proc);
      manager->increment_outstanding();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::decrement_processor_executing(Processor proc)
    //--------------------------------------------------------------------------------------------
    {
      ProcessorManager *manager = find_manager(proc);
      manager->decrement_outstanding();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::initialize_mappable(Mappable *mappable, Processor target, MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
      find_manager(target)->initialize_mappable(mappable, map_id);
    }

    //--------------------------------------------------------------------------------------------
    Event HighLevelRuntime::increment_task_window(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
#endif
      if (context_windows.find(parent) == context_windows.end())
      {
        // Didn't exist before so make it 
        context_windows[parent] = WindowState(1/*num children*/,false/*blocked*/);
      }
      else
      {
        WindowState &state = context_windows[parent];
#ifdef DEBUG_HIGH_LEVEL
        // We should never be here if we're blocked
        assert(!state.blocked);
        assert(state.active_children < max_task_window_per_context);
#endif
        // Check to see if we've reached the maximum window size
        if ((++state.active_children) == max_task_window_per_context)
        {
          // Mark that we're blocked, create a user event and set it
          state.blocked = true; 
          state.notify_event = UserEvent::create_user_event();
          return state.notify_event; 
        }
        // Otherwise no need to do anything
      }
      return Event::NO_EVENT;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::decrement_task_window(Context parent)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(parent != NULL);
      // The state better exist too
      assert(context_windows.find(parent) != context_windows.end());
#endif
      WindowState &state = context_windows[parent];
      state.active_children--;
      if (state.blocked)
      {
        state.blocked = false;
        state.notify_event.trigger();
      }
    }

    //--------------------------------------------------------------------------------------------
    InstanceID HighLevelRuntime::get_unique_instance_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      InstanceID result = next_instance_id;
      next_instance_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    UniqueID HighLevelRuntime::get_unique_op_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      UniqueID result = next_op_id;
      next_op_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    IndexPartition HighLevelRuntime::get_unique_partition_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      IndexPartition result = next_partition_id;
      next_partition_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    RegionTreeID HighLevelRuntime::get_unique_tree_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      RegionTreeID result = next_region_tree_id;
      next_region_tree_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    FieldSpaceID HighLevelRuntime::get_unique_field_space_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      FieldSpaceID result = next_field_space_id;
      next_field_space_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    FieldID HighLevelRuntime::get_unique_field_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      FieldID result = next_field_id;
      next_field_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    UniqueManagerID HighLevelRuntime::get_unique_manager_id(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock ulock(unique_lock);
      UniqueManagerID result = next_manager_id;
      next_manager_id += unique_stride;
      return result;
    }

    //--------------------------------------------------------------------------------------------
    Color HighLevelRuntime::get_start_color(void) const
    //--------------------------------------------------------------------------------------------
    {
      return start_color;
    }

    //--------------------------------------------------------------------------------------------
    unsigned HighLevelRuntime::get_color_modulus(void) const
    //--------------------------------------------------------------------------------------------
    {
      return unique_stride;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_dependence_queue(Processor proc, GeneralizedOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      find_manager(proc)->add_to_dependence_queue(op);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, IndividualTask *task, bool remote)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_ready_queue(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // Note if it's remote we still need to add it to the drain queue
        // so that it actually gets executed
        if (remote)
        {
          drain_queue.push_back(task);
        }
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, IndexTask *task)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_ready_queue(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, SliceTask *task)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_ready_queue(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // This is not a program level task so put it on the drain queue
        // to be handled.
        drain_queue.push_back(task);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, PointTask *task)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_ready_queue(task);
#else
      if (!program_order_execution)
      {
        ready_queues[task->map_id].push_back(task);
      }
      else
      {
        // This is not a program level task so put it on the drain queue
        // to be handled.
        drain_queue.push_back(task);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, MappingOperation *op)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_other_queue(op);
#else
      if (!program_order_execution)
      {
        other_ready_queue.push_back(op);
      }
#endif
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_ready_queue(Processor proc, DeletionOperation *op)
    //--------------------------------------------------------------------------------------------
    {
#ifndef INORDER_EXECUTION
      // If we're doing inorder execution we don't need to do this since
      // we already have the task in the right queue, but we should still
      // enable the idle task to see if anything is ready to execute.
      find_manager(proc)->add_to_other_queue(op);
#else
      if (!program_order_execution)
      {
        other_ready_queue.push_back(op);
      }
#endif
    } 

#ifdef INORDER_EXECUTION
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, TaskContext *task)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_task(task); 
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        UtilityProcessor copy = utility_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, MappingOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_op(op);
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        UtilityProcessor copy = utility_proc;
        copy.enable_idle_task();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::add_to_inorder_queue(Context parent, DeletionOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(program_order_execution);
#endif
      if (inorder_queues.find(parent) == inorder_queues.end())
      {
        inorder_queues[parent] = new InorderQueue();
      }
      inorder_queues[parent]->enqueue_op(op);
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        UtilityProcessor copy = utility_proc;
        copy.enable_idle_task();
      }
    }
#endif // INORDER_EXECUTION

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::send_task(Processor target, TaskContext *task) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(target.exists());
#endif
      if (local_procs.find(target) == local_procs.end())
      {
        Processor target_utility = target.get_utility_processor();
        size_t buffer_size = sizeof(Processor) + sizeof(size_t) + sizeof(bool);
        task->lock_context();
        buffer_size += task->compute_task_size();
        Serializer rez(buffer_size); 
        rez.serialize<Processor>(target);
        rez.serialize<size_t>(1); // only one task
        rez.serialize<bool>(task->is_single());
        task->pack_task(rez);
        task->unlock_context();
        // Send the result back to the target's utility processor
        target_utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), buffer_size);
        return false;
      }
      else
      {
        // Send it to the local manager
        find_manager(target)->add_to_ready_queue(task);
        return true;
      }
    }

    //--------------------------------------------------------------------------------------------
    bool HighLevelRuntime::send_tasks(Processor target, const std::set<TaskContext*> &tasks) const
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(target.exists());
#endif
      if (local_procs.find(target) == local_procs.end())
      {
        Processor target_utility = target.get_utility_processor();
        size_t total_buffer_size = sizeof(Processor) + sizeof(size_t) + (tasks.size() * sizeof(bool));
        Serializer rez(total_buffer_size); 
        rez.serialize<Processor>(target);
        rez.serialize<size_t>(tasks.size());
        for (std::set<TaskContext*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++)
        {
          rez.serialize<bool>((*it)->is_single());
          (*it)->lock_context();
          size_t task_size = (*it)->compute_task_size();
          total_buffer_size += task_size;
          rez.grow(task_size);
          (*it)->pack_task(rez);
          (*it)->unlock_context();
        }
        // Send the result back to the target's utility processor
        target_utility.spawn(ENQUEUE_TASK_ID, rez.get_buffer(), total_buffer_size);
        return false;
      }
      else
      {
        ProcessorManager *manager = find_manager(target);
        for (std::set<TaskContext*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++)
        {
          manager->add_to_ready_queue(*it);
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::send_steal_request(const std::multimap<Processor,MapperID> &targets, Processor thief) const
    //--------------------------------------------------------------------------------------------
    {
      for (std::multimap<Processor,MapperID>::const_iterator it = targets.begin();
            it != targets.end(); )
      {
        Processor target = it->first;
        // Check to see if it is a local task or not
        if (local_procs.find(target) == local_procs.end())
        {
          Processor utility_target = target.get_utility_processor();
          int num_mappers = targets.count(target);
          log_task(LEVEL_SPEW,"Processor %x attempting steal on processor %d",
                                thief.id,target.id);
          size_t buffer_size = sizeof(Processor)+sizeof(Processor)+sizeof(int)+num_mappers*sizeof(MapperID);
          // Allocate a buffer for launching the steal task
          Serializer rez(buffer_size);
          // Give the actual target processor
          rez.serialize<Processor>(utility_target);
          // Give the stealing (this) processor
          rez.serialize<Processor>(thief);
          rez.serialize<int>(num_mappers);
          for ( ; it != targets.upper_bound(target); it++)
          {
            rez.serialize<MapperID>(it->second);
          }

          // Now launch the task to perform the steal operation
          utility_target.spawn(STEAL_TASK_ID,rez.get_buffer(),buffer_size);
        }
        else
        {
          // Local processor so do the steal here
          ProcessorManager *manager  = find_manager(target);
          std::vector<MapperID> thieves;
          for ( ; it != targets.upper_bound(target); it++)
            thieves.push_back(it->second);
          manager->process_steal_request(thief, thieves);
        }
#ifdef DEBUG_HIGH_LEVEL
        if (it != targets.end())
          assert(!((target.id) == (it->first.id)));
#endif
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::send_advertisements(const std::set<Processor> &targets, 
                                                MapperID map_id, Processor source) const
    //--------------------------------------------------------------------------------------------
    {
      // make sure we don't send an advertisement twice since the highlevel
      // runtime will already broadcast it out to all its local processors
      std::set<Processor> already_sent;
      size_t buffer_size = sizeof(Processor)+sizeof(Processor)+sizeof(MapperID);
      for (std::set<Processor>::const_iterator it = targets.begin();
            it != targets.end(); it++)
      {
        if (local_procs.find(*it) != local_procs.end())
        {
          // Local processor, do the notify here
          find_manager(*it)->process_advertisement(source, map_id);
        }
        else
        {
          Processor utility_target = it->get_utility_processor();
          if (already_sent.find(utility_target) != already_sent.end())
            continue;
          Serializer rez(buffer_size);
          // Send a message to the processor saying that a specific mapper has work now
          rez.serialize<Processor>(*it); // The actual target processor
          rez.serialize<Processor>(source); // The advertising processor
          rez.serialize<MapperID>(map_id);
          // Send the advertisement
          utility_target.spawn(ADVERTISEMENT_ID,rez.get_buffer(),buffer_size);
          already_sent.insert(utility_target);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_tasks(const void * args, size_t arglen, Processor target)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      // Then get the number of tasks to process
      size_t num_tasks; 
      derez.deserialize<size_t>(num_tasks);
      for (unsigned idx = 0; idx < num_tasks; idx++)
      {
        // Figure out whether this is a individual task or a slice task
        // Note it can never be a point task because they never move without their slice
        // and it can't be an index task because they never move.
        bool single;
        derez.deserialize<bool>(single);
        if (single)
        {
          IndividualTask *task = get_available_individual_task(NULL/*no parent on this node*/);
          task->lock_context();
          task->unpack_task(derez);
          task->unlock_context();
          add_to_ready_queue(target, task, true/*remote*/);
        }
        else
        {
          SliceTask *task = get_available_slice_task(NULL/*no parent on this node*/);
          task->lock_context();
          task->unpack_task(derez);
          task->unlock_context();
          add_to_ready_queue(target, task);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_steal(const void * args, size_t arglen, Processor target)
    //--------------------------------------------------------------------------------------------
    {
#ifdef INORDER_EXECUTION
      if (program_order_execution)
        assert(false); // should never get a steal request during INORDER_EXECUTION
#endif
      Deserializer derez(args,arglen);
      // Unpack the stealing processor
      Processor thief;
      derez.deserialize<Processor>(thief);	
      // Get the number of mappers that requested this processor for stealing 
      int num_stealers;
      derez.deserialize<int>(num_stealers);
      std::vector<MapperID> thieves(num_stealers);
      for (int idx = 0; idx < num_stealers; idx++)
        derez.deserialize(thieves[idx]);

      find_manager(target)->process_steal_request(thief, thieves); 
    }
    
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"All child tasks mapped for task %s (ID %d) with unique id %d on processor %x",
        ctx->variants->name,ctx->task_id,ctx->get_unique_id(),utility_proc.id);
#endif
      ctx->children_mapped();
    }    

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context from the arguments
      Context ctx = *((const Context*)args);
#ifdef DEBUG_HIGH_LEVEL
      log_task(LEVEL_DEBUG,"Task %s (ID %d) with unique id %d finished on processor %x", 
        ctx->variants->name,ctx->task_id, ctx->get_unique_id(), utility_proc.id);
#endif
      ctx->finish_task();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_start(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack context, task, and event info
      const char * ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);
     
      local_ctx->remote_start(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_children_mapped(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the context
      const char *ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);

      local_ctx->remote_children_mapped(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_notify_finish(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      // Unpack the user event to be trigged when we finished
      const char *ptr = (const char*)args;
      TaskContext *local_ctx = *((TaskContext**)ptr);
      ptr += sizeof(TaskContext*);

      local_ctx->remote_finish(ptr, arglen-sizeof(TaskContext*));
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_termination(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      // Unpack the event to wait on and then wait for it
      Event termination_event;
      derez.deserialize<Event>(termination_event);
      // Wait until the set event has completed
      termination_event.wait();
      log_task(LEVEL_SPEW,"Computation has terminated, shutting down high level runtime...");
      // Once this is over shutdown the machine
      machine->shutdown();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_advertisement(const void * args, size_t arglen)
    //--------------------------------------------------------------------------------------------
    {
      Deserializer derez(args,arglen);
      // Get the processor that is advertising work
      Processor advertiser;
      derez.deserialize<Processor>(advertiser);
      MapperID map_id;
      derez.deserialize<MapperID>(map_id);
      // Notify all of our processor managers about the advertisement
      for (std::map<Processor,ProcessorManager*>::const_iterator it = proc_managers.begin();
            it != proc_managers.end(); it++)
      {
        it->second->process_advertisement(advertiser, map_id);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::process_schedule_request(Processor proc)
    //--------------------------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(local_procs.find(proc) != local_procs.end());
#endif
      log_run(LEVEL_DEBUG,"Running scheduler on processor %x", proc.id);
#ifdef LEGION_PROF
      LegionProf::Recorder<PROF_SCHEDULER> rec(0, 0, DomainPoint());
#endif
      ProcessorManager *manager = proc_managers[proc];
      // first perform the dependence analysis 
      manager->perform_dependence_analysis();

      // Perform these before doing any mappings
#ifdef DYNAMIC_TESTS
      if (dynamic_independence_tests)
        perform_dynamic_tests();
#endif

#ifdef INORDER_EXECUTION
      // Short circuit for inorder case
      if (program_order_execution)
      {
        perform_inorder_scheduling();
        return;
      }
#endif
      // Now perform any other operations that are not tasks to enusre
      // that as many tasks are eligible for mapping as possible
      manager->perform_other_operations();
      // Finally do any scheduling 
      manager->perform_scheduling();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::perform_dependence_analysis(void)
    //--------------------------------------------------------------------------------------------
    {
      // There is something fancy going on here: since there are some operations
      // (e.g. get_subregion_by_color) that introspect this queue, we can't clear
      // the queue until those operations have been applied to the RegionTreeForest.
      // So we pull out the operations we are going to perform, but don't clear
      // them from the queue until we are actually done applying the operations.
      // This two-phase approach to handling dependence analysis allows us to keep
      // the operations in the queue so they can be queried while they still may
      // yet to be performed.
      std::list<GeneralizedOperation*> ops;
      size_t num_ops_handled;
      // Empty out the queue
      {
        // Only reading the dependence queue this time
        AutoLock d_lock(dependence_lock,1,false/*exclusive*/);
        num_ops_handled = dependence_queue.size();
        ops.insert(ops.end(),dependence_queue.begin(),dependence_queue.end());
      }
      // VERY VERY IMPORTANT that these get done in order
      for (std::list<GeneralizedOperation*>::iterator it = ops.begin();
            it != ops.end(); /*nothing*/)
      {
        // Only keep them in the list if we should deactivate them 
        if ((*it)->perform_dependence_analysis())
          it++;
        else
          it = ops.erase(it);
      }
      // Now that we're done applying the operations, pull them off the dependence queue
      {
        AutoLock d_lock(dependence_lock);
        for (unsigned idx = 0; idx < num_ops_handled; idx++)
          dependence_queue.pop_front();
      }
      // Finally we can deactivate the operations which are finished
      for (std::list<GeneralizedOperation*>::const_iterator it = ops.begin();
            it != ops.end(); it++)
      {
        (*it)->deactivate(); 
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::perform_other_operations(void)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<GeneralizedOperation*> ops;
      // Empty out the queue
      {
        AutoLock q_lock(queue_lock);
        ops = other_ready_queue;
        other_ready_queue.clear();
      }
      std::vector<GeneralizedOperation*> failed_ops;
      for (unsigned idx = 0; idx < ops.size(); idx++)
      {
        // Perform each operation
        bool success = ops[idx]->perform_operation();
        if (!success)
          failed_ops.push_back(ops[idx]);
      }
      if (!failed_ops.empty())
      {
        AutoLock q_lock(queue_lock);
        // Put them on the back since this is a vector 
        // and it won't make much difference what order
        // they get performed in
        other_ready_queue.insert(other_ready_queue.end(),failed_ops.begin(),failed_ops.end());
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------------------------
    {
      // Do a quick check to see if we should run the scheduler
      bool do_scheduling;
      {
        AutoLock i_lock(idle_lock);
        do_scheduling = (current_outstanding < min_outstanding);
      }
      // Get the lists of tasks to map
      std::vector<TaskContext*> tasks_to_map;
      // Also get the list of any steals the mappers want to perform
      std::multimap<Processor,MapperID> targets;
      // Whether or not any of the mappers are still actively trying to
      // steal (e.g. guess a processor not on the blacklist)
      bool still_stealing = false; 
      if (do_scheduling)
      {
        AutoLock q_lock(queue_lock);
        AutoLock m_lock(mapping_lock);
        // Also perform stealing here
        AutoLock steal_lock(stealing_lock);
        for (unsigned map_id = 0; map_id < ready_queues.size(); map_id++)
        {
          // Check for invalid mappers
          if (mapper_objects[map_id] == NULL)
            continue;
          std::vector<bool> mask(ready_queues[map_id].size());
          for (unsigned idx = 0; idx < mask.size(); idx++)
            mask[idx] = false;
          // Acquire the mapper lock
          {
            AutoLock map_lock(mapper_locks[map_id]);
            DetailedTimer::ScopedPush sp(TIME_MAPPER); 
            // Only do this if the list isn't empty
            if (!ready_queues[map_id].empty())
            {
              // Watch me stomp all over the C++ type system here
              const std::list<Task*> &ready_tasks = *((std::list<Task*>*)(&(ready_queues[map_id])));
              mapper_objects[map_id]->select_tasks_to_schedule(ready_tasks, mask);
            }
            // Now ask about stealing
            std::set<Processor> &blacklist = outstanding_steals[map_id];
            if (blacklist.size() <= max_outstanding_steals)
            {
              Processor p = mapper_objects[map_id]->target_task_steal(blacklist);
              if (p.exists() && (p != local_proc) && (blacklist.find(p) == blacklist.end()))
              {
                targets.insert(std::pair<Processor,MapperID>(p,map_id));
                // Update the list of outstanding steal requests, add in all the processors for
                // the utility processor of the target processor
                blacklist.insert(p);
                still_stealing = true;
              }
            }
          }
          std::list<TaskContext*>::iterator list_it = ready_queues[map_id].begin();
          for (unsigned idx = 0; idx < mask.size(); idx++)
          {
            if (mask[idx])
            {
              tasks_to_map.push_back(*list_it);
              list_it = ready_queues[map_id].erase(list_it);
            }
            else
            {
              list_it++;
            }
          }
        }
      } // release the queue lock and the mapping lock

      std::vector<TaskContext*> failed_mappings;
      // Now we've got our list of tasks to map so map all of them
      for (unsigned idx = 0; idx < tasks_to_map.size(); idx++)
      {
        bool mapped = tasks_to_map[idx]->perform_operation();
        if (!mapped)
          failed_mappings.push_back(tasks_to_map[idx]);
      }

      // Also send out any steal requests that might have been made
      if (!targets.empty())
        runtime->send_steal_request(targets, local_proc);
      
      // If we had any failed mappings, put them back on the (front of the) ready queue
      if (!failed_mappings.empty())
      {
        AutoLock q_lock(queue_lock);
        for (std::vector<TaskContext*>::reverse_iterator it = failed_mappings.rbegin();
              it != failed_mappings.rend(); it++)
        {
          ready_queues[(*it)->map_id].push_front(*it);
        }
        failed_mappings.clear();
      }

      std::vector<MapperID> mappers_with_work;
      // Now we need to determine if should disable the idle task
      {
        // Need to hold all the queue locks while doing this
        // Disable only if all of the following are true
        bool disable = true;
        // 1. No ops in dependence queue
        AutoLock d_lock(dependence_lock,1,false/*exclusive*/); // only reading
        disable = disable && dependence_queue.empty();
        // 2. No ops in other_ready_queue
        AutoLock q_lock(queue_lock);
        disable = disable && other_ready_queue.empty();
        bool have_ready_tasks = false;
        // Need to run this loop anyway to get the set of mappers with work for advertising
        for (unsigned map_id = 0; map_id < ready_queues.size(); map_id++)
        {
          if (mapper_objects[map_id] == NULL)
            continue;
          if (!ready_queues[map_id].empty())
          {
            mappers_with_work.push_back(map_id);
            have_ready_tasks = true;
          }
        }
        // 3. Have enough tasks in flight or (all the ready_queues are empty and not still stealing)
        AutoLock i_lock(idle_lock);
        disable = disable && ((current_outstanding >= min_outstanding) || (!have_ready_tasks && !still_stealing));
        if (disable)
        {
          idle_task_enabled = false;
          Processor copy = local_proc;
          copy.disable_idle_task();
        }
      }
      // If we had any mappers with work, advertise to anyone who had tried to steal before
      for (std::vector<MapperID>::const_iterator it = mappers_with_work.begin();
            it != mappers_with_work.end(); it++)
      {
        advertise(*it);
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::process_steal_request(Processor thief, const std::vector<MapperID> &thieves)
    //--------------------------------------------------------------------------------------------
    {
      log_run(LEVEL_SPEW,"handling a steal request on processor %x from processor %x",
              local_proc.id,thief.id);

      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal them
      std::set<TaskContext*> stolen;
      // Need read-write access to the ready queue to try stealing
      {
        AutoLock q_lock(queue_lock);
        for (std::vector<MapperID>::const_iterator it = thieves.begin();
              it != thieves.end(); it++)
        {
          // Get the mapper id out of the buffer
          MapperID stealer = *it;
          
          // Handle a race condition here where some processors can issue steal
          // requests to another processor before the mappers have been initialized
          // on that processor.  There's no correctness problem for ignoring a steal
          // request so just do that.
          if (mapper_objects.size() <= stealer)
            continue;

          // Go through the ready queue and construct the list of tasks
          // that this mapper has access to
          std::vector<const Task*> mapper_tasks;
          for (std::list<TaskContext*>::iterator it = ready_queues[stealer].begin();
                it != ready_queues[stealer].end(); it++)
          {
            // The tasks also must be stealable
            if ((*it)->is_stealable() && ((*it)->map_id == stealer) && !(*it)->is_locally_mapped())
              mapper_tasks.push_back(*it);
          }
          // Now call the mapper and get back the results
          std::set<const Task*> to_steal;
          if (!mapper_tasks.empty())
          {
            // Need read-only access to the mapper vector to access the mapper objects
#ifdef LOW_LEVEL_LOCKS
            AutoLock map_lock(mapping_lock,1,false/*exclusive*/);
#else
            AutoLock map_lock(mapping_lock);
#endif
            // Also need exclusive access to the mapper itself
            AutoLock mapper_lock(mapper_locks[stealer]);
            DetailedTimer::ScopedPush sp(TIME_MAPPER);
            mapper_objects[stealer]->permit_task_steal(thief, mapper_tasks, to_steal);
          }
          // Add the results to the set of stolen tasks
          // Do this explicitly since we need to upcast the pointers
          if (!to_steal.empty())
          {
            for (std::set<const Task*>::iterator it = to_steal.begin();
                  it != to_steal.end(); it++)
            {
              // Mark the task as stolen
              Task *t = const_cast<Task*>(*it);
              TaskContext *tt = static_cast<TaskContext*>(t);
#ifdef DEBUG_HIGH_LEVEL
              assert(stolen.find(tt) == stolen.end());
#endif
              // Make sure we're going to be able to steal this task
              if (tt->prepare_steal())
              {
                stolen.insert(tt);
              }
            }
            // Also remove any stolen tasks from the queue
            std::list<TaskContext*>::iterator it = ready_queues[stealer].begin();
            while (it != ready_queues[stealer].end())
            {
              if (stolen.find(*it) != stolen.end())
                it = ready_queues[stealer].erase(it);
              else
                it++;
            }
          }
          else
          {
            AutoLock thief_lock(thieving_lock);
            // Mark a failed steal attempt
            failed_thiefs.insert(std::pair<MapperID,Processor>(stealer,thief));
          }
        }
      } // Release the queue lock
      
      // Send the tasks back
      if (!stolen.empty())
      {

        // Send the tasks back  
        bool still_local = runtime->send_tasks(thief, stolen);

        // Delete any remote tasks that we will no longer have a reference to
        for (std::set<TaskContext*>::iterator it = stolen.begin();
              it != stolen.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          log_task(LEVEL_DEBUG,"task %s (ID %d) with unique id %d stolen from processor %x",
                                (*it)->variants->name,
                                (*it)->task_id,(*it)->get_unique_id(),local_proc.id);
#endif
          // If they are remote, deactivate the instance
          // If it's not remote, its parent will deactivate it
          if (!still_local && (*it)->is_remote())
            (*it)->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::process_advertisement(Processor advertiser, MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
      {
        // Need exclusive access to the list steal data structures
        AutoLock steal_lock(stealing_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(outstanding_steals.find(map_id) != outstanding_steals.end());
#endif
        std::set<Processor> &procs = outstanding_steals[map_id];
        // Erase the utility users from the set
        procs.erase(advertiser);
      }
      // Enable the idle task since some mappers might make new decisions
      enable_scheduler();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::add_to_dependence_queue(GeneralizedOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      // The call to this function should always come from 
      // an application task calling into the runtime, so 
      // there are no other runtime locks held.  This allows
      // us to do non-blocking locking so processors are always busy.
      {
        Event dep_event = dependence_lock.lock(0,true/*exclusive*/); 
        dep_event.wait(true/*block*/);
        dependence_queue.push_back(op);
        dependence_lock.unlock();
      }
      // Now enable the scheduler without block-waiting on an event
      // Note by not continuing to hold the dependence queue lock
      // while setting this, we can cause spurious runs of the
      // scheduler, but that is better than blocking a compute thread.
      {
        Event idle_event = idle_lock.lock(0,true/*exclusive*/);
        idle_event.wait(true/*block*/);
        if (!idle_task_enabled)
        {
          idle_task_enabled = true;
          Processor copy = local_proc;
          copy.enable_idle_task();
        }
        idle_lock.unlock();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::add_to_ready_queue(TaskContext *task)
    //--------------------------------------------------------------------------------------------
    {
      // Anytime we add something to ready queue, update its mappers so
      // that they point to this processor's mappers
      initialize_mappable(task, task->get_mapper_id());
      AutoLock q_lock(queue_lock);
      ready_queues[task->map_id].push_back(task);
      enable_scheduler();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::add_to_other_queue(GeneralizedOperation *op)
    //--------------------------------------------------------------------------------------------
    {
      // Anytime we add something to our queue to be mapped, update
      // its mappers so they point to this processor's mappers
      initialize_mappable(op, op->get_mapper_id());
      AutoLock q_lock(queue_lock);
      other_ready_queue.push_back(op);
      enable_scheduler();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::enable_scheduler(void)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock i_lock(idle_lock);
      if (!idle_task_enabled)
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task(); 
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::increment_outstanding(void)
    //--------------------------------------------------------------------------------------------
    {
      // Assume that we're not going to be holding any other locks when doing this
      Event idle_event = idle_lock.lock(0,true/*exclusive*/);
      idle_event.wait(true/*block*/);
      current_outstanding++;
      idle_lock.unlock();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::decrement_outstanding(void)
    //--------------------------------------------------------------------------------------------
    {
      // Assume that we're not going to be holding any other locks when doing this
      Event idle_event = idle_lock.lock(0,true/*exclusive*/);
      idle_event.wait(true/*block*/);
#ifdef DEBUG_HIGH_LEVEL
      assert(current_outstanding > 0);
#endif
      current_outstanding--;
      if (!idle_task_enabled && (current_outstanding < min_outstanding))
      {
        idle_task_enabled = true;
        Processor copy = local_proc;
        copy.enable_idle_task();
      }
      idle_lock.unlock();
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::lock_dependence_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      // Assume we're not holding any other locks when this is called
      // and that we're only going to be reading the queue
      Event lock_event = dependence_lock.lock(1,false/*exclusive*/);
      lock_event.wait(true/*block*/);
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::unlock_dependence_queue(void)
    //--------------------------------------------------------------------------------------------
    {
      dependence_lock.unlock();
    }

    //--------------------------------------------------------------------------------------------
    const std::list<GeneralizedOperation*>& HighLevelRuntime::ProcessorManager::get_dependence_queue(void) const
    //--------------------------------------------------------------------------------------------
    {
      return dependence_queue;
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::ProcessorManager::advertise(MapperID map_id)
    //--------------------------------------------------------------------------------------------
    {
      // Create a clone of the processors we want to advertise so that
      // we don't call into the high level runtime holding a lock
      std::set<Processor> failed_waiters;
      // Check to see if we have any failed thieves with the mapper id
      {
        AutoLock theif_lock(thieving_lock);
        if (failed_thiefs.lower_bound(map_id) != failed_thiefs.upper_bound(map_id))
        {
          for (std::multimap<MapperID,Processor>::iterator it = failed_thiefs.lower_bound(map_id);
                it != failed_thiefs.upper_bound(map_id); it++)
          {
            failed_waiters.insert(it->second);
          } 
          // Erase all the failed theives
          failed_thiefs.erase(failed_thiefs.lower_bound(map_id),failed_thiefs.upper_bound(map_id));
        }
      }
      if (!failed_waiters.empty())
        runtime->send_advertisements(failed_waiters, map_id, local_proc);
    }

#ifdef INORDER_EXECUTION
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_inorder_scheduling(void)
    //--------------------------------------------------------------------------------------------
    {
      std::vector<TaskContext*> drain_to_map;
      std::map<Context,TaskContext*> tasks_to_map;
      std::map<Context,GeneralizedOperation*> ops_to_map;
      // Take the queue lock and get the next operations
      {
        AutoLock q_lock(queue_lock);
        // Get all the tasks out of the drain queue
        drain_to_map.insert(drain_to_map.end(),drain_queue.begin(),drain_queue.end());
        drain_queue.clear();
        for (std::map<Context,InorderQueue*>::const_iterator it = inorder_queues.begin();
              it != inorder_queues.end(); it++)
        {
          if (it->second->has_ready())
          {
            it->second->schedule_next(it->first,tasks_to_map,ops_to_map);
          }
        }
      }
      std::vector<TaskContext*> failed_drain;
      for (std::vector<TaskContext*>::const_iterator it = drain_to_map.begin();
            it != drain_to_map.end(); it++)
      {
        bool success = (*it)->perform_operation();
        if (!success)
          failed_drain.push_back(*it);
      }
      if (!failed_drain.empty())
      {
        AutoLock q_lock(queue_lock);
        drain_queue.insert(drain_queue.end(),failed_drain.begin(),failed_drain.end());
      }
      // Perform all the operations and tasks
      for (std::map<Context,TaskContext*>::const_iterator it = tasks_to_map.begin();
            it != tasks_to_map.end(); it++)
      {
        bool success = it->second->perform_operation();
        if (!success)
        {
          AutoLock q_lock(queue_lock);
          inorder_queues[it->first]->requeue_task(it->second);
        }
      }
      for (std::map<Context,GeneralizedOperation*>::const_iterator it = ops_to_map.begin();
            it != ops_to_map.end(); it++)
      {
        bool success = it->second->perform_operation();
        if (!success)
        {
          AutoLock q_lock(queue_lock);
          inorder_queues[it->first]->requeue_op(it->second);
        }
      }
      // Now check to see whether any of the queues have inorder tasks
      {
        AutoLock q_lock(queue_lock);
        bool has_ready = !drain_queue.empty();
        for (std::map<Context,InorderQueue*>::const_iterator it = inorder_queues.begin();
              it != inorder_queues.end(); it++)
        {
          if (it->second->has_ready())
          {
            has_ready = true;
            break;
          }
        }
        if (!has_ready)
        {
          idle_task_enabled = false;
          UtilityProcessor copy = utility_proc;
          copy.disable_idle_task(); 
        }
      }
    }
#endif

#ifdef DYNAMIC_TESTS
    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::perform_dynamic_tests(void)
    //--------------------------------------------------------------------------------------------
    {
      // Copy out the dynamic forests while holding the lock
      std::set<RegionTreeForest*> targets;
      {
        AutoLock f_lock(forest_lock);
        targets.insert(dynamic_forests.begin(),dynamic_forests.end());
        // Empty out the current buffer
        dynamic_forests.clear();
      }
      // Now we can do our thing
      for (std::set<RegionTreeForest*>::const_iterator it = targets.begin();
            it != targets.end(); it++)
      {
        RegionTreeForest *forest = *it;
        forest->lock_context();
        if (forest->fix_dynamic_test_set())
        {
          forest->unlock_context();
          forest->perform_dynamic_tests();
          forest->lock_context();
          forest->publish_dynamic_test_results();
        }
        forest->unlock_context();
      }
    }

    //--------------------------------------------------------------------------------------------
    void HighLevelRuntime::request_dynamic_tests(RegionTreeForest *forest)
    //--------------------------------------------------------------------------------------------
    {
      AutoLock f_lock(forest_lock);
      dynamic_forests.push_back(forest);
    }
#endif

  };
};


// EOF

