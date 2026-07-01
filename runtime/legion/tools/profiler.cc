/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/tools/profiler.h"
#include "legion/contexts/inner.h"
#include "legion/operations/operation.h"
#include "legion/tools/serializer.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    ArrivalInfo::ArrivalInfo(void)
      : arrival_time(0), trigger_time(std::numeric_limits<timestamp_t>::min())
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ArrivalInfo::ArrivalInfo(const ArrivalInfo& rhs)
      : arrival_time(rhs.arrival_time), trigger_time(rhs.trigger_time.load()),
        arrival_precondition(rhs.arrival_precondition), fevent(rhs.fevent)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ArrivalInfo::ArrivalInfo(LgEvent pre)
      : arrival_time(Realm::Clock::current_time_in_nanoseconds()),
        trigger_time(arrival_time), arrival_precondition(pre),
        fevent(implicit_fevent)
    //--------------------------------------------------------------------------
    {
      legion_assert(fevent.exists());
    }

    //--------------------------------------------------------------------------
    ArrivalInfo::ArrivalInfo(
        timestamp_t arrival, timestamp_t trigger, LgEvent pre, LgEvent f)
      : arrival_time(arrival), trigger_time(trigger), arrival_precondition(pre),
        fevent(f)
    //--------------------------------------------------------------------------
    { }

    /*static*/ const ArrivalInfo BarrierArrivalReduction::identity =
        ArrivalInfo();

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void BarrierArrivalReduction::apply<true>(
        LHS& lhs, const RHS& rhs)
    //--------------------------------------------------------------------------
    {
      if (lhs.trigger_time < rhs.trigger_time)
      {
        lhs.arrival_time = rhs.arrival_time;
        lhs.arrival_precondition = rhs.arrival_precondition;
        lhs.fevent = rhs.fevent;
        lhs.trigger_time.store(rhs.trigger_time.load());
      }
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void BarrierArrivalReduction::apply<false>(
        LHS& lhs, const RHS& rhs)
    //--------------------------------------------------------------------------
    {
      timestamp_t previous = lhs.trigger_time.load();
      while (true)
      {
        // Spin until the previous is not the sentinel
        while (previous == SENTINEL) previous = lhs.trigger_time.load();
        // Quick test to see if we even need to do the compare and swap
        if (rhs.trigger_time <= previous)
          break;
        // Once it's not the sentinel then do a compare and swap to
        // see if we can add ourselves as the sentinel
        if (!lhs.trigger_time.compare_exchange_weak(previous, SENTINEL))
          // Exchange was not successful so go around again
          continue;
        // Save our state since we know we're later than the previous
        lhs.arrival_time = rhs.arrival_time;
        lhs.arrival_precondition = rhs.arrival_precondition;
        lhs.fevent = rhs.fevent;
        lhs.trigger_time.store(rhs.trigger_time);
        break;
      }
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void BarrierArrivalReduction::fold<true>(
        RHS& rhs1, const RHS& rhs2)
    //--------------------------------------------------------------------------
    {
      if (rhs1.trigger_time < rhs2.trigger_time)
      {
        rhs1.arrival_time = rhs2.arrival_time;
        rhs1.arrival_precondition = rhs2.arrival_precondition;
        rhs1.fevent = rhs2.fevent;
        rhs1.trigger_time.store(rhs2.trigger_time.load());
      }
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void BarrierArrivalReduction::fold<false>(
        LHS& rhs1, const RHS& rhs2)
    //--------------------------------------------------------------------------
    {
      timestamp_t previous = rhs1.trigger_time.load();
      while (true)
      {
        // Spin until the previous is not the sentinel
        while (previous == SENTINEL) previous = rhs1.trigger_time.load();
        // Quick test to see if we even need to do the compare and swap
        if (rhs2.trigger_time <= previous)
          break;
        // Once it's not the sentinel then do a compare and swap to
        // see if we can add ourselves as the sentinel
        if (!rhs1.trigger_time.compare_exchange_weak(previous, SENTINEL))
          // Exchange was not successful so go around again
          continue;
        // Save our state since we know we're later than the previous
        rhs1.arrival_time = rhs2.arrival_time;
        rhs1.arrival_precondition = rhs2.arrival_precondition;
        rhs1.fevent = rhs2.fevent;
        rhs1.trigger_time.store(rhs2.trigger_time);
        break;
      }
    }

    //--------------------------------------------------------------------------
    template<size_t ENTRIES>
    SmallNameClosure<ENTRIES>::SmallNameClosure(
        IndexSpaceExpression* expr, ReductionOpID r)
      : copy_expr((expr == nullptr) ? 0 : expr->record_profiler_expression()),
        redop(r)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < ENTRIES; idx++)
        instances[idx] = PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    template<size_t ENTRIES>
    void SmallNameClosure<ENTRIES>::record_instance_name(
        PhysicalInstance instance, LgEvent name, DistributedID subspace)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < ENTRIES; idx++)
      {
        if (!instances[idx].exists())
        {
          instances[idx] = instance;
          names[idx] = name;
          subspaces[idx] = subspace;
          return;
        }
        if (instances[idx] == instance)
        {
          legion_assert(names[idx] == name);
          legion_assert(subspaces[idx] == subspace);
          return;
        }
      }
      // Should not run out of space
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<size_t ENTRIES>
    LgEvent SmallNameClosure<ENTRIES>::find_instance_name(
        PhysicalInstance inst) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < ENTRIES; idx++)
        if (instances[idx] == inst)
          return names[idx];
      // Should always find it before this
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<size_t ENTRIES>
    DistributedID SmallNameClosure<ENTRIES>::find_instance_subspace(
        PhysicalInstance inst) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < ENTRIES; idx++)
        if (instances[idx] == inst)
          return subspaces[idx];
      // Should always find it before this
      std::abort();
    }

    // Explicit instantiations for 1 and 2
    template class SmallNameClosure<1>;
    template class SmallNameClosure<2>;

    //--------------------------------------------------------------------------
    LargeNameClosure::LargeNameClosure(
        IndexSpaceExpression* expr, FieldID f, size_t expected, ReductionOpID r)
      : copy_expr((expr == nullptr) ? 0 : expr->record_profiler_expression()),
        redop(r), fid(f)
    //--------------------------------------------------------------------------
    {
      entries.reserve(expected);
    }

    //--------------------------------------------------------------------------
    void LargeNameClosure::record_instance_name(
        PhysicalInstance instance, LgEvent name, DistributedID subspace)
    //--------------------------------------------------------------------------
    {
      entries.emplace_back(Entry{instance, name, subspace});
    }

    //--------------------------------------------------------------------------
    LgEvent LargeNameClosure::find_instance_name(
        PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      for (const Entry& entry : entries)
        if (entry.instance == instance)
          return entry.name;
      // Should always find it
      std::abort();
    }

    //--------------------------------------------------------------------------
    DistributedID LargeNameClosure::find_instance_subspace(
        PhysicalInstance instance) const
    //--------------------------------------------------------------------------
    {
      for (const Entry& entry : entries)
        if (entry.instance == instance)
          return entry.subspace;
      // Should always find it
      std::abort();
    }

    //--------------------------------------------------------------------------
    LegionProfMarker::LegionProfMarker(const char* _name)
      : name(_name), stopped(false)
    //--------------------------------------------------------------------------
    {
      proc = Realm::Processor::get_executing_processor();
      start = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    LegionProfMarker::~LegionProfMarker()
    //--------------------------------------------------------------------------
    {
      if (!stopped)
        mark_stop();
      log_prof.print(
          "Prof User Info " IDFMT " %llu %llu %s", proc.id, start, stop, name);
    }

    //--------------------------------------------------------------------------
    void LegionProfMarker::mark_stop()
    //--------------------------------------------------------------------------
    {
      stop = Realm::Clock::current_time_in_nanoseconds();
      stopped = true;
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::ProfilingInfo::ProfilingInfo(
        ProfilingResponseHandler* h, UniqueID uid)
      : ProfilingResponseBase(h, uid), creator(implicit_fevent)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(
        LegionProfiler* own, Processor local, LgEvent ext)
      : external_fevent(ext), local_proc(local),
        external_start(
            external_fevent.exists() ?
                Realm::Clock::current_time_in_nanoseconds() :
                0),
        owner(own)
    //--------------------------------------------------------------------------
    {
      if (external_fevent.exists() && !implicit_fevent.exists())
        implicit_fevent = external_fevent;
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(
        LegionProfiler* own, Processor local, LgEvent ext, long long start)
      : external_fevent(ext), local_proc(local), external_start(start),
        owner(own)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LegionProfInstance::~LegionProfInstance(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LegionProfInstance* LegionProfInstance::dump(void)
    //--------------------------------------------------------------------------
    {
      LegionProfInstance* dump = new LegionProfInstance(
          owner, local_proc, external_fevent, external_start);
      operation_instances.swap(dump->operation_instances);
      multi_tasks.swap(dump->multi_tasks);
      slice_owners.swap(dump->slice_owners);
      task_infos.swap(dump->task_infos);
      implicit_infos.swap(dump->implicit_infos);
      gpu_task_infos.swap(dump->gpu_task_infos);
      ispace_rect_desc.swap(dump->ispace_rect_desc);
      ispace_point_desc.swap(dump->ispace_point_desc);
      ispace_empty_desc.swap(dump->ispace_empty_desc);
      field_desc.swap(dump->field_desc);
      field_space_desc.swap(dump->field_space_desc);
      index_part_desc.swap(dump->index_part_desc);
      index_space_desc.swap(dump->index_space_desc);
      index_subspace_desc.swap(dump->index_subspace_desc);
      index_partition_desc.swap(dump->index_partition_desc);
      lr_desc.swap(dump->lr_desc);
      phy_inst_rdesc.swap(dump->phy_inst_rdesc);
      phy_inst_layout_rdesc.swap(dump->phy_inst_layout_rdesc);
      phy_inst_dim_order_rdesc.swap(dump->phy_inst_dim_order_rdesc);
      phy_inst_spaces.swap(dump->phy_inst_spaces);
      phy_inst_usage.swap(dump->phy_inst_usage);
      index_space_size_desc.swap(dump->index_space_size_desc);
      meta_infos.swap(dump->meta_infos);
      message_infos.swap(dump->message_infos);
      copy_infos.swap(dump->copy_infos);
      fill_infos.swap(dump->fill_infos);
      inst_timeline_infos.swap(dump->inst_timeline_infos);
      partition_infos.swap(dump->partition_infos);
      mapper_call_infos.swap(dump->mapper_call_infos);
      runtime_call_infos.swap(dump->runtime_call_infos);
      application_call_infos.swap(dump->application_call_infos);
      async_effect_infos.swap(dump->async_effect_infos);
      event_wait_infos.swap(dump->event_wait_infos);
      event_merger_infos.swap(dump->event_merger_infos);
      event_trigger_infos.swap(dump->event_trigger_infos);
      event_poison_infos.swap(dump->event_poison_infos);
      barrier_arrival_infos.swap(dump->barrier_arrival_infos);
      reservation_acquire_infos.swap(dump->reservation_acquire_infos);
      instance_ready_infos.swap(dump->instance_ready_infos);
      instance_redistrict_infos.swap(dump->instance_redistrict_infos);
      completion_queue_infos.swap(dump->completion_queue_infos);
      make_valid_infos.swap(dump->make_valid_infos);
      fetch_metadata_infos.swap(dump->fetch_metadata_infos);
      external_wait_infos.swap(dump->external_wait_infos);
      prof_task_infos.swap(dump->prof_task_infos);
      footprint = 0;
      return dump;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_operation(Operation* op)
    //--------------------------------------------------------------------------
    {
      operation_instances.emplace_back(OperationInstance());
      OperationInstance& inst = operation_instances.back();
      inst.op_id = op->get_unique_op_id();
      InnerContext* parent_ctx = op->get_context();
      // Legion prof uses ULLONG_MAX to represent the unique IDs of the root
      inst.parent_id = (parent_ctx->get_depth() < 0) ?
                           std::numeric_limits<UniqueID>::max() :
                           parent_ctx->get_unique_id();
      inst.kind = op->get_operation_kind();
      Provenance* prov = op->get_provenance();
      if (prov != nullptr)
        inst.provenance = prov->pid;
      else
        inst.provenance = 0;
      owner->update_footprint(sizeof(OperationInstance), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_multi_task(Operation* op, TaskID task_id)
    //--------------------------------------------------------------------------
    {
      multi_tasks.emplace_back(MultiTask());
      MultiTask& task = multi_tasks.back();
      task.op_id = op->get_unique_op_id();
      task.task_id = task_id;
      owner->update_footprint(sizeof(MultiTask), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_slice_owner(UniqueID pid, UniqueID id)
    //--------------------------------------------------------------------------
    {
      slice_owners.emplace_back(SliceOwner());
      SliceOwner& task = slice_owners.back();
      task.parent_id = pid;
      task.op_id = id;
      owner->update_footprint(sizeof(SliceOwner), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_space_rect(
        IndexSpaceRectDesc& _ispace_rect_desc)
    //--------------------------------------------------------------------------
    {
      ispace_rect_desc.emplace_back(IndexSpaceRectDesc());
      IndexSpaceRectDesc& desc = ispace_rect_desc.back();
      desc = _ispace_rect_desc;
      owner->update_footprint(sizeof(IndexSpaceRectDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_space_point(
        IndexSpacePointDesc& _ispace_point_desc)
    //--------------------------------------------------------------------------
    {
      ispace_point_desc.emplace_back(IndexSpacePointDesc());
      IndexSpacePointDesc& desc = ispace_point_desc.back();
      desc = _ispace_point_desc;
      owner->update_footprint(sizeof(IndexSpacePointDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_empty_index_space(DistributedID handle)
    //--------------------------------------------------------------------------
    {
      ispace_empty_desc.emplace_back(IndexSpaceEmptyDesc());
      IndexSpaceEmptyDesc& desc = ispace_empty_desc.back();
      desc.unique_id = handle;
      owner->update_footprint(sizeof(IndexSpaceEmptyDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_field(
        UniqueID unique_id, unsigned field_id, size_t size, const char* name)
    //--------------------------------------------------------------------------
    {
      field_desc.emplace_back(FieldDesc());
      FieldDesc& desc = field_desc.back();
      desc.unique_id = unique_id;
      desc.field_id = field_id;
      desc.size = (long long)size;
      desc.name = strdup(name);
      const size_t diff = sizeof(FieldDesc) + strlen(name);
      owner->update_footprint(diff, this);
    }
    //--------------------------------------------------------------------------
    void LegionProfInstance::register_field_space(
        UniqueID unique_id, const char* name)
    //--------------------------------------------------------------------------
    {
      field_space_desc.emplace_back(FieldSpaceDesc());
      FieldSpaceDesc& desc = field_space_desc.back();
      desc.unique_id = unique_id;
      desc.name = strdup(name);
      const size_t diff = sizeof(FieldSpaceDesc) + strlen(name);
      owner->update_footprint(diff, this);
    }
    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_part(
        UniqueID unique_id, const char* name)
    //--------------------------------------------------------------------------
    {
      index_part_desc.emplace_back(IndexPartDesc());
      IndexPartDesc& desc = index_part_desc.back();
      desc.unique_id = unique_id;
      desc.name = strdup(name);
      const size_t diff = sizeof(IndexPartDesc) + strlen(name);
      owner->update_footprint(diff, this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_space(
        UniqueID unique_id, const char* name)
    //--------------------------------------------------------------------------
    {
      index_space_desc.emplace_back(IndexSpaceDesc());
      IndexSpaceDesc& desc = index_space_desc.back();
      desc.unique_id = unique_id;
      desc.name = strdup(name);
      const size_t diff = sizeof(IndexSpaceDesc) + strlen(name);
      owner->update_footprint(diff, this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_subspace(
        DistributedID parent_id, DistributedID unique_id,
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      index_subspace_desc.emplace_back(IndexSubSpaceDesc());
      IndexSubSpaceDesc& desc = index_subspace_desc.back();
      desc.parent_id = parent_id;
      desc.unique_id = unique_id;
      owner->update_footprint(sizeof(IndexSubSpaceDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_partition(
        DistributedID parent_id, DistributedID unique_id, bool disjoint,
        LegionColor point)
    //--------------------------------------------------------------------------
    {
      index_partition_desc.emplace_back(IndexPartitionDesc());
      IndexPartitionDesc& desc = index_partition_desc.back();
      desc.parent_id = parent_id;
      desc.unique_id = unique_id;
      desc.disjoint = disjoint;
      desc.point = point;
      owner->update_footprint(sizeof(IndexPartitionDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_logical_region(
        DistributedID index_space, DistributedID field_space, unsigned tree_id,
        const char* name)
    //--------------------------------------------------------------------------
    {
      lr_desc.emplace_back(LogicalRegionDesc());
      LogicalRegionDesc& desc = lr_desc.back();
      desc.ispace_id = index_space;
      desc.fspace_id = field_space;
      desc.tree_id = tree_id;
      desc.name = strdup(name);
      const size_t diff = sizeof(LogicalRegionDesc) + strlen(name);
      owner->update_footprint(diff, this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_field(
        LgEvent inst_uid, unsigned field_id, DistributedID field_sp,
        unsigned align, bool align_set, EqualityKind eqk)
    //--------------------------------------------------------------------------
    {
      phy_inst_layout_rdesc.emplace_back(PhysicalInstLayoutDesc());
      PhysicalInstLayoutDesc& pdesc = phy_inst_layout_rdesc.back();
      pdesc.inst_uid = inst_uid;
      pdesc.field_id = field_id;
      pdesc.fspace_id = field_sp;
      pdesc.eqk = eqk;
      pdesc.alignment = align;
      pdesc.has_align = align_set;
      owner->update_footprint(sizeof(PhysicalInstLayoutDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_region(
        LgEvent inst_uid, LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      phy_inst_rdesc.emplace_back(PhysicalInstRegionDesc());
      PhysicalInstRegionDesc& phy_instance_rdesc = phy_inst_rdesc.back();
      phy_instance_rdesc.inst_uid = inst_uid;
      phy_instance_rdesc.ispace_id = handle.get_index_space().get_id();
      phy_instance_rdesc.fspace_id = handle.get_field_space().get_id();
      phy_instance_rdesc.tree_id = handle.get_tree_id();
      owner->update_footprint(sizeof(PhysicalInstRegionDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_layout(
        LgEvent unique_event, FieldSpace fs, const LayoutConstraintSet& lc)
    //--------------------------------------------------------------------------
    {
      // get fields_constraints
      // get_alignment_constraints
      std::map<FieldID, AlignmentConstraint> align_map;
      const std::vector<AlignmentConstraint>& alignment_constraints =
          lc.alignment_constraints;
      for (const AlignmentConstraint& constraint : alignment_constraints)
        align_map[constraint.fid] = constraint;
      const std::vector<FieldID>& fields = lc.field_constraint.field_set;
      for (const FieldID& field : fields)
      {
        std::map<FieldID, AlignmentConstraint>::const_iterator align =
            align_map.find(field);
        bool has_align = false;
        unsigned alignment = 0;
        EqualityKind eqk = LEGION_LT_EK;
        if (align != align_map.end())
        {
          has_align = true;
          alignment = align->second.alignment;
          eqk = align->second.eqk;
        }
        register_physical_instance_field(
            unique_event, field, fs.get_id(), alignment, has_align, eqk);
      }
      const std::vector<DimensionKind>& dim_ordering_constr =
          lc.ordering_constraint.ordering;
      unsigned dim = 0;
      for (const DimensionKind& dimension : dim_ordering_constr)
      {
        register_physical_instance_dim_order(unique_event, dim, dimension);
        dim++;
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_dim_order(
        LgEvent inst_uid, unsigned dim, DimensionKind k)
    //--------------------------------------------------------------------------
    {
      phy_inst_dim_order_rdesc.emplace_back(PhysicalInstDimOrderDesc());
      PhysicalInstDimOrderDesc& phy_instance_d_rdesc =
          phy_inst_dim_order_rdesc.back();
      phy_instance_d_rdesc.inst_uid = inst_uid;
      phy_instance_d_rdesc.dim = dim;
      phy_instance_d_rdesc.k = k;
      owner->update_footprint(sizeof(PhysicalInstDimOrderDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_spaces(
        LgEvent inst_uid, DistributedID union_space, DistributedID piece_space)
    //--------------------------------------------------------------------------
    {
      PhysicalInstanceSpaces& spaces =
          phy_inst_spaces.emplace_back(PhysicalInstanceSpaces());
      spaces.inst_uid = inst_uid;
      spaces.union_space = union_space;
      spaces.piece_space = piece_space;
      owner->update_footprint(sizeof(PhysicalInstanceSpaces), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_physical_instance_use(
        LgEvent inst_uid, UniqueID op_id, DistributedID index_expr,
        FieldID field, PrivilegeMode mode, ReductionOpID redop,
        timestamp_t start, timestamp_t stop, int index)
    //--------------------------------------------------------------------------
    {
      PhysicalInstanceUsage& usage =
          phy_inst_usage.emplace_back(PhysicalInstanceUsage());
      usage.inst_uid = inst_uid;
      usage.fevent = implicit_fevent;
      usage.op_id = op_id;
      usage.index_expr = index_expr;
      usage.mode = mode & LEGION_READ_WRITE;  // only need the privilege bits
      usage.redop = redop;
      usage.field = field;
      usage.index = index;
      usage.start = start;
      usage.stop = stop;
      owner->update_footprint(sizeof(PhysicalInstanceUsage), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_index_space_size(
        UniqueID id, unsigned long long dense_size,
        unsigned long long sparse_size, bool is_sparse)
    //--------------------------------------------------------------------------
    {
      index_space_size_desc.emplace_back(IndexSpaceSizeDesc());
      IndexSpaceSizeDesc& size_info = index_space_size_desc.back();
      size_info.id = id;
      size_info.dense_size = dense_size;
      size_info.sparse_size = sparse_size;
      size_info.is_sparse = is_sparse;
      owner->update_footprint(sizeof(IndexSpaceSizeDesc), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_event_merger(
        LgEvent result, const LgEvent* preconditions, size_t count)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      // Realm can return one of the preconditions as the result of
      // an event merger as an optimization, to handle that we check
      // if the result is the same as any of the preconditions, if it
      // is then there is nothing needed for us to record
      for (unsigned idx = 0; idx < count; idx++)
        if (preconditions[idx] == result)
          return;
      EventMergerInfo& info =
          event_merger_infos.emplace_back(EventMergerInfo());
      // Take the timing measurement of when this happened first
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.preconditions.resize(count);
      for (unsigned idx = 0; idx < count; idx++)
      {
        info.preconditions[idx] = preconditions[idx];
        if (preconditions[idx].is_barrier())
          record_barrier_use(preconditions[idx], implicit_unique_op_id);
      }
      info.fevent = implicit_fevent;
      owner->update_footprint(sizeof(info) + count * sizeof(LgEvent), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_event_trigger(LgEvent result, LgEvent pre)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      EventTriggerInfo& info =
          event_trigger_infos.emplace_back(EventTriggerInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.precondition = pre;
      if (pre.is_barrier())
        record_barrier_use(pre, implicit_unique_op_id);
      info.fevent = implicit_fevent;
      // See if we're triggering this node on the same node where it was made
      // If not we need to eventually notify the node where it was made that
      // it was triggered here and what the fevent was for it
      const Realm::ID id(result.id);
      const AddressSpaceID creator_node = id.event_creator_node();
      if (creator_node != runtime->address_space)
      {
        // Triggered on a remote node, send a message back to the creator
        // node of the event so that even partial profile loading can know
        // where the triggering of this event occurred
        ProfilerEventTriggerMessage rez;
        rez.serialize(info);
        rez.dispatch(creator_node);
      }
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_event_poison(LgEvent result)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      EventPoisonInfo& info =
          event_poison_infos.emplace_back(EventPoisonInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.fevent = implicit_fevent;
      // See if we're poisoning this node on the same node where it was made
      // If not we need to eventually notify the node where it was made that
      // it was triggered here and what the fevent was for it
      const Realm::ID id(result.id);
      const AddressSpaceID creator_node = id.event_creator_node();
      if (creator_node != runtime->address_space)
      {
        // Triggered on a remote node, send a message back to the creator
        // node of the event so that even partial profile loading can know
        // where the triggering of this event occurred
        ProfilerEventPoisonMessage rez;
        rez.serialize(info);
        rez.dispatch(creator_node);
      }
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_barrier_arrival(LgEvent result, LgEvent pre)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      legion_assert(result.is_barrier());
      legion_assert(owner->all_critical_arrivals);
      BarrierArrivalInfo& info =
          barrier_arrival_infos.emplace_back(BarrierArrivalInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.precondition = pre;
      if (pre.is_barrier())
        record_barrier_use(pre, implicit_unique_op_id);
      legion_assert(implicit_fevent.exists());
      info.fevent = implicit_fevent;
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_barrier_use(LgEvent bar, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      legion_assert(bar.is_barrier());
      // We don't need to record this if we're recording all barrier arrivals
      // since the profiler will be able to look at all the log files and
      // do this by itself
      if (owner->no_critical_paths || owner->all_critical_arrivals)
        return;
      Realm::Barrier barrier;
      barrier.id = bar.id;
      barrier.timestamp = 0;
      // See if the barrier has already triggered
      bool poisoned = false;
      if (barrier.has_triggered_faultaware(poisoned) || poisoned)
      {
        // See how many generations to record as we need to record all of
        // them from the previous generation up to now
        Realm::Barrier previous;
        if (owner->update_previous_recorded_barrier(barrier, previous))
        {
          while (barrier.id != previous.id)
          {
            ArrivalInfo arrival_info;
            // TODO: what happens if the barrier is poisoned
            legion_no_skip_assert(
                barrier.get_result(&arrival_info, sizeof(arrival_info)));
            legion_assert(arrival_info.fevent.exists());
            BarrierArrivalInfo& info =
                barrier_arrival_infos.emplace_back(BarrierArrivalInfo());
            info.result = LgEvent(barrier);
            info.fevent = arrival_info.fevent;
            info.precondition = arrival_info.arrival_precondition;
            info.performed = arrival_info.arrival_time;
            owner->update_footprint(sizeof(info), this);
            barrier = barrier.get_previous_phase();
          }
        }
      }
      else
        // The barrier hasn't triggered yet so launch a profiling task to
        // record it has triggered
        owner->profile_barrier_trigger(barrier, uid);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_reservation_acquire(
        Reservation r, LgEvent result, LgEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      ReservationAcquireInfo& info =
          reservation_acquire_infos.emplace_back(ReservationAcquireInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.precondition = precondition;
      if (precondition.is_barrier())
        record_barrier_use(precondition, implicit_unique_op_id);
      info.reservation = r;
      info.fevent = implicit_fevent;
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_instance_ready(
        LgEvent result, LgEvent unique_event, LgEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      InstanceReadyInfo& info =
          instance_ready_infos.emplace_back(InstanceReadyInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.unique = unique_event;
      info.precondition = precondition;
      if (precondition.is_barrier())
        record_barrier_use(precondition, implicit_unique_op_id);
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_instance_redistrict(
        LgEvent& result, LgEvent previous_unique, LgEvent next_unique,
        LgEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      // If the result is the same as the precondition make a new event
      if (result == precondition)
        Runtime::rename_event(result);
      InstanceRedistrictInfo& info =
          instance_redistrict_infos.emplace_back(InstanceRedistrictInfo());
      info.performed = Realm::Clock::current_time_in_nanoseconds();
      info.result = result;
      info.previous = previous_unique;
      info.next = next_unique;
      info.precondition = precondition;
      if (precondition.is_barrier())
        record_barrier_use(precondition, implicit_unique_op_id);
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_completion_queue_event(
        LgEvent result, LgEvent fevent, timestamp_t performed,
        const LgEvent* preconditions, size_t count)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      // Realm can return one of the preconditions as the result of
      // an event merger as an optimization, to handle that we check
      // if the result is the same as any of the preconditions, if it
      // is then there is nothing needed for us to record
      for (unsigned idx = 0; idx < count; idx++)
        if (preconditions[idx] == result)
          return;
      CompletionQueueInfo& info =
          completion_queue_infos.emplace_back(CompletionQueueInfo());
      info.result = result;
      info.preconditions.resize(count);
      for (unsigned idx = 0; idx < count; idx++)
      {
        info.preconditions[idx] = preconditions[idx];
        if (preconditions[idx].is_barrier())
          record_barrier_use(preconditions[idx], implicit_unique_op_id);
      }
      info.fevent = fevent;
      info.performed = performed;
      owner->update_footprint(sizeof(info) + count * sizeof(LgEvent), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_make_valid(
        LgEvent event, DistributedID space)
    //--------------------------------------------------------------------------
    {
      if (!event.exists() || owner->no_critical_paths)
        return;
      bool poisoned = false;
      if (event.has_triggered_faultaware(poisoned) || poisoned)
      {
        MakeValidInfo& info = make_valid_infos.emplace_back(MakeValidInfo());
        // Do this first to make it look like it triggered before
        // we created the event which it probably did
        info.triggered = Realm::Clock::current_time_in_nanoseconds();
        info.created = Realm::Clock::current_time_in_nanoseconds();
        info.result = event;
        info.space = space;
        info.fevent = implicit_fevent;
        owner->update_footprint(sizeof(info), this);
      }
      else
        owner->measure_event_trigger(
            event, LegionProfiler::MAKE_VALID_EFFECT, implicit_fevent, space);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_fetch_metadata(
        LgEvent event, LgEvent inst_uid)
    //--------------------------------------------------------------------------
    {
      if (!event.exists() || owner->no_critical_paths)
        return;
      bool poisoned = false;
      if (event.has_triggered_faultaware(poisoned) || poisoned)
      {
        FetchMetadataInfo& info =
            fetch_metadata_infos.emplace_back(FetchMetadataInfo());
        // Do this first to make it look like it triggered before
        // we created the event which it probably did
        info.triggered = Realm::Clock::current_time_in_nanoseconds();
        info.created = Realm::Clock::current_time_in_nanoseconds();
        info.result = event;
        info.inst_uid = inst_uid;
        info.fevent = implicit_fevent;
        owner->update_footprint(sizeof(info), this);
      }
      else
        owner->measure_event_trigger(
            event, LegionProfiler::FETCH_METADATA_EFFECT, implicit_fevent,
            inst_uid.id);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_task(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::OperationProcessorUsage& usage)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();
      Realm::ProfilingMeasurements::OperationEventWaits waits;
      if (!response.get_measurement(waits))
        std::abort();
      legion_assert(timeline.is_valid());
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);
      Realm::ProfilingMeasurements::OperationTimelineGPU timeline_gpu;
      if (response.get_measurement(timeline_gpu))
      {
        legion_assert(timeline_gpu.is_valid());
        gpu_task_infos.emplace_back(GPUTaskInfo());
        GPUTaskInfo& info = gpu_task_infos.back();
        info.op_id = prof_info->op_id;
        info.task_id = prof_info->id;
        info.variant_id = prof_info->extra.id2;
        info.proc_id = usage.proc.id;
        info.create = timeline.create_time;
        info.ready = timeline.ready_time;
        info.start = timeline.start_time;
        info.stop = timeline.end_time;

        // record gpu time
        info.gpu_start = timeline_gpu.start_time;
        info.gpu_stop = timeline_gpu.end_time;

        unsigned num_intervals = waits.intervals.size();
        if (num_intervals > 0)
        {
          for (unsigned idx = 0; idx < num_intervals; ++idx)
          {
            info.wait_intervals.emplace_back(WaitInfo());
            WaitInfo& wait_info = info.wait_intervals.back();
            wait_info.wait_start = waits.intervals[idx].wait_start;
            wait_info.wait_ready = waits.intervals[idx].wait_ready;
            wait_info.wait_end = waits.intervals[idx].wait_end;
            wait_info.wait_event = LgEvent(waits.intervals[idx].wait_event);
          }
        }
        info.creator = prof_info->creator;
        info.critical = prof_info->critical;
        Realm::ProfilingMeasurements::OperationFinishEvent finish;
        if (!response.get_measurement(finish))
          std::abort();
        info.finish_event = LgEvent(finish.finish_event);
        const size_t diff =
            sizeof(GPUTaskInfo) + num_intervals * sizeof(WaitInfo);
        owner->update_footprint(diff, this);
      }
      else
      {
        task_infos.emplace_back(TaskInfo());
        TaskInfo& info = task_infos.back();
        info.op_id = prof_info->op_id;
        info.task_id = prof_info->id;
        info.variant_id = prof_info->extra.id2;
        info.proc_id = usage.proc.id;
        info.create = timeline.create_time;
        info.ready = timeline.ready_time;
        info.start = timeline.start_time;
        // use complete_time instead of end_time to include async work
        info.stop = timeline.complete_time;
        unsigned num_intervals = waits.intervals.size();
        if (num_intervals > 0)
        {
          for (unsigned idx = 0; idx < num_intervals; ++idx)
          {
            info.wait_intervals.emplace_back(WaitInfo());
            WaitInfo& wait_info = info.wait_intervals.back();
            wait_info.wait_start = waits.intervals[idx].wait_start;
            wait_info.wait_ready = waits.intervals[idx].wait_ready;
            wait_info.wait_end = waits.intervals[idx].wait_end;
            wait_info.wait_event = LgEvent(waits.intervals[idx].wait_event);
          }
        }
        info.creator = prof_info->creator;
        info.critical = prof_info->critical;
        Realm::ProfilingMeasurements::OperationFinishEvent finish;
        if (!response.get_measurement(finish))
          std::abort();
        info.finish_event = LgEvent(finish.finish_event);
        const size_t diff = sizeof(TaskInfo) + num_intervals * sizeof(WaitInfo);
        owner->update_footprint(diff, this);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_meta(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::OperationProcessorUsage& usage)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();
      Realm::ProfilingMeasurements::OperationEventWaits waits;
      if (!response.get_measurement(waits))
        std::abort();
      legion_assert(timeline.is_valid());
      meta_infos.emplace_back(MetaInfo());
      MetaInfo& info = meta_infos.back();
      info.op_id = prof_info->op_id;
      info.lg_id = prof_info->id;
      info.proc_id = usage.proc.id;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      unsigned num_intervals = waits.intervals.size();
      if (num_intervals > 0)
      {
        for (unsigned idx = 0; idx < num_intervals; ++idx)
        {
          info.wait_intervals.emplace_back(WaitInfo());
          WaitInfo& wait_info = info.wait_intervals.back();
          wait_info.wait_start = waits.intervals[idx].wait_start;
          wait_info.wait_ready = waits.intervals[idx].wait_ready;
          wait_info.wait_end = waits.intervals[idx].wait_end;
          wait_info.wait_event = LgEvent(waits.intervals[idx].wait_event);
        }
      }
      info.creator = prof_info->creator;
      info.critical = prof_info->critical;
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);
      Realm::ProfilingMeasurements::OperationFinishEvent finish;
      if (!response.get_measurement(finish))
        std::abort();
      info.finish_event = LgEvent(finish.finish_event);
      const size_t diff = sizeof(MetaInfo) + num_intervals * sizeof(WaitInfo);
      owner->update_footprint(diff, this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_message(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::OperationProcessorUsage& usage)
    //--------------------------------------------------------------------------
    {
      Realm::ProfilingMeasurements::OperationFinishEvent finish;
      if (!response.get_measurement(finish))
        std::abort();
      // Do this first to avoid leaking anything
      const LgEvent original_fevent = owner->find_message_fevent(
          LgEvent(finish.finish_event), true /*remove*/);
      // Do a quick check to see if this is a message task we're profiling
      // If it is then we only profile it if we're self-profiling the profiler
      const MessageKind kind = (MessageKind)(prof_info->id - LG_MESSAGE_ID);
      legion_assert(kind < LAST_SEND_KIND);
      const VirtualChannelKind vc = MessageManager::find_message_vc(kind);
      if ((vc == PROFILING_VIRTUAL_CHANNEL) && !owner->self_profile)
        return;
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();
      Realm::ProfilingMeasurements::OperationEventWaits waits;
      if (!response.get_measurement(waits))
        std::abort();
      legion_assert(timeline.is_valid());
      message_infos.emplace_back(MessageInfo());
      MessageInfo& info = message_infos.back();
      info.op_id = prof_info->op_id;
      info.lg_id = prof_info->id;
      info.proc_id = usage.proc.id;
      info.spawn = prof_info->extra.spawn_time;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      unsigned num_intervals = waits.intervals.size();
      if (num_intervals > 0)
      {
        for (unsigned idx = 0; idx < num_intervals; ++idx)
        {
          info.wait_intervals.emplace_back(WaitInfo());
          WaitInfo& wait_info = info.wait_intervals.back();
          wait_info.wait_start = waits.intervals[idx].wait_start;
          wait_info.wait_ready = waits.intervals[idx].wait_ready;
          wait_info.wait_end = waits.intervals[idx].wait_end;
          wait_info.wait_event = LgEvent(waits.intervals[idx].wait_event);
        }
      }
      info.creator = prof_info->creator;
      info.critical = prof_info->critical;
      info.finish_event = original_fevent;
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);

      const size_t diff =
          sizeof(MessageInfo) + num_intervals * sizeof(WaitInfo);
      owner->update_footprint(diff, this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_copy(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::OperationMemoryUsage& usage)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationCopyInfo>());
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationFinishEvent>());

      Realm::ProfilingMeasurements::OperationCopyInfo cpinfo;
      if (!response.get_measurement(cpinfo))
        std::abort();

      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();

      Realm::ProfilingMeasurements::OperationFinishEvent fevent;
      fevent.finish_event = Realm::Event::NO_EVENT;
      if (!response.get_measurement(fevent))
        std::abort();

      legion_assert(timeline.is_valid());
      copy_infos.emplace_back(CopyInfo());
      CopyInfo& info = copy_infos.back();
      info.op_id = prof_info->op_id;
      info.size = usage.size;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      info.fevent = LgEvent(fevent.finish_event);
      info.collective = (CollectiveKind)prof_info->id;
      legion_assert(!cpinfo.inst_info.empty());
      InstanceNameClosure* closure = prof_info->extra.closure;
      info.copy_expr = closure->find_copy_expression();
      info.redop = closure->find_redop();
      typedef Realm::ProfilingMeasurements::OperationCopyInfo::InstInfo
          InstInfo;
      for (const InstInfo& inst_info_item : cpinfo.inst_info)
      {
        legion_assert(
            inst_info_item.src_fields.size() ==
            inst_info_item.dst_fields.size());
        if (inst_info_item.src_indirection_inst.exists() ||
            inst_info_item.dst_indirection_inst.exists())
        {
          // Apparently we have to do the full cross-product of
          // everything here. I don't really understand so just
          // log what the Realm developers say and redirect any
          // questions from the profiler back to Realm
          unsigned offset = info.inst_infos.size();
          info.inst_infos.resize(
              offset +
              (inst_info_item.src_insts.size() *
               inst_info_item.src_fields.size() *
               inst_info_item.dst_insts.size() *
               inst_info_item.dst_fields.size()) +
              1 /*extra for indirection*/);
          // Finally log the indirection instance(s)
          CopyInstInfo& indirect = info.inst_infos[offset++];
          indirect.indirect = true;
          indirect.num_hops = inst_info_item.num_hops;
          if (inst_info_item.src_indirection_inst.exists())
          {
            indirect.src =
                inst_info_item.src_indirection_inst.get_location().id;
            indirect.src_fid = inst_info_item.src_indirection_field;
            indirect.src_inst_uid = closure->find_instance_name(
                inst_info_item.src_indirection_inst);
            indirect.src_expr = closure->find_instance_subspace(
                inst_info_item.src_indirection_inst);
          }
          else
          {
            indirect.src = 0;
            indirect.src_fid = 0;
            indirect.src_inst_uid = LgEvent::NO_LG_EVENT;
            indirect.src_expr = 0;
          }
          if (inst_info_item.dst_indirection_inst.exists())
          {
            indirect.dst =
                inst_info_item.dst_indirection_inst.get_location().id;
            indirect.dst_fid = inst_info_item.dst_indirection_field;
            indirect.dst_inst_uid = closure->find_instance_name(
                inst_info_item.dst_indirection_inst);
            indirect.dst_expr = closure->find_instance_subspace(
                inst_info_item.dst_indirection_inst);
          }
          else
          {
            indirect.dst = 0;
            indirect.dst_fid = 0;
            indirect.dst_inst_uid = LgEvent::NO_LG_EVENT;
            indirect.dst_expr = 0;
          }
          for (const PhysicalInstance& src_inst : inst_info_item.src_insts)
          {
            Memory src_location = src_inst.get_location();
            LgEvent src_name = closure->find_instance_name(src_inst);
            DistributedID src_expr = closure->find_instance_subspace(src_inst);
            for (const PhysicalInstance& dst_inst : inst_info_item.dst_insts)
            {
              Memory dst_location = dst_inst.get_location();
              LgEvent dst_name = closure->find_instance_name(dst_inst);
              DistributedID dst_expr =
                  closure->find_instance_subspace(dst_inst);
              for (const Realm::FieldID& src_fid : inst_info_item.src_fields)
              {
                for (const Realm::FieldID& dst_fid : inst_info_item.dst_fields)
                {
                  CopyInstInfo& inst_info = info.inst_infos[offset++];
                  inst_info.src = src_location.id;
                  inst_info.dst = dst_location.id;
                  inst_info.src_fid = src_fid;
                  inst_info.dst_fid = dst_fid;
                  inst_info.src_inst_uid = src_name;
                  inst_info.dst_inst_uid = dst_name;
                  inst_info.src_expr = src_expr;
                  inst_info.dst_expr = dst_expr;
                  inst_info.num_hops = inst_info_item.num_hops;
                  inst_info.indirect = false;
                }
              }
            }
          }
        }
        else
        {
          // Ask the Realm developers about why these assertions are true
          // because I still don't completely understand the logic
          legion_assert(inst_info_item.src_insts.size() == 1);
          legion_assert(inst_info_item.dst_insts.size() == 1);
          PhysicalInstance src_inst = inst_info_item.src_insts.front();
          PhysicalInstance dst_inst = inst_info_item.dst_insts.front();
          Memory src_location = src_inst.get_location();
          Memory dst_location = dst_inst.get_location();
          LgEvent src_name = closure->find_instance_name(src_inst);
          LgEvent dst_name = closure->find_instance_name(dst_inst);
          DistributedID src_expr = closure->find_instance_subspace(src_inst);
          DistributedID dst_expr = closure->find_instance_subspace(dst_inst);
          const unsigned offset = info.inst_infos.size();
          info.inst_infos.resize(offset + inst_info_item.src_fields.size());
          unsigned idx = 0;
          for (const Realm::FieldID& src_field : inst_info_item.src_fields)
          {
            CopyInstInfo& inst_info = info.inst_infos[offset + idx];
            inst_info.src = src_location.id;
            inst_info.dst = dst_location.id;
            inst_info.src_fid = src_field;
            inst_info.dst_fid = inst_info_item.dst_fields[idx];
            inst_info.src_inst_uid = src_name;
            inst_info.dst_inst_uid = dst_name;
            inst_info.src_expr = src_expr;
            inst_info.dst_expr = dst_expr;
            inst_info.num_hops = inst_info_item.num_hops;
            inst_info.indirect = false;
            idx++;
          }
        }
      }
      info.creator = prof_info->creator;
      info.critical = prof_info->critical;
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);
      owner->update_footprint(
          sizeof(CopyInfo) + info.inst_infos.size() * sizeof(CopyInstInfo),
          this);
      if (closure->remove_reference())
        delete closure;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_fill(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::OperationMemoryUsage& usage)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationCopyInfo>());
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      Realm::ProfilingMeasurements::OperationCopyInfo cpinfo;
      if (!response.get_measurement(cpinfo))
        std::abort();

      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();
      legion_assert(timeline.is_valid());
      fill_infos.emplace_back(FillInfo());
      FillInfo& info = fill_infos.back();
      info.op_id = prof_info->op_id;
      info.size = usage.size;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      Realm::ProfilingMeasurements::OperationFinishEvent fevent;
      if (!response.get_measurement(fevent))
        std::abort();
      info.fevent = LgEvent(fevent.finish_event);
      info.collective = (CollectiveKind)prof_info->id;
      InstanceNameClosure* closure = prof_info->extra.closure;
      info.fill_expr = closure->find_copy_expression();
      typedef Realm::ProfilingMeasurements::OperationCopyInfo::InstInfo
          InstInfo;
      for (const InstInfo& inst_info_item : cpinfo.inst_info)
      {
        legion_assert(!inst_info_item.dst_fields.empty());
        legion_assert(inst_info_item.dst_insts.size() == 1);
        PhysicalInstance instance = inst_info_item.dst_insts.front();
        Memory location = instance.get_location();
        LgEvent name = closure->find_instance_name(instance);
        unsigned offset = info.inst_infos.size();
        info.inst_infos.resize(offset + inst_info_item.dst_fields.size());
        unsigned idx = 0;
        for (const Realm::FieldID& dst_field : inst_info_item.dst_fields)
        {
          FillInstInfo& inst_info = info.inst_infos[offset + idx];
          inst_info.dst = location.id;
          inst_info.fid = dst_field;
          inst_info.dst_inst_uid = name;
          idx++;
        }
      }
      info.creator = prof_info->creator;
      info.critical = prof_info->critical;
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);
      owner->update_footprint(
          sizeof(FillInfo) + info.inst_infos.size() * sizeof(FillInstInfo),
          this);
      if (closure->remove_reference())
        delete closure;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst_timeline(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response,
        const Realm::ProfilingMeasurements::InstanceMemoryUsage& usage,
        const Realm::ProfilingMeasurements::InstanceTimeline& timeline)
    //--------------------------------------------------------------------------
    {
      inst_timeline_infos.emplace_back(InstTimelineInfo());
      InstTimelineInfo& info = inst_timeline_infos.back();
      info.inst_uid.id = prof_info->id;
      info.inst_id = usage.instance.id;
      info.mem_id = usage.memory.id;
      info.size = usage.bytes;
      info.op_id = prof_info->op_id;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.destroy = timeline.delete_time;
      info.creator = prof_info->creator;
      owner->update_footprint(sizeof(InstTimelineInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_partition(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingResponse& response)
    //--------------------------------------------------------------------------
    {
      legion_assert(response.has_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>());
      // Check to see if this dependent partition operation has a finish event
      // If it doesn't that means that it was executed inline and we don't
      // need to bother recording
      Realm::ProfilingMeasurements::OperationFinishEvent fevent;
      if (!response.get_measurement(fevent) || !fevent.finish_event.exists())
        return;
      Realm::ProfilingMeasurements::OperationTimeline timeline;
      if (!response.get_measurement(timeline))
        std::abort();
      partition_infos.emplace_back(PartitionInfo());
      PartitionInfo& info = partition_infos.back();
      info.op_id = prof_info->op_id;
      info.part_op = (DepPartOpKind)prof_info->id;
      info.create = timeline.create_time;
      info.ready = timeline.ready_time;
      info.start = timeline.start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline.complete_time;
      info.creator = prof_info->creator;
      info.critical = prof_info->critical;
      if (prof_info->critical.is_barrier())
        record_barrier_use(prof_info->critical, prof_info->op_id);
      info.fevent = LgEvent(fevent.finish_event);
      if (prof_info->extra.closure != nullptr)
      {
        LargeNameClosure* closure =
            legion_safe_cast<LargeNameClosure*>(prof_info->extra.closure);
        info.inst_infos.reserve(closure->entries.size());
        for (const LargeNameClosure::Entry& entry : closure->entries)
        {
          info.inst_infos.emplace_back(PartInstInfo{
              entry.instance.get_location(), closure->fid, entry.name,
              entry.subspace});
        }
        if (closure->remove_reference())
          delete closure;
      }
      owner->update_footprint(sizeof(PartitionInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_arrival(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingMeasurements::OperationTimeline& timeline)
    //--------------------------------------------------------------------------
    {
      // The arrival occurred when we created the no-op task
      // The precondition event triggered when the no-op task became ready
      const ArrivalInfo info(
          timeline.create_time, timeline.ready_time, prof_info->critical,
          prof_info->creator);
      // Do the barrier arrival with the arrival info argument
      // Still chain on the precondition to propagate poison (if any)
      Realm::Barrier bar;
      bar.id = prof_info->id;
      bar.timestamp = 0;
      bar.arrive(
          prof_info->extra.id2, prof_info->critical, &info, sizeof(info));
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_async_effect(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingMeasurements::OperationTimeline& timeline)
    //--------------------------------------------------------------------------
    {
      AsyncEffectInfo& info =
          async_effect_infos.emplace_back(AsyncEffectInfo());
      info.external = prof_info->critical;
      info.fevent = prof_info->creator;
      info.created = timeline.create_time;
      info.triggered = timeline.ready_time;
      info.pid = prof_info->id;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_make_valid(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingMeasurements::OperationTimeline& timeline)
    //--------------------------------------------------------------------------
    {
      MakeValidInfo& info = make_valid_infos.emplace_back(MakeValidInfo());
      info.result = prof_info->critical;
      info.fevent = prof_info->creator;
      info.space = prof_info->id;
      info.created = timeline.create_time;
      info.triggered = timeline.ready_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_fetch_metadata(
        const ProfilingInfo* prof_info,
        const Realm::ProfilingMeasurements::OperationTimeline& timeline)
    //--------------------------------------------------------------------------
    {
      FetchMetadataInfo& info =
          fetch_metadata_infos.emplace_back(FetchMetadataInfo());
      info.result = prof_info->critical;
      info.fevent = prof_info->creator;
      info.inst_uid.id = prof_info->id;
      info.created = timeline.create_time;
      info.triggered = timeline.ready_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_implicit(
        UniqueID op_id, TaskID tid, long long start_time, long long stop_time,
        std::deque<WaitInfo>& waits, LgEvent finish_event)
    //--------------------------------------------------------------------------
    {
      TaskInfo& info = implicit_infos.emplace_back(TaskInfo());
      info.op_id = op_id;
      info.task_id = tid;
      info.variant_id = 0;  // no variants for implicit tasks
      info.proc_id = local_proc.id;
      // We make create, ready, and start all the same for implicit tasks
      info.create = start_time;
      info.ready = start_time;
      info.start = start_time;
      info.stop = stop_time;
      info.wait_intervals.swap(waits);
      info.finish_event = finish_event;
      // Also record an implicit wait on the external thread for this task
      // to make it seem like we were blocked waiting for it
      WaitInfo& wait = external_wait_infos.emplace_back(WaitInfo());
      wait.wait_start = start_time;
      wait.wait_ready = stop_time;
      wait.wait_end = stop_time;
      wait.wait_event = finish_event;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_mem_desc(const Memory& m)
    //--------------------------------------------------------------------------
    {
      if (m == Memory::NO_MEMORY)
        return;
      if (std::binary_search(mem_ids.begin(), mem_ids.end(), m.id))
        return;
      mem_ids.emplace_back(m.id);
      std::sort(mem_ids.begin(), mem_ids.end());
      owner->record_memory(m);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_proc_desc(const Processor& p)
    //--------------------------------------------------------------------------
    {
      if (std::binary_search(proc_ids.begin(), proc_ids.end(), p.id))
        return;
      proc_ids.emplace_back(p.id);
      std::sort(proc_ids.begin(), proc_ids.end());
      owner->record_processor(p);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_event_trigger(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      EventTriggerInfo& info =
          event_trigger_infos.emplace_back(EventTriggerInfo());
      derez.deserialize(info);
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProfilerEventTriggerMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      implicit_profiler->process_event_trigger(derez);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_event_poison(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      EventPoisonInfo& info =
          event_poison_infos.emplace_back(EventPoisonInfo());
      derez.deserialize(info);
      owner->update_footprint(sizeof(info), this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProfilerEventPoisonMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      implicit_profiler->process_event_poison(derez);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_mapper_call(
        MapperID mapper, Processor mapper_proc, MappingCallKind kind,
        UniqueID uid, long long start, long long stop)
    //--------------------------------------------------------------------------
    {
      // Check to see if it exceeds the call threshold
      if ((stop - start) < owner->minimum_call_threshold)
        return;
      mapper_call_infos.emplace_back(MapperCallInfo());
      MapperCallInfo& info = mapper_call_infos.back();
      info.mapper = mapper;
      info.mapper_proc = mapper_proc.id;
      info.kind = kind;
      info.op_id = uid;
      info.start = start;
      info.stop = stop;
      info.proc_id = local_proc.id;
      info.finish_event = implicit_fevent;
      owner->update_footprint(sizeof(MapperCallInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_runtime_call(
        RuntimeCallKind kind, long long start, long long stop)
    //--------------------------------------------------------------------------
    {
      // Check to see if it exceeds the call threshold
      if ((stop - start) < owner->minimum_call_threshold)
        return;
      runtime_call_infos.emplace_back(RuntimeCallInfo());
      RuntimeCallInfo& info = runtime_call_infos.back();
      info.kind = kind;
      info.start = start;
      info.stop = stop;
      info.proc_id = local_proc.id;
      info.finish_event = implicit_fevent;
      owner->update_footprint(sizeof(RuntimeCallInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_application_range(
        ProvenanceID pid, long long start, long long stop)
    //--------------------------------------------------------------------------
    {
      // We don't filter application call ranges currently since presumably
      // the application knows what its doing and wants to see everything
      application_call_infos.emplace_back(ApplicationCallInfo());
      ApplicationCallInfo& info = application_call_infos.back();
      info.pid = pid;
      info.start = start;
      info.stop = stop;
      info.proc_id = local_proc.id;
      info.finish_event = implicit_fevent;
      owner->update_footprint(sizeof(ApplicationCallInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_async_effect(
        ApEvent effect, const char* prov)
    //--------------------------------------------------------------------------
    {
      if (owner->no_critical_paths)
        return;
      bool poisoned = false;
      if (effect.has_triggered_faultaware(poisoned) || poisoned)
      {
        AsyncEffectInfo& info =
            async_effect_infos.emplace_back(AsyncEffectInfo());
        info.triggered = Realm::Clock::current_time_in_nanoseconds();
        // Doing created after triggered makes it look like the event was
        // ready before it was created, which it probably was since it
        // triggered almost immediately after it was created, so while
        // not strictly accurate it reflects what actually happened so
        // the profiler can give good feedback
        info.created = Realm::Clock::current_time_in_nanoseconds();
        info.external = effect;
        info.fevent = implicit_fevent;
        if (prov != nullptr)
        {
          Provenance* provenance =
              runtime->find_or_create_provenance(prov, strlen(prov));
          info.pid = provenance->pid;
        }
        else
          info.pid = 0;
      }
      else
        owner->measure_event_trigger(
            effect, LegionProfiler::ASYNC_EFFECT, implicit_fevent, 0, prov);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_event_wait(
        LgEvent event, ProvenanceID pid, Realm::Backtrace& bt)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have a backtrace ID for this backtrace yet
      unsigned long long backtrace_id = owner->find_backtrace_id(bt);
      event_wait_infos.emplace_back(EventWaitInfo{
          local_proc.id, implicit_fevent, event, pid, backtrace_id});
      if (event.is_barrier())
        record_barrier_use(event, implicit_unique_op_id);
      owner->update_footprint(sizeof(EventWaitInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::begin_external_wait(LgEvent event)
    //--------------------------------------------------------------------------
    {
      // You cannot do anything in here that waits on an event!
      WaitInfo& info = external_wait_infos.emplace_back(WaitInfo());
      info.wait_event = event;
      info.wait_start = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::end_external_wait(LgEvent event)
    //--------------------------------------------------------------------------
    {
      // You cannot do anything in here that waits on an event!
      legion_assert(!external_wait_infos.empty());
      WaitInfo& info = external_wait_infos.back();
      legion_assert(info.wait_event == event);
      info.wait_ready = Realm::Clock::current_time_in_nanoseconds();
      info.wait_end = info.wait_ready;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_proftask(
        Processor proc, UniqueID op_id, long long start, long long stop,
        LgEvent creator, LgEvent finish_event, bool complete)
    //--------------------------------------------------------------------------
    {
      prof_task_infos.emplace_back(ProfTaskInfo());
      ProfTaskInfo& info = prof_task_infos.back();
      info.proc_id = proc.id;
      info.op_id = op_id;
      info.start = start;
      info.stop = stop;
      info.creator = creator;
      info.finish_event = finish_event;
      info.completion = complete;
      owner->update_footprint(sizeof(ProfTaskInfo), this);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::dump_state(LegionProfSerializer* serializer)
    //--------------------------------------------------------------------------
    {
      for (const OperationInstance& op_instance : operation_instances)
        serializer->serialize(op_instance);
      for (const MultiTask& multi_task : multi_tasks)
        serializer->serialize(multi_task);
      for (const SliceOwner& slice_owner : slice_owners)
        serializer->serialize(slice_owner);
      for (const TaskInfo& task_info : task_infos)
      {
        serializer->serialize(task_info, false /*not implicit*/);
        for (const WaitInfo& wait_info : task_info.wait_intervals)
          serializer->serialize(wait_info, task_info);
      }
      for (const TaskInfo& task_info : implicit_infos)
      {
        serializer->serialize(task_info, true /*implicit*/);
        for (const WaitInfo& wait_info : task_info.wait_intervals)
          serializer->serialize(wait_info, task_info);
      }
      for (const GPUTaskInfo& gpu_task_info : gpu_task_infos)
      {
        serializer->serialize(gpu_task_info);
        for (const WaitInfo& wait_info : gpu_task_info.wait_intervals)
          serializer->serialize(wait_info, gpu_task_info);
      }
      for (const IndexSpaceRectDesc& rect_desc : ispace_rect_desc)
        serializer->serialize(rect_desc);
      for (const IndexSpacePointDesc& point_desc : ispace_point_desc)
        serializer->serialize(point_desc);
      for (const IndexSpaceEmptyDesc& empty_desc : ispace_empty_desc)
        serializer->serialize(empty_desc);
      for (const FieldDesc& field_desc_item : field_desc)
        serializer->serialize(field_desc_item);
      for (const FieldSpaceDesc& field_space_desc_item : field_space_desc)
        serializer->serialize(field_space_desc_item);
      for (const IndexSpaceDesc& index_space_desc_item : index_space_desc)
        serializer->serialize(index_space_desc_item);
      for (const IndexPartDesc& index_part_desc_item : index_part_desc)
        serializer->serialize(index_part_desc_item);
      for (const IndexSubSpaceDesc& index_subspace_desc_item :
           index_subspace_desc)
        serializer->serialize(index_subspace_desc_item);
      for (const IndexPartitionDesc& index_partition_desc_item :
           index_partition_desc)
        serializer->serialize(index_partition_desc_item);
      for (const LogicalRegionDesc& lr_desc_item : lr_desc)
        serializer->serialize(lr_desc_item);
      for (const PhysicalInstRegionDesc& phy_inst_region_desc : phy_inst_rdesc)
        serializer->serialize(phy_inst_region_desc);
      for (const PhysicalInstLayoutDesc& phy_inst_layout_desc :
           phy_inst_layout_rdesc)
        serializer->serialize(phy_inst_layout_desc);
      for (const PhysicalInstDimOrderDesc& phy_inst_dim_order_desc :
           phy_inst_dim_order_rdesc)
        serializer->serialize(phy_inst_dim_order_desc);
      for (const PhysicalInstanceSpaces& phy_inst_space : phy_inst_spaces)
        serializer->serialize(phy_inst_space);
      for (const PhysicalInstanceUsage& phy_inst_usage_item : phy_inst_usage)
        serializer->serialize(phy_inst_usage_item);
      for (const IndexSpaceSizeDesc& index_space_size_desc_item :
           index_space_size_desc)
        serializer->serialize(index_space_size_desc_item);
      for (const MetaInfo& meta_info : meta_infos)
      {
        serializer->serialize(meta_info);
        for (const WaitInfo& wait_info : meta_info.wait_intervals)
          serializer->serialize(wait_info, meta_info);
      }
      for (const MessageInfo& message_info : message_infos)
      {
        serializer->serialize(message_info);
        for (const WaitInfo& wait_info : message_info.wait_intervals)
          serializer->serialize(wait_info, message_info);
      }
      for (const FillInfo& fill_info : fill_infos)
        serializer->serialize(fill_info);
      for (const CopyInfo& copy_info : copy_infos)
        serializer->serialize(copy_info);
      for (const InstTimelineInfo& inst_timeline_info : inst_timeline_infos)
        serializer->serialize(inst_timeline_info);
      for (const PartitionInfo& partition_info : partition_infos)
        serializer->serialize(partition_info);
      for (const MapperCallInfo& mapper_call_info : mapper_call_infos)
        serializer->serialize(mapper_call_info);
      for (const RuntimeCallInfo& runtime_call_info : runtime_call_infos)
        serializer->serialize(runtime_call_info);
      for (const ApplicationCallInfo& application_call_info :
           application_call_infos)
        serializer->serialize(application_call_info);
      for (const AsyncEffectInfo& async_effect_info : async_effect_infos)
        serializer->serialize(async_effect_info);
      for (const EventWaitInfo& event_wait_info : event_wait_infos)
        serializer->serialize(event_wait_info);
      for (const EventMergerInfo& event_merger_info : event_merger_infos)
        serializer->serialize(event_merger_info);
      for (const EventTriggerInfo& event_trigger_info : event_trigger_infos)
        serializer->serialize(event_trigger_info);
      for (const EventPoisonInfo& event_poison_info : event_poison_infos)
        serializer->serialize(event_poison_info);
      for (const BarrierArrivalInfo& barrier_arrival_info :
           barrier_arrival_infos)
        serializer->serialize(barrier_arrival_info);
      for (const ReservationAcquireInfo& reservation_acquire_info :
           reservation_acquire_infos)
        serializer->serialize(reservation_acquire_info);
      for (const InstanceReadyInfo& instance_ready_info : instance_ready_infos)
        serializer->serialize(instance_ready_info);
      for (const InstanceRedistrictInfo& instance_redistrict_info :
           instance_redistrict_infos)
        serializer->serialize(instance_redistrict_info);
      for (const CompletionQueueInfo& completion_queue_info :
           completion_queue_infos)
        serializer->serialize(completion_queue_info);
      for (const MakeValidInfo& make_valid_info : make_valid_infos)
        serializer->serialize(make_valid_info);
      for (const FetchMetadataInfo& fetch_metadata_info : fetch_metadata_infos)
        serializer->serialize(fetch_metadata_info);
      for (const ProfTaskInfo& prof_task_info : prof_task_infos)
        serializer->serialize(prof_task_info);
      operation_instances.clear();
      multi_tasks.clear();
      slice_owners.clear();
      task_infos.clear();
      implicit_infos.clear();
      gpu_task_infos.clear();
      ispace_rect_desc.clear();
      ispace_point_desc.clear();
      ispace_empty_desc.clear();
      field_desc.clear();
      field_space_desc.clear();
      index_part_desc.clear();
      index_space_desc.clear();
      index_subspace_desc.clear();
      index_partition_desc.clear();
      lr_desc.clear();
      phy_inst_rdesc.clear();
      phy_inst_layout_rdesc.clear();
      phy_inst_dim_order_rdesc.clear();
      phy_inst_spaces.clear();
      phy_inst_usage.clear();
      index_space_size_desc.clear();
      meta_infos.clear();
      message_infos.clear();
      copy_infos.clear();
      fill_infos.clear();
      inst_timeline_infos.clear();
      partition_infos.clear();
      mapper_call_infos.clear();
      runtime_call_infos.clear();
      application_call_infos.clear();
      async_effect_infos.clear();
      event_wait_infos.clear();
      event_merger_infos.clear();
      event_trigger_infos.clear();
      event_poison_infos.clear();
      barrier_arrival_infos.clear();
      reservation_acquire_infos.clear();
      instance_ready_infos.clear();
      instance_redistrict_infos.clear();
      completion_queue_infos.clear();
      make_valid_infos.clear();
      fetch_metadata_infos.clear();
      prof_task_infos.clear();
      // Finally if we're an external thread, dump our implicit
      // top-level task information for ourselves
      if (external_fevent.exists())
      {
        TaskInfo external_info;
        external_info.op_id = runtime->get_unique_operation_id();
        external_info.task_id = owner->get_external_implicit_task();
        external_info.variant_id = 0;
        external_info.proc_id = local_proc.id;
        external_info.create = external_start;
        external_info.ready = external_start;
        external_info.start = external_start;
        external_info.stop = Realm::Clock::current_time_in_nanoseconds();
        external_info.finish_event = external_fevent;
        serializer->serialize(external_info, true /*implicit*/);
        for (const WaitInfo& wait_info : external_wait_infos)
          serializer->serialize(wait_info, external_info);
      }
    }

    //--------------------------------------------------------------------------
    bool LegionProfInstance::dump_inter(
        LegionProfSerializer* serializer, const long long t_stop)
    //--------------------------------------------------------------------------
    {
      while (!operation_instances.empty())
      {
        OperationInstance& front = operation_instances.front();
        serializer->serialize(front);
        operation_instances.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!multi_tasks.empty())
      {
        MultiTask& front = multi_tasks.front();
        serializer->serialize(front);
        multi_tasks.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!slice_owners.empty())
      {
        SliceOwner& front = slice_owners.front();
        serializer->serialize(front);
        slice_owners.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!task_infos.empty())
      {
        TaskInfo& front = task_infos.front();
        serializer->serialize(front, false /*not implicit*/);
        // Have to do all of these now
        for (const WaitInfo& wait_info : front.wait_intervals)
          serializer->serialize(wait_info, front);
        task_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!implicit_infos.empty())
      {
        TaskInfo& front = implicit_infos.front();
        serializer->serialize(front, true /*implicit*/);
        // Have to do all of these now
        for (const WaitInfo& wait_info : front.wait_intervals)
          serializer->serialize(wait_info, front);
        implicit_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!gpu_task_infos.empty())
      {
        GPUTaskInfo& front = gpu_task_infos.front();
        serializer->serialize(front);
        // Have to do all of these now
        for (const WaitInfo& wait_info : front.wait_intervals)
          serializer->serialize(wait_info, front);
        gpu_task_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!ispace_rect_desc.empty())
      {
        IndexSpaceRectDesc& front = ispace_rect_desc.front();
        serializer->serialize(front);
        ispace_rect_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!ispace_point_desc.empty())
      {
        IndexSpacePointDesc& front = ispace_point_desc.front();
        serializer->serialize(front);
        ispace_point_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!ispace_empty_desc.empty())
      {
        IndexSpaceEmptyDesc& front = ispace_empty_desc.front();
        serializer->serialize(front);
        ispace_empty_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!field_desc.empty())
      {
        FieldDesc& front = field_desc.front();
        serializer->serialize(front);
        free(const_cast<char*>(front.name));
        field_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!field_space_desc.empty())
      {
        FieldSpaceDesc& front = field_space_desc.front();
        serializer->serialize(front);
        free(const_cast<char*>(front.name));
        field_space_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!index_part_desc.empty())
      {
        IndexPartDesc& front = index_part_desc.front();
        serializer->serialize(front);
        free(const_cast<char*>(front.name));
        index_part_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!index_space_desc.empty())
      {
        IndexSpaceDesc& front = index_space_desc.front();
        serializer->serialize(front);
        free(const_cast<char*>(front.name));
        index_space_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!index_subspace_desc.empty())
      {
        IndexSubSpaceDesc& front = index_subspace_desc.front();
        serializer->serialize(front);
        index_subspace_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!index_partition_desc.empty())
      {
        IndexPartitionDesc& front = index_partition_desc.front();
        serializer->serialize(front);
        index_partition_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!lr_desc.empty())
      {
        LogicalRegionDesc& front = lr_desc.front();
        serializer->serialize(front);
        free(const_cast<char*>(front.name));
        lr_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!phy_inst_rdesc.empty())
      {
        PhysicalInstRegionDesc& front = phy_inst_rdesc.front();
        serializer->serialize(front);
        phy_inst_rdesc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!phy_inst_dim_order_rdesc.empty())
      {
        PhysicalInstDimOrderDesc& front = phy_inst_dim_order_rdesc.front();
        serializer->serialize(front);
        phy_inst_dim_order_rdesc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!phy_inst_spaces.empty())
      {
        PhysicalInstanceSpaces& spaces = phy_inst_spaces.front();
        serializer->serialize(spaces);
        phy_inst_spaces.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!phy_inst_usage.empty())
      {
        PhysicalInstanceUsage& usage = phy_inst_usage.front();
        serializer->serialize(usage);
        phy_inst_usage.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!index_space_size_desc.empty())
      {
        IndexSpaceSizeDesc& front = index_space_size_desc.front();
        serializer->serialize(front);
        index_space_size_desc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!phy_inst_layout_rdesc.empty())
      {
        PhysicalInstLayoutDesc& front = phy_inst_layout_rdesc.front();
        serializer->serialize(front);
        phy_inst_layout_rdesc.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!meta_infos.empty())
      {
        MetaInfo& front = meta_infos.front();
        serializer->serialize(front);
        // Have to do all of these now
        for (const WaitInfo& wait_info : front.wait_intervals)
          serializer->serialize(wait_info, front);
        meta_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!message_infos.empty())
      {
        MessageInfo& front = message_infos.front();
        serializer->serialize(front);
        // Have to do all of these now
        for (const WaitInfo& wait_info : front.wait_intervals)
          serializer->serialize(wait_info, front);
        message_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!copy_infos.empty())
      {
        CopyInfo& front = copy_infos.front();
        serializer->serialize(front);
        copy_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!fill_infos.empty())
      {
        FillInfo& front = fill_infos.front();
        serializer->serialize(front);
        fill_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!inst_timeline_infos.empty())
      {
        InstTimelineInfo& front = inst_timeline_infos.front();
        serializer->serialize(front);
        inst_timeline_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!partition_infos.empty())
      {
        PartitionInfo& front = partition_infos.front();
        serializer->serialize(front);
        partition_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!mapper_call_infos.empty())
      {
        MapperCallInfo& front = mapper_call_infos.front();
        serializer->serialize(front);
        mapper_call_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!runtime_call_infos.empty())
      {
        RuntimeCallInfo& front = runtime_call_infos.front();
        serializer->serialize(front);
        runtime_call_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!application_call_infos.empty())
      {
        ApplicationCallInfo& front = application_call_infos.front();
        serializer->serialize(front);
        application_call_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!async_effect_infos.empty())
      {
        AsyncEffectInfo& front = async_effect_infos.front();
        serializer->serialize(front);
        async_effect_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!event_wait_infos.empty())
      {
        EventWaitInfo& info = event_wait_infos.front();
        serializer->serialize(info);
        event_wait_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!event_merger_infos.empty())
      {
        EventMergerInfo& info = event_merger_infos.front();
        serializer->serialize(info);
        event_merger_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!event_trigger_infos.empty())
      {
        EventTriggerInfo& info = event_trigger_infos.front();
        serializer->serialize(info);
        event_trigger_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!event_poison_infos.empty())
      {
        EventPoisonInfo& info = event_poison_infos.front();
        serializer->serialize(info);
        event_poison_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!barrier_arrival_infos.empty())
      {
        BarrierArrivalInfo& info = barrier_arrival_infos.front();
        serializer->serialize(info);
        barrier_arrival_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!reservation_acquire_infos.empty())
      {
        ReservationAcquireInfo& info = reservation_acquire_infos.front();
        serializer->serialize(info);
        reservation_acquire_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!instance_ready_infos.empty())
      {
        InstanceReadyInfo& info = instance_ready_infos.front();
        serializer->serialize(info);
        instance_ready_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!instance_redistrict_infos.empty())
      {
        InstanceRedistrictInfo& info = instance_redistrict_infos.front();
        serializer->serialize(info);
        instance_redistrict_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!completion_queue_infos.empty())
      {
        CompletionQueueInfo& info = completion_queue_infos.front();
        serializer->serialize(info);
        completion_queue_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!make_valid_infos.empty())
      {
        MakeValidInfo& info = make_valid_infos.front();
        serializer->serialize(info);
        make_valid_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!fetch_metadata_infos.empty())
      {
        FetchMetadataInfo& info = fetch_metadata_infos.front();
        serializer->serialize(info);
        fetch_metadata_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      while (!prof_task_infos.empty())
      {
        ProfTaskInfo& front = prof_task_infos.front();
        serializer->serialize(front);
        prof_task_infos.pop_front();
        if (t_stop <= Realm::Clock::current_time_in_microseconds())
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(
        Processor target, const Machine& machine, unsigned num_meta_tasks,
        const char* const * const task_descriptions, unsigned num_message_kinds,
        const char* const * const message_descriptions,
        unsigned num_operation_kinds,
        const char* const * const operation_kind_descriptions,
        const char* serializer_type, const char* prof_logfile,
        const size_t total_runtime_instances, const size_t footprint_threshold,
        const size_t target_latency, const size_t call_threshold,
        const bool slow_config_ok, const bool self_prof, const bool no_critical,
        const bool all_arrivals)
      : done_event(Realm::UserEvent::create_user_event()),
        minimum_call_threshold(call_threshold * 1000 /*convert us to ns*/),
        output_footprint_threshold(footprint_threshold),
        output_target_latency(target_latency), target_proc(target),
        self_profile(self_prof), no_critical_paths(no_critical),
#ifdef LEGION_DEBUG_COLLECTIVES
        // Can't rely on the barrier reduction in this case
        all_critical_arrivals(true),
#else
        all_critical_arrivals(all_arrivals),
#endif
#ifndef LEGION_DEBUG
        total_outstanding_requests(1 /*start with guard*/),
#endif
        total_memory_footprint(0), implicit_top_level_task_proc(0),
        need_default_mapper_warning(!slow_config_ok)
    //--------------------------------------------------------------------------
    {
      legion_assert(target_proc.exists());
      if (!strcmp(serializer_type, "binary"))
      {
        if (prof_logfile == nullptr)
        {
          Error error(LEGION_STARTUP_EXCEPTION);
          error << "Please specify -lg:prof_logfile <logfile_name> when "
                   "running with -lg:serializer binary.";
          error.raise();
        }
        std::string filename(prof_logfile);
        size_t pct = filename.find_first_of('%', 0);
        if (pct == std::string::npos)
        {
          // This is only an error if we have multiple runtimes
          if (total_runtime_instances > 1)
          {
            Error error(LEGION_STARTUP_EXCEPTION);
            error << "The logfile name must contain '%' which will be replaced "
                     "with the node id.";
            error.raise();
          }
          serializer = new LegionProfBinarySerializer(filename.c_str());
        }
        else
        {
          // replace % with node number
          std::stringstream ss;
          ss << filename.substr(0, pct) << target.address_space()
             << filename.substr(pct + 1);
          serializer = new LegionProfBinarySerializer(ss.str());
        }
      }
      else if (!strcmp(serializer_type, "ascii"))
      {
        if (prof_logfile != nullptr)
        {
          Warning warning;
          warning << "You should not specify -lg:prof_logfile <logfile_name> "
                     "when running with -lg:serializer ascii. "
                  << "Legion_prof output will be written to '-logfile "
                     "<logfile_name>' instead.";
          warning.raise();
        }
        serializer = new LegionProfASCIISerializer();
      }
      else
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Invalid serializer (" << serializer_type
              << "), must be 'binary' or 'ascii'.";
        error.raise();
      }

      // log machine info, this needs to be the first log
      MachineDesc machine_desc;

      machine.get_process_info(target, &machine_desc.process_info);
      machine_desc.node_id = static_cast<unsigned>(runtime->address_space);
      machine_desc.num_nodes =
          static_cast<unsigned>(runtime->total_address_spaces);

      serializer->serialize(machine_desc);

      ZeroTime zero_time;
      zero_time.zero_time = Legion::Runtime::get_zero_time();

      serializer->serialize(zero_time);

      for (unsigned idx = 0; idx < num_meta_tasks; idx++)
      {
        MetaDesc meta_desc;
        meta_desc.kind = idx;
        meta_desc.message = false;
        meta_desc.ordered_vc = false;
        meta_desc.name = task_descriptions[idx];
        serializer->serialize(meta_desc);
      }
      // Messages are appended as kinds of meta descriptions
      for (unsigned idx = 0; idx < num_message_kinds; idx++)
      {
        MetaDesc meta_desc;
        meta_desc.kind = num_meta_tasks + idx;
        meta_desc.message = true;
        const VirtualChannelKind vc =
            MessageManager::find_message_vc((MessageKind)idx);
        meta_desc.ordered_vc = (vc <= LAST_UNORDERED_VIRTUAL_CHANNEL);
        meta_desc.name = message_descriptions[idx];
        serializer->serialize(meta_desc);
      }
      for (unsigned idx = 0; idx < num_operation_kinds; idx++)
      {
        OpDesc op_desc;
        op_desc.kind = idx;
        op_desc.name = operation_kind_descriptions[idx];
        serializer->serialize(op_desc);
      }
      // log max dim
      MaxDimDesc max_dim_desc;
      max_dim_desc.max_dim = LEGION_MAX_DIM;
      serializer->serialize(max_dim_desc);
      // Log the runtime configuration
      const RuntimeConfig config = {
#ifdef LEGION_DEBUG
          true,
#else
          false,
#endif
          spy_logging_level > NO_SPY_LOGGING,
#ifdef LEGION_GC
          true,
#else
          false,
#endif
          runtime->program_order_execution,
          runtime->safe_mapper,
          runtime->safe_model,
          runtime->safe_control_replication > 0,
          runtime->verify_partitions,
#ifdef LEGION_BOUNDS_CHECKS
          true,
#else
          false,
#endif
          runtime->resilient_mode,
      };
      serializer->serialize(config);
#ifdef LEGION_DEBUG
      for (unsigned idx = 0; idx < LEGION_PROF_LAST; idx++)
        total_outstanding_requests[idx] = 0;
      total_outstanding_requests[LEGION_PROF_META] = 1;  // guard
#endif
    }

    //--------------------------------------------------------------------------
    LegionProfiler::~LegionProfiler(void)
    //--------------------------------------------------------------------------
    {
      LegionProfInstance* inst = instances.load();
      while (inst != nullptr)
      {
        LegionProfInstance* next = inst->next;
        delete inst;
        inst = next;
      }
      legion_assert(dump_list == nullptr);
      // remove our serializer
      delete serializer;
    }

    //--------------------------------------------------------------------------
    LegionProfiler::ProfilingInfo::ProfilingInfo(
        LegionProfiler* p, ProfilingKind k, Operation* op)
      : LegionProfInstance::ProfilingInfo(
            p, (op == nullptr) ? 0 : op->get_unique_op_id()),
        kind(k)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_task_kind(
        TaskID task_id, const char* name, bool overwrite)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(profiler_lock);
      const TaskKind& kind =
          task_kinds.emplace_back(TaskKind{task_id, name, overwrite});
      update_footprint(sizeof(kind) + kind.name.size());
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_task_variant(
        TaskID task_id, VariantID variant_id, const char* variant_name)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(profiler_lock);
      const TaskVariant& variant = task_variants.emplace_back(
          TaskVariant{task_id, variant_id, variant_name});
      update_footprint(sizeof(variant) + variant.name.size());
    }

    //--------------------------------------------------------------------------
    unsigned long long LegionProfiler::find_backtrace_id(Realm::Backtrace& bt)
    //--------------------------------------------------------------------------
    {
      // Very important: we cannot block in this function!
      // It is called within the event wait code so we can't wait on
      // any events ourselves. We need to defer this code out to the future
      // if we do need to do any blocking behavior.
      const uintptr_t hash = bt.hash();
      {
        AutoTryLock p_lock(profiler_lock, false /*exclusive*/);
        if (p_lock.has_lock())
        {
          std::map<uintptr_t, unsigned long long>::const_iterator finder =
              backtrace_ids.find(hash);
          if (finder != backtrace_ids.end())
            return finder->second;
        }
      }
      // First time seeing this backtrace so capture the symbols
      std::stringstream ss;
      ss << bt;
      std::string str = ss.str();
      // Need this for computing the ID of the backtrace
      std::vector<std::string> symbols;
      bt.print_symbols(symbols);
      // Use the hash of the string for the backtrace so that the profiler
      // can deduplicate it across different processes, make sure to use
      // the Murmur3Hasher here and not std::hash because we want the
      // result to be the same across processes and that is not true
      // of the std::hash algorithm
      Murmur3Hasher hasher;
      for (const std::string& symbol : symbols)
        hasher.hash(symbol.c_str(), symbol.size());
      uint64_t hashes[2];
      hasher.finalize(hashes);
      const unsigned long long result = hashes[0] ^ hashes[1];
      // Now retake the lock and see if we lost the race
      AutoTryLock p_lock(profiler_lock);
      if (p_lock.has_lock())
      {
        drain_pending_backtraces(true /*update footprint*/);
        std::map<uintptr_t, unsigned long long>::const_iterator finder =
            backtrace_ids.find(hash);
        if (finder != backtrace_ids.end())
        {
          legion_assert(finder->second == result);
          return finder->second;
        }
        // Didn't lose the race so generate a new ID for this backtrace
        backtrace_ids[hash] = result;
        const Backtrace& backtrace =
            backtraces.emplace_back(Backtrace{result, std::move(str)});
        update_footprint(sizeof(backtrace) + backtrace.backtrace.size());
      }
      else
      {
        // Make a pending backtrace and add it to the lock-free list
        PendingBacktrace* pending =
            new PendingBacktrace{{result, std::move(str)}, hash, nullptr};
        PendingBacktrace* head = pending_backtraces.load();
        pending->next = head;
        while (!pending_backtraces.compare_exchange_weak(head, pending))
          pending->next = head;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::drain_pending_backtraces(bool track_diff)
    //--------------------------------------------------------------------------
    {
      // Atomically swap the entire list out and then drain it
      PendingBacktrace* bt = pending_backtraces.exchange(nullptr);
      size_t diff = 0;
      while (bt != nullptr)
      {
        // See if we need to save it
        if (backtrace_ids.find(bt->hash) == backtrace_ids.end())
        {
          backtrace_ids[bt->hash] = bt->id;
          const Backtrace& backtrace = backtraces.emplace_back(
              Backtrace{bt->id, std::move(bt->backtrace)});
          if (track_diff)
            diff += sizeof(backtrace) + backtrace.backtrace.size();
        }
        PendingBacktrace* next = bt->next;
        delete bt;
        bt = next;
      }
      if (diff > 0)
        update_footprint(diff);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_memory(Memory m)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock p_lock(profiler_lock, false /*exclusive*/);
        if (std::binary_search(
                recorded_memories.begin(), recorded_memories.end(), m))
          return;
      }
      AutoLock p_lock(profiler_lock);
      if (std::binary_search(
              recorded_memories.begin(), recorded_memories.end(), m))
        return;
      // Also log all the affinities for this memory
      std::vector<Memory> memories_to_log(1, m);
      record_affinities(memories_to_log);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_processor(Processor p)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock p_lock(profiler_lock, false /*exclusive*/);
        if (std::binary_search(
                recorded_processors.begin(), recorded_processors.end(), p))
          return;
      }
      AutoLock p_lock(profiler_lock);
      if (std::binary_search(
              recorded_processors.begin(), recorded_processors.end(), p))
        return;
      ProcDesc& proc = processor_descriptions.emplace_back(ProcDesc());
      proc.proc_id = p.id;
      proc.kind = p.kind();
#ifdef LEGION_USE_CUDA
      if (!Realm::Cuda::get_cuda_device_uuid(p, &proc.cuda_device_uuid))
        proc.cuda_device_uuid[0] = 0;
#endif
      update_footprint(sizeof(proc));
      recorded_processors.emplace_back(p);
      std::sort(recorded_processors.begin(), recorded_processors.end());
      std::vector<Memory> memories_to_log;
      std::vector<ProcessorMemoryAffinity> affinities;
      runtime->machine.get_proc_mem_affinity(affinities, p);
      for (const ProcessorMemoryAffinity& affinity : affinities)
        if (!std::binary_search(
                recorded_memories.begin(), recorded_memories.end(), affinity.m))
          memories_to_log.emplace_back(affinity.m);
      record_affinities(memories_to_log);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_affinities(std::vector<Memory>& memories_to_log)
    //--------------------------------------------------------------------------
    {
      size_t total_update = 0;
      while (!memories_to_log.empty())
      {
        const Memory m = memories_to_log.back();
        memories_to_log.pop_back();
        // Eagerly log the processor description to the logging file so
        // that it appears before anything that needs it
        const MemDesc& mem = memory_descriptions.emplace_back(
            MemDesc{m.id, m.kind(), m.capacity()});
        total_update += sizeof(mem);
        recorded_memories.emplace_back(m);
        std::sort(recorded_memories.begin(), recorded_memories.end());
        std::vector<ProcessorMemoryAffinity> memory_affinities;
        runtime->machine.get_proc_mem_affinity(
            memory_affinities, Processor::NO_PROC, m);
        for (const ProcessorMemoryAffinity& mem_affinity : memory_affinities)
        {
          if (!std::binary_search(
                  recorded_processors.begin(), recorded_processors.end(),
                  mem_affinity.p))
          {
            ProcDesc& proc = processor_descriptions.emplace_back(ProcDesc());
            proc.proc_id = mem_affinity.p.id;
            proc.kind = mem_affinity.p.kind();
#ifdef LEGION_USE_CUDA
            if (!Realm::Cuda::get_cuda_device_uuid(
                    mem_affinity.p, &proc.cuda_device_uuid))
              proc.cuda_device_uuid[0] = 0;
#endif
            total_update += sizeof(proc);
            recorded_processors.emplace_back(mem_affinity.p);
            std::sort(recorded_processors.begin(), recorded_processors.end());
            std::vector<ProcessorMemoryAffinity> processor_affinities;
            runtime->machine.get_proc_mem_affinity(
                processor_affinities, mem_affinity.p);
            for (const ProcessorMemoryAffinity& proc_affinity :
                 processor_affinities)
              if (!std::binary_search(
                      recorded_memories.begin(), recorded_memories.end(),
                      proc_affinity.m))
                memories_to_log.emplace_back(proc_affinity.m);
          }
          const ProcMemDesc& affinity =
              procmem_affinities.emplace_back(ProcMemDesc{
                  mem_affinity.p.id, m.id, mem_affinity.bandwidth,
                  mem_affinity.latency});
          total_update += sizeof(affinity);
        }
      }
      update_footprint(total_update);
    }

    //--------------------------------------------------------------------------
    ProcID LegionProfiler::get_implicit_processor(void)
    //--------------------------------------------------------------------------
    {
      ProcID proc = implicit_top_level_task_proc.load();
      if (proc > 0)
        return proc;
      // Figure out how many local processors there are on this node
      Machine::ProcessorQuery query(runtime->machine);
      query.local_address_space();
      proc =
          Realm::ID::make_processor(runtime->address_space, query.count()).id;
      AutoLock p_lock(profiler_lock);
      // Check to see if we lost the race
      if (implicit_top_level_task_proc.load() > 0)
      {
        legion_assert(proc == implicit_top_level_task_proc.load());
        return proc;
      }
      implicit_top_level_task_proc.store(proc);
      assert(!external_implicit_task);
      external_implicit_task = runtime->generate_dynamic_task_id(false);
      // Record the processor kind as being an I/O kind so that the profiler
      // renders all implicit top-level tasks separately
      ProcDesc& desc = processor_descriptions.emplace_back(ProcDesc());
      desc.proc_id = proc;
      desc.kind = Processor::NO_KIND;
      // Also record a task and variant for external threads
      const TaskKind& kind = task_kinds.emplace_back(
          TaskKind{*external_implicit_task, "External Thread", true});
      const TaskVariant& variant = task_variants.emplace_back(
          TaskVariant{*external_implicit_task, 0, "External Thread"});
      update_footprint(sizeof(desc) + sizeof(kind) + sizeof(variant));
      return proc;
    }

    //--------------------------------------------------------------------------
    TaskID LegionProfiler::get_external_implicit_task(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(external_implicit_task);
      return *external_implicit_task;
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_task_request(
        Realm::ProfilingRequestSet& requests, TaskID tid, VariantID vid,
        UniqueID task_uid, Processor p, LgEvent critical)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_TASK);
      ProfilingInfo info(this, LEGION_PROF_TASK, task_uid);
      info.id = tid;
      info.extra.id2 = vid;
      info.critical = critical;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
          Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationEventWaits>();
      if (p.kind() == Processor::TOC_PROC)
        req.add_measurement<
            Realm::ProfilingMeasurements::OperationTimelineGPU>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(
        Realm::ProfilingRequestSet& requests, LgTaskID tid, Operation* op,
        LgEvent critical)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_META);
      ProfilingInfo info(this, LEGION_PROF_META, op);
      info.id = tid;
      info.critical = critical;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
          Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationEventWaits>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionProfiler::add_message_request(
        Realm::ProfilingRequestSet& requests, MessageKind k,
        Processor remote_target, LgEvent critical)
    //--------------------------------------------------------------------------
    {
      // Don't increment here, we'll increment on the remote side since we
      // that is where we know the profiler is going to handle the results
      ProfilingInfo info(nullptr, LEGION_PROF_MESSAGE, implicit_unique_op_id);
      info.id = LG_MESSAGE_ID + (int)k;
      info.critical = critical;
      // Record the spawn time which is different than the create_time in
      // the Realm profiling response because the create time is not recorded
      // until the active message makes it to the remote node and we want to
      // see how long it took for that active message to make it there
      // Do this last so it is as close the actual spawn as possible
      info.extra.spawn_time = Realm::Clock::current_time_in_nanoseconds();
      Realm::ProfilingRequest& req = requests.add_request(
          remote_target, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
          Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationEventWaits>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_copy_request(
        Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
        Operation* op, LgEvent critical, unsigned count,
        CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_COPY, count);
      ProfilingInfo info(this, LEGION_PROF_COPY, op);
      // Use ID to encode the collective copy kind
      info.id = collective;
      info.critical = critical;
      closure->add_reference(count);
      info.extra.closure = closure;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationCopyInfo>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_fill_request(
        Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
        Operation* op, LgEvent critical, CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_FILL);
      ProfilingInfo info(this, LEGION_PROF_FILL, op);
      // Use ID to encode the collective copy kind
      info.id = collective;
      info.critical = critical;
      closure->add_reference();
      info.extra.closure = closure;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationCopyInfo>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_inst_request(
        Realm::ProfilingRequestSet& requests, Operation* op,
        LgEvent unique_event)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_INST);
      ProfilingInfo info(this, LEGION_PROF_INST, op);
      info.id = unique_event.id;
      // Instances use two profiling requests so that we can get MemoryUsage
      // right away - the Timeline doesn't come until we delete the instance
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      req.add_measurement<Realm::ProfilingMeasurements::InstanceMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::InstanceTimeline>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_partition_request(
        Realm::ProfilingRequestSet& requests, Operation* op,
        DepPartOpKind part_op, LgEvent critical, LargeNameClosure* closure)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_PARTITION);
      ProfilingInfo info(this, LEGION_PROF_PARTITION, op);
      // Pass the part_op as the ID
      info.id = part_op;
      info.critical = critical;
      info.extra.closure = closure;
      if (closure != nullptr)
        closure->add_reference();
      Realm::ProfilingRequest& req = requests.add_request(
          (target_proc.exists()) ? target_proc :
                                   Processor::get_executing_processor(),
          LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_task_request(
        Realm::ProfilingRequestSet& requests, TaskID tid, VariantID vid,
        UniqueID uid, LgEvent critical)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_TASK);
      ProfilingInfo info(this, LEGION_PROF_TASK, uid);
      info.id = tid;
      info.extra.id2 = vid;
      info.critical = critical;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
          Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationEventWaits>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(
        Realm::ProfilingRequestSet& requests, LgTaskID tid, UniqueID uid,
        LgEvent critical)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_META);
      ProfilingInfo info(this, LEGION_PROF_META, uid);
      info.id = tid;
      info.critical = critical;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
          Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationEventWaits>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_copy_request(
        Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
        UniqueID uid, LgEvent critical, unsigned count,
        CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_COPY, count);
      ProfilingInfo info(this, LEGION_PROF_COPY, uid);
      // Use ID to encode the collective copy kind
      info.id = collective;
      info.critical = critical;
      closure->add_reference(count);
      info.extra.closure = closure;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationCopyInfo>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_fill_request(
        Realm::ProfilingRequestSet& requests, InstanceNameClosure* closure,
        UniqueID uid, LgEvent critical, CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_FILL);
      ProfilingInfo info(this, LEGION_PROF_FILL, uid);
      // Use ID to encode the collective copy kind
      info.id = collective;
      info.critical = critical;
      closure->add_reference();
      info.extra.closure = closure;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationCopyInfo>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_inst_request(
        Realm::ProfilingRequestSet& requests, UniqueID uid,
        LgEvent unique_event)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_INST);
      ProfilingInfo info(this, LEGION_PROF_INST, uid);
      info.id = unique_event.id;
      // Instances use two profiling requests so that we can get MemoryUsage
      // right away - the Timeline doesn't come until we delete the instance
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      req.add_measurement<Realm::ProfilingMeasurements::InstanceMemoryUsage>();
      req.add_measurement<Realm::ProfilingMeasurements::InstanceTimeline>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_partition_request(
        Realm::ProfilingRequestSet& requests, UniqueID uid,
        DepPartOpKind part_op, LgEvent critical, LargeNameClosure* closure)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_PARTITION);
      ProfilingInfo info(this, LEGION_PROF_PARTITION, uid);
      // Pass the partition op kind as the ID
      info.id = part_op;
      info.extra.closure = closure;
      if (closure != nullptr)
        closure->add_reference();
      info.critical = critical;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<Realm::ProfilingMeasurements::OperationFinishEvent>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::profile_barrier_arrival(
        Realm::Barrier bar, size_t count, LgEvent precondition,
        Realm::Event protected_precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(precondition.exists());
      increment_total_outstanding_requests(LEGION_PROF_ARRIVAL);
      // This is tricky: to measure when the arrival for this barrier is
      // actually done then we are going to run a no-op task when the
      // protected precondition triggers. It's a no-op because it is not
      // actually going to do anything, we're just using the 'ready' time
      // from its timeline to establish when the precondition has triggered
      // and then we'll use that to feed into the reduction in the barrier
      // to establish the last arrival for the barrier
      ProfilingInfo info(this, LEGION_PROF_ARRIVAL, implicit_unique_op_id);
      info.id = bar.id;
      info.extra.id2 = count;
      info.creator = implicit_fevent;
      info.critical = precondition;
      Realm::ProfilingRequestSet requests;
      // Give this high priority since it actually needs to arrive on the bar
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_RESOURCE_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      // Also give this high priority to run as soon as possible when ready
      // so we can get its profiling response back as well
      target_proc.spawn(
          Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, requests,
          protected_precondition, LG_RESOURCE_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::profile_barrier_trigger(
        Realm::Barrier bar, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_BARRIER);
      ProfilingInfo info(this, LEGION_PROF_BARRIER, uid);
      info.id = bar.id;
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingRequest& req = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_LOW_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::OperationStatus>();
      // Launch a no-op task with low priority just to get a profiling
      // response back once the barrier has triggered. This will also
      // ensure we subscribe to the barrier and get its result
      target_proc.spawn(
          Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, requests, bar,
          LG_LOW_PRIORITY);
    }

    //--------------------------------------------------------------------------
    bool LegionProfiler::update_previous_recorded_barrier(
        Realm::Barrier bar, Realm::Barrier& previous)
    //--------------------------------------------------------------------------
    {
      Realm::ID id(bar.id);
      legion_assert(bar.exists());
      legion_assert(id.is_barrier());
      const std::pair<unsigned, unsigned> key(
          id.barrier_creator_node(), id.barrier_barrier_idx());
      const unsigned generation = id.barrier_generation();
      AutoLock prof_lock(profiler_lock);
      std::map<std::pair<unsigned, unsigned>, unsigned>::iterator finder =
          recorded_barriers.find(key);
      if (finder != recorded_barriers.end())
      {
        // Already recorded through this generation
        if (generation <= finder->second)
          return false;
        previous.id =
            Realm::ID::make_barrier(
                finder->first.first, finder->first.second, finder->second)
                .id;
        if ((generation + 1) == Realm::Barrier::MAX_PHASES)
          recorded_barriers.erase(finder);
        else
          finder->second = generation;
      }
      else
      {
        previous.id = Realm::ID::make_barrier(
                          key.first, key.second, 0 /*base generation*/)
                          .id;
        if ((generation + 1) < Realm::Barrier::MAX_PHASES)
          recorded_barriers[key] = generation;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LegionProfiler::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      long long start = 0;
      if (self_profile)
        start = Realm::Clock::current_time_in_nanoseconds();
      legion_assert(response.user_data_size() == sizeof(ProfilingInfo));
      const ProfilingInfo* info =
          static_cast<const ProfilingInfo*>(response.user_data());
      switch (info->kind)
      {
        case LEGION_PROF_TASK:
          {
            Realm::ProfilingMeasurements::OperationProcessorUsage usage;
            // Check for predication and speculation
            if (response.get_measurement(usage))
            {
              implicit_profiler->process_proc_desc(usage.proc);
              implicit_profiler->process_task(info, response, usage);
            }
            break;
          }
        case LEGION_PROF_META:
          {
            Realm::ProfilingMeasurements::OperationProcessorUsage usage;
            // Check for predication and speculation
            if (response.get_measurement(usage))
            {
              implicit_profiler->process_proc_desc(usage.proc);
              implicit_profiler->process_meta(info, response, usage);
            }
            break;
          }
        case LEGION_PROF_MESSAGE:
          {
            Realm::ProfilingMeasurements::OperationProcessorUsage usage;
            // Check for predication and speculation
            if (response.get_measurement(usage))
            {
              implicit_profiler->process_proc_desc(usage.proc);
              implicit_profiler->process_message(info, response, usage);
            }
            break;
          }
        case LEGION_PROF_COPY:
          {
            Realm::ProfilingMeasurements::OperationMemoryUsage usage;
            // Check for predication and speculation
            if (response.get_measurement(usage))
            {
              implicit_profiler->process_mem_desc(usage.source);
              implicit_profiler->process_mem_desc(usage.target);
              implicit_profiler->process_copy(info, response, usage);
            }
            break;
          }
        case LEGION_PROF_FILL:
          {
            Realm::ProfilingMeasurements::OperationMemoryUsage usage;
            // Check for predication and speculation
            if (response.get_measurement(usage))
            {
              implicit_profiler->process_mem_desc(usage.target);
              implicit_profiler->process_fill(info, response, usage);
            }
            break;
          }
        case LEGION_PROF_INST:
          {
            // Record data based on which measurements we got back this time
            Realm::ProfilingMeasurements::InstanceAllocResult result;
            Realm::ProfilingMeasurements::InstanceTimeline timeline;
            Realm::ProfilingMeasurements::InstanceMemoryUsage usage;
            if (response.get_measurement(result) && result.success)
            {
              if (response.get_measurement(timeline) &&
                  response.get_measurement(usage))
              {
                implicit_profiler->process_mem_desc(usage.memory);
                implicit_profiler->process_inst_timeline(
                    info, response, usage, timeline);
              }
              else
                std::abort();
            }
            else
              failed_alloc = true;
            break;
          }
        case LEGION_PROF_PARTITION:
          {
            implicit_profiler->process_partition(info, response);
            break;
          }
        case LEGION_PROF_ARRIVAL:
          {
            Realm::ProfilingMeasurements::OperationTimeline timeline;
            if (!response.get_measurement(timeline))
              std::abort();
            implicit_profiler->process_arrival(info, timeline);
            break;
          }
        case LEGION_PROF_BARRIER:
          {
            Realm::ProfilingMeasurements::OperationStatus status;
            // Check for predication and specuation
            if (response.get_measurement(status) &&
                (status.result == Realm::ProfilingMeasurements::
                                      OperationStatus::COMPLETED_SUCCESSFULLY))
            {
              LgEvent barrier;
              barrier.id = info->id;
              implicit_profiler->record_barrier_use(barrier, info->op_id);
            }
            break;
          }
        case LEGION_PROF_TRIGGER:
          {
            Realm::ProfilingMeasurements::OperationTimeline timeline;
            if (!response.get_measurement(timeline))
              std::abort();
            switch (static_cast<EffectKind>(info->extra.id2))
            {
              case ASYNC_EFFECT:
                implicit_profiler->process_async_effect(info, timeline);
                break;
              case MAKE_VALID_EFFECT:
                implicit_profiler->process_make_valid(info, timeline);
                break;
              case FETCH_METADATA_EFFECT:
                implicit_profiler->process_fetch_metadata(info, timeline);
                break;
            }
            break;
          }
        default:
          std::abort();
      }
      // Have to do self-profiling here before the decrement to avoid races
      // with the shutdown code
      if (self_profile)
      {
        const Processor proc = Processor::get_executing_processor();
        implicit_profiler->process_proc_desc(proc);
        if (info->kind == LEGION_PROF_INST)
        {
          if (failed_alloc)
            fevent = info->creator;
          else
            fevent.id = info->id;
          const long long stop = Realm::Clock::current_time_in_nanoseconds();
          implicit_profiler->record_proftask(
              proc, info->op_id, start, stop, fevent, implicit_fevent,
              true /*completion*/);
        }
        else
        {
          Realm::ProfilingMeasurements::OperationFinishEvent finish;
          if (!response.get_measurement(finish))
            std::abort();
          const long long stop = Realm::Clock::current_time_in_nanoseconds();
          implicit_profiler->record_proftask(
              proc, info->op_id, start, stop, LgEvent(finish.finish_event),
              implicit_fevent, true /*completion*/);
        }
      }
      decrement_total_outstanding_requests(info->kind);
      // Already recorded the prof task profiling in this case
      return false;
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::finalize(void)
    //--------------------------------------------------------------------------
    {
      // Remove our guard outstanding request
      decrement_total_outstanding_requests(LEGION_PROF_META);
      CalibrationErr calibration_err;
      calibration_err.calibration_err = Realm::Clock::get_calibration_error();
      if (!done_event.has_triggered())
        done_event.wait();
      // Make sure the last dumping task is done as well
      if (!last_dump_task.has_triggered())
        last_dump_task.wait();
      for (const ProcDesc& proc : processor_descriptions)
        serializer->serialize(proc);
      for (const MemDesc& mem : memory_descriptions) serializer->serialize(mem);
      for (const ProcMemDesc& desc : procmem_affinities)
        serializer->serialize(desc);
      for (const TaskKind& kind : task_kinds) serializer->serialize(kind);
      for (const TaskVariant& variant : task_variants)
        serializer->serialize(variant);
      drain_pending_backtraces(false /*update footprint*/);
      for (const Backtrace& bt : backtraces) serializer->serialize(bt);
      for (const MapperName& mapper : mapper_names)
        serializer->serialize(mapper);
      for (const Provenance& prov : provenances) serializer->serialize(prov);
      LegionProfInstance* inst = instances;
      while (inst != nullptr)
      {
        inst->dump_state(serializer);
        inst = inst->next.load();
      }
      serializer->serialize(calibration_err);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_mapper_name(
        MapperID mapper, Processor proc, const char* name)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(profiler_lock);
      const MapperName& entry =
          mapper_names.emplace_back(MapperName{mapper, proc.id, name});
      update_footprint(sizeof(entry) + entry.name.size());
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_mapper_call_kinds(
        const char* const * const mapper_call_names,
        unsigned int num_mapper_calls)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_mapper_calls; idx++)
      {
        MapperCallDesc mapper_call_desc;
        mapper_call_desc.kind = idx;
        mapper_call_desc.name = mapper_call_names[idx];
        serializer->serialize(mapper_call_desc);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_runtime_call_kinds(
        const char* const * const runtime_call_names,
        unsigned int num_runtime_calls)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_runtime_calls; idx++)
      {
        RuntimeCallDesc runtime_call_desc;
        runtime_call_desc.kind = idx;
        runtime_call_desc.name = runtime_call_names[idx];
        serializer->serialize(runtime_call_desc);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_provenance(
        ProvenanceID pid, const char* provenance, size_t size)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(profiler_lock);
      const Provenance& prov = provenances.emplace_back(
          Provenance{pid, std::string(provenance, size)});
      update_footprint(sizeof(prov) + prov.provenance.size());
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::increment_outstanding_message_request(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_fevent.exists());
      // This part is a bit tricky: we want the implicit_fevent to always be
      // an event local to this node so that the profiler can always look up
      // which node to consult based on the fevent for a task. However, Realm
      // creates the finish_event for messages on the node where they are
      // spawned and not where they are run, so we have to rename the
      // implicit_fevent here to an event local to this node, so we do that
      // here before we handle the message, and then we pretend like the
      // actuall finish event is a user event triggered after we get the
      // profiling response for this task.
      const LgEvent original_fevent = implicit_fevent;
      Runtime::rename_event(implicit_fevent);
      // Well this is fun, we might even block on this lock acquire so
      // make sure we've set up our implicit fevent correctly
      AutoLock prof_lock(profiler_lock);
      // Save the current implicit fevent so we can look it up later
      message_fevents[implicit_fevent] = original_fevent;
#ifdef LEGION_DEBUG
      total_outstanding_requests[LEGION_PROF_MESSAGE]++;
#else
      total_outstanding_requests.fetch_add(1);
#endif
    }

    //--------------------------------------------------------------------------
    LgEvent LegionProfiler::find_message_fevent(LgEvent fevent, bool remove)
    //--------------------------------------------------------------------------
    {
      AutoLock prof_lock(profiler_lock);
      std::map<LgEvent, LgEvent>::iterator finder =
          message_fevents.find(fevent);
      legion_assert(finder != message_fevents.end());
      const LgEvent result = finder->second;
      message_fevents.erase(finder);
      // Reverse the order so we can find it the other way in the response
      if (!remove)
        message_fevents[result] = fevent;
      return result;
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::increment_total_outstanding_requests(
        ProfilingKind kind, unsigned cnt)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      AutoLock p_lock(profiler_lock);
      total_outstanding_requests[kind] += cnt;
#else
      total_outstanding_requests.fetch_add(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::decrement_total_outstanding_requests(
        ProfilingKind kind, unsigned cnt)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      AutoLock p_lock(profiler_lock);
      legion_assert(total_outstanding_requests[kind] >= cnt);
      total_outstanding_requests[kind] -= cnt;
      if (total_outstanding_requests[kind] > 0)
        return;
      for (unsigned idx = 0; idx < LEGION_PROF_LAST; idx++)
      {
        if (idx == kind)
          continue;
        if (total_outstanding_requests[idx] > 0)
          return;
      }
      legion_assert(!done_event.has_triggered());
      done_event.trigger();
#else
      unsigned prev = total_outstanding_requests.fetch_sub(cnt);
      legion_assert(prev >= cnt);
      // If we were the last outstanding event we can trigger the event
      if (prev == cnt)
      {
        legion_assert(!done_event.has_triggered());
        done_event.trigger();
      }
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::measure_event_trigger(
        LgEvent event, EffectKind kind, LgEvent fevent, size_t extra_data,
        const char* prov)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests(LEGION_PROF_TRIGGER);
      ProfilingInfo info(this, LEGION_PROF_TRIGGER, implicit_provenance);
      if (prov != nullptr)
      {
        legion_assert(extra_data == 0);
        Legion::Internal::Provenance* provenance =
            runtime->find_or_create_provenance(prov, strlen(prov));
        info.id = provenance->pid;
      }
      else
        info.id = extra_data;
      info.extra.id2 = kind;
      info.creator = fevent;
      info.critical = event;
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingRequest& request = requests.add_request(
          target_proc, LG_LEGION_PROFILING_ID, &info, sizeof(info),
          LG_MIN_PRIORITY);
      request
          .add_measurement<Realm::ProfilingMeasurements::OperationTimeline>();
      // Spawn a no-op task with a profiling response to measure when
      // the event triggered
      target_proc.spawn(
          Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, requests,
          Realm::Event::ignorefaults(event), LG_MIN_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::update_footprint(size_t diff, LegionProfInstance* inst)
    //--------------------------------------------------------------------------
    {
      // This function must be lock-free because it can be called inside
      // an event wait when we're already trying to wait on an event and
      // therefore it is not safe to take a lock and try to wait again
      inst->footprint += diff;
      size_t footprint = total_memory_footprint.fetch_add(diff) + diff;
      if (footprint > output_footprint_threshold)
      {
        // Extract a new prof instance so we can dump out its current state
        // and save the clone as the new implicit profiler
        total_memory_footprint.fetch_sub(inst->footprint);
        LegionProfInstance* dump = inst->dump();
        // Enqueue this on the dump list and if it is the first one
        // then launch a profiler dump task to start writing it out
        LegionProfInstance* head = dump_list.load();
        dump->next.store(head);
        while (!dump_list.compare_exchange_weak(head, dump))
          dump->next.store(head);
        if (head == nullptr)
        {
          const RtUserEvent dump_event = Runtime::create_rt_user_event();
          ProfilerDumpArgs args(this, dump_event);
          const RtEvent precondition = last_dump_task;
          last_dump_task = dump_event;
          runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, precondition);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::update_footprint(size_t diff)
    //--------------------------------------------------------------------------
    {
      // We're already holding the lock when this is called so we can't take
      // it again but it also doesn't protect us with races from the calls to
      // update_footprint from the instances, so we need to be careful
      size_t footprint = total_memory_footprint.fetch_add(diff) + diff;
      LegionProfInstance* head = dump_list.load();
      if ((footprint > output_footprint_threshold) && (head == nullptr) &&
          dump_list.compare_exchange_strong(
              head, reinterpret_cast<LegionProfInstance*>(DUMP_NOW)))
      {
        const RtUserEvent dump_event = Runtime::create_rt_user_event();
        ProfilerDumpArgs args(this, dump_event);
        const RtEvent precondition = last_dump_task;
        last_dump_task = dump_event;
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY, precondition);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::dump_instances(RtUserEvent dump_event)
    //--------------------------------------------------------------------------
    {
      // Start the timing so we know how long we are taking
      const long long t_start = Realm::Clock::current_time_in_microseconds();
      // Scale our latency by how much we are over the space limit
      const long long t_stop = t_start + output_target_latency;
      LegionProfInstance* instance = nullptr;
      do {
        // Extract out the things we're going to dump
        std::deque<ProcDesc> dump_processor_descriptions;
        std::deque<MemDesc> dump_memory_descriptions;
        std::deque<ProcMemDesc> dump_procmem_affinities;
        std::deque<TaskKind> dump_task_kinds;
        std::deque<TaskVariant> dump_task_variants;
        std::deque<Backtrace> dump_backtraces;
        std::deque<MapperName> dump_mapper_names;
        std::deque<Provenance> dump_provenances;
        {
          // Need to hold the lock to swap these
          AutoLock p_lock(profiler_lock);
          drain_pending_backtraces(false /*update footprint*/);
          dump_processor_descriptions.swap(processor_descriptions);
          dump_memory_descriptions.swap(memory_descriptions);
          dump_procmem_affinities.swap(procmem_affinities);
          dump_task_kinds.swap(task_kinds);
          dump_task_variants.swap(task_variants);
          dump_backtraces.swap(backtraces);
          dump_mapper_names.swap(mapper_names);
          dump_provenances.swap(provenances);
          // Lock doesn't protect us from popping the first thing off
          // the list so have to use a lock-free algorithm here
          instance = dump_list.load();
          if (instance != nullptr)
          {
            while (reinterpret_cast<uintptr_t>(instance) == DUMP_NOW)
            {
              if (dump_list.compare_exchange_weak(instance, nullptr))
                break;
            }
            while (reinterpret_cast<uintptr_t>(instance) != DUMP_NOW)
            {
              LegionProfInstance* next = instance->next.load();
              if (dump_list.compare_exchange_weak(
                      instance,
                      next == reinterpret_cast<LegionProfInstance*>(DUMP_NOW) ?
                          nullptr :
                          next))
                break;
            }
          }
        }
        size_t diff = 0;
        // Very important that we always write out all the deques before
        // dumping any profiling instances since the profiler assumes
        // that some of these things are ordered in the log file. It
        // doesn't matter what order we dump out the profiling instances
        // themselves though which is why we can get away with LIFO
        for (const ProcDesc& proc : dump_processor_descriptions)
          serializer->serialize(proc);
        diff += dump_processor_descriptions.size() * sizeof(ProcDesc);
        for (const MemDesc& mem : dump_memory_descriptions)
          serializer->serialize(mem);
        diff += dump_memory_descriptions.size() * sizeof(MemDesc);
        for (const ProcMemDesc& desc : dump_procmem_affinities)
          serializer->serialize(desc);
        diff += dump_procmem_affinities.size() * sizeof(ProcMemDesc);
        for (const TaskKind& kind : dump_task_kinds)
        {
          serializer->serialize(kind);
          diff += kind.name.size();
        }
        diff += dump_task_kinds.size() * sizeof(TaskKind);
        for (const TaskVariant& variant : dump_task_variants)
        {
          serializer->serialize(variant);
          diff += variant.name.size();
        }
        diff += dump_task_variants.size() * sizeof(TaskVariant);
        for (const Backtrace& bt : dump_backtraces)
        {
          serializer->serialize(bt);
          diff += bt.backtrace.size();
        }
        diff += dump_backtraces.size() * sizeof(Backtrace);
        for (const MapperName& mapper : dump_mapper_names)
        {
          serializer->serialize(mapper);
          diff += mapper.name.size();
        }
        diff += dump_mapper_names.size() * sizeof(MapperName);
        for (const Provenance& prov : dump_provenances)
        {
          serializer->serialize(prov);
          diff += prov.provenance.size();
        }
        diff += dump_provenances.size() * sizeof(Provenance);
        total_memory_footprint.fetch_sub(diff);
        // If we don't have an instance to handle we're done
        if ((instance == nullptr) ||
            (reinterpret_cast<uintptr_t>(instance) == DUMP_NOW) ||
            !instance->dump_inter(serializer, t_stop))
          break;
        // Done with this instance, go around again
        delete instance;
      } while (true);
      if ((instance != nullptr) &&
          (reinterpret_cast<uintptr_t>(instance) != DUMP_NOW))
      {
        // We still have a partial instance, if the partial
        // instance has a next then that means there was always
        // a guard in place and we need to launch a continuation
        const bool launch_continuation =
            (instance->next.load() != nullptr) &&
            (reinterpret_cast<uintptr_t>(instance->next.load()) != DUMP_NOW);
        // Add the instance back into the list
        LegionProfInstance* head = dump_list.load();
        instance->next.store(head);
        while (!dump_list.compare_exchange_weak(head, instance))
          instance->next.store(head);
        if (launch_continuation || (head == nullptr))
        {
          ProfilerDumpArgs args(this, dump_event);
          // Don't use a precondition or you'll make us depend on our
          // own dump_event!
          legion_assert(last_dump_task == dump_event);
          runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
        }
        else
          Runtime::trigger_event(dump_event);
      }
      else
        Runtime::trigger_event(dump_event);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::ProfilerDumpArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      profiler->dump_instances(dump_event);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::issue_default_mapper_warning(
        Operation* op, const char* mapper_call_name)
    //--------------------------------------------------------------------------
    {
      // We'll skip any warnings for now with no operation
      if (op == nullptr)
        return;
      // We'll only issue this warning once on each node for now
      if (!need_default_mapper_warning.exchange(false /*no longer needed*/))
        return;
      // Check to see if the application has registered other mappers other
      // than the default mapper, if it has then we don't issue this warning
      if (runtime->has_non_default_mapper())
        return;
      // Give a massive warning for profilig when using the default mapper
      for (int i = 0; i < 2; i++)
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      for (int i = 0; i < 4; i++)
        fprintf(stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
      for (int i = 0; i < 2; i++)
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      fprintf(stderr, "!!! YOU ARE PROFILING USING THE DEFAULT MAPPER!!!\n");
      fprintf(stderr, "!!! THE DEFAULT MAPPER IS NOT FOR PERFORMANCE !!!\n");
      fprintf(stderr, "!!! PLEASE CUSTOMIZE YOUR MAPPER TO YOUR      !!!\n");
      fprintf(stderr, "!!! APPLICATION AND TO YOUR TARGET MACHINE    !!!\n");
      InnerContext* context = op->get_context();
      if (op->get_operation_kind() == TASK_OP_KIND)
      {
        TaskOp* task = static_cast<TaskOp*>(op);
        if (context->get_owner_task() != nullptr)
          fprintf(
              stderr,
              "First use of the default mapper in address space %d\n"
              "occurred when task %s (UID %lld) in parent task %s "
              "(UID %lld)\ninvoked the \"%s\" mapper call\n",
              runtime->address_space, task->get_task_name(),
              task->get_unique_op_id(), context->get_task_name(),
              context->get_unique_id(), mapper_call_name);
        else
          fprintf(
              stderr,
              "First use of the default mapper in address space %d\n"
              "occurred when task %s (UID %lld) invoked the \"%s\" "
              "mapper call\n",
              runtime->address_space, task->get_task_name(),
              task->get_unique_op_id(), mapper_call_name);
      }
      else
        fprintf(
            stderr,
            "First use of the default mapper in address space %d\n"
            "occurred when %s (UID %lld) in parent task %s "
            "(UID %lld)\ninvoked the \"%s\" mapper call\n",
            runtime->address_space, op->get_logging_name(),
            op->get_unique_op_id(), context->get_task_name(),
            context->get_unique_id(), mapper_call_name);
      for (int i = 0; i < 2; i++)
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      for (int i = 0; i < 4; i++)
        fprintf(stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
      for (int i = 0; i < 2; i++)
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      fprintf(stderr, "\n");
      fflush(stderr);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::instantiate_profiling_instance(void)
    //--------------------------------------------------------------------------
    {
      if (implicit_profiler != nullptr)
        return;
      // This function needs to be lock free to avoid taking a lock and
      // possibly waiting on an event before we've had a chance to even
      // create an LegionProfInstance to record the data associated with it.
      Processor current = Processor::get_executing_processor();
      LgEvent external;
      if (!current.exists())
      {
        const Realm::UserEvent ext = Realm::UserEvent::create_user_event();
        ext.trigger();
        external = LgEvent(ext);
        // Also get the implicit processor to make sure it exists
        // and register the external top-level task
        // Note this call to get_implicit_processor could possibly take
        // a lock and end up waiting on an event, but we don't really care
        // that much because we're on an external thread
        current.id = get_implicit_processor();
      }
      // We can share these profiler instances on anything other than I/O
      // processors which can have multiple threads running at the same time
      else if (current.kind() != Processor::IO_PROC)
      {
        // Scan the list as it exists right now checking to see if the
        // processor already has a profiling instance. While this list
        // could be stale immediately we know its fine because any processor
        // adding to the list must be different than ours so either our
        // processor is already in the list or not. In this sense races
        // are completely benign.
        LegionProfInstance* instance = instances.load();
        while (instance != nullptr)
        {
          if (instance->local_proc == current)
          {
            implicit_profiler = instance;
            return;
          }
          instance = instance->next.load();
        }
      }
      // Make our instance
      implicit_profiler = new LegionProfInstance(this, current, external);
      // Only do this after we've set up our thread-local profiler as
      // it might take a lock and end up waiting on an event
      if (!external.exists())
        record_processor(current);
      // Add it to the list lock free
      LegionProfInstance* head = instances.load();
      implicit_profiler->next.store(head);
      while (!instances.compare_exchange_weak(head, implicit_profiler))
        implicit_profiler->next.store(head);
    }

  }  // namespace Internal
}  // namespace Legion
