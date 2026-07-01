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

#include "legion/operations/operation.h"
#include "legion/analysis/logical.h"
#include "legion/analysis/projection.h"
#include "legion/analysis/update.h"
#include "legion/analysis/valid.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/virtual.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/region.h"
#include "legion/operations/mustepoch.h"
#include "legion/tools/spy.h"
#include "legion/tracing/logical.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"
#include "legion/views/replicate.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Operation
    /////////////////////////////////////////////////////////////

    const char* const Operation::op_names[] = {
#define LEGION_OPERATION_NAMES(kind, name) name,
        LEGION_OPERATION_KINDS(LEGION_OPERATION_NAMES)
#undef LEGION_OPERATION_NAMES
    };

    //--------------------------------------------------------------------------
    Operation::Operation(void)
      : gen(0), unique_op_id(0), context_index(0),
        outstanding_mapping_references(0), hardened_notifications(0),
        hardened(false), track_parent(false), tracing(false), trace(nullptr),
        trace_local_id(0), parent_ctx(nullptr), must_epoch(nullptr),
        provenance(nullptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      activated = false;
#endif
    }

    //--------------------------------------------------------------------------
    Operation::~Operation(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ const char* Operation::get_string_rep(OpKind kind)
    //--------------------------------------------------------------------------
    {
      return op_names[kind];
    }

    //--------------------------------------------------------------------------
    void Operation::activate(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      legion_assert(!activated);
      activated = true;
#endif
      // Get a new unique ID for this operation
      unique_op_id = runtime->get_unique_operation_id();
      context_index = 0;
      exception_handler = 0;
      // Set a guard reference that will be removed when the logical
      // dependence analysis finishes
      remaining_mapping_dependences.store(1);
      // Set a guard reference that will be removed when the operation
      // becomes complete
      remaining_commit_dependences.store(1);
      outstanding_mapping_references = 0;
      hardened_notifications = 0;
      prepipelined = 0;
      mapped = false;
      executed = false;
      completed = false;
      committed = false;
      hardened = false;
      track_parent = false;
      parent_ctx = nullptr;
      prepipelined_event = RtUserEvent::NO_RT_USER_EVENT;
      mapped_event = RtUserEvent::NO_RT_USER_EVENT;
      execution_fence_event = ApEvent::NO_AP_EVENT;
      completion_event.pending = ApUserEvent::NO_AP_USER_EVENT;
      completion_set = false;
      commit_event = RtUserEvent::NO_RT_USER_EVENT;
      trace = nullptr;
      tracing = false;
      trace_local_id = (unsigned)-1;
      must_epoch = nullptr;
      provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    void Operation::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      legion_assert(!freeop);
      legion_assert(activated);
      legion_assert(mapped_event.has_triggered());
      legion_assert(commit_event.has_triggered());
#ifdef LEGION_DEBUG
      activated = false;
#endif
      // Generation is bumped when we committed
      incoming.clear();
      outgoing.clear();
      verification_notifications.clear();
      completion_effects.clear();
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    size_t Operation::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return 0;
    }

    //--------------------------------------------------------------------------
    void Operation::analyze_region_requirements(
        IndexSpaceNode* launch_space, ShardingFunction* func,
        IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
      // We can skip doing the analysis for logical regions if we
      // are replaying a logical trace
      if ((trace != nullptr) && !trace->is_recording())
        return;

      LogicalAnalysis logical_analysis(this, get_output_offset());

      unsigned req_count = get_region_count();
      for (unsigned i = 0; i < req_count; i++)
      {
        const RegionRequirement& req = get_requirement(i);

        ProjectionInfo projection_info(&req, launch_space, func, shard_space);

        perform_dependence_analysis(i, req, projection_info, logical_analysis);
      }
    }

    //--------------------------------------------------------------------------
    unsigned Operation::get_output_offset() const
    //--------------------------------------------------------------------------
    {
      return LogicalAnalysis::NO_OUTPUT_OFFSET;
    }

    //--------------------------------------------------------------------------
    Mappable* Operation::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      // should never be called on this class
      return nullptr;
    }

    //--------------------------------------------------------------------------
    unsigned Operation::get_operation_depth(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(parent_ctx != nullptr);
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    void Operation::set_trace(LogicalTrace* t, bool recording, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      legion_assert(trace == nullptr);
      legion_assert(t != nullptr);
      trace = t;
      tracing = recording;
      if (runtime->safe_tracing)
        record_trace_hash(*trace, opidx);
    }

    //--------------------------------------------------------------------------
    uint64_t Operation::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return (must_epoch == nullptr) ? context_index :
                                       must_epoch->get_context_index();
    }

    //--------------------------------------------------------------------------
    std::optional<uint64_t> Operation::get_context_index(GenerationID g) const
    //--------------------------------------------------------------------------
    {
      if (g < gen.load())
        return std::optional<uint64_t>();
      AutoLock o_lock(op_lock, false /*exclusive*/);
      // Make sure we didn't lose the race
      if (g < gen.load())
        return std::optional<uint64_t>();
      legion_assert(g == gen.load());
      legion_assert(track_parent);
      return std::optional<uint64_t>(context_index);
    }

    //--------------------------------------------------------------------------
    void Operation::set_context_index(
        uint64_t index, ExceptionHandlerID handler)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch == nullptr);
      track_parent = true;
      context_index = index;
      exception_handler = handler;
      LegionSpy::log_child_operation_index(
          parent_ctx->get_unique_id(), context_index, unique_op_id);
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID Operation::get_exception_handler(void)
    //--------------------------------------------------------------------------
    {
      if (must_epoch == nullptr)
        return exception_handler;
      else
        return must_epoch->get_exception_handler();
    }

    //--------------------------------------------------------------------------
    void Operation::set_must_epoch(MustEpochOp* epoch, bool do_registration)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch == nullptr);
      legion_assert(epoch != nullptr);
      must_epoch = epoch;
      if (do_registration)
        must_epoch->register_subop(this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::localize_region_requirement(RegionRequirement& r)
    //--------------------------------------------------------------------------
    {
      legion_assert(r.handle_type == LEGION_SINGULAR_PROJECTION);
      r.parent = r.region;
      r.prop = LEGION_EXCLUSIVE;
      // If we're doing a write discard, then we can add read privileges
      // inside our task since it is safe to read what we wrote
      if (IS_WRITE_DISCARD(r))
        r.privilege |= (LEGION_READ_PRIV | LEGION_REDUCE_PRIV);
      // Then remove any discard and collective masks from the privileges
      r.privilege = FILTER_DISCARD(r);
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::release_nonempty_acquired_instances(
        RtEvent perform,
        std::map<PhysicalManager*, unsigned>& acquired_instances)
    //--------------------------------------------------------------------------
    {
      if (perform.exists() && !perform.has_triggered())
      {
        std::vector<std::pair<PhysicalManager*, unsigned> >* to_release =
            nullptr;
        for (std::map<PhysicalManager*, unsigned>::iterator it =
                 acquired_instances.begin();
             it != acquired_instances.end();)
        {
          size_t instance_size = it->first->get_instance_size();
          if (instance_size > 0)
          {
            if (to_release == nullptr)
              to_release =
                  new std::vector<std::pair<PhysicalManager*, unsigned> >();
            to_release->emplace_back(std::make_pair(it->first, it->second));
            std::map<PhysicalManager*, unsigned>::iterator to_delete = it++;
            acquired_instances.erase(to_delete);
          }
          else
            it++;
        }
        if (to_release != nullptr)
        {
          DeferReleaseAcquiredArgs args(this, to_release);
          return runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, perform);
        }
        else
          return perform;
      }
      for (std::map<PhysicalManager*, unsigned>::iterator it =
               acquired_instances.begin();
           it != acquired_instances.end();)
      {
        size_t instance_size = it->first->get_instance_size();
        if (instance_size > 0)
        {
          if (it->first->remove_base_valid_ref(MAPPING_ACQUIRE_REF, it->second))
            delete it->first;
          std::map<PhysicalManager*, unsigned>::iterator to_delete = it++;
          acquired_instances.erase(to_delete);
        }
        else
          it++;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::release_acquired_instances(
        std::map<PhysicalManager*, unsigned>& acquired_instances)
    //--------------------------------------------------------------------------
    {
      for (std::pair<PhysicalManager* const, unsigned>& instance_pair :
           acquired_instances)
        if (instance_pair.first->remove_base_valid_ref(
                MAPPING_ACQUIRE_REF, instance_pair.second))
          delete instance_pair.first;
      acquired_instances.clear();
    }

    //--------------------------------------------------------------------------
    void Operation::log_mapping_decision(
        unsigned index, const RegionRequirement& req,
        const InstanceSet& targets, bool postmapping /*=false*/) const
    //--------------------------------------------------------------------------
    {
      if ((spy_logging_level == NO_SPY_LOGGING) &&
          (runtime->profiler == nullptr))
        return;
      FieldSpaceNode* node =
          (req.handle_type != LEGION_PARTITION_PROJECTION) ?
              runtime->get_node(req.region.get_field_space()) :
              runtime->get_node(req.partition.get_field_space());
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const InstanceRef& inst = targets[idx];
        const FieldMask& valid_mask = inst.get_valid_fields();
        std::vector<FieldID> valid_fields;
        node->get_field_set(valid_mask, parent_ctx, valid_fields);
        InstanceManager* manager = inst.get_manager();
        const LgEvent inst_event =
            manager->is_virtual_manager() ?
                LgEvent::NO_LG_EVENT :
                manager->as_physical_manager()->get_unique_event();
        if (spy_logging_level > NO_SPY_LOGGING)
        {
          for (const FieldID& field_id : valid_fields)
          {
            if (postmapping)
              LegionSpy::log_post_mapping_decision(
                  unique_op_id, index, field_id, inst_event);
            else
              LegionSpy::log_mapping_decision(
                  unique_op_id, index, field_id, inst_event);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Operation::log_virtual_mapping(
        unsigned index, const RegionRequirement& req) const
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      for (const FieldID& field_id : req.privilege_fields)
        LegionSpy::log_mapping_decision(
            unique_op_id, index, field_id, ApEvent::NO_AP_EVENT /*inst event*/);
    }

    //--------------------------------------------------------------------------
    void Operation::DeferReleaseAcquiredArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<PhysicalManager*, unsigned>& instance : *instances)
      {
        if (instance.first->remove_base_valid_ref(
                MAPPING_ACQUIRE_REF, instance.second))
          delete instance.first;
      }
      delete instances;
    }

    //--------------------------------------------------------------------------
    void Operation::initialize_operation(
        InnerContext* ctx, Provenance* prov /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      parent_ctx = ctx;
      provenance = prov;
      if (provenance != nullptr)
      {
        provenance->add_reference();
        LegionSpy::log_operation_provenance(unique_op_id, prov->human);
      }
      if (implicit_profiler != nullptr)
        implicit_profiler->register_operation(this);
    }

    //--------------------------------------------------------------------------
    void Operation::set_provenance(Provenance* prov, bool has_ref)
    //--------------------------------------------------------------------------
    {
      legion_assert(provenance == nullptr);
      provenance = prov;
      if ((provenance != nullptr) && !has_ref)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::execute_prepipeline_stage(
        GenerationID generation, bool from_logical_analysis)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock op(op_lock);
        legion_assert(generation <= gen);
        if (generation < gen)
          return RtEvent::NO_RT_EVENT;
        // Check to see if we've already started the analysis
        if (prepipelined > 0)
        {
          // Someone else already started, figure out if we need to wait
          if (prepipelined == 1)
          {
            // Only partially through so make an event to trigger when done
            legion_assert(from_logical_analysis);
            legion_assert(!prepipelined_event.exists());
            prepipelined_event = Runtime::create_rt_user_event();
            return prepipelined_event;
          }
          else
            return RtEvent::NO_RT_EVENT;
        }
        else
        {
          // We got here first, mark that we're doing it
          prepipelined = from_logical_analysis ? 2 : 1;
        }
      }
      trigger_prepipeline_stage();
      // Trigger any guard events we might have
      if (!from_logical_analysis)
      {
        AutoLock op(op_lock);
        legion_assert(prepipelined == 1);
        prepipelined = 2;
        if (prepipelined_event.exists())
          Runtime::trigger_event(prepipelined_event);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void Operation::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Check to make sure our prepipeline stage is done if we have one
      if (has_prepipeline_stage())
      {
        RtEvent wait_on = execute_prepipeline_stage(gen, true /*need wait*/);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Always wrap this call with calls to begin/end dependence analysis
      begin_dependence_analysis();
      trigger_dependence_analysis();
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    bool Operation::has_prepipeline_stage(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called by inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do in the base case
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Put this thing on the ready queue
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we finished mapping
      complete_mapping();
      // The execution stage only gets invoked if you call it explicitly
      // We do so here to ensure that we call the complete execution method
      trigger_execution();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we finished execution
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void Operation::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/);
    }

    //--------------------------------------------------------------------------
    void Operation::verify_requirement(
        const RegionRequirement& req, unsigned index,
        bool allow_projections) const
    //--------------------------------------------------------------------------
    {
      // If this is a NO-ACCESS requirement then we don't care
      if (IS_NO_ACCESS(req) || (req.flags & LEGION_VERIFIED_FLAG))
        return;
      // Check the sanity of the privileges
      // Make sure that none of the unused-bits are used
      if (req.privilege & ~(LEGION_DISCARD_OUTPUT_MASK | LEGION_WRITE_DISCARD))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Region requirement " << index << " of " << *this
              << " has an improperly formed privilege mode " << std::hex
              << req.privilege << std::dec << ".";
        error.raise();
      }
      if (IS_REDUCE(req))
      {
        // Must have a non-zero reduction operator
        if (req.redop == 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Region requirement " << index << " of " << *this
                << " must have a non-zero reduction operator "
                << "when requesting reduction privileges.";
          error.raise();
        }
        // No discards allowed
        if (IS_WRITE_DISCARD(req))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Region requirement " << index << " of " << *this
                << " requested illegal discard-input modifier "
                << "with reduction privileges. Reduction privileges are not "
                << "permitted to specify any kind of discard modifier.";
          error.raise();
        }
        if (IS_OUTPUT_DISCARD(req))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Region requirement " << index << " of " << *this
                << " requested illegal discard-output modifier "
                << "with reduction privileges. Reduction privileges are not "
                << "permitted to specify any kind of discard modifier.";
          error.raise();
        }
      }
      else
      {
        // Make sure reduction operator is zero
        if (req.redop != 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Region requirement " << index << " of " << *this
                << " must not specify a reduction operator when "
                << "using non-reduction privileges.";
          error.raise();
        }
        // Make sure no input discards on read-only privileges
        if (IS_READ_ONLY(req) && IS_WRITE_DISCARD(req))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Region requirement " << index << " of " << *this
                << " requested illegal discard-input modifier "
                << "with read-only privileges. This is guaranteed to result in "
                   "the "
                << "use of uninitialized data and is therefore illegal.";
          error.raise();
        }
        if (IS_WRITE_ONLY(req) && IS_OUTPUT_DISCARD(req) &&
            ((req.flags & LEGION_SUPPRESS_WARNINGS_FLAG) == 0))
        {
          Warning warning;
          warning
              << "Region requirement " << index << " of " << *this
              << "combinded output-discard qualifer with a write-only "
              << "privilege which means this region is never used anywhere. "
              << "Are you sure you know what you are doing?";
          warning.raise();
        }
      }
      // Make sure that none of the unused bits are set for coherence
      if (req.prop & ~LEGION_COLLECTIVE_RELAXED)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Region requirement " << index << " of " << *this
              << " has an improperly formed coherence mode " << std::hex
              << req.prop << std::dec << ".";
        error.raise();
      }
      if (req.privilege_fields.empty() &&
          ((req.flags & LEGION_SUPPRESS_WARNINGS_FLAG) == 0))
      {
        Warning warning;
        warning << "Region requirement " << index << " of " << *this
                << " does not contain any privilege fields. "
                << "Did you forget them?";
        warning.raise();
      }
      // Check that the handle names are all sound
      if (!req.parent.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "The 'parent' region of region requirement " << index << " of "
              << *this
              << ") does not exist. The 'parent' region must always be set on "
                 "every region requirement.";
        error.raise();
      }
      if (!req.parent.valid())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "The 'parent' region of region requirement " << index << " of "
              << *this
              << " is not well-formed. This likely means it was corrupted by "
                 "application code.";
        error.raise();
      }
      if ((req.handle_type != LEGION_SINGULAR_PROJECTION) &&
          (req.handle_type != LEGION_REGION_PROJECTION) &&
          (req.handle_type != LEGION_PARTITION_PROJECTION))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid value of 'handle_type' " << req.handle_type
              << " for region requirement " << index << " of " << *this
              << ". The 'handle_type' of the region "
              << "requirement must be one of LEGION_SINGULAR_PROJECTION, "
              << "LEGION_REGION_PROJECTION, or LEGION_PARTITION_PROJECTION.";
        error.raise();
      }
      if (req.handle_type == LEGION_PARTITION_PROJECTION)
      {
        if (!req.partition.exists())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "The 'partition' of region requirement " << index << " of "
                << *this
                << " does not exist. The 'partition' must always be set when "
                << "'handle_type' is LEGION_SINGULAR_PROJECTION or "
                << "LEGION_REGION_PROJECTION.";
          error.raise();
        }
        if (!req.partition.valid())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "The 'partition' of region requirement " << index << " of "
                << *this
                << " is not well-formed. This likely means it was corrupted "
                << "by application code.";
          error.raise();
        }
        // Check that partition  is in the same region tree
        if (req.partition.get_tree_id() != req.parent.get_tree_id())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Partition " << req.partition
                << " is not from the same region tree (tree="
                << req.partition.get_tree_id()
                << ") as the 'parent' region (tree=" << req.parent.get_tree_id()
                << ") for region requirement " << index << " of " << *this
                << ". The partition for a projection region requirement "
                << "must always be from the same tree as the 'parent' region.";
          error.raise();
        }
        // Check to see if the partition is a below in parent in the tree
        if (!runtime->has_partition_path(
                req.parent.index_space, req.partition.index_partition))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Partition " << req.partition
                << " does not have parent region " << req.parent
                << " as an ancestor in the region tree for region requirement "
                << index << " of " << *this << "). The partition must always "
                << "have the 'parent' region as an ancestor for privileges.";
          error.raise();
        }
      }
      else
      {
        if (!req.region.exists())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "The 'region' of region requirement " << index << " of "
                << *this
                << " does not exist. The 'region' must always be set when "
                << "'handle_type' is LEGION_SINGULAR_PROJECTION "
                << "or LEGION_REGION_PROJECTION.";
          error.raise();
        }
        if (!req.region.valid())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "The 'region' of region requirement " << index << " of "
                << *this
                << " is not well-formed. This likely means it was corrupted "
                << "by application code.";
          error.raise();
        }
        // Check that the region is in the same region tree
        if (req.region.get_tree_id() != req.parent.get_tree_id())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Region " << req.region
                << " is not from the same region tree (tree="
                << req.region.get_tree_id()
                << ") as the 'parent' region (tree=" << req.parent.get_tree_id()
                << ") for region requirement " << index << " of " << *this
                << ". The region for a region requirement "
                << "must always be from the same tree as the 'parent' region.";
          error.raise();
        }
        // Check to see if the partition is a below in parent in the tree
        if (!runtime->has_index_path(
                req.parent.index_space, req.region.index_space))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Region " << req.region << " does not have parent region "
                << req.parent
                << " as an ancestor in the region tree for region requirement "
                << index << " of " << *this
                << ". The region must always have the "
                << "'parent' region as an ancestor for privileges.";
          error.raise();
        }
      }
      // Check the projection properties of the requirement
      if (req.handle_type != LEGION_SINGULAR_PROJECTION)
      {
        if (allow_projections)
        {
          ProjectionFunction* function = runtime->find_projection_function(
              req.projection, true /*can fail*/);
          if (function == nullptr)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unable to find projection function " << req.projection
                  << " for region requirement " << index << " of " << *this
                  << ". This means a projection function was not registered "
                  << "with that projection function ID.";
            error.raise();
          }
        }
        else
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Detected a projection region requirement for region "
                << "requirement " << index << " of " << *this << ". Projection "
                << "region requirements are not supported for this kind of "
                   "operation.";
          error.raise();
        }
      }
      // Check that all the fields are contained in the field space
      FieldSpaceNode* fs = runtime->get_node(req.parent.get_field_space());
      for (const FieldID& field_id : req.privilege_fields)
        if (!fs->has_field(field_id))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Field " << field_id
                << " in privilege fields of region requirement " << index
                << " of " << *this << " is not contained with field space "
                << fs->handle << " of the parent region requirement. All "
                << "privilege fields must be contained within the parent "
                << "region's field space.";
          error.raise();
        }
      // Check that the instance fields are unique and represented by privilege
      std::vector<FieldID> instance_fields(req.instance_fields);
      std::sort(instance_fields.begin(), instance_fields.end());
      for (unsigned idx = 0; idx < instance_fields.size(); idx++)
      {
        if ((idx > 0) && (instance_fields[idx - 1] == instance_fields[idx]))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "Duplicate field " << instance_fields[idx]
              << " found in the 'instance_fields' of region requirement "
              << index << " of " << *this
              << ". Each field in the 'privilege_fields' should be represented "
              << "exactly once in the 'instance_fields' of the region "
                 "requirement.";
          error.raise();
        }
        if (req.privilege_fields.find(instance_fields[idx]) ==
            req.privilege_fields.end())
        {
          const void* name = nullptr;
          size_t name_size = 0;
          if (fs->retrieve_semantic_information(
                  instance_fields[idx], LEGION_NAME_SEMANTIC_TAG, name,
                  name_size, true /*can fail*/, false /*wait until*/))
          {
            std::string_view field_name((const char*)name, name_size);
            Error error(LEGION_INTERFACE_EXCEPTION);
            error
                << "Field " << field_name
                << " in 'instance_fields' of region requirement " << index
                << " of " << *this
                << " is not contained in the 'privilege_fields'. "
                << "Each field in the 'instance_fields' must also be contained "
                   "in "
                << "the 'privilege_fields' of the region requirement.";
            error.raise();
          }
          else
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error
                << "Field " << instance_fields[idx]
                << " in 'instance_fields' of region requirement " << index
                << " of " << *this
                << " is not contained in the 'privilege_fields'. "
                << "Each field in the 'instance_fields' must also be contained "
                   "in the "
                << "'privilege_fields' of the region requirement.";
            error.raise();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Operation::report_interfering_requirements(
        unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
      // should only be called if overridden
      std::abort();
    }

    //--------------------------------------------------------------------------
    unsigned Operation::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool Operation::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return recorder.record_operation_untraceable(this, opidx);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::hash_requirement(
        Murmur3Hasher& hasher, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      if (req.region.exists())
      {
        hasher.hash<bool>(true);  // is_reg
        hasher.hash(req.region.get_index_space().get_id());
        hasher.hash(req.region.get_field_space().get_id());
        hasher.hash(req.region.get_tree_id());
      }
      else
      {
        hasher.hash<bool>(false);  // is_reg
        hasher.hash(req.partition.get_index_partition().get_id());
        hasher.hash(req.partition.get_field_space().get_id());
        hasher.hash(req.partition.get_tree_id());
      }
      for (const FieldID& field_id : req.privilege_fields)
        hasher.hash(field_id);
      for (const FieldID& field_id : req.instance_fields) hasher.hash(field_id);
      hasher.hash(req.privilege);
      hasher.hash(req.prop);
      hasher.hash(req.parent.get_index_space().get_id());
      hasher.hash(req.parent.get_field_space().get_id());
      hasher.hash(req.parent.get_tree_id());
      hasher.hash(req.redop);
      // Excluding the fields: tag and flags.
      hasher.hash(req.handle_type);
      hasher.hash(req.projection);
    }

    //--------------------------------------------------------------------------
    void Operation::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    size_t Operation::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return 1;
    }

    //--------------------------------------------------------------------------
    bool Operation::perform_collective_analysis(
        CollectiveMapping*& mapping, bool& first_local)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool Operation::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      // Should only be called in derived types
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::convert_collective_views(
        unsigned requirement_index, unsigned analysis_index,
        LogicalRegion region, const InstanceSet& targets,
        InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
        bool& first_local,
        op::vector<op::FieldMaskMap<InstanceView> >& target_views,
        std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      // Should only be called in derived types
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      // Should only be called in derived types
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::report_uninitialized_usage(
        const unsigned index, const char* field_string, RtUserEvent reported)
    //--------------------------------------------------------------------------
    {
      legion_assert(reported.exists());
      legion_assert(!reported.has_triggered());
      const RegionRequirement& req = get_requirement(index);
      // Read-only or reduction usage of uninitialized data is always an error
      if (IS_READ_ONLY(req))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Region requirement " << index << " of " << *this
              << " is using uninitialized data for field(s) " << field_string
              << " of logical region " << req.region
              << " with read-only privileges.";
        error.raise();
      }
      else if (IS_REDUCE(req))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Region requirement " << index << " of " << *this
              << " is using uninitialized data for field(s) " << field_string
              << " of logical region " << req.region
              << " with reduction privileges.";
        error.raise();
      }
      // Read-write usage is just a warning
      else if ((req.flags & LEGION_SUPPRESS_WARNINGS_FLAG) == 0)
      {
        Warning warning;
        warning << "WARNING: Region requirement " << index << " of " << *this
                << " is using uninitialized data for field(s) " << field_string
                << " of logical region " << req.region
                << " with read-write privileges. This is safe but you may be "
                << "accessing invalid data.";
        warning.raise();
      }
      Runtime::trigger_event(reported);
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* Operation::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      // should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::update_atomic_locks(
        const unsigned index, Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    /*static*/ ApEvent Operation::merge_sync_preconditions(
        const TraceInfo& trace_info, const std::vector<Grant>& grants,
        const std::vector<PhaseBarrier>& wait_barriers)
    //--------------------------------------------------------------------------
    {
      if (!grants.empty())
        std::abort();  // Figure out how to deduplicate grant acquires
      if (wait_barriers.empty())
        return ApEvent::NO_AP_EVENT;
      if (wait_barriers.size() == 1)
        return Runtime::get_previous_phase(wait_barriers[0].phase_barrier);
      std::set<ApEvent> wait_events;
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        wait_events.insert(
            Runtime::get_previous_phase(wait_barriers[idx].phase_barrier));
      return Runtime::merge_events(&trace_info, wait_events);
    }

    //--------------------------------------------------------------------------
    int Operation::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool Operation::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      // Should only be called for inherited types
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::compute_task_tree_coordinates(TaskTreeCoordinates& coords)
    //--------------------------------------------------------------------------
    {
      parent_ctx->compute_task_tree_coordinates(coords);
      coords.emplace_back(get_task_tree_coordinate());
    }

    //--------------------------------------------------------------------------
    ContextCoordinate Operation::get_task_tree_coordinate(void) const
    //--------------------------------------------------------------------------
    {
      return ContextCoordinate(context_index, DomainPoint());
    }

    //--------------------------------------------------------------------------
    void Operation::filter_copy_request_kinds(
        MapperManager* mapper, const std::set<ProfilingMeasurementID>& requests,
        std::vector<ProfilingMeasurementID>& results, bool warn_if_not_copy)
    //--------------------------------------------------------------------------
    {
      for (const ProfilingMeasurementID& measurement_id : requests)
      {
        switch ((Realm::ProfilingMeasurementID)measurement_id)
        {
          case Realm::PMID_OP_STATUS:
          case Realm::PMID_OP_STATUS_ABNORMAL:
          case Realm::PMID_OP_BACKTRACE:
          case Realm::PMID_OP_TIMELINE:
          case Realm::PMID_OP_TIMELINE_GPU:
          case Realm::PMID_OP_MEM_USAGE:
          case Realm::PMID_OP_COPY_INFO:
            {
              results.emplace_back(measurement_id);
              break;
            }
          default:
            {
              if (warn_if_not_copy)
              {
                Warning warning;
                warning << "Mapper " << *mapper
                        << "requested a profiling measurement of type "
                        << measurement_id
                        << "which is not applicable to operation " << *this
                        << "and therefore it will be ignored";
                warning.raise();
              }
            }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Operation::enqueue_ready_operation(
        RtEvent wait_on /*=Event::NO_EVENT*/,
        LgPriority priority /*= LG_THROUGHPUT_WORK_PRIORITY*/)
    //--------------------------------------------------------------------------
    {
      TriggerOpArgs args(this);
      runtime->issue_runtime_meta_task(args, priority, wait_on);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_mapping(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        parent_ctx->add_to_deferred_mapped_queue(this, wait_on);
        return;
      }
      ApEvent effects;
      bool trigger_now = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!mapped);
        mapped = true;
        if (mapped_event.exists())
          Runtime::trigger_event(mapped_event);
        // Notify all our mapping dependences, note we can do this while
        // holding the lock since notifying them doesn't involve taking
        // their locks
        for (const std::pair<Operation* const, GenerationID>& op_pair :
             outgoing)
          op_pair.first->satisfy_mapping_dependence();
        if (executed)
        {
          trigger_now = true;
          effects = compute_effects();
        }
      }
      // Do the trigger complete call if necessary
      if (trigger_now)
        trigger_complete(effects);
    }

    //--------------------------------------------------------------------------
    void Operation::complete_execution(RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        // We have to defer the execution of this operation
        parent_ctx->add_to_deferred_execution_queue(this, wait_on);
        return;
      }
      ApEvent effects;
      bool trigger_now = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!executed);
        executed = true;
        if (mapped)
        {
          trigger_now = true;
          effects = compute_effects();
        }
      }
      if (trigger_now)
        trigger_complete(effects);
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::compute_effects(void)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      ApEvent effects_done;
      if (!completion_effects.empty())
        effects_done = Runtime::merge_events(nullptr, completion_effects);
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        // Operations with regions and tasks do their own logging
        const OpKind op_kind = get_operation_kind();
        if ((op_kind != TASK_OP_KIND) && (op_kind != MAP_OP_KIND) &&
            (op_kind != ACQUIRE_OP_KIND) && (op_kind != RELEASE_OP_KIND) &&
            (op_kind != DEPENDENT_PARTITION_OP_KIND) &&
            (op_kind != ATTACH_OP_KIND) && (op_kind != DETACH_OP_KIND))
        {
          legion_assert(!completion_set);
          if (!completion_event.pending.exists())
            completion_event.pending = Runtime::create_ap_user_event(nullptr);
          LegionSpy::log_operation_events(
              unique_op_id, effects_done, completion_event.pending);
        }
      }
      return effects_done;
    }

    //--------------------------------------------------------------------------
    void Operation::complete_operation(ApEvent effects, bool first_invocation)
    //--------------------------------------------------------------------------
    {
      if (effects.exists() && first_invocation)
      {
        {
          AutoLock o_lock(op_lock);
          legion_assert(!completion_set);
          if (completion_event.pending.exists())
          {
            ApUserEvent to_trigger = completion_event.pending;
            Runtime::trigger_event_untraced(to_trigger, effects);
            completion_event.effects = to_trigger;
          }
          else
            completion_event.effects = effects;
          completion_set = true;
        }
        parent_ctx->add_to_deferred_completion_queue(
            this, effects, track_parent);
        return;
      }
      bool do_commit = false;
      std::vector<Operation*> to_notify;
      {
        AutoLock o_lock(op_lock);
        legion_assert(mapped);
        legion_assert(executed);
        legion_assert(!completed);
        completed = true;
        if (!completion_set)
        {
          if (completion_event.pending.exists())
          {
            ApUserEvent to_trigger = completion_event.pending;
            Runtime::trigger_event_untraced(to_trigger, effects);
            completion_event.effects = to_trigger;
          }
          else
            completion_event.effects = effects;
          completion_set = true;
        }
        // Check to see if we need to trigger commit
        if (runtime->resilient_mode)
        {
          // Always do any verification notifications once we are
          // actually complete and will not raise any region exceptions
          to_notify.insert(
              to_notify.end(), verification_notifications.begin(),
              verification_notifications.end());
          if ((outstanding_mapping_references == 0) &&
              (hardened_notifications == outgoing.size()))
          {
            do_commit = true;
            if (hardened)
            {
              // If we're a hardened operation and complete then we can
              // notify our upstream operations that they too are hardened
              // but skip any operations that are themselves hardened as
              // they would already be included from the verification users
              for (const std::pair<Operation* const, GenerationID>& op_pair :
                   incoming)
                if (verification_notifications.find(op_pair.first) ==
                    verification_notifications.end())
                  to_notify.emplace_back(op_pair.first);
            }
          }
        }
        else
          do_commit = true;
      }
      // finally notify all the operations we dependended on
      // that we validated their regions note we don't need
      // the lock since this was all set when we did our mapping analysis
      for (Operation* const & operation : to_notify)
        operation->notify_hardened();
      if (do_commit)
      {
        if (track_parent)
          parent_ctx->register_child_complete(this);
        // Remove the guard reference for commit
        satisfy_commit_dependence();
      }
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::get_completion_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (!completion_set)
      {
        if (!completion_event.pending.exists())
          completion_event.pending = Runtime::create_ap_user_event(nullptr);
        return completion_event.pending;
      }
      else
        return completion_event.effects;
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::get_mapped_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (mapped)
        return RtEvent::NO_RT_EVENT;
      if (!mapped_event.exists())
        mapped_event = Runtime::create_rt_user_event();
      return mapped_event;
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::get_commit_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(!committed);
      if (!commit_event.exists())
        commit_event = Runtime::create_rt_user_event();
      return commit_event;
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::get_commit_event(GenerationID g)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(g <= gen);
      if ((g < gen) || committed)
        return RtEvent::NO_RT_EVENT;
      if (!commit_event.exists())
        commit_event = Runtime::create_rt_user_event();
      return commit_event;
    }

    //--------------------------------------------------------------------------
    void Operation::record_completion_effect(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      AutoLock o_lock(op_lock);
      legion_assert(!mapped || !executed);
      completion_effects.insert(effect);
    }

    //--------------------------------------------------------------------------
    void Operation::record_completion_effect(
        ApEvent effect, std::set<RtEvent>& map_applied_events)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      AutoLock o_lock(op_lock);
      legion_assert(!mapped || !executed);
      completion_effects.insert(effect);
    }

    //--------------------------------------------------------------------------
    void Operation::record_completion_effects(const std::set<ApEvent>& effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      AutoLock o_lock(op_lock);
      legion_assert(!mapped || !executed);
      for (const ApEvent& event : effects)
        if (event.exists())
          completion_effects.insert(event);
    }

    //--------------------------------------------------------------------------
    void Operation::record_completion_effects(
        const std::vector<ApEvent>& effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      AutoLock o_lock(op_lock);
      legion_assert(!mapped || !executed);
      for (const ApEvent& event : effects)
        if (event.exists())
          completion_effects.insert(event);
    }

    //--------------------------------------------------------------------------
    void Operation::forward_completion_effects(Operation* target)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock, false /*exclusive*/);
      if (!completion_effects.empty())
        target->record_completion_effects(completion_effects);
    }

    //--------------------------------------------------------------------------
    void Operation::commit_operation(
        bool do_deactivate, RtEvent wait_on /*= Event::NO_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (wait_on.exists() && !wait_on.has_triggered())
      {
        parent_ctx->add_to_deferred_commit_queue(this, wait_on, do_deactivate);
        return;
      }
      if (track_parent)
        parent_ctx->register_child_commit(this);
      // Mark that we are committed
      {
        AutoLock o_lock(op_lock);
        legion_assert(mapped);
        legion_assert(executed);
        legion_assert(completed);
        legion_assert(!committed);
        committed = true;
        // At this point we bumb the generation as we can never roll back
        // after we have committed the operation
        gen++;
      }
      // Signal all of our dependent operations that they have a commit
      // reference satisfied before anything happends that can clean up
      for (std::map<Operation*, GenerationID>::const_iterator it =
               outgoing.begin();
           it != outgoing.end(); it++)
        it->first->satisfy_commit_dependence();
      // Trigger the commit event
      if (commit_event.exists())
      {
        Runtime::trigger_event(commit_event);
        commit_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (do_deactivate)
        deactivate();
    }

    //--------------------------------------------------------------------------
    void Operation::quash_operation(GenerationID gen, bool restart)
    //--------------------------------------------------------------------------
    {
      // TODO: actually handle quashing of operations
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::begin_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Should have a guard reference here already
      legion_assert(remaining_mapping_dependences.load() == 1);
      // Register ourselves with our trace if there is one
      // This will also add any necessary dependences
      if ((trace != nullptr) && !is_tracing_fence())
        trace_local_id = trace->register_operation(this, gen);
      if (tracing)
      {
        // Temporarily disable tracing while recording implicit dependences
        tracing = false;
        parent_ctx->register_implicit_dependences(this);
        tracing = true;
      }
      else  // See if we have any fence dependences
        parent_ctx->register_implicit_dependences(this);
    }

    //--------------------------------------------------------------------------
    void Operation::end_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Remove the guard that we added previously
      satisfy_mapping_dependence();
    }

    //--------------------------------------------------------------------------
    bool Operation::register_dependence(
        Operation* target, GenerationID target_gen)
    //--------------------------------------------------------------------------
    {
      if (must_epoch != nullptr)
        must_epoch->verify_dependence(this, gen, target, target_gen);
      if (tracing)
      {
        legion_assert(trace != nullptr);
        // If we're tracing check to see if the target is even in the
        // trace, if it's not then there's no need to record the dependence
        // because it will be handled by the mapping fence at the that
        // was issued at the beginning of the trace
        if (!trace->record_dependence(target, target_gen, this, gen))
          return true;
      }
      // The rest of this method is the same as the one below
      if (target == this)
      {
        legion_assert(target_gen < gen);
        // Can prune it if we're not tracing
        return !tracing;
      }
      bool registered_dependence = false;
      const bool prune =
          target->perform_registration(
              target_gen, this, gen, registered_dependence,
              remaining_mapping_dependences, remaining_commit_dependences,
              verification_notifications) &&
          !tracing;
      if (registered_dependence)
        incoming[target] = target_gen;
      return prune;
    }

    //--------------------------------------------------------------------------
    bool Operation::register_region_dependence(
        unsigned idx, Operation* target, GenerationID target_gen,
        unsigned target_idx, DependenceType dtype,
        const FieldMask& dependent_mask)
    //--------------------------------------------------------------------------
    {
      bool do_registration = true;
      if (must_epoch != nullptr)
        do_registration = must_epoch->record_dependence(
            this, gen, target, target_gen, idx, target_idx, dtype);
      if (tracing)
      {
        legion_assert(trace != nullptr);
        // If we're tracing check to see if the target is even in the
        // trace, if it's not then there's no need to record the dependence
        // because it will be handled by the mapping fence at the that
        // was issued at the beginning of the trace
        if (!trace->record_region_dependence(
                target, target_gen, this, gen, target_idx, idx, dtype,
                dependent_mask))
          return true;
      }
      // Can never register a dependence on ourself since it means
      // that the target was recycled and will never register. Return
      // true if the generation is older than our current generation.
      if (target == this)
      {
        if (target_gen == gen)
          report_interfering_requirements(target_idx, idx);
        // Can prune it if we're not tracing
        return !tracing;
      }
      bool prune = false;
      bool registered_dependence = false;
      if (do_registration)
      {
        prune = target->perform_registration(
                    target_gen, this, gen, registered_dependence,
                    remaining_mapping_dependences, remaining_commit_dependences,
                    verification_notifications) &&
                !tracing;
      }
      if (registered_dependence)
        incoming[target] = target_gen;
      return prune;
    }

    //--------------------------------------------------------------------------
    bool Operation::perform_registration(
        GenerationID our_gen, Operation* op, GenerationID op_gen,
        bool& registered_dependence, std::atomic<unsigned>& mapping_dependences,
        std::atomic<unsigned>& commit_dependences,
        std::set<Operation*>& notifications)
    //--------------------------------------------------------------------------
    {
      legion_assert(our_gen <= gen);  // better not be ahead of where we are now
      // If the generations match and we haven't committed yet,
      // register an outgoing dependence
      if (our_gen == gen)
      {
        AutoLock o_lock(op_lock);
        // Retest generation to see if we lost the race
        if (our_gen == gen)
        {
          // should still have some mapping references
          // if other operations are trying to register dependences
          // This assertion no longer holds because of how we record
          // fence dependences from context operation lists which
          // don't track mapping dependences
          // legion_assert(outstanding_mapping_references > 0);
          // Check to see if we've already recorded this dependence
          std::map<Operation*, GenerationID>::const_iterator finder =
              outgoing.find(op);
          if ((finder == outgoing.end()) || (finder->second != op_gen))
          {
            outgoing[op] = op_gen;
            if (!mapped)
              mapping_dependences.fetch_add(1);
            commit_dependences.fetch_add(1);
            // If we're a hardened operation then have the operation
            // tell us when it is complete so we know our data is good
            if (hardened)
              notifications.insert(this);
            registered_dependence = true;
          }
          else
          {
            // We already registered it
            registered_dependence = false;
          }
          // Cannot prune this operation from the list since it
          // is still not committed
          return false;
        }
      }
      // We already committed so we're done and this
      // operation can be pruned from the list of users
      registered_dependence = false;
      return true;
    }

    //--------------------------------------------------------------------------
    void Operation::satisfy_mapping_dependence(void)
    //--------------------------------------------------------------------------
    {
      const unsigned remaining = remaining_mapping_dependences.fetch_sub(1);
      legion_assert(remaining > 0);
      if (remaining == 1)
      {
        if (must_epoch == nullptr)
          parent_ctx->add_to_ready_queue(this);
        else
          must_epoch->satisfy_mapping_dependence();
      }
    }

    //--------------------------------------------------------------------------
    void Operation::satisfy_commit_dependence(void)
    //--------------------------------------------------------------------------
    {
      const unsigned remaining = remaining_commit_dependences.fetch_sub(1);
      legion_assert(remaining > 0);
      if (remaining == 1)
      {
        if (track_parent)
          parent_ctx->add_to_trigger_commit_queue(this);
        else
          trigger_commit();
      }
    }

    //--------------------------------------------------------------------------
    bool Operation::is_operation_committed(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      // If we're on an old generation then it's definitely committed
      return (our_gen < gen);
    }

    //--------------------------------------------------------------------------
    bool Operation::add_mapping_reference(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(our_gen <= gen);  // better not be ahead of where we are now
      if (our_gen < gen)
        return false;
      outstanding_mapping_references++;
      return true;
    }

    //--------------------------------------------------------------------------
    void Operation::remove_mapping_reference(GenerationID our_gen)
    //--------------------------------------------------------------------------
    {
      bool do_commit = false;
      std::vector<Operation*> to_notify;
      {
        AutoLock o_lock(op_lock);
        legion_assert(
            our_gen <= gen);  // better not be ahead of where we are now
        if ((our_gen == gen) && !committed)
        {
          legion_assert(outstanding_mapping_references > 0);
          outstanding_mapping_references--;
          if (runtime->resilient_mode &&
              (outstanding_mapping_references == 0) &&
              (hardened_notifications == outgoing.size()))
          {
            // If we're hardened we notify all upstream operations that
            // we're officially hardened, if we're not hardened we only
            // notify the upstream ones that are not themselves hardened
            for (const std::pair<Operation* const, GenerationID>& op_pair :
                 incoming)
              if (hardened || (verification_notifications.find(op_pair.first) ==
                               verification_notifications.end()))
                to_notify.emplace_back(op_pair.first);
            if (completed)
              do_commit = true;
          }
        }
        // otherwise we were already recycled and are no longer valid
      }
      // finally notify all the operations we dependended on
      // that we validated their regions note we don't need
      // the lock since this was all set when we did our mapping analysis
      for (Operation* const & operation : to_notify)
        operation->notify_hardened();
      if (do_commit)
      {
        if (track_parent)
          parent_ctx->register_child_complete(this);
        // Remove the guard reference for commit
        satisfy_commit_dependence();
      }
    }

    //--------------------------------------------------------------------------
    void Operation::notify_hardened(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(runtime->resilient_mode);
      bool do_commit = false;
      std::vector<Operation*> to_notify;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!committed);
        legion_assert(hardened_notifications < outgoing.size());
        hardened_notifications++;
        if ((outstanding_mapping_references == 0) &&
            (hardened_notifications == outgoing.size()))
        {
          // If we're hardened we notify all upstream operations that
          // we're officially hardened, if we're not hardened we only
          // notify the upstream ones that are not themselves hardened
          for (const std::pair<Operation* const, GenerationID>& op_pair :
               incoming)
            if (hardened || (verification_notifications.find(op_pair.first) ==
                             verification_notifications.end()))
              to_notify.emplace_back(op_pair.first);
          if (completed)
            do_commit = true;
        }
      }
      for (Operation* const & operation : to_notify)
        operation->notify_hardened();
      if (do_commit)
      {
        if (track_parent)
          parent_ctx->register_child_complete(this);
        // Remove the guard reference for commit
        satisfy_commit_dependence();
      }
    }

    //--------------------------------------------------------------------------
    InnerContext* Operation::find_physical_context(unsigned index)
    //--------------------------------------------------------------------------
    {
      return parent_ctx->find_parent_physical_context(find_parent_index(index));
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(
        PhysicalManager* manager, MappingInstance& instance)
    //--------------------------------------------------------------------------
    {
      instance = MappingInstance(manager);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(
        const std::vector<InstanceView*>& views,
        std::vector<MappingInstance>& input_valid,
        std::vector<MappingCollective>& collectives)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < views.size(); idx++)
      {
        if (views[idx]->is_individual_view())
        {
          IndividualView* view = views[idx]->as_individual_view();
          input_valid.emplace_back(MappingInstance(view->get_manager()));
        }
        else
        {
          CollectiveView* view = views[idx]->as_collective_view();
          collectives.emplace_back(MappingCollective(view));
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(
        const InstanceSet& valid,
        const local::FieldMaskMap<ReplicatedView>& collectives,
        std::vector<MappingInstance>& input_valid,
        std::vector<MappingCollective>& collectives_valid)
    //--------------------------------------------------------------------------
    {
      if (!valid.empty())
      {
        unsigned offset = input_valid.size();
        input_valid.resize(offset + valid.size());
        for (unsigned idx = 0; idx < valid.size(); idx++)
        {
          const InstanceRef& ref = valid[idx];
          legion_assert(!ref.is_virtual_ref());
          MappingInstance& inst = input_valid[offset + idx];
          inst = ref.get_mapping_instance();
        }
      }
      if (!collectives.empty())
      {
        collectives_valid.reserve(
            collectives_valid.size() + collectives.size());
        for (const std::pair<ReplicatedView* const, FieldMask>&
                 collective_pair : collectives)
          collectives_valid.emplace_back(
              MappingCollective(collective_pair.first));
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::prepare_for_mapping(
        const InstanceSet& valid,
        const local::FieldMaskMap<ReplicatedView>& collectives,
        const std::set<Memory>& visible_filter,
        std::vector<MappingInstance>& input_valid,
        std::vector<MappingCollective>& collectives_valid)
    //--------------------------------------------------------------------------
    {
      if (!valid.empty())
      {
        unsigned offset = input_valid.size();
        input_valid.reserve(offset + valid.size());
        unsigned next_index = offset;
        for (unsigned idx = 0; idx < valid.size(); idx++)
        {
          const InstanceRef& ref = valid[idx];
          PhysicalManager* manager = ref.get_physical_manager();
          if (!manager->has_visible_from(visible_filter))
            continue;
          input_valid.resize(next_index + 1);
          MappingInstance& inst = input_valid[next_index++];
          inst = ref.get_mapping_instance();
        }
      }
      if (!collectives.empty())
      {
        collectives_valid.reserve(
            collectives_valid.size() + collectives.size());
        for (const std::pair<ReplicatedView* const, FieldMask>&
                 collective_pair : collectives)
          collectives_valid.emplace_back(
              MappingCollective(collective_pair.first));
      }
    }

    //--------------------------------------------------------------------------
    void Operation::compute_ranking(
        MapperManager* mapper, const std::deque<MappingInstance>& output,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& collective_insts) const
    //--------------------------------------------------------------------------
    {
      ranking.reserve(output.size());
      std::vector<bool> unique_sources(sources.size(), false);
      for (const MappingInstance& mapping_instance : output)
      {
        const InstanceManager* man = mapping_instance.impl;
        if (!man->is_physical_manager())
          continue;
        PhysicalManager* manager = man->as_physical_manager();
        bool found = false;
        bool has_collectives = false;
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          if (!sources[idx]->is_individual_view())
          {
            has_collectives = true;
            continue;
          }
          IndividualView* src = sources[idx]->as_individual_view();
          if (src->get_manager() == manager)
          {
            found = true;
            if (unique_sources[idx])
            {
              Warning warning;
              warning
                  << "Detected duplicate entries of physical instance "
                  << output[idx] << " in select sources call for " << *this
                  << " by mapper " << *mapper << ". This may be indicative of "
                  << "a mapper bug. Legion will ignore later entries of this "
                  << "instance in the ranking.";
              warning.raise();
            }
            else
            {
              ranking.emplace_back(idx);
              unique_sources[idx] = true;
            }
            break;
          }
        }
        if (!found && has_collectives)
        {
          for (unsigned idx = 0; idx < sources.size(); idx++)
          {
            if (!sources[idx]->is_collective_view())
              continue;
            CollectiveView* src = sources[idx]->as_collective_view();
            if (src->contains(manager))
            {
              found = true;
              // Only need to save the first instance from each collective
              if (unique_sources[idx])
              {
                Warning warning;
                warning
                    << "Detected duplicate points of collective view "
                    << output[idx] << " in select sources call for " << *this
                    << " by mapper " << *mapper << ". This may be indicative "
                    << "of a mapper bug. Legion will ignore later entries of "
                    << "this instance in the ranking.";
                warning.raise();
              }
              else
              {
                unique_sources[idx] = true;
                ranking.emplace_back(idx);
                collective_insts[idx] = manager;
                break;
              }
            }
          }
        }
        // Ignore any instances which are not in the original set of sources
        if (!found)
        {
          Warning warning;
          warning << "Ignoring invalid instance output from mapper " << *mapper
                  << " by select source call for " << *this << ".";
          warning.raise();
        }
      }
      legion_assert(ranking.size() <= sources.size());
      // Fill in any sources not covered by the mapper
      if (ranking.size() < sources.size())
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
          if (!unique_sources[idx])
            ranking.push_back(idx);
      }
      legion_assert(ranking.size() == sources.size());
    }

    //--------------------------------------------------------------------------
    void Operation::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      // should only be called on derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::pack_local_remote_operation(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      legion_assert(parent_ctx != nullptr);
      rez.serialize(get_operation_kind());
      rez.serialize(this);
      rez.serialize(runtime->address_space);
      rez.serialize(unique_op_id);
      parent_ctx->pack_inner_context(rez);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      rez.serialize<bool>(tracing);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Operation::add_launch_space_reference(IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      node->add_base_valid_ref(CONTEXT_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ bool Operation::remove_launch_space_reference(
        IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      return (node != nullptr) && node->remove_base_valid_ref(CONTEXT_REF);
    }

    //--------------------------------------------------------------------------
    bool Operation::is_pointwise_analyzable(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void Operation::register_pointwise_dependence(
        unsigned idx, const LogicalUser& previous)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::replay_pointwise_dependences(
        std::map<unsigned, std::vector<PointwiseDependence> >& dependences)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::find_pointwise_dependence(
        const DomainPoint& point, GenerationID gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_index)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void Operation::log_launch_space(IndexSpace handle) const
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = runtime->get_node(handle);
      node->log_launch_space(unique_op_id);
    }

    //--------------------------------------------------------------------------
    void Operation::perform_dependence_analysis(
        unsigned idx, const RegionRequirement& req,
        const ProjectionInfo& proj_info, LogicalAnalysis& logical_analysis)
    //--------------------------------------------------------------------------
    {
      // If this is a NO_ACCESS, then we'll have no dependences so we're done
      if (IS_NO_ACCESS(req))
        return;

      ProjectionType htype = req.handle_type;
      bool is_ispace_htype =
          (htype == LEGION_SINGULAR_PROJECTION ||
           htype == LEGION_REGION_PROJECTION);
      legion_assert(is_ispace_htype || htype == LEGION_PARTITION_PROJECTION);
      IndexTreeNode* child_node =
          is_ispace_htype ? runtime->get_node(req.region.get_index_space()) :
                            (IndexTreeNode*)runtime->get_node(
                                req.partition.get_index_partition());

      RegionNode* parent_node = runtime->get_node(req.parent);

      RegionTreePath path;
      legion_assert(child_node->depth >= parent_node->row_source->depth);
      path.initialize(parent_node->row_source->depth, child_node->depth);
      while (child_node != parent_node->row_source)
      {
        legion_assert(child_node->depth > 0);
        path.register_child(child_node->depth - 1, child_node->color);
        child_node = child_node->get_parent();
      }

      FieldMask user_mask =
          parent_node->column_source->get_field_mask(req.privilege_fields);
      LogicalTraceInfo trace_info(this, idx, req, user_mask);
      if (trace_info.skip_analysis)
        return;
      // Then compute the logical user
      ProjectionSummary* shard_proj = nullptr;
      if (proj_info.is_projecting())
      {
        if (runtime->enable_pointwise_analysis)
        {
          RegionTreeNode* destination =
              (req.handle_type == LEGION_PARTITION_PROJECTION) ?
                  static_cast<RegionTreeNode*>(
                      runtime->get_node(req.partition)) :
                  static_cast<RegionTreeNode*>(runtime->get_node(req.region));
          shard_proj = destination->compute_projection_summary(
              this, idx, req, logical_analysis, proj_info);
        }
        else
        {
          if (proj_info.is_sharding())
          {
            // If we're doing a projection in a control replicated context then
            // we need to compute the shard projection up front since it might
            // involve a collective if we don't hit in the cache and we want
            // that to appear nice and deterministic
            RegionTreeNode* destination =
                (req.handle_type == LEGION_PARTITION_PROJECTION) ?
                    static_cast<RegionTreeNode*>(
                        runtime->get_node(req.partition)) :
                    static_cast<RegionTreeNode*>(runtime->get_node(req.region));
            shard_proj = destination->compute_projection_summary(
                this, idx, req, logical_analysis, proj_info);
          }
        }
      }

      LogicalUser* user = new LogicalUser(
          this, idx, RegionUsage(req), shard_proj,
          (get_must_epoch_op() == nullptr) ?
              std::numeric_limits<unsigned>::max() :
              get_must_epoch_op()->find_operation_index(
                  this, get_generation()));
      user->add_reference();
      // Finally do the traversal, note that we don't need to hold the
      // context lock since the runtime guarantees that all dependence
      // analysis for a single context are performed in order
      {
        FieldMask unopened_mask = user_mask;
        FieldMask refinement_mask;
        // We disallow refinements for operations that are part of
        // a must epoch launch because refinements are too hard to
        // implement correctly in that case
        // We also don't try to update refinements if we're doing a reset
        // operation since that is an internal kind of operation
        if ((get_must_epoch_op() == nullptr) &&
            (get_operation_kind() != RESET_OP_KIND))
          refinement_mask = user_mask;
        FieldMaskMap<
            RefinementOp, TASK_LOCAL_LIFETIME,
            LogicalAnalysis::RefinementComparator>
            refinements;
        parent_node->register_logical_user(
            req.parent, *user, path, trace_info, proj_info, user_mask,
            unopened_mask, refinement_mask, logical_analysis, refinements);
      }
      if (user->remove_reference())
        delete user;
    }

    //--------------------------------------------------------------------------
    void Operation::perform_versioning_analysis(
        unsigned index, const RegionRequirement& req, VersionInfo& version_info,
        std::set<RtEvent>& ready_events, RtEvent* output_region_ready,
        bool collective_rendezvous)
    //--------------------------------------------------------------------------
    {
      if (IS_NO_ACCESS(req))
        return;
      InnerContext* context = find_physical_context(index);
      ContextID ctx = context->get_physical_tree_context();
      legion_assert(
          (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
          ((req.handle_type == LEGION_REGION_PROJECTION) &&
           (req.projection == 0)));
      RegionNode* region_node = runtime->get_node(req.region);
      FieldMask user_mask =
          region_node->column_source->get_field_mask(req.privilege_fields);
      region_node->perform_versioning_analysis(
          ctx, parent_ctx, &version_info, user_mask, this, index,
          find_parent_index(index), ready_events, output_region_ready,
          collective_rendezvous);
    }

    //--------------------------------------------------------------------------
    void Operation::physical_premap_region(
        unsigned index, RegionRequirement& req, const VersionInfo& version_info,
        InstanceSet& targets, local::FieldMaskMap<ReplicatedView>& collectives,
        std::set<RtEvent>& map_applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
          (req.handle_type == LEGION_REGION_PROJECTION));
      // If we are a NO_ACCESS or there are no fields then we are already done
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return;
      // Iterate over the equivalence sets and get all the instances that
      // are valid for all the different equivalence classes
      IndexSpaceNode* expr_node =
          runtime->get_node(req.region.get_index_space());
      ValidInstAnalysis analysis(
          this, index, expr_node, IS_REDUCE(req) ? req.redop : 0);
      const RtEvent traversal_done = analysis.perform_traversal(
          RtEvent::NO_RT_EVENT, version_info, map_applied_events);
      RtEvent ready;
      if (traversal_done.exists() || analysis.has_remote_sets())
        ready = analysis.perform_remote(traversal_done, map_applied_events);
      // Wait for all the responses to be ready
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      op::FieldMaskMap<LogicalView> instances;
      if (analysis.report_instances(instances))
        req.flags |= LEGION_RESTRICTED_FLAG;
      const std::vector<LogicalRegion> to_meet(1, req.region);
      for (const std::pair<LogicalView* const, FieldMask>& instance_pair :
           instances)
      {
        legion_assert(instance_pair.first->is_instance_view());
        if (instance_pair.first->is_individual_view())
        {
          IndividualView* view = instance_pair.first->as_individual_view();
          PhysicalManager* manager = view->get_manager();
          if (manager->meets_regions(to_meet))
            targets.add_instance(InstanceRef(manager, instance_pair.second));
        }
        else
        {
          legion_assert(instance_pair.first->is_replicated_view());
          ReplicatedView* view = instance_pair.first->as_replicated_view();
          if (view->meets_regions(to_meet))
            collectives.insert(view, instance_pair.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent Operation::physical_perform_updates(
        const RegionRequirement& req, const VersionInfo& version_info,
        unsigned index, ApEvent precondition, ApEvent term_event,
        const InstanceSet& targets,
        const std::vector<PhysicalManager*>& sources,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events, UpdateAnalysis*& analysis,
        const bool collective_rendezvous, const bool record_valid,
        const bool check_initialized, const bool defer_copies)
    //--------------------------------------------------------------------------
    {
      // If we are a NO_ACCESS or there are no fields then we are already done
      if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
        return RtEvent::NO_RT_EVENT;
      legion_assert(
          (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
          (req.handle_type == LEGION_REGION_PROJECTION));
      legion_assert(!targets.empty());
      legion_assert(!targets.is_virtual_mapping());
      RegionNode* region_node = runtime->get_node(req.region);
      const FieldMask user_mask =
          region_node->column_source->get_field_mask(req.privilege_fields);
      // Perform the registration
      legion_assert(analysis == nullptr);
      // Should be recording or must be read-only
      legion_assert(record_valid || IS_READ_ONLY(req));
      analysis = new UpdateAnalysis(
          this, index, req, region_node, trace_info, precondition, term_event,
          check_initialized, record_valid);
      analysis->add_reference();
      const RtEvent views_ready = analysis->convert_views(
          req.region, targets, &sources, &analysis->usage,
          collective_rendezvous);
      const RtEvent traversal_done = analysis->perform_traversal(
          views_ready, version_info, map_applied_events);
      // Send out any remote updates
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready =
            analysis->perform_remote(traversal_done, map_applied_events);
      // Issue any release copies/fills that need to be done
      const RtEvent updates_done =
          analysis->perform_updates(traversal_done, map_applied_events);
      if (remote_ready.exists())
      {
        if (updates_done.exists())
          return Runtime::merge_events(remote_ready, updates_done);
        else
          return remote_ready;
      }
      else
        return updates_done;
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::physical_perform_registration(
        RtEvent precondition, UpdateAnalysis* analysis,
        std::set<RtEvent>& map_applied_events, bool symbolic)
    //--------------------------------------------------------------------------
    {
      // If we are a NO_ACCESS or there are no fields then analysis will be
      // nullptr
      if (analysis == nullptr)
        return ApEvent::NO_AP_EVENT;
      ApEvent instances_ready;
      const RtEvent registered = analysis->perform_registration(
          precondition, analysis->usage, map_applied_events,
          analysis->precondition, analysis->term_event, instances_ready,
          symbolic);
      // Perform any output copies (e.g. for restriction) that need to be done
      if (registered.exists() || analysis->has_output_updates())
        analysis->perform_output(registered, map_applied_events);
      // Remove the reference that we added in the updates step
      if (analysis->remove_reference())
        delete analysis;
      return instances_ready;
    }

    //--------------------------------------------------------------------------
    ApEvent Operation::physical_perform_updates_and_registration(
        const RegionRequirement& req, const VersionInfo& version_info,
        unsigned index, ApEvent precondition, ApEvent term_event,
        const InstanceSet& targets, const std::vector<PhysicalManager*>& src,
        const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& map_applied_events, const bool collective_rendezvous,
        const bool record_valid, const bool check_initialized)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      // These are some basic sanity checks that each field is represented
      // by exactly one instance and that the total number of fields
      // represented matches the number of privilege fields.
      // There has been at least one case where this invariant was violated
      // for attach operations and there were more fields represented in
      // instances than there were privileges, see the attach_2d example.
      FieldMask check_mask;
      for (unsigned idx = 0; idx < targets.size(); idx++)
      {
        const FieldMask& mask = targets[idx].get_valid_fields();
        legion_assert(check_mask * mask);
        check_mask |= mask;
      }
      legion_assert(check_mask.pop_count() == req.privilege_fields.size());
#endif
      UpdateAnalysis* analysis = nullptr;
      const RtEvent registration_precondition = physical_perform_updates(
          req, version_info, index, precondition, term_event, targets, src,
          trace_info, map_applied_events, analysis, collective_rendezvous,
          record_valid, check_initialized, false /*defer copies*/);
      return physical_perform_registration(
          registration_precondition, analysis, map_applied_events);
    }

    //--------------------------------------------------------------------------
    void Operation::physical_convert_sources(
        const RegionRequirement& req,
        const std::vector<MappingInstance>& sources,
        std::vector<PhysicalManager*>& result,
        std::map<PhysicalManager*, unsigned>* acquired)
    //--------------------------------------------------------------------------
    {
      const RegionTreeID req_tid = req.parent.get_tree_id();
      std::vector<PhysicalManager*> unacquired;
      for (const MappingInstance& instance : sources)
      {
        InstanceManager* man = instance.impl;
        if (man == nullptr)
          continue;
        if (man->is_virtual_manager())
          continue;
        PhysicalManager* manager = man->as_physical_manager();
        // Check to see if the region trees are the same
        if (req_tid != manager->tree_id)
          continue;
        if ((acquired != nullptr) &&
            (acquired->find(manager) == acquired->end()))
          unacquired.emplace_back(manager);
        result.emplace_back(manager);
      }
      if (!unacquired.empty())
      {
        perform_missing_acquires(*acquired, unacquired);
        unsigned unacquired_index = 0;
        for (std::vector<PhysicalManager*>::iterator it = result.begin();
             it != result.end();
             /*nothing*/)
        {
          if ((*it) == unacquired[unacquired_index])
          {
            if (acquired->find(unacquired[unacquired_index]) == acquired->end())
              it = result.erase(it);
            else
              it++;
            if ((++unacquired_index) == unacquired.size())
              break;
          }
          else
            it++;
        }
      }
    }

    //--------------------------------------------------------------------------
    int Operation::physical_convert_mapping(
        const RegionRequirement& req, std::vector<MappingInstance>& chosen,
        InstanceSet& result, RegionTreeID& bad_tree,
        std::vector<FieldID>& missing_fields,
        std::map<PhysicalManager*, unsigned>* acquired,
        std::vector<PhysicalManager*>& unacquired, const bool do_acquire_checks,
        const bool allow_partial_virtual)
    //--------------------------------------------------------------------------
    {
      // Can be a part projection if we are closing to a partition node
      FieldSpaceNode* node = runtime->get_node(req.parent.get_field_space());
      // Get the field mask for the fields we need
      FieldMask needed_fields = node->get_field_mask(req.privilege_fields);
      const RegionTreeID req_tid = req.parent.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_virtual = false;
      // If we're doing safe mapping, then sort these in order for determinism
      if (runtime->safe_mapper)
        std::sort(chosen.begin(), chosen.end());
      for (const MappingInstance& instance : chosen)
      {
        InstanceManager* man = instance.impl;
        if (man == nullptr)
          continue;
        if (man->is_virtual_manager())
        {
          has_virtual = true;
          continue;
        }
        PhysicalManager* manager = man->as_physical_manager();
        // Check to see if the region trees are the same
        if (req_tid != manager->tree_id)
        {
          bad_tree = manager->tree_id;
          return -1;
        }
        // See if we should be checking the acquired sets
        if (do_acquire_checks && (acquired->find(manager) == acquired->end()))
          unacquired.emplace_back(manager);
        // See which fields need to be made valid here
        FieldMask valid_fields =
            manager->layout->allocated_fields & needed_fields;
        if (!valid_fields)
          continue;
        result.add_instance(InstanceRef(manager, valid_fields));
        // We can remove the update fields from the needed mask since
        // we now have space for them
        needed_fields -= valid_fields;
        // If we've seen all our needed fields then we are done
        if (!needed_fields)
          break;
      }
      // If we don't have needed fields, see if we had a composite instance
      // if we did, put all the fields in there, otherwise we put report
      // them as missing fields figure out what field IDs they are
      if (!!needed_fields)
      {
        if (has_virtual)
        {
          if (!allow_partial_virtual)
          {
            // If we don't allow partial virtual results then clear
            // the results and make all the needed fields the result
            result.clear();
            needed_fields = node->get_field_mask(req.privilege_fields);
          }
          int composite_idx = result.size();
          result.add_instance(
              InstanceRef(runtime->virtual_manager, needed_fields));
          return composite_idx;
        }
        else
        {
          // This can be slow because if we get here we are just
          // going to be reporting an error so performance no
          // longer matters
          std::set<FieldID> missing;
          node->get_field_set(needed_fields, parent_ctx, missing);
          missing_fields.insert(
              missing_fields.end(), missing.begin(), missing.end());
        }
      }
      // We'll only run this code when we're checking for errors
      if (!unacquired.empty())
      {
        legion_assert(acquired != nullptr);
        perform_missing_acquires(*acquired, unacquired);
      }
      return -1;  // no composite index
    }

    //--------------------------------------------------------------------------
    bool Operation::physical_convert_postmapping(
        const RegionRequirement& req, std::vector<MappingInstance>& chosen,
        InstanceSet& result, RegionTreeID& bad_tree,
        std::map<PhysicalManager*, unsigned>* acquired,
        std::vector<PhysicalManager*>& unacquired, const bool do_acquire_checks)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      RegionNode* reg_node = runtime->get_node(req.region);
      // Get the field mask for the fields we need
      FieldMask optional_fields =
          reg_node->column_source->get_field_mask(req.privilege_fields);
      const RegionTreeID reg_tree = req.region.get_tree_id();
      // Iterate over each one of the chosen instances
      bool has_composite = false;
      // If we're doing safe mapping, then sort these in order for determinism
      if (runtime->safe_mapper)
        std::sort(chosen.begin(), chosen.end());
      for (const MappingInstance& instance : chosen)
      {
        InstanceManager* man = instance.impl;
        if (man == nullptr)
          continue;
        if (man->is_virtual_manager())
        {
          has_composite = true;
          continue;
        }
        PhysicalManager* manager = man->as_physical_manager();
        // Check to see if the tree IDs are the same
        if (reg_tree != manager->tree_id)
        {
          bad_tree = manager->tree_id;
          return -1;
        }
        // See if we should be checking the acquired sets
        if (do_acquire_checks && (acquired->find(manager) == acquired->end()))
          unacquired.emplace_back(manager);
        FieldMask valid_fields =
            manager->layout->allocated_fields & optional_fields;
        if (!valid_fields)
          continue;
        result.add_instance(InstanceRef(manager, valid_fields));
      }
      if (!unacquired.empty())
      {
        legion_assert(acquired != nullptr);
        perform_missing_acquires(*acquired, unacquired);
      }
      return has_composite;
    }

    //--------------------------------------------------------------------------
    void Operation::perform_missing_acquires(
        std::map<PhysicalManager*, unsigned>& acquired,
        const std::vector<PhysicalManager*>& unacquired)
    //--------------------------------------------------------------------------
    {
      // This code is very similar to what we see in the MapperManager
      for (unsigned idx = 0; idx < unacquired.size(); idx++)
      {
        PhysicalManager* manager = unacquired[idx];
        // Try and do the acquires for any instances that weren't acquired
        if (manager->acquire_instance(MAPPING_ACQUIRE_REF))
          // We already know it wasn't there before
          acquired.insert(std::pair<PhysicalManager*, unsigned>(manager, 1));
      }
    }

    /////////////////////////////////////////////////////////////
    // External Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_mappable(
        const Mappable& mappable, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(mappable.map_id);
      rez.serialize(mappable.tag);
      rez.serialize(mappable.mapper_data_size);
      if (mappable.mapper_data_size > 0)
        rez.serialize(mappable.mapper_data, mappable.mapper_data_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_index_space_requirement(
        const IndexSpaceRequirement& req, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.handle);
      rez.serialize(req.privilege);
      rez.serialize(req.parent);
      // no need to send verified
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_index_space_requirement(
        IndexSpaceRequirement& req, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(req.handle);
      derez.deserialize(req.privilege);
      derez.deserialize(req.parent);
      req.verified = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_region_requirement(
        const RegionRequirement& req, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(req.region);
      rez.serialize(req.partition);
      rez.serialize(req.privilege_fields.size());
      for (const FieldID& field_id : req.privilege_fields)
      {
        rez.serialize(field_id);
      }
      rez.serialize(req.instance_fields.size());
      for (const FieldID& field_id : req.instance_fields)
      {
        rez.serialize(field_id);
      }
      rez.serialize(req.privilege);
      rez.serialize(req.prop);
      rez.serialize(req.parent);
      rez.serialize(req.redop);
      rez.serialize(req.tag);
      rez.serialize(req.flags);
      rez.serialize(req.handle_type);
      rez.serialize(req.projection);
      size_t projection_size = 0;
      const void* projection_args = req.get_projection_args(&projection_size);
      rez.serialize(projection_size);
      if (projection_size > 0)
        rez.serialize(projection_args, projection_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_mappable(
        Mappable& mappable, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(mappable.map_id);
      derez.deserialize(mappable.tag);
      derez.deserialize(mappable.mapper_data_size);
      if (mappable.mapper_data_size > 0)
      {
        // If we already have mapper data, then we are going to replace it
        if (mappable.mapper_data != nullptr)
          free(mappable.mapper_data);
        mappable.mapper_data = malloc(mappable.mapper_data_size);
        derez.deserialize(mappable.mapper_data, mappable.mapper_data_size);
      }
      else if (mappable.mapper_data != nullptr)
      {
        // If we freed it remotely then we can free it here too
        free(mappable.mapper_data);
        mappable.mapper_data = nullptr;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_region_requirement(
        RegionRequirement& req, Deserializer& derez)
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
        req.instance_fields.emplace_back(fid);
      }
      derez.deserialize(req.privilege);
      derez.deserialize(req.prop);
      derez.deserialize(req.parent);
      derez.deserialize(req.redop);
      derez.deserialize(req.tag);
      derez.deserialize(req.flags);
      derez.deserialize(req.handle_type);
      derez.deserialize(req.projection);
      req.flags |= LEGION_VERIFIED_FLAG;
      size_t projection_size;
      derez.deserialize(projection_size);
      if (projection_size > 0)
      {
        void* projection_ptr = malloc(projection_size);
        derez.deserialize(projection_ptr, projection_size);
        req.set_projection_args(projection_ptr, projection_size, true /*own*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_grant(
        const Grant& grant, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      grant.impl->pack_grant(rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_grant(
        Grant& grant, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Create a new grant impl object to perform the unpack
      grant = Grant(new GrantImpl());
      grant.impl->unpack_grant(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::pack_phase_barrier(
        const PhaseBarrier& barrier, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(barrier.phase_barrier);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalMappable::unpack_phase_barrier(
        PhaseBarrier& barrier, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(barrier.phase_barrier);
    }

  }  // namespace Internal
}  // namespace Legion
