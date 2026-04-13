/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/contexts/replicate.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/analysis/projection.h"
#include "legion/api/argument_map_impl.h"
#include "legion/api/data_impl.h"
#include "legion/api/future_impl.h"
#include "legion/api/predicate_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/kdtree.h"
#include "legion/nodes/region.h"
#include "legion/operations/acquire.h"
#include "legion/operations/allreduce.h"
#include "legion/operations/attach.h"
#include "legion/operations/begin.h"
#include "legion/operations/boolean.h"
#include "legion/operations/close.h"
#include "legion/operations/complete.h"
#include "legion/operations/copy.h"
#include "legion/operations/creation.h"
#include "legion/operations/deletion.h"
#include "legion/operations/dependent.h"
#include "legion/operations/detach.h"
#include "legion/operations/discard.h"
#include "legion/operations/dynamic.h"
#include "legion/operations/fence.h"
#include "legion/operations/fill.h"
#include "legion/operations/frame.h"
#include "legion/operations/mapping.h"
#include "legion/operations/mustepoch.h"
#include "legion/operations/partition.h"
#include "legion/operations/recurrent.h"
#include "legion/operations/refinement.h"
#include "legion/operations/release.h"
#include "legion/operations/reset.h"
#include "legion/operations/timing.h"
#include "legion/operations/tunable.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/tracing/logical.h"
#include "legion/tracing/shard.h"
#include "legion/utilities/provenance.h"
#include "legion/views/fill.h"

#define SWAP_PART_KINDS(k1, k2) std::swap(k1, k2)

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Replicate Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplicateContext::ReplicateContext(
        const Mapper::ContextConfigOutput& config, ShardTask* owner, int d,
        bool full, const std::vector<RegionRequirement>& reqs,
        const std::vector<OutputRequirement>& out_reqs,
        const std::vector<unsigned>& parent_indexes,
        const std::vector<bool>& virt_mapped, TaskPriority priority,
        ApEvent exec_fence, ShardManager* manager, bool inline_task,
        bool implicit_task, bool concurrent)
      : HeapifyMixin<ReplicateContext, InnerContext, CONTEXT_LIFETIME>(
            config, owner, d, full, reqs, out_reqs, parent_indexes, virt_mapped,
            priority, exec_fence, 0 /*did*/, inline_task, implicit_task,
            concurrent),
        owner_shard(owner), shard_manager(manager),
        total_shards(shard_manager->total_shards),
        next_close_mapped_bar_index(0), next_refinement_ready_bar_index(0),
        next_refinement_mapped_bar_index(0), next_indirection_bar_index(0),
        next_collective_map_bar_index(0), distributed_id_allocator_shard(0),
        index_space_allocator_shard(0), index_partition_allocator_shard(0),
        field_space_allocator_shard(0), field_allocator_shard(0),
        logical_region_allocator_shard(0), dynamic_id_allocator_shard(0),
        equivalence_set_allocator_shard(0), next_available_collective_index(0),
        next_logical_collective_index(1), next_physical_template_index(0),
        next_replicate_bar_index(0), next_logical_bar_index(0),
        unordered_ops_counter(0), unordered_ops_epoch(MIN_UNORDERED_OPS_EPOCH),
        unordered_collective(nullptr), minimize_repeats_collective(nullptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG_COLLECTIVES
      collective_guard_reentrant = false;
      logical_guard_reentrant = false;
#endif
      shard_manager->add_nested_resource_ref(did);
      size_t num_barriers = LEGION_CONTROL_REPLICATION_COMMUNICATION_BARRIERS;
      close_mapped_barriers.resize(num_barriers);
      refinement_ready_barriers.resize(num_barriers);
      refinement_mapped_barriers.resize(num_barriers);
      indirection_barriers.resize(num_barriers);
      collective_map_barriers.resize(num_barriers);
      // Configure our collective settings
      shard_collective_radix = runtime->legion_collective_radix;
      configure_collective_settings(
          total_shards, owner->shard_id, shard_collective_radix,
          shard_collective_log_radix, shard_collective_stages,
          shard_collective_participating_shards, shard_collective_last_radix);
    }

    //--------------------------------------------------------------------------
    ReplicateContext::~ReplicateContext(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(unordered_collective == nullptr);
      if (shard_manager->remove_nested_resource_ref(did))
        delete shard_manager;
      if (returned_resource_ready_barrier.exists())
        returned_resource_ready_barrier.destroy_barrier();
      if (returned_resource_mapped_barrier.exists())
        returned_resource_mapped_barrier.destroy_barrier();
      if (returned_resource_execution_barrier.exists())
        returned_resource_execution_barrier.destroy_barrier();
      if (minimize_repeats_collective != nullptr)
      {
        minimize_repeats_collective->wait_all_reduce();
        delete minimize_repeats_collective;
      }
    }

    //--------------------------------------------------------------------------
    ContextID ReplicateContext::get_physical_tree_context(void) const
    //--------------------------------------------------------------------------
    {
      // We have all the shards on the same node use the same physical
      // tree context. This is vital for the correct implementation of
      // some parts of physical analysis equivalence set discovery.
      return shard_manager->get_first_shard_tree_context();
    }

    //--------------------------------------------------------------------------
    DistributedID ReplicateContext::get_replication_id(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->did;
    }

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void ReplicateContext::perform_global_registration_callbacks(
        Realm::DSOReferenceImplementation* dso, const void* buffer,
        size_t buffer_size, bool withargs, size_t dedup_tag, RtEvent local_done,
        RtEvent global_done, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_PERFORM_REGISTRATION_CALLBACK, __func__);
        hasher.hash(dso->dso_name.c_str(), dso->dso_name.size(), "dso_name");
        hasher.hash(
            dso->symbol_name.c_str(), dso->symbol_name.size(), "symbol_name");
        hasher.hash(withargs, "withargs");
        hasher.hash(dedup_tag, "dedup_tag");
        if (runtime->safe_control_replication > 1)
          hasher.hash(buffer, buffer_size, "buffer");
        if (hasher.verify(__func__))
          break;
      }
      shard_manager->perform_global_registration_callbacks(
          dso, buffer, buffer_size, withargs, dedup_tag, local_done,
          global_done, preconditions);
    }
#endif

    //--------------------------------------------------------------------------
    void ReplicateContext::print_once(FILE* f, const char* message) const
    //--------------------------------------------------------------------------
    {
      // Only print from shard 0
      if (owner_shard->shard_id == 0)
        fprintf(f, "%s", message);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::log_once(Realm::LoggerMessage& message) const
    //--------------------------------------------------------------------------
    {
      // Deactivate all the messages except shard 0
      if (owner_shard->shard_id != 0)
        message.deactivate();
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::from_value(
        const void* value, size_t size, bool owned, Provenance* provenance,
        bool shard_local)
    //--------------------------------------------------------------------------
    {
      Future result =
          TaskContext::from_value(value, size, owned, provenance, shard_local);
      for (int i = 0;
           runtime->safe_control_replication && !shard_local && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_FUTURE_FROM_VALUE, __func__);
        hash_future(
            hasher, runtime->safe_control_replication, result, "future");
        hasher.hash(size, "size");
        hasher.hash(owned, "owned");
        if (hasher.verify(__func__))
          break;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::from_value(
        const void* buffer, size_t size, bool owned,
        const Realm::ExternalInstanceResource& resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        Provenance* provenance, bool shard_local)
    //--------------------------------------------------------------------------
    {
      Future result = TaskContext::from_value(
          buffer, size, owned, resource, freefunc, provenance, shard_local);
      for (int i = 0;
           runtime->safe_control_replication && !shard_local && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_FUTURE_FROM_VALUE, __func__);
        hash_future(
            hasher, runtime->safe_control_replication, result, "future");
        hasher.hash(size, "size");
        hasher.hash(owned, "owned");
        if (hasher.verify(__func__))
          break;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::consensus_match(
        const void* input, void* output, size_t num_elements,
        size_t element_size, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CONSENSUS_MATCH, __func__);
        if (hasher.verify(__func__))
          break;
      }
      const size_t future_size = sizeof(num_elements);
      FutureImpl* future = new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance);
      // Create the handle to add a reference before doing the match
      Future result(future);
      future->set_future_result_size(future_size, runtime->address_space);
      // The consensus match object will delete itself!
      ConsensusMatchExchange* collective = new ConsensusMatchExchange(
          this, COLLECTIVE_LOC_89, future, input, output, element_size,
          num_elements);
      collective->perform_collective_async();
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID ReplicateContext::register_variant(
        const TaskVariantRegistrar& registrar, const void* user_data,
        size_t user_data_size, const CodeDescriptor& desc, size_t ret_size,
        bool has_ret_size, VariantID vid, bool check_task_id)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::register_variant(
            registrar, user_data, user_data_size, desc, ret_size, has_ret_size,
            vid, check_task_id);
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_REGISTER_TASK_VARIANT, __func__);
        hasher.hash(registrar.task_id, "task_id");
        hasher.hash(registrar.global_registration, "global_registration");
        if (registrar.task_variant_name != nullptr)
          hasher.hash(
              registrar.task_variant_name, strlen(registrar.task_variant_name),
              "task_variant_name");
        hash_execution_constraints(hasher, registrar.execution_constraints);
        for (const std::multimap<unsigned, LayoutConstraintID>::value_type& it :
             registrar.layout_constraints.layouts)
        {
          hasher.hash(it.first, "layout constraints");
          hasher.hash(it.second, "layout_constraints");
        }
        for (const TaskID& it : registrar.generator_tasks)
          hasher.hash(it, "generator_tasks");
        hasher.hash(registrar.leaf_variant, "leaf_variant");
        hasher.hash(registrar.inner_variant, "inner_variant");
        hasher.hash(registrar.idempotent_variant, "idempotent_variant");
        hasher.hash(registrar.replicable_variant, "replicable_variant");
        if (has_ret_size)
          hasher.hash(ret_size, "ret_size");
        if ((user_data != nullptr) && (runtime->safe_control_replication > 1))
          hasher.hash(user_data, user_data_size, "user_data");
        hasher.hash(vid, "vid");
        if (hasher.verify(__func__))
          break;
      }
      // If the task registration is marked as global, then one shard will do
      // the registration and broadcast the variant information to all other
      // shards. If not, all shards performing the registration will
      // independently register the variant.
      VariantID result;
      if (registrar.global_registration)
      {
        ValueBroadcast<VariantID> collective(
            this, dynamic_id_allocator_shard, COLLECTIVE_LOC_17);
        if (owner_shard->shard_id == dynamic_id_allocator_shard)
        {
          result = runtime->register_variant(
              registrar, user_data, user_data_size, desc, ret_size,
              has_ret_size, vid, check_task_id, false /*check context*/);
          collective.broadcast(result);
        }
        else
          result = collective.get_value();
        if (++dynamic_id_allocator_shard == total_shards)
          dynamic_id_allocator_shard = 0;
      }
      else
      {
        // We have to be a little careful here when assigning the variant ID for
        // the registered task. If the user specified it already, then can just
        // use the ID passed in. However, if we are supposed to generate the
        // variant ID, then we'll need to pick an ID (on one shard) and tell
        // everyone else to register the variant with this ID.
        if (vid == LEGION_AUTO_GENERATE_ID)
        {
          TaskImpl* impl = runtime->find_or_create_task_impl(registrar.task_id);
          ValueBroadcast<VariantID> collective(
              this, this->dynamic_id_allocator_shard, COLLECTIVE_LOC_17);
          if (this->owner_shard->shard_id == this->dynamic_id_allocator_shard)
          {
            vid = impl->get_unique_variant_id();
            collective.broadcast(vid);
          }
          else
            vid = collective.get_value();
          if (++dynamic_id_allocator_shard == total_shards)
            dynamic_id_allocator_shard = 0;
        }

        // Finally, if there are multiple shards in the same address space, only
        // one of the shards needs to do the registration.
        if (this->shard_manager->is_first_local_shard(this->owner_shard))
        {
          result = runtime->register_variant(
              registrar, user_data, user_data_size, desc, ret_size,
              has_ret_size, vid, check_task_id, false /*check context*/);
        }
        else
        {
          result = vid;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    VariantImpl* ReplicateContext::select_inline_variant(
        TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
        std::deque<InstanceSet>& physical_instances)
    //--------------------------------------------------------------------------
    {
      VariantImpl* variant_impl = TaskContext::select_inline_variant(
          child, parent_regions, physical_instances);
      if (!variant_impl->is_replicable())
      {
        MapperManager* child_mapper =
            runtime->find_mapper(executing_processor, child->map_id);
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of "
                 "'select_task_variant' on mapper "
              << child_mapper->get_mapper_name()
              << ". Mapper selected an invalid variant ID " << variant_impl->vid
              << " for inlining of task " << child->get_task_name() << " (UID "
              << child->get_unique_id() << "). Parent task "
              << owner_task->get_task_name() << " (UID "
              << owner_task->get_unique_id()
              << ") is a control-replicated task but mapper "
              << "selected non-replicable variant " << variant_impl->vid
              << " for task " << child->get_task_name() << ".";
        error.raise();
      }
      return variant_impl;
    }

    //--------------------------------------------------------------------------
    TraceID ReplicateContext::generate_dynamic_trace_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_trace_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_TRACE_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      TraceID result;
      ValueBroadcast<TraceID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_9);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_trace_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    MapperID ReplicateContext::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_mapper_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_MAPPER_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      MapperID result;
      ValueBroadcast<MapperID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_10);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_mapper_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    ProjectionID ReplicateContext::generate_dynamic_projection_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_projection_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_PROJECTION_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      ProjectionID result;
      ValueBroadcast<ProjectionID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_11);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result =
            runtime->generate_dynamic_projection_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    ShardingID ReplicateContext::generate_dynamic_sharding_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_sharding_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_SHARDING_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      ShardingID result;
      ValueBroadcast<ShardingID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_12);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_sharding_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    ConcurrentID ReplicateContext::generate_dynamic_concurrent_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_concurrent_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_CONCURRENT_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      ConcurrentID result;
      ValueBroadcast<ConcurrentID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_68);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result =
            runtime->generate_dynamic_concurrent_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID ReplicateContext::generate_dynamic_exception_handler_id(
        void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_exception_handler_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_EXCEPTION_HANDLER_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      ExceptionHandlerID result;
      ValueBroadcast<ExceptionHandlerID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_110);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_exception_handler_id(
            false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    TaskID ReplicateContext::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_task_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_TASK_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      TaskID result;
      ValueBroadcast<TaskID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_13);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_task_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionOpID ReplicateContext::generate_dynamic_reduction_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_reduction_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_REDUCTION_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      ReductionOpID result;
      ValueBroadcast<ReductionOpID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_14);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result =
            runtime->generate_dynamic_reduction_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    CustomSerdezID ReplicateContext::generate_dynamic_serdez_id(void)
    //--------------------------------------------------------------------------
    {
      // If we're inside a registration callback we don't care
      if (inside_registration_callback)
        return TaskContext::generate_dynamic_serdez_id();
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_GENERATE_DYNAMIC_SERDEZ_ID, __func__);
        if (hasher.verify(__func__))
          break;
      }
      // Otherwise have one shard make it and broadcast it to everyone else
      CustomSerdezID result;
      ValueBroadcast<CustomSerdezID> collective(
          this, dynamic_id_allocator_shard, COLLECTIVE_LOC_16);
      if (owner_shard->shard_id == dynamic_id_allocator_shard)
      {
        result = runtime->generate_dynamic_serdez_id(false /*check context*/);
        collective.broadcast(result);
      }
      else
        result = collective.get_value();
      if (++dynamic_id_allocator_shard == total_shards)
        dynamic_id_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::perform_semantic_attach(
        const char* func, unsigned kind, const void* arg, size_t arglen,
        SemanticTag tag, const void* buffer, size_t size, bool is_mutable,
        bool& global, const void* arg2, size_t arg2len)
    //--------------------------------------------------------------------------
    {
      if (inside_registration_callback)
        return TaskContext::perform_semantic_attach(
            func, kind, arg, arglen, tag, buffer, size, is_mutable, global,
            arg2, arg2len);
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(kind, func);
        hasher.hash(
            arg, arglen,
            (kind == REPLICATE_ATTACH_TASK_INFO) ? "task_id" : "handle");
        hasher.hash(tag, "tag");
        if (runtime->safe_control_replication > 1)
          hasher.hash(buffer, size, "buffer");
        hasher.hash(is_mutable, "is_mutable");
        hasher.hash(global, "send_to_owner");
        if (arg2 != nullptr)
          hasher.hash(arg2, arg2len, "fid");
        if (hasher.verify(func))
          break;
      }
      // Before we do anything else here, we need to make sure that all
      // the shards are done reading before we attempt to mutate the value
      const RtBarrier bar = semantic_attach_barrier.next(this);
      runtime->phase_barrier_arrive(bar, 1 /*count*/);
      // Check to see if we can downgrade this to a local_only update
      if (global && shard_manager->is_total_sharding())
        global = false;
      // Wait until all the reads of the semantic info are done
      if (!bar.has_triggered())
        bar.wait();
      if (global)
      {
        // If we're still global then just have shard 0 do this for now
        if (owner_shard->shard_id == 0)
          return true;
        post_semantic_attach();
        return false;
      }
      else
      {
        // See if we're the local shard to perform the attach operation
        if (shard_manager->perform_semantic_attach())
          return true;
        post_semantic_attach();
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::post_semantic_attach(void)
    //--------------------------------------------------------------------------
    {
      if (inside_registration_callback)
        return;
      const RtBarrier bar = semantic_attach_barrier.next(this);
      runtime->phase_barrier_arrive(bar, 1 /*count*/);
      if (!bar.has_triggered())
        bar.wait();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::push_exception_handler(ExceptionHandlerID handler)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, (i > 0));
        hasher.hash(REPLICATE_PUSH_EXCEPTION_HANDLER, __func__);
        hasher.hash(handler, "exception_handler");
        if (hasher.verify(__func__))
          break;
      }
      InnerContext::push_exception_handler(handler);
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::pop_exception_handler(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, (i > 0), provenance);
        hasher.hash(REPLICATE_POP_EXCEPTION_HANDLER, __func__);
        if (hasher.verify(__func__))
          break;
      }
      return InnerContext::pop_exception_handler(provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::hash_future(
        HashVerifier& hasher, const unsigned safe_level, const Future& future,
        const char* description) const
    //--------------------------------------------------------------------------
    {
      if (future.impl == nullptr)
        return;
      ContextCoordinate coordinate;
      if (future.impl->get_context_coordinate(this, coordinate))
      {
        // If it came from this context make sure they are the same across
        // the shards, if the future didn't come from this context then by
        // definition it must be the same across the shards
        hasher.hash(coordinate.context_index, description);
        for (int idx = 0; idx < coordinate.index_point.get_dim(); idx++)
          hasher.hash(coordinate.index_point[idx], description);
      }
      else if (safe_level > 1)
      {
        size_t size = 0;
        const void* result = future.impl->get_buffer(
            executing_processor, Memory::SYSTEM_MEM, &size, false /*check*/,
            true /*silence warn*/);
        hasher.hash(result, size, description);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_future_map(
        HashVerifier& hasher, const FutureMap& map, const char* description)
    //--------------------------------------------------------------------------
    {
      if (map.impl == nullptr)
        return;
      hasher.hash(map.impl->blocking_index, description);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_index_space_requirements(
        HashVerifier& hasher, const std::vector<IndexSpaceRequirement>& reqs)
    //--------------------------------------------------------------------------
    {
      if (reqs.empty())
        return;
      Serializer rez;
      for (const IndexSpaceRequirement& it : reqs)
        ExternalMappable::pack_index_space_requirement(it, rez);
      hasher.hash(
          rez.get_buffer(), rez.get_used_bytes(), "index space requirement");
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_region_requirements(
        HashVerifier& hasher, const std::vector<RegionRequirement>& regions)
    //--------------------------------------------------------------------------
    {
      if (regions.empty())
        return;
      Serializer rez;
      for (const RegionRequirement& it : regions)
        ExternalMappable::pack_region_requirement(it, rez);
      hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "region requirement");
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_output_requirements(
        HashVerifier& hasher, const std::vector<OutputRequirement>& outputs)
    //--------------------------------------------------------------------------
    {
      if (outputs.empty())
        return;
      Serializer rez;
      for (const OutputRequirement& it : outputs)
        ExternalTask::pack_output_requirement(it, rez);
      hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "output requirement");
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_grants(
        HashVerifier& hasher, const std::vector<Grant>& grants)
    //--------------------------------------------------------------------------
    {
      if (grants.empty())
        return;
      Serializer rez;
      for (const Grant& it : grants) ExternalMappable::pack_grant(it, rez);
      hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "grants");
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_phase_barriers(
        HashVerifier& hasher, const std::vector<PhaseBarrier>& barriers)
    //--------------------------------------------------------------------------
    {
      if (barriers.empty())
        return;
      // We're not handling phase barriers that come from hanshakes correctly
      // right now because those can be safely different across the shards
      // so only check this with precise checks for now
      if (!hasher.precise)
        return;
      Serializer rez;
      for (const PhaseBarrier& it : barriers)
        ExternalMappable::pack_phase_barrier(it, rez);
      hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "phase barriers");
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_argument(
        HashVerifier& hasher, unsigned safe_level,
        const UntypedBuffer& argument, const char* description)
    //--------------------------------------------------------------------------
    {
      if (safe_level == 1)
        return;
      if (argument.get_size() > 0)
        hasher.hash(argument.get_ptr(), argument.get_size(), description);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_predicate(
        HashVerifier& hasher, const Predicate& pred, const char* description)
    //--------------------------------------------------------------------------
    {
      if (pred == Predicate::TRUE_PRED)
        hasher.hash(0, description);
      else if (pred == Predicate::FALSE_PRED)
        hasher.hash(SIZE_MAX, description);
      else
      {
        ReplPredicateImpl* impl =
            legion_safe_cast<ReplPredicateImpl*>(pred.impl);
        hasher.hash(impl->predicate_coordinate, description);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::hash_static_dependences(
        HashVerifier& hasher, const std::vector<StaticDependence>* dependences)
    //--------------------------------------------------------------------------
    {
      if ((dependences == nullptr) || dependences->empty())
        return;
      Serializer rez;
      for (const StaticDependence& it : *dependences)
      {
        hasher.hash(it.previous_offset, "static dependence previous_offset");
        hasher.hash(
            it.previous_req_index, "static dependence previous_req_index");
        hasher.hash(
            it.current_req_index, "static dependence current_req_index");
        hasher.hash(it.dependence_type, "static dependence dependence_type");
        hasher.hash(it.validates, "static dependence validates");
        hasher.hash(it.shard_only, "static dependence shard_only");
        for (const FieldID& fit : it.dependent_fields)
          hasher.hash(fit, "static dependence field");
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::hash_task_launcher(
        HashVerifier& hasher, const unsigned safe_level,
        const TaskLauncher& launcher) const
    //--------------------------------------------------------------------------
    {
      hasher.hash(launcher.task_id, "task_id");
      hash_index_space_requirements(hasher, launcher.index_requirements);
      hash_region_requirements(hasher, launcher.region_requirements);
      for (const Future& it : launcher.futures)
        hash_future(hasher, safe_level, it, "futures");
      hash_grants(hasher, launcher.grants);
      hash_phase_barriers(hasher, launcher.wait_barriers);
      hash_phase_barriers(hasher, launcher.arrive_barriers);
      hash_argument(hasher, safe_level, launcher.argument, "argument");
      hash_predicate(hasher, launcher.predicate, "predicate");
      hasher.hash(launcher.map_id, "map_id");
      hasher.hash(launcher.tag, "tag");
      hash_argument(hasher, safe_level, launcher.map_arg, "map_arg");
      for (int idx = 0; idx < launcher.point.get_dim(); idx++)
        hasher.hash(launcher.point[idx], "point");
      hasher.hash(launcher.sharding_space, "sharding_space");
      hash_future(
          hasher, safe_level, launcher.predicate_false_future,
          "predicate_false_future");
      hash_argument(
          hasher, safe_level, launcher.predicate_false_result,
          "predicate_false_result");
      hash_static_dependences(hasher, launcher.static_dependences);
      hasher.hash(launcher.enable_inlining, "enable_inlining");
      hasher.hash(launcher.local_function_task, "local_function_task");
      hasher.hash(
          launcher.independent_requirements, "independent_requirements");
      hasher.hash(launcher.silence_warnings, "silence_warnings");
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::hash_index_launcher(
        HashVerifier& hasher, const unsigned safe_level,
        const IndexTaskLauncher& launcher)
    //--------------------------------------------------------------------------
    {
      hasher.hash(launcher.task_id, "task_id");
      hasher.hash(launcher.launch_domain, "launch_domain");
      hasher.hash(launcher.launch_space, "launch_space");
      hasher.hash(launcher.sharding_space, "sharding_space");
      hash_index_space_requirements(hasher, launcher.index_requirements);
      hash_region_requirements(hasher, launcher.region_requirements);
      for (const Future& it : launcher.futures)
        hash_future(hasher, safe_level, it, "futures");
      for (const ArgumentMap& it : launcher.point_futures)
        hash_future_map(
            hasher, it.impl->freeze(this, hasher.provenance), "point_futures");
      hash_grants(hasher, launcher.grants);
      hash_phase_barriers(hasher, launcher.wait_barriers);
      hash_phase_barriers(hasher, launcher.arrive_barriers);
      hash_argument(hasher, safe_level, launcher.global_arg, "global_arg");
      if (launcher.argument_map.impl != nullptr)
        hash_future_map(
            hasher, launcher.argument_map.impl->freeze(this, hasher.provenance),
            "argument_map");
      hash_predicate(hasher, launcher.predicate, "predicate");
      hasher.hash(launcher.concurrent, "concurrent");
      hasher.hash(launcher.concurrent_functor, "concurrent_functor");
      hasher.hash(launcher.must_parallelism, "must_parallelism");
      hasher.hash(launcher.map_id, "map_id");
      hasher.hash(launcher.tag, "tag");
      hash_argument(hasher, safe_level, launcher.map_arg, "map_arg");
      hash_future(
          hasher, safe_level, launcher.predicate_false_future,
          "predicate_false_future");
      hash_future(hasher, safe_level, launcher.initial_value, "initial_value");
      hash_argument(
          hasher, safe_level, launcher.predicate_false_result,
          "predicate_false_result");
      hash_static_dependences(hasher, launcher.static_dependences);
      hasher.hash(launcher.enable_inlining, "enable_inlining");
      hasher.hash(
          launcher.independent_requirements, "independent_requirements");
      hasher.hash(launcher.silence_warnings, "silence_warnings");
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::hash_execution_constraints(
        HashVerifier& hasher, const ExecutionConstraintSet& constraints)
    //--------------------------------------------------------------------------
    {
      hasher.hash(constraints.isa_constraint.isa_prop, "ISA Constraint");
      for (const Processor::Kind& it :
           constraints.processor_constraint.valid_kinds)
        hasher.hash(it, "Processor Constraint");
      for (const ResourceConstraint& it : constraints.resource_constraints)
      {
        hasher.hash(it.resource_kind, "Resource Constraint resource_kind");
        hasher.hash(it.equality_kind, "Resource Constraint equality_kind");
        hasher.hash(it.value, "Resource Constraint value");
      }
      for (const LaunchConstraint& it : constraints.launch_constraints)
      {
        hasher.hash(it.launch_kind, "Launch Constraint launch_kind");
        hasher.hash(it.dims, "Launch Constraint dims");
        for (int i = 0; i < it.dims; i++)
          hasher.hash(it.values[i], "Launch Constraint value");
      }
      for (const ColocationConstraint& cit : constraints.colocation_constraints)
      {
        for (const FieldID& it : cit.fields)
          hasher.hash(it, "Colocation Constraint fields");
        for (const unsigned& it : cit.indexes)
          hasher.hash(it, "Colocation Constraint indexes");
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::hash_layout_constraints(
        HashVerifier& hasher, const LayoutConstraintSet& constraints,
        bool hash_pointers)
    //--------------------------------------------------------------------------
    {
      hasher.hash(
          constraints.specialized_constraint.kind,
          "Specialized Constraint kind");
      hasher.hash(
          constraints.specialized_constraint.redop,
          "Specialized Constraint redop");
      hasher.hash(
          constraints.specialized_constraint.max_pieces,
          "Specialized Constraint max_pieces");
      hasher.hash(
          constraints.specialized_constraint.max_overhead,
          "Specialized Constraint max_overhead");
      hasher.hash(
          constraints.specialized_constraint.no_access,
          "Specialized Constraint no_access");
      hasher.hash(
          constraints.specialized_constraint.exact,
          "Specialized Constraint exact");
      for (const FieldID& it : constraints.field_constraint.field_set)
        hasher.hash(it, "Field Constraint fields");
      hasher.hash(
          constraints.field_constraint.contiguous,
          "Field Constraint contiguous");
      hasher.hash(
          constraints.field_constraint.inorder, "Field Constraint inorder");
      if (constraints.memory_constraint.has_kind)
        hasher.hash(
            constraints.memory_constraint.kind, "Memory Constraint kind");
      if (hash_pointers && constraints.pointer_constraint.is_valid)
      {
        hasher.hash(
            constraints.pointer_constraint.memory, "Pointer Constraint memory");
        hasher.hash(
            constraints.pointer_constraint.ptr, "Pointer Constraint ptr");
      }
      for (const DimensionKind& it : constraints.ordering_constraint.ordering)
        hasher.hash(it, "Ordering Constraint ordering");
      hasher.hash(
          constraints.ordering_constraint.contiguous,
          "Ordering Constraint contiguous");
      for (const TilingConstraint& it : constraints.tiling_constraints)
      {
        hasher.hash(it.dim, "Tiling Constraint dim");
        hasher.hash(it.value, "Tiling Constraint value");
        hasher.hash(it.tiles, "Tiling Constraint tiles");
      }
      for (const DimensionConstraint& it : constraints.dimension_constraints)
      {
        hasher.hash(it.kind, "Dimension Constraint kind");
        hasher.hash(it.eqk, "Dimension Constraint eqk");
        hasher.hash(it.value, "Splitting Constraint value");
      }
      for (const AlignmentConstraint& it : constraints.alignment_constraints)
      {
        hasher.hash(it.fid, "Alignment Constraint fid");
        hasher.hash(it.eqk, "Alignment Constraint eqk");
        hasher.hash(it.alignment, "Alignment Constraint alignment");
      }
      for (const OffsetConstraint& it : constraints.offset_constraints)
      {
        hasher.hash(it.fid, "Offset Constraint fid");
        hasher.hash(it.offset, "Offset Constraint offset");
      }
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::verify_hash(
        const uint64_t hash[2], const char* description, Provenance* provenance,
        bool verify_every_call)
    //--------------------------------------------------------------------------
    {
      VerifyReplicableExchange exchange(COLLECTIVE_LOC_82, this);
      const VerifyReplicableExchange::ShardHashes& hashes =
          exchange.exchange(hash);
      // If all shards had the same hashes then we are done
      if (hashes.size() == 1)
        return true;
      if (!verify_every_call)
      {
        // First pass, we detected a violation so go around again and see
        // if we can find out exactly which member is bad
        // Report the warning on one of the lowest hashes
        const std::pair<uint64_t, uint64_t> key(hash[0], hash[1]);
        const VerifyReplicableExchange::ShardHashes::const_iterator finder =
            hashes.find(key);
        legion_assert(finder != hashes.end());
        if (finder->second == owner_shard->shard_id)
          log_legion.error(
              "Detected control replication violation when invoking %s in "
              "task %s (UID %lld) on shard %d [Provenance: %.*s]. The hash "
              "summary"
              " for the function does not align with the hash summaries from "
              "other call sites. We'll run the hash algorithm again to try to "
              "recognize what value differs between the shards, hang tight...",
              description, get_task_name(), get_unique_id(),
              owner_shard->shard_id,
              (provenance == nullptr) ? 7 : int(provenance->human.length()),
              (provenance == nullptr) ? "unknown" : provenance->human.data());
      }
      else
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Specific control replication violation occurred from member "
              << description << ".";
        error.raise();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* ReplicateContext::create_initial_equivalence_set(
        unsigned idx, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      return shard_manager->get_initial_equivalence_set(
          idx, req.region, find_parent_physical_context(idx),
          (owner_shard->shard_id == 0));
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::receive_created_region_contexts(
        const std::vector<RegionNode*>& created_nodes,
        const std::vector<EqKDTree*>& created_trees,
        std::set<RtEvent>& applied_events, const ShardMapping* mapping,
        ShardID source_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(created_nodes.size() == created_trees.size());
      if ((mapping == nullptr) || (mapping->size() != total_shards))
      {
        // Do the volumetric extraction to send all of the equivalence sets
        // from the source shard to the right shards in this context
        local::map<
            ShardID,
            local::map<RegionNode*, local::FieldMaskMap<EquivalenceSet> > >
            eq_sets;
        for (unsigned idx = 0; idx < created_nodes.size(); idx++)
          if (created_trees[idx] != nullptr)
            created_trees[idx]->find_shard_equivalence_sets(
                eq_sets, source_shard, 0 /*lower shard id*/,
                total_shards - 1 /*upper shard id*/, created_nodes[idx]);
        for (const local::map<
                 ShardID,
                 local::map<
                     RegionNode*, local::FieldMaskMap<EquivalenceSet> > >::
                 value_type& sit : eq_sets)
        {
          ReplCreatedRegions rez;
          rez.serialize(shard_manager->did);
          rez.serialize(sit.first);
          rez.serialize<size_t>(sit.second.size());
          for (const local::map<
                   RegionNode*,
                   local::FieldMaskMap<EquivalenceSet> >::value_type& rit :
               sit.second)
          {
            rez.serialize(rit.first->handle);
            rit.first->pack_global_ref();
            rez.serialize(rit.second.size());
            for (const std::pair<EquivalenceSet*, FieldMask>& it : rit.second)
            {
              it.first->pack_global_ref();
              rez.serialize(it.first->did);
              rez.serialize(it.second);
            }
          }
          shard_manager->send_created_region_contexts(
              sit.first, rez, applied_events);
        }
      }
      else
      {
        // If we have the same number of shards then we know that the
        // equivalence set k-d trees will be the same so we can just
        // use the base case. However we still need to send the
        // information to the right shard
        if (source_shard != owner_shard->shard_id)
        {
          // Pack it up and send it to the source shard
          ReplCreatedRegions rez;
          rez.serialize(shard_manager->did);
          rez.serialize(source_shard);
          rez.serialize<size_t>(created_nodes.size());
          for (unsigned idx = 0; idx < created_nodes.size(); idx++)
          {
            RegionNode* region = created_nodes[idx];
            rez.serialize(region->handle);
            region->pack_global_ref();
            local::FieldMaskMap<EquivalenceSet> eq_sets;
            if (created_trees[idx] != nullptr)
              created_trees[idx]->find_local_equivalence_sets(
                  eq_sets, source_shard);
            rez.serialize<size_t>(eq_sets.size());
            for (local::FieldMaskMap<EquivalenceSet>::const_iterator it =
                     eq_sets.begin();
                 it != eq_sets.end(); it++)
            {
              it->first->pack_global_ref();
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
          }
          shard_manager->send_created_region_contexts(
              source_shard, rez, applied_events);
        }
        else
          InnerContext::receive_created_region_contexts(
              created_nodes, created_trees, applied_events, mapping,
              source_shard);
      }
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::compute_shard_to_shard_mapping(
        const ShardMapping& src_mapping,
        std::multimap<ShardID, ShardID>& result) const
    //--------------------------------------------------------------------------
    {
      // Build a mapping between the shards of the source context and
      // the shards of this context and then decide what to do with
      // the data from the source shard. Note that this mapping needs
      // to be constructed deterministically across all the shards so
      // that they all agree to do the same thing. The resulting mapping
      // needs to satisfy two properties:
      // 1. Every shard in the destination context must get at least
      //    one update from a shard in the source context (surjective)
      // 2. Every shard in the source context has to got to at least
      //    one shard in the destination context
      const ShardMapping& dst_mapping = shard_manager->get_mapping();
      const CollectiveMapping& dst_spaces =
          shard_manager->get_collective_mapping();
      std::vector<ShardID> address_space_shard_offsets(dst_spaces.size(), 0);
      std::vector<bool> targeted(dst_mapping.size(), false);
      size_t total_targets = 0;
      for (ShardID src = 0; src < src_mapping.size(); src++)
      {
        AddressSpaceID src_space = src_mapping[src];
        AddressSpaceID dst_space = dst_spaces.contains(src_space) ?
                                       src_space :
                                       dst_spaces.find_nearest(src_space);
        ShardID dst = address_space_shard_offsets[dst_space];
        // Find the next shard in our map at the dst space
        for (unsigned offset = 0; offset < dst_mapping.size(); offset++, dst++)
        {
          legion_assert(dst <= dst_mapping.size());
          if (dst == dst_mapping.size())
            dst = 0;  // reset back to the first shard on wrap around
          if (dst_mapping[dst] != dst_space)
            continue;
          result.insert(std::make_pair(src, dst));
          address_space_shard_offsets[dst_space] = dst + 1;
          if (!targeted[dst])
          {
            targeted[dst] = true;
            total_targets++;
          }
          break;
        }
        // Should have assigned something for this shard
        legion_assert(result.lower_bound(src)->first == src);
      }
      if (total_targets < dst_mapping.size())
      {
        // Not all the destination shards have targets
        // Find the nearest shards in the source to the destination
        // and have them send their results to those destinations too
        // If the source shard is the one we're receiving then
        // add to the destination shard to the targets
        const CollectiveMapping src_spaces(
            src_mapping, runtime->legion_collective_radix);
        address_space_shard_offsets.resize(src_spaces.size());
        for (ShardID shard = 0; shard < src_spaces.size(); shard++)
          address_space_shard_offsets[shard] = 0;
        for (ShardID dst = 0; dst < targeted.size(); dst++)
        {
          if (targeted[dst])
            continue;
          AddressSpace dst_space = dst_mapping[dst];
          AddressSpace src_space = src_spaces.contains(dst_space) ?
                                       dst_space :
                                       src_spaces.find_nearest(dst_space);
          ShardID src = address_space_shard_offsets[src_space];
          for (unsigned offset = 0; offset < src_mapping.size();
               offset++, src++)
          {
            legion_assert(src <= src_mapping.size());
            if (src == src_mapping.size())
              src = 0;  // reset back to the first shard on wrap around
            if (src_mapping[src] != src_space)
              continue;
            result.insert(std::make_pair(src, dst));
            address_space_shard_offsets[src_space] = src + 1;
            break;
          }
          legion_assert(
              (src % src_mapping.size()) !=
              address_space_shard_offsets[src_space]);
        }
      }
      // Do the identity check
      for (const std::multimap<ShardID, ShardID>::value_type& it : result)
        if (it.first != it.second)
          return false;
      // If we get here we passed the identity check so clear the result
      result.clear();
      return true;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space(
        const Domain& domain, bool take_ownership, TypeTag type_tag,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, (i > 0), provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE, __func__);
        hasher.hash(domain, "domain");
        hasher.hash(take_ownership, "take ownership");
        hasher.hash(type_tag, "type_tag");
        if (hasher.verify(__func__))
          break;
      }
      return create_index_space_replicated(
          domain, type_tag, provenance, take_ownership);
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_replicated(
        const Domain& domain, TypeTag type_tag, Provenance* provenance,
        bool take_ownership)
    //--------------------------------------------------------------------------
    {
      // Seed this with the first index space broadcast
      if (pending_index_spaces.empty())
      {
        increase_pending_index_spaces(1 /*count*/, false /*double*/);
        pending_index_space_check = 0;
      }
      IndexSpace handle;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
          pending_index_spaces.front();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, type_tag);
        double_buffer = value.double_buffer;
        runtime->create_node(
            handle, domain, take_ownership, nullptr /*parent*/, 0 /*color*/,
            creation_bar, provenance, ApEvent::NO_AP_EVENT, value.expr_id,
            &collective_mapping, true /*add root reference*/);
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_index_space(value.did);
        LegionSpy::log_top_index_space(
            handle.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, type_tag);
        legion_assert(handle.exists());
        double_buffer = value.double_buffer;
        runtime->create_node(
            handle, domain, take_ownership, nullptr /*parent*/, 0 /*color*/,
            creation_bar, provenance, ApEvent::NO_AP_EVENT, value.expr_id,
            &collective_mapping, true /*add root reference*/);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      // Record this in our context
      register_index_space_creation(handle);
      if (++pending_index_space_check == pending_index_spaces.size())
        pending_index_space_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_index_spaces(
          double_buffer ? pending_index_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_index_spaces.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_unbound_index_space(
        TypeTag type_tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_UNBOUND_INDEX_SPACE, __func__);
        hasher.hash(type_tag, "type_tag");
        if (hasher.verify(__func__))
          break;
      }
      return create_index_space_replicated(
          Domain::NO_DOMAIN, type_tag, provenance, true);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_index_spaces(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<ISBroadcast>* collective =
            new ValueBroadcast<ISBroadcast>(
                this, index_space_allocator_shard, COLLECTIVE_LOC_3);
        if (owner_shard->shard_id == index_space_allocator_shard)
        {
          const DistributedID did = runtime->get_unique_index_space_id();
          // We're the owner, so make it locally and then broadcast it
          runtime->record_pending_index_space(did);
          // Do our arrival on this generation, should be the last one
          collective->broadcast(ISBroadcast(
              did, runtime->get_unique_index_tree_id(),
              runtime->get_unique_index_space_expr_id(), double_next));
          pending_index_spaces.emplace_back(
              std::pair<ValueBroadcast<ISBroadcast>*, bool>(collective, true));
        }
        else
        {
          register_collective(collective);
          pending_index_spaces.emplace_back(
              std::pair<ValueBroadcast<ISBroadcast>*, bool>(collective, false));
        }
        index_space_allocator_shard++;
        if (index_space_allocator_shard == total_shards)
          index_space_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space(
        const Future& future, TypeTag type_tag, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE, __func__);
        hash_future(
            hasher, runtime->safe_control_replication, future, "future");
        hasher.hash(type_tag, "type_tag");
        if (hasher.verify(__func__))
          break;
      }
      // Seed this with the first index space broadcast
      if (pending_index_spaces.empty())
      {
        increase_pending_index_spaces(1 /*count*/, false /*double*/);
        pending_index_space_check = 0;
      }
      IndexSpace handle;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
          pending_index_spaces.front();
      IndexSpaceNode* node = nullptr;
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const ApEvent ready = creator_op->get_completion_event();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, type_tag);
        double_buffer = value.double_buffer;
        node = runtime->create_node(
            handle, Domain::NO_DOMAIN, true /*task ownership*/,
            nullptr /*parent*/, 0 /*color*/, creation_bar, provenance, ready,
            value.expr_id, &collective_mapping, true /*add root reference*/);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_index_space(value.did);
        LegionSpy::log_top_index_space(
            handle.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, type_tag);
        legion_assert(handle.exists());
        double_buffer = value.double_buffer;
        node = runtime->create_node(
            handle, Domain::NO_DOMAIN, true /*take ownership*/,
            nullptr /*parent*/, 0 /*color*/, creation_bar, provenance, ready,
            value.expr_id, &collective_mapping, true /*add root reference*/);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      creator_op->initialize_index_space(
          this, node, future, provenance,
          shard_manager->is_first_local_shard(owner_shard),
          &(shard_manager->get_collective_mapping()));
      add_to_dependence_queue(creator_op);
      // Record this in our context
      register_index_space_creation(handle);
      if (++pending_index_space_check == pending_index_spaces.size())
        pending_index_space_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_index_spaces(
          double_buffer ? pending_index_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_index_spaces.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space(
        const std::vector<DomainPoint>& points, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE, __func__);
        for (unsigned idx = 0; idx < points.size(); idx++)
          hasher.hash(points[idx], "points");
        if (hasher.verify(__func__))
          break;
      }
      switch (points[0].get_dim())
      {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      std::vector<Realm::Point<DIM, coord_t> > realm_points(points.size());  \
      for (unsigned idx = 0; idx < points.size(); idx++)                     \
        realm_points[idx] = Point<DIM, coord_t>(points[idx]);                \
      const DomainT<DIM, coord_t> realm_is(                                  \
          (Realm::IndexSpace<DIM, coord_t>(realm_points)));                  \
      const Domain bounds(realm_is);                                         \
      return create_index_space_replicated(                                  \
          bounds, NT_TemplateHelper::encode_tag<DIM, coord_t>(), provenance, \
          true);                                                             \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space(
        const std::vector<Domain>& rects, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE, __func__);
        for (unsigned idx = 0; idx < rects.size(); idx++)
          hasher.hash(rects[idx], "rects");
        if (hasher.verify(__func__))
          break;
      }
      switch (rects[0].get_dim())
      {
#define DIMFUNC(DIM)                                                         \
  case DIM:                                                                  \
    {                                                                        \
      std::vector<Realm::Rect<DIM, coord_t> > realm_rects(rects.size());     \
      for (unsigned idx = 0; idx < rects.size(); idx++)                      \
        realm_rects[idx] = Rect<DIM, coord_t>(rects[idx]);                   \
      const DomainT<DIM, coord_t> realm_is(                                  \
          (Realm::IndexSpace<DIM, coord_t>(realm_rects)));                   \
      const Domain bounds(realm_is);                                         \
      return create_index_space_replicated(                                  \
          bounds, NT_TemplateHelper::encode_tag<DIM, coord_t>(), provenance, \
          true);                                                             \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return IndexSpace::NO_SPACE;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::union_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_UNION_INDEX_SPACES, __func__);
        for (const IndexSpace& it : spaces) hasher.hash(it, "spaces");
        if (hasher.verify(__func__))
          break;
      }
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      bool none_exists = true;
      for (const IndexSpace& it : spaces)
      {
        if (none_exists && it.exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it.get_type_tag())
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'union_index_spaces' performed in "
                   "task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
          error.raise();
        }
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      // Seed this with the first index space broadcast
      if (pending_index_spaces.empty())
      {
        increase_pending_index_spaces(1 /*count*/, false /*double*/);
        pending_index_space_check = 0;
      }
      IndexSpace handle;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
          pending_index_spaces.front();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, spaces[0].get_type_tag());
        double_buffer = value.double_buffer;
        runtime->create_union_space(
            handle, provenance, spaces, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_index_space(value.did);
        LegionSpy::log_top_index_space(
            handle.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, spaces[0].get_type_tag());
        legion_assert(handle.exists());
        double_buffer = value.double_buffer;
        runtime->create_union_space(
            handle, provenance, spaces, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      // Record this in our context
      register_index_space_creation(handle);
      if (++pending_index_space_check == pending_index_spaces.size())
        pending_index_space_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_index_spaces(
          double_buffer ? pending_index_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_index_spaces.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::intersect_index_spaces(
        const std::vector<IndexSpace>& spaces, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_INTERSECT_INDEX_SPACES, __func__);
        for (const IndexSpace& it : spaces) hasher.hash(it, "spaces");
        if (hasher.verify(__func__))
          break;
      }
      if (spaces.empty())
        return IndexSpace::NO_SPACE;
      bool none_exists = true;
      for (const IndexSpace& it : spaces)
      {
        if (none_exists && it.exists())
          none_exists = false;
        if (spaces[0].get_type_tag() != it.get_type_tag())
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'intersect_index_spaces' "
                   "performed in task "
                << get_task_name() << " (UID " << get_unique_id() << ")";
          error.raise();
        }
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      // Seed this with the first index space broadcast
      if (pending_index_spaces.empty())
      {
        increase_pending_index_spaces(1 /*count*/, false /*double*/);
        pending_index_space_check = 0;
      }
      IndexSpace handle;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
          pending_index_spaces.front();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, spaces[0].get_type_tag());
        double_buffer = value.double_buffer;
        runtime->create_intersection_space(
            handle, provenance, spaces, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_index_space(value.did);
        LegionSpy::log_top_index_space(
            handle.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, spaces[0].get_type_tag());
        legion_assert(handle.exists());
        double_buffer = value.double_buffer;
        runtime->create_intersection_space(
            handle, provenance, spaces, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      // Record this in our context
      register_index_space_creation(handle);
      if (++pending_index_space_check == pending_index_spaces.size())
        pending_index_space_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_index_spaces(
          double_buffer ? pending_index_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_index_spaces.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::subtract_index_spaces(
        IndexSpace left, IndexSpace right, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_SUBTRACT_INDEX_SPACES, __func__);
        hasher.hash(left, "left");
        hasher.hash(right, "right");
        if (hasher.verify(__func__))
          break;
      }
      if (!left.exists())
        return IndexSpace::NO_SPACE;
      if (right.exists() && left.get_type_tag() != right.get_type_tag())
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "Dynamic type mismatch in 'create_difference_spaces' "
                 "performed in task "
              << get_task_name() << " (UID " << get_unique_id() << ")";
        error.raise();
      }
      // Seed this with the first index space broadcast
      if (pending_index_spaces.empty())
      {
        increase_pending_index_spaces(1 /*count*/, false /*double*/);
        pending_index_space_check = 0;
      }
      IndexSpace handle;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
          pending_index_spaces.front();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, left.get_type_tag());
        double_buffer = value.double_buffer;
        runtime->create_difference_space(
            handle, provenance, left, right, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_index_space(value.did);
        LegionSpy::log_top_index_space(
            handle.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const ISBroadcast value = collective.first->get_value(false);
        handle = IndexSpace(value.did, value.tid, left.get_type_tag());
        legion_assert(handle.exists());
        double_buffer = value.double_buffer;
        runtime->create_difference_space(
            handle, provenance, left, right, creation_bar, &collective_mapping,
            value.expr_id);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      // Record this in our context
      register_index_space_creation(handle);
      if (++pending_index_space_check == pending_index_spaces.size())
        pending_index_space_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_index_spaces(
          double_buffer ? pending_index_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_index_spaces.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::create_shared_ownership(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CREATE_SHARED_OWNERSHIP, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_index_space(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to create shared ownership for index space "
              << handle << " in task " << get_task_name() << " (UID "
              << get_unique_id()
              << ") which is not a top-level index space. Legion only permits "
              << "top-level index spaces to have shared ownership.";
        error.raise();
      }
      if (shard_manager->is_total_sharding() &&
          shard_manager->is_first_local_shard(owner_shard))
        runtime->create_shared_ownership(handle, true /*total sharding*/);
      else if (owner_shard->shard_id == 0)
        runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexSpace, unsigned>::iterator finder =
          created_index_spaces.find(handle);
      if (finder != created_index_spaces.end())
        finder->second++;
      else
        created_index_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_index_space(
        IndexSpace handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DESTROY_INDEX_SPACE, __func__);
        hasher.hash(handle, "handle");
        hasher.hash(recurse, "recurse");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      // Check to see if this is a top-level index space, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_index_space(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to destroy index space " << handle << " in task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") which is not a top-level index space. Legion only permits "
              << "top-level index spaces to be destroyed.";
        error.raise();
      }
      // Check to see if this is one that we should be allowed to destory
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexSpace, unsigned>::iterator finder =
            created_index_spaces.find(handle);
        if (finder == created_index_spaces.end())
        {
          // If we didn't make the index space in this context, just
          // record it and keep going, it will get handled later
          deleted_index_spaces.emplace_back(
              DeletedIndexSpace(handle, recurse, provenance));
          return;
        }
        else
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_index_spaces.erase(finder);
          else
            return;
        }
        if (recurse)
        {
          // Also remove any index partitions for this index space tree
          for (std::map<IndexPartition, unsigned>::iterator it =
                   created_index_partitions.begin();
               it != created_index_partitions.end();
               /*nothing*/)
          {
            if (it->first.get_tree_id() == handle.get_tree_id())
            {
              sub_partitions.emplace_back(it->first);
              legion_assert(it->second > 0);
              if (--it->second == 0)
              {
                std::map<IndexPartition, unsigned>::iterator to_delete = it++;
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
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_index_space_deletion(
          this, handle, sub_partitions, unordered, provenance);
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal unordered index space deletion performed after task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") has finished executing. All unordered operations must be "
                 "performed "
              << "before the end of the execution of the parent task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::create_shared_ownership(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CREATE_SHARED_OWNERSHIP, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      if (shard_manager->is_total_sharding() &&
          shard_manager->is_first_local_shard(owner_shard))
        runtime->create_shared_ownership(handle, true /*total sharding*/);
      else if (owner_shard->shard_id == 0)
        runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<IndexPartition, unsigned>::iterator finder =
          created_index_partitions.find(handle);
      if (finder != created_index_partitions.end())
        finder->second++;
      else
        created_index_partitions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_index_partition(
        IndexPartition handle, const bool unordered, const bool recurse,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DESTROY_INDEX_PARTITION, __func__);
        hasher.hash(handle, "handle");
        hasher.hash(recurse, "recurse");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      std::vector<IndexPartition> sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        std::map<IndexPartition, unsigned>::iterator finder =
            created_index_partitions.find(handle);
        if (finder != created_index_partitions.end())
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_index_partitions.erase(finder);
          else
            return;
          if (recurse)
          {
            // Remove any other partitions that this partition dominates
            for (std::map<IndexPartition, unsigned>::iterator it =
                     created_index_partitions.begin();
                 it != created_index_partitions.end();
                 /*nothing*/)
            {
              if ((handle.get_tree_id() == it->first.get_tree_id()) &&
                  runtime->is_dominated_tree_only(it->first, handle))
              {
                sub_partitions.emplace_back(it->first);
                legion_assert(it->second > 0);
                if (--it->second == 0)
                {
                  std::map<IndexPartition, unsigned>::iterator to_delete = it++;
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
          // If we didn't make the partition, record it and keep going
          deleted_index_partitions.emplace_back(
              DeletedPartition(handle, recurse, provenance));
          return;
        }
      }
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_index_part_deletion(
          this, handle, sub_partitions, unordered, provenance);
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal unordered index partition deletion performed after "
                 "task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") has finished executing. All unordered operations must be "
                 "performed "
              << "before the end of the execution of the parent task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_partitions(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<IPBroadcast>* collective =
            new ValueBroadcast<IPBroadcast>(
                this, index_partition_allocator_shard, COLLECTIVE_LOC_7);
        pending_index_partitions.emplace_back(
            std::pair<ValueBroadcast<IPBroadcast>*, ShardID>(
                collective, index_partition_allocator_shard));
        if (owner_shard->shard_id == index_partition_allocator_shard)
        {
          const DistributedID did = runtime->get_unique_index_partition_id();
          // We're the owner, so make it locally and then broadcast it
          runtime->record_pending_partition(did);
          // Do our arrival on this generation, should be the last one
          collective->broadcast(IPBroadcast(did, double_next));
        }
        else
          register_collective(collective);
        index_partition_allocator_shard++;
        if (index_partition_allocator_shard == total_shards)
          index_partition_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::create_shard_partition(
        Operation* op, IndexPartition& pid, IndexSpace parent,
        IndexSpace color_space, Provenance* provenance, PartitionKind part_kind,
        LegionColor partition_color, bool color_generated)
    //--------------------------------------------------------------------------
    {
      if (pending_index_partitions.empty())
      {
        increase_pending_partitions(1 /*count*/, false /*double*/);
        pending_index_partition_check = 0;
      }
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<IPBroadcast>*, ShardID>& collective =
          pending_index_partitions.front();
      const bool is_owner = (collective.second == owner_shard->shard_id);
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (is_owner)
      {
        const IPBroadcast value = collective.first->get_value(false);
        pid.did = value.did;
        double_buffer = value.double_buffer;
        // Have to do our registration before broadcasting
        RtEvent safe_event = create_pending_partition_internal(
            pid, parent, color_space, partition_color, part_kind, provenance,
            &collective_mapping, creation_bar);
        // Broadcast the color if we have to generate it
        if (color_generated)
        {
          legion_assert(
              partition_color != INVALID_COLOR);  // we should have an ID
          ValueBroadcast<LegionColor> color_collective(
              this, collective.second, COLLECTIVE_LOC_8);
          color_collective.broadcast(partition_color);
        }
        // Signal that we're done our creation
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, safe_event);
        runtime->revoke_pending_partition(value.did);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const IPBroadcast value = collective.first->get_value(false);
        pid.did = value.did;
        double_buffer = value.double_buffer;
        legion_assert(pid.exists());
        // If we need a color then we can get that too
        if (color_generated)
        {
          ValueBroadcast<LegionColor> color_collective(
              this, collective.second, COLLECTIVE_LOC_8);
          partition_color = color_collective.get_value();
          legion_assert(partition_color != INVALID_COLOR);
        }
        // Do our registration
        RtEvent safe_event = create_pending_partition_internal(
            pid, parent, color_space, partition_color, part_kind, provenance,
            &collective_mapping, creation_bar);
        // Signal that we're done our creation
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, safe_event);
      }
      if (++pending_index_partition_check == pending_index_partitions.size())
        pending_index_partition_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_partitions(
          double_buffer ? pending_index_partitions.size() + 1 : 1, double_next);
      // Clean up the collective
      delete collective.first;
      pending_index_partitions.pop_front();
      return is_owner;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_equal_partition(
        IndexSpace parent, IndexSpace color_space, size_t granularity,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_EQUAL_PARTITION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(color_space, "color_space");
        hasher.hash(granularity, "granularity");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance,
          LEGION_DISJOINT_COMPLETE_KIND, partition_color, color_generated);
      part_op->initialize_equal_partition(this, pid, granularity, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_weights(
        IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
        size_t granularity, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_WEIGHTS, __func__);
        hasher.hash(parent, "parent");
        hash_future_map(hasher, weights, "weights");
        hasher.hash(color_space, "color_space");
        hasher.hash(granularity, "granularity");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance,
          LEGION_DISJOINT_COMPLETE_KIND, partition_color, color_generated);
      part_op->initialize_weight_partition(
          this, pid, weights, granularity, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_union(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_UNION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(handle1, "handle1");
        hasher.hash(handle2, "handle2");
        hasher.hash(color_space, "color_space");
        hasher.hash(kind, "kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "IndexPartition " << handle1
              << " is not part of the same index tree as "
              << "IndexSpace " << parent << " in create partition by union.";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "IndexPartition " << handle2
              << " is not part of the same index tree as "
              << "IndexSpace " << parent << " in create partition by union.";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      // If either partition is aliased the result is aliased
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        // If one of these partitions is aliased then the result is aliased
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (p1->is_disjoint(true /*from app*/))
        {
          IndexPartNode* p2 = runtime->get_node(handle2);
          if (!p2->is_disjoint(true /*from app*/))
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
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, kind, partition_color,
          color_generated);
      part_op->initialize_union_partition(
          this, pid, handle1, handle2, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_INTERSECTION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(handle1, "handle1");
        hasher.hash(handle2, "handle2");
        hasher.hash(color_space, "color_space");
        hasher.hash(kind, "kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1
              << " is not part of the same index tree as IndexSpace " << parent
              << " in create partition by intersection.";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle2
              << " is not part of the same index tree as IndexSpace " << parent
              << " in create partition by intersection.";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      // If either partition is disjoint then the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (!p1->is_disjoint(true /*from app*/))
        {
          IndexPartNode* p2 = runtime->get_node(handle2);
          if (p2->is_disjoint(true /*from app*/))
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
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, kind, partition_color,
          color_generated);
      part_op->initialize_intersection_partition(
          this, pid, handle1, handle2, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_intersection(
        IndexSpace parent, IndexPartition partition, PartitionKind kind,
        Color color, bool dominates, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_INTERSECTION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(partition, "partition");
        hasher.hash(kind, "kind");
        hasher.hash(color, "color");
        hasher.hash(dominates, "dominates");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      if (parent.get_type_tag() != partition.get_type_tag())
      {
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        error << "IndexPartition " << partition
              << " does not have the same type as the parent index space "
              << parent << " in task " << get_task_name() << " (UID "
              << get_unique_id() << ").";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      IndexPartNode* part_node = runtime->get_node(partition);
      // See if we can determine disjointness if we weren't told
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (part_node->is_disjoint(true /*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, part_node->color_space->handle, provenance,
          kind, partition_color, color_generated);
      part_op->initialize_intersection_partition(
          this, pid, partition, dominates, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_difference(
        IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
        IndexSpace color_space, PartitionKind kind, Color color,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_DIFFERENCE, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(handle1, "handle1");
        hasher.hash(handle2, "handle2");
        hasher.hash(color_space, "color_space");
        hasher.hash(kind, "kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      if (parent.get_tree_id() != handle1.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1
              << " is not part of the same index tree as IndexSpace " << parent
              << " in create partition by difference.";
        error.raise();
      }
      if (parent.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle2
              << " is not part of the same index tree as IndexSpace " << parent
              << " in create partition by difference.";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      else
        color_generated = true;
      // If the left-hand-side is disjoint the result is disjoint
      if ((kind == LEGION_COMPUTE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p1 = runtime->get_node(handle1);
        if (p1->is_disjoint(true /*from app*/))
        {
          if (kind == LEGION_COMPUTE_KIND)
            kind = LEGION_DISJOINT_KIND;
          else if (kind == LEGION_COMPUTE_COMPLETE_KIND)
            kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, kind, partition_color,
          color_generated);
      part_op->initialize_difference_partition(
          this, pid, handle1, handle2, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    Color ReplicateContext::create_cross_product_partitions(
        IndexPartition handle1, IndexPartition handle2,
        std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
        Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_CROSS_PRODUCT_PARTITIONS, __func__);
        hasher.hash(handle1, "handle1");
        hasher.hash(handle2, "handle2");
        hasher.hash(kind, "kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, kind);
      if (handle1.get_tree_id() != handle2.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "IndexPartition " << handle1
              << " is not part of the same index tree as IndexPartition "
              << handle2 << " in create cross product partitions.";
        error.raise();
      }
      LegionColor partition_color = INVALID_COLOR;
      if (color != LEGION_AUTO_GENERATE_ID)
        partition_color = color;
      std::set<RtEvent> safe_events;
      // We need an owner node to decide which color everyone is going to use
      if (owner_shard->shard_id == index_partition_allocator_shard)
      {
        // Do the call on the owner node
        if (partition_color == INVALID_COLOR)
        {
          ValueBroadcast<LegionColor> color_collective(
              this, index_partition_allocator_shard, COLLECTIVE_LOC_15);
          create_pending_cross_product_internal(
              handle1, handle2, handles, kind, provenance, partition_color,
              safe_events, owner_shard->shard_id, &shard_manager->get_mapping(),
              &color_collective);
        }
        else
          create_pending_cross_product_internal(
              handle1, handle2, handles, kind, provenance, partition_color,
              safe_events, owner_shard->shard_id,
              &shard_manager->get_mapping());
      }
      else
      {
        // Get the color result from the owner node
        if (partition_color == INVALID_COLOR)
        {
          ValueBroadcast<LegionColor> color_collective(
              this, index_partition_allocator_shard, COLLECTIVE_LOC_15);
          partition_color = color_collective.get_value();
          legion_assert(partition_color != INVALID_COLOR);
        }
        // Now we can do the call from this node
        create_pending_cross_product_internal(
            handle1, handle2, handles, kind, provenance, partition_color,
            safe_events, owner_shard->shard_id, &shard_manager->get_mapping());
      }
      // Signal that we're done with our creation
      RtEvent safe_event;
      if (!safe_events.empty())
        safe_event = Runtime::merge_events(safe_events);
      const RtBarrier creation_bar = creation_barrier.next(this);
      runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, safe_event);
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      part_op->initialize_cross_product(
          this, handle1, handle2, partition_color, provenance,
          owner_shard->shard_id, &shard_manager->get_mapping());
      // Also have to wait for creation to finish on all shards because
      // any shard can handle requests for any cross-product partition
      creation_bar.wait();
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      // Perform the exchange of all the handle names so that we can record
      // all the valid partition names here in this context
      {
        CrossProductCollective collective(this, COLLECTIVE_LOC_36);
        collective.exchange_partitions(handles);
        // Record these partition handles that were created
        AutoLock priv_lock(privilege_lock);
        for (const std::map<IndexSpace, IndexPartition>::value_type& it :
             handles)
        {
          legion_assert(
              (created_index_partitions.find(it.second) ==
               created_index_partitions.end()) ||
              (created_index_partitions[it.second] == 1));
          created_index_partitions[it.second] = 1;
        }
      }
      // Update our allocation shard
      index_partition_allocator_shard++;
      if (index_partition_allocator_shard == total_shards)
        index_partition_allocator_shard = 0;
      if (runtime->verify_partitions)
      {
        Domain color_space = runtime->get_index_partition_color_space(handle1);
        // This code will only work if the color space has type coord_t
        TypeTag type_tag = 0;
        switch (color_space.get_dim())
        {
#define DIMFUNC(DIM)                                            \
  case DIM:                                                     \
    {                                                           \
      type_tag = NT_TemplateHelper::encode_tag<DIM, coord_t>(); \
      break;                                                    \
    }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        for (Domain::DomainPointIterator itr(color_space); itr; itr++)
        {
          IndexSpace subspace;
          switch (color_space.get_dim())
          {
#define DIMFUNC(DIM)                                                 \
  case DIM:                                                          \
    {                                                                \
      const Point<DIM, coord_t> p(itr.p);                            \
      subspace = runtime->get_index_subspace(handle1, &p, type_tag); \
      break;                                                         \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
          IndexPartition part =
              runtime->get_index_partition(subspace, partition_color);
          verify_partition(part, verify_kind, __func__);
        }
      }
      return partition_color;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::create_association(
        LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
        IndexSpace range, MapperID id, MappingTagID tag,
        const UntypedBuffer& marg, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_ASSOCIATION, __func__);
        hasher.hash(domain, "domain");
        hasher.hash(domain_parent, "domain_parent");
        hasher.hash(domain_fid, "domain_fid");
        hasher.hash(range, "range");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_37));
      part_op->initialize_by_association(
          this, domain, domain_parent, domain_fid, range, id, tag, marg,
          provenance);
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping physical regions "
                     "around create_association call in task "
                  << get_task_name() << " (UID " << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_restricted_partition(
        IndexSpace parent, IndexSpace color_space, const void* transform,
        size_t transform_size, const void* extent, size_t extent_size,
        PartitionKind part_kind, Color color, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_RESTRICTED_PARTITION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(color_space, "color_space");
        hasher.hash(transform, transform_size, "transform_size");
        hasher.hash(extent, extent_size, "extent_size");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, part_kind, part_color,
          color_generated);
      part_op->initialize_restricted_partition(
          this, pid, transform, transform_size, extent, extent_size,
          provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_domain(
        IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
        bool perform_intersections, PartitionKind part_kind, Color color,
        Provenance* provenance, bool skip_check)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !skip_check && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_DOMAIN, __func__);
        hasher.hash(parent, "parent");
        hash_future_map(hasher, domains, "domains");
        hasher.hash(color_space, "color_space");
        hasher.hash(perform_intersections, "perform_intersections");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, part_kind, part_color,
          color_generated);
      part_op->initialize_by_domain(
          this, pid, domains, perform_intersections, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_field(
        LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
        IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
        PartitionKind part_kind, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_FIELD, __func__);
        hasher.hash(handle, "handle");
        hasher.hash(parent_priv, "parent_priv");
        hasher.hash(fid, "fid");
        hasher.hash(color_space, "color_space");
        hasher.hash(color, "color");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hasher.hash(part_kind, "part_kind");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      // Partition by field is disjoint by construction
      PartitionKind verify_kind = LEGION_DISJOINT_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      IndexSpace parent = handle.get_index_space();
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      create_shard_partition(
          part_op, pid, parent, color_space, provenance, part_kind, part_color,
          color_generated);
      part_op->initialize_by_field(
          this, pid, handle, parent_priv, color_space, fid, id, tag, marg,
          provenance);
      part_op->initialize_replication(this);
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_38));
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_field call "
              "in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_image(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_IMAGE, __func__);
        hasher.hash(handle, "handle");
        hasher.hash(projection, "projection");
        hasher.hash(parent, "parent");
        hasher.hash(fid, "fid");
        hasher.hash(color_space, "color_space");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, handle.get_tree_id(), handle.get_type_tag());
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      create_shard_partition(
          part_op, pid, handle, color_space, provenance, part_kind, part_color,
          color_generated);
      part_op->initialize_by_image(
          this, pid, handle, projection, parent, fid, id, tag, marg,
          provenance);
      part_op->initialize_replication(this);
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_39));
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_image call "
              "in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_image_range(
        IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_IMAGE_RANGE, __func__);
        hasher.hash(handle, "handle");
        hasher.hash(projection, "projection");
        hasher.hash(parent, "parent");
        hasher.hash(fid, "fid");
        hasher.hash(color_space, "color_space");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, handle.get_tree_id(), handle.get_type_tag());
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      create_shard_partition(
          part_op, pid, handle, color_space, provenance, part_kind, part_color,
          color_generated);
      part_op->initialize_by_image_range(
          this, pid, handle, projection, parent, fid, id, tag, marg,
          provenance);
      part_op->initialize_replication(this);
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_40));
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_image_range call "
              "in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_preimage(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_PREIMAGE, __func__);
        hasher.hash(projection, "projection");
        hasher.hash(handle, "handle");
        hasher.hash(parent, "parent");
        hasher.hash(fid, "fid");
        hasher.hash(color_space, "color_space");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      // If the source of the preimage is disjoint then the result is disjoint
      // Note this only applies here and not to range
      if ((part_kind == LEGION_COMPUTE_KIND) ||
          (part_kind == LEGION_COMPUTE_COMPLETE_KIND) ||
          (part_kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        IndexPartNode* p = runtime->get_node(projection);
        if (p->is_disjoint(true /*from app*/))
        {
          if (part_kind == LEGION_COMPUTE_KIND)
            part_kind = LEGION_DISJOINT_KIND;
          else if (part_kind == LEGION_COMPUTE_COMPLETE_KIND)
            part_kind = LEGION_DISJOINT_COMPLETE_KIND;
          else
            part_kind = LEGION_DISJOINT_INCOMPLETE_KIND;
        }
      }
      IndexPartition pid(
          0 /*temp*/, handle.get_index_space().get_tree_id(),
          handle.get_type_tag());
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      create_shard_partition(
          part_op, pid, handle.get_index_space(), color_space, provenance,
          part_kind, part_color, color_generated);
      part_op->initialize_by_preimage(
          this, pid, projection, handle, parent, fid, id, tag, marg,
          provenance);
      part_op->initialize_replication(this);
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_41));
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_preimage call "
              "in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_partition_by_preimage_range(
        IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
        FieldID fid, IndexSpace color_space, PartitionKind part_kind,
        Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PARTITION_BY_PREIMAGE_RANGE, __func__);
        hasher.hash(projection, "projection");
        hasher.hash(handle, "handle");
        hasher.hash(parent, "parent");
        hasher.hash(fid, "fid");
        hasher.hash(color_space, "color_space");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        hasher.hash(id, "id");
        hasher.hash(tag, "tag");
        hash_argument(hasher, runtime->safe_control_replication, marg, "marg");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, handle.get_index_space().get_tree_id(),
          handle.get_type_tag());
      ReplDependentPartitionOp* part_op =
          runtime->get_operation<ReplDependentPartitionOp>();
      create_shard_partition(
          part_op, pid, handle.get_index_space(), color_space, provenance,
          part_kind, part_color, color_generated);
      part_op->initialize_by_preimage_range(
          this, pid, projection, handle, parent, fid, id, tag, marg,
          provenance);
      part_op->initialize_replication(this);
      if (runtime->safe_mapper)
        part_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_42));
      // Now figure out if we need to unmap and re-map any inline mappings
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(part_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around create_partition_by_preimage_range call "
              "in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(part_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      if (runtime->verify_partitions)
        verify_partition(pid, verify_kind, __func__);
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexPartition ReplicateContext::create_pending_partition(
        IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
        Color color, Provenance* provenance, bool trust)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !trust && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_PENDING_PARTITION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(color_space, "color_space");
        hasher.hash(part_kind, "part_kind");
        hasher.hash(color, "color");
        if (hasher.verify(__func__))
          break;
      }
      PartitionKind verify_kind = LEGION_COMPUTE_KIND;
      if (runtime->verify_partitions && !trust)
        SWAP_PART_KINDS(verify_kind, part_kind);
      LegionColor part_color = INVALID_COLOR;
      bool color_generated = false;
      if (color != LEGION_AUTO_GENERATE_ID)
        part_color = color;
      else
        color_generated = true;
      IndexPartition pid(
          0 /*temp*/, parent.get_tree_id(), parent.get_type_tag());
      create_shard_partition(
          nullptr /*op*/, pid, parent, color_space, provenance, part_kind,
          part_color, color_generated);
      if (runtime->verify_partitions && !trust)
      {
        // We can't block to check this here because the user needs
        // control back in order to fill in the pieces of the partitions
        // so just launch a meta-task to check it when we can
        VerifyPartitionArgs args(this, pid, verify_kind, __func__);
        runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
      }
      return pid;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE_UNION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(realm_color, color_size, "realm_color");
        hasher.hash(type_tag, "type_tag");
        for (const IndexSpace& it : handles) hasher.hash(it, "handles");
        if (hasher.verify(__func__))
          break;
      }
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_union(this, result, handles, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_union(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE_UNION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(realm_color, color_size, "realm_color");
        hasher.hash(type_tag, "type_tag");
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_union(this, result, handle, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, const std::vector<IndexSpace>& handles,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE_INTERSECTION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(realm_color, color_size, "realm_color");
        hasher.hash(type_tag, "type_tag");
        for (const IndexSpace& it : handles) hasher.hash(it, "handles");
        if (hasher.verify(__func__))
          break;
      }
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_intersection(
          this, result, handles, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_intersection(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexPartition handle, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE_INTERSECTION, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(realm_color, color_size, "realm_color");
        hasher.hash(type_tag, "type_tag");
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_intersection(
          this, result, handle, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace ReplicateContext::create_index_space_difference(
        IndexPartition parent, const void* realm_color, size_t color_size,
        TypeTag type_tag, IndexSpace initial,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_INDEX_SPACE_DIFFERENCE, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(realm_color, color_size, "realm_color");
        hasher.hash(type_tag, "type_tag");
        hasher.hash(initial, "initial");
        for (const IndexSpace& it : handles) hasher.hash(it, "handles");
        if (hasher.verify(__func__))
          break;
      }
      ReplPendingPartitionOp* part_op =
          runtime->get_operation<ReplPendingPartitionOp>();
      IndexSpace result = instantiate_subspace(parent, realm_color, type_tag);
      part_op->initialize_index_space_difference(
          this, result, initial, handles, provenance);
      // Now we can add the operation to the queue
      add_to_dependence_queue(part_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::verify_partition(
        IndexPartition pid, PartitionKind kind, const char* function_name)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(pid);
      // Check containment first
      for (ColorSpaceIterator itr(node, true /*local only*/); itr; itr++)
      {
        IndexSpaceNode* child_node = node->get_child(*itr);
        IndexSpaceExpression* diff =
            runtime->subtract_index_spaces(child_node, node->parent);
        if (!diff->is_empty())
        {
          const DomainPoint bad =
              node->color_space->delinearize_color_to_point(*itr);
          switch (bad.get_dim())
          {
            case 1:
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Call to partition function " << function_name
                      << " in " << get_task_name() << " (UID "
                      << get_unique_id()
                      << ") has non-dominated child sub-region at color ("
                      << bad[0] << ").";
                error.raise();
              }
            case 2:
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Call to partition function " << function_name
                      << " in " << get_task_name() << " (UID "
                      << get_unique_id()
                      << ") has non-dominated child sub-region at color ("
                      << bad[0] << "," << bad[1] << ").";
                error.raise();
              }
            case 3:
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Call to partition function " << function_name
                      << " in " << get_task_name() << " (UID "
                      << get_unique_id()
                      << ") has non-dominated child sub-region at color ("
                      << bad[0] << "," << bad[1] << "," << bad[2] << ").";
                error.raise();
              }
            default:
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Call to partition function " << function_name
                      << " in " << get_task_name() << " (UID "
                      << get_unique_id()
                      << ") has non-dominated child sub-region at color " << bad
                      << ".";
                error.raise();
              }
          }
        }
      }
      // Only need to do the rest of this on shard 0
      if (owner_shard->shard_id > 0)
        return;
      // Check disjointness
      if ((kind == LEGION_DISJOINT_KIND) ||
          (kind == LEGION_DISJOINT_COMPLETE_KIND) ||
          (kind == LEGION_DISJOINT_INCOMPLETE_KIND))
      {
        if (!node->is_disjoint(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_DISJOINT_KIND) ? "DISJOINT_KIND" :
                    (kind == LEGION_DISJOINT_COMPLETE_KIND) ?
                                                     "DISJOINT_COMPLETE_KIND" :
                                                     "DISJOINT_INCOMPLETE_KIND")
                << " but the partition is aliased.";
          error.raise();
        }
      }
      else if (
          (kind == LEGION_ALIASED_KIND) ||
          (kind == LEGION_ALIASED_COMPLETE_KIND) ||
          (kind == LEGION_ALIASED_INCOMPLETE_KIND))
      {
        if (node->is_disjoint(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_ALIASED_KIND) ? "ALIASED_KIND" :
                    (kind == LEGION_ALIASED_COMPLETE_KIND) ?
                                                    "ALIASED_COMPLETE_KIND" :
                                                    "ALIASED_INCOMPLETE_KIND")
                << " but the partition is disjoint.";
          error.raise();
        }
      }
      // Check completeness
      if ((kind == LEGION_DISJOINT_COMPLETE_KIND) ||
          (kind == LEGION_ALIASED_COMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_COMPLETE_KIND))
      {
        if (!node->is_complete(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_DISJOINT_COMPLETE_KIND) ?
                        "DISJOINT_COMPLETE_KIND" :
                    (kind == LEGION_ALIASED_COMPLETE_KIND) ?
                        "ALIASED_COMPLETE_KIND" :
                        "COMPUTE_COMPLETE_KIND")
                << " but the partition is incomplete.";
          error.raise();
        }
      }
      else if (
          (kind == LEGION_DISJOINT_INCOMPLETE_KIND) ||
          (kind == LEGION_ALIASED_INCOMPLETE_KIND) ||
          (kind == LEGION_COMPUTE_INCOMPLETE_KIND))
      {
        if (node->is_complete(true /*from application*/))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Call to partitioning function " << function_name << " in "
                << get_task_name() << " (UID " << get_unique_id()
                << ") specified partition was "
                << ((kind == LEGION_DISJOINT_INCOMPLETE_KIND) ?
                        "DISJOINT_INCOMPLETE_KIND" :
                    (kind == LEGION_ALIASED_INCOMPLETE_KIND) ?
                        "ALIASED_INCOMPLETE_KIND" :
                        "COMPUTE_INCOMPLETE_KIND")
                << " but the partition is complete.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    FieldSpace ReplicateContext::create_field_space(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_FIELD_SPACE, __func__);
        if (hasher.verify(__func__))
          break;
      }
      return create_replicated_field_space(provenance);
    }

    //--------------------------------------------------------------------------
    FieldSpace ReplicateContext::create_replicated_field_space(
        Provenance* provenance, ShardID* creator)
    //--------------------------------------------------------------------------
    {
      // Seed this with the first field space broadcast
      if (pending_field_spaces.empty())
      {
        increase_pending_field_spaces(1 /*count*/, false /*double*/);
        pending_field_space_check = 0;
      }
      FieldSpace space;
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<FSBroadcast>*, bool>& collective =
          pending_field_spaces.front();
      if (creator != nullptr)
        *creator = collective.first->origin;
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const FSBroadcast value = collective.first->get_value(false);
        space = FieldSpace(value.did);
        double_buffer = value.double_buffer;
        // Need to register this before broadcasting
        runtime->create_node(
            space, creation_bar, provenance, &collective_mapping);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_field_space(value.did);
        LegionSpy::log_field_space(
            space.get_id(), runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const FSBroadcast value = collective.first->get_value(false);
        space = FieldSpace(value.did);
        double_buffer = value.double_buffer;
        legion_assert(space.exists());
        runtime->create_node(
            space, creation_bar, provenance, &collective_mapping);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      if (++pending_field_space_check == pending_field_spaces.size())
        pending_field_space_check = 0;
      else
        double_buffer = false;
      // Record this in our context
      register_field_space_creation(space);
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_field_spaces(
          double_buffer ? pending_field_spaces.size() + 1 : 1, double_next);
      delete collective.first;
      pending_field_spaces.pop_front();
      return space;
    }

    //--------------------------------------------------------------------------
    FieldSpace ReplicateContext::create_field_space(
        const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_FIELD_SPACE, __func__);
        for (const size_t& it : sizes) hasher.hash(it, "sizes");
        for (const FieldID& it : resulting_fields)
          hasher.hash(it, "resulting_fields");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify(__func__))
          break;
      }
      ShardID creator_shard = 0;
      const FieldSpace space =
          create_replicated_field_space(provenance, &creator_shard);
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
        {
          if (pending_fields.empty())
          {
            increase_pending_fields(1 /*count*/, false /*double*/);
            pending_field_check = 0;
          }
          bool double_next = false;
          bool double_buffer = false;
          std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
              pending_fields.front();
          if (collective.second)
          {
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          else
          {
            const RtEvent done = collective.first->get_done_event();
            if (!done.has_triggered())
            {
              double_next = true;
              done.wait();
            }
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          if (++pending_field_check == pending_fields.size())
            pending_field_check = 0;
          else
            double_buffer = false;
          increase_pending_fields(
              double_buffer ? pending_fields.size() + 1 : 1, double_next);
          delete collective.first;
          pending_fields.pop_front();
        }
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << get_task_name() << " (ID " << get_unique_id()
                << ") attempted to allocate a field with ID "
                << resulting_fields[idx]
                << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound "
                   "set in legion_config.h.";
          error.raise();
        }
      }
      // Figure out if we're going to do the field initialization on this node
      const AddressSpaceID owner_space = FieldSpaceNode::get_owner_space(space);
      const bool local_shard =
          (owner_space == runtime->address_space) ?
              (creator_shard == owner_shard->shard_id) :
              shard_manager->is_first_local_shard(owner_shard);
      const RtBarrier creation_bar = creation_barrier.next(this);
      // This deduplicates multiple shards on the same node
      if (local_shard)
      {
        const bool non_owner = (creator_shard != owner_shard->shard_id);
        FieldSpaceNode* node = runtime->get_node(space);
        node->initialize_fields(
            sizes, resulting_fields, serdez_id, provenance,
            true /*collective*/);
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        if (!non_owner)
          for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
            LegionSpy::log_field_creation(
                space.get_id(), resulting_fields[idx], sizes[idx],
                (provenance == nullptr) ? std::string_view() :
                                          provenance->human);
      }
      else
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      register_all_field_creations(space, false /*loca*/, resulting_fields);
      // Make sure all the field allocations are done on all shards
      creation_bar.wait();
      return space;
    }

    //--------------------------------------------------------------------------
    FieldSpace ReplicateContext::create_field_space(
        const std::vector<Future>& sizes,
        std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_FIELD_SPACE, __func__);
        for (const Future& it : sizes)
          hash_future(hasher, runtime->safe_control_replication, it, "sizes");
        for (const FieldID& it : resulting_fields)
          hasher.hash(it, "resulting_fields");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify(__func__))
          break;
      }
      ShardID creator_shard = 0;
      const FieldSpace space =
          create_replicated_field_space(provenance, &creator_shard);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
        {
          if (pending_fields.empty())
          {
            increase_pending_fields(1 /*count*/, false /*double*/);
            pending_field_check = 0;
          }
          bool double_next = false;
          bool double_buffer = false;
          std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
              pending_fields.front();
          if (collective.second)
          {
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          else
          {
            const RtEvent done = collective.first->get_done_event();
            if (!done.has_triggered())
            {
              double_next = true;
              done.wait();
            }
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          if (++pending_field_check == pending_fields.size())
            pending_field_check = 0;
          else
            double_buffer = false;
          increase_pending_fields(
              double_buffer ? pending_fields.size() + 1 : 1, double_next);
          delete collective.first;
          pending_fields.pop_front();
        }
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << get_task_name() << " (ID " << get_unique_id()
                << ") attempted to allocate a field with ID "
                << resulting_fields[idx]
                << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound "
                   "set in legion_config.h.";
          error.raise();
        }
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
        if (sizes[idx].impl == nullptr)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid empty future passed to field allocation for field "
                << resulting_fields[idx] << " in task " << get_task_name()
                << " (UID " << get_unique_id() << ").";
          error.raise();
        }
      // Figure out if we're going to do the field initialization on this node
      const AddressSpaceID owner_space = FieldSpaceNode::get_owner_space(space);
      const bool local_shard =
          (owner_space == runtime->address_space) ?
              (creator_shard == owner_shard->shard_id) :
              shard_manager->is_first_local_shard(owner_shard);
      const RtBarrier creation_bar = creation_barrier.next(this);
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      FieldSpaceNode* node = runtime->get_node(space);
      // This deduplicates multiple shards on the same node
      if (local_shard)
      {
        const ApEvent ready = creator_op->get_completion_event();
        const bool owner = (creator_shard == owner_shard->shard_id);
        node->initialize_fields(
            ready, resulting_fields, serdez_id, provenance,
            true /*collective*/);
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        creator_op->initialize_fields(
            this, node, resulting_fields, sizes, provenance, owner);
      }
      else
      {
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        creator_op->initialize_fields(
            this, node, resulting_fields, sizes, provenance, false /*owner*/);
      }
      register_all_field_creations(space, false /*local*/, resulting_fields);
      // Make sure the field IDs are valid everywhere
      creation_bar.wait();
      // Launch the creation op in this context to act as a fence to ensure
      // that the allocations are done on all shard nodes before anyone else
      // tries to use them or their meta-data
      add_to_dependence_queue(creator_op);
      return space;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_field_spaces(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<FSBroadcast>* collective =
            new ValueBroadcast<FSBroadcast>(
                this, field_space_allocator_shard, COLLECTIVE_LOC_31);
        if (owner_shard->shard_id == field_space_allocator_shard)
        {
          const DistributedID did = runtime->get_unique_field_space_id();
          // We're the owner, so make it locally and then broadcast it
          runtime->record_pending_field_space(did);
          // Do our arrival on this generation, should be the last one
          collective->broadcast(FSBroadcast(did, double_next));
          pending_field_spaces.emplace_back(
              std::pair<ValueBroadcast<FSBroadcast>*, bool>(collective, true));
        }
        else
        {
          register_collective(collective);
          pending_field_spaces.emplace_back(
              std::pair<ValueBroadcast<FSBroadcast>*, bool>(collective, false));
        }
        field_space_allocator_shard++;
        if (field_space_allocator_shard == total_shards)
          field_space_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::create_shared_ownership(FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return;
      if (shard_manager->is_total_sharding() &&
          shard_manager->is_first_local_shard(owner_shard))
        runtime->create_shared_ownership(handle, true /*total sharding*/);
      else if (owner_shard->shard_id == 0)
        runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<FieldSpace, unsigned>::iterator finder =
          created_field_spaces.find(handle);
      if (finder != created_field_spaces.end())
        finder->second++;
      else
        created_field_spaces[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_field_space(
        FieldSpace handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DESTROY_FIELD_SPACE, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock);
        std::map<FieldSpace, unsigned>::iterator finder =
            created_field_spaces.find(handle);
        if (finder != created_field_spaces.end())
        {
          legion_assert(finder->second > 0);
          if (--finder->second == 0)
            created_field_spaces.erase(finder);
          else
            return;
          // Count how many regions are still using this field space
          // that still need to be deleted before we can remove the
          // list of created fields
          std::set<LogicalRegion> latent_regions;
          for (const std::map<LogicalRegion, unsigned>::value_type& it :
               created_regions)
            if (it.first.get_field_space() == handle)
              latent_regions.insert(it.first);
          for (const std::map<LogicalRegion, bool>::value_type& it :
               local_regions)
            if (it.first.get_field_space() == handle)
              latent_regions.insert(it.first);
          if (latent_regions.empty())
          {
            // No remaining regions so we can remove any created fields now
            for (std::set<std::pair<FieldSpace, FieldID> >::iterator it =
                     created_fields.begin();
                 it != created_fields.end();
                 /*nothing*/)
            {
              if (it->first == handle)
              {
                std::set<std::pair<FieldSpace, FieldID> >::iterator to_delete =
                    it++;
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
          // If we didn't make this field space, record the deletion
          // and keep going. It will be handled by the context that
          // made the field space
          deleted_field_spaces.emplace_back(handle);
          return;
        }
      }
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_field_space_deletion(this, handle, unordered, provenance);
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal unordered field space deletion performed after task "
              << *this
              << " has finished executing. All unordered operations must "
              << "be performed before the end of the execution of the parent "
                 "task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    FieldID ReplicateContext::allocate_field(
        FieldSpace space, size_t field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ALLOCATE_FIELD, __func__);
        hasher.hash(space, "space");
        hasher.hash(field_size, "field_size");
        hasher.hash(fid, "fid");
        hasher.hash(local, "local");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify(__func__))
          break;
      }
      if (local)
      {
        Fatal fatal;
        fatal << "Local field creation is not currently supported "
              << "for control replication with task " << *this << ".";
        fatal.raise();
      }
      if (fid == LEGION_AUTO_GENERATE_ID)
      {
        if (pending_fields.empty())
        {
          increase_pending_fields(1 /*count*/, false /*double*/);
          pending_field_check = 0;
        }
        bool double_next = false;
        bool double_buffer = false;
        std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
            pending_fields.front();
        if (collective.second)
        {
          const FIDBroadcast value = collective.first->get_value(false);
          fid = value.field_id;
          double_buffer = value.double_buffer;
        }
        else
        {
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
          {
            double_next = true;
            done.wait();
          }
          const FIDBroadcast value = collective.first->get_value(false);
          fid = value.field_id;
          double_buffer = value.double_buffer;
        }
        if (++pending_field_check == pending_fields.size())
          pending_field_check = 0;
        else
          double_buffer = false;
        increase_pending_fields(
            double_buffer ? pending_fields.size() + 1 : 1, double_next);
        delete collective.first;
        pending_fields.pop_front();
      }
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task " << *this << " attempted to allocate a field with "
              << "ID " << fid << " which exceeds the "
              << "LEGION_MAX_APPLICATION_FIELD_ID bound set in legion_config.h";
        error.raise();
      }
      std::map<FieldSpace, std::pair<ShardID, bool> >::const_iterator finder =
          field_allocator_owner_shards.find(space);
      legion_assert(finder != field_allocator_owner_shards.end());
      RtEvent precondition;
      // This deduplicates multiple shards on the same node
      if (finder->second.second)
      {
        const bool non_owner = (finder->second.first != owner_shard->shard_id);
        precondition = runtime->allocate_field(
            space, field_size, fid, serdez_id, provenance, non_owner);
        if (!non_owner)
          LegionSpy::log_field_creation(
              space.get_id(), fid, field_size,
              (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      const RtBarrier creation_bar = creation_barrier.next(this);
      runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, precondition);
      register_field_creation(space, fid, local);
      // Make sure the field IDs are valid everywhere
      creation_bar.wait();
      return fid;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_fields(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<FIDBroadcast>* collective =
            new ValueBroadcast<FIDBroadcast>(
                this, field_allocator_shard, COLLECTIVE_LOC_33);
        if (owner_shard->shard_id == field_allocator_shard)
        {
          const FieldID fid = runtime->get_unique_field_id();
          // Do our arrival on this generation, should be the last one
          collective->broadcast(FIDBroadcast(fid, double_next));
          pending_fields.emplace_back(
              std::pair<ValueBroadcast<FIDBroadcast>*, bool>(collective, true));
        }
        else
        {
          register_collective(collective);
          pending_fields.emplace_back(
              std::pair<ValueBroadcast<FIDBroadcast>*, bool>(
                  collective, false));
        }
        field_allocator_shard++;
        if (field_allocator_shard == total_shards)
          field_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    FieldID ReplicateContext::allocate_field(
        FieldSpace space, const Future& field_size, FieldID fid, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ALLOCATE_FIELD, __func__);
        hasher.hash(space, "space");
        hash_future(
            hasher, runtime->safe_control_replication, field_size,
            "field_size");
        hasher.hash(fid, "fid");
        hasher.hash(local, "local");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify(__func__))
          break;
      }
      if (local)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Local field creation is not currently supported "
              << "for control replication with task " << *this << ".";
        error.raise();
      }
      if (fid == LEGION_AUTO_GENERATE_ID)
      {
        if (pending_fields.empty())
        {
          increase_pending_fields(1 /*count*/, false /*double*/);
          pending_field_check = 0;
        }
        bool double_next = false;
        bool double_buffer = false;
        std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
            pending_fields.front();
        if (collective.second)
        {
          const FIDBroadcast value = collective.first->get_value(false);
          fid = value.field_id;
          double_buffer = value.double_buffer;
        }
        else
        {
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
          {
            double_next = true;
            done.wait();
          }
          const FIDBroadcast value = collective.first->get_value(false);
          fid = value.field_id;
          double_buffer = value.double_buffer;
        }
        if (++pending_field_check == pending_fields.size())
          pending_field_check = 0;
        else
          double_buffer = false;
        increase_pending_fields(
            double_buffer ? pending_fields.size() + 1 : 1, double_next);
        delete collective.first;
        pending_fields.pop_front();
      }
      else if (fid >= LEGION_MAX_APPLICATION_FIELD_ID)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task " << *this << " attempted to allocate a field with "
              << "ID " << fid << " which exceeds the "
              << "LEGION_MAX_APPLICATION_FIELD_ID "
              << "bound set in legion_config.h";
        error.raise();
      }
      if (field_size.impl == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid empty future passed to field allocation for field "
              << fid << " in task " << *this << ".";
        error.raise();
      }
      std::map<FieldSpace, std::pair<ShardID, bool> >::const_iterator finder =
          field_allocator_owner_shards.find(space);
      legion_assert(finder != field_allocator_owner_shards.end());
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const RtBarrier creation_bar = creation_barrier.next(this);
      // This deduplicates multiple shards on the same node
      if (finder->second.second)
      {
        const ApEvent ready = creator_op->get_completion_event();
        const bool owner = (finder->second.first == owner_shard->shard_id);
        RtEvent precondition;
        FieldSpaceNode* node = runtime->allocate_field(
            space, ready, fid, serdez_id, provenance, precondition, !owner);
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, precondition);
        creator_op->initialize_field(
            this, node, fid, field_size, provenance, owner);
      }
      else
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      register_field_creation(space, fid, local);
      // Make sure the IDs are valid everywhere
      creation_bar.wait();
      if (!finder->second.second)
      {
        FieldSpaceNode* node = runtime->get_node(space);
        creator_op->initialize_field(
            this, node, fid, field_size, provenance, false /*owner*/);
      }
      // Launch the creation op in this context to act as a fence to ensure
      // that the allocations are done on all shard nodes before anyone else
      // tries to use them or their meta-data
      add_to_dependence_queue(creator_op);
      return fid;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::free_field(
        FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
        const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_FREE_FIELD, __func__);
        hasher.hash(space, "space");
        hasher.hash(fid, "fid");
        if (hasher.verify(__func__))
          break;
      }
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        const std::pair<FieldSpace, FieldID> key(space, fid);
        // This field will actually be removed in analyze_destroy_fields
        std::set<std::pair<FieldSpace, FieldID> >::const_iterator finder =
            created_fields.find(key);
        if (finder == created_fields.end())
        {
          std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
              local_finder = local_fields.find(key);
          if (local_finder == local_fields.end())
          {
            // If we didn't make this field, record the deletion and
            // then have a later context handle it
            deleted_fields.emplace_back(DeletedField(space, fid, provenance));
            return;
          }
          else
            local_finder->second = true;
        }
      }
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_field_deletion(
          this, space, fid, unordered, allocator, provenance,
          (owner_shard->shard_id != 0));
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal unordered field space deletion performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed before the end of the execution of the parent task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::allocate_fields(
        FieldSpace space, const std::vector<size_t>& sizes,
        std::vector<FieldID>& resulting_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ALLOCATE_FIELDS, __func__);
        hasher.hash(space, "space");
        for (const size_t& it : sizes) hasher.hash(it, "sizes");
        for (const FieldID& it : resulting_fields)
          hasher.hash(it, "resulting_fields");
        hasher.hash(local, "local");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify(__func__))
          break;
      }
      if (local)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Local field creation is not currently supported for control "
                 "replication with task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      if (resulting_fields.size() < sizes.size())
        resulting_fields.resize(sizes.size(), LEGION_AUTO_GENERATE_ID);
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
        {
          if (pending_fields.empty())
          {
            increase_pending_fields(1 /*count*/, false /*double*/);
            pending_field_check = 0;
          }
          bool double_next = false;
          bool double_buffer = false;
          std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
              pending_fields.front();
          if (collective.second)
          {
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          else
          {
            const RtEvent done = collective.first->get_done_event();
            if (!done.has_triggered())
            {
              double_next = true;
              done.wait();
            }
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          if (++pending_field_check == pending_fields.size())
            pending_field_check = 0;
          else
            double_buffer = false;
          increase_pending_fields(
              double_buffer ? pending_fields.size() + 1 : 1, double_next);
          delete collective.first;
          pending_fields.pop_front();
        }
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << get_task_name() << " (ID " << get_unique_id()
                << ") attempted to allocate a field with ID "
                << resulting_fields[idx]
                << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound "
                   "set in legion_config.h.";
          error.raise();
        }
      }
      std::map<FieldSpace, std::pair<ShardID, bool> >::const_iterator finder =
          field_allocator_owner_shards.find(space);
      legion_assert(finder != field_allocator_owner_shards.end());
      RtEvent precondition;
      // This deduplicates multiple shards on the same node
      if (finder->second.second)
      {
        const bool non_owner = (finder->second.first != owner_shard->shard_id);
        precondition = runtime->allocate_fields(
            space, sizes, resulting_fields, serdez_id, provenance, non_owner);
        if (!non_owner)
          for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
            LegionSpy::log_field_creation(
                space.get_id(), resulting_fields[idx], sizes[idx],
                (provenance == nullptr) ? std::string_view() :
                                          provenance->human);
      }
      const RtBarrier creation_bar = creation_barrier.next(this);
      runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, precondition);
      register_all_field_creations(space, local, resulting_fields);
      // Make sure all the field IDs are valid everywhere
      creation_bar.wait();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::allocate_fields(
        FieldSpace space, const std::vector<Future>& sizes,
        std::vector<FieldID>& resulting_fields, bool local,
        CustomSerdezID serdez_id, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ALLOCATE_FIELDS, __func__);
        hasher.hash(space, "space");
        for (const Future& it : sizes)
          hash_future(hasher, runtime->safe_control_replication, it, "sizes");
        for (const FieldID& it : resulting_fields)
          hasher.hash(it, "resulting_fields");
        hasher.hash(local, "local");
        hasher.hash(serdez_id, "serdez_id");
        if (hasher.verify("allocate_fields"))
          break;
      }
      if (local)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Local field creation is not currently supported for control "
                 "replication with task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      for (unsigned idx = 0; idx < resulting_fields.size(); idx++)
      {
        if (resulting_fields[idx] == LEGION_AUTO_GENERATE_ID)
        {
          if (pending_fields.empty())
          {
            increase_pending_fields(1 /*count*/, false /*double*/);
            pending_field_check = 0;
          }
          bool double_next = false;
          bool double_buffer = false;
          std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
              pending_fields.front();
          if (collective.second)
          {
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          else
          {
            const RtEvent done = collective.first->get_done_event();
            if (!done.has_triggered())
            {
              double_next = true;
              done.wait();
            }
            const FIDBroadcast value = collective.first->get_value(false);
            resulting_fields[idx] = value.field_id;
            double_buffer = value.double_buffer;
          }
          if (++pending_field_check == pending_fields.size())
            pending_field_check = 0;
          else
            double_buffer = false;
          increase_pending_fields(
              double_buffer ? pending_fields.size() + 1 : 1, double_next);
          delete collective.first;
          pending_fields.pop_front();
        }
        else if (resulting_fields[idx] >= LEGION_MAX_APPLICATION_FIELD_ID)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Task " << get_task_name() << " (ID " << get_unique_id()
                << ") attempted to allocate a field with ID "
                << resulting_fields[idx]
                << " which exceeds the LEGION_MAX_APPLICATION_FIELD_ID bound "
                   "set in legion_config.h.";
          error.raise();
        }
      }
      for (unsigned idx = 0; idx < sizes.size(); idx++)
      {
        if (sizes[idx].impl == nullptr)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid empty future passed to field allocation for field "
                << resulting_fields[idx] << " in task " << *this << ".";
          error.raise();
        }
      }
      std::map<FieldSpace, std::pair<ShardID, bool> >::const_iterator finder =
          field_allocator_owner_shards.find(space);
      legion_assert(finder != field_allocator_owner_shards.end());
      // Get a new creation operation
      CreationOp* creator_op = runtime->get_operation<CreationOp>();
      const RtBarrier creation_bar = creation_barrier.next(this);
      // This deduplicates multiple shards on the same node
      if (finder->second.second)
      {
        const ApEvent ready = creator_op->get_completion_event();
        const bool owner = (finder->second.first == owner_shard->shard_id);
        RtEvent precondition;
        FieldSpaceNode* node = runtime->allocate_fields(
            space, ready, resulting_fields, serdez_id, provenance, precondition,
            !owner);
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/, precondition);
        creator_op->initialize_fields(
            this, node, resulting_fields, sizes, provenance, owner);
      }
      else
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      register_all_field_creations(space, local, resulting_fields);
      // Make sure the field IDs are valid everywhere
      creation_bar.wait();
      if (!finder->second.second)
      {
        FieldSpaceNode* node = runtime->get_node(space);
        creator_op->initialize_fields(
            this, node, resulting_fields, sizes, provenance, false /*owner*/);
      }
      // Launch the creation op in this context to act as a fence to ensure
      // that the allocations are done on all shard nodes before anyone else
      // tries to use them or their meta-data
      add_to_dependence_queue(creator_op);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::free_fields(
        FieldAllocatorImpl* allocator, FieldSpace space,
        const std::set<FieldID>& to_free, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_FREE_FIELDS, __func__);
        hasher.hash(space, "space");
        for (const FieldID& it : to_free) hasher.hash(it, "to_free");
        if (hasher.verify(__func__))
          break;
      }
      std::set<FieldID> free_now;
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        // These fields will actually be removed in analyze_destroy_fields
        for (const FieldID& it : to_free)
        {
          const std::pair<FieldSpace, FieldID> key(space, it);
          std::set<std::pair<FieldSpace, FieldID> >::const_iterator finder =
              created_fields.find(key);
          if (finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
                local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
            {
              local_finder->second = true;
              free_now.insert(it);
            }
            else
              deleted_fields.emplace_back(DeletedField(space, it, provenance));
          }
          else
            free_now.insert(it);
        }
      }
      if (free_now.empty())
        return;
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_field_deletions(
          this, space, free_now, unordered, allocator, provenance,
          (owner_shard->shard_id != 0));
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal unordered free fields performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed before the end of the execution of the parent task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    LogicalRegion ReplicateContext::create_logical_region(
        IndexSpace index_space, FieldSpace field_space, const bool task_local,
        Provenance* provenance, const bool output_region)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CREATE_LOGICAL_REGION, __func__);
        hasher.hash(index_space, "index_space");
        hasher.hash(field_space, "field_space");
        hasher.hash(task_local, "task_local");
        if (hasher.verify(__func__))
          break;
      }
      // Seed this with the first field space broadcast
      if (pending_region_trees.empty())
      {
        increase_pending_region_trees(1 /*count*/, false /*double*/);
        pending_region_tree_check = 0;
      }
      LogicalRegion handle(0 /*temp*/, index_space, field_space);
      bool double_next = false;
      bool double_buffer = false;
      std::pair<ValueBroadcast<LRBroadcast>*, bool>& collective =
          pending_region_trees.front();
      CollectiveMapping& collective_mapping =
          shard_manager->get_collective_mapping();
      const RtBarrier creation_bar = creation_barrier.next(this);
      if (collective.second)
      {
        const LRBroadcast value = collective.first->get_value(false);
        handle.tree_did = value.tid;
        double_buffer = value.double_buffer;
        // Have to register this before doing the broadcast
        runtime->create_node(
            handle, nullptr /*parent*/, creation_bar, value.did, provenance,
            &collective_mapping);
        // Arrive on the creation barrier
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
        runtime->revoke_pending_region_tree(value.tid);
        LegionSpy::log_top_region(
            index_space.get_id(), field_space.get_id(), handle.get_tree_id(),
            runtime->address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
      }
      else
      {
        const RtEvent done = collective.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
        const LRBroadcast value = collective.first->get_value(false);
        handle.tree_did = value.tid;
        double_buffer = value.double_buffer;
        legion_assert(handle.exists());
        runtime->create_node(
            handle, nullptr /*parent*/, creation_bar, value.did, provenance,
            &collective_mapping);
        // Signal that we are done our creation
        runtime->phase_barrier_arrive(creation_bar, 1 /*count*/);
      }
      // Register the creation of a top-level region with the context
      const unsigned created_index =
          register_region_creation(handle, task_local, output_region);
      if (output_region)
      {
        // If this is an output region make sure nobody tries to compute
        // the equivalence sets for it until we know it is ready
        AutoLock priv_lock(privilege_lock);
        legion_assert(
            equivalence_set_trees.find(created_index) ==
            equivalence_set_trees.end());
        legion_assert(
            pending_equivalence_set_trees.find(created_index) ==
            pending_equivalence_set_trees.end());
        // Put in a guard so that nobody else tries to make it
        pending_equivalence_set_trees[created_index] =
            RtUserEvent::NO_RT_USER_EVENT;
      }
      if (++pending_region_tree_check == pending_region_trees.size())
        pending_region_tree_check = 0;
      else
        double_buffer = false;
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_region_trees(
          double_buffer ? pending_region_trees.size() + 1 : 1, double_next);
      delete collective.first;
      pending_region_trees.pop_front();
      return handle;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_region_trees(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<LRBroadcast>* collective =
            new ValueBroadcast<LRBroadcast>(
                this, logical_region_allocator_shard, COLLECTIVE_LOC_34);
        if (owner_shard->shard_id == logical_region_allocator_shard)
        {
          const RegionTreeID tid = runtime->get_unique_region_tree_id();
          const DistributedID did = runtime->get_available_distributed_id();
          // We're the owner, so make it locally and then broadcast it
          runtime->record_pending_region_tree(tid);
          // Do our arrival on this generation, should be the last one
          collective->broadcast(LRBroadcast(tid, did, double_next));
          pending_region_trees.emplace_back(
              std::pair<ValueBroadcast<LRBroadcast>*, bool>(collective, true));
        }
        else
        {
          register_collective(collective);
          pending_region_trees.emplace_back(
              std::pair<ValueBroadcast<LRBroadcast>*, bool>(collective, false));
        }
        logical_region_allocator_shard++;
        if (logical_region_allocator_shard == total_shards)
          logical_region_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::create_shared_ownership(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CREATE_SHARED_OWNERSHIP, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify("create_shared_ownership"))
          break;
      }
      if (!handle.exists())
        return;
      if (!runtime->is_top_level_region(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal call to create shared ownership for logical region "
            << handle << " in task " << get_task_name() << " (UID "
            << get_unique_id()
            << ") which is not a top-level logical region. Legion only permits "
            << "top-level logical regions to have shared ownerships.";
        error.raise();
      }
      if (shard_manager->is_total_sharding() &&
          shard_manager->is_first_local_shard(owner_shard))
        runtime->create_shared_ownership(handle, true /*total sharding*/);
      else if (owner_shard->shard_id == 0)
        runtime->create_shared_ownership(handle);
      AutoLock priv_lock(privilege_lock);
      std::map<LogicalRegion, unsigned>::iterator finder =
          created_regions.find(handle);
      if (finder != created_regions.end())
        finder->second++;
      else
        created_regions[handle] = 1;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_logical_region(
        LogicalRegion handle, const bool unordered, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DESTROY_LOGICAL_REGION, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      if (!handle.exists())
        return;
      // Check to see if this is a top-level logical region, if not then
      // we shouldn't even be destroying it
      if (!runtime->is_top_level_region(handle))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to destroy logical region " << handle
              << " in task " << get_task_name() << " (UID " << get_unique_id()
              << ") which is not a top-level logical region.";
        error.raise();
      }
      // Check to see if this is one that we should be allowed to destory
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<LogicalRegion, unsigned>::iterator finder =
            created_regions.find(handle);
        if (finder == created_regions.end())
        {
          // Check to see if it is a local region
          std::map<LogicalRegion, bool>::iterator local_finder =
              local_regions.find(handle);
          // Mark that this region is deleted, safe even though this
          // is a read-only lock because we're not changing the structure
          // of the map
          if (local_finder == local_regions.end())
          {
            // Record the deletion for later and propagate it up
            deleted_regions.emplace_back(DeletedRegion(handle, provenance));
            return;
          }
          else
            local_finder->second = true;
        }
        else
        {
          if (finder->second == 0)
          {
            Warning warning;
            warning << "Duplicate deletions were performed for region "
                    << handle << " in task tree rooted by " << get_task_name()
                    << ".";
            warning.raise();
            return;
          }
          if (--finder->second > 0)
            return;
          // Don't remove anything from created regions yet, we still might
          // need it as part of the logical dependence analysis for earlier
          // operations, but the reference count is zero so we're protected
        }
      }
      ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
      op->initialize_logical_region_deletion(
          this, handle, unordered, provenance);
      op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal unordered logical region deletion performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed "
            << "before the end of the execution of the parent task.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::reset_equivalence_sets(
        LogicalRegion parent, LogicalRegion region,
        const std::set<FieldID>& fields)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_RESET_EQUIVALENCE_SETS, __func__);
        hasher.hash(parent, "parent");
        hasher.hash(region, "region");
        for (const FieldID& it : fields) hasher.hash(it, "fields");
        if (hasher.verify(__func__))
          break;
      }
      // Ignore reset calls inside of traces replays
      if ((current_trace != nullptr) && current_trace->is_fixed())
      {
        Warning warning;
        warning << "Ignoring equivalence sets reset in " << get_task_name()
                << " (UID " << get_unique_id()
                << ") because it was made inside of a trace.";
        warning.raise();
        return;
      }
      if (fields.empty())
      {
        Warning warning;
        warning << "Ignoring equivalence sets reset in " << get_task_name()
                << " (UID " << get_unique_id() << ") because "
                << "it contains no fields.";
        warning.raise();
        return;
      }
      ReplResetOp* reset = runtime->get_operation<ReplResetOp>();
      reset->initialize(this, parent, region, fields);
      add_to_dependence_queue(reset);
    }

    //--------------------------------------------------------------------------
    FieldAllocatorImpl* ReplicateContext::create_field_allocator(
        FieldSpace handle, bool unordered)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CREATE_FIELD_ALLOCATOR, __func__);
        hasher.hash(handle, "handle");
        if (hasher.verify(__func__))
          break;
      }
      {
        AutoLock priv_lock(privilege_lock, false /*exclusive*/);
        std::map<FieldSpace, FieldAllocatorImpl*>::const_iterator finder =
            field_allocators.find(handle);
        if (finder != field_allocators.end())
          return finder->second;
      }
      FieldSpaceNode* node = runtime->get_node(handle);
      if (unordered)
      {
        // This next part is unsafe to perform in a control replicated
        // context if we are unordered, so just make a fresh allocator
        const RtEvent ready = node->create_allocator(runtime->address_space);
        // Don't have one so make a new one
        FieldAllocatorImpl* result = new FieldAllocatorImpl(node, this, ready);
        // DO NOT SAVE THIS!
        return result;
      }
      // Didn't find it, so have to make, retake the lock in exclusive mode
      AutoLock priv_lock(privilege_lock);
      // Check to see if we lost the race
      std::map<FieldSpace, FieldAllocatorImpl*>::const_iterator finder =
          field_allocators.find(handle);
      if (finder != field_allocators.end())
        return finder->second;
      // Check to see which shard (if any) owns this field space
      const AddressSpaceID owner_space =
          FieldSpaceNode::get_owner_space(handle);
      // Figure out which shard is the owner
      bool found = false;
      std::pair<ShardID, bool> owner(0, false);
      const ShardMapping& mapping = shard_manager->get_mapping();
      for (unsigned idx = 0; idx < mapping.size(); idx++)
      {
        if (mapping[idx] != owner_space)
          continue;
        owner.first = idx;
        found = true;
        break;
      }
      // Pick a shard to be the owner if we don't have a local shard
      if (!found)
      {
        owner.first = field_allocator_shard++;
        if (field_allocator_shard == total_shards)
          field_allocator_shard = 0;
      }
      if (owner_space == runtime->address_space)
        owner.second = (owner.first == owner_shard->shard_id);
      else
        owner.second = shard_manager->is_first_local_shard(owner_shard);
      legion_assert(
          field_allocator_owner_shards.find(handle) ==
          field_allocator_owner_shards.end());
      field_allocator_owner_shards[handle] = owner;
      RtEvent ready;
      if (owner.second)
        ready = node->create_allocator(
            runtime->address_space, RtUserEvent::NO_RT_USER_EVENT,
            true /*sharded context*/, (owner.first == owner_shard->shard_id));
      // Don't have one so make a new one
      FieldAllocatorImpl* result = new FieldAllocatorImpl(node, this, ready);
      // Save it for later
      field_allocators[handle] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_field_allocator(FieldSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      bool found = false;
      std::pair<ShardID, bool> result;
      {
        AutoLock priv_lock(privilege_lock);
        // Check to see if we still have one
        std::map<FieldSpace, FieldAllocatorImpl*>::iterator finder =
            field_allocators.find(node->handle);
        if (finder != field_allocators.end())
        {
          found = true;
          field_allocators.erase(finder);
          std::map<FieldSpace, std::pair<ShardID, bool> >::iterator
              owner_finder = field_allocator_owner_shards.find(node->handle);
          legion_assert(owner_finder != field_allocator_owner_shards.end());
          result = owner_finder->second;

          field_allocator_owner_shards.erase(owner_finder);
        }
      }
      if (found && result.second)
      {
        const RtEvent ready = node->destroy_allocator(
            runtime->address_space, true /*sharded*/,
            (result.first == owner_shard->shard_id));
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else if (!found)
      {
        const RtEvent ready = node->destroy_allocator(runtime->address_space);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::initialize_unordered_collective(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(unordered_collective == nullptr);
      unordered_ops_counter = 0;
      unordered_collective = new UnorderedExchange(this, COLLECTIVE_LOC_88);
      unordered_collective->start_unordered_exchange(unordered_ops);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::finalize_unordered_collective(AutoLock& d_lock)
    //--------------------------------------------------------------------------
    {
      legion_assert(unordered_collective != nullptr);
      const RtEvent ready =
          unordered_collective->perform_collective_wait(false /*block*/);
      if (ready.exists() && !ready.has_triggered())
      {
        d_lock.release();
        ready.wait();
        d_lock.reacquire();
      }
      std::vector<Operation*> ready_operations;
      if (!unordered_ops.empty())
        unordered_collective->find_ready_operations(ready_operations);
      delete unordered_collective;
      unordered_collective = nullptr;
      if (!ready_operations.empty())
      {
        // Filter out the ready operations
        for (std::vector<Operation*>::iterator it = unordered_ops.begin();
             it != unordered_ops.end();
             /*nothing*/)
        {
          bool filter = false;
          for (unsigned idx = 0; idx < ready_operations.size(); idx++)
          {
            if ((*it) != ready_operations[idx])
              continue;
            filter = true;
            break;
          }
          if (filter)
            it = unordered_ops.erase(it);
          else
            it++;
        }
        // Now we can do the insertion
        issue_unordered_operations(d_lock, ready_operations);
        // Reset the epoch counter back to the minimum since we found some
        // matches so we should probably be doing this more often
        unordered_ops_epoch = MIN_UNORDERED_OPS_EPOCH;
      }
      else if (unordered_ops_epoch < MAX_UNORDERED_OPS_EPOCH)
        // If there were no ready unordered ops then we double the epoch
        unordered_ops_epoch *= 2;
      legion_assert(MIN_UNORDERED_OPS_EPOCH <= unordered_ops_epoch);
      legion_assert(unordered_ops_epoch <= MAX_UNORDERED_OPS_EPOCH);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::insert_unordered_ops(AutoLock& d_lock)
    //--------------------------------------------------------------------------
    {
      // If we have a trace then we're definitely not inserting operations
      if (current_trace != nullptr)
        return;
      // For control replication, we need to have an algorithm to determine
      // when the shards try to sync up to insert operations that doesn't
      // rely on knowing if or when any one shard has unordered ops
      // We employ a sampling based algorithm here with exponential backoff
      // to detect when we are doing unordered ops since it's likely a
      // binary state where either we are or we aren't doing unordered ops
      legion_assert(unordered_ops_counter < unordered_ops_epoch);
      // If we're doing progress then we can skip this check and
      // reset the counter back to zero since we're doing an exchange
      if (++unordered_ops_counter < unordered_ops_epoch)
        return;
      // Check to see if the previous exchange had any matching unordered
      // operations for us to perform
      if (unordered_collective != nullptr)
        finalize_unordered_collective(d_lock);
      // Start the next exchange
      initialize_unordered_collective();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::progress_unordered_operations(bool end_task)
    //--------------------------------------------------------------------------
    {
      AutoLock d_lock(dependence_lock);
      if (end_task)
      {
        // This is the end of this parent task so mark that we're done
        legion_assert(!finished_execution);
        legion_assert(current_trace == nullptr);
        finished_execution = true;
      }
      // No progress can occur inside of a trace
      else if (current_trace != nullptr)
        return;
      // With control replication we're always doing half phases for detecting
      // when we have unordered operations across all shards that are ready to
      // be performed, but in this case the user has asked us to actually
      // perform a full phase, so finish the previous one and then start a
      // new phase if we're not the end last phase
      if (unordered_collective != nullptr)
        finalize_unordered_collective(d_lock);
      initialize_unordered_collective();
      finalize_unordered_collective(d_lock);
      if (end_task)
      {
        if (!unordered_ops.empty())
        {
          Warning warning;
          warning
              << "Control replicated task " << get_task_name() << " (UID "
              << get_unique_id() << ") had " << unordered_ops.size()
              << " mismatched unordered operations at the end of its execution "
              << "that are now leaked.";
          warning.raise();
        }
      }
      else
        initialize_unordered_collective();
    }

    //--------------------------------------------------------------------------
    unsigned ReplicateContext::minimize_repeat_results(
        unsigned ready, bool& double_wait_interval)
    //--------------------------------------------------------------------------
    {
      unsigned result = 0;
      if (minimize_repeats_collective != nullptr)
      {
        result = minimize_repeats_collective->get_result();
        legion_assert(result <= ready);
        ready -= result;
        double_wait_interval = (result == 0);
      }
      else
        double_wait_interval = false;
      minimize_repeats_collective =
          new AllReduceCollective<MinReduction<unsigned>, false>(
              COLLECTIVE_LOC_106, this);
      minimize_repeats_collective->async_all_reduce(ready);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::execute_task(
        const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_EXECUTE_TASK, __func__);
        hash_task_launcher(hasher, runtime->safe_control_replication, launcher);
        if (outputs != nullptr)
          hash_output_requirements(hasher, *outputs);
        if (hasher.verify(__func__))
          break;
      }
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_task_false(launcher, provenance);
      // If we're doing a local-function task then we can run that with just
      // a normal individual task in each shard since it is safe to duplicate
      if (launcher.local_function_task)
        return InnerContext::execute_task(launcher, outputs, provenance);
      ReplIndividualTask* task = runtime->get_operation<ReplIndividualTask>();
      Future result = task->initialize_task(
          this, launcher, provenance, false /*top_level*/, false /*must epoch*/,
          outputs);
      if (runtime->safe_mapper)
        task->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_43));
      // Now initialize the particular information for replication
      task->initialize_replication(this);
      if (launcher.enable_inlining && !launcher.silence_warnings)
      {
        Warning warning;
        warning << "Inlining is not currently supported for replicated tasks "
                   "such as "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
      }
      execute_task_launch(
          task, false /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, false /*no inlining*/);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplicateContext::execute_index_space(
        const IndexTaskLauncher& launcher,
        std::vector<OutputRequirement>* outputs, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        FutureMap result = execute_must_epoch(epoch_launcher, provenance);
        return result;
      }
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_EXECUTE_INDEX_SPACE, __func__);
        hash_index_launcher(
            hasher, runtime->safe_control_replication, launcher);
        if (outputs != nullptr)
          hash_output_requirements(hasher, *outputs);
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        log_legion.warning(
            "Ignoring empty index task launch in task %s (ID %lld)",
            get_task_name(), get_unique_id());
        return FutureMap();
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_index_task_false(launch_space, launcher, provenance);
      ReplIndexTask* task = runtime->get_operation<ReplIndexTask>();
      FutureMap result = task->initialize_task(
          this, launcher, launch_space, provenance, true /*track*/, outputs);
      if (runtime->safe_mapper)
        task->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_44));
      task->initialize_replication(this);
      if (launcher.enable_inlining && !launcher.silence_warnings)
      {
        Warning warning;
        warning << "Inlining is not currently supported for replicated tasks "
                   "such as "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
      }
      execute_task_launch(
          task, true /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, false /*no inlining*/);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::execute_index_space(
        const IndexTaskLauncher& launcher, ReductionOpID redop,
        bool deterministic, std::vector<OutputRequirement>* outputs,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.must_parallelism)
      {
        // Turn around and use a must epoch launcher
        MustEpochLauncher epoch_launcher(launcher.map_id, launcher.tag);
        epoch_launcher.add_index_task(launcher);
        FutureMap result = execute_must_epoch(epoch_launcher, provenance);
        // Reduce the future map down to a future
        return reduce_future_map(
            result, redop, deterministic, launcher.map_id, launcher.tag,
            provenance, launcher.initial_value);
      }
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_EXECUTE_INDEX_SPACE, __func__);
        hash_index_launcher(
            hasher, runtime->safe_control_replication, launcher);
        hasher.hash(redop, "redop");
        hasher.hash<bool>(deterministic, "deterministic");
        if (outputs != nullptr)
          hash_output_requirements(hasher, *outputs);
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        if (!launcher.initial_value.is_empty())
          return launcher.initial_value;

        Warning warning;
        warning << "Ignoring empty index task launch in task "
                << get_task_name() << " (ID " << get_unique_id() << ").";
        warning.raise();
        const ReductionOp* reduction_op = runtime->get_reduction(redop);
        FutureImpl* result = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance);
        result->set_local(
            reduction_op->identity, reduction_op->sizeof_rhs, false /*own*/);
        return Future(result);
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      // Quick out for predicate false
      if (launcher.predicate == Predicate::FALSE_PRED)
        return predicate_index_task_reduce_false(
            launcher, launch_space, redop, provenance);
      ReplIndexTask* task = runtime->get_operation<ReplIndexTask>();
      Future result = task->initialize_task(
          this, launcher, launch_space, provenance, redop, deterministic,
          true /*track*/, outputs);
      if (runtime->safe_mapper)
        task->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_45));
      task->initialize_replication(this);
      if (launcher.enable_inlining && !launcher.silence_warnings)
      {
        Warning warning;
        warning << "Inlining is not currently supported for replicated tasks "
                   "such as "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
      }
      execute_task_launch(
          task, true /*index*/, launcher.static_dependences, provenance,
          launcher.silence_warnings, false /*no inlining*/);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::reduce_future_map(
        const FutureMap& future_map, ReductionOpID redop, bool deterministic,
        MapperID mapper_id, MappingTagID tag, Provenance* provenance,
        Future initial_value)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_REDUCE_FUTURE_MAP, __func__);
        hash_future_map(hasher, future_map, "future_map");
        hash_future(
            hasher, runtime->safe_control_replication, initial_value,
            "initial_value");
        hasher.hash(redop, "redop");
        hasher.hash(deterministic, "deterministic");
        if (hasher.verify(__func__))
          break;
      }
      if (future_map.impl == nullptr)
      {
        const ReductionOp* reduction_op = runtime->get_reduction(redop);
        FutureImpl* result = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance);
        result->set_local(
            reduction_op->identity, reduction_op->sizeof_rhs, false /*own*/);
        return Future(result);
      }
      // Check to see if this is just a normal future map, if so then
      // we can just do the standard thing here
      if (!future_map.impl->is_replicate_future_map())
        return InnerContext::reduce_future_map(
            future_map, redop, deterministic, mapper_id, tag, provenance,
            initial_value);
      ReplAllReduceOp* all_reduce_op =
          runtime->get_operation<ReplAllReduceOp>();
      Future result = all_reduce_op->initialize(
          this, future_map, redop, deterministic, mapper_id, tag, provenance,
          initial_value);
      all_reduce_op->initialize_replication(this);
      add_to_dependence_queue(all_reduce_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::increase_pending_distributed_ids(
        unsigned count, bool double_next)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < count; idx++)
      {
        ValueBroadcast<DIDBroadcast>* collective =
            new ValueBroadcast<DIDBroadcast>(
                this, distributed_id_allocator_shard, COLLECTIVE_LOC_2);
        if (owner_shard->shard_id == distributed_id_allocator_shard)
        {
          const DistributedID did = runtime->get_available_distributed_id();
          // Do our arrival on this generation, should be the last one
          collective->broadcast(DIDBroadcast(did, double_next));
          pending_distributed_ids.emplace_back(
              std::pair<ValueBroadcast<DIDBroadcast>*, bool>(collective, true));
        }
        else
        {
          register_collective(collective);
          pending_distributed_ids.emplace_back(
              std::pair<ValueBroadcast<DIDBroadcast>*, bool>(
                  collective, false));
        }
        distributed_id_allocator_shard++;
        if (distributed_id_allocator_shard == total_shards)
          distributed_id_allocator_shard = 0;
        double_next = false;
      }
    }

    //--------------------------------------------------------------------------
    DistributedID ReplicateContext::get_next_distributed_id(void)
    //--------------------------------------------------------------------------
    {
      if (pending_distributed_ids.empty())
      {
        increase_pending_distributed_ids(1 /*count*/, false /*double*/);
        pending_distributed_id_check = 0;
      }
      bool double_next = false;
      std::pair<ValueBroadcast<DIDBroadcast>*, bool>& pending_did =
          pending_distributed_ids.front();
      if (!pending_did.second)
      {
        const RtEvent done = pending_did.first->get_done_event();
        if (!done.has_triggered())
        {
          double_next = true;
          done.wait();
        }
      }
      const DIDBroadcast value = pending_did.first->get_value(false);
      bool double_buffer = false;
      if (++pending_distributed_id_check == pending_distributed_ids.size())
      {
        double_buffer = value.double_buffer;
        pending_distributed_id_check = 0;
      }
      // Get new handles in flight for the next time we need them
      // Always add a new one to replace the old one, but double the number
      // in flight if we're not hiding the latency
      increase_pending_distributed_ids(
          double_buffer ? pending_distributed_ids.size() + 1 : 1, double_next);
      delete pending_did.first;
      pending_distributed_ids.pop_front();
      return value.did;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplicateContext::construct_future_map(
        IndexSpace space, const std::map<DomainPoint, UntypedBuffer>& data,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CONSTRUCT_FUTURE_MAP, __func__);
        if (check_space)
          hasher.hash(space, "space");
        if (!collective)
        {
          for (const std::map<DomainPoint, UntypedBuffer>::value_type& it :
               data)
          {
            hasher.hash(it.first, "data");
            if (runtime->safe_control_replication > 1)
              hasher.hash(it.second.get_ptr(), it.second.get_size(), "data");
          }
        }
        else if (!implicit)
          hasher.hash(sid, "sid");
        if (hasher.verify(__func__))
          break;
      }
      IndexSpaceNode* domain_node = runtime->get_node(space);
      Domain domain = domain_node->get_tight_domain();
      FutureMap result;
      if (collective)
      {
        const DistributedID map_did = get_next_distributed_id();
        result = shard_manager->deduplicate_future_map_creation(
            this, domain_node, domain_node, map_did, provenance);
        ReplFutureMapImpl* map = static_cast<ReplFutureMapImpl*>(result.impl);
        ShardingFunction* function;
        if (implicit)
        {
          // Do an exchange between the shards to compute the implicit sharding
          // No need to wait for it to be done before continuing
          ImplicitShardingFunctor* functor =
              new ImplicitShardingFunctor(this, COLLECTIVE_LOC_101, map);
          functor->compute_sharding(data);
          function = new ShardingFunction(
              functor, shard_manager, sid, false, true /*own functor*/);
          if (!map->set_sharding_function(function, true /*own*/))
          {
            // Wait for the collective to be done before we delete it
            functor->perform_collective_wait();
            delete function;
          }
        }
        else
        {
          function = shard_manager->find_sharding_function(sid);
          map->set_sharding_function(function, false /*own*/);
        }
        // Check that all the points abide by the sharding function
        for (const std::map<DomainPoint, UntypedBuffer>::value_type& it : data)
          if (function->find_owner(it.first, domain) != owner_shard->shard_id)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Sharding function does not match described sharding for "
                     "future map "
                  << "construction in " << get_task_name() << " (UID "
                  << get_unique_id() << ").";
            error.raise();
          }
      }
      else
      {
        if (data.size() != domain_node->get_volume())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "The number of buffers passed into a future map construction ("
              << data.size() << ") does not match the volume of the domain ("
              << domain_node->get_volume() << ") for the future map in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
          error.raise();
        }
        const DistributedID did = runtime->get_available_distributed_id();
        result = FutureMap(new FutureMapImpl(
            this, domain_node, did, NO_BLOCKING_INDEX,
            std::optional<uint64_t>(), provenance));
      }
      for (const std::map<DomainPoint, UntypedBuffer>::value_type& it : data)
      {
        if (!domain.contains(it.first))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Point passed into future map construction is not contained "
                   "within "
                << "the bounds of the domain in task " << get_task_name()
                << " (UID " << get_unique_id() << ").";
          error.raise();
        }
        const size_t future_size = it.second.get_size();
        FutureImpl* future = new FutureImpl(
            this, true /*register*/, runtime->get_available_distributed_id(),
            provenance);
        future->set_local(it.second.get_ptr(), future_size);
        result.impl->set_future(it.first, future);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplicateContext::construct_future_map(
        IndexSpace space, const std::map<DomainPoint, Future>& futures,
        Provenance* provenance, bool collective, ShardingID sid, bool implicit,
        bool check_space)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_CONSTRUCT_FUTURE_MAP, __func__);
        if (check_space)
          hasher.hash(space, "space");
        if (!collective)
        {
          for (const std::map<DomainPoint, Future>::value_type& it : futures)
          {
            hasher.hash(it.first, "futures");
            hash_future(
                hasher, runtime->safe_control_replication, it.second,
                "futures");
          }
        }
        else if (!implicit)
          hasher.hash(sid, "sid");
        if (hasher.verify(__func__))
          break;
      }
      IndexSpaceNode* domain_node = runtime->get_node(space);
      CreationOp* creation_op = runtime->get_operation<CreationOp>();
      creation_op->initialize_map(this, provenance, futures);
      FutureMap result;
      if (collective)
      {
        const DistributedID map_did = get_next_distributed_id();
        result = shard_manager->deduplicate_future_map_creation(
            this, creation_op, domain_node, domain_node, map_did, provenance);
        ReplFutureMapImpl* map = static_cast<ReplFutureMapImpl*>(result.impl);
        ShardingFunction* function;
        if (implicit)
        {
          // Do an exchange between the shards to compute the implicit sharding
          // No need to wait for it to be done before continuing
          ImplicitShardingFunctor* functor =
              new ImplicitShardingFunctor(this, COLLECTIVE_LOC_102, map);
          functor->compute_sharding(futures);
          function = new ShardingFunction(
              functor, shard_manager, sid, false, true /*own functor*/);
          if (!map->set_sharding_function(function, true /*own*/))
          {
            // Wait for the collective to be done before we delete it
            functor->perform_collective_wait();
            delete function;
          }
        }
        else
        {
          function = shard_manager->find_sharding_function(sid);
          map->set_sharding_function(function, false /*own*/);
        }
        // Check that all the points abide by the sharding function
        Domain domain = domain_node->get_tight_domain();
        for (const std::map<DomainPoint, Future>::value_type& it : futures)
          if (function->find_owner(it.first, domain) != owner_shard->shard_id)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Sharding function does not match described sharding for "
                     "future map "
                  << "construction in " << get_task_name() << " (UID "
                  << get_unique_id() << ").";
            error.raise();
          }
      }
      else
      {
        if (futures.size() != domain_node->get_volume())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "The number of futures passed into a future map construction ("
              << futures.size() << ") does not match the volume of the domain ("
              << domain_node->get_volume() << ") for the future map in task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
          error.raise();
        }
        const DistributedID did = runtime->get_available_distributed_id();
        result = FutureMap(
            new FutureMapImpl(this, creation_op, domain_node, did, provenance));
      }
      add_to_dependence_queue(creation_op);
      result.impl->set_all_futures(futures);
      return result;
    }

    //--------------------------------------------------------------------------
    PhysicalRegion ReplicateContext::map_region(
        const InlineLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_MAP_REGION, __func__);
        Serializer rez;
        ExternalMappable::pack_region_requirement(launcher.requirement, rez);
        hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "requirement");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        hasher.hash(launcher.layout_constraint_id, "layout_constraints");
        hash_static_dependences(hasher, launcher.static_dependences);
        if (hasher.verify(__func__))
          break;
      }
      if (IS_NO_ACCESS(launcher.requirement))
        return PhysicalRegion();
      ReplMapOp* map_op = runtime->get_operation<ReplMapOp>();
      PhysicalRegion result = map_op->initialize(
          this, launcher, provenance, physical_region_count++);
      map_op->initialize_replication(this);
      if (current_trace != nullptr)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal attempt to perform an unordered operation in task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") after execution has completed.";
        error.raise();
      }
      bool parent_conflict = false, inline_conflict = false;
      const int index =
          has_conflicting_regions(map_op, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an external attach operation on region "
              << launcher.requirement.region << " in task " << get_task_name()
              << " (UID " << get_unique_id()
              << ") which would conflict with mapped parent region "
              << "requirement " << index << ".";
        error.raise();
      }
      if (inline_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an external attach operation on region "
              << launcher.requirement.region << " in task " << get_task_name()
              << " (UID " << get_unique_id()
              << ") which would conflict with previous inline mapping.";
        error.raise();
      }
      register_inline_mapped_region(result);
      add_to_dependence_queue(map_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent ReplicateContext::remap_region(
        const PhysicalRegion& region, Provenance* provenance, bool internal)
    //--------------------------------------------------------------------------
    {
      if (!internal)
      {
        for (int i = 0;
             runtime->safe_control_replication && (i < 2) &&
             ((current_trace == nullptr) || !current_trace->is_fixed());
             i++)
        {
          HashVerifier hasher(
              this, runtime->safe_control_replication > 1, i > 0, provenance);
          hasher.hash(REPLICATE_REMAP_REGION, __func__);
          Serializer rez;
          ExternalMappable::pack_region_requirement(
              region.impl->get_requirement(), rez);
          hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "requirement");
          hasher.hash<bool>(region.is_mapped(), "is_mapped");
          if (hasher.verify(__func__))
            break;
        }
        return remap_region(region, provenance, true /*internal*/);
      }
      // Check to see if the region is already mapped,
      // if it is then we are done
      if (region.is_mapped())
        return ApEvent::NO_AP_EVENT;
      if (current_trace != nullptr)
      {
        const RegionRequirement& req = region.impl->get_requirement();
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Attempted an inline mapping of region " << req.region
              << " inside of trace " << current_trace->tid << " of task "
              << *this << ". It is illegal to perform inline mappings inside "
              << "of traces.";
        error.raise();
      }
      ReplMapOp* map_op = runtime->get_operation<ReplMapOp>();
      map_op->initialize(this, region, provenance);
      map_op->initialize_replication(this);
      register_inline_mapped_region(region);
      const ApEvent result = map_op->get_completion_event();
      add_to_dependence_queue(map_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::fill_fields(
        const FillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_FILL_FIELDS, __func__);
        hasher.hash(launcher.handle, "handle");
        hasher.hash(launcher.parent, "parent");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.argument,
            "argument");
        hash_future(
            hasher, runtime->safe_control_replication, launcher.future,
            "future");
        hash_predicate(hasher, launcher.predicate, "predicate");
        for (const FieldID& it : launcher.fields) hasher.hash(it, "fields");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        for (int idx = 0; idx < launcher.point.get_dim(); idx++)
          hasher.hash(launcher.point[idx], "point");
        hasher.hash(launcher.sharding_space, "sharding_space");
        hash_static_dependences(hasher, launcher.static_dependences);
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring fill request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
        return;
      }
      ReplFillOp* fill_op = runtime->get_operation<ReplFillOp>();
      fill_op->initialize(this, launcher, provenance);
      fill_op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "fill_fields call in task " << get_task_name() << " (UID "
              << get_unique_id() << ").";
          warning.raise();
        }
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(fill_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::fill_fields(
        const IndexFillLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_FILL_FIELDS, __func__);
        hasher.hash(launcher.launch_domain, "launch_domain");
        hasher.hash(launcher.launch_space, "launch_space");
        hasher.hash(launcher.sharding_space, "sharding_space");
        hasher.hash(launcher.region, "region");
        hasher.hash(launcher.partition, "partition");
        hasher.hash(launcher.parent, "parent");
        hasher.hash(launcher.projection, "projection");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.argument,
            "argument");
        hash_future(
            hasher, runtime->safe_control_replication, launcher.future,
            "future");
        hash_predicate(hasher, launcher.predicate, "predicate");
        for (const FieldID& it : launcher.fields) hasher.hash(it, "fields");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        hash_static_dependences(hasher, launcher.static_dependences);
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring index fill request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
        return;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        log_legion.warning(
            "Ignoring empty index space fill in task %s (ID %lld)",
            get_task_name(), get_unique_id());
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      ReplIndexFillOp* fill_op = runtime->get_operation<ReplIndexFillOp>();
      fill_op->initialize(this, launcher, launch_space, provenance);
      if (runtime->safe_mapper)
        fill_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_46));
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(fill_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around fill_fields call in task %s (UID %lld).",
              get_task_name(), get_unique_id());
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(fill_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::discard_fields(
        const DiscardLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DISCARD_FIELDS, __func__);
        hasher.hash(launcher.handle, "handle");
        hasher.hash(launcher.parent, "parent");
        for (const FieldID& it : launcher.fields) hasher.hash(it, "fields");
        hash_static_dependences(hasher, launcher.static_dependences);
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.fields.empty())
      {
        Warning warning;
        warning << "Ignoring discard request with no fields in task "
                << get_task_name() << " (UID " << get_unique_id() << ").";
        warning.raise();
        return;
      }
      ReplDiscardOp* discard_op = runtime->get_operation<ReplDiscardOp>();
      discard_op->initialize(this, launcher, provenance);
      discard_op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      // We still unamp conflicting regions for discard, but we wil never
      // remap them afterwards since this is invalidating the data
      if (!runtime->unsafe_launch)
      {
        std::vector<PhysicalRegion> unmapped_regions;
        find_conflicting_regions(discard_op, unmapped_regions);
        if (!unmapped_regions.empty())
        {
          if (runtime->runtime_warnings && !launcher.silence_warnings)
          {
            Warning warning;
            warning
                << "Runtime is unmapping and remapping physical regions around "
                << "discard_fields call in task " << get_task_name() << " (UID "
                << get_unique_id() << ").";
            warning.raise();
          }
          // Unmap any regions which are conflicting
          for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
            unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
        }
      }
      add_to_dependence_queue(discard_op, launcher.static_dependences);
      // Do not remap the previously mapped regions, they are uninitialized
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::issue_copy(
        const CopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ISSUE_COPY, __func__);
        hash_region_requirements(hasher, launcher.src_requirements);
        hash_region_requirements(hasher, launcher.dst_requirements);
        hash_region_requirements(hasher, launcher.src_indirect_requirements);
        hash_region_requirements(hasher, launcher.dst_indirect_requirements);
        for (const bool& it : launcher.src_indirect_is_range)
          hasher.hash<bool>(it, "src_indirect_is_range");
        for (const bool& it : launcher.dst_indirect_is_range)
          hasher.hash<bool>(it, "dst_indirect_is_range");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hash_predicate(hasher, launcher.predicate, "predicate");
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        for (int idx = 0; idx < launcher.point.get_dim(); idx++)
          hasher.hash(launcher.point[idx], "point");
        hasher.hash(launcher.sharding_space, "sharding_space");
        hash_static_dependences(hasher, launcher.static_dependences);
        hasher.hash(
            launcher.possible_src_indirect_out_of_range,
            "possible_src_indirect_out_of_range");
        hasher.hash(
            launcher.possible_dst_indirect_out_of_range,
            "possible_dst_indirect_out_of_range");
        hasher.hash(
            launcher.possible_dst_indirect_aliasing,
            "possible_dst_indirect_aliasing");
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      ReplCopyOp* copy_op = runtime->get_operation<ReplCopyOp>();
      copy_op->initialize(this, launcher, provenance);
      if (runtime->safe_mapper)
        copy_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_47));
      copy_op->initialize_replication(this);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).",
              get_task_name(), get_unique_id());
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(copy_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::issue_copy(
        const IndexCopyLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ISSUE_COPY, __func__);
        hash_region_requirements(hasher, launcher.src_requirements);
        hash_region_requirements(hasher, launcher.dst_requirements);
        hash_region_requirements(hasher, launcher.src_indirect_requirements);
        hash_region_requirements(hasher, launcher.dst_indirect_requirements);
        for (const bool& it : launcher.src_indirect_is_range)
          hasher.hash<bool>(it, "src_indirect_is_range");
        for (const bool& it : launcher.dst_indirect_is_range)
          hasher.hash<bool>(it, "dst_indirect_is_range");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hash_predicate(hasher, launcher.predicate, "predicate");
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        hasher.hash(launcher.launch_domain, "launch_domain");
        hasher.hash(launcher.launch_space, "launch_space");
        hasher.hash(launcher.sharding_space, "sharding_space");
        hash_static_dependences(hasher, launcher.static_dependences);
        hasher.hash(
            launcher.possible_src_indirect_out_of_range,
            "possible_src_indirect_out_of_range");
        hasher.hash(
            launcher.possible_dst_indirect_out_of_range,
            "possible_dst_indirect_out_of_range");
        hasher.hash(
            launcher.possible_dst_indirect_aliasing,
            "possible_dst_indirect_aliasing");
        hasher.hash(
            launcher.collective_src_indirect_points,
            "collective_src_indirect_points");
        hasher.hash(
            launcher.collective_dst_indirect_points,
            "collective_dst_indirect_points");
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.launch_domain.exists() &&
          (launcher.launch_domain.get_volume() == 0))
      {
        log_legion.warning(
            "Ignoring empty index space copy in task %s "
            "(ID %lld)",
            get_task_name(), get_unique_id());
        return;
      }
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
        launch_space =
            find_index_launch_space(launcher.launch_domain, provenance);
      ReplIndexCopyOp* copy_op = runtime->get_operation<ReplIndexCopyOp>();
      copy_op->initialize(this, launcher, launch_space, provenance);
      if (runtime->safe_mapper)
        copy_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_48));
      copy_op->initialize_replication(this);
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this copy operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(copy_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
          log_legion.warning(
              "WARNING: Runtime is unmapping and remapping "
              "physical regions around issue_copy_operation call in "
              "task %s (UID %lld).",
              get_task_name(), get_unique_id());
        // Unmap any regions which are conflicting
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the copy operation
      add_to_dependence_queue(copy_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::issue_acquire(
        const AcquireLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_ACQUIRE, __func__);
        hasher.hash(launcher.logical_region, "logical_region");
        hasher.hash(launcher.parent_region, "parent_region");
        for (const FieldID& it : launcher.fields) hasher.hash(it, "fields");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        if (launcher.physical_region.impl != nullptr)
        {
          Serializer rez;
          ExternalMappable::pack_region_requirement(
              launcher.physical_region.impl->get_requirement(), rez);
          hasher.hash(
              rez.get_buffer(), rez.get_used_bytes(), "physical_region");
          hasher.hash<bool>(launcher.physical_region.is_mapped(), "is_mapped");
        }
        if (hasher.verify(__func__))
          break;
      }
      ReplAcquireOp* acquire_op = runtime->get_operation<ReplAcquireOp>();
      acquire_op->initialize(this, launcher, provenance);
      acquire_op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      // Check to see if we need to do any unmappings and remappings
      // before we can issue this acquire operation.
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(acquire_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "issue_acquire call in task " << get_task_name() << " (UID "
              << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the acquire operation
      add_to_dependence_queue(acquire_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::issue_release(
        const ReleaseLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_RELEASE, __func__);
        hasher.hash(launcher.logical_region, "logical_region");
        hasher.hash(launcher.parent_region, "parent_region");
        for (const FieldID& it : launcher.fields) hasher.hash(it, "fields");
        hash_grants(hasher, launcher.grants);
        hash_phase_barriers(hasher, launcher.wait_barriers);
        hash_phase_barriers(hasher, launcher.arrive_barriers);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.map_arg,
            "map_arg");
        if (launcher.physical_region.impl != nullptr)
        {
          Serializer rez;
          ExternalMappable::pack_region_requirement(
              launcher.physical_region.impl->get_requirement(), rez);
          hasher.hash(
              rez.get_buffer(), rez.get_used_bytes(), "physical_region");
          hasher.hash<bool>(launcher.physical_region.is_mapped(), "is_mappped");
        }
        if (hasher.verify(__func__))
          break;
      }
      ReplReleaseOp* release_op = runtime->get_operation<ReplReleaseOp>();
      release_op->initialize(this, launcher, provenance);
      release_op->initialize_replication(
          this, shard_manager->is_first_local_shard(owner_shard));
      // Check to see if we need to do any unmappings and remappings
      // before we can issue the release operation
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        find_conflicting_regions(release_op, unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning
              << "Runtime is unmapping and remapping physical regions around "
              << "issue_release call in task " << get_task_name() << " (UID "
              << get_unique_id() << ").";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Issue the release operation
      add_to_dependence_queue(release_op, launcher.static_dependences);
      // Remap any regions which we unmapped
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
    }

    //--------------------------------------------------------------------------
    PhysicalRegion ReplicateContext::attach_resource(
        const AttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_ATTACH_RESOURCE, __func__);
        hasher.hash(launcher.resource, "resource");
        hasher.hash(launcher.handle, "handle");
        hasher.hash(launcher.parent, "parent");
        hasher.hash(launcher.restricted, "restricted");
        hasher.hash(launcher.mapped, "mapped");
        hasher.hash(launcher.collective, "collective");
        hasher.hash(
            launcher.deduplicate_across_shards, "deduplicate_across_shards");
        for (const std::map<FieldID, const char*>::value_type& it :
             launcher.field_files)
        {
          hasher.hash(it.first, "field_files");
          hasher.hash(it.second, strlen(it.second), "field_files");
        }
        hash_layout_constraints(
            hasher, launcher.constraints, false /*pointers*/);
        for (const FieldID& it : launcher.privilege_fields)
          hasher.hash(it, "privilege_fields");
        hasher.hash(launcher.footprint, "footprint");
        hash_static_dependences(hasher, launcher.static_dependences);
        if (hasher.verify(__func__))
          break;
      }
      if (launcher.restricted)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attach operations in control replication context "
              << get_task_name() << " (UID " << get_unique_id()
              << ") requested a restriction. "
              << "Restrictions are only permitted for attach operations in "
              << "non-control-replicated contexts currently.";
        error.raise();
      }
      ReplAttachOp* attach_op = runtime->get_operation<ReplAttachOp>();
      PhysicalRegion result = attach_op->initialize(
          this, launcher, provenance, physical_region_count++);
      attach_op->initialize_replication(
          this, launcher.collective, launcher.deduplicate_across_shards,
          shard_manager->is_first_local_shard(owner_shard));
      bool parent_conflict = false, inline_conflict = false;
      int index =
          has_conflicting_regions(attach_op, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an external attach operation on region "
              << launcher.handle << " in task " << get_task_name() << " (UID "
              << get_unique_id()
              << ") which would conflict with mapped parent region "
              << "requirement " << index << ".";
        error.raise();
      }
      if (inline_conflict)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Attempted an external attach operation on region "
              << launcher.handle << " in task " << get_task_name() << " (UID "
              << get_unique_id()
              << ") which would conflict with previous inline mapping.";
        error.raise();
      }
      // If we're counting this region as mapped we need to register it
      if (launcher.mapped)
        register_inline_mapped_region(result);
      add_to_dependence_queue(attach_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    ExternalResources ReplicateContext::attach_resources(
        const IndexAttachLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_INDEX_ATTACH_RESOURCE, __func__);
        hasher.hash(launcher.resource, "resource");
        hasher.hash(launcher.parent, "parent");
        hasher.hash(launcher.restricted, "restricted");
        hasher.hash(
            launcher.deduplicate_across_shards, "deduplicate_across_shards");
        hash_layout_constraints(
            hasher, launcher.constraints, false /*pointers*/);
        // Everything else other than the privilege fields is sharded already
        // Make sure we include privilege fields from the files too
        // Effectively the direct privilege fields or privilege fields
        // mentioned by any of the other data structures need to be the same
        std::set<FieldID> all_privilege_fields(launcher.privilege_fields);
        for (const std::map<FieldID, std::vector<const char*> >::value_type&
                 it : launcher.field_files)
          all_privilege_fields.insert(it.first);
        for (const FieldID& it : all_privilege_fields)
          hasher.hash(it, "privilege_fields");
        hash_static_dependences(hasher, launcher.static_dependences);
        if (hasher.verify(__func__))
          break;
      }
      std::vector<unsigned> indexes;
      if (!launcher.deduplicate_across_shards)
      {
        indexes.resize(launcher.handles.size());
        for (unsigned idx = 0; idx < indexes.size(); idx++) indexes[idx] = idx;
      }
      else  // ask the shard manager to deduplicate here
        shard_manager->deduplicate_attaches(launcher, indexes);
      // Start this inflight before we compute the upper bound
      IndexAttachLaunchSpace collective(this, COLLECTIVE_LOC_28);
      collective.exchange_counts(indexes.size());
      // Compute the upper bound partition node from this launcher
      RegionTreeNode* node =
          compute_index_attach_upper_bound(launcher, indexes);
      ReplIndexAttachOp* attach_op =
          runtime->get_operation<ReplIndexAttachOp>();
      IndexSpaceNode* launch_space = collective.get_launch_space(provenance);
      ExternalResources result = attach_op->initialize(
          this, node, launch_space, launcher, indexes, provenance,
          true /*replicated*/, physical_region_count);
      physical_region_count += indexes.size();
      attach_op->initialize_replication(this);
      const RegionRequirement& req = attach_op->get_requirement();
      bool parent_conflict = false, inline_conflict = false;
      int index =
          has_conflicting_internal(req, parent_conflict, inline_conflict);
      if (parent_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Attempted an index attach operation with upper bound "
                   "partition "
                << req.partition << " that conflicts with mapped region "
                << regions[index].region << " at index " << index
                << " of parent task " << get_task_name() << " (ID "
                << get_unique_id()
                << ") which would conflict with mapped parent region.";
          error.raise();
        }
        else
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Attempted an index attach operation with upper bound region "
              << req.region << " that conflicts with mapped region "
              << regions[index].region << " at index " << index
              << " of parent task " << get_task_name() << " (ID "
              << get_unique_id()
              << ") which would conflict with mapped parent region.";
          error.raise();
        }
      }
      if (inline_conflict)
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Attempted an index attach operation with upper bound "
                   "partition "
                << req.partition
                << " that conflicts with previous inline mapping in task "
                << get_task_name() << " (ID " << get_unique_id()
                << ") which would conflict with previous inline mapping.";
          error.raise();
        }
        else
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Attempted an index attach operation with upper bound region "
              << req.region
              << " that conflicts with previous inline mapping in task "
              << get_task_name() << " (ID " << get_unique_id()
              << ") which would conflict with previous inline mapping.";
          error.raise();
        }
      }
      add_to_dependence_queue(attach_op, launcher.static_dependences);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeNode* ReplicateContext::compute_index_attach_upper_bound(
        const IndexAttachLauncher& launcher,
        const std::vector<unsigned>& indexes)
    //--------------------------------------------------------------------------
    {
      // Call the base version if our indexes are not empty
      RegionTreeNode* result =
          indexes.empty() ?
              nullptr :
              InnerContext::compute_index_attach_upper_bound(launcher, indexes);
      // Do the exchange between the shards
      IndexAttachUpperBound exchange(this, COLLECTIVE_LOC_26);
      result = exchange.find_upper_bound(result);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::detach_resource(
        PhysicalRegion region, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_DETACH_RESOURCE, __func__);
        Serializer rez;
        if (region.impl != nullptr)
          ExternalMappable::pack_region_requirement(
              region.impl->get_requirement(), rez);
        hasher.hash(rez.get_buffer(), rez.get_used_bytes(), "requirement");
        hasher.hash<bool>(region.is_mapped(), "is_mapped");
        hasher.hash<bool>(flush, "flush");
        if (hasher.verify(__func__))
          break;
      }
      ReplDetachOp* op = runtime->get_operation<ReplDetachOp>();
      Future result =
          op->initialize_detach(this, region, flush, unordered, provenance);
      op->initialize_replication(
          this, region.impl->collective,
          shard_manager->is_first_local_shard(owner_shard));
      // If the region is still mapped, then unmap it
      if (region.is_mapped())
      {
        unregister_inline_mapped_region(region);
        region.impl->unmap_region(false /*escaped*/);
      }
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal unordered detach operation performed after task "
              << get_task_name() << " (UID " << get_unique_id()
              << ") has finished executing. All unordered operations must be "
                 "performed "
              << "before the end of the execution of the parent task.";
        error.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::detach_resources(
        ExternalResources resources, const bool flush, const bool unordered,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && !unordered && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_INDEX_DETACH_RESOURCE, __func__);
        if (resources.impl != nullptr)
        {
          hasher.hash(resources.impl->parent, "parent");
          for (const FieldID& it : resources.impl->privilege_fields)
            hasher.hash(it, "privilege_fields");
          if (resources.impl->upper_bound->is_region())
            hasher.hash(
                resources.impl->upper_bound->as_region_node()->handle,
                "region");
          else
            hasher.hash(
                resources.impl->upper_bound->as_partition_node()->handle,
                "partition");
        }
        hasher.hash<bool>(flush, "flush");
        if (hasher.verify(__func__))
          break;
      }
      if (resources.impl == nullptr)
        return Future();
      ReplIndexDetachOp* op = runtime->get_operation<ReplIndexDetachOp>();
      Future result =
          resources.impl->detach(this, op, flush, unordered, provenance);
      op->initialize_replication(this);
      if (!add_to_dependence_queue(op, nullptr /*deps*/, unordered))
      {
        legion_assert(unordered);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal unordered index detach operation performed after task "
            << get_task_name() << " (UID " << get_unique_id()
            << ") has finished executing. All unordered operations must be "
               "performed "
            << "before the end of the execution of the parent task.";
        error.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplicateContext::execute_must_epoch(
        const MustEpochLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_MUST_EPOCH, __func__);
        hasher.hash(launcher.map_id, "map_id");
        hasher.hash(launcher.mapping_tag, "mapping_tag");
        for (const TaskLauncher& it : launcher.single_tasks)
          hash_task_launcher(hasher, runtime->safe_control_replication, it);
        for (const IndexTaskLauncher& it : launcher.index_tasks)
          hash_index_launcher(hasher, runtime->safe_control_replication, it);
        hasher.hash(launcher.launch_domain, "launch_domain");
        hasher.hash(launcher.launch_space, "launch_space");
        hasher.hash(launcher.sharding_space, "sharding_space");
        hasher.hash(launcher.silence_warnings, "silence_warnings");
        if (hasher.verify(__func__))
          break;
      }
      ReplMustEpochOp* epoch_op = runtime->get_operation<ReplMustEpochOp>();
      FutureMap result = epoch_op->initialize(this, launcher, provenance);
      if (runtime->safe_mapper)
        epoch_op->set_sharding_collective(new ShardingGatherCollective(
            this, 0 /*owner shard*/, COLLECTIVE_LOC_49));
      epoch_op->initialize_replication(this);
      // Now find all the parent task regions we need to invalidate
      std::vector<PhysicalRegion> unmapped_regions;
      if (!runtime->unsafe_launch)
        epoch_op->find_conflicted_regions(unmapped_regions);
      if (!unmapped_regions.empty())
      {
        if (runtime->runtime_warnings && !launcher.silence_warnings)
        {
          Warning warning;
          warning << "Runtime is unmapping and remapping "
                  << "physical regions around issue_release call in task"
                  << *this << ".";
          warning.raise();
        }
        for (unsigned idx = 0; idx < unmapped_regions.size(); idx++)
          unmapped_regions[idx].impl->unmap_region(false /*escaped*/);
      }
      // Now we can issue the must epoch
      add_to_dependence_queue(epoch_op);
      // Remap any unmapped regions
      if (!unmapped_regions.empty())
        remap_unmapped_regions(current_trace, unmapped_regions, provenance);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::issue_timing_measurement(
        const TimingLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_TIMING_MEASUREMENT, __func__);
        hasher.hash(launcher.measurement, "measurement");
        for (const Future& it : launcher.preconditions)
          hash_future(
              hasher, runtime->safe_control_replication, it, "preconditions");
        if (hasher.verify(__func__))
          break;
      }
      ReplTimingOp* timing_op = runtime->get_operation<ReplTimingOp>();
      Future result = timing_op->initialize(this, launcher, provenance);
      ValueBroadcast<long long>* timing_collective =
          new ValueBroadcast<long long>(
              this, 0 /*shard 0 is always the owner*/, COLLECTIVE_LOC_35);
      timing_op->set_timing_collective(timing_collective);
      add_to_dependence_queue(timing_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::select_tunable_value(
        const TunableLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_TUNABLE_SELECTION, __func__);
        hasher.hash(launcher.tunable, "tunable");
        hasher.hash(launcher.mapper, "mapper");
        hasher.hash(launcher.tag, "tag");
        hash_argument(
            hasher, runtime->safe_control_replication, launcher.arg, "arg");
        for (const Future& it : launcher.futures)
          hash_future(hasher, runtime->safe_control_replication, it, "futures");
        if (hasher.verify(__func__))
          break;
      }
      ReplTunableOp* tunable_op = runtime->get_operation<ReplTunableOp>();
      Future result = tunable_op->initialize(this, launcher, provenance);
      tunable_op->initialize_replication(this);
      add_to_dependence_queue(tunable_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::issue_mapping_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_MAPPING_FENCE, __func__);
        if (hasher.verify(__func__))
          break;
      }
      ReplFenceOp* fence_op = runtime->get_operation<ReplFenceOp>();
      Future result =
          fence_op->initialize(this, FenceOp::MAPPING_FENCE, true, provenance);
      add_to_dependence_queue(fence_op);
      return result;
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::issue_execution_fence(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_EXECUTION_FENCE, __func__);
        if (hasher.verify(__func__))
          break;
      }
      ReplFenceOp* fence_op = runtime->get_operation<ReplFenceOp>();
      Future result = fence_op->initialize(
          this, FenceOp::EXECUTION_FENCE, true, provenance);
      add_to_dependence_queue(fence_op);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::begin_trace(
        TraceID tid, bool logical_only, bool static_trace,
        const std::set<RegionTreeID>* trees, bool deprecated,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_BEGIN_TRACE, __func__);
        hasher.hash(tid, "tid");
        hasher.hash<bool>(logical_only, "logical_only");
        hasher.hash<bool>(static_trace, "static_trace");
        hasher.hash<bool>(deprecated, "deprecated");
        if (trees != nullptr)
          for (const RegionTreeID& it : *trees) hasher.hash(it, "trees");
        if (hasher.verify(__func__))
          break;
      }
      if (runtime->no_tracing)
        return;
      if (runtime->no_physical_tracing)
        logical_only = true;
      // No need to hold the lock here, this is only ever called
      // by the one thread that is running the task.
      if (current_trace != nullptr)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal nested trace with ID " << tid << " attempted in task "
              << get_task_name() << " (ID " << get_unique_id() << ").";
        error.raise();
      }
      LogicalTrace* trace = nullptr;
      std::map<TraceID, LogicalTrace*>::const_iterator finder =
          traces.find(tid);
      if (finder == traces.end())
      {
        // Trace does not exist yet, so make one and record it
        trace = new LogicalTrace(
            this, tid, logical_only, static_trace, provenance, trees);
        trace->add_reference();
        if (!deprecated)
        {
          // Need the lock her to avoid look-ups racing with modifying
          // the trace data structure
          AutoLock t_lock(trace_lock);
          traces[tid] = trace;
        }
      }
      else
        trace = finder->second;
      legion_assert(trace != nullptr);
      ReplTraceOp* trace_op = nullptr;
      if (previous_trace == nullptr)
      {
        // Issue a begin op
        ReplTraceBeginOp* begin = runtime->get_operation<ReplTraceBeginOp>();
        begin->initialize_begin(this, trace, provenance);
        trace_op = begin;
      }
      else
      {
        ReplTraceRecurrentOp* recurrent =
            runtime->get_operation<ReplTraceRecurrentOp>();
        recurrent->initialize_recurrent(
            this, trace, previous_trace, provenance,
            (traces.find(previous_trace->tid) == traces.end()));
        trace_op = recurrent;
        previous_trace = nullptr;
      }
      if (trace->is_fixed() && trace->has_physical_trace())
      {
        // Record the event for when the trace replay is ready
        physical_trace_replay_status.store(trace_op->get_mapped_event().id);
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          tracing_replay_event = trace_op->get_completion_event();
      }
      add_to_dependence_queue(trace_op);
      // Now mark that we are starting a trace
      current_trace = trace;
      current_trace_blocking_index = next_blocking_index;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::end_trace(
        TraceID tid, bool deprecated, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
      {
        HashVerifier hasher(
            this, runtime->safe_control_replication > 1, i > 0, provenance);
        hasher.hash(REPLICATE_END_TRACE, __func__);
        hasher.hash(tid, "tid");
        hasher.hash<bool>(deprecated, "deprecated");
        if (hasher.verify(__func__))
          break;
      }
      InnerContext::end_trace(tid, deprecated, provenance);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::wait_on_future(FutureImpl* future, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_FUTURE_WAIT, __func__);
        hash_future(
            hasher, runtime->safe_control_replication, Future(future),
            "future");
        if (hasher.verify(__func__))
          break;
      }
      const RtBarrier wait_bar = get_next_future_wait_barrier();
      runtime->phase_barrier_arrive(wait_bar, 1 /*count*/, ready);
      wait_bar.wait();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::wait_on_future_map(FutureMapImpl* map, RtEvent ready)
    //--------------------------------------------------------------------------
    {
      for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_FUTURE_MAP_WAIT_ALL_FUTURES, __func__);
        hash_future_map(hasher, FutureMap(map), "future map");
        if (hasher.verify(__func__))
          break;
      }
      const RtBarrier wait_bar = get_next_future_wait_barrier();
      runtime->phase_barrier_arrive(wait_bar, 1 /*count*/, ready);
      wait_bar.wait();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::end_task(
        const void* res, size_t res_size, bool owned,
        PhysicalInstance deferred_result_instance,
        FutureFunctor* callback_functor,
        const Realm::ExternalInstanceResource* resource,
        void (*freefunc)(const Realm::ExternalInstanceResource& resource),
        const void* metadataptr, size_t metadatasize, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // We have an extra one of these here to handle the case where some
      // shards do an extra runtime call than other shards. This should
      // avoid that case hanging at least.
      for (int i = 0; runtime->safe_control_replication && (i < 2); i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_END_TASK, __func__);
        hasher.hash(res_size, "res_size");
        hasher.hash(metadatasize, "metadatasize");
        if (hasher.verify(__func__))
          break;
      }
      InnerContext::end_task(
          res, res_size, owned, deferred_result_instance, callback_functor,
          resource, freefunc, metadataptr, metadatasize, effects);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::post_end_task(void)
    //--------------------------------------------------------------------------
    {
      // Pull any pending collectives here on the stack so we can delete them
      // after the end task call, even though this context might be reclaimed
      std::deque<std::pair<ValueBroadcast<ISBroadcast>*, bool> >
          release_index_spaces;
      if (!pending_index_spaces.empty())
        release_index_spaces.swap(pending_index_spaces);
      std::deque<std::pair<ValueBroadcast<IPBroadcast>*, ShardID> >
          release_index_partitions;
      if (!pending_index_partitions.empty())
        release_index_partitions.swap(pending_index_partitions);
      std::deque<std::pair<ValueBroadcast<FSBroadcast>*, bool> >
          release_field_spaces;
      if (!pending_field_spaces.empty())
        release_field_spaces.swap(pending_field_spaces);
      std::deque<std::pair<ValueBroadcast<FIDBroadcast>*, bool> >
          release_fields;
      if (!pending_fields.empty())
        release_fields.swap(pending_fields);
      std::deque<std::pair<ValueBroadcast<LRBroadcast>*, bool> >
          release_region_trees;
      if (!pending_region_trees.empty())
        release_region_trees.swap(pending_region_trees);
      std::deque<std::pair<ValueBroadcast<DIDBroadcast>*, bool> >
          release_distributed_ids;
      if (!pending_distributed_ids.empty())
        release_distributed_ids.swap(pending_distributed_ids);
      // Grab this now before the context might be deleted
      const ShardID local_shard = owner_shard->shard_id;
      // Do the base call
      InnerContext::post_end_task();
      // Then delete all the pending collectives that we had
      while (!release_index_spaces.empty())
      {
        std::pair<ValueBroadcast<ISBroadcast>*, bool>& collective =
            release_index_spaces.front();
        if (collective.second)
        {
          const ISBroadcast value = collective.first->get_value(false);
          runtime->revoke_pending_index_space(value.did);
        }
        else
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_index_spaces.pop_front();
      }
      while (!release_index_partitions.empty())
      {
        std::pair<ValueBroadcast<IPBroadcast>*, ShardID>& collective =
            release_index_partitions.front();
        if (collective.second == local_shard)
        {
          const IPBroadcast value = collective.first->get_value(false);
          runtime->revoke_pending_partition(value.did);
        }
        else
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_index_partitions.pop_front();
      }
      while (!release_field_spaces.empty())
      {
        std::pair<ValueBroadcast<FSBroadcast>*, bool>& collective =
            release_field_spaces.front();
        if (collective.second)
        {
          const FSBroadcast value = collective.first->get_value(false);
          runtime->revoke_pending_field_space(value.did);
        }
        else
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_field_spaces.pop_front();
      }
      while (!release_fields.empty())
      {
        std::pair<ValueBroadcast<FIDBroadcast>*, bool>& collective =
            release_fields.front();
        if (!collective.second)
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_fields.pop_front();
      }
      while (!release_region_trees.empty())
      {
        std::pair<ValueBroadcast<LRBroadcast>*, bool>& collective =
            release_region_trees.front();
        if (collective.second)
        {
          const LRBroadcast value = collective.first->get_value(false);
          runtime->revoke_pending_region_tree(value.tid);
        }
        else
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_region_trees.pop_front();
      }
      while (!release_distributed_ids.empty())
      {
        std::pair<ValueBroadcast<DIDBroadcast>*, bool>& collective =
            release_distributed_ids.front();
        if (!collective.second)
        {
          // Make sure this collective is done before we delete it
          const RtEvent done = collective.first->get_done_event();
          if (!done.has_triggered())
            done.wait();
        }
        delete collective.first;
        release_distributed_ids.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::add_to_dependence_queue(
        Operation* op, const std::vector<StaticDependence>* dependences,
        bool unordered, bool outermost)
    //--------------------------------------------------------------------------
    {
      // We disable program order execution when we are replaying a
      // fixed trace since it might not be sound to block
      if (runtime->program_order_execution && !unordered && outermost &&
          !is_replaying_physical_trace())
      {
        const RtEvent commit_event = op->get_commit_event();
        InnerContext::add_to_dependence_queue(
            op, dependences, unordered, false /*outermost*/);
        const RtBarrier inorder_bar = inorder_barrier.next(this);
        runtime->phase_barrier_arrive(inorder_bar, 1 /*count*/, commit_event);
        inorder_bar.wait();
        // Issue any unordered operations now
        AutoLock d_lock(dependence_lock);
        insert_unordered_ops(d_lock);
        // Not unordered so it must have succeeded
        return true;
      }
      else
        return InnerContext::add_to_dependence_queue(
            op, dependences, unordered, outermost);
    }

    //--------------------------------------------------------------------------
    FenceOp* ReplicateContext::initialize_trace_completion(Provenance* prov)
    //--------------------------------------------------------------------------
    {
      legion_assert(previous_trace != nullptr);
      ReplTraceCompleteOp* op = runtime->get_operation<ReplTraceCompleteOp>();
      op->initialize_complete(
          this, previous_trace, prov,
          (traces.find(previous_trace->tid) == traces.end()));
      return op;
    }

    //--------------------------------------------------------------------------
    PredicateImpl* ReplicateContext::create_predicate_impl(Operation* op)
    //--------------------------------------------------------------------------
    {
      return new ReplPredicateImpl(
          op, get_next_blocking_index(),
          get_next_collective_index(COLLECTIVE_LOC_1));
    }

    //--------------------------------------------------------------------------
    InnerContext::CollectiveResult*
        ReplicateContext::find_or_create_collective_view(
            RegionTreeID tid, const std::vector<DistributedID>& instances,
            RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(instances.size() > 1);
      // Find which shard is the owner
      const ShardID tid_shard = shard_manager->find_collective_owner(tid);
      if (tid_shard != owner_shard->shard_id)
      {
        const RtUserEvent to_trigger = Runtime::create_rt_user_event();
        CollectiveResult* result = new CollectiveResult(instances);
        result->add_reference();
        ReplFindCollectiveView rez;
        rez.serialize(shard_manager->did);
        rez.serialize(tid_shard);
        rez.serialize(tid);
        rez.serialize<size_t>(instances.size());
        for (unsigned idx = 0; idx < instances.size(); idx++)
          rez.serialize(instances[idx]);
        rez.serialize(result);
        rez.serialize(runtime->address_space);
        rez.serialize(to_trigger);
        shard_manager->send_find_or_create_collective_view(tid_shard, rez);
        ready = to_trigger;
        return result;
      }
      else
        return InnerContext::find_or_create_collective_view(
            tid, instances, ready);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary* ReplicateContext::construct_projection_summary(
        Operation* op, unsigned index, const RegionRequirement& req,
        LogicalState* state, const ProjectionInfo& proj_info)
    //--------------------------------------------------------------------------
    {
      const ShardID local_shard = owner_shard->shard_id;
      ProjectionNode* result = proj_info.projection->construct_projection_tree(
          op, index, req, local_shard, state->owner, proj_info);
      // Now we need to exchange this between the shards. The secret to this
      // function is knowing that it is only called in the logical dependence
      // analysis stage of the pipeline so we can get a collective ID here to
      // perform the exchange between the shards of their neighbor sharding
      // information. This is unfortunately a blocking process, but it should
      // be memoized in most cases to reduce the latency of it happening
      if (req.projection == 0)
      {
        // For the identity projection function we know how to compute this
        // without performing any communication between the shards
        IndexSpaceNode* launch_space = proj_info.projection_space;
        Domain launch_domain = launch_space->get_tight_domain();
        Domain shard_domain;
        if (proj_info.sharding_space != nullptr)
          shard_domain = proj_info.sharding_space->get_tight_domain();
        else
          shard_domain = launch_domain;
        if (state->owner->is_region())
        {
          // Iterate all the points in the launch space and compute their
          // shards and record them in the shard users
          ProjectionRegion* projection = result->as_region_projection();
          for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
          {
            const ShardID shard =
                proj_info.sharding_function->find_owner(*itr, shard_domain);
            projection->add_user(shard);
          }
          return new ProjectionSummary(
              proj_info, result, op, index, req, state, true /*disjoint*/,
              false /*unique*/);
        }
        else
        {
          // Iterate all the points in the launch space and linearize
          // their colors to add to the summary
          IndexPartNode* partition =
              state->owner->as_partition_node()->row_source;
          IndexSpaceNode* color_space = partition->color_space;
          ProjectionPartition* projection = result->as_partition_projection();
          for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
          {
            const ShardID shard =
                proj_info.sharding_function->find_owner(*itr, shard_domain);
            if (shard == local_shard)
              continue;
            const LegionColor color = color_space->linearize_color(*itr);
            legion_assert(
                projection->local_children.find(color) ==
                projection->local_children.end());
            projection->shard_children.add_child(color);
          }
          const bool disjoint = partition->is_disjoint(false /*from app*/);
          return new ProjectionSummary(
              proj_info, result, op, index, req, state, disjoint,
              true /*unique*/);
        }
      }
      else
        return new ProjectionSummary(
            proj_info, result, op, index, req, state, this);
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::has_interfering_shards(
        ProjectionSummary* one, ProjectionSummary* two, bool& dominates)
    //--------------------------------------------------------------------------
    {
      legion_assert(dominates);
      const bool result = one->get_tree()->interferes(
          two->get_tree(), owner_shard->shard_id, dominates);
      // This is a bit tricky so pay attention: we're going to exchange both
      // the interference result and the dominates result using a single
      // all-reduce collective. We can do this because the interfering shards
      // result is a bool OR all-reduce and the result of dominates is a
      // bool AND all-reduce. Therefore we can encode this as a single max
      // all-reduce computation. The states will be:
      // 0: not-interfering and dominating
      // 1: not-interfering and not-dominating
      // 2: interfering (dominate doesn't matter in this case)
      // We can then do a max all-reduce and determine if all the shards
      // are non-interfering and dominating
      uint32_t code = (result ? 2 : (dominates ? 0 : 1));
      // Now we need to perform a collective to make sure that all the
      // shards agree on the result of the interference
      AllReduceCollective<MaxReduction<uint32_t>, false> any_interfering(
          this,
          get_next_collective_index(COLLECTIVE_LOC_105, true /*logical*/));
      code = any_interfering.sync_all_reduce(code);
      legion_assert(code <= 2);
      if (code == 0)
      {
        dominates = true;
        return false;
      }
      else if (code == 1)
      {
        dominates = false;
        return false;
      }
      else
        return true;
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::match_timeouts(
        std::vector<LogicalUser*>& timeouts, TimeoutMatchExchange*& exchange)
    //--------------------------------------------------------------------------
    {
      bool previous_ready = true;
      bool double_latency = false;
      std::vector<std::pair<size_t, unsigned> > remaining_timeouts;
      if (exchange != nullptr)
      {
        RtEvent ready = exchange->perform_collective_wait(false /*block*/);
        if (ready.exists() && !ready.has_triggered())
        {
          previous_ready = false;
          ready.wait();
        }
        double_latency =
            exchange->complete_exchange(timeouts, remaining_timeouts);
        delete exchange;
      }
      else
      {
        // No prior exchange means that they all go in the remaining timeouts
        remaining_timeouts.reserve(timeouts.size());
        for (LogicalUser* user : timeouts)
          remaining_timeouts.emplace_back(
              std::make_pair(user->ctx_index, user->internal_idx));
        // We can't prune any of these users yet without consensus
        timeouts.clear();
      }
      exchange = new TimeoutMatchExchange(this, COLLECTIVE_LOC_79);
      exchange->perform_exchange(remaining_timeouts, previous_ready);
      return double_latency;
    }

    //--------------------------------------------------------------------------
    std::pair<bool, bool> ReplicateContext::has_pointwise_dominance(
        ProjectionSummary* one, ProjectionSummary* two)
    //--------------------------------------------------------------------------
    {
      std::pair<bool, bool> local =
          InnerContext::has_pointwise_dominance(one, two);
      PointwiseAllreduce allreduce(
          this, get_next_collective_index(COLLECTIVE_LOC_109, true /*logical*/),
          local);
      allreduce.perform_collective_sync();
      return local;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplicateContext::find_pointwise_dependence(
        uint64_t context_index, const DomainPoint& point, ShardID shard,
        RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      Operation* op;
      GenerationID gen;
      if (shard == owner_shard->shard_id)
      {
        AutoLock child_lock(child_op_lock);
        if (reorder_buffer.empty())
        {
          // Already been retired so there is nothing to do
          if (context_index < total_children_count)
          {
            if (to_trigger.exists())
              Runtime::trigger_event(to_trigger);
            return RtEvent::NO_RT_EVENT;
          }
          // Since shards execute independently, we could get a request
          // for a pointwise dependence for an operation that this shard
          // has not even created yet
          std::map<DomainPoint, RtUserEvent>& pending_points =
              pending_pointwise_dependences[context_index];
          std::map<DomainPoint, RtUserEvent>::const_iterator finder =
              pending_points.find(point);
          if (finder == pending_points.end())
          {
            if (!to_trigger.exists())
              to_trigger = Runtime::create_rt_user_event();
            pending_points.emplace(std::make_pair(point, to_trigger));
            return to_trigger;
          }
          else
          {
            if (to_trigger.exists())
              Runtime::trigger_event(to_trigger, finder->second);
            return finder->second;
          }
        }
        // Operation has already been retired
        if (context_index < reorder_buffer.front().operation_index)
        {
          if (to_trigger.exists())
            Runtime::trigger_event(to_trigger);
          return RtEvent::NO_RT_EVENT;
        }
        if (reorder_buffer.back().operation_index < context_index)
        {
          // Since shards execute independently, we could get a request
          // for a pointwise dependence for an operation that this shard
          // has not even created yet
          std::map<DomainPoint, RtUserEvent>& pending_points =
              pending_pointwise_dependences[context_index];
          std::map<DomainPoint, RtUserEvent>::const_iterator finder =
              pending_points.find(point);
          if (finder == pending_points.end())
          {
            if (!to_trigger.exists())
              to_trigger = Runtime::create_rt_user_event();
            pending_points.emplace(std::make_pair(point, to_trigger));
            return to_trigger;
          }
          else
          {
            if (to_trigger.exists())
              Runtime::trigger_event(to_trigger, finder->second);
            return finder->second;
          }
        }
        size_t offset = context_index - reorder_buffer.front().operation_index;
        const ReorderBufferEntry& entry = reorder_buffer[offset];
        if (entry.complete)
        {
          if (to_trigger.exists())
            Runtime::trigger_event(to_trigger);
          return RtEvent::NO_RT_EVENT;
        }
        legion_assert(entry.operation_index == context_index);
        op = entry.operation;
        // Have to do this while holding the lock to ensure it isn't
        // committed while we're getting the generation
        gen = op->get_generation();
        // Fall through and do the local call without the lock
      }
      else
        return shard_manager->find_pointwise_dependence(
            context_index, point, shard, to_trigger);
      return op->find_pointwise_dependence(point, gen, to_trigger);
    }

    //--------------------------------------------------------------------------
    FillView* ReplicateContext::find_or_create_fill_view(
        FillOp* op, const void* value, size_t value_size, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      // Two versions of this method depending on whether we are doing
      // Legion Spy or not, Legion Spy wants to know exactly which op
      // made each fill view so we can't cache them
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        // See if we can find this in the cache first
        // Need the lock to protect against races with collectives coming back
        AutoLock f_lock(fill_view_lock);
        for (std::list<FillViewEntry>::iterator it = fill_view_cache.begin();
             it != fill_view_cache.end(); it++)
        {
          // Skip comparing against fill views from futures
          if (it->future_did > 0)
            continue;
          if (it->collective != nullptr)
          {
            if (!it->collective->matches(value, value_size))
              continue;
            it->pending_references++;
            ready = it->collective->get_done_event();
          }
          // Safe to do this since we know we're only comparing against other
          // fill views that were also made with eager values
          else if (!it->view->matches(value, value_size))
            continue;
          else  // Safe to record a reference on it
            it->view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
          // Move it back to the front of the list
          fill_view_cache.splice(fill_view_cache.begin(), fill_view_cache, it);
          return it->view;
        }
      }
      // Deduplicate the allocation so we have a name for the fill view
      // This is just the allocation
      void* allocation = shard_manager->deduplicate_fill_view_allocation(
          op->get_context_index());
      CollectiveID collective_id =
          get_next_collective_index(COLLECTIVE_LOC_77, true /*logical*/);
      // At this point we have to make a fill view using a collective
      CreateCollectiveFillView* collective = new CreateCollectiveFillView(
          this, collective_id, allocation, op->get_unique_op_id(), value,
          value_size);
      if (collective->is_origin())
      {
        // Make the fill view and then broadcast it out
        DistributedID fresh_did = runtime->get_available_distributed_id();
        // This comes back with a MAPPING_ACQUIRE_REF already held
        bool first_arrival = false;
        FillView* result = shard_manager->deduplicate_fill_view_creation(
            allocation, fresh_did, op->get_unique_op_id(), &first_arrival);
        legion_assert(first_arrival);
        if (first_arrival)
          result->set_value(value, value_size);
        collective->broadcast_fill_view_did(fresh_did);
        delete collective;
        if (spy_logging_level <= LIGHT_SPY_LOGGING)
        {
          result->add_nested_valid_ref(did);
          AutoLock f_lock(fill_view_lock);
          fill_view_cache.emplace_front(
              FillViewEntry{result, 0 /*future did*/, nullptr, 0});
          if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
          {
            FillViewEntry& last = fill_view_cache.back();
            // See if it is done or not
            if (last.collective == nullptr)
            {
              if (last.view->remove_nested_valid_ref(did))
                delete last.view;
            }
            else if (last.pending_references > 0)
              pending_fill_views.emplace(last.view, last.pending_references);
            fill_view_cache.pop_back();
          }
        }
        return result;
      }
      else
      {
        // Static cast this as it will be constructed later
        FillView* result = static_cast<FillView*>(allocation);
        // Save this in the right cache for when the result comes back
        if (spy_logging_level <= LIGHT_SPY_LOGGING)
        {
          // Need the lock to modify the collective entry
          AutoLock f_lock(fill_view_lock);
          fill_view_cache.emplace_front(
              FillViewEntry{result, 0 /*future did*/, collective, 0});
          if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
          {
            FillViewEntry& last = fill_view_cache.back();
            // See if it is done or not
            if (last.collective == nullptr)
            {
              if (last.view->remove_nested_valid_ref(did))
                delete last.view;
            }
            else if (last.pending_references > 0)
              pending_fill_views.emplace(last.view, last.pending_references);
            fill_view_cache.pop_back();
          }
        }
        ready = collective->perform_collective_wait(false /*block*/);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    FillView* ReplicateContext::find_or_create_fill_view(
        FillOp* op, const Future& future, bool& set_value, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(!set_value);
      const DistributedID future_did = future.impl->did;
      legion_assert(future_did != 0);
      // Two versions of this method depending on whether we are doing
      // Legion Spy or not, Legion Spy wants to know exactly which op
      // made each fill view so we can't cache them
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        // See if we can find this in the cache first
        // Need the lock to protect against races with collectives coming back
        for (std::list<FillViewEntry>::iterator it = fill_view_cache.begin();
             it != fill_view_cache.end(); it++)
        {
          if (future_did != it->future_did)
            continue;
          AutoLock f_lock(fill_view_lock);
          if (it->collective != nullptr)
          {
            it->pending_references++;
            ready = it->collective->get_done_event();
          }
          else
            it->view->add_base_valid_ref(MAPPING_ACQUIRE_REF);
          // Move it back to the front of the list
          fill_view_cache.splice(fill_view_cache.begin(), fill_view_cache, it);
          return it->view;
        }
      }
      // Deduplicate the allocation so we have a name for the fill view
      // This is just the allocation
      void* allocation = shard_manager->deduplicate_fill_view_allocation(
          op->get_context_index(), &set_value);
      CollectiveID collective_id =
          get_next_collective_index(COLLECTIVE_LOC_93, true /*logical*/);
      // At this point we have to make a fill view using a collective
      CreateCollectiveFillView* collective = new CreateCollectiveFillView(
          this, collective_id, allocation, op->get_unique_op_id());
      if (collective->is_origin())
      {
        // Make the fill view and then broadcast it out
        DistributedID fresh_did = runtime->get_available_distributed_id();
        // This comes back with a MAPPING_ACQUIRE_REF already held
        FillView* result = shard_manager->deduplicate_fill_view_creation(
            allocation, fresh_did, op->get_unique_op_id());
        collective->broadcast_fill_view_did(fresh_did);
        delete collective;
        if (spy_logging_level <= LIGHT_SPY_LOGGING)
        {
          result->add_nested_valid_ref(did);
          AutoLock f_lock(fill_view_lock);
          fill_view_cache.emplace_front(
              FillViewEntry{result, future_did, nullptr, 0});
          if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
          {
            FillViewEntry& last = fill_view_cache.back();
            // See if it is done or not
            if (last.collective == nullptr)
            {
              if (last.view->remove_nested_valid_ref(did))
                delete last.view;
            }
            else if (last.pending_references > 0)
              pending_fill_views.emplace(last.view, last.pending_references);
            fill_view_cache.pop_back();
          }
        }
        return result;
      }
      else
      {
        // Static cast this as it will be constructed later
        FillView* result = static_cast<FillView*>(allocation);
        // Save this in the right cache for when the result comes back
        if (spy_logging_level <= LIGHT_SPY_LOGGING)
        {
          // Need the lock to modify the collective entry
          AutoLock f_lock(fill_view_lock);
          fill_view_cache.emplace_front(
              FillViewEntry{result, future_did, collective, 0});
          if (fill_view_cache.size() > MAX_FILL_VIEW_CACHE_SIZE)
          {
            FillViewEntry& last = fill_view_cache.back();
            // See if it is done or not
            if (last.collective == nullptr)
            {
              if (last.view->remove_nested_valid_ref(did))
                delete last.view;
            }
            else if (last.pending_references > 0)
              pending_fill_views.emplace(last.view, last.pending_references);
            fill_view_cache.pop_back();
          }
        }
        ready = collective->perform_collective_wait(false /*block*/);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::finalize_collective_fill_view(
        DistributedID view_did, void* allocation, UniqueID op_uid,
        const void* value, size_t value_size,
        CreateCollectiveFillView* collective)
    //--------------------------------------------------------------------------
    {
      // Deduplicate the fill view creation across the shards
      // This comes back with a MAPPING_ACQUIRE_REF on it
      bool first = false;
      FillView* view = shard_manager->deduplicate_fill_view_creation(
          allocation, view_did, op_uid, &first);
      if (first && (value != nullptr))
        view->set_value(value, value_size);
      AutoLock f_lock(fill_view_lock);
      // Delete the collective associated with this fill view
      // It's not safe to do this until we're holding the lock because other
      // threads might still be invoking the 'matches' function on collectives
      // in the list so we need to make sure none of them are outstanding
      delete collective;
      // Check the pending set first since that is easy
      std::map<FillView*, size_t>::iterator finder =
          pending_fill_views.find(view);
      if (finder != pending_fill_views.end())
      {
        view->add_base_valid_ref(MAPPING_ACQUIRE_REF, finder->second);
        pending_fill_views.erase(finder);
        return;
      }
      // Search first in the fill view cache
      for (FillViewEntry& entry : fill_view_cache)
      {
        if (entry.view != view)
          continue;
        legion_assert(entry.collective == collective);
        entry.collective = nullptr;
        if (entry.pending_references > 0)
          view->add_base_valid_ref(
              MAPPING_ACQUIRE_REF, entry.pending_references);
        // Also add our nested reference now
        view->add_nested_valid_ref(did);
        return;
      }
    }

    //--------------------------------------------------------------------------
    Lock ReplicateContext::create_lock(void)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal create lock operation performed in control replicated "
               "task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return Lock();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_lock(Lock l)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal destroy lock operation performed in control replicated "
               "task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Grant ReplicateContext::acquire_grant(
        const std::vector<LockRequest>& requests)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal acquire grant operation performed in control "
               "replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return Grant();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::release_grant(Grant g)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal release grant operation performed in control "
               "replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    PhaseBarrier ReplicateContext::create_phase_barrier(unsigned arrivals)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_CREATE_PHASE_BARRIER, __func__);
        hasher.hash(arrivals, "arrivals");
        if (hasher.verify(__func__))
          break;
      }
      ValueBroadcast<PhaseBarrier> bar_collective(
          this, 0 /*origin*/, COLLECTIVE_LOC_71);
      // Shard 0 will make the barrier and broadcast it
      if (owner_shard->shard_id == 0)
      {
        PhaseBarrier result = InnerContext::create_phase_barrier(arrivals);
        bar_collective.broadcast(result);
        return result;
      }
      else
        return bar_collective.get_value();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_DESTROY_PHASE_BARRIER, __func__);
        hasher.hash(pb.phase_barrier, "phase_barrier");
        if (hasher.verify(__func__))
          break;
      }
      // Shard 0 has to wait for all the other shards to get here
      // too before it can do the deletion
      ShardSyncTree sync_point(this, 0 /*origin*/, COLLECTIVE_LOC_72);
      sync_point.perform_collective_sync();
      if (owner_shard->shard_id == 0)
        InnerContext::destroy_phase_barrier(pb);
    }

    //--------------------------------------------------------------------------
    PhaseBarrier ReplicateContext::advance_phase_barrier(PhaseBarrier bar)
    //--------------------------------------------------------------------------
    {
      // For now we issue a mapping fence whenever we do this because
      // we do not have any logical dependence analysis on phase barriers
      issue_mapping_fence(nullptr /*provenance*/);
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_ADVANCE_PHASE_BARRIER, __func__);
        hasher.hash(bar, "bar");
        if (hasher.verify(__func__))
          break;
      }
      PhaseBarrier result = bar;
      Runtime::advance_barrier(result);
      if (owner_shard->shard_id == 0)
        LegionSpy::log_event_dependence(
            bar.phase_barrier, result.phase_barrier);
      return result;
    }

    //--------------------------------------------------------------------------
    DynamicCollective ReplicateContext::create_dynamic_collective(
        unsigned arrivals, ReductionOpID redop, const void* init_value,
        size_t init_size)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal create dynamic collective operation performed in "
               "control replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return DynamicCollective();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::destroy_dynamic_collective(DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal destroy dynamic collective operation performed in "
               "control replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::arrive_dynamic_collective(
        DynamicCollective dc, const void* buffer, size_t size, unsigned count)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal dynamic collective arrival operation performed in "
               "control replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::defer_dynamic_collective_arrival(
        DynamicCollective dc, const Future& f, unsigned count)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal defer dynamic collective arrival operation performed "
               "in control replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
    }

    //--------------------------------------------------------------------------
    Future ReplicateContext::get_dynamic_collective_result(
        DynamicCollective dc, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Illegal get dynamic collective result operation performed in "
               "control replicated task "
            << get_task_name() << " (UID " << get_unique_id() << ").";
      error.raise();
      return Future();
    }

    //--------------------------------------------------------------------------
    DynamicCollective ReplicateContext::advance_dynamic_collective(
        DynamicCollective dc)
    //--------------------------------------------------------------------------
    {
      // For now we issue a mapping fence whenever we do this because
      // we do not have any logical dependence analysis on phase barriers
      issue_mapping_fence(nullptr /*provenance*/);
      for (int i = 0;
           runtime->safe_control_replication && (i < 2) &&
           ((current_trace == nullptr) || !current_trace->is_fixed());
           i++)
      {
        HashVerifier hasher(this, runtime->safe_control_replication > 1, i > 0);
        hasher.hash(REPLICATE_ADVANCE_DYNAMIC_COLLECTIVE, __func__);
        hasher.hash(dc, "dc");
        if (hasher.verify(__func__))
          break;
      }
      DynamicCollective result = dc;
      Runtime::advance_barrier(result);
      if (owner_shard->shard_id == 0)
        LegionSpy::log_event_dependence(dc.phase_barrier, result.phase_barrier);
      return result;
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_COLLECTIVES
    MergeCloseOp* ReplicateContext::get_merge_close_op(
        Operation* op, RegionTreeNode* node)
#else
    MergeCloseOp* ReplicateContext::get_merge_close_op(void)
#endif
    //--------------------------------------------------------------------------
    {
      ReplMergeCloseOp* result = runtime->get_operation<ReplMergeCloseOp>();
      // Get the mapped barrier for the close operation
      const RtBarrier mapped_bar = get_next_close_mapped_barrier();
#ifdef LEGION_DEBUG_COLLECTIVES
      CloseCheckReduction::RHS barrier(
          op, mapped_bar, node, false /*read only*/);
      const RtBarrier close_check_bar = close_check_barrier.next(
          this, CloseCheckReduction::REDOP, &CloseCheckReduction::IDENTITY,
          sizeof(CloseCheckReduction::IDENTITY));
      runtime->phase_barrier_arrive(
          close_check_bar, 1 /*count*/, RtEvent::NO_RT_EVENT, &barrier,
          sizeof(barrier));
      close_check_bar.wait();
      CloseCheckReduction::RHS actual_barrier;
      legion_no_skip_assert(Runtime::get_barrier_result(
          close_check_bar, &actual_barrier, sizeof(actual_barrier)));
      legion_assert(actual_barrier == barrier);
#endif
      result->set_repl_close_info(mapped_bar);
      return result;
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_COLLECTIVES
    RefinementOp* ReplicateContext::get_refinement_op(
        Operation* op, RegionTreeNode* node)
#else
    RefinementOp* ReplicateContext::get_refinement_op(void)
#endif
    //--------------------------------------------------------------------------
    {
      ReplRefinementOp* result = runtime->get_operation<ReplRefinementOp>();
      // Get the mapped barrier for the refinement operation
      RtBarrier mapped_bar = get_next_refinement_mapped_barrier();
#ifdef LEGION_DEBUG_COLLECTIVES
      CloseCheckReduction::RHS barrier(
          op, mapped_bar, node, false /*read only*/);
      const RtBarrier refinement_check_bar = refinement_check_barrier.next(
          this, CloseCheckReduction::REDOP, &CloseCheckReduction::IDENTITY,
          sizeof(CloseCheckReduction::IDENTITY));
      runtime->phase_barrier_arrive(
          refinement_check_bar, 1 /*count*/, RtEvent::NO_RT_EVENT, &barrier,
          sizeof(barrier));
      refinement_check_bar.wait();
      CloseCheckReduction::RHS actual_barrier;
      legion_no_skip_assert(Runtime::get_barrier_result(
          refinement_check_bar, &actual_barrier, sizeof(actual_barrier)));
      legion_assert(actual_barrier == barrier);
#endif
      const RtBarrier next_refinement_bar = get_next_refinement_barrier();
      result->set_repl_refinement_info(mapped_bar, next_refinement_bar);
      return result;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::pack_task_context(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(did);  // pack our distributed ID
      rez.serialize<DistributedID>(shard_manager->did);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::pack_remote_context(
        Serializer& rez, AddressSpaceID target, bool replicate)
    //--------------------------------------------------------------------------
    {
      // Do the normal inner pack with replicate true
      InnerContext::pack_remote_context(rez, target, true /*replicate*/);
      // Then pack our additional information
      rez.serialize(owner_shard->shard_id);
      rez.serialize<size_t>(total_shards);
      rez.serialize(shard_manager->shard_points[owner_shard->shard_id]);
      rez.serialize(shard_manager->shard_domain);
      rez.serialize(shard_manager->did);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_collective_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardCollective* collective = find_or_buffer_collective(derez);
      if (collective != nullptr)
        collective->handle_collective_message(derez);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_rendezvous(ShardRendezvous* rendezvous)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<void*, size_t> > to_handle;
      {
        AutoLock repl_lock(replication_lock);
        legion_assert(
            shard_rendezvous.find(rendezvous->origin_shard) ==
            shard_rendezvous.end());
        shard_rendezvous[rendezvous->origin_shard] = rendezvous;
        std::map<ShardID, std::vector<std::pair<void*, size_t> > >::iterator
            finder = pending_rendezvous_updates.find(rendezvous->origin_shard);
        if (finder != pending_rendezvous_updates.end())
        {
          to_handle.swap(finder->second);
          pending_rendezvous_updates.erase(finder);
        }
      }
      for (const std::pair<void*, size_t>& it : to_handle)
      {
        Deserializer derez(it.first, it.second);
        if (rendezvous->receive_message(derez))
        {
          AutoLock repl_lock(replication_lock);
          std::map<ShardID, ShardRendezvous*>::iterator finder =
              shard_rendezvous.find(rendezvous->origin_shard);
          legion_assert(finder != shard_rendezvous.end());
          legion_assert(finder->second == rendezvous);
          shard_rendezvous.erase(finder);
        }
        free(it.first);
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_rendezvous_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardRendezvous* rendezvous = find_or_buffer_rendezvous(derez);
      if ((rendezvous != nullptr) && rendezvous->receive_message(derez))
      {
        AutoLock repl_lock(replication_lock);
        std::map<ShardID, ShardRendezvous*>::iterator finder =
            shard_rendezvous.find(rendezvous->origin_shard);
        legion_assert(finder != shard_rendezvous.end());
        legion_assert(finder->second == rendezvous);
        shard_rendezvous.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    ShardRendezvous* ReplicateContext::find_or_buffer_rendezvous(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardID origin_shard;
      derez.deserialize(origin_shard);
      AutoLock repl_lock(replication_lock);
      // See if we already have a rendezvous here to rendezvous with
      std::map<ShardID, ShardRendezvous*>::const_iterator finder =
          shard_rendezvous.find(origin_shard);
      if (finder != shard_rendezvous.end())
        return finder->second;
      // If we couldn't find it then we have to buffer it for the future
      const size_t remaining_bytes = derez.get_remaining_bytes();
      void* buffer = malloc(remaining_bytes);
      memcpy(buffer, derez.get_current_pointer(), remaining_bytes);
      derez.advance_pointer(remaining_bytes);
      pending_rendezvous_updates[origin_shard].emplace_back(
          std::pair<void*, size_t>(buffer, remaining_bytes));
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_resource_update(
        Deserializer& derez, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      uint64_t return_index;
      derez.deserialize(return_index);
      RtBarrier ready_barrier, mapped_barrier, execution_barrier;
      derez.deserialize(ready_barrier);
      derez.deserialize(mapped_barrier);
      derez.deserialize(execution_barrier);
      size_t num_created_regions;
      derez.deserialize(num_created_regions);
      std::map<LogicalRegion, unsigned> created_regs;
      for (unsigned idx = 0; idx < num_created_regions; idx++)
      {
        LogicalRegion reg;
        derez.deserialize(reg);
        derez.deserialize(created_regs[reg]);
      }
      size_t num_deleted_regions;
      derez.deserialize(num_deleted_regions);
      std::vector<DeletedRegion> deleted_regs(num_deleted_regions);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
        deleted_regs[idx].deserialize(derez);
      size_t num_created_fields;
      derez.deserialize(num_created_fields);
      std::set<std::pair<FieldSpace, FieldID> > created_fids;
      for (unsigned idx = 0; idx < num_created_fields; idx++)
      {
        std::pair<FieldSpace, FieldID> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        created_fids.insert(key);
      }
      size_t num_deleted_fields;
      derez.deserialize(num_deleted_fields);
      std::vector<DeletedField> deleted_fids(num_deleted_fields);
      for (unsigned idx = 0; idx < num_deleted_fields; idx++)
        deleted_fields[idx].deserialize(derez);
      size_t num_created_field_spaces;
      derez.deserialize(num_created_field_spaces);
      std::map<FieldSpace, unsigned> created_fs;
      for (unsigned idx = 0; idx < num_created_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_fs[sp]);
      }
      size_t num_latent_field_spaces;
      derez.deserialize(num_latent_field_spaces);
      std::map<FieldSpace, std::set<LogicalRegion> > latent_fs;
      for (unsigned idx = 0; idx < num_latent_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        std::set<LogicalRegion>& regions = latent_fs[sp];
        size_t num_regions;
        derez.deserialize(num_regions);
        for (unsigned idx2 = 0; idx2 < num_regions; idx2++)
        {
          LogicalRegion region;
          derez.deserialize(region);
          regions.insert(region);
        }
      }
      size_t num_deleted_field_spaces;
      derez.deserialize(num_deleted_field_spaces);
      std::vector<DeletedFieldSpace> deleted_fs(num_deleted_field_spaces);
      for (unsigned idx = 0; idx < num_deleted_field_spaces; idx++)
        deleted_fs[idx].deserialize(derez);
      size_t num_created_index_spaces;
      derez.deserialize(num_created_index_spaces);
      std::map<IndexSpace, unsigned> created_is;
      for (unsigned idx = 0; idx < num_created_index_spaces; idx++)
      {
        IndexSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_is[sp]);
      }
      size_t num_deleted_index_spaces;
      derez.deserialize(num_deleted_index_spaces);
      std::vector<DeletedIndexSpace> deleted_is(num_deleted_index_spaces);
      for (unsigned idx = 0; idx < num_deleted_index_spaces; idx++)
        deleted_is[idx].deserialize(derez);
      size_t num_created_index_partitions;
      derez.deserialize(num_created_index_partitions);
      std::map<IndexPartition, unsigned> created_partitions;
      for (unsigned idx = 0; idx < num_created_index_partitions; idx++)
      {
        IndexPartition ip;
        derez.deserialize(ip);
        derez.deserialize(created_partitions[ip]);
      }
      size_t num_deleted_index_partitions;
      derez.deserialize(num_deleted_index_partitions);
      std::vector<DeletedPartition> deleted_partitions(
          num_deleted_index_partitions);
      for (unsigned idx = 0; idx < num_deleted_index_partitions; idx++)
        deleted_partitions[idx].deserialize(derez);
      // Send this down to the base class to avoid re-broadcasting
      receive_replicate_resources(
          return_index, created_regs, deleted_regs, created_fids, deleted_fids,
          created_fs, latent_fs, deleted_fs, created_is, deleted_is,
          created_partitions, deleted_partitions, applied, ready_barrier,
          mapped_barrier, execution_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_created_region_contexts(
        Deserializer& derez, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<RegionNode*> created_nodes(num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        RegionNode* node = runtime->get_node(handle);
        created_nodes[idx1] = node;
        size_t num_sets;
        derez.deserialize(num_sets);
        std::vector<RtEvent> ready_events;
        local::FieldMaskMap<EquivalenceSet> eq_sets;
        for (unsigned idx2 = 0; idx2 < num_sets; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          EquivalenceSet* set =
              runtime->find_or_request_equivalence_set(did, ready);
          FieldMask mask;
          derez.deserialize(mask);
          eq_sets.insert(set, mask);
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
        if (!ready_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists() && !wait_on.has_triggered())
            wait_on.wait();
        }
        EqKDTree* current = nullptr;
        {
          AutoLock priv_lock(privilege_lock);
          unsigned index = add_created_region(
              handle, false /*task local*/, false /*output region*/);
          std::map<unsigned, EqKDRoot>::const_iterator finder =
              equivalence_set_trees.find(index);
          if (finder == equivalence_set_trees.end())
          {
            // If we're the first make the KD-tree here
            current =
                node->row_source->create_equivalence_set_kd_tree(total_shards);
            equivalence_set_trees.emplace(index, EqKDRoot(current));
          }
          else
            current = finder->second.tree;
        }
        const ShardID local_shard = get_shard_id();
        // Put the equivalence sets in the tree but in the previous set
        // of equivalence sets so new accesses will make new sets
        for (local::FieldMaskMap<EquivalenceSet>::const_iterator it =
                 eq_sets.begin();
             it != eq_sets.end(); it++)
        {
          it->first->set_expr->initialize_equivalence_set_kd_tree(
              current, it->first, it->second, local_shard, false /*current*/);
          it->first->unpack_global_ref();
        }
        // Remove the global reference that we held on the node
        node->unpack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_trace_update(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShardedPhysicalTemplate* tpl = find_or_buffer_trace_update(derez, source);
      // If the template is nullptr then the request was buffered
      if (tpl == nullptr)
        return;
      tpl->handle_trace_update(derez, source);
    }

    //--------------------------------------------------------------------------
    ApBarrier ReplicateContext::handle_find_trace_shard_event(
        size_t template_index, ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      ShardedPhysicalTemplate* physical_template = nullptr;
      {
        AutoLock r_lock(replication_lock);
        std::map<size_t, ShardedPhysicalTemplate*>::const_iterator finder =
            physical_templates.find(template_index);
        // If we can't find the template index that means it hasn't been
        // started here so it can't have produced the event we're looking for
        // Note it also can't have been reclaimed yet as all the shard
        // templates need to come to the same decision on whether they
        // are replayable before any of them can be deleted and so if one
        // is still tracing then they all are
        if (finder == physical_templates.end())
          return ApBarrier::NO_AP_BARRIER;
        physical_template = finder->second;
      }
      return physical_template->find_trace_shard_event(event, remote_shard);
    }

    //--------------------------------------------------------------------------
    ApBarrier ReplicateContext::handle_find_trace_shard_frontier(
        size_t template_index, ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      ShardedPhysicalTemplate* physical_template = nullptr;
      {
        AutoLock r_lock(replication_lock);
        std::map<size_t, ShardedPhysicalTemplate*>::const_iterator finder =
            physical_templates.find(template_index);
        // If we can't find the template index that means it hasn't been
        // started here so it can't have produced the event we're looking for
        // Note it also can't have been reclaimed yet as all the shard
        // templates need to come to the same decision on whether they
        // are replayable before any of them can be deleted and so if one
        // is still tracing then they all are
        if (finder == physical_templates.end())
          return ApBarrier::NO_AP_BARRIER;
        physical_template = finder->second;
      }
      return physical_template->find_trace_shard_frontier(event, remote_shard);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::record_intra_space_dependence(
        uint64_t context_index, const DomainPoint& point, RtEvent point_mapped,
        ShardID next_shard)
    //--------------------------------------------------------------------------
    {
      const std::pair<uint64_t, DomainPoint> key(context_index, point);
      AutoLock r_lock(replication_lock);
      IntraSpaceDeps& deps = intra_space_deps[key];
      // Check to see if someone has already registered this
      std::map<ShardID, RtUserEvent>::iterator finder =
          deps.pending_deps.find(next_shard);
      if (finder != deps.pending_deps.end())
      {
        Runtime::trigger_event(finder->second, point_mapped);
        deps.pending_deps.erase(finder);
        if (deps.pending_deps.empty() && deps.ready_deps.empty())
          intra_space_deps.erase(key);
      }
      else
      {
        // Not seen yet so just record our entry for this shard
        legion_assert(
            deps.ready_deps.find(next_shard) == deps.ready_deps.end());
        deps.ready_deps[next_shard] = point_mapped;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_intra_space_dependence(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      std::pair<uint64_t, DomainPoint> key;
      derez.deserialize(key.first);
      derez.deserialize(key.second);
      RtUserEvent pending_event;
      derez.deserialize(pending_event);
      ShardID requesting_shard;
      derez.deserialize(requesting_shard);

      AutoLock r_lock(replication_lock);
      IntraSpaceDeps& deps = intra_space_deps[key];
      // Check to see if someone has already registered this shard
      std::map<ShardID, RtEvent>::iterator finder =
          deps.ready_deps.find(requesting_shard);
      if (finder != deps.ready_deps.end())
      {
        Runtime::trigger_event(pending_event, finder->second);
        deps.ready_deps.erase(finder);
        if (deps.ready_deps.empty() && deps.pending_deps.empty())
          intra_space_deps.erase(key);
      }
      else
      {
        // Not seen yet so just record our entry for this shard
        legion_assert(
            deps.pending_deps.find(requesting_shard) ==
            deps.pending_deps.end());
        deps.pending_deps[requesting_shard] = pending_event;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::receive_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID> >& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion> >& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // We need to broadcast these updates out to other shards
      Serializer rez;
      // If we have any deletions make barriers for use with
      // the deletion operations we may need to perform
      if (!deleted_regs.empty() || !deleted_fids.empty() ||
          !deleted_fs.empty() || !deleted_is.empty() ||
          !deleted_partitions.empty())
      {
        if (!returned_resource_ready_barrier.exists())
          returned_resource_ready_barrier =
              runtime->create_rt_barrier(total_shards);
        if (!returned_resource_mapped_barrier.exists())
          returned_resource_mapped_barrier =
              runtime->create_rt_barrier(total_shards);
        if (!returned_resource_execution_barrier.exists())
          returned_resource_execution_barrier =
              runtime->create_rt_barrier(total_shards);
      }
      rez.serialize(return_index);
      rez.serialize(returned_resource_ready_barrier);
      rez.serialize(returned_resource_mapped_barrier);
      rez.serialize(returned_resource_execution_barrier);
      rez.serialize<size_t>(created_regs.size());
      if (!created_regs.empty())
      {
        for (const std::pair<const LogicalRegion, unsigned>& it : created_regs)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
      rez.serialize<size_t>(deleted_regs.size());
      if (!deleted_regs.empty())
      {
        for (const DeletedRegion& it : deleted_regs) it.serialize(rez);
      }
      rez.serialize<size_t>(created_fids.size());
      if (!created_fids.empty())
      {
        for (const std::pair<FieldSpace, FieldID>& it : created_fids)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
      rez.serialize<size_t>(deleted_fids.size());
      if (!deleted_fids.empty())
      {
        for (const DeletedField& it : deleted_fids) it.serialize(rez);
      }
      rez.serialize<size_t>(created_fs.size());
      if (!created_fs.empty())
      {
        for (const std::pair<const FieldSpace, unsigned>& it : created_fs)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
      rez.serialize<size_t>(latent_fs.size());
      if (!latent_fs.empty())
      {
        for (const std::pair<const FieldSpace, std::set<LogicalRegion> >& it :
             latent_fs)
        {
          rez.serialize(it.first);
          rez.serialize<size_t>(it.second.size());
          for (const LogicalRegion& it2 : it.second) rez.serialize(it2);
        }
      }
      rez.serialize<size_t>(deleted_fs.size());
      if (!deleted_fs.empty())
      {
        for (const DeletedFieldSpace& it : deleted_fs) it.serialize(rez);
      }
      rez.serialize<size_t>(created_is.size());
      if (!created_is.empty())
      {
        for (const std::pair<const IndexSpace, unsigned>& it : created_is)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
      rez.serialize<size_t>(deleted_is.size());
      if (!deleted_is.empty())
      {
        for (const DeletedIndexSpace& it : deleted_is) it.serialize(rez);
      }
      rez.serialize<size_t>(created_partitions.size());
      if (!created_partitions.empty())
      {
        for (const std::pair<const IndexPartition, unsigned>& it :
             created_partitions)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
      rez.serialize<size_t>(deleted_partitions.size());
      if (!deleted_partitions.empty())
      {
        for (const DeletedPartition& it : deleted_partitions) it.serialize(rez);
      }
      shard_manager->broadcast_resource_update(owner_shard, rez, preconditions);
      // Now we can handle this for ourselves
      receive_replicate_resources(
          return_index, created_regs, deleted_regs, created_fids, deleted_fids,
          created_fs, latent_fs, deleted_fs, created_is, deleted_is,
          created_partitions, deleted_partitions, preconditions,
          returned_resource_ready_barrier, returned_resource_mapped_barrier,
          returned_resource_execution_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::receive_replicate_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID> >& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion> >& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
        RtBarrier& mapped_barrier, RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      bool need_deletion_dependences = true;
      std::map<Operation*, GenerationID> dependences;
      if (!created_regs.empty())
        register_region_creations(created_regs);
      if (!deleted_regs.empty())
      {
        compute_return_deletion_dependences(return_index, dependences);
        need_deletion_dependences = false;
        register_region_deletions(
            dependences, deleted_regs, preconditions, ready_barrier,
            mapped_barrier, execution_barrier);
      }
      if (!created_fids.empty())
        register_field_creations(created_fids);
      if (!deleted_fids.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_deletions(
            dependences, deleted_fids, preconditions, ready_barrier,
            mapped_barrier, execution_barrier);
      }
      if (!created_fs.empty())
        register_field_space_creations(created_fs);
      if (!latent_fs.empty())
        register_latent_field_spaces(latent_fs);
      if (!deleted_fs.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_field_space_deletions(
            dependences, deleted_fs, preconditions, ready_barrier,
            mapped_barrier, execution_barrier);
      }
      if (!created_is.empty())
        register_index_space_creations(created_is);
      if (!deleted_is.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_space_deletions(
            dependences, deleted_is, preconditions, ready_barrier,
            mapped_barrier, execution_barrier);
      }
      if (!created_partitions.empty())
        register_index_partition_creations(created_partitions);
      if (!deleted_partitions.empty())
      {
        if (need_deletion_dependences)
        {
          compute_return_deletion_dependences(return_index, dependences);
          need_deletion_dependences = false;
        }
        register_index_partition_deletions(
            dependences, deleted_partitions, preconditions, ready_barrier,
            mapped_barrier, execution_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_region_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedRegion>& regions, std::set<RtEvent>& preconditions,
        RtBarrier& ready_barrier, RtBarrier& mapped_barrier,
        RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedRegion> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedRegion& rit : regions)
        {
          std::map<LogicalRegion, unsigned>::iterator region_finder =
              created_regions.find(rit.region);
          if (region_finder == created_regions.end())
          {
            if (local_regions.find(rit.region) != local_regions.end())
            {
              Error error(LEGION_RESOURCE_EXCEPTION);
              error << "Local logical region ("
                    << rit.region.index_space.get_id() << ","
                    << rit.region.field_space.get_id() << ","
                    << rit.region.get_tree_id() << ") in task "
                    << get_task_name() << " (UID " << get_unique_id()
                    << ") was not deleted by this task. Local regions can only "
                       "be deleted "
                    << "by the task that made them.";
              error.raise();
            }
            // Deletion keeps going up
            deleted_regions.emplace_back(rit);
          }
          else
          {
            // One of ours to delete
            legion_assert(region_finder->second > 0);
            if (--region_finder->second == 0)
            {
              // Don't remove this from created regions yet,
              // That will happen when we make the deletion operation
              delete_now.emplace_back(rit);
            }
          }
        }
      }
      if (!delete_now.empty())
      {
        for (const DeletedRegion& it : delete_now)
        {
          ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
          op->initialize_logical_region_deletion(
              this, it.region, true /*unordered*/, it.provenance);
          op->initialize_replication(
              this, shard_manager->is_first_local_shard(owner_shard),
              &ready_barrier, &mapped_barrier, &execution_barrier);
          preconditions.insert(op->get_commit_event());
          op->set_deletion_preconditions(dependences);
          op->execute_dependence_analysis();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_field_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedField>& fields, std::set<RtEvent>& preconditions,
        RtBarrier& ready_barrier, RtBarrier& mapped_barrier,
        RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      std::map<std::pair<FieldSpace, Provenance*>, std::set<FieldID> >
          delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedField& fit : fields)
        {
          const std::pair<FieldSpace, FieldID> key(fit.space, fit.fid);
          std::set<std::pair<FieldSpace, FieldID> >::const_iterator
              field_finder = created_fields.find(key);
          if (field_finder == created_fields.end())
          {
            std::map<std::pair<FieldSpace, FieldID>, bool>::iterator
                local_finder = local_fields.find(key);
            if (local_finder != local_fields.end())
            {
              Error error(LEGION_RESOURCE_EXCEPTION);
              error << "Local field " << fit.fid << " in field space "
                    << fit.space.get_id() << " in task " << get_task_name()
                    << " (UID " << get_unique_id()
                    << ") was not deleted by this task. Local fields can only "
                       "be deleted "
                    << "by the task that made them.";
              error.raise();
            }
            deleted_fields.emplace_back(fit);
          }
          else
          {
            // One of ours to delete
            std::pair<FieldSpace, Provenance*> now_key(
                fit.space, fit.provenance);
            delete_now[now_key].insert(fit.fid);
            // No need to delete this now, it will be deleted
            // when the deletion op makes its region requirements
          }
        }
      }
      if (!delete_now.empty())
      {
        for (const std::pair<
                 const std::pair<FieldSpace, Provenance*>, std::set<FieldID> >&
                 it : delete_now)
        {
          ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
          FieldAllocatorImpl* allocator =
              create_field_allocator(it.first.first, true /*unordered*/);
          op->initialize_field_deletions(
              this, it.first.first, it.second, true /*unordered*/, allocator,
              it.first.second, (owner_shard->shard_id != 0));
          op->initialize_replication(
              this, shard_manager->is_first_local_shard(owner_shard),
              &ready_barrier, &mapped_barrier, &execution_barrier);
          preconditions.insert(op->get_commit_event());
          op->set_deletion_preconditions(dependences);
          op->execute_dependence_analysis();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_field_space_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedFieldSpace>& spaces,
        std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
        RtBarrier& mapped_barrier, RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedFieldSpace> delete_now;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedFieldSpace& fit : spaces)
        {
          std::map<FieldSpace, unsigned>::iterator finder =
              created_field_spaces.find(fit.space);
          if (finder != created_field_spaces.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(
                  DeletedFieldSpace(fit.space, fit.provenance));
              created_field_spaces.erase(finder);
              // Count how many regions are still using this field space
              // that still need to be deleted before we can remove the
              // list of created fields
              std::set<LogicalRegion> remaining_regions;
              for (const std::pair<const LogicalRegion, unsigned>& it :
                   created_regions)
                if (it.first.get_field_space() == fit.space)
                  remaining_regions.insert(it.first);
              for (const std::pair<const LogicalRegion, bool>& it :
                   local_regions)
                if (it.first.get_field_space() == fit.space)
                  remaining_regions.insert(it.first);
              if (remaining_regions.empty())
              {
                // No remaining regions so we can remove any created fields now
                for (std::set<std::pair<FieldSpace, FieldID> >::iterator it =
                         created_fields.begin();
                     it != created_fields.end();
                     /*nothing*/)
                {
                  if (it->first == fit.space)
                  {
                    std::set<std::pair<FieldSpace, FieldID> >::iterator
                        to_delete = it++;
                    created_fields.erase(to_delete);
                  }
                  else
                    it++;
                }
              }
              else
                latent_field_spaces[fit.space] = remaining_regions;
            }
          }
          else
            // If we didn't make this field space, record the deletion
            // and keep going. It will be handled by the context that
            // made the field space
            deleted_field_spaces.emplace_back(fit);
        }
      }
      if (!delete_now.empty())
      {
        for (const DeletedFieldSpace& it : delete_now)
        {
          ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
          op->initialize_field_space_deletion(
              this, it.space, true /*unordered*/, it.provenance);
          op->initialize_replication(
              this, shard_manager->is_first_local_shard(owner_shard),
              &ready_barrier, &mapped_barrier, &execution_barrier);
          preconditions.insert(op->get_commit_event());
          op->set_deletion_preconditions(dependences);
          op->execute_dependence_analysis();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_index_space_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedIndexSpace>& spaces,
        std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
        RtBarrier& mapped_barrier, RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedIndexSpace> delete_now;
      std::vector<std::vector<IndexPartition> > sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedIndexSpace& sit : spaces)
        {
          std::map<IndexSpace, unsigned>::iterator finder =
              created_index_spaces.find(sit.space);
          if (finder != created_index_spaces.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(sit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_spaces.erase(finder);
              if (sit.recurse)
              {
                std::vector<IndexPartition>& subs = sub_partitions.back();
                // Also remove any index partitions for this index space tree
                for (std::map<IndexPartition, unsigned>::iterator it =
                         created_index_partitions.begin();
                     it != created_index_partitions.end();
                     /*nothing*/)
                {
                  if (it->first.get_tree_id() == sit.space.get_tree_id())
                  {
                    legion_assert(it->second > 0);
                    if (--it->second == 0)
                    {
                      subs.emplace_back(it->first);
                      std::map<IndexPartition, unsigned>::iterator to_delete =
                          it++;
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
            deleted_index_spaces.emplace_back(sit);
        }
      }
      if (!delete_now.empty())
      {
        legion_assert(delete_now.size() == sub_partitions.size());
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
          op->initialize_index_space_deletion(
              this, delete_now[idx].space, sub_partitions[idx],
              true /*unordered*/, delete_now[idx].provenance);
          op->initialize_replication(
              this, shard_manager->is_first_local_shard(owner_shard),
              &ready_barrier, &mapped_barrier, &execution_barrier);
          preconditions.insert(op->get_commit_event());
          op->set_deletion_preconditions(dependences);
          op->execute_dependence_analysis();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_index_partition_deletions(
        const std::map<Operation*, GenerationID>& dependences,
        std::vector<DeletedPartition>& parts, std::set<RtEvent>& preconditions,
        RtBarrier& ready_barrier, RtBarrier& mapped_barrier,
        RtBarrier& execution_barrier)
    //--------------------------------------------------------------------------
    {
      std::vector<DeletedPartition> delete_now;
      std::vector<std::vector<IndexPartition> > sub_partitions;
      {
        AutoLock priv_lock(privilege_lock);
        for (const DeletedPartition& pit : parts)
        {
          std::map<IndexPartition, unsigned>::iterator finder =
              created_index_partitions.find(pit.partition);
          if (finder != created_index_partitions.end())
          {
            legion_assert(finder->second > 0);
            if (--finder->second == 0)
            {
              delete_now.emplace_back(pit);
              sub_partitions.resize(sub_partitions.size() + 1);
              created_index_partitions.erase(finder);
              if (pit.recurse)
              {
                std::vector<IndexPartition>& subs = sub_partitions.back();
                // Remove any other partitions that this partition dominates
                for (std::map<IndexPartition, unsigned>::iterator it =
                         created_index_partitions.begin();
                     it != created_index_partitions.end();
                     /*nothing*/)
                {
                  if ((pit.partition.get_tree_id() ==
                       it->first.get_tree_id()) &&
                      runtime->is_dominated_tree_only(it->first, pit.partition))
                  {
                    legion_assert(it->second > 0);
                    if (--it->second == 0)
                    {
                      subs.emplace_back(it->first);
                      std::map<IndexPartition, unsigned>::iterator to_delete =
                          it++;
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
            deleted_index_partitions.emplace_back(pit);
        }
      }
      if (!delete_now.empty())
      {
        legion_assert(delete_now.size() == sub_partitions.size());
        for (unsigned idx = 0; idx < delete_now.size(); idx++)
        {
          ReplDeletionOp* op = runtime->get_operation<ReplDeletionOp>();
          op->initialize_index_part_deletion(
              this, delete_now[idx].partition, sub_partitions[idx],
              true /*unordered*/, delete_now[idx].provenance);
          op->initialize_replication(
              this, shard_manager->is_first_local_shard(owner_shard),
              &ready_barrier, &mapped_barrier, &execution_barrier);
          preconditions.insert(op->get_commit_event());
          op->set_deletion_preconditions(dependences);
          op->execute_dependence_analysis();
        }
      }
    }

    //--------------------------------------------------------------------------
    CollectiveID ReplicateContext::get_next_collective_index(
        CollectiveIndexLocation loc, bool logical)
    //--------------------------------------------------------------------------
    {
      // No need for a lock, should only be coming from the creation
      // of operations directly from the application and therefore
      // should be deterministic
      // Count by 2s to avoid conflicts with the collectives from the
      // logical depedence analysis stage of the pipeline
      if (logical)
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        if (!logical_guard_reentrant)
        {
          CollectiveCheckReduction::RHS location = loc;
          // Guard against coming back in here when advancing the barrier
          logical_guard_reentrant = true;
          const RtBarrier logical_check_bar = logical_check_barrier.next(
              this, CollectiveCheckReduction::REDOP,
              &CollectiveCheckReduction::IDENTITY,
              sizeof(CollectiveCheckReduction::IDENTITY));
          logical_guard_reentrant = false;
          runtime->phase_barrier_arrive(
              logical_check_bar, 1 /*count*/, RtEvent::NO_RT_EVENT, &location,
              sizeof(location));
          logical_check_bar.wait();
          CollectiveCheckReduction::RHS actual_location;
          legion_no_skip_assert(Runtime::get_barrier_result(
              logical_check_bar, &actual_location, sizeof(actual_location)));
          legion_assert(location == actual_location);
        }
#endif
        const CollectiveID result = next_logical_collective_index;
        next_logical_collective_index += 2;
        return result;
      }
      else
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        if (!collective_guard_reentrant)
        {
          CollectiveCheckReduction::RHS location = loc;
          // Guard against coming back in here when advancing the barrier
          collective_guard_reentrant = true;
          const RtBarrier collective_check_bar = collective_check_barrier.next(
              this, CollectiveCheckReduction::REDOP,
              &CollectiveCheckReduction::IDENTITY,
              sizeof(CollectiveCheckReduction::IDENTITY));
          collective_guard_reentrant = false;
          runtime->phase_barrier_arrive(
              collective_check_bar, 1 /*count*/, RtEvent::NO_RT_EVENT,
              &location, sizeof(location));
          collective_check_bar.wait();
          CollectiveCheckReduction::RHS actual_location;
          legion_no_skip_assert(Runtime::get_barrier_result(
              collective_check_bar, &actual_location, sizeof(actual_location)));
          legion_assert(location == actual_location);
        }
#endif
        const CollectiveID result = next_available_collective_index;
        next_available_collective_index += 2;
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::register_collective(ShardCollective* collective)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<void*, size_t> > to_apply;
      {
        AutoLock repl_lock(replication_lock);
        legion_assert(
            collectives.find(collective->collective_index) ==
            collectives.end());
        legion_assert(shard_manager != nullptr);
        // If the collectives are empty then we add a reference to the
        // shard manager to prevent it being collected before we're
        // done handling all the collectives
        if (collectives.empty())
          shard_manager->add_nested_gc_ref(did);
        collectives[collective->collective_index] = collective;
        std::map<CollectiveID, std::vector<std::pair<void*, size_t> > >::
            iterator finder =
                pending_collective_updates.find(collective->collective_index);
        if (finder != pending_collective_updates.end())
        {
          to_apply.swap(finder->second);
          pending_collective_updates.erase(finder);
        }
      }
      if (!to_apply.empty())
      {
        for (const std::pair<void*, size_t>& it : to_apply)
        {
          Deserializer derez(it.first, it.second);
          collective->handle_collective_message(derez);
          free(it.first);
        }
      }
    }

    //--------------------------------------------------------------------------
    ShardCollective* ReplicateContext::find_or_buffer_collective(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      CollectiveID collective_index;
      derez.deserialize(collective_index);
      AutoLock repl_lock(replication_lock);
      // See if we already have the collective in which case we can just
      // return it, otherwise we need to buffer the deserializer
      std::map<CollectiveID, ShardCollective*>::const_iterator finder =
          collectives.find(collective_index);
      if (finder != collectives.end())
        return finder->second;
      // If we couldn't find it then we have to buffer it for the future
      const size_t remaining_bytes = derez.get_remaining_bytes();
      void* buffer = malloc(remaining_bytes);
      memcpy(buffer, derez.get_current_pointer(), remaining_bytes);
      derez.advance_pointer(remaining_bytes);
      pending_collective_updates[collective_index].emplace_back(
          std::pair<void*, size_t>(buffer, remaining_bytes));
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::unregister_collective(ShardCollective* collective)
    //--------------------------------------------------------------------------
    {
      bool remove_reference = false;
      {
        AutoLock repl_lock(replication_lock);
        std::map<CollectiveID, ShardCollective*>::iterator finder =
            collectives.find(collective->collective_index);
        // Sometimes collectives are not used
        if (finder != collectives.end())
        {
          collectives.erase(finder);
          // Once we've done all our collectives then we can remove the
          // reference that we added on the shard manager
          remove_reference = collectives.empty();
        }
      }
      if (remove_reference && shard_manager->remove_nested_gc_ref(did))
        delete shard_manager;
    }

    //--------------------------------------------------------------------------
    size_t ReplicateContext::register_trace_template(
        ShardedPhysicalTemplate* physical_template)
    //--------------------------------------------------------------------------
    {
      size_t index;
      std::vector<PendingTemplateUpdate> to_apply;
      {
        AutoLock r_lock(replication_lock);
        index = next_physical_template_index++;
        legion_assert(
            physical_templates.find(index) == physical_templates.end());
        physical_templates[index] = physical_template;
        // Check to see if we have any pending updates to perform
        std::map<size_t, std::vector<PendingTemplateUpdate> >::iterator finder =
            pending_template_updates.find(index);
        if (finder != pending_template_updates.end())
        {
          to_apply.swap(finder->second);
          pending_template_updates.erase(finder);
        }
      }
      if (!to_apply.empty())
      {
        for (const PendingTemplateUpdate& it : to_apply)
        {
          Deserializer derez(it.ptr, it.size);
          physical_template->handle_trace_update(derez, it.source);
          free(it.ptr);
        }
      }
      return index;
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate* ReplicateContext::find_or_buffer_trace_update(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t trace_index;
      derez.deserialize(trace_index);
      AutoLock r_lock(replication_lock);
      std::map<size_t, ShardedPhysicalTemplate*>::const_iterator finder =
          physical_templates.find(trace_index);
      if (finder != physical_templates.end())
        return finder->second;
      legion_assert(next_physical_template_index <= trace_index);
      // If we couldn't find it then we have to buffer it for the future
      const size_t remaining_bytes = derez.get_remaining_bytes();
      void* buffer = malloc(remaining_bytes);
      memcpy(buffer, derez.get_current_pointer(), remaining_bytes);
      derez.advance_pointer(remaining_bytes);
      pending_template_updates[trace_index].emplace_back(
          PendingTemplateUpdate(buffer, remaining_bytes, source));
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::unregister_trace_template(size_t index)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(replication_lock);
      std::map<size_t, ShardedPhysicalTemplate*>::iterator finder =
          physical_templates.find(index);
      legion_assert(finder != physical_templates.end());
      physical_templates.erase(finder);
    }

    //--------------------------------------------------------------------------
    ShardID ReplicateContext::get_next_equivalence_set_origin(void)
    //--------------------------------------------------------------------------
    {
      const ShardID result = equivalence_set_allocator_shard++;
      if (equivalence_set_allocator_shard == total_shards)
        equivalence_set_allocator_shard = 0;
      return result;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplicateContext::compute_equivalence_sets(
        unsigned req_index, const std::vector<EqSetTracker*>& targets,
        const std::vector<AddressSpaceID>& target_spaces,
        AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
        const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(targets.size() == target_spaces.size());
      legion_assert(std::is_sorted(target_spaces.begin(), target_spaces.end()));
      legion_assert(std::binary_search(
          target_spaces.begin(), target_spaces.end(), creation_target_space));
      // If this is virtual mapped, then continue up to the parent
      if ((req_index < regions.size()) && virtual_mapped[req_index])
        return find_parent_context()->compute_equivalence_sets(
            parent_req_indexes[req_index], targets, target_spaces,
            creation_target_space, expr, mask);
      // Find the equivalence set tree for this region requirement
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      // Then ask the index space expression to traverse the tree for
      // all of its rectangles and find the equivalence sets that are needed
      op::FieldMaskMap<EqKDTree> to_create;
      op::FieldMaskMap<EquivalenceSet> eq_sets;
      std::vector<RtEvent> pending_sets;
      op::FieldMaskMap<EqKDTree> new_subscriptions;
      op::map<EqKDTree*, Domain> creation_rects;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask> > creation_srcs;
      op::map<ShardID, op::map<Domain, FieldMask> > remote_shard_rects;
      std::vector<unsigned> new_target_references(targets.size(), 0);
      expr->compute_equivalence_sets(
          tree, tree_lock, mask, targets, target_spaces, new_target_references,
          eq_sets, pending_sets, new_subscriptions, to_create, creation_rects,
          creation_srcs, remote_shard_rects, owner_shard->shard_id);
      legion_assert(to_create.size() == creation_rects.size());
      // Send out messages to any shards we need to compute remotely
      for (op::map<ShardID, op::map<Domain, FieldMask> >::const_iterator sit =
               remote_shard_rects.begin();
           sit != remote_shard_rects.end(); sit++)
      {
        const RtUserEvent ready = Runtime::create_rt_user_event();
        ReplComputeEquivalenceSets rez;
        rez.serialize(shard_manager->did);
        rez.serialize(sit->first);
        rez.serialize(targets.size());
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          rez.serialize(targets[idx]);
          rez.serialize(target_spaces[idx]);
        }
        rez.serialize(creation_target_space);
        rez.serialize(req_index);
        rez.serialize(mask);
        rez.serialize<size_t>(sit->second.size());
        for (op::map<Domain, FieldMask>::const_iterator it =
                 sit->second.begin();
             it != sit->second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(remote_shard_rects.size() + 1);
        rez.serialize(ready);
        shard_manager->send_compute_equivalence_sets(sit->first, rez);
        pending_sets.emplace_back(ready);
      }
      const CollectiveMapping target_mapping(
          target_spaces, runtime->legion_collective_radix);
      return report_equivalence_sets(
          req_index, target_mapping, targets, creation_target_space, mask,
          new_target_references, eq_sets, new_subscriptions, to_create,
          creation_rects, creation_srcs, remote_shard_rects.size() + 1,
          pending_sets);
    }

    //--------------------------------------------------------------------------
    RtEvent ReplicateContext::record_output_equivalence_set(
        EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
        EquivalenceSet* set, const FieldMask& mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(regions.size() <= req_index);
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_or_create_output_set_kd_tree(req_index, tree_lock);
      local::FieldMaskMap<EqKDTree> new_subscriptions;
      op::map<ShardID, op::map<Domain, FieldMask> > remote_shard_rects;
      unsigned references = set->set_expr->record_output_equivalence_set(
          tree, tree_lock, set, mask, source, source_space, new_subscriptions,
          remote_shard_rects, owner_shard->shard_id);
      std::vector<RtEvent> recorded_events;
      // Send out messages to any shards we need to do the recording on
      for (op::map<ShardID, op::map<Domain, FieldMask> >::const_iterator sit =
               remote_shard_rects.begin();
           sit != remote_shard_rects.end(); sit++)
      {
        const RtUserEvent ready = Runtime::create_rt_user_event();
        ReplOutputEquivalenceSet rez;
        rez.serialize(shard_manager->did);
        rez.serialize(sit->first);
        rez.serialize(source);
        rez.serialize(source_space);
        rez.serialize(req_index);
        rez.serialize(set->did);
        set->pack_global_ref();
        rez.serialize<size_t>(sit->second.size());
        for (op::map<Domain, FieldMask>::const_iterator it =
                 sit->second.begin();
             it != sit->second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        rez.serialize(ready);
        shard_manager->send_output_equivalence_set(sit->first, rez);
        recorded_events.emplace_back(ready);
      }
      if (!new_subscriptions.empty())
      {
        RtEvent recorded = report_output_registrations(
            source, source_space, references, new_subscriptions);
        if (recorded.exists())
          recorded_events.emplace_back(recorded);
      }
      return Runtime::merge_events(recorded_events);
    }

    //--------------------------------------------------------------------------
    EqKDTree* ReplicateContext::create_equivalence_set_kd_tree(
        IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
      // Tell it how many shards we have so it can create an initial spatial
      // partitioning for that number of shards
      return node->create_equivalence_set_kd_tree(total_shards);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::refine_equivalence_sets(
        unsigned req_index, IndexSpaceNode* node,
        const FieldMask& refinement_mask, std::vector<RtEvent>& applied_events,
        bool sharded, bool first, const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(!sharded || first);
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        if (!first)
          find_parent_context()->refine_equivalence_sets(
              parent_req_indexes[req_index], node, refinement_mask,
              applied_events, false /*sharded*/, false /*first*/, mapping);
        else if (sharded)
          find_parent_context()->refine_equivalence_sets(
              parent_req_indexes[req_index], node, refinement_mask,
              applied_events, false /*sharded*/, false /*first*/);
        else if (shard_manager->is_first_local_shard(owner_shard))
          find_parent_context()->refine_equivalence_sets(
              parent_req_indexes[req_index], node, refinement_mask,
              applied_events, false /*sharded*/, false /*first*/,
              shard_manager->collective_mapping);
        return;
      }
      if (sharded || !first)
      {
        LocalLock* tree_lock = nullptr;
        EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
        op::map<ShardID, op::map<Domain, FieldMask> > remote_shard_rects;
        node->invalidate_shard_equivalence_set_kd_tree(
            tree, tree_lock, refinement_mask, applied_events,
            remote_shard_rects, owner_shard->shard_id);
        // If there are any remote then send them to the target shard
        for (op::map<ShardID, op::map<Domain, FieldMask> >::const_iterator sit =
                 remote_shard_rects.begin();
             sit != remote_shard_rects.end(); sit++)
        {
          // If there is a collective mapping and it already contains the
          // address space of the node where the target shard is then we
          // don't need to send that message as it will be handled by the
          // call done on that node
          if (mapping != nullptr)
          {
            AddressSpace target = shard_manager->get_shard_space(sit->first);
            if ((target != local_space) &&
                (mapping->contains(target) ||
                 (local_space != mapping->find_nearest(target))))
              continue;
          }
          const RtUserEvent refined_event = Runtime::create_rt_user_event();
          ReplRefineEquivalenceSets rez;
          rez.serialize(shard_manager->did);
          rez.serialize(sit->first);
          rez.serialize(req_index);
          rez.serialize<size_t>(sit->second.size());
          for (op::map<Domain, FieldMask>::const_iterator it =
                   sit->second.begin();
               it != sit->second.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          rez.serialize(refined_event);
          shard_manager->send_refine_equivalence_sets(sit->first, rez);
          applied_events.emplace_back(refined_event);
        }
      }
      else
        InnerContext::refine_equivalence_sets(
            req_index, node, refinement_mask, applied_events, sharded);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::find_trace_local_sets(
        unsigned req_index, const FieldMask& mask,
        std::map<EquivalenceSet*, unsigned>& current_sets, IndexSpaceNode* node,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      const bool first = (node == nullptr);
      if (first)
      {
        LogicalRegion region = find_logical_region(req_index);
        node = runtime->get_node(region.get_index_space());
      }
      if ((req_index < regions.size()) && virtual_mapped[req_index])
      {
        if (!first)
          find_parent_context()->find_trace_local_sets(
              parent_req_indexes[req_index], mask, current_sets, node, mapping);
        else if (shard_manager->is_first_local_shard(owner_shard))
          find_parent_context()->find_trace_local_sets(
              parent_req_indexes[req_index], mask, current_sets, node,
              shard_manager->collective_mapping);
        return;
      }
      if (!first)
      {
        LocalLock* tree_lock = nullptr;
        EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
        local::map<ShardID, FieldMask> remote_shards;
        node->find_shard_trace_local_sets_kd_tree(
            tree, tree_lock, mask, req_index, current_sets, remote_shards,
            owner_shard->shard_id);
        if (!remote_shards.empty())
        {
          // Need a lock for coordinating access to the target
          LocalLock current_set_lock;
          std::vector<RtEvent> ready_events;
          // If there are any remote then send them to the target shard
          for (local::map<ShardID, FieldMask>::const_iterator it =
                   remote_shards.begin();
               it != remote_shards.end(); it++)
          {
            // If there is a collective mapping and it already contains the
            // address space of the node where the target shard is then we
            // don't need to send that message as it will be handled by the
            // call done on that node
            if (mapping != nullptr)
            {
              AddressSpace target = shard_manager->get_shard_space(it->first);
              if ((target != local_space) &&
                  (mapping->contains(target) ||
                   (local_space != mapping->find_nearest(target))))
                continue;
            }
            const RtUserEvent ready_event = Runtime::create_rt_user_event();
            ReplFindTraceSets rez;
            rez.serialize(shard_manager->did);
            rez.serialize(it->first);
            rez.serialize(req_index);
            rez.serialize(it->second);
            rez.serialize(node->handle);
            rez.serialize(&current_sets);
            rez.serialize(&current_set_lock);
            rez.serialize(ready_event);
            shard_manager->send_find_trace_local_sets(it->first, rez);
            ready_events.emplace_back(ready_event);
          }
          Runtime::merge_events(ready_events).wait();
        }
      }
      else
        InnerContext::find_trace_local_sets(
            req_index, mask, current_sets, node, mapping);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_find_trace_local_sets(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      unsigned req_index;
      derez.deserialize(req_index);
      FieldMask mask;
      derez.deserialize(mask);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode* node = runtime->get_node(handle);
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      std::map<EquivalenceSet*, unsigned> local_sets;
      node->find_trace_local_sets_kd_tree(
          tree, tree_lock, mask, req_index, get_shard_id(), local_sets);
      std::map<EquivalenceSet*, unsigned>* target;
      derez.deserialize(target);
      LocalLock* target_lock;
      derez.deserialize(target_lock);
      RtUserEvent done;
      derez.deserialize(done);
      RemoteContextFindTraceLocalResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(target_lock);
        rez.serialize(req_index);
        rez.serialize<size_t>(local_sets.size());
        for (const std::pair<EquivalenceSet* const, unsigned>& it : local_sets)
        {
          legion_assert(req_index == it.second);
          rez.serialize(it.first->did);
        }
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_refine_equivalence_sets(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      unsigned req_index;
      derez.deserialize(req_index);
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      size_t num_rects;
      derez.deserialize(num_rects);
      std::vector<RtEvent> invalidated;
      // Need exclusive access for invalidations
      AutoLock t_lock(*tree_lock);
      for (unsigned idx = 0; idx < num_rects; idx++)
      {
        Domain domain;
        derez.deserialize(domain);
        FieldMask mask;
        derez.deserialize(mask);
        tree->invalidate_shard_tree(domain, mask, invalidated);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (!invalidated.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(invalidated));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_compute_equivalence_sets(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_targets;
      derez.deserialize(num_targets);
      std::vector<EqSetTracker*> targets(num_targets);
      std::vector<AddressSpaceID> target_spaces(num_targets);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        derez.deserialize(targets[idx]);
        derez.deserialize(target_spaces[idx]);
      }
      AddressSpaceID creation_target_space;
      derez.deserialize(creation_target_space);
      unsigned req_index;
      derez.deserialize(req_index);
      FieldMask mask;
      derez.deserialize(mask);
      size_t num_rects;
      derez.deserialize(num_rects);

      op::FieldMaskMap<EqKDTree> to_create;
      op::FieldMaskMap<EquivalenceSet> eq_sets;
      std::vector<RtEvent> pending_sets;
      op::FieldMaskMap<EqKDTree> new_subscriptions;
      op::map<EqKDTree*, Domain> creation_rects;
      op::map<EquivalenceSet*, op::map<Domain, FieldMask> > creation_srcs;
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_equivalence_set_kd_tree(req_index, tree_lock);
      std::vector<unsigned> new_target_references(num_targets, 0);
      {
        // Non-exclusive access to the tree for
        AutoLock t_lock(*tree_lock, false /*exclusive*/);
        for (unsigned idx = 0; idx < num_rects; idx++)
        {
          Domain rect;
          derez.deserialize(rect);
          FieldMask rect_mask;
          derez.deserialize(rect_mask);
          tree->compute_shard_equivalence_sets(
              rect, rect_mask, targets, target_spaces, new_target_references,
              eq_sets, pending_sets, new_subscriptions, to_create,
              creation_rects, creation_srcs, owner_shard->shard_id);
        }
      }
      size_t expected_responses;
      derez.deserialize(expected_responses);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      // Now we can send the responses
      const CollectiveMapping target_mapping(
          target_spaces, runtime->legion_collective_radix);
      RtEvent ready = report_equivalence_sets(
          req_index, target_mapping, targets, creation_target_space, mask,
          new_target_references, eq_sets, new_subscriptions, to_create,
          creation_rects, creation_srcs, expected_responses, pending_sets);
      Runtime::trigger_event(ready_event, ready);
    }

    //--------------------------------------------------------------------------
    void ReplicateContext::handle_output_equivalence_set(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      EqSetTracker* source;
      derez.deserialize(source);
      AddressSpaceID source_space;
      derez.deserialize(source_space);
      unsigned req_index;
      derez.deserialize(req_index);
      DistributedID did;
      derez.deserialize(did);
      RtEvent set_ready;
      EquivalenceSet* set =
          runtime->find_or_request_equivalence_set(did, set_ready);
      size_t num_rects;
      derez.deserialize(num_rects);

      local::FieldMaskMap<EqKDTree> new_subscriptions;
      LocalLock* tree_lock = nullptr;
      EqKDTree* tree = find_or_create_output_set_kd_tree(req_index, tree_lock);
      if (set_ready.exists() && !set_ready.has_triggered())
        set_ready.wait();
      unsigned references = 0;
      {
        // Non exclusive accessor for recording shard output sets
        AutoLock t_lock(*tree_lock, false /*exclusive*/);
        for (unsigned idx = 0; idx < num_rects; idx++)
        {
          Domain rect;
          derez.deserialize(rect);
          FieldMask rect_mask;
          derez.deserialize(rect_mask);
          references += tree->record_shard_output_equivalence_set(
              set, rect, rect_mask, source, source_space, new_subscriptions,
              owner_shard->shard_id);
        }
      }
      RtUserEvent recorded_event;
      derez.deserialize(recorded_event);
      Runtime::trigger_event(
          recorded_event,
          report_output_registrations(
              source, source_space, references, new_subscriptions));
      set->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::create_new_replicate_barrier(
        RtBarrier& bar,
#ifdef LEGION_DEBUG_COLLECTIVES
        ReductionOpID redop, const void* init, size_t init_size,
#endif
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(!bar.exists());
      legion_assert(next_replicate_bar_index < total_shards);
      bool created = false;
      ValueBroadcast<RtBarrier> collective(
          this, next_replicate_bar_index, COLLECTIVE_LOC_83);
      if (owner_shard->shard_id == next_replicate_bar_index++)
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        bar = RtBarrier(
            Realm::Barrier::create_barrier(arrivals, redop, init, init_size));
#else
        bar = runtime->create_rt_barrier(arrivals);
#endif
        collective.broadcast(bar);
        created = true;
      }
      else
        bar = collective.get_value();
      // Check to see if we need to reset the next_replicate_bar_index
      if (next_replicate_bar_index == total_shards)
        next_replicate_bar_index = 0;
      return created;
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::create_new_replicate_barrier(
        ApBarrier& bar,
#ifdef LEGION_DEBUG_COLLECTIVES
        ReductionOpID redop, const void* init, size_t init_size,
#endif
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(!bar.exists());
      legion_assert(next_replicate_bar_index < total_shards);
      bool created = false;
      ValueBroadcast<ApBarrier> collective(
          this, next_replicate_bar_index, COLLECTIVE_LOC_84);
      if (owner_shard->shard_id == next_replicate_bar_index++)
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        bar = ApBarrier(
            Realm::Barrier::create_barrier(arrivals, redop, init, init_size));
#else
        bar = runtime->create_ap_barrier(arrivals);
#endif
        collective.broadcast(bar);
        created = true;
      }
      else
        bar = collective.get_value();
      // Check to see if we need to reset the next_replicate_bar_index
      if (next_replicate_bar_index == total_shards)
        next_replicate_bar_index = 0;
      return created;
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::create_new_logical_barrier(
        RtBarrier& bar,
#ifdef LEGION_DEBUG_COLLECTIVES
        ReductionOpID redop, const void* init, size_t init_size,
#endif
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(!bar.exists());
      legion_assert(next_logical_bar_index < total_shards);
      bool created = false;
      const CollectiveID cid =
          get_next_collective_index(COLLECTIVE_LOC_18, true /*logical*/);
      ValueBroadcast<RtBarrier> collective(cid, this, next_logical_bar_index);
      if (owner_shard->shard_id == next_logical_bar_index++)
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        bar = RtBarrier(
            Realm::Barrier::create_barrier(arrivals, redop, init, init_size));
#else
        bar = runtime->create_rt_barrier(arrivals);
#endif
        collective.broadcast(bar);
        created = true;
      }
      else
        bar = collective.get_value();
      // Check to see if we need to reset the next_replicate_bar_index
      if (next_logical_bar_index == total_shards)
        next_logical_bar_index = 0;
      return created;
    }

    //--------------------------------------------------------------------------
    bool ReplicateContext::create_new_logical_barrier(
        ApBarrier& bar,
#ifdef LEGION_DEBUG_COLLECTIVES
        ReductionOpID redop, const void* init, size_t init_size,
#endif
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(!bar.exists());
      legion_assert(next_logical_bar_index < total_shards);
      bool created = false;
      const CollectiveID cid =
          get_next_collective_index(COLLECTIVE_LOC_24, true /*logical*/);
      ValueBroadcast<ApBarrier> collective(cid, this, next_logical_bar_index);
      if (owner_shard->shard_id == next_logical_bar_index++)
      {
#ifdef LEGION_DEBUG_COLLECTIVES
        bar = ApBarrier(
            Realm::Barrier::create_barrier(arrivals, redop, init, init_size));
#else
        bar = runtime->create_ap_barrier(arrivals);
#endif
        collective.broadcast(bar);
        created = true;
      }
      else
        bar = collective.get_value();
      // Check to see if we need to reset the next_replicate_bar_index
      if (next_logical_bar_index == total_shards)
        next_logical_bar_index = 0;
      return created;
    }

    //--------------------------------------------------------------------------
    const DomainPoint& ReplicateContext::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->shard_points[owner_shard->shard_id];
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate* ReplicateContext::find_current_shard_template(
        TraceID tid) const
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(trace_lock, false /*exclusive*/);
      std::map<TraceID, LogicalTrace*>::const_iterator finder =
          traces.find(tid);
      legion_assert(finder != traces.end());
      legion_assert(finder->second->has_physical_trace());
      PhysicalTrace* physical = finder->second->get_physical_trace();
      legion_assert(physical->is_recording());
      legion_assert(physical->has_current_template());
      PhysicalTemplate* tpl = physical->get_current_template();
      return legion_safe_cast<ShardedPhysicalTemplate*>(tpl);
    }

    //--------------------------------------------------------------------------
    ShardID ReplicateContext::AttachDetachShardingFunctor::shard(
        const DomainPoint& point, const Domain& domain,
        const size_t total_shards)
    //--------------------------------------------------------------------------
    {
      const Point<2> p = point;
      return p[0];  // First dimension is always the shard dimension
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::register_attach_detach_sharding_functor(
        void)
    //--------------------------------------------------------------------------
    {
      // See Runtime::get_current_static_sharding_id for how we get this ID
      runtime->register_sharding_functor(
          LEGION_MAX_APPLICATION_SHARDING_ID, new AttachDetachShardingFunctor(),
          false /*need check*/, true /*silence warnings*/, nullptr,
          true /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ReplicateContext::get_attach_detach_sharding_function(
        void)
    //--------------------------------------------------------------------------
    {
      // See Runtime::get_current_static_sharding_id for how we get this ID
      return shard_manager->find_sharding_function(
          LEGION_MAX_APPLICATION_SHARDING_ID);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* ReplicateContext::compute_index_attach_launch_spaces(
        std::vector<size_t>& shard_sizes, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // No need for a lock, we're in the logical dependence stage
      for (const AttachLaunchSpace* space : index_attach_launch_spaces)
      {
        legion_assert(space->shard_sizes.size() == shard_sizes.size());
        bool match = true;
        for (unsigned idx = 0; idx < shard_sizes.size(); idx++)
        {
          if (space->shard_sizes[idx] == shard_sizes[idx])
            continue;
          match = false;
          break;
        }
        if (match)
          return space->launch_space;
      }
      // Make the index space first
      // See if we can make this a rect or a ragged collection of rects
      coord_t upper_bound = 0;
      for (const size_t& it : shard_sizes)
      {
        if (&it != &shard_sizes[0])
        {
          if (coord_t(it) != upper_bound)
            upper_bound = -1;
        }
        else
          upper_bound = it;
      }
      IndexSpace handle;
      if (upper_bound > 0)
      {
        const Domain domain = Rect<2>(
            Point<2>(0, 0), Point<2>(shard_sizes.size() - 1, upper_bound - 1));
        handle = InnerContext::create_index_space(
            domain, true /*take ownership*/,
            NT_TemplateHelper::encode_tag<2, coord_t>(), provenance);
      }
      else
      {
        std::vector<Rect<2> > rects(shard_sizes.size());
        // Use prefix sum here so we know where each shard begins and ends
        // in the color space of the projection
        coord_t offset = 0;
        for (unsigned idx = 0; idx < shard_sizes.size(); idx++)
        {
          rects[idx] = Rect<2>(
              Point<2>(idx, offset),
              Point<2>(idx, offset + shard_sizes[idx] - 1));
          offset += shard_sizes[idx];
        }
        const Domain domain = Realm::IndexSpace<2, coord_t>(rects);
        handle = InnerContext::create_index_space(
            domain, true /*take ownership*/,
            NT_TemplateHelper::encode_tag<2, coord_t>(), provenance);
      }
      IndexSpaceNode* node = runtime->get_node(handle);
      AttachLaunchSpace* space = new AttachLaunchSpace(node);
      space->shard_sizes.swap(shard_sizes);
      index_attach_launch_spaces.emplace_back(space);
      return node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReplicateContext::register_universal_sharding_functor(void)
    //--------------------------------------------------------------------------
    {
      // See Runtime::get_current_static_sharding_id for how we get this ID
      runtime->register_sharding_functor(
          LEGION_MAX_APPLICATION_SHARDING_ID + 1,
          new UniversalShardingFunctor(), false /*need check*/,
          true /*silence warnings*/, nullptr, true /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ReplicateContext::get_universal_sharding_function(void)
    //--------------------------------------------------------------------------
    {
      // See Runtime::get_current_static_sharding_id for how we get this ID
      // Note the universal sharding function is special and can skip checks
      // on the output because it's not actually used for sharding
      return shard_manager->find_sharding_function(
          LEGION_MAX_APPLICATION_SHARDING_ID + 1, true /*skip checks*/);
    }

    /////////////////////////////////////////////////////////////
    // Consensus Match Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConsensusMatchExchange::ConsensusMatchExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc, FutureImpl* f,
        const void* in, void* out, size_t size, size_t count)
      : AllGatherCollective<false>(loc, ctx), to_complete(f),
        total_elements(count), input(in), output(out), element_size(size)
    //--------------------------------------------------------------------------
    {
      to_complete->add_base_gc_ref(COLLECTIVE_REF);
      uintptr_t ptr = reinterpret_cast<uintptr_t>(input);
      valid_elements.resize(count);
      for (unsigned idx = 0; idx < count; idx++)
        valid_elements[idx] = reinterpret_cast<void*>(ptr + idx * element_size);
      // Sort the valid elements so that we can efficiently query them
      std::sort(
          valid_elements.begin(), valid_elements.end(),
          ElementComparator(element_size));
    }

    //--------------------------------------------------------------------------
    ConsensusMatchExchange::~ConsensusMatchExchange(void)
    //--------------------------------------------------------------------------
    {
      if (to_complete->remove_base_gc_ref(COLLECTIVE_REF))
        delete to_complete;
    }

    //--------------------------------------------------------------------------
    void ConsensusMatchExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(valid_elements.size());
      for (unsigned idx = 0; idx < valid_elements.size(); idx++)
        rez.serialize(valid_elements[idx], element_size);
    }

    //--------------------------------------------------------------------------
    void ConsensusMatchExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_elements;
      derez.deserialize(num_elements);
      if (num_elements > 0)
      {
        uintptr_t ptr =
            reinterpret_cast<uintptr_t>(derez.get_current_pointer());
        std::vector<const void*> incoming_elements(num_elements);
        for (unsigned idx = 0; idx < num_elements; idx++)
          incoming_elements[idx] =
              reinterpret_cast<const void*>(ptr + idx * element_size);
        derez.advance_pointer(num_elements * element_size);
        // Elements are already sorted in order
        for (std::vector<const void*>::iterator it = valid_elements.begin();
             it != valid_elements.end();
             /*nothing*/)
        {
          if (std::binary_search(
                  incoming_elements.begin(), incoming_elements.end(), *it,
                  ElementComparator(element_size)))
            it++;
          else
            it = valid_elements.erase(it);
        }
      }
      else
        valid_elements.clear();
    }

    //--------------------------------------------------------------------------
    RtEvent ConsensusMatchExchange::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      const size_t valid_count = valid_elements.size();
      if (!valid_elements.empty())
      {
        // We guarantee that the elements appear in the same order across
        // all the shards so we make that the sorted order
        uintptr_t out_ptr = reinterpret_cast<uintptr_t>(output);
        for (unsigned idx = 0; idx < valid_elements.size(); idx++)
        {
          std::memcpy(
              reinterpret_cast<void*>(out_ptr), valid_elements[idx],
              element_size);
          out_ptr += element_size;
        }
      }
      to_complete->set_local(&valid_count, sizeof(valid_count), false /*own*/);
      // A little bit of fun here, we can self delete because the
      // implementation of AllGatherCollective allows that
      delete this;
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // VerifyReplicableExchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VerifyReplicableExchange::VerifyReplicableExchange(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : AllGatherCollective<false>(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    VerifyReplicableExchange::~VerifyReplicableExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void VerifyReplicableExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(unique_hashes.size());
      for (const ShardHashes::value_type& it : unique_hashes)
      {
        rez.serialize(it.first.first);
        rez.serialize(it.first.second);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void VerifyReplicableExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_hashes;
      derez.deserialize(num_hashes);
      for (unsigned idx = 0; idx < num_hashes; idx++)
      {
        std::pair<uint64_t, uint64_t> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        ShardHashes::iterator finder = unique_hashes.find(key);
        if (finder != unique_hashes.end())
        {
          ShardID sid;
          derez.deserialize(sid);
          if (sid < finder->second)
            finder->second = sid;
        }
        else
          derez.deserialize(unique_hashes[key]);
      }
    }

    //--------------------------------------------------------------------------
    const VerifyReplicableExchange::ShardHashes&
        VerifyReplicableExchange::exchange(const uint64_t hash[2])
    //--------------------------------------------------------------------------
    {
      const std::pair<uint64_t, uint64_t> key(hash[0], hash[1]);
      unique_hashes[key] = local_shard;
      perform_collective_sync();
      return unique_hashes;
    }

    /////////////////////////////////////////////////////////////
    // Cross Product Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CrossProductCollective::CrossProductCollective(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CrossProductCollective::~CrossProductCollective(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CrossProductCollective::exchange_partitions(
        std::map<IndexSpace, IndexPartition>& handles)
    //--------------------------------------------------------------------------
    {
      // Only put the non-empty partitions into our local set
      for (const std::map<IndexSpace, IndexPartition>::value_type& it : handles)
      {
        if (!it.second.exists())
          continue;
        non_empty_handles.insert(it);
      }
      // Now we do the exchange
      perform_collective_sync();
      // When we wake up we should have all the handles and no need the lock
      // to access them
      handles = non_empty_handles;
    }

    //--------------------------------------------------------------------------
    void CrossProductCollective::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(non_empty_handles.size());
      for (const std::map<IndexSpace, IndexPartition>::value_type& it :
           non_empty_handles)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void CrossProductCollective::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_handles;
      derez.deserialize(num_handles);
      for (unsigned idx = 0; idx < num_handles; idx++)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        derez.deserialize(non_empty_handles[handle]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Unordered Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UnorderedExchange::UnorderedExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UnorderedExchange::~UnorderedExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::update_future_counts(
        const int stage, std::map<int, std::map<T, unsigned> >& future_counts,
        std::map<T, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      typename std::map<int, std::map<T, unsigned> >::iterator next =
          future_counts.find(stage - 1);
      if (next != future_counts.end())
      {
        for (typename std::map<T, unsigned>::const_iterator it =
                 next->second.begin();
             it != next->second.end(); it++)
        {
          typename std::map<T, unsigned>::iterator finder =
              counts.find(it->first);
          if (finder == counts.end())
            counts.insert(*it);
          else
            finder->second += it->second;
        }
        future_counts.erase(next);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::pack_counts(
        Serializer& rez, const std::map<T, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(counts.size());
      for (const typename std::map<T, unsigned>::value_type& it : counts)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::unpack_counts(
        const int stage, Deserializer& derez, std::map<T, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      size_t num_counts;
      derez.deserialize(num_counts);
      if (num_counts == 0)
        return;
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        T key;
        derez.deserialize(key);
        typename std::map<T, unsigned>::iterator finder = counts.find(key);
        if (finder != counts.end())
        {
          unsigned count;
          derez.deserialize(count);
          finder->second += count;
        }
        else
          derez.deserialize(counts[key]);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::pack_field_counts(
        Serializer& rez,
        const std::map<std::pair<T, FieldID>, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(counts.size());
      for (const std::pair<const typename std::pair<T, FieldID>, unsigned>&
               count : counts)
      {
        rez.serialize(count.first.first);
        rez.serialize(count.first.second);
        rez.serialize(count.second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::unpack_field_counts(
        const int stage, Deserializer& derez,
        std::map<std::pair<T, FieldID>, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      size_t num_counts;
      derez.deserialize(num_counts);
      if (num_counts == 0)
        return;
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        std::pair<T, FieldID> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        typename std::map<std::pair<T, FieldID>, unsigned>::iterator finder =
            counts.find(key);
        if (finder != counts.end())
        {
          unsigned count;
          derez.deserialize(count);
          finder->second += count;
        }
        else
          derez.deserialize(counts[key]);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, typename OP>
    void UnorderedExchange::initialize_counts(
        const std::map<T, OP*>& ops, std::map<T, unsigned>& counts)
    //--------------------------------------------------------------------------
    {
      for (const typename std::map<T, OP*>::value_type& it : ops)
        counts[it.first] = 1;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename OP>
    void UnorderedExchange::find_ready_ops(
        const size_t total_shards, const std::map<T, unsigned>& final_counts,
        const std::map<T, OP*>& ops, std::vector<Operation*>& ready_ops)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<T, unsigned>::const_iterator it =
               final_counts.begin();
           it != final_counts.end(); it++)
      {
        legion_assert(it->second <= total_shards);
        if (it->second == total_shards)
        {
          typename std::map<T, OP*>::const_iterator finder =
              ops.find(it->first);
          legion_assert(finder != ops.end());
          ready_ops.emplace_back(finder->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      pack_counts(rez, index_space_counts);
      pack_counts(rez, index_partition_counts);
      pack_counts(rez, field_space_counts);
      pack_field_counts(rez, field_counts);
      pack_counts(rez, logical_region_counts);
      pack_field_counts(rez, region_detach_counts);
      pack_field_counts(rez, partition_detach_counts);
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      // If we are not a participating stage then we already contributed our
      // data into the output so we clear ourself to avoid double counting
      if (!participating)
      {
        legion_assert(stage == -1);
        index_space_counts.clear();
        index_partition_counts.clear();
        field_space_counts.clear();
        field_counts.clear();
        logical_region_counts.clear();
        region_detach_counts.clear();
        partition_detach_counts.clear();
      }
      unpack_counts(stage, derez, index_space_counts);
      unpack_counts(stage, derez, index_partition_counts);
      unpack_counts(stage, derez, field_space_counts);
      unpack_field_counts(stage, derez, field_counts);
      unpack_counts(stage, derez, logical_region_counts);
      unpack_field_counts(stage, derez, region_detach_counts);
      unpack_field_counts(stage, derez, partition_detach_counts);
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::start_unordered_exchange(
        const std::vector<Operation*>& unordered_ops)
    //--------------------------------------------------------------------------
    {
      // Sort our operations
      if (!unordered_ops.empty())
      {
        for (Operation* const & it : unordered_ops)
        {
          switch (it->get_operation_kind())
          {
            case DELETION_OP_KIND:
              {
                ReplDeletionOp* op = legion_safe_cast<ReplDeletionOp*>(it);
                op->record_unordered_kind(
                    index_space_deletions, index_partition_deletions,
                    field_space_deletions, field_deletions,
                    logical_region_deletions);
                break;
              }
            case DETACH_OP_KIND:
              {
                ReplDetachOp* op = dynamic_cast<ReplDetachOp*>(it);
                if (op == nullptr)
                {
                  ReplIndexDetachOp* index =
                      legion_safe_cast<ReplIndexDetachOp*>(it);
                  index->record_unordered_kind(
                      region_detachments, partition_detachments);
                }
                else
                  op->record_unordered_kind(region_detachments);
                break;
              }
            default:  // Unimplemented operation kind
              std::abort();
          }
        }
        // Set the initial counts to one for all our unordered ops
        initialize_counts(index_space_deletions, index_space_counts);
        initialize_counts(index_partition_deletions, index_partition_counts);
        initialize_counts(field_space_deletions, field_space_counts);
        initialize_counts(field_deletions, field_counts);
        initialize_counts(logical_region_deletions, logical_region_counts);
        initialize_counts(region_detachments, region_detach_counts);
        initialize_counts(partition_detachments, partition_detach_counts);
      }
      // Perform the exchange
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::find_ready_operations(
        std::vector<Operation*>& ready_ops)
    //--------------------------------------------------------------------------
    {
      // Now look and see which operations have keys for all shards
      // Only need to do this if we have ops, if we didn't have ops then
      // it's impossible for anyone else to have them all too
      const size_t total_shards = manager->total_shards;
      // The order in which we add these operations is actually important
      // We need to do them in the order in which they might actually depend
      // on themselves based on how they were issued
      // Do detach operations first since they should preced all deletions
      find_ready_ops(
          total_shards, region_detach_counts, region_detachments, ready_ops);
      find_ready_ops(
          total_shards, partition_detach_counts, partition_detachments,
          ready_ops);
      // Next do field deletions since they should precede deletions of
      // logical regions and field spaces
      find_ready_ops(total_shards, field_counts, field_deletions, ready_ops);
      // Then do logical region deletions which should precede field
      // space deletions
      find_ready_ops(
          total_shards, logical_region_counts, logical_region_deletions,
          ready_ops);
      find_ready_ops(
          total_shards, field_space_counts, field_space_deletions, ready_ops);
      // Do index partition deletions before index space deletions
      find_ready_ops(
          total_shards, index_partition_counts, index_partition_deletions,
          ready_ops);
      find_ready_ops(
          total_shards, index_space_counts, index_space_deletions, ready_ops);
    }

    /////////////////////////////////////////////////////////////
    // Implicit Sharding Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ImplicitShardingFunctor::ImplicitShardingFunctor(
        ReplicateContext* ctx, CollectiveIndexLocation loc,
        ReplFutureMapImpl* m)
      : AllGatherCollective<false>(loc, ctx), ShardingFunctor(), map(m)
    //--------------------------------------------------------------------------
    {
      // Add this reference here, it will be removed after the exchange is
      // complete and that will break the cycle on deleting things since
      // technically the future map will have a reference to this as well
      map->add_base_resource_ref(PENDING_UNBOUND_REF);
    }

    //--------------------------------------------------------------------------
    ImplicitShardingFunctor::~ImplicitShardingFunctor(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ImplicitShardingFunctor::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(implicit_sharding.size());
      for (const std::pair<const DomainPoint, ShardID>& it : implicit_sharding)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void ImplicitShardingFunctor::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_points;
      derez.deserialize(num_points);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(implicit_sharding[point]);
      }
    }

    //--------------------------------------------------------------------------
    ShardID ImplicitShardingFunctor::shard(
        const DomainPoint& point, const Domain& full_space,
        const size_t total_shards)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      std::map<DomainPoint, ShardID>::const_iterator finder =
          implicit_sharding.find(point);
      legion_assert(finder != implicit_sharding.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    RtEvent ImplicitShardingFunctor::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Remove our reference on the map
      if (map->remove_base_resource_ref(PENDING_UNBOUND_REF))
        delete map;
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Pointwise Allreduce
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointwiseAllreduce::PointwiseAllreduce(
        ReplicateContext* ctx, CollectiveID id, std::pair<bool, bool>& loc)
      : AllGatherCollective<false>(ctx, id), local(loc)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointwiseAllreduce::~PointwiseAllreduce(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointwiseAllreduce::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(local.first);
      rez.serialize(local.second);
    }

    //--------------------------------------------------------------------------
    void PointwiseAllreduce::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      bool next;
      derez.deserialize(next);
      if (!next)
        local.first = false;
      derez.deserialize(next);
      if (!next)
        local.second = false;
    }

    /////////////////////////////////////////////////////////////
    // Shard Sync Tree
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardSyncTree::ShardSyncTree(
        ReplicateContext* ctx, ShardID origin, CollectiveIndexLocation loc)
      : GatherCollective(loc, ctx, origin), done(get_done_event())
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardSyncTree::~ShardSyncTree(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ShardSyncTree::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(done);
    }

    //--------------------------------------------------------------------------
    void ShardSyncTree::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      RtEvent postcondition;
      derez.deserialize(postcondition);
      postconditions.emplace_back(postcondition);
    }

    //--------------------------------------------------------------------------
    RtEvent ShardSyncTree::post_gather(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::merge_events(postconditions);
    }

    /////////////////////////////////////////////////////////////
    // Sharded Rendezvous
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardRendezvous::ShardRendezvous(
        ReplicateContext* ctx, ShardID origin,
        const std::vector<ShardID>& parts)
      : context(ctx), origin_shard(origin),
        local_shard(ctx->owner_shard->shard_id), participants(parts),
        all_shards_participating(parts.size() == 1)
    //--------------------------------------------------------------------------
    {
      legion_assert(!parts.empty());
      legion_assert(
          !all_shards_participating || (parts.back() == ctx->total_shards));
    }

    //--------------------------------------------------------------------------
    void ShardRendezvous::prefix_message(Serializer& rez, ShardID target) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(context->shard_manager->did);
      rez.serialize(target);
      rez.serialize(origin_shard);
    }

    //--------------------------------------------------------------------------
    void ShardRendezvous::register_rendezvous(void)
    //--------------------------------------------------------------------------
    {
      context->register_rendezvous(this);
    }

    //--------------------------------------------------------------------------
    size_t ShardRendezvous::get_total_participants(void) const
    //--------------------------------------------------------------------------
    {
      if (all_shards_participating)
        return context->total_shards;
      else
        return participants.size();
    }

    //--------------------------------------------------------------------------
    ShardID ShardRendezvous::get_parent(void) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local_shard);
      const unsigned origin_index = find_index(origin_shard);
      legion_assert(local_index < get_total_participants());
      legion_assert(origin_index < get_total_participants());
      const unsigned offset = convert_to_offset(local_index, origin_index);
      const unsigned index = convert_to_index(
          (offset - 1) / runtime->legion_collective_radix, origin_index);
      return get_index(index);
    }

    //--------------------------------------------------------------------------
    size_t ShardRendezvous::count_children(void) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local_shard);
      const unsigned origin_index = find_index(origin_shard);
      const size_t total_participants = get_total_participants();
      legion_assert(local_index < total_participants);
      legion_assert(origin_index < total_participants);
      const unsigned radix = runtime->legion_collective_radix;
      const unsigned offset =
          radix * convert_to_offset(local_index, origin_index);
      size_t result = 0;
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_participants)
          result++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ShardRendezvous::get_children(std::vector<ShardID>& children) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local_shard);
      const unsigned origin_index = find_index(origin_shard);
      const size_t total_participants = get_total_participants();
      legion_assert(local_index < total_participants);
      legion_assert(origin_index < total_participants);
      const unsigned radix = runtime->legion_collective_radix;
      const unsigned offset =
          radix * convert_to_offset(local_index, origin_index);
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_participants)
        {
          const unsigned index = convert_to_index(child_offset, origin_index);
          children.emplace_back(get_index(index));
        }
      }
    }

    //--------------------------------------------------------------------------
    unsigned ShardRendezvous::find_index(ShardID shard) const
    //--------------------------------------------------------------------------
    {
      if (!all_shards_participating)
      {
        std::vector<ShardID>::const_iterator finder =
            std::lower_bound(participants.begin(), participants.end(), shard);
        legion_assert(finder != participants.end());
        legion_assert(*finder == shard);
        return std::distance(participants.begin(), finder);
      }
      else
        return shard;
    }

    //--------------------------------------------------------------------------
    ShardID ShardRendezvous::get_index(unsigned index) const
    //--------------------------------------------------------------------------
    {
      if (all_shards_participating)
      {
        legion_assert(index < context->total_shards);
        return index;
      }
      else
      {
        legion_assert(index < participants.size());
        return participants[index];
      }
    }

    //--------------------------------------------------------------------------
    unsigned ShardRendezvous::convert_to_offset(
        unsigned index, unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
      const size_t total_participants = get_total_participants();
      legion_assert(index < total_participants);
      legion_assert(origin_index < total_participants);
      if (index < origin_index)
      {
        // Modulus arithmetic here
        return ((index + total_participants) - origin_index);
      }
      else
        return (index - origin_index);
    }

    //--------------------------------------------------------------------------
    unsigned ShardRendezvous::convert_to_index(
        unsigned offset, unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
      const size_t total_participants = get_total_participants();
      legion_assert(offset < total_participants);
      legion_assert(origin_index < total_participants);
      unsigned result = origin_index + offset;
      if (result >= total_participants)
        result -= total_participants;
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Timeout Match Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TimeoutMatchExchange::TimeoutMatchExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(
            ctx, ctx->get_next_collective_index(loc, true /*logical*/)),
        double_latency(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TimeoutMatchExchange::~TimeoutMatchExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TimeoutMatchExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(all_timeouts.size());
      for (unsigned idx = 0; idx < all_timeouts.size(); idx++)
      {
        rez.serialize(all_timeouts[idx].first);
        rez.serialize(all_timeouts[idx].second);
      }
      rez.serialize<bool>(double_latency);
    }

    //--------------------------------------------------------------------------
    void TimeoutMatchExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_timeouts;
      derez.deserialize(num_timeouts);
      std::vector<std::pair<size_t, unsigned> > next;
      for (unsigned idx = 0; idx < num_timeouts; idx++)
      {
        std::pair<size_t, unsigned> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        if (std::binary_search(all_timeouts.begin(), all_timeouts.end(), key))
          next.emplace_back(key);
      }
      if (next.size() < all_timeouts.size())
        all_timeouts.swap(next);
      bool not_ready;
      derez.deserialize<bool>(not_ready);
      if (not_ready)
        double_latency = true;
    }

    //--------------------------------------------------------------------------
    void TimeoutMatchExchange::perform_exchange(
        std::vector<std::pair<size_t, unsigned> >& timeouts, bool ready)
    //--------------------------------------------------------------------------
    {
      if (!ready)
        double_latency = true;
      if (!timeouts.empty())
      {
        all_timeouts.swap(timeouts);
        std::sort(all_timeouts.begin(), all_timeouts.end());
        // Now uniquify in case there are duplicates since we might have
        // multiple logical users for different requirements of the same
        // operation, but if the operation is committed then we know that they
        // all will have been committed so we don't need to track them all
        std::vector<std::pair<size_t, unsigned> >::iterator end =
            std::unique(all_timeouts.begin(), all_timeouts.end());
        all_timeouts.resize(std::distance(all_timeouts.begin(), end));
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool TimeoutMatchExchange::complete_exchange(
        std::vector<LogicalUser*>& timeout_users,
        std::vector<std::pair<size_t, unsigned> >& remaining_timeouts)
    //--------------------------------------------------------------------------
    {
      // perform collective wait already called by caller
      if (!all_timeouts.empty())
      {
        for (std::vector<LogicalUser*>::iterator it = timeout_users.begin();
             it != timeout_users.end();
             /*nothing*/)
        {
          const std::pair<size_t, unsigned> key(
              (*it)->ctx_index, (*it)->internal_idx);
          if (!std::binary_search(
                  all_timeouts.begin(), all_timeouts.end(), key))
          {
            // If we didn't find it then it goes in the remaining
            remaining_timeouts.emplace_back(
                std::make_pair((*it)->ctx_index, (*it)->internal_idx));
            // We remove it since we can't prune it until everyone agrees
            it = timeout_users.erase(it);
          }
          else  // Everyone agreed to remove it so it stays
            it++;
        }
      }
      else
      {
        // Everything goes into the remaining
        remaining_timeouts.reserve(timeout_users.size());
        for (LogicalUser* user : timeout_users)
          remaining_timeouts.emplace_back(
              std::make_pair(user->ctx_index, user->internal_idx));
        // We can't prune any of these users yet without consensus
        timeout_users.clear();
      }
      return double_latency;
    }

    /////////////////////////////////////////////////////////////
    // Cross Product Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CrossProductExchange::CrossProductExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void CrossProductExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(child_ids.size());
      for (const std::pair<const LegionColor, IndexPartition>& it : child_ids)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void CrossProductExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_ids;
      derez.deserialize(num_ids);
      for (unsigned idx = 0; idx < num_ids; idx++)
      {
        LegionColor color;
        derez.deserialize(color);
        derez.deserialize(child_ids[color]);
      }
    }

    //--------------------------------------------------------------------------
    void CrossProductExchange::exchange_ids(
        LegionColor color, IndexPartition pid)
    //--------------------------------------------------------------------------
    {
      child_ids.emplace(std::make_pair(color, pid));
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void CrossProductExchange::sync_child_ids(
        LegionColor color, IndexPartition& pid)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      std::map<LegionColor, IndexPartition>::iterator finder =
          child_ids.find(color);
      legion_assert(finder != child_ids.end());
      pid = finder->second;
    }

    /////////////////////////////////////////////////////////////
    // Create Collective Fill View
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreateCollectiveFillView::CreateCollectiveFillView(
        ReplicateContext* ctx, CollectiveID id, void* alloc, UniqueID uid,
        const void* val, size_t size)
      : BroadcastCollective(ctx, id, 0 /*origin shard*/), allocation(alloc),
        value((size > 0) ? std::malloc(size) : nullptr), value_size(size),
        op_uid(uid), view_did(0)
    //--------------------------------------------------------------------------
    {
      if (size > 0)
        std::memcpy(value, val, size);
    }

    //--------------------------------------------------------------------------
    CreateCollectiveFillView::~CreateCollectiveFillView(void)
    //--------------------------------------------------------------------------
    {
      if (value != nullptr)
        std::free(value);
    }

    //--------------------------------------------------------------------------
    bool CreateCollectiveFillView::matches(const void* val, size_t size) const
    //--------------------------------------------------------------------------
    {
      legion_assert(value_size > 0);
      if (value_size != size)
        return false;
      return (std::memcmp(value, val, size) == 0);
    }

    //--------------------------------------------------------------------------
    void CreateCollectiveFillView::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(view_did);
    }

    //--------------------------------------------------------------------------
    void CreateCollectiveFillView::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(view_did);
    }

    //--------------------------------------------------------------------------
    RtEvent CreateCollectiveFillView::post_broadcast(void)
    //--------------------------------------------------------------------------
    {
      // Tell the replicate context about the new DID for the fill view
      // DO NOT DO ANYTHING AFTER THIS LINE AS THE CONTEXT MIGHT DELETE THIS
      context->finalize_collective_fill_view(
          view_did, allocation, op_uid, value, value_size, this);
      return RtEvent::NO_RT_EVENT;
    }

  }  // namespace Internal
}  // namespace Legion
