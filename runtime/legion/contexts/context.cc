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

#include <thread>  // yield

#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/memory.h"
#include "legion/nodes/region.h"
#include "legion/tasks/single.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Task Context
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskContext::TaskContext(
        SingleTask* owner, int d, const std::vector<RegionRequirement>& reqs,
        const std::vector<OutputRequirement>& out_reqs, DistributedID id,
        bool perform_registration, bool inline_t, bool implicit_t,
        CollectiveMapping* mapping)
      : DistributedCollectable(id, perform_registration, mapping),
        owner_task(owner), regions(reqs), output_reqs(out_reqs), depth(d),
        executing_processor(Processor::NO_PROC), inlined_tasks(0),
        total_tunable_count(0), overhead_profiler(nullptr),
        implicit_task_profiler(nullptr), implicit_effects(nullptr),
        safe_cast_semaphore(0), task_executed(false), mutable_priority(false),
        inline_task(inline_t), implicit_task(implicit_t)
    //--------------------------------------------------------------------------
    {
      if (implicit_task && (runtime->profiler != nullptr))
        implicit_task_profiler = new ImplicitTaskProfiler();
    }

    //--------------------------------------------------------------------------
    TaskContext::~TaskContext(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(task_local_instances.empty());
      // Clean up any local variables that we have
      if (!task_local_variables.empty())
      {
        for (const std::pair<
                 const LocalVariableID, std::pair<void*, void (*)(void*)>>& it :
             task_local_variables)
        {
          if (it.second.second != nullptr)
            (*it.second.second)(it.second.first);
        }
      }
      if (overhead_profiler != nullptr)
        delete overhead_profiler;
      if (implicit_task_profiler != nullptr)
        delete implicit_task_profiler;
      if (implicit_effects != nullptr)
        delete implicit_effects;
    }

    //--------------------------------------------------------------------------
    const Task* TaskContext::get_task(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task;
    }

    //--------------------------------------------------------------------------
    UniqueID TaskContext::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return owner_task->get_unique_id();
    }

    //--------------------------------------------------------------------------
    InnerContext* TaskContext::find_parent_context(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner_task != nullptr);
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
        Realm::DSOReferenceImplementation* dso, const void* buffer,
        size_t buffer_size, bool withargs, size_t dedup_tag, RtEvent local_done,
        RtEvent global_done, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // Send messages to all the other nodes to perform it
      for (AddressSpaceID space = 0; space < runtime->total_address_spaces;
           space++)
      {
        if (space == runtime->address_space)
          continue;
        runtime->send_registration_callback(
            space, dso, global_done, preconditions, buffer, buffer_size,
            withargs, true /*deduplicate*/, dedup_tag);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void TaskContext::print_once(FILE* f, const char* message) const
    //--------------------------------------------------------------------------
    {
      fprintf(f, "%s", message);
    }

    //--------------------------------------------------------------------------
    void TaskContext::log_once(Realm::LoggerMessage& message) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, just don't deactivate it
    }

    //--------------------------------------------------------------------------
    Future TaskContext::from_value(
        const void* value, size_t size, bool owned, Provenance* provenance,
        bool shard_local)
    //--------------------------------------------------------------------------
    {
      Future result(new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance));
      // Set the future result
      FutureInstance* instance = nullptr;
      if (size > 0)
      {
        if (owned)
        {
          const Realm::ExternalMemoryResource resource(
              reinterpret_cast<uintptr_t>(value), size, true /*read only*/);
          instance = new FutureInstance(
              value, size, true /*own allocation*/, resource.clone(),
              FutureInstance::free_host_memory, executing_processor);
        }
        else
          instance = copy_to_future_inst(value, size);
      }
      result.impl->set_result(ApEvent::NO_AP_EVENT, instance);
      return result;
    }

    //--------------------------------------------------------------------------
    Future TaskContext::from_value(
        const void* buffer, size_t size, bool owned,
        const Realm::ExternalInstanceResource& resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        Provenance* provenance, bool shard_local)
    //--------------------------------------------------------------------------
    {
      Future result(new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance));
      FutureInstance* instance = new FutureInstance(
          buffer, size, owned, resource.clone(), freefunc,
          (freefunc == nullptr) ? Processor::NO_PROC : executing_processor);
      result.impl->set_result(ApEvent::NO_AP_EVENT, instance);
      return result;
    }

    //--------------------------------------------------------------------------
    Future TaskContext::consensus_match(
        const void* input, void* output, size_t num_elements,
        size_t element_size, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // No need to do a match here, there is just one shard
      if (num_elements > 0)
        std::memcpy(output, input, num_elements * element_size);
      Future result(new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance));
      result.impl->set_local(&num_elements, sizeof(num_elements));
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID TaskContext::register_variant(
        const TaskVariantRegistrar& registrar, const void* user_data,
        size_t user_data_size, const CodeDescriptor& desc, size_t ret_size,
        bool has_ret_size, VariantID vid, bool check_task_id)
    //--------------------------------------------------------------------------
    {
      return runtime->register_variant(
          registrar, user_data, user_data_size, desc, ret_size, has_ret_size,
          vid, check_task_id, false /*check context*/);
    }

    //--------------------------------------------------------------------------
    TraceID TaskContext::generate_dynamic_trace_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_trace_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    MapperID TaskContext::generate_dynamic_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_mapper_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    ProjectionID TaskContext::generate_dynamic_projection_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_projection_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    ShardingID TaskContext::generate_dynamic_sharding_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_sharding_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    ConcurrentID TaskContext::generate_dynamic_concurrent_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_concurrent_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID TaskContext::generate_dynamic_exception_handler_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_exception_handler_id(
          false /*check context*/);
    }

    //--------------------------------------------------------------------------
    TaskID TaskContext::generate_dynamic_task_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_task_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    ReductionOpID TaskContext::generate_dynamic_reduction_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_reduction_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID TaskContext::generate_dynamic_serdez_id(void)
    //--------------------------------------------------------------------------
    {
      return runtime->generate_dynamic_serdez_id(false /*check context*/);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::perform_semantic_attach(
        const char* func, unsigned kind, const void* arg, size_t arglen,
        SemanticTag tag, const void* buffer, size_t size, bool is_mutable,
        bool& global, const void* arg2, size_t arg2len)
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
    void TaskContext::push_exception_handler(ExceptionHandlerID hid)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_context == this);
      // Check to make sure we can find it
      if (runtime->find_exception_handler(hid, true /*can fail*/) == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "No exception handler registered with ID " << hid
              << " for exception handler push in task " << *this << ".";
        error.raise();
      }
      else
        exception_handler_stack.push_back(hid);
    }

    //--------------------------------------------------------------------------
    Future TaskContext::pop_exception_handler(Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_context == this);
      if (exception_handler_stack.empty())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "No exception handlers found to pop in " << *this << ".";
        error.raise();
      }
      exception_handler_stack.pop_back();
      Future result(new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance));
      result.impl->set_local(nullptr, 0 /*size*/);
      return result;
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID TaskContext::get_current_exception_handler(void) const
    //--------------------------------------------------------------------------
    {
      if (exception_handler_stack.empty())
        return 0;  // null exception handler
      else
        return exception_handler_stack.back();
    }

    //--------------------------------------------------------------------------
    void TaskContext::record_task_tree_trace(
        Exception& exception, Operation* op) const
    //--------------------------------------------------------------------------
    {
      std::ostream stream(&exception);
      stream << "\n-----------------------------------\n";
      stream << "Task Tree Trace:\n";
      stream << "[depth]<op#,(point)>\n";
      std::vector<const TaskContext*> contexts;
      const TaskContext* context = this;
      while (context->owner_task != nullptr)
      {
        contexts.push_back(context);
        context = context->owner_task->get_context();
      }
      unsigned depth = 0;
      while (!contexts.empty())
      {
        context = contexts.back();
        stream << "[" << depth++ << "]"
               << context->owner_task->get_task_tree_coordinate() << ": "
               << *context << "\n";
        contexts.pop_back();
      }
      if (op != nullptr)
        stream << "[" << depth << "]" << op->get_task_tree_coordinate() << ": "
               << *op << "\n";
    }

    //--------------------------------------------------------------------------
    void TaskContext::add_output_region(
        const OutputRequirement& req, const InstanceSet& instances, bool global,
        bool bounded, bool grouped)
    //--------------------------------------------------------------------------
    {
      size_t index = output_regions.size();
      OutputRegionImpl* impl = new OutputRegionImpl(
          index, req, instances, this, global, bounded, grouped);
      output_regions.emplace_back(OutputRegion(impl));
    }

    //--------------------------------------------------------------------------
    PhysicalRegion TaskContext::get_physical_region(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          idx < regions.size());  // should be one of our original regions
      return physical_regions[idx];
    }

    //--------------------------------------------------------------------------
    OutputRegion TaskContext::get_output_region(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(
          idx < output_regions.size());  // should be one of our output regions
      return output_regions[idx];
    }

    //--------------------------------------------------------------------------
    void TaskContext::finalize_output_regions(
        RtEvent safe_effects, std::vector<RtEvent>& output_regions_finalized)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
      {
        // Check if the task returned data for all the output fields
        // before we fianlize this output region.
        OutputRegion& output_region = output_regions[idx];
        FieldID unbound_field = 0;
        if (!output_region.impl->is_complete(unbound_field))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Task " << *owner_task << " did not return any instance "
                << "for field " << unbound_field << "of output requirement "
                << idx << ".";
          error.raise();
        }
        const RtEvent finalized = output_region.impl->finalize(safe_effects);
        if (finalized.exists())
          output_regions_finalized.push_back(finalized);
      }
      // Clear this to remove references in output region data structures
      output_regions.clear();
    }

    //--------------------------------------------------------------------------
    void TaskContext::raise_poison_exception(void)
    //--------------------------------------------------------------------------
    {
      // TODO: handle poisoned task
      std::abort();
    }

    //--------------------------------------------------------------------------
    void TaskContext::raise_region_exception(
        PhysicalRegion region, bool nuclear)
    //--------------------------------------------------------------------------
    {
      // TODO: handle region exception
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool TaskContext::safe_cast(
        IndexSpace handle, const void* realm_point, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      // We allow multiple threads to call in to safe_cast at the same
      // time to handle when a processor has multiple threads such as
      // with OpenMP processors. We don't even bother having a lock to
      // look-up the node
      IndexSpaceNode* node = nullptr;
      // Try to take the semaphore in read-only mode to do the look-up
      int current = safe_cast_semaphore.load();
      while (true)
      {
        if (current < 0)
          current = safe_cast_semaphore.load();
        else if (safe_cast_semaphore.compare_exchange_weak(
                     current, current + 1))
          break;
      }
      // Read-only look-up
      std::map<IndexSpace, IndexSpaceNode*>::const_iterator finder =
          safe_cast_spaces.find(handle);
      if (finder != safe_cast_spaces.end())
        node = finder->second;
      // Decrement the semaphore counter
      safe_cast_semaphore.fetch_sub(1);
      if (node == nullptr)
      {
        node = runtime->get_node(handle);
        // Take the semaphore in exclusive mode to update the data structure
        current = 0;
        while (!safe_cast_semaphore.compare_exchange_weak(current, -1))
          current = 0;
        safe_cast_spaces[handle] = node;
        safe_cast_semaphore.store(0);
      }
      return node->contains_point(realm_point, type_tag);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::is_region_mapped(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < physical_regions.size());
      return physical_regions[idx].is_mapped();
    }

    //--------------------------------------------------------------------------
    void TaskContext::record_padded_fields(VariantImpl* variant)
    //--------------------------------------------------------------------------
    {
      variant->record_padded_fields(regions, physical_regions);
    }

    //--------------------------------------------------------------------------
    bool TaskContext::check_region_dependence(
        RegionTreeID our_tid, IndexSpace our_space,
        const RegionRequirement& our_req, const RegionUsage& our_usage,
        const RegionRequirement& req, bool check_privileges) const
    //--------------------------------------------------------------------------
    {
      if ((req.handle_type == LEGION_SINGULAR_PROJECTION) ||
          (req.handle_type == LEGION_REGION_PROJECTION))
      {
        // If the trees are different we're done
        if (our_tid != req.region.get_tree_id())
          return false;
        // Check to see if there is a path between
        // the index spaces
        if (!runtime->has_index_path(our_space, req.region.get_index_space()))
          return false;
      }
      else
      {
        // Check if the trees are different
        if (our_tid != req.partition.get_tree_id())
          return false;
        if (!runtime->has_partition_path(
                our_space, req.partition.get_index_partition()))
          return false;
      }
      // Check to see if any privilege fields overlap
      std::vector<FieldID> intersection(our_req.privilege_fields.size());
      std::vector<FieldID>::iterator intersect_it = std::set_intersection(
          our_req.privilege_fields.begin(), our_req.privilege_fields.end(),
          req.privilege_fields.begin(), req.privilege_fields.end(),
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
      switch (check_dependence_type<false, true>(our_usage, usage))
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
    const std::vector<PhysicalRegion>& TaskContext::begin_task(Processor proc)
    //--------------------------------------------------------------------------
    {
      implicit_context = this;
      implicit_enclosing_context = did;
      implicit_provenance = 0;
      implicit_unique_op_id = owner_task->get_unique_op_id();
      if (overhead_profiler != nullptr)
        overhead_profiler->previous_profiling_time =
            Realm::Clock::current_time_in_nanoseconds();
      if (implicit_task_profiler != nullptr)
        implicit_task_profiler->start_time =
            Realm::Clock::current_time_in_nanoseconds();
      if (Processor::get_executing_processor().exists())
      {
        realm_done_event = ApEvent(Processor::get_current_finish_event());
        implicit_fevent = realm_done_event;
      }
      else if (runtime->profiler != nullptr)
        implicit_fevent = owner_task->get_completion_event();
      if ((runtime->profiler != nullptr) && (implicit_profiler == nullptr))
        runtime->profiler->instantiate_profiling_instance();
      // Switch over the executing processor to the one
      // that has actually been assigned to run this task.
      executing_processor = proc;
      owner_task->current_proc = executing_processor;
      LegionSpy::log_task_processor(get_unique_id(), executing_processor.id);
      legion_assert(regions.size() == physical_regions.size());
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      if (!inline_task)
        find_parent_context()->decrement_pending(owner_task);
      return physical_regions;
    }

    //--------------------------------------------------------------------------
    void TaskContext::end_task(
        const void* res, size_t res_size, bool owned,
        PhysicalInstance deferred_result_instance,
        FutureFunctor* callback_functor,
        const Realm::ExternalInstanceResource* resource,
        void (*freefunc)(const Realm::ExternalInstanceResource&),
        const void* metadataptr, size_t metadatasize, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      if (implicit_effects != nullptr)
      {
        // Fold in any asynchronous implicit effects here
        legion_assert(implicit_task);
        if (effects.exists())
          implicit_effects->push_back(effects);
        effects = Runtime::merge_events(nullptr, *implicit_effects);
      }
      // Finalize output regions by setting realm instances created during
      // task execution to the output regions' physical managers
      RtEvent safe_effects;
      std::vector<RtEvent> output_regions_finalized;
      if (!output_regions.empty())
      {
        if (effects.exists())
          safe_effects = Runtime::protect_event(effects);
        finalize_output_regions(safe_effects, output_regions_finalized);
      }
      if (!user_profiling_ranges.empty())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Detected mismatched profiling range calls, missing "
              << user_profiling_ranges.size() << " stop calls at the end of "
              << *owner_task << ".";
        error.raise();
      }
      // See if we need to pull the data in from a callback in the case
      // where we are going to be doing a reduction immediately, if we
      // are then we're going to overwrite 'owned' so save it to callback_owned
      bool callback_owned = false;
      bool eager_callback = false;
      if (callback_functor != nullptr)
      {
        legion_assert(res == nullptr);
        legion_assert(metadataptr == nullptr);
        legion_assert(metadatasize == 0);
        if (owner_task->is_reducing_future())
        {
          eager_callback = true;
          callback_owned = owned;
          res = callback_functor->callback_get_future(
              res_size, owned, resource, freefunc, metadataptr, metadatasize);
        }
      }
      // If we have a deferred result instance we need to escape that too
      FutureInstance* instance = nullptr;
      if (deferred_result_instance.exists())
      {
        legion_assert(res != nullptr);
        legion_assert(freefunc == nullptr);
        // Find the unique event for this instance if there is one
        LgEvent unique_event;
        if (effects.exists() && !safe_effects.exists())
          safe_effects = Runtime::protect_event(effects);
        // escape this task local instance
        const RtEvent ready = escape_task_local_instance(
            deferred_result_instance, safe_effects, 1 /*size*/,
            &deferred_result_instance, &unique_event);
        instance = new FutureInstance(
            res, res_size, false /*external*/, true /*own alloc*/, unique_event,
            deferred_result_instance, executing_processor, ready);
      }
      else if (resource != nullptr)
      {
        if (!owned)
        {
          void* buffer = malloc(res_size);
          instance = new FutureInstance(
              buffer, res_size, true /*external*/, true /*own allocation*/);
          if (!FutureInstance::check_meta_visible(resource->suggested_memory()))
          {
            FutureInstance source(
                res, res_size, false /*own allocation*/, resource->clone(),
                freefunc, executing_processor);
            effects = instance->copy_from(&source, owner_task, effects);
            // Need to wait for the copy to be done before returning
            if (effects.exists())
              effects.wait_faultignorant();
          }
          else
          {
            // We can do a simple memory copy here but we need to wait
            // for the results to be ready first before returning
            if (effects.exists())
              effects.wait_faultignorant();
            memcpy(buffer, res, res_size);
          }
        }
        else
          instance = new FutureInstance(
              res, res_size, true /*own allocation*/, resource->clone(),
              freefunc, executing_processor);
      }
      else if (res_size > 0)
      {
        legion_assert(res != nullptr);
        if (owned)
        {
          const Realm::ExternalMemoryResource resource(
              reinterpret_cast<uintptr_t>(res), res_size, true /*read only*/);
          instance = new FutureInstance(
              res, res_size, true /*own allocation*/, resource.clone(),
              FutureInstance::free_host_memory, executing_processor);
        }
        else
        {
          // Wait for any effects for immediate values this is
          // not a Realm task (e.g. an implicit task) because
          // we can't track sub-effects on them so we need to
          // trust that the effects the user is giving us are correct
          if (effects.exists() &&
              !Processor::get_executing_processor().exists())
            effects.wait_faultignorant();
          instance = copy_to_future_inst(res, res_size);
        }
      }
      // If we did an eager callback, restore whether we own it now
      bool release_callback = false;
      if (eager_callback)
      {
        // Release the callback here if we do not own the output and
        // therefore going to make a copy of it
        release_callback = !owned;
        owned = callback_owned;
      }
      // Once there are no more escaping instances we can release the rest
      release_task_local_instances(effects, safe_effects);
      // Grab some information before doing the next step in case it
      // results in the deletion of 'this'
      legion_assert(owner_task != nullptr);
      const TaskID owner_task_id = owner_task->task_id;
      // Tell the parent context that we are ready for post-end
      InnerContext* parent_ctx = owner_task->get_context();
      if (inline_task)
        parent_ctx->decrement_inlined();
      owner_task->handle_future(
          realm_done_event, instance, metadataptr, metadatasize,
          release_callback ? nullptr : callback_functor, executing_processor,
          owned);
      if (output_regions_finalized.empty())
        owner_task->complete_execution();
      else
        owner_task->complete_execution(
            Runtime::merge_events(output_regions_finalized));
      // Clear the thread local task context to prevent users from
      // calling back into this context now that the task has finished
      // Do this before calling post-end task to make sure no references
      // are recorded to this context when we wait on events
      implicit_context = nullptr;
      implicit_enclosing_context = 0;
      // If this is an implicit top-level task then we need to finish
      // the implicit profiling of the execution of that top-level task
      // now that everything else is done running
      if (implicit_task_profiler != nullptr)
      {
        // If we're an implicit top-level task then pull a bunch of
        // data onto the stack before we do any of the cleanup because
        // we might end up deleting this
        const UniqueID local_uid = get_unique_id();
        const ApEvent local_completion = owner_task->get_completion_event();
        ImplicitTaskProfiler* local_task_profiler = implicit_task_profiler;
        implicit_task_profiler = nullptr;  // We take ownership
        // Cannot invoke any local methods after this call
        post_end_task();
        const long long stop = Realm::Clock::current_time_in_nanoseconds();
        // log this with the profiler
        implicit_profiler->process_implicit(
            local_uid, owner_task_id, local_task_profiler->start_time, stop,
            local_task_profiler->waits, local_completion);
        runtime->decrement_total_outstanding_tasks(
            owner_task_id, false /*meta*/);
        delete local_task_profiler;
      }
      else
      {
        post_end_task();
        runtime->decrement_total_outstanding_tasks(
            owner_task_id, false /*meta*/);
      }
      // See if we can release our callback down
      if (release_callback)
      {
        callback_functor->callback_release_future();
        if (callback_owned)
          delete callback_functor;
      }
    }

    //--------------------------------------------------------------------------
    FutureInstance* TaskContext::copy_to_future_inst(
        const void* value, size_t size)
    //--------------------------------------------------------------------------
    {
      // Make a simple memory copy here now
      if (size > LEGION_MAX_RETURN_SIZE)
      {
        FutureInstance* instance =
            create_task_local_future(runtime->runtime_system_memory, size);
        memcpy(const_cast<void*>(instance->get_data()), value, size);
        return instance;
      }
      else
      {
        void* buffer = malloc(size);
        memcpy(buffer, value, size);
        return new FutureInstance(
            buffer, size, true /*external*/, true /*own allocation*/);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent TaskContext::escape_task_local_instance(
        PhysicalInstance instance, RtEvent safe_effects, size_t num_results,
        PhysicalInstance* results, LgEvent* unique_events,
        const Realm::InstanceLayoutGeneric** layouts, bool redistrict_only)
    //--------------------------------------------------------------------------
    {
      legion_assert(num_results > 0);
      legion_assert((layouts != nullptr) || (num_results == 1));
      std::map<PhysicalInstance, std::pair<LgEvent, bool>>::iterator finder =
          task_local_instances.find(instance);
      LgEvent old_unique_event;
      if ((finder != task_local_instances.end()) && !redistrict_only)
      {
        if (!finder->second.second)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Task " << *owner_task << " attempted to escape "
              << "an unescapable instance in memory " << instance.get_location()
              << ". Escaping instances must be allocated from escaping pools.";
          error.raise();
        }
        // Special case where we can reuse the existing instance because
        // we're escaping this into exactly one other instance with the
        // same unique event result
        if ((layouts == nullptr) && (num_results == 1) &&
            (!unique_events[0].exists() ||
             (unique_events[0] == finder->second.first)))
        {
          unique_events[0] = finder->second.first;
          task_local_instances.erase(finder);
          return RtEvent::NO_RT_EVENT;
        }
        // Everything else falls through and we redistrict instance
        old_unique_event = finder->second.first;
        task_local_instances.erase(finder);
      }
      std::vector<Realm::ProfilingRequestSet> requests(num_results);
#ifdef LEGION_DEBUG
      std::vector<MemoryManager::TaskLocalInstanceAllocator> allocators;
      allocators.reserve(num_results);
      std::vector<ProfilingResponseBase> bases;
      bases.reserve(num_results);
#endif
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        if (runtime->profiler != nullptr)
        {
          if (!unique_events[idx].exists())
            Runtime::rename_event(unique_events[idx]);
          runtime->profiler->add_inst_request(
              requests[idx], get_unique_id(), unique_events[idx]);
        }
#ifdef LEGION_DEBUG
        allocators.emplace_back(
            MemoryManager::TaskLocalInstanceAllocator(unique_events[idx]));
        bases.emplace_back(
            ProfilingResponseBase(&allocators[idx], get_unique_id(), false));
        Realm::ProfilingRequest& req = requests[idx].add_request(
            runtime->find_local_group(), LG_LEGION_PROFILING_ID, &bases[idx],
            sizeof(bases[idx]), LG_RESOURCE_PRIORITY);
        req.add_measurement<
            Realm::ProfilingMeasurements::InstanceAllocResult>();
#endif
      }
      RtEvent ready;
      const Realm::InstanceLayoutGeneric* layout = instance.get_layout();
      if (layouts == nullptr)
      {
        legion_assert(num_results == 1);
        ready = RtEvent(instance.redistrict(
            results, &layout, num_results, &requests.front()));
      }
      else
      {
        // Compute the difference in sizes so we can update the memory
        // manager with any space that has been freed up
        size_t remainder = layout->bytes_used;
        for (unsigned idx = 0; idx < num_results; idx++)
        {
          legion_assert(layouts[idx]->bytes_used <= remainder);
          remainder -= layouts[idx]->bytes_used;
        }
        if (remainder > 0)
        {
          MemoryManager* manager =
              runtime->find_memory_manager(instance.get_location());
          manager->update_remaining_capacity(remainder);
        }
        ready = RtEvent(instance.redistrict(
            results, layouts, num_results, &requests.front()));
      }
      if (ready.exists() && (runtime->profiler != NULL))
      {
        if (old_unique_event.exists())
        {
          // This happens when the instance being escaped was
          // one we made and should be the common case
          for (unsigned idx = 0; idx < num_results; idx++)
            implicit_profiler->record_instance_redistrict(
                ready, old_unique_event, unique_events[idx]);
        }
        else
        {
          // This happens when the instance being escaped wasn't
          // actually made by Legion, e.g. when the user passes
          // in an external instance to an output region
          for (unsigned idx = 0; idx < num_results; idx++)
            implicit_profiler->record_instance_ready(ready, unique_events[idx]);
        }
      }
#ifdef LEGION_DEBUG
      for (unsigned idx = 0; idx < allocators.size(); idx++)
        legion_no_skip_assert(allocators[idx].succeeded());
#endif
      return ready;
    }

    //--------------------------------------------------------------------------
    void TaskContext::release_task_local_instances(
        ApEvent effects, RtEvent safe_effects)
    //--------------------------------------------------------------------------
    {
      if (task_local_instances.empty())
        return;
      if (effects.exists() && !safe_effects.exists())
        safe_effects = Runtime::protect_event(effects);
      for (const std::pair<const PhysicalInstance, std::pair<LgEvent, bool>>&
               it : task_local_instances)
      {
        MemoryManager* manager =
            runtime->find_memory_manager(it.first.get_location());
#ifdef LEGION_MALLOC_INSTANCES
        manager->free_legion_instance(safe_effects, it.first);
#else
        manager->free_task_local_instance(it.first, safe_effects);
#endif
      }
      task_local_instances.clear();
    }

    //--------------------------------------------------------------------------
    void TaskContext::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // Issue a utility task to decrement the number of outstanding
      // tasks now that this task has started running
      owner_task->get_context()->decrement_pending(owner_task);
      legion_assert(owner_task != nullptr);
      runtime->decrement_total_outstanding_tasks(
          owner_task->task_id, false /*meta*/);
      owner_task->complete_execution();
      owner_task->trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    Lock TaskContext::create_lock(void)
    //--------------------------------------------------------------------------
    {
      return Lock(Reservation::create_reservation());
    }

    //--------------------------------------------------------------------------
    PhaseBarrier TaskContext::create_phase_barrier(unsigned arrivals)
    //--------------------------------------------------------------------------
    {
      return PhaseBarrier(runtime->create_ap_barrier(arrivals));
    }

    //--------------------------------------------------------------------------
    PhaseBarrier TaskContext::advance_phase_barrier(PhaseBarrier pb)
    //--------------------------------------------------------------------------
    {
      PhaseBarrier result = pb;
      Runtime::advance_barrier(result);
      LegionSpy::log_event_dependence(pb.phase_barrier, result.phase_barrier);
      return result;
    }

    //--------------------------------------------------------------------------
    void TaskContext::initialize_overhead_profiler(void)
    //--------------------------------------------------------------------------
    {
      // Make an overhead tracker
      legion_assert(overhead_profiler == nullptr);
      overhead_profiler = new OverheadProfiler();
    }

    //--------------------------------------------------------------------------
    void TaskContext::start_profiling_range(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->profiler != nullptr)
      {
        const long long start = Realm::Clock::current_time_in_nanoseconds();
        user_profiling_ranges.emplace_back(start);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::stop_profiling_range(const char* prov)
    //--------------------------------------------------------------------------
    {
      if (prov == nullptr)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Missing provenance string for application profiling range in "
              << *owner_task << ".";
        error.raise();
      }
      if (implicit_profiler != nullptr)
      {
        Provenance* provenance =
            runtime->find_or_create_provenance(prov, strlen(prov));
        if (user_profiling_ranges.empty())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Detected mismatched profiling range calls, received a stop "
                << "call without a corresponding start call in task "
                << *owner_task << ".";
          error.raise();
        }
        const long long stop = Realm::Clock::current_time_in_nanoseconds();
        implicit_profiler->record_application_range(
            provenance->pid, user_profiling_ranges.back(), stop);
        user_profiling_ranges.pop_back();
        if (provenance->remove_reference())
          delete provenance;
      }
    }

    //--------------------------------------------------------------------------
    void* TaskContext::get_local_task_variable(LocalVariableID id)
    //--------------------------------------------------------------------------
    {
      std::map<LocalVariableID, std::pair<void*, void (*)(void*)>>::
          const_iterator finder = task_local_variables.find(id);
      if (finder == task_local_variables.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find task local variable " << id << " in "
              << *owner_task << ".";
        error.raise();
      }
      return finder->second.first;
    }

    //--------------------------------------------------------------------------
    void TaskContext::set_local_task_variable(
        LocalVariableID id, const void* value, void (*destructor)(void*))
    //--------------------------------------------------------------------------
    {
      std::map<LocalVariableID, std::pair<void*, void (*)(void*)>>::iterator
          finder = task_local_variables.find(id);
      if (finder != task_local_variables.end())
      {
        // See if we need to clean things up first
        if (finder->second.second != nullptr)
          (*finder->second.second)(finder->second.first);
        finder->second = std::pair<void*, void (*)(void*)>(
            const_cast<void*>(value), destructor);
      }
      else
        task_local_variables[id] = std::pair<void*, void (*)(void*)>(
            const_cast<void*>(value), destructor);
    }

    //--------------------------------------------------------------------------
    void TaskContext::yield(void)
    //--------------------------------------------------------------------------
    {
      const Processor proc = Processor::get_executing_processor();
      if (proc.exists())
      {
        // Normal realm task
        YieldArgs args;
        // Run this as the lowest possible priority task on the same processor
        // which will give all other ready tasks an opportunity to run. Once
        // the meta-task does run though then we know we can wake-up.
        const RtEvent wait_for = runtime->issue_application_processor_task(
            args, LG_MIN_PRIORITY, proc);
        wait_for.wait();
      }
      else  // external implicit top-level task
        std::this_thread::yield();
    }

    //--------------------------------------------------------------------------
    void TaskContext::record_asynchronous_effect(
        ApEvent effect, const char* provenance)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      if (implicit_task)
      {
        if (implicit_effects == nullptr)
          implicit_effects = new std::vector<ApEvent>();
        implicit_effects->push_back(effect);
        // No need to bother with profiling here since nothing is going
        // to depend on our implicit top-level task anyway
      }
      else
      {
        // Ask Realm to add the event to the finish precondition
        legion_no_skip_assert(
            Processor::add_finish_event_precondition(effect) == REALM_SUCCESS);
        // If we have a profiler need to tell it about the async effect
        if (implicit_profiler != nullptr)
          implicit_profiler->record_async_effect(effect, provenance);
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      owner_task->perform_concurrent_task_barrier();
    }

    //--------------------------------------------------------------------------
    void TaskContext::increment_inlined(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inline_lock);
      inlined_tasks++;
    }

    //--------------------------------------------------------------------------
    void TaskContext::decrement_inlined(void)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inline_lock);
      legion_assert(inlined_tasks > 0);
      if ((--inlined_tasks == 0) && inlining_done.exists())
      {
        Runtime::trigger_event(inlining_done);
        inlining_done = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void TaskContext::wait_for_inlined(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      {
        AutoLock i_lock(inline_lock);
        if (inlined_tasks > 0)
        {
          legion_assert(!inlining_done.exists());
          inlining_done = Runtime::create_rt_user_event();
          wait_on = inlining_done;
        }
      }
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    Future TaskContext::predicate_task_false(
        const TaskLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.elide_future_return)
        return Future();
      if (launcher.predicate_false_future.impl != nullptr)
        return launcher.predicate_false_future;
      // Otherwise check to see if we have a value
      FutureImpl* result = new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance);
      const size_t future_size = launcher.predicate_false_result.get_size();
      if (future_size > 0)
        result->set_local(
            launcher.predicate_false_result.get_ptr(), future_size,
            false /*own*/);
      else
        result->set_result(ApEvent::NO_AP_EVENT, nullptr);
      return Future(result);
    }

    //--------------------------------------------------------------------------
    FutureMap TaskContext::predicate_index_task_false(
        IndexSpace launch_space, const IndexTaskLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.elide_future_return)
        return FutureMap();
      Domain launch_domain = launcher.launch_domain;
      if (!launch_domain.exists())
        runtime->find_domain(launch_space, launch_domain);
      IndexSpaceNode* launch_node = runtime->get_node(launch_space);
      FutureMapImpl* result = new FutureMapImpl(
          this, launch_node, runtime->get_available_distributed_id(),
          InnerContext::NO_BLOCKING_INDEX, std::optional<uint64_t>(),
          provenance);
      if (launcher.predicate_false_future.impl != nullptr)
      {
        for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
        {
          Future f = result->get_future(itr.p, true /*internal*/);
          f.impl->set_result(this, launcher.predicate_false_future.impl);
        }
      }
      else if (launcher.predicate_false_result.get_size() == 0)
      {
        // Just initialize all the futures
        for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
        {
          Future f = result->get_future(itr.p, true /*internal*/);
          f.impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
        }
      }
      else
      {
        const void* ptr = launcher.predicate_false_result.get_ptr();
        size_t ptr_size = launcher.predicate_false_result.get_size();
        for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
        {
          Future f = result->get_future(itr.p, true /*internal*/);
          f.impl->set_local(ptr, ptr_size, false /*own*/);
        }
      }
      return FutureMap(result);
    }

    //--------------------------------------------------------------------------
    Future TaskContext::predicate_index_task_reduce_false(
        const IndexTaskLauncher& launcher, IndexSpace launch_space,
        ReductionOpID redop, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (launcher.elide_future_return)
        return Future();
      // If there is an initial value for the reduction use that
      if (launcher.initial_value.impl != nullptr)
        return launcher.initial_value;
      // Otherwise set it to the identity value of the reduction operator
      FutureImpl* result = new FutureImpl(
          this, true /*register*/, runtime->get_available_distributed_id(),
          provenance);
      const ReductionOp* reduction_op = runtime->get_reduction(redop);
      result->set_local(&reduction_op->identity, reduction_op->sizeof_rhs);
      return Future(result);
    }

    //--------------------------------------------------------------------------
    VariantImpl* TaskContext::select_inline_variant(
        TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
        std::deque<InstanceSet>& physical_instances)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectVariantInput input;
      Mapper::SelectVariantOutput output;
      input.processor = executing_processor;
      input.chosen_instances.resize(parent_regions.size());
      // Extract the specific field instances for each region requirement
      for (unsigned idx1 = 0; idx1 < parent_regions.size(); idx1++)
      {
        const RegionRequirement& child_req = child->regions[idx1];
        FieldSpaceNode* space =
            runtime->get_node(child_req.parent.get_field_space());
        FieldMask mask = space->get_field_mask(child_req.privilege_fields);
        InstanceSet instances;
        parent_regions[idx1].impl->get_references(instances);
        for (unsigned idx2 = 0; idx2 < instances.size(); idx2++)
        {
          const InstanceRef& ref = instances[idx2];
          const FieldMask overlap = mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          physical_instances[idx1].add_instance(
              InstanceRef(ref.get_manager(), overlap));
          input.chosen_instances[idx1].emplace_back(
              MappingInstance(ref.get_manager()));
          mask -= overlap;
          if (!mask)
            break;
        }
        legion_assert(!mask);
      }
      // Always do this with the child mapper
      MapperManager* child_mapper =
          runtime->find_mapper(executing_processor, child->map_id);
      child_mapper->invoke_select_task_variant(child, input, output);
      VariantImpl* variant_impl = runtime->find_variant_impl(
          child->task_id, output.chosen_variant, true /*can fail*/);
      if (variant_impl == nullptr)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invoction of "
              << "'select_task_variant' by mapper " << *child_mapper
              << ". Mapper selected an invalid variant ID "
              << output.chosen_variant << " for inlining of task " << *child
              << ".";
        error.raise();
      }
      if (runtime->safe_mapper)
        child->validate_variant_selection(
            child_mapper, variant_impl, executing_processor.kind(),
            physical_instances, "select_task_variant");
      return variant_impl;
    }

  }  // namespace Internal
}  // namespace Legion
