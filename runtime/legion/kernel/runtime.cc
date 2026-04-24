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

#include "legion/kernel/runtime.h"
#include "legion/analysis/acquire.h"
#include "legion/analysis/across.h"
#include "legion/analysis/aggregator.h"
#include "legion/analysis/filter.h"
#include "legion/analysis/overwrite.h"
#include "legion/analysis/release.h"
#include "legion/analysis/valid.h"
#include "legion/analysis/update.h"
#include "legion/contexts/remote.h"
#include "legion/contexts/replicate.h"
#include "legion/contexts/toplevel.h"
#include "legion/instances/virtual.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/message.h"
#include "legion/managers/processor.h"
#include "legion/managers/shard.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/index.h"
#include "legion/nodes/region.h"
#include "legion/operations/copy.h"
#include "legion/operations/deletion.h"
#include "legion/operations/mustepoch.h"
#include "legion/operations/remote.h"
#include "legion/operations/timing.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/slice.h"
#include "legion/tracing/recognizer.h"
#include "legion/tracing/shard.h"
#include "legion/utilities/provenance.h"
#include "legion/views/allreduce.h"
#include "legion/views/fill.h"
#include "legion/views/materialized.h"
#include "legion/views/phi.h"
#include "legion/views/reduction.h"
#include "legion/views/replicate.h"
#include "mappers/default_mapper.h"
#include "mappers/test_mapper.h"
#include "mappers/replay_mapper.h"
#include "mappers/debug_mapper.h"

#include <algorithm>
#include <stdlib.h>
#include <unistd.h>  // sleep for warnings

#ifdef LEGION_TRACE_ALLOCATION
#include <sys/resource.h>
#endif

namespace Legion {
  namespace Internal {

    // If you add a logger, update the forward delarations in types.h
    Realm::Logger log_legion("legion");
    Realm::Logger log_allocation("allocation");
    Realm::Logger log_migration("migration");
    Realm::Logger log_prof("legion_prof");
    Realm::Logger log_garbage("legion_gc");
    Realm::Logger log_shutdown("shutdown");
    Realm::Logger log_tracing("tracing");
    Realm::Logger log_auto_trace("auto_trace");
    Realm::Logger log_spy("legion_spy");
    Realm::Logger log_registration("registration");

#ifdef LEGION_DEBUG_CALLERS
    thread_local LgTaskID implicit_task_kind = LG_SCHEDULER_ID;
    thread_local LgTaskID implicit_task_caller = LG_SCHEDULER_ID;
#endif

    const LgEvent LgEvent::NO_LG_EVENT = {};
    const ApEvent ApEvent::NO_AP_EVENT = {};
    const ApUserEvent ApUserEvent::NO_AP_USER_EVENT = {};
    const ApBarrier ApBarrier::NO_AP_BARRIER = {};
    const RtEvent RtEvent::NO_RT_EVENT = {};
    const RtUserEvent RtUserEvent::NO_RT_USER_EVENT = {};
    const RtBarrier RtBarrier::NO_RT_BARRIER = {};
    const PredEvent PredEvent::NO_PRED_EVENT = {};
    const PredUserEvent PredUserEvent::NO_PRED_USER_EVENT = {};

    /*static*/ Runtime* Runtime::the_runtime = nullptr;

    //--------------------------------------------------------------------------
    void LgEvent::begin_wait(Context ctx, bool from_application) const
    //--------------------------------------------------------------------------
    {
      if (ctx != nullptr)
        ctx->begin_wait(*this, from_application);
      else if (
          (implicit_profiler != nullptr) &&
          implicit_profiler->is_external_thread())
        implicit_profiler->begin_external_wait(*this);
    }

    //--------------------------------------------------------------------------
    void LgEvent::end_wait(Context ctx, bool from_application) const
    //--------------------------------------------------------------------------
    {
      if ((runtime->profiler != nullptr) && (implicit_profiler == nullptr) &&
          implicit_fevent.exists())
      {
        // This can occur when a (meta-)task that was previously running,
        // but waited on a event, and Realm wakes it up on a new kernel
        // thread that has never run a task before and therefore hasn't
        // been assigned an implicit profiler yet
        runtime->profiler->instantiate_profiling_instance();
      }
      if (ctx != nullptr)
        ctx->end_wait(*this, from_application);
      else if (
          (implicit_profiler != nullptr) &&
          implicit_profiler->is_external_thread())
        implicit_profiler->end_external_wait(*this);
    }

    //--------------------------------------------------------------------------
    void LgEvent::begin_mapper_call_wait(MappingCallInfo* call) const
    //--------------------------------------------------------------------------
    {
      call->begin_wait();
    }

    //--------------------------------------------------------------------------
    void LgEvent::record_event_wait(
        Realm::Backtrace& bt, ProvenanceID pid) const
    //--------------------------------------------------------------------------
    {
      legion_assert(exists());
      legion_assert(implicit_profiler != nullptr);
      implicit_profiler->record_event_wait(*this, pid, bt);
    }

    //--------------------------------------------------------------------------
    void LgEvent::record_event_trigger(LgEvent precondition) const
    //--------------------------------------------------------------------------
    {
      legion_assert(exists());
      legion_assert(implicit_profiler != nullptr);
      implicit_profiler->record_event_trigger(*this, precondition);
    }

    //--------------------------------------------------------------------------
    static inline bool compare_expressions(
        IndexSpaceExpression* one, IndexSpaceExpression* two)
    //--------------------------------------------------------------------------
    {
      return (one->expr_id < two->expr_id);
    }

    struct CompareExpressions {
    public:
      inline bool operator()(
          IndexSpaceExpression* one, IndexSpaceExpression* two) const
      {
        return compare_expressions(one, two);
      }
    };

    /////////////////////////////////////////////////////////////
    // Operation Creator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OperationCreator::OperationCreator(void) : result(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OperationCreator::~OperationCreator(void)
    //--------------------------------------------------------------------------
    {
      // If we still have a result then it's because it wasn't consumed need
      // we need to remove it's reference that was added by the constructor
      if ((result != nullptr) &&
          result->remove_base_resource_ref(REGION_TREE_REF))
        delete result;
    }

    //--------------------------------------------------------------------------
    void OperationCreator::produce(IndexSpaceOperation* op)
    //--------------------------------------------------------------------------
    {
      legion_assert(result == nullptr);
      result = op;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* OperationCreator::consume(void)
    //--------------------------------------------------------------------------
    {
      if (result == nullptr)
        create_operation();
      legion_assert(result != nullptr);
      // Add an expression reference here since this is going to be put
      // into the region tree expression trie data structure, the reference
      // will be removed when the expressions is removed from the trie
      result->add_base_gc_ref(REGION_TREE_REF);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Pending Registrations
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingVariantRegistration::PendingVariantRegistration(
        VariantID v, size_t return_size, bool has_return_size,
        const TaskVariantRegistrar& reg, const void* udata, size_t udata_size,
        const CodeDescriptor& realm, const char* task_name)
      : vid(v), return_type_size(return_size),
        has_return_type_size(has_return_size), registrar(reg),
        realm_desc(realm), logical_task_name(nullptr)
    //--------------------------------------------------------------------------
    {
      // If we're doing a pending registration, this is a static
      // registration so we don't have to register it globally
      registrar.global_registration = false;
      // Make sure we own the task variant name
      if (reg.task_variant_name != nullptr)
        registrar.task_variant_name = strdup(reg.task_variant_name);
      // We need to own the user data too
      if (udata != nullptr)
      {
        user_data_size = udata_size;
        user_data = malloc(user_data_size);
        memcpy(user_data, udata, user_data_size);
      }
      else
      {
        user_data_size = 0;
        user_data = nullptr;
      }
      if (task_name != nullptr)
        logical_task_name = strdup(task_name);
    }

    //--------------------------------------------------------------------------
    PendingVariantRegistration::~PendingVariantRegistration(void)
    //--------------------------------------------------------------------------
    {
      if (registrar.task_variant_name != nullptr)
        free(const_cast<char*>(registrar.task_variant_name));
      if (user_data != nullptr)
        free(user_data);
      if (logical_task_name != nullptr)
        free(logical_task_name);
    }

    //--------------------------------------------------------------------------
    void PendingVariantRegistration::perform_registration(void)
    //--------------------------------------------------------------------------
    {
      // If we have a logical task name, attach the name info
      // Do this first before any logging for the variant
      if (logical_task_name != nullptr)
        runtime->attach_semantic_information(
            registrar.task_id, LEGION_NAME_SEMANTIC_TAG, logical_task_name,
            strlen(logical_task_name) + 1, false /*mutable*/,
            false /*send to owner*/);
      runtime->register_variant(
          registrar, user_data, user_data_size, realm_desc, return_type_size,
          has_return_type_size, vid, false /*check task*/,
          true /*check context*/, true /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    PendingRegistrationCallback::PendingRegistrationCallback(
        RegistrationCallback call, bool dedup, size_t tag)
      : withoutargs(call), dedup_tag(tag), deduplicate(dedup), has_args(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(withoutargs);
    }

    //--------------------------------------------------------------------------
    PendingRegistrationCallback::PendingRegistrationCallback(
        RegistrationWithArgsCallback call, const UntypedBuffer& buf, bool dedup,
        size_t tag)
      : withargs(call), buffer(buf), dedup_tag(tag), deduplicate(dedup),
        has_args(true)
    //--------------------------------------------------------------------------
    {
      legion_assert(withargs);
    }

    //--------------------------------------------------------------------------
    PendingRegistrationCallback::PendingRegistrationCallback(
        PendingRegistrationCallback&& rhs)
      : withargs(nullptr), buffer(rhs.buffer), dedup_tag(rhs.dedup_tag),
        deduplicate(rhs.deduplicate), has_args(rhs.has_args)
    //--------------------------------------------------------------------------
    {
      rhs.buffer = UntypedBuffer();
      if (has_args)
      {
        withargs = std::move(rhs.withargs);
        legion_assert(withargs);
      }
      else
      {
        withoutargs = std::move(rhs.withoutargs);
        legion_assert(withoutargs);
      }
    }

    //--------------------------------------------------------------------------
    PendingRegistrationCallback::~PendingRegistrationCallback(void)
    //--------------------------------------------------------------------------
    {
      if (has_args)
        withargs.~RegistrationWithArgsCallback();
      else
        withoutargs.~RegistrationCallback();
    }

    /////////////////////////////////////////////////////////////
    // Legion Runtime
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Runtime::Runtime(
        Machine m, const LegionConfiguration& config, bool background,
        InputArgs args, AddressSpaceID unique, Memory system,
        const std::set<Processor>& locals,
        const std::set<Processor>& local_utilities,
        const std::set<AddressSpaceID>& address_spaces, bool default_mapper)
      : external(new Legion::Runtime(this)),
        mapper_runtime(new Legion::Mapping::MapperRuntime(this)), machine(m),
        runtime_system_memory(system), address_space(unique),
        total_address_spaces(address_spaces.size()),
        runtime_stride(address_spaces.size()), profiler(nullptr),
        virtual_manager(nullptr),
        num_utility_procs(
            local_utilities.empty() ? locals.size() : local_utilities.size()),
        input_args(args),
        initial_task_window_size(config.initial_task_window_size),
        initial_task_window_hysteresis(config.initial_task_window_hysteresis),
        initial_tasks_to_schedule(config.initial_tasks_to_schedule),
        initial_meta_task_vector_width(config.initial_meta_task_vector_width),
        max_message_size(config.max_message_size),
        gc_epoch_size(config.gc_epoch_size),
        max_control_replication_contexts(
            config.max_control_replication_contexts),
        max_local_fields(config.max_local_fields),
        max_replay_parallelism(config.max_replay_parallelism),
        safe_control_replication(config.safe_control_replication),
        program_order_execution(config.program_order_execution),
        dump_physical_traces(config.dump_physical_traces),
        no_tracing(config.no_tracing),
        no_physical_tracing(
            config.no_physical_tracing || no_tracing ||
            program_order_execution),
        no_auto_tracing(
            config.no_auto_tracing || no_tracing || program_order_execution),
        no_trace_optimization(config.no_trace_optimization),
        no_fence_elision(config.no_fence_elision),
        no_transitive_reduction(config.no_transitive_reduction),
        inline_transitive_reduction(config.inline_transitive_reduction),
        replay_on_cpus(config.replay_on_cpus),
        verify_partitions(config.verify_partitions),
        runtime_warnings(config.runtime_warnings),
        warnings_backtrace(config.warnings_backtrace),
        warnings_are_errors(config.warnings_are_errors),
        report_leaks(config.report_leaks),
        record_registration(config.record_registration),
        stealing_disabled(config.stealing_disabled),
        resilient_mode(config.resilient_mode),
        unsafe_launch(config.unsafe_launch),
#ifdef LEGION_DEBUG
        safe_mapper(!config.unsafe_mapper),
#else
        safe_mapper(config.safe_mapper),
#endif
        safe_model(config.safe_model),
        safe_tracing(config.safe_tracing || config.safe_model),
        disable_independence_tests(config.disable_independence_tests),
        enable_pointwise_analysis(config.enable_pointwise_analysis),
        supply_default_mapper(default_mapper),
        enable_test_mapper(config.enable_test_mapper),
        legion_ldb_enabled(!config.ldb_file.empty()),
        replay_file(legion_ldb_enabled ? config.ldb_file : config.replay_file),
        verbose_logging(config.verbose_logging),
        dump_free_ranges(config.dump_free_ranges),
        legion_collective_radix(config.legion_collective_radix),
        mpi_rank_table(
            (mpi_rank >= 0) ? new MPIRankTable(
                                  legion_collective_radix, address_space,
                                  total_address_spaces) :
                              nullptr),
        prepared_for_shutdown(false), total_outstanding_tasks(0),
        outstanding_top_level_tasks(initialize_outstanding_top_level_tasks(
            address_space, total_address_spaces, legion_collective_radix)),
        local_procs(locals), local_utils(local_utilities),
        unique_index_tree_id((unique == 0) ? runtime_stride : unique),
        unique_field_id(
            LEGION_MAX_APPLICATION_FIELD_ID +
            ((unique == 0) ? runtime_stride : unique)),
        unique_operation_id((unique == 0) ? runtime_stride : unique),
        unique_code_descriptor_id(
            LG_TASK_ID_AVAILABLE + ((unique == 0) ? runtime_stride : unique)),
        unique_constraint_id(
            LEGION_MAX_APPLICATION_LAYOUT_ID +
            (((LEGION_MAX_APPLICATION_LAYOUT_ID % runtime_stride) <= unique) ?
                 (runtime_stride -
                  ((LEGION_MAX_APPLICATION_LAYOUT_ID % runtime_stride) -
                   unique)) :
                 (unique -
                  (LEGION_MAX_APPLICATION_LAYOUT_ID % runtime_stride)))),
        unique_is_expr_id((unique == 0) ? runtime_stride : unique),
        unique_top_level_task_id((unique == 0) ? runtime_stride : unique),
        unique_implicit_top_level_task_id(0),
        unique_indirections_id((unique == 0) ? runtime_stride : unique),
        unique_task_id(get_current_static_task_id() + unique),
        unique_mapper_id(get_current_static_mapper_id() + unique),
        unique_trace_id(get_current_static_trace_id() + unique),
        unique_projection_id(get_current_static_projection_id() + unique),
        unique_sharding_id(get_current_static_sharding_id() + unique),
        unique_concurrent_id(get_current_static_concurrent_id() + unique),
        unique_exception_handler_id(
            get_current_static_exception_handler_id() + unique),
        unique_redop_id(get_current_static_reduction_id() + unique),
        unique_serdez_id(get_current_static_serdez_id() + unique),
        unique_library_mapper_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_trace_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_projection_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_sharding_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_concurrent_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_task_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_redop_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_library_serdez_id(LEGION_INITIAL_LIBRARY_ID_OFFSET),
        unique_distributed_id((unique == 0) ? runtime_stride : unique)
    //--------------------------------------------------------------------------
    {
      legion_assert(runtime == nullptr);
      runtime = this;
      the_runtime = this;  // for debugging
      legion_assert((unique_constraint_id % runtime_stride) == unique);
      if (LEGION_MAX_NUM_NODES <= address_space)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error
            << "Maximum number of nodes exceeded. Detected node "
            << address_space << " but 'LEGION_MAX_NUM_NODES' is set to "
            << LEGION_MAX_NUM_NODES
            << ". Change the value of 'LEGION_MAX_NUM_NODES' in "
               "legion_config.h "
            << "and recompile. Please note that 'LEGION_MAX_NUM_NODES' must be "
            << "a power of two.";
        error.raise();
      }
      // Construct a local utility processor group
      if (local_utils.empty())
      {
        // make the utility group the set of all the local processors
        legion_assert(!locals.empty());
        if (locals.size() == 1)
          utility_group = *(locals.begin());
        else
        {
          std::vector<Processor> util_group(locals.begin(), locals.end());
          utility_group = ProcessorGroup::create_group(util_group);
        }
        local_group = utility_group;
      }
      else
      {
        if (local_utils.size() == 1)
          utility_group = *(local_utils.begin());
        else
        {
          std::vector<Processor> util_g(local_utils.begin(), local_utils.end());
          utility_group = ProcessorGroup::create_group(util_g);
        }
        std::vector<Processor> all_local(locals.begin(), locals.end());
        all_local.insert(
            all_local.end(), local_utils.begin(), local_utils.end());
        local_group = ProcessorGroup::create_group(all_local);
      }
      legion_assert(utility_group.exists());
      // For each of the processors in our local set construct a manager
      for (const Processor& proc : local_procs)
      {
        legion_assert(proc.kind() != Processor::UTIL_PROC);
        ProcessorManager* manager = new ProcessorManager(
            proc, proc.kind(), LEGION_DEFAULT_MAPPER_SLOTS, stealing_disabled,
            !replay_file.empty());
        proc_managers[proc] = manager;
      }
      // Register our meta tasks
#define META_TASK_REGISTRATION(kind, type, name)   \
  legion_assert(meta_task_table[kind] == nullptr); \
  meta_task_table[kind] = type::handle;
      LEGION_META_TASKS(META_TASK_REGISTRATION)
#undef META_TASK_REGISTRATION
      // Register active message handlers
      MessageManager::register_handlers();
      // Initialize the message manager array so that we can construct
      // message managers lazily as they are needed
      for (unsigned idx = 0; idx < LEGION_MAX_NUM_NODES; idx++)
        message_managers[idx].store(nullptr);

      // Make the default number of contexts
      // No need to hold the lock yet because nothing is running
      total_contexts = LEGION_DEFAULT_CONTEXTS;
      available_contexts.resize(LEGION_DEFAULT_CONTEXTS);
      // Add in reverse order so lower numbers get popped off first
      for (unsigned idx = 0; idx < LEGION_DEFAULT_CONTEXTS; idx++)
        available_contexts[idx] = LEGION_DEFAULT_CONTEXTS - (idx + 1);
      // Initialize our random number generator state
      random_state[0] = address_space & 0xFFFF;  // low-order bits of node ID
      random_state[1] = (address_space >> 16) & 0xFFFF;  // high-order bits
      random_state[2] = LEGION_INIT_SEED;
      // Do some mixing
      for (int i = 0; i < 256; i++) nrand48(random_state);
      // We've intentionally switched this to profile all the nodes if we're
      // profiling any nodes since some information about things like copies
      // usage of instances are now split across multiple log files
      if (config.num_profiling_nodes > 0)
        initialize_legion_prof(config);

      if (spy_logging_level > NO_SPY_LOGGING)
        log_local_machine();

#ifdef LEGION_TRACE_ALLOCATION
      allocation_tracing_count.store(0);
#endif
#ifdef LEGION_GC
      {
        REFERENCE_NAMES_ARRAY(reference_names);
        for (unsigned idx = 0; idx < LAST_SOURCE_REF; idx++)
        {
          log_garbage.info("GC Source Kind %d %s", idx, reference_names[idx]);
        }
      }
#endif
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      outstanding_counts = std::vector<std::atomic<int> >(LG_LAST_TASK_ID);
      for (unsigned idx = 0; idx < outstanding_counts.size(); idx++)
        outstanding_counts[idx].store(0);
#endif
    }

    //--------------------------------------------------------------------------
    Runtime::~Runtime(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(outstanding_operations.empty());
      if (profiler != nullptr)
      {
        delete profiler;
        profiler = nullptr;
      }
      // Make sure we don't send anymore messages
      for (unsigned idx = 0; idx < LEGION_MAX_NUM_NODES; idx++)
      {
        MessageManager* manager = message_managers[idx].load();
        if (manager != nullptr)
        {
          delete manager;
          message_managers[idx].store(nullptr);
        }
      }
      // Free any input arguments
      if (input_args.argc > 0)
      {
        for (int i = 0; i < input_args.argc; i++)
          if (input_args.argv[i] != nullptr)
            free(input_args.argv[i]);
        free(input_args.argv);
      }
      delete external;
      delete mapper_runtime;
      // Avoid duplicate deletions on these for separate runtime
      // instances by just leaking them for now
      for (std::pair<const ProjectionID, ProjectionFunction*>& proj_pair :
           projection_functions)
      {
        delete proj_pair.second;
      }
      projection_functions.clear();
      for (std::pair<const ShardingID, ShardingFunctor*>& shard_pair :
           sharding_functors)
      {
        delete shard_pair.second;
      }
      sharding_functors.clear();
      for (std::pair<const ConcurrentID, ConcurrentColoringFunctor*>&
               conc_pair : concurrent_functors)
        delete conc_pair.second;
      concurrent_functors.clear();
      for (std::pair<const ExceptionHandlerID, ExceptionHandler*>& exc_pair :
           exception_handlers)
        delete exc_pair.second;
      exception_handlers.clear();
      for (const std::pair<const Processor, ProcessorManager*>& proc_pair :
           proc_managers)
      {
        delete proc_pair.second;
      }
      proc_managers.clear();
      for (const std::pair<const TaskID, TaskImpl*>& task_pair : task_table)
      {
        delete task_pair.second;
      }
      task_table.clear();
      for (VariantImpl* const & variant : variant_table)
      {
        delete variant;
      }
      variant_table.clear();
      // Skip this if we are in separate runtime mode
      while (!layout_constraints_table.empty())
      {
        std::map<LayoutConstraintID, LayoutConstraints*>::iterator next_it =
            layout_constraints_table.begin();
        LayoutConstraints* next = next_it->second;
        layout_constraints_table.erase(next_it);
        if (next->remove_base_resource_ref(RUNTIME_REF))
          delete (next);
      }
      while (!layout_constraints_table.empty())
      {
        std::map<LayoutConstraintID, LayoutConstraints*>::iterator next_it =
            layout_constraints_table.begin();
        LayoutConstraints* next = next_it->second;
        layout_constraints_table.erase(next_it);
        if (next->remove_base_resource_ref(RUNTIME_REF))
          delete (next);
      }
      // We can also delete all of our reduction operators
      ReductionOpTable& redop_table = get_reduction_table(true /*safe*/);
      while (!redop_table.empty())
      {
        ReductionOpTable::iterator it = redop_table.begin();
        // Free ReductionOp *'s with free, not delete!
        static_assert(
            std::is_trivially_destructible<
                typename std::decay<decltype(*(it->second))>::type>::value,
            "ReducionOp must be trivially destructible");
        free(it->second);
        redop_table.erase(it);
      }
      for (rt::map<uint64_t, rt::deque<ProcessorGroupInfo> >::const_iterator
               git = processor_groups.begin();
           git != processor_groups.end(); git++)
        for (rt::deque<ProcessorGroupInfo>::const_iterator it =
                 git->second.begin();
             it != git->second.end(); it++)
          it->processor_group.destroy();
      for (const std::pair<const Memory, MemoryManager*>& mem_pair :
           memory_managers)
      {
        delete mem_pair.second;
      }
      memory_managers.clear();
      for (const std::pair<const uint64_t, Provenance*>& prov_pair :
           provenances)
        if (prov_pair.second->remove_reference())
          delete prov_pair.second;
      provenances.clear();
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_variants(void)
    //--------------------------------------------------------------------------
    {
      std::deque<PendingVariantRegistration*>& pending_variants =
          get_pending_variant_table();
      if (!pending_variants.empty())
      {
        for (PendingVariantRegistration* pending : pending_variants)
        {
          pending->perform_registration();
          delete pending;
        }
        pending_variants.clear();
      }
    }

    //--------------------------------------------------------------------------
    CollectiveMapping* Runtime::register_static_constraints(
        uint64_t& next_static_did, LayoutConstraintID& virtual_layout_id)
    //--------------------------------------------------------------------------
    {
      // Register any pending constraint sets
      std::map<LayoutConstraintID, LayoutConstraintRegistrar>&
          pending_constraints = get_pending_constraint_table();
      // Create a collective mapping for all the nodes
      CollectiveMapping* mapping = nullptr;
      if (total_address_spaces > 1)
      {
        std::vector<AddressSpaceID> all_spaces(total_address_spaces);
        for (unsigned idx = 0; idx < total_address_spaces; idx++)
          all_spaces[idx] = idx;
        mapping = new CollectiveMapping(all_spaces, legion_collective_radix);
      }
      unsigned already_used = 0;
      // Now do the registrations
      std::map<AddressSpaceID, unsigned> address_counts;
      for (const std::pair<const LayoutConstraintID, LayoutConstraintRegistrar>&
               it : pending_constraints)
      {
        if (LEGION_MAX_APPLICATION_LAYOUT_ID < it.first)
          already_used++;
        register_layout(
            it.second, it.first,
            get_next_static_distributed_id(next_static_did), mapping);
      }
      // Now register the virtual layout constraints
      LayoutConstraintRegistrar virtual_registrar;
      virtual_registrar.add_constraint(
          SpecializedConstraint(LEGION_VIRTUAL_SPECIALIZE));
      virtual_layout_id = LEGION_MAX_APPLICATION_LAYOUT_ID + ++already_used;
      register_layout(
          virtual_registrar, virtual_layout_id,
          get_next_static_distributed_id(next_static_did), mapping);
      // Bump up our unique constraint ID if we already used the IDs statically
      while (unique_constraint_id <=
             (LEGION_MAX_APPLICATION_LAYOUT_ID + already_used))
        unique_constraint_id += runtime_stride;
      pending_constraints.clear();
      return mapping;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_projections(void)
    //--------------------------------------------------------------------------
    {
      std::map<ProjectionID, ProjectionFunctor*>& pending_projection_functors =
          get_pending_projection_table();
      for (const std::pair<const ProjectionID, ProjectionFunctor*>& it :
           pending_projection_functors)
      {
        it.second->set_runtime(external);
        register_projection_functor(
            it.first, it.second, true /*need check*/,
            true /*was preregistered*/, nullptr, true /*pregistered*/);
      }
      register_projection_functor(
          0, new IdentityProjectionFunctor(this->external),
          false /*need check*/, true /*was preregistered*/, nullptr,
          true /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_sharding_functors(void)
    //--------------------------------------------------------------------------
    {
      std::map<ShardingID, ShardingFunctor*>& pending_sharding_functors =
          get_pending_sharding_table();
      for (const std::pair<const ShardingID, ShardingFunctor*>& it :
           pending_sharding_functors)
        register_sharding_functor(
            it.first, it.second, true /*zero check*/,
            true /*was preregistered*/, nullptr, true /*preregistered*/);
      register_sharding_functor(
          0, new CyclicShardingFunctor(), false /*need check*/,
          true /*was preregistered*/, nullptr, true /*preregistered*/);
      // Register the attach-detach sharding functor
      ReplicateContext::register_attach_detach_sharding_functor();
      // Register the universal sharding functor
      ReplicateContext::register_universal_sharding_functor();
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_concurrent_functors(void)
    //--------------------------------------------------------------------------
    {
      std::map<ConcurrentID, ConcurrentColoringFunctor*>&
          pending_concurrent_functors = get_pending_concurrent_table();
      for (const std::pair<const ConcurrentID, ConcurrentColoringFunctor*>& it :
           pending_concurrent_functors)
        register_concurrent_functor(
            it.first, it.second, true /*zero check*/,
            true /*was preregistered*/, nullptr, true /*preregistered*/);
      register_concurrent_functor(
          0, new ZeroColoringFunctor(), false /*need check*/,
          true /*was preregistered*/, nullptr, true /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_static_exception_handlers(void)
    //--------------------------------------------------------------------------
    {
      std::map<ExceptionHandlerID, ExceptionHandler*>&
          pending_exception_handlers = get_pending_exception_handler_table();
      for (const std::pair<const ExceptionHandlerID, ExceptionHandler*>& it :
           pending_exception_handlers)
        register_exception_handler(
            it.first, it.second, true /*zero check*/,
            true /*was preregistered*/);
      register_exception_handler(
          0, new ExceptionHandler(), false /*need check*/,
          true /*was preregistered*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::initialize_legion_prof(const LegionConfiguration& config)
    //--------------------------------------------------------------------------
    {
      // For the profiler we want to find as many "holes" in the execution
      // as possible in which to run profiler tasks so we can minimize the
      // overhead on the application. To do this we want profiler tasks to
      // run on any processor that has a dedicated core which is either any
      // CPU processor a utility processor. There's no need to use GPU or
      // I/O processors since they share the same cores as the utility cores.
      // In the future we can relax this to use any processor core that doesn't
      // support multiple threads executing concurrently (e.g. I/O procs) as
      // that could lead to a lot of profiling instances being made since this
      // will clear the implicit_profiler if we're not self-profiling and then
      // we'll have to make new instances in
      // LegionProfiler::instantiate_profiling_instance the next time we
      // go to profile anything on that processor.
      std::vector<Processor> prof_procs(local_utils.begin(), local_utils.end());
      for (const Processor& proc : local_procs)
      {
        if (proc.kind() == Processor::LOC_PROC)
          prof_procs.emplace_back(proc);
      }
      legion_assert(!prof_procs.empty());
      const Processor target_proc_for_profiler =
          prof_procs.size() > 1 ? ProcessorGroup::create_group(prof_procs) :
                                  prof_procs.front();
      const char* lg_task_descriptions[LG_LAST_TASK_ID] = {
#define META_TASK_NAMES(kind, type, name) name,
          LEGION_META_TASKS(META_TASK_NAMES)
#undef META_TASK_NAMES
      };
      const char* lg_message_descriptions[LAST_SEND_KIND] = {
#define CTRL_REPL_MESSAGE_NAMES(kind, name) name,
          LEGION_SHARD_COLLECTIVE_ACTIVE_MESSAGES(CTRL_REPL_MESSAGE_NAMES)
#undef CTRL_REPL_MESSAGE_NAMES
#define MESSAGE_NAMES(kind, type, name, resp, escape_ctx, escape_op) name,
              LEGION_ACTIVE_MESSAGES(MESSAGE_NAMES)
#undef MESSAGE_NAMES
      };
      static_assert(
          (LG_MESSAGE_ID + 1) == LG_LAST_TASK_ID,
          "LG_MESSAGE_ID must always be the last meta-task ID");
      profiler = new LegionProfiler(
          target_proc_for_profiler, machine, LG_MESSAGE_ID,
          lg_task_descriptions, LAST_SEND_KIND, lg_message_descriptions,
          LAST_OP_KIND, Operation::op_names, config.serializer_type.c_str(),
          config.prof_logfile.c_str(), total_address_spaces,
          config.prof_footprint_threshold << 20, config.prof_target_latency,
          config.prof_call_threshold, config.slow_config_ok,
          config.prof_self_profile, config.prof_no_critical_paths,
          config.prof_all_critical_arrivals);
      MAPPER_CALL_NAMES(lg_mapper_calls);
      profiler->record_mapper_call_kinds(lg_mapper_calls, LAST_MAPPER_CALL);
      RUNTIME_CALL_DESCRIPTIONS(lg_runtime_calls);
      profiler->record_runtime_call_kinds(
          lg_runtime_calls, LAST_RUNTIME_CALL_KIND);
    }

    //--------------------------------------------------------------------------
    void Runtime::log_local_machine(void) const
    //--------------------------------------------------------------------------
    {
      std::set<Processor::Kind> proc_kinds;
      Machine::ProcessorQuery local_procs(machine);
      local_procs.local_address_space();
#define COUNTER(X, Y) +1
      constexpr size_t num_procs = REALM_PROCESSOR_KINDS(COUNTER);
      static_assert(num_procs == 9, "Add new processor kinds");
#undef COUNTER
      // Log processors
      for (Machine::ProcessorQuery::iterator it = local_procs.begin();
           it != local_procs.end(); it++)
      {
        Processor::Kind kind = it->kind();
        if (proc_kinds.find(kind) == proc_kinds.end())
        {
          switch (kind)
          {
            case Processor::NO_KIND:
              {
                LegionSpy::log_processor_kind(kind, "NoProc");
                break;
              }
            case Processor::TOC_PROC:
              {
                LegionSpy::log_processor_kind(kind, "GPU");
                break;
              }
            case Processor::LOC_PROC:
              {
                LegionSpy::log_processor_kind(kind, "CPU");
                break;
              }
            case Processor::UTIL_PROC:
              {
                LegionSpy::log_processor_kind(kind, "Utility");
                break;
              }
            case Processor::IO_PROC:
              {
                LegionSpy::log_processor_kind(kind, "IO");
                break;
              }
            case Processor::PROC_GROUP:
              {
                LegionSpy::log_processor_kind(kind, "ProcGroup");
                break;
              }
            case Processor::PROC_SET:
              {
                LegionSpy::log_processor_kind(kind, "ProcSet");
                break;
              }
            case Processor::OMP_PROC:
              {
                LegionSpy::log_processor_kind(kind, "OpenMP");
                break;
              }
            case Processor::PY_PROC:
              {
                LegionSpy::log_processor_kind(kind, "Python");
                break;
              }
            default:
              std::abort();  // unknown processor kind
          }
          proc_kinds.insert(kind);
        }
        LegionSpy::log_processor(it->id, kind);
      }
      // Log memories
      std::set<Memory::Kind> mem_kinds;
      Machine::MemoryQuery local_mems(machine);
      local_mems.local_address_space();
#define COUNTER(X, Y) +1
      constexpr size_t num_mems = REALM_MEMORY_KINDS(COUNTER);
      static_assert(num_mems == 15, "Add new memory kinds");
#undef COUNTER
      for (Machine::MemoryQuery::iterator it = local_mems.begin();
           it != local_mems.end(); it++)
      {
        Memory::Kind kind = it->kind();
        if (mem_kinds.find(kind) == mem_kinds.end())
        {
          switch (kind)
          {
            case Memory::GLOBAL_MEM:
              {
                LegionSpy::log_memory_kind(kind, "GASNet");
                break;
              }
            case Memory::SYSTEM_MEM:
              {
                LegionSpy::log_memory_kind(kind, "System");
                break;
              }
            case Memory::REGDMA_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Registered");
                break;
              }
            case Memory::SOCKET_MEM:
              {
                LegionSpy::log_memory_kind(kind, "NUMA");
                break;
              }
            case Memory::Z_COPY_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Zero-Copy");
                break;
              }
            case Memory::GPU_FB_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Framebuffer");
                break;
              }
            case Memory::DISK_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Disk");
                break;
              }
            case Memory::HDF_MEM:
              {
                LegionSpy::log_memory_kind(kind, "HDF");
                break;
              }
            case Memory::FILE_MEM:
              {
                LegionSpy::log_memory_kind(kind, "File");
                break;
              }
            case Memory::LEVEL3_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L3");
                break;
              }
            case Memory::LEVEL2_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L2");
                break;
              }
            case Memory::LEVEL1_CACHE:
              {
                LegionSpy::log_memory_kind(kind, "L1");
                break;
              }
            case Memory::GPU_MANAGED_MEM:
              {
                LegionSpy::log_memory_kind(kind, "UVM");
                break;
              }
            case Memory::GPU_DYNAMIC_MEM:
              {
                LegionSpy::log_memory_kind(kind, "Dynamic Framebuffer");
                break;
              }
            default:
              std::abort();  // unknown memory kind
          }
        }
        LegionSpy::log_memory(it->id, it->capacity(), it->kind());
      }
      // Log Proc-Mem Affinity
      Machine::ProcessorQuery local_procs2(machine);
      local_procs2.local_address_space();
      for (Machine::ProcessorQuery::iterator pit = local_procs2.begin();
           pit != local_procs2.end(); pit++)
      {
        std::vector<ProcessorMemoryAffinity> affinities;
        machine.get_proc_mem_affinity(affinities, *pit);
        for (const ProcessorMemoryAffinity& it : affinities)
        {
          LegionSpy::log_proc_mem_affinity(
              pit->id, it.m.id, it.bandwidth, it.latency);
        }
      }
      // Log Mem-Mem Affinity
      Machine::MemoryQuery local_mems2(machine);
      local_mems2.local_address_space();
      for (Machine::MemoryQuery::iterator mit = local_mems2.begin();
           mit != local_mems2.begin(); mit++)
      {
        std::vector<MemoryMemoryAffinity> affinities;
        machine.get_mem_mem_affinity(affinities, *mit);
        for (const MemoryMemoryAffinity& it : affinities)
        {
          LegionSpy::log_mem_mem_affinity(
              it.m1.id, it.m2.id, it.bandwidth, it.latency);
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::initialize_mappers(void)
    //--------------------------------------------------------------------------
    {
      if (replay_file.empty())  // This is the normal path
      {
        if (enable_test_mapper)
        {
          // Make test mappers for everyone
          for (const std::pair<const Processor, ProcessorManager*>& it :
               proc_managers)
          {
            Mapper* mapper =
                new Mapping::TestMapper(mapper_runtime, machine, it.first);
            MapperManager* wrapper = wrap_mapper(mapper, 0, it.first);
            it.second->add_mapper(0, wrapper, false /*check*/);
          }
        }
        else if (supply_default_mapper)
        {
          // Make default mappers for everyone
          for (const std::pair<const Processor, ProcessorManager*>& it :
               proc_managers)
          {
            Mapper* mapper =
                new Mapping::DefaultMapper(mapper_runtime, machine, it.first);
            MapperManager* wrapper =
                wrap_mapper(mapper, 0, it.first, true /*is default mapper*/);
            it.second->add_mapper(0, wrapper, false /*check*/);
          }
        }
      }
      else  // This is the replay/debug path
      {
        if (legion_ldb_enabled)
        {
          // This path is not quite ready yet
          std::abort();
          for (const std::pair<const Processor, ProcessorManager*>& it :
               proc_managers)
          {
            Mapper* mapper = new Mapping::DebugMapper(
                mapper_runtime, machine, it.first, replay_file.c_str());
            MapperManager* wrapper = wrap_mapper(mapper, 0, it.first);
            it.second->add_mapper(
                0, wrapper, false /*check*/, true /*skip replay*/);
          }
        }
        else
        {
          for (const std::pair<const Processor, ProcessorManager*>& it :
               proc_managers)
          {
            Mapper* mapper = new Mapping::ReplayMapper(
                mapper_runtime, machine, it.first, replay_file.c_str());
            MapperManager* wrapper = wrap_mapper(mapper, 0, it.first);
            it.second->add_mapper(
                0, wrapper, false /*check*/, true /*skip replay*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::initialize_virtual_manager(
        uint64_t& next_static_did, LayoutConstraintID virtual_layout_id,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      legion_assert(virtual_manager == nullptr);
      // make a layout constraints
      FieldMask all_ones(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      std::vector<unsigned> mask_index_map;
      std::vector<CustomSerdezID> serdez;
      std::vector<std::pair<FieldID, size_t> > field_sizes;
      LayoutConstraints* virtual_constraints =
          find_layout_constraints(virtual_layout_id);
      LayoutDescription* layout =
          new LayoutDescription(all_ones, virtual_constraints);
      virtual_manager = new VirtualManager(0 /*did*/, layout, mapping);
      virtual_manager->add_base_gc_ref(NEVER_GC_REF);
    }

    //--------------------------------------------------------------------------
    TopLevelContext* Runtime::initialize_runtime(Processor first_proc)
    //--------------------------------------------------------------------------
    {
      // If we have an MPI rank table do the exchanges before initializing
      // the mappers as they may want to look at the rank table
      if (mpi_rank_table != nullptr)
        mpi_rank_table->perform_rank_exchange();
      // Starts at 1 since 0 is a reserved ID for the virtual manager
      uint64_t next_static_did = 1;
      // Pull in any static registrations that were done
      LayoutConstraintID virtual_layout_id = 0;
      CollectiveMapping* mapping =
          register_static_constraints(next_static_did, virtual_layout_id);
      register_static_variants();
      register_static_projections();
      register_static_sharding_functors();
      register_static_concurrent_functors();
      register_static_exception_handlers();
      // Has to come after registring the static constraints
      initialize_virtual_manager(next_static_did, virtual_layout_id, mapping);
      // Initialize the mappers
      initialize_mappers();
      // Finally perform the registration callback methods
      std::vector<PendingRegistrationCallback>& registration_callbacks =
          get_pending_registration_callbacks();
      if (!registration_callbacks.empty())
      {
        for (const PendingRegistrationCallback& it : registration_callbacks)
        {
          perform_registration_callback(
              it, false /*global*/, true /*preregistered*/);
          if (it.buffer.get_size() > 0)
            free(it.buffer.get_ptr());
        }
        registration_callbacks.clear();
      }
      // If we have main top-level task, make a context for it
      if (legion_main_set)
      {
        TopLevelContext* top_context = new TopLevelContext(
            first_proc, get_unique_top_level_task_id(), 0 /*implicit*/,
            get_next_static_distributed_id(next_static_did), mapping);
        top_context->register_with_runtime();
        return top_context;
      }
      else
        return nullptr;
    }

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void Runtime::send_registration_callback(
        AddressSpaceID target, Realm::DSOReferenceImplementation* dso,
        RtEvent global_done_event, std::set<RtEvent>& applied_events,
        const void* buffer, size_t buffer_size, bool withargs, bool deduplicate,
        size_t dedup_tag)
    //--------------------------------------------------------------------------
    {
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      RegistrationCallbackMessage rez;
      {
        RezCheck z(rez);
        const size_t dso_size = dso->dso_name.size() + 1;
        const size_t sym_size = dso->symbol_name.size() + 1;
        rez.serialize(dso_size);
        rez.serialize(dso->dso_name.c_str(), dso_size);
        rez.serialize(sym_size);
        rez.serialize(dso->symbol_name.c_str(), sym_size);
        rez.serialize(buffer_size);
        if (buffer_size > 0)
          rez.serialize(buffer, buffer_size);
        rez.serialize<bool>(withargs);
        rez.serialize<bool>(deduplicate);
        rez.serialize(dedup_tag);
        rez.serialize(global_done_event);
        rez.serialize(done_event);
      }
      rez.dispatch(target);
      applied_events.insert(done_event);
    }
#endif  // LEGION_USE_LIBDL

    //--------------------------------------------------------------------------
    RtEvent Runtime::perform_registration_callback(
        const PendingRegistrationCallback& callback, bool global,
        bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (inside_registration_callback)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Nested registration callbacks are not permitted.";
        error.raise();
      }
      RegistrationKey global_key;
#ifdef LEGION_USE_LIBDL
      Realm::DSOReferenceImplementation* dso = nullptr;
      if (global)
      {
        // No such thing as global registration if there's only one addres space
        if (total_address_spaces > 1)
        {
          // First check to that we can convert the std::function back into
          // a function pointer, if we can't do that then there's no hope
          void (*raw_fnptr)() = nullptr;
          if (callback.has_args)
          {
            const RegistrationWithArgsCallbackFnptr* p =
                callback.withargs.target<RegistrationWithArgsCallbackFnptr>();
            if (p != nullptr)
              raw_fnptr = reinterpret_cast<void (*)()>(*p);
          }
          else
          {
            const RegistrationCallbackFnptr* p =
                callback.withoutargs.target<RegistrationCallbackFnptr>();
            if (p != nullptr)
              raw_fnptr = reinterpret_cast<void (*)()>(*p);
          }
          if (raw_fnptr == nullptr)
          {
            Fatal fatal;
            fatal
                << "Global registration callback function pointer " << raw_fnptr
                << " is not portable. All registration callbacks requesting to "
                << "be performed 'globally' must be able to be recognized by "
                << "a call to 'dladdr'. This requires that they come from a "
                << "shared object or the binary is linked with the '-rdynamic' "
                << "flag.";
#ifdef __clang__
            fatal << " It looks like you compiled Legion with clang so if "
                  << "your callback is actually a function pointer it's likely "
                  << "that you are hitting this LLVM bug: "
                  << "https://github.com/llvm/llvm-project/issues/36746 "
                  << "Please double check before reporting any issues.";
#endif
            fatal.raise();
          }
          const Realm::FunctionPointerImplementation impl(raw_fnptr);
          // Convert this to it's portable representation or raise an error
          // This is a little scary, we could still be inside of dlopen when
          // we get this call as part of the constructor for a shared object
          // and yet we're about to do a call to dladdr. This seems to work
          // but there is no documentation anywhere about whether this is
          // legal or safe to do...
          legion_assert(callback_translator.can_translate(
              typeid(Realm::FunctionPointerImplementation),
              typeid(Realm::DSOReferenceImplementation)));
          dso = static_cast<Realm::DSOReferenceImplementation*>(
              callback_translator.translate(
                  &impl, typeid(Realm::DSOReferenceImplementation)));
          if (dso == nullptr)
          {
            Fatal fatal;
            fatal
                << "Global registration callback function pointer " << raw_fnptr
                << " is not portable. All registration callbacks requesting to "
                << "be performed 'globally' must be able to be recognized by "
                << "a call to 'dladdr'. This requires that they come from a "
                << "shared object or the binary is linked with the '-rdynamic' "
                << "flag.";
            fatal.raise();
          }
          global_key = RegistrationKey(
              callback.dedup_tag, dso->dso_name, dso->symbol_name);
        }
        else
          global = false;
      }
#else
      if (global)
      {
        if (total_address_spaces > 1)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Global registration callbacks are not supported in "
                   "multi-node "
                << "executions without support for libdl. Please build Legion "
                << "with LEGION_USE_LIBDL defined.";
          error.raise();
        }
        else
          global = false;
      }
#endif
      RtEvent local_done, global_done;
      RtUserEvent local_perform, global_perform;
      if (callback.deduplicate)
      {
        AutoLock c_lock(callback_lock);
        if (global)
        {
          // See if we're going to perform this or not
          std::map<RegistrationKey, RtEvent>::const_iterator local_finder =
              global_local_done.find(global_key);
          if (local_finder == global_local_done.end())
          {
            local_perform = Runtime::create_rt_user_event();
            global_local_done[global_key] = local_perform;
            // Check to see if we have any pending global callbacks to
            // notify about being done locally
            std::map<RegistrationKey, std::set<RtUserEvent> >::iterator
                pending_finder = pending_remote_callbacks.find(global_key);
            if (pending_finder != pending_remote_callbacks.end())
            {
              for (const RtUserEvent& it : pending_finder->second)
                Runtime::trigger_event(it, local_perform);
              pending_remote_callbacks.erase(pending_finder);
            }
          }
          else
            local_done = local_finder->second;
          // Now see if we need to do our global registration callbacks
          std::map<RegistrationKey, RtEvent>::const_iterator global_finder =
              global_callbacks_done.find(global_key);
          if (global_finder == global_callbacks_done.end())
          {
            global_perform = Runtime::create_rt_user_event();
            global_callbacks_done[global_key] = global_perform;
          }
          else
            global_done = global_finder->second;
        }
        else
        {
          void* fnptr = nullptr;
          if (callback.has_args)
          {
            const RegistrationWithArgsCallbackFnptr* p =
                callback.withargs.target<RegistrationWithArgsCallbackFnptr>();
            if (p != nullptr)
              fnptr = reinterpret_cast<void*>(*p);
          }
          else
          {
            const RegistrationCallbackFnptr* p =
                callback.withoutargs.target<RegistrationCallbackFnptr>();
            if (p != nullptr)
              fnptr = reinterpret_cast<void*>(*p);
          }
          if (fnptr == nullptr)
          {
            Fatal fatal;
            fatal << "Deduplication of registration callbacks is only "
                     "permitted on "
                  << "callbacks that can be converted to function pointers to "
                     "facilitate "
                  << "identification of the same callback from different "
                     "sources.";
#ifdef __clang__
            fatal << " It looks like you compiled Legion with clang so if "
                  << "your callback is actually a function pointer it's likely "
                  << "that you are hitting this LLVM bug: "
                  << "https://github.com/llvm/llvm-project/issues/36746 "
                  << "Please double check before reporting any issues.";
#endif
            fatal.raise();
          }
          std::map<void*, RtEvent>::const_iterator local_finder =
              local_callbacks_done.find(fnptr);
          if (local_finder == local_callbacks_done.end())
          {
            local_perform = Runtime::create_rt_user_event();
            local_callbacks_done[fnptr] = local_perform;
          }
          else
            return local_finder->second;
        }
      }
      else if (global)
        global_perform = Runtime::create_rt_user_event();
      // Do the local callback and record it now
      if (!callback.deduplicate || local_perform.exists())
      {
        // All the pregistered cases are effectively global too
        if (global || preregistered)
          inside_registration_callback = GLOBAL_REGISTRATION_CALLBACK;
        else
          inside_registration_callback = LOCAL_REGISTRATION_CALLBACK;
        if (callback.has_args)
        {
          RegistrationCallbackArgs args{
              machine, external, local_procs, callback.buffer};
          callback.withargs(args);
        }
        else
          callback.withoutargs(machine, external, local_procs);
        inside_registration_callback = NO_REGISTRATION_CALLBACK;
        if (local_perform.exists())
          Runtime::trigger_event(local_perform);
        if (!global)
          return local_perform;
      }
#ifdef LEGION_USE_LIBDL
      legion_assert(global);
      if (global_done.exists())
      {
        delete dso;
        return global_done;
      }
      legion_assert(global_perform.exists());
      // See if we're inside of a task and can use that to help do the
      // global invocations of this registration callback
      if (!callback.deduplicate || (implicit_context == nullptr))
      {
        // This means we're in an external thread asking for us to
        // perform a global registration so just send out messages
        // to all the nodes asking them to do the registration
        std::set<RtEvent> preconditions;
        for (AddressSpaceID space = 0; space < total_address_spaces; space++)
        {
          if (space == address_space)
            continue;
          send_registration_callback(
              space, dso, global_perform, preconditions,
              callback.buffer.get_ptr(), callback.buffer.get_size(),
              callback.has_args, callback.deduplicate, callback.dedup_tag);
        }
        if (!preconditions.empty())
          Runtime::trigger_event(
              global_perform, Runtime::merge_events(preconditions));
        else
          Runtime::trigger_event(global_perform);
      }
      else
      {
        std::set<RtEvent> preconditions;
        implicit_context->perform_global_registration_callbacks(
            dso, callback.buffer.get_ptr(), callback.buffer.get_size(),
            callback.has_args, callback.dedup_tag, local_done, global_perform,
            preconditions);
        if (!preconditions.empty())
          Runtime::trigger_event(
              global_perform, Runtime::merge_events(preconditions));
        else
          Runtime::trigger_event(global_perform);
      }
      delete dso;
#endif  // LEGION_USE_LIBDL
      return global_perform;
    }

    //--------------------------------------------------------------------------
    void Runtime::finalize_runtime(
        std::vector<Realm::Event>& shutdown_preconditions)
    //--------------------------------------------------------------------------
    {
      // We send messages to the next address spaces up the
      // tree from us to do the shutdown
      Realm::ProfilingRequestSet empty_requests;
      AddressSpaceID start = address_space * legion_collective_radix + 1;
      for (int idx = 0; idx < legion_collective_radix; idx++)
      {
        AddressSpaceID next = start + idx;
        if (total_address_spaces <= next)
          break;
        MessageManager* messenger = find_messenger(next);
        shutdown_preconditions.emplace_back(messenger->target.spawn(
            LG_SHUTDOWN_TASK_ID, nullptr, 0, empty_requests));
      }
      // Have the memory managers for deletion of all their instances
      for (const std::pair<const Memory, MemoryManager*>& it : memory_managers)
        it.second->finalize();
      if (profiler != nullptr)
        profiler->finalize();
    }

    //--------------------------------------------------------------------------
    ApEvent Runtime::launch_mapper_task(
        Mapper* mapper, Processor proc, TaskID tid, const UntypedBuffer& arg,
        MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Get a remote task to serve as the top of the top-level task
      TopLevelContext* map_context = new TopLevelContext(
          proc, get_unique_top_level_task_id(), 0 /*implicit*/);
      map_context->add_base_gc_ref(RUNTIME_REF);
      TaskLauncher launcher(tid, arg, Predicate::TRUE_PRED, map_id);
      // Get an individual task to be the top-level task
      IndividualTask* mapper_task = get_operation<IndividualTask>();
      Future f = mapper_task->initialize_task(
          map_context, launcher, nullptr /*provenance*/, true /*top level*/);
      mapper_task->set_current_proc(proc);
      mapper_task->select_task_options(false /*prioritize*/);
      // Add a reference to the future impl to prevent it being collected
      f.impl->add_base_gc_ref(META_TASK_REF);
      // Create a meta-task to return the results to the mapper
      MapperTaskArgs args(f.impl, map_id, proc, map_context);
      ApEvent post(issue_runtime_meta_task(
          args, LG_LATENCY_WORK_PRIORITY, mapper_task->get_commit_event()));
      // Mark that we have another outstanding top level task
      increment_outstanding_top_level_tasks();
      // Now we can put it on the queue
      add_to_ready_queue(proc, mapper_task);
      return post;
    }

    //--------------------------------------------------------------------------
    void Runtime::MapperTaskArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      runtime->process_mapper_task_result(this);
      // Now indicate that we are done with the future
      if (future->remove_base_gc_ref(META_TASK_REF))
        delete future;
      // We can also deactivate the enclosing context
      if (ctx->remove_nested_gc_ref(RUNTIME_REF))
        delete ctx;
      // Finally tell the runtime we have one less top level task
      runtime->decrement_outstanding_top_level_tasks();
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_task_result(const MapperTaskArgs* args)
    //--------------------------------------------------------------------------
    {
#if 0
      MapperManager *mapper = find_mapper(args->proc, args->map_id);
      Mapper::MapperTaskResult result;
      result.mapper_event = args->event;
      result.result = args->future->get_untyped_result();
      result.result_size = args->future->get_untyped_size();
      mapper->invoke_handle_task_result(result);
#else
      std::abort();  // update this
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::create_shared_ownership(
        IndexSpace handle, const bool total_sharding_collective,
        const bool unpack_reference)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = runtime->get_node(handle);
      if (!node->check_valid_and_increment(APPLICATION_REF))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to add shared ownership to index space "
              << handle.get_id() << " which has already been deleted.";
        error.raise();
      }
      if (!node->is_owner())
      {
        legion_assert(!unpack_reference);
        if (!total_sharding_collective)
        {
          node->pack_valid_ref();
          SharedOwnershipMessage rez;
          {
            RezCheck z(rez);
            rez.serialize<int>(0);
            rez.serialize(handle);
          }
          rez.serialize(node->owner_space);
        }
        node->remove_base_valid_ref(APPLICATION_REF);
      }
      else if (unpack_reference)
        node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void Runtime::create_shared_ownership(
        IndexPartition handle, const bool total_sharding_collective,
        const bool unpack_reference)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(handle);
      if (!node->check_valid_and_increment(APPLICATION_REF))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to add shared ownership to index partition "
              << handle.get_id() << " which has already been deleted.";
        error.raise();
      }
      if (!node->is_owner())
      {
        legion_assert(!unpack_reference);
        if (!total_sharding_collective)
        {
          node->pack_valid_ref();
          SharedOwnershipMessage rez;
          {
            RezCheck z(rez);
            rez.serialize<int>(1);
            rez.serialize(handle);
          }
          rez.dispatch(node->owner_space);
        }
        node->remove_base_valid_ref(APPLICATION_REF);
      }
      else if (unpack_reference)
        node->unpack_valid_ref();
    }

    //--------------------------------------------------------------------------
    void Runtime::create_shared_ownership(
        FieldSpace handle, const bool total_sharding_collective,
        const bool unpack_reference)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = runtime->get_node(handle);
      if (!node->check_global_and_increment(APPLICATION_REF))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to add shared ownership to field space "
              << handle.get_id() << " which has already been deleted.";
        error.raise();
      }
      if (!node->is_owner())
      {
        legion_assert(!unpack_reference);
        if (!total_sharding_collective)
        {
          node->pack_global_ref();
          SharedOwnershipMessage rez;
          {
            RezCheck z(rez);
            rez.serialize<int>(2);
            rez.serialize(handle);
          }
          rez.dispatch(node->owner_space);
        }
        node->remove_base_gc_ref(APPLICATION_REF);
      }
      else if (unpack_reference)
        node->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    void Runtime::create_shared_ownership(
        LogicalRegion handle, const bool total_sharding_collective,
        const bool unpack_reference)
    //--------------------------------------------------------------------------
    {
      RegionNode* node = runtime->get_node(handle);
      if (!node->check_global_and_increment(APPLICATION_REF))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Illegal call to add shared ownership to logical region "
              << "(" << handle.index_space.get_id() << ","
              << handle.field_space.get_id() << "," << handle.get_tree_id()
              << ") which has already been deleted.";
        error.raise();
      }
      if (!node->is_owner())
      {
        legion_assert(!unpack_reference);
        if (!total_sharding_collective)
        {
          node->pack_global_ref();
          SharedOwnershipMessage rez;
          {
            RezCheck z(rez);
            rez.serialize<int>(3);
            rez.serialize(handle);
          }
          rez.dispatch(node->owner_space);
        }
        node->remove_base_gc_ref(APPLICATION_REF);
      }
      else if (unpack_reference)
        node->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    TraceID Runtime::generate_dynamic_trace_id(bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_trace_id();
      TraceID result = unique_trace_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Trace IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    TraceID Runtime::generate_library_trace_ids(const char* name, size_t count)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (count == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryTraceIDs>::const_iterator finder =
            library_trace_ids.find(library_name);
        if (finder != library_trace_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "TraceID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryTraceIDs>::const_iterator finder =
            library_trace_ids.find(library_name);
        if (finder != library_trace_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "TraceID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryTraceIDs& record = library_trace_ids[library_name];
          record.count = count;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_trace_id;
            unique_library_trace_id += count;
            legion_assert(unique_library_trace_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        TraceLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(count);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryTraceIDs>::const_iterator finder =
          library_trace_ids.find(library_name);
      legion_assert(finder != library_trace_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ TraceID& Runtime::get_current_static_trace_id(void)
    //--------------------------------------------------------------------------
    {
      static TraceID next_trace_id = LEGION_MAX_APPLICATION_TRACE_ID;
      return next_trace_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ TraceID Runtime::generate_static_trace_id(void)
    //--------------------------------------------------------------------------
    {
      TraceID& next_trace = get_current_static_trace_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_trace_id' after "
              << "the runtime has been started.";
        error.raise();
      }
      return next_trace++;
    }

    //--------------------------------------------------------------------------
    Mapper* Runtime::get_mapper(MapperID id, Processor target)
    //--------------------------------------------------------------------------
    {
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(target);
      if (finder == proc_managers.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid processor " << target
              << " passed to get mapper call.";
        error.raise();
      }
      return finder->second->find_mapper(id)->mapper;
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* Runtime::begin_mapper_call(
        MapperID id, Processor target, Operation* op)
    //--------------------------------------------------------------------------
    {
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(target);
      if (finder == proc_managers.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid processor " << target
              << " passed to begin mapper call.";
        error.raise();
      }
      MapperManager* manager = finder->second->find_mapper(id);
      return new MappingCallInfo(manager, APPLICATION_MAPPER_CALL, op);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_mapper_call(MappingCallInfo* info)
    //--------------------------------------------------------------------------
    {
      delete info;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_MPI_interop_configured(void)
    //--------------------------------------------------------------------------
    {
      return (mpi_rank_table != nullptr);
    }

    //--------------------------------------------------------------------------
    const std::map<int, AddressSpace>& Runtime::find_forward_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (mpi_rank_table == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Forward MPI mapping call not supported without "
              << "calling configure_MPI_interoperability during start up";
        error.raise();
      }
      legion_assert(!mpi_rank_table->forward_mapping.empty());
      return mpi_rank_table->forward_mapping;
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace, int>& Runtime::find_reverse_MPI_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (mpi_rank_table == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Reverse MPI mapping call not supported without "
              << "calling configure_MPI_interoperability during start up";
        error.raise();
      }
      legion_assert(!mpi_rank_table->reverse_mapping.empty());
      return mpi_rank_table->reverse_mapping;
    }

    //--------------------------------------------------------------------------
    int Runtime::find_local_MPI_rank(void)
    //-------------------------------------------------------------------------
    {
      if (mpi_rank_table == nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Findling local MPI rank not supported without "
              << "calling configure_MPI_interoperability during start up";
        error.raise();
      }
      return mpi_rank;
    }

    //--------------------------------------------------------------------------
    void Runtime::add_mapper(MapperID map_id, Mapper* mapper, Processor proc)
    //--------------------------------------------------------------------------
    {
      // If we have a custom mapper then silently ignore this
      if (!replay_file.empty() || enable_test_mapper)
      {
        // We take ownership of these things so delete it now
        delete mapper;
        return;
      }
      const bool all_local_procs = !proc.exists();
      if (all_local_procs)
      {
        std::vector<Processor> all_local_processors;
        all_local_processors.reserve(proc_managers.size());
        for (const std::pair<const Processor, ProcessorManager*>& it :
             proc_managers)
          all_local_processors.emplace_back(it.first);
        proc = find_processor_group(all_local_processors);
      }
      // First, wrap this mapper in a mapper manager
      MapperManager* manager = wrap_mapper(mapper, map_id, proc);
      if (all_local_procs)
      {
        // Save it to all the managers
        for (const std::pair<const Processor, ProcessorManager*>& it :
             proc_managers)
          it.second->add_mapper(map_id, manager, true /*check*/);
      }
      else if (proc.address_space() == address_space)
      {
        legion_assert(proc_managers.find(proc) != proc_managers.end());
        proc_managers[proc]->add_mapper(map_id, manager, true /*check*/);
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal attempt to register mapper "
              << mapper->get_mapper_name() << " as mapper " << map_id
              << " for processor " << proc
              << ". That processor is not local to the "
              << "process where 'Runtime::add_mapper' was called.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    Mapping::MapperRuntime* Runtime::get_mapper_runtime(void)
    //--------------------------------------------------------------------------
    {
      return mapper_runtime;
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_dynamic_mapper_id(bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_mapper_id();
      MapperID result = unique_mapper_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Mapper IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    MapperID Runtime::generate_library_mapper_ids(const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryMapperIDs>::const_iterator finder =
            library_mapper_ids.find(library_name);
        if (finder != library_mapper_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "MapperID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryMapperIDs>::const_iterator finder =
            library_mapper_ids.find(library_name);
        if (finder != library_mapper_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "MapperID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryMapperIDs& record = library_mapper_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_mapper_id;
            unique_library_mapper_id += cnt;
            legion_assert(unique_library_mapper_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        MapperLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryMapperIDs>::const_iterator finder =
          library_mapper_ids.find(library_name);
      legion_assert(finder != library_mapper_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID& Runtime::get_current_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      static MapperID current_mapper_id = LEGION_MAX_APPLICATION_MAPPER_ID;
      return current_mapper_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperID Runtime::generate_static_mapper_id(void)
    //--------------------------------------------------------------------------
    {
      MapperID& next_mapper = get_current_static_mapper_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_mapper_id' after "
                 "the runtime has been started!";
        error.raise();
      }
      return next_mapper++;
    }

    //--------------------------------------------------------------------------
    void Runtime::replace_default_mapper(Mapper* mapper, Processor proc)
    //--------------------------------------------------------------------------
    {
      // If we have a custom mapper then silently ignore this
      if (!replay_file.empty() || enable_test_mapper)
      {
        // We take ownership of mapper so delete it now
        delete mapper;
        return;
      }
      const bool all_local_procs = !proc.exists();
      if (all_local_procs)
      {
        std::vector<Processor> all_local_processors;
        all_local_processors.reserve(proc_managers.size());
        for (const std::pair<const Processor, ProcessorManager*>& it :
             proc_managers)
          all_local_processors.emplace_back(it.first);
        proc = find_processor_group(all_local_processors);
      }
      // First, wrap this mapper in a mapper manager
      MapperManager* manager = wrap_mapper(mapper, 0, proc);
      if (all_local_procs)
      {
        // Save it to all the managers
        for (const std::pair<const Processor, ProcessorManager*>& it :
             proc_managers)
          it.second->replace_default_mapper(manager);
      }
      else if (local_procs.find(proc) != local_procs.end())
      {
        legion_assert(proc_managers.find(proc) != proc_managers.end());
        proc_managers[proc]->replace_default_mapper(manager);
      }
      else
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal attempt to register mapper " << *manager
              << "as the default mapper for processor " << proc
              << ". That processor is not local to the "
              << "process where 'Runtime::replace_default_mapper' was called.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ MapperManager* Runtime::wrap_mapper(
        Mapper* mapper, MapperID map_id, Processor p, bool is_default)
    //--------------------------------------------------------------------------
    {
      MapperManager* manager = nullptr;
      switch (mapper->get_mapper_sync_model())
      {
        case Mapper::CONCURRENT_MAPPER_MODEL:
          {
            manager = new ConcurrentManager(mapper, map_id, p, is_default);
            break;
          }
        case Mapper::SERIALIZED_REENTRANT_MAPPER_MODEL:
          {
            manager = new SerializingManager(
                mapper, map_id, p, true /*reentrant*/, is_default);
            break;
          }
        case Mapper::SERIALIZED_NON_REENTRANT_MAPPER_MODEL:
          {
            manager = new SerializingManager(
                mapper, map_id, p, false /*reentrant*/, is_default);
            break;
          }
        default:
          std::abort();
      }
      return manager;
    }

    //--------------------------------------------------------------------------
    MapperManager* Runtime::find_mapper(MapperID map_id)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
      {
        MapperManager* result = it.second->find_mapper(map_id);
        if (result != nullptr)
          return result;
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------
    MapperManager* Runtime::find_mapper(Processor target, MapperID map_id)
    //--------------------------------------------------------------------------
    {
      legion_assert(target.exists());
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(target);
      legion_assert(finder != proc_managers.end());
      return finder->second->find_mapper(map_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_non_default_mapper(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
        if (it.second->has_non_default_mapper())
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    ProjectionID Runtime::generate_dynamic_projection_id(
        bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_projection_id();
      ProjectionID result = unique_projection_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Projection IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ProjectionID Runtime::generate_library_projection_ids(
        const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryProjectionIDs>::const_iterator finder =
            library_projection_ids.find(library_name);
        if (finder != library_projection_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ProjectionID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryProjectionIDs>::const_iterator finder =
            library_projection_ids.find(library_name);
        if (finder != library_projection_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ProjectionID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryProjectionIDs& record = library_projection_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_projection_id;
            unique_library_projection_id += cnt;
            legion_assert(unique_library_projection_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        ProjectionLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryProjectionIDs>::const_iterator finder =
          library_projection_ids.find(library_name);
      legion_assert(finder != library_projection_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID& Runtime::get_current_static_projection_id(void)
    //--------------------------------------------------------------------------
    {
      static ProjectionID current_projection_id =
          LEGION_MAX_APPLICATION_PROJECTION_ID;
      return current_projection_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionID Runtime::generate_static_projection_id(void)
    //--------------------------------------------------------------------------
    {
      ProjectionID& next_projection = get_current_static_projection_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_projection_id' after "
              << "the runtime has been started!";
        error.raise();
      }
      return next_projection++;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_projection_functor(
        ProjectionID pid, ProjectionFunctor* functor, bool need_zero_check,
        bool silence_warnings, const char* warning_string, bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (need_zero_check && (pid == 0))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ProjectionID zero is reserved.";
        error.raise();
      }
      if (!preregistered && !inside_registration_callback && !silence_warnings)
      {
        Warning warning;
        warning
            << "Projection functor " << pid
            << " was dynamically registered outside of a "
            << "registration callback invocation. In the near future this will "
            << "become an error in order to support task subprocesses. Please "
            << "use 'perform_registration_callback' to generate a callback "
               "where "
            << "it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      if (!silence_warnings && (total_address_spaces > 1) &&
          (inside_registration_callback != GLOBAL_REGISTRATION_CALLBACK))
      {
        Warning warning;
        warning << "Projection functor " << pid << " is being dynamically "
                << "registered for a multi-node run with "
                << total_address_spaces << " nodes. It is "
                << "currently the responsibility of the application to "
                << "ensure that this projection functor is registered on "
                << "all nodes where it will be required. "
                << "Warning string: "
                << (warning_string == nullptr ? "" : warning_string);
        warning.raise();
      }
      ProjectionFunction* function = new ProjectionFunction(pid, functor);
      AutoLock p_lock(projection_lock);
      std::map<ProjectionID, ProjectionFunction*>::const_iterator finder =
          projection_functions.find(pid);
      if (finder != projection_functions.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ProjectionID " << pid << " has already been used in "
              << "the region projection table.";
        error.raise();
      }
      projection_functions[pid] = function;
      LegionSpy::log_projection_function(
          pid, function->depth, function->is_invertible);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_projection_functor(
        ProjectionID pid, ProjectionFunctor* functor)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_projection_functor' after "
              << "the runtime has started!";
        error.raise();
      }
      if (pid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ProjectionID zero is reserved.";
        error.raise();
      }
      std::map<ProjectionID, ProjectionFunctor*>& pending_projection_functors =
          get_pending_projection_table();
      std::map<ProjectionID, ProjectionFunctor*>::const_iterator finder =
          pending_projection_functors.find(pid);
      if (finder != pending_projection_functors.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ProjectionID " << pid << " has already been used in "
              << "the region projection table.";
        error.raise();
      }
      pending_projection_functors[pid] = functor;
    }

    //--------------------------------------------------------------------------
    ProjectionFunction* Runtime::find_projection_function(
        ProjectionID pid, bool can_fail)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(projection_lock, false /*exclusive*/);
      std::map<ProjectionID, ProjectionFunction*>::const_iterator finder =
          projection_functions.find(pid);
      if (finder == projection_functions.end())
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find registered region projection ID " << pid
              << ". "
              << "Please upgrade to using projection functors!";
        error.raise();
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionFunctor* Runtime::get_projection_functor(
        ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        ProjectionFunction* func =
            runtime->find_projection_function(pid, true /*can fail*/);
        if (func != nullptr)
          return func->functor;
      }
      else
      {
        std::map<ProjectionID, ProjectionFunctor*>&
            pending_projection_functors = get_pending_projection_table();
        std::map<ProjectionID, ProjectionFunctor*>::const_iterator finder =
            pending_projection_functors.find(pid);
        if (finder != pending_projection_functors.end())
          return finder->second;
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------
    ShardingID Runtime::generate_dynamic_sharding_id(
        bool check_context /*true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_sharding_id();
      ShardingID result = unique_sharding_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Sharding IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ShardingID Runtime::generate_library_sharding_ids(
        const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryShardingIDs>::const_iterator finder =
            library_sharding_ids.find(library_name);
        if (finder != library_sharding_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ShardingID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryShardingIDs>::const_iterator finder =
            library_sharding_ids.find(library_name);
        if (finder != library_sharding_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ShardingID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryShardingIDs& record = library_sharding_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_sharding_id;
            unique_library_sharding_id += cnt;
            legion_assert(unique_library_sharding_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        ShardingLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryShardingIDs>::const_iterator finder =
          library_sharding_ids.find(library_name);
      legion_assert(finder != library_sharding_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ ShardingID& Runtime::get_current_static_sharding_id(void)
    //--------------------------------------------------------------------------
    {
      // + 2 since we use that for first one for the attach-detach functor
      // and the second one for the universal functor
      static ShardingID current_sharding_id =
          LEGION_MAX_APPLICATION_SHARDING_ID + 2;
      return current_sharding_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ShardingID Runtime::generate_static_sharding_id(void)
    //--------------------------------------------------------------------------
    {
      ShardingID& next_sharding = get_current_static_sharding_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_sharding_id' after "
              << "the runtime has been started!";
        error.raise();
      }
      return next_sharding++;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_sharding_functor(
        ShardingID sid, ShardingFunctor* functor, bool need_zero_check,
        bool silence_warnings, const char* warning_string, bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (sid == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: " << sid << " (UINT_MAX) is a reserved sharding ID.";
        error.raise();
      }
      else if (need_zero_check && (sid == 0))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: ShardingID zero is reserved.";
        error.raise();
      }
      if (!preregistered && !inside_registration_callback && !silence_warnings)
      {
        Warning warning;
        warning
            << "Sharding functor " << sid
            << " was dynamically registered outside of a "
            << "registration callback invocation. In the near future this will "
            << "become an error in order to support task subprocesses. Please "
            << "use 'perform_registration_callback' to generate a callback "
               "where "
            << "it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      if (!silence_warnings && (total_address_spaces > 1) &&
          (inside_registration_callback != GLOBAL_REGISTRATION_CALLBACK))
      {
        Warning warning;
        warning << "WARNING: Sharding functor " << sid
                << " is being dynamically "
                << "registered for a multi-node run with "
                << total_address_spaces << " nodes. It is "
                << "currently the responsibility of the application to "
                << "ensure that this sharding functor is registered on "
                << "all nodes where it will be required. "
                << "Warning string: "
                << (warning_string == nullptr ? "" : warning_string);
        warning.raise();
      }
      AutoLock s_lock(sharding_lock);
      std::map<ShardingID, ShardingFunctor*>::const_iterator finder =
          sharding_functors.find(sid);
      if (finder != sharding_functors.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: ShardingID " << sid
              << " has already been used by another "
              << "sharding functor.";
        error.raise();
      }
      sharding_functors[sid] = functor;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_sharding_functor(
        ShardingID sid, ShardingFunctor* functor)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_sharding_functor' after "
              << "the runtime has started!";
        error.raise();
      }
      if (sid == std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: " << sid << " (UINT_MAX) is a reserved sharding ID.";
        error.raise();
      }
      else if (sid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: ShardingID zero is reserved.";
        error.raise();
      }
      std::map<ShardingID, ShardingFunctor*>& pending_sharding_functors =
          get_pending_sharding_table();
      std::map<ShardID, ShardingFunctor*>::const_iterator finder =
          pending_sharding_functors.find(sid);
      if (finder != pending_sharding_functors.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ERROR: ShardingID " << sid
              << " has already been used by another "
              << "sharding functor.";
        error.raise();
      }
      pending_sharding_functors[sid] = functor;
    }

    //--------------------------------------------------------------------------
    ShardingFunctor* Runtime::find_sharding_functor(
        ShardingID sid, bool can_fail)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(sharding_lock, false /*exclusive*/);
      std::map<ShardingID, ShardingFunctor*>::const_iterator finder =
          sharding_functors.find(sid);
      if (finder == sharding_functors.end())
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find registered sharding functor ID " << sid << ".";
        error.raise();
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ ShardingFunctor* Runtime::get_sharding_functor(ShardingID sid)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started)
      {
        std::map<ShardingID, ShardingFunctor*>& pending_sharding_functors =
            get_pending_sharding_table();
        std::map<ShardID, ShardingFunctor*>::const_iterator finder =
            pending_sharding_functors.find(sid);
        if (finder == pending_sharding_functors.end())
          return nullptr;
        else
          return finder->second;
      }
      else
        return runtime->find_sharding_functor(sid, true /*can fail*/);
    }

    //--------------------------------------------------------------------------
    ConcurrentID Runtime::generate_dynamic_concurrent_id(
        bool check_context /*true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_concurrent_id();
      ConcurrentID result = unique_concurrent_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Concurrent Coloring IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ConcurrentID Runtime::generate_library_concurrent_ids(
        const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryConcurrentIDs>::const_iterator finder =
            library_concurrent_ids.find(library_name);
        if (finder != library_concurrent_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ConcurrentID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryConcurrentIDs>::const_iterator finder =
            library_concurrent_ids.find(library_name);
        if (finder != library_concurrent_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ConcurrentID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryConcurrentIDs& record = library_concurrent_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_concurrent_id;
            unique_library_concurrent_id += cnt;
            legion_assert(unique_library_concurrent_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        ConcurrentLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryConcurrentIDs>::const_iterator finder =
          library_concurrent_ids.find(library_name);
      legion_assert(finder != library_concurrent_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ ConcurrentID& Runtime::get_current_static_concurrent_id(void)
    //--------------------------------------------------------------------------
    {
      static ConcurrentID current_concurrent_id =
          LEGION_MAX_APPLICATION_CONCURRENT_ID;
      return current_concurrent_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ConcurrentID Runtime::generate_static_concurrent_id(void)
    //--------------------------------------------------------------------------
    {
      ConcurrentID& next_concurrent = get_current_static_concurrent_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_concurrent_id' after the "
                 "runtime has been started.";
        error.raise();
      }
      return next_concurrent++;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_concurrent_functor(
        ConcurrentID cid, ConcurrentColoringFunctor* functor,
        bool need_zero_check, bool silence_warnings, const char* warning_string,
        bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (cid == std::numeric_limits<ConcurrentID>::max())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ConcurrentID " << std::numeric_limits<ConcurrentID>::max()
              << " (UINT_MAX) is a reserved concurrent ID.";
        error.raise();
      }
      else if (need_zero_check && (cid == 0))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ConcurrentID zero is reserved.";
        error.raise();
      }
      if (!preregistered && !inside_registration_callback && !silence_warnings)
      {
        Warning warning;
        warning << "Concurrent coloring functor " << cid
                << " was dynamically registered outside of a registration "
                   "callback invocation. "
                << "In the near future this will become an error in order to "
                   "support task subprocesses. "
                << "Please use 'perform_registration_callback' to generate a "
                   "callback where "
                << "it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      if (!silence_warnings && (total_address_spaces > 1) &&
          (inside_registration_callback != GLOBAL_REGISTRATION_CALLBACK))
      {
        Warning warning;
        warning << "Concurrent coloring functor " << cid
                << " is being dynamically registered for a multi-node run with "
                << total_address_spaces
                << " nodes. It is currently the responsibility "
                << "of the application to ensure that this concurrent coloring "
                   "functor "
                << "is registered on all nodes where it will be required. "
                   "Warning string: "
                << ((warning_string == nullptr) ? "" : warning_string);
        warning.raise();
      }
      AutoLock s_lock(concurrent_lock);
      std::map<ConcurrentID, ConcurrentColoringFunctor*>::const_iterator
          finder = concurrent_functors.find(cid);
      if (finder != concurrent_functors.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "ConcurrentID " << cid
            << " has already been used by another concurrent coloring functor.";
        error.raise();
      }
      concurrent_functors[cid] = functor;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_concurrent_functor(
        ConcurrentID cid, ConcurrentColoringFunctor* functor)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_concurrent_coloring_functor' "
                 "after the runtime has started.";
        error.raise();
      }
      if (cid == std::numeric_limits<ConcurrentID>::max())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ConcurrentID " << std::numeric_limits<ConcurrentID>::max()
              << " (UINT_MAX) is a reserved concurrent ID.";
        error.raise();
      }
      else if (cid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ConcurrentID zero is reserved.";
        error.raise();
      }
      std::map<ConcurrentID, ConcurrentColoringFunctor*>&
          pending_concurrent_functors = get_pending_concurrent_table();
      std::map<ConcurrentID, ConcurrentColoringFunctor*>::const_iterator
          finder = pending_concurrent_functors.find(cid);
      if (finder != pending_concurrent_functors.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "ConcurrentID " << cid
            << " has already been used by another concurrent coloring functor.";
        error.raise();
      }
      pending_concurrent_functors[cid] = functor;
    }

    //--------------------------------------------------------------------------
    ConcurrentColoringFunctor* Runtime::find_concurrent_coloring_functor(
        ConcurrentID cid, bool can_fail)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(concurrent_lock, false /*exclusive*/);
      std::map<ConcurrentID, ConcurrentColoringFunctor*>::const_iterator
          finder = concurrent_functors.find(cid);
      if (finder == concurrent_functors.end())
      {
        if (can_fail)
          return nullptr;
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find registered concurrent coloring functor ID "
                << cid << ".";
          error.raise();
        }
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ ConcurrentColoringFunctor* Runtime::get_concurrent_functor(
        ConcurrentID cid)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started)
      {
        std::map<ConcurrentID, ConcurrentColoringFunctor*>&
            pending_concurrent_functors = get_pending_concurrent_table();
        std::map<ConcurrentID, ConcurrentColoringFunctor*>::const_iterator
            finder = pending_concurrent_functors.find(cid);
        if (finder == pending_concurrent_functors.end())
          return nullptr;
        else
          return finder->second;
      }
      else
        return runtime->find_concurrent_coloring_functor(
            cid, true /*can fail*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_projection_functor(ProjectionID pid)
    //--------------------------------------------------------------------------
    {
      legion_assert(runtime_started);
      AutoLock p_lock(projection_lock);
      std::map<ProjectionID, ProjectionFunction*>::iterator finder =
          projection_functions.find(pid);
      if (finder != projection_functions.end())
      {
        delete finder->second;
        projection_functions.erase(finder);
        return;
      }
      std::abort();
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID Runtime::generate_dynamic_exception_handler_id(
        bool check_context /*true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_exception_handler_id();
      ExceptionHandlerID result =
          unique_exception_handler_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Exception Handler IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ExceptionHandlerID Runtime::generate_library_exception_handler_ids(
        const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryExceptionIDs>::const_iterator finder =
            library_exception_ids.find(library_name);
        if (finder != library_exception_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ExceptionHandlerID generation counts "
                  << finder->second.count << " and " << cnt
                  << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryExceptionIDs>::const_iterator finder =
            library_exception_ids.find(library_name);
        if (finder != library_exception_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ExceptionHandlerID generation counts "
                  << finder->second.count << " and " << cnt
                  << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryExceptionIDs& record = library_exception_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_exception_handler_id;
            unique_library_exception_handler_id += cnt;
            legion_assert(unique_library_exception_handler_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        ExceptionLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryExceptionIDs>::const_iterator finder =
          library_exception_ids.find(library_name);
      legion_assert(finder != library_exception_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    /*static*/ ExceptionHandlerID&
        Runtime::get_current_static_exception_handler_id(void)
    //--------------------------------------------------------------------------
    {
      static ExceptionHandlerID current_exception_handler_id =
          LEGION_MAX_APPLICATION_EXCEPTION_HANDLER_ID;
      return current_exception_handler_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ExceptionHandlerID Runtime::generate_static_exception_handler_id(
        void)
    //--------------------------------------------------------------------------
    {
      ExceptionHandlerID& next_exception =
          get_current_static_exception_handler_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_exception_handler_id' after "
                 "the "
                 "runtime has been started.";
        error.raise();
      }
      return next_exception++;
    }

    //--------------------------------------------------------------------------
    void Runtime::register_exception_handler(
        ExceptionHandlerID hid, ExceptionHandler* handler, bool need_zero_check,
        bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (need_zero_check && (hid == 0))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ExceptionHandlerID 0 is a reserved ExceptionHandlerID that "
              << "is reserved for the null exception handler.";
        error.raise();
      }
      if (!preregistered && !inside_registration_callback)
      {
        Warning warning;
        warning << "ExceptionHandler " << hid
                << " was dynamically registered outside of a registration "
                   "callback invocation. "
                << "In the near future this will become an error in order to "
                   "support task subprocesses. "
                << "Please use 'perform_registration_callback' to generate a "
                   "callback where "
                << "it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      if (!preregistered && (total_address_spaces > 1) &&
          (inside_registration_callback != GLOBAL_REGISTRATION_CALLBACK))
      {
        Warning warning;
        warning << "ExceptionHandler " << hid
                << " is being dynamically registered for a multi-node run with "
                << total_address_spaces
                << " nodes. It is currently the responsibility "
                << "of the application to ensure that this exception handler "
                << "is registered on all nodes where it will be required.";
        warning.raise();
      }
      AutoLock s_lock(exception_handler_lock);
      std::map<ExceptionHandlerID, ExceptionHandler*>::const_iterator finder =
          exception_handlers.find(hid);
      if (finder != exception_handlers.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ExceptionHandlerID " << hid
              << " has already been used by another exception handler.";
        error.raise();
      }
      exception_handlers[hid] = handler;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::preregister_exception_handler(
        ExceptionHandlerID hid, ExceptionHandler* handler)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_exception_handler' "
                 "after the runtime has started.";
        error.raise();
      }
      if (hid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ExceptionHandlerID 0 is a reserved ExceptionHandlerID that "
              << "is reserved for the null exception handler.";
        error.raise();
      }
      std::map<ExceptionHandlerID, ExceptionHandler*>&
          pending_exception_handlers = get_pending_exception_handler_table();
      std::map<ExceptionHandlerID, ExceptionHandler*>::const_iterator finder =
          pending_exception_handlers.find(hid);
      if (finder != pending_exception_handlers.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ExceptionHandlerID " << hid
              << " has already been used by another exception handler.";
        error.raise();
      }
      pending_exception_handlers[hid] = handler;
    }

    //--------------------------------------------------------------------------
    ExceptionHandler* Runtime::find_exception_handler(
        ExceptionHandlerID hid, bool can_fail)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(exception_handler_lock, false /*exclusive*/);
      std::map<ExceptionHandlerID, ExceptionHandler*>::const_iterator finder =
          exception_handlers.find(hid);
      if (finder == exception_handlers.end())
      {
        if (can_fail)
          return nullptr;
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find registered excpetion handler ID " << hid
                << ".";
          error.raise();
        }
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ ExceptionHandler* Runtime::get_exception_handler(
        ExceptionHandlerID hid)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started)
      {
        std::map<ExceptionHandlerID, ExceptionHandler*>&
            pending_exception_handlers = get_pending_exception_handler_table();
        std::map<ExceptionHandlerID, ExceptionHandler*>::const_iterator finder =
            pending_exception_handlers.find(hid);
        if (finder == pending_exception_handlers.end())
          return nullptr;
        else
          return finder->second;
      }
      else
        return runtime->find_exception_handler(hid, true /*can fail*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        TaskID task_id, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_TASK_INFO, &task_id,
              sizeof(task_id), tag, buffer, size, is_mutable, send_to_owner))
        return;
      if (tag == LEGION_NAME_SEMANTIC_TAG)
        LegionSpy::log_task_name(task_id, static_cast<const char*>(buffer));
      TaskImpl* impl = find_or_create_task_impl(task_id);
      impl->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, send_to_owner);
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        IndexSpace handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_INDEX_SPACE_INFO,
              &handle, sizeof(handle), tag, buffer, size, is_mutable, global))
        return;
      get_node(handle)->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_index_space_name(handle.get_id(), name);
        if (implicit_profiler != nullptr)
          implicit_profiler->register_index_space(handle.get_id(), name);
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        IndexPartition handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_INDEX_PARTITION_INFO,
              &handle, sizeof(handle), tag, buffer, size, is_mutable, global))
        return;
      get_node(handle)->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_index_partition_name(handle.get_id(), name);
        if (implicit_profiler != nullptr)
          implicit_profiler->register_index_part(handle.get_id(), name);
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        FieldSpace handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_FIELD_SPACE_INFO,
              &handle, sizeof(handle), tag, buffer, size, is_mutable, global))
        return;
      get_node(handle)->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_field_space_name(handle.get_id(), name);
        if (implicit_profiler != nullptr)
          implicit_profiler->register_field_space(handle.get_id(), name);
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        FieldSpace handle, FieldID fid, SemanticTag tag, const void* buffer,
        size_t size, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_FIELD_INFO, &handle,
              sizeof(handle), tag, buffer, size, is_mutable, global, &fid,
              sizeof(fid)))
        return;
      get_node(handle)->attach_semantic_information(
          fid, tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_field_name(handle.get_id(), fid, name);
        if (implicit_profiler != nullptr)
        {
          FieldSpaceNode* node = get_node(handle);
          implicit_profiler->register_field(
              handle.get_id(), fid, node->get_field_size(fid), name);
        }
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        LogicalRegion handle, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__, ReplicateContext::REPLICATE_ATTACH_LOGICAL_REGION_INFO,
              &handle, sizeof(handle), tag, buffer, size, is_mutable, global))
        return;
      get_node(handle)->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_logical_region_name(
            handle.get_index_space().get_id(),
            handle.get_field_space().get_id(), handle.get_tree_id(), name);
        if (implicit_profiler != nullptr)
          implicit_profiler->register_logical_region(
              handle.get_index_space().get_id(),
              handle.get_field_space().get_id(), handle.get_tree_id(), name);
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    void Runtime::attach_semantic_information(
        LogicalPartition handle, SemanticTag tag, const void* buffer,
        size_t size, bool is_mutable)
    //--------------------------------------------------------------------------
    {
      bool global = true;
      if ((implicit_context != nullptr) &&
          !implicit_context->perform_semantic_attach(
              __func__,
              ReplicateContext::REPLICATE_ATTACH_LOGICAL_PARTITION_INFO,
              &handle, sizeof(handle), tag, buffer, size, is_mutable, global))
        return;
      get_node(handle)->attach_semantic_information(
          tag, address_space, buffer, size, is_mutable, !global);
      if (tag == LEGION_NAME_SEMANTIC_TAG)
      {
        const char* name = static_cast<const char*>(buffer);
        LegionSpy::log_logical_partition_name(
            handle.get_index_partition().get_id(),
            handle.get_field_space().get_id(), handle.get_tree_id(), name);
      }
      if (implicit_context != nullptr)
        implicit_context->post_semantic_attach();
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        TaskID task_id, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      TaskImpl* impl = find_or_create_task_impl(task_id);
      return impl->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        IndexSpace handle, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle, nullptr, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        IndexPartition handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = get_node(handle, nullptr, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        FieldSpace handle, SemanticTag tag, const void*& result, size_t& size,
        bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle, nullptr, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        FieldSpace handle, FieldID fid, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle, nullptr, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          fid, tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        LogicalRegion handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      RegionNode* node = get_node(handle, true /*need check*/, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    bool Runtime::retrieve_semantic_information(
        LogicalPartition handle, SemanticTag tag, const void*& result,
        size_t& size, bool can_fail, bool wait_until)
    //--------------------------------------------------------------------------
    {
      PartitionNode* node = get_node(handle, true /*need check*/, can_fail);
      if (node == nullptr)
        return false;
      return node->retrieve_semantic_information(
          tag, result, size, can_fail, wait_until);
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_dynamic_task_id(bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_task_id();
      TaskID result = unique_task_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Task IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    TaskID Runtime::generate_library_task_ids(const char* name, size_t cnt)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (cnt == 0)
        return LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryTaskIDs>::const_iterator finder =
            library_task_ids.find(library_name);
        if (finder != library_task_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "TaskID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryTaskIDs>::const_iterator finder =
            library_task_ids.find(library_name);
        if (finder != library_task_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != cnt)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "TaskID generation counts " << finder->second.count
                  << " and " << cnt << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryTaskIDs& record = library_task_ids[library_name];
          record.count = cnt;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_task_id;
            unique_library_task_id += cnt;
            legion_assert(unique_library_task_id > record.result);
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        TaskLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(cnt);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryTaskIDs>::const_iterator finder =
          library_task_ids.find(library_name);
      legion_assert(finder != library_task_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    VariantID Runtime::register_variant(
        const TaskVariantRegistrar& registrar, const void* user_data,
        size_t user_data_size, const CodeDescriptor& realm_code_desc,
        size_t return_type_size, bool has_return_size,
        VariantID vid /*= AUTO_GENERATE_ID*/, bool check_task_id /*= true*/,
        bool check_context /*= true*/, bool preregistered /*= false*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->register_variant(
            registrar, user_data, user_data_size, realm_code_desc,
            return_type_size, has_return_size, vid, check_task_id);
      // TODO: figure out a way to make this check safe with dynamic
      // generation
#if 0
      if (check_task_id && 
          (registrar.task_id >= LEGION_MAX_APPLICATION_TASK_ID))
        REPORT_LEGION_ERROR(ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED, 
                      "Error registering task with ID %d. Exceeds the "
                      "statically set bounds on application task IDs of %d. "
                      "See %s in legion_config.h.", 
                      registrar.task_id, LEGION_MAX_APPLICATION_TASK_ID, 
                      LEGION_MACRO_TO_STRING(LEGION_MAX_APPLICATION_TASK_ID))
#endif
      // First find the task implementation
      TaskImpl* task_impl = find_or_create_task_impl(registrar.task_id);
      // See if we need to make a new variant ID
      if (vid == LEGION_AUTO_GENERATE_ID)  // Make a variant ID to use
        vid = task_impl->get_unique_variant_id();
      else if (vid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Variant ID 0 is reserved for task generators when "
                 "registering variant for task ID "
              << registrar.task_id << ".";
        error.raise();
      }
      // Make our variant and add it to the set of variants
      VariantImpl* impl = new VariantImpl(
          vid, task_impl, registrar, return_type_size, has_return_size,
          realm_code_desc, user_data, user_data_size);
      // Add this variant to the owner
      task_impl->add_variant(impl);
      {
        AutoLock tv_lock(task_variant_lock);
        variant_table.emplace_back(impl);
      }
      // If this is a global registration we need to broadcast the variant
      if (registrar.global_registration && (total_address_spaces > 1))
      {
        RtUserEvent done_event = Runtime::create_rt_user_event();
        impl->broadcast_variant(done_event, address_space, 0);
        done_event.wait();
      }
      LegionSpy::log_task_variant(
          registrar.task_id, vid, impl->is_inner(), impl->is_leaf(),
          impl->is_idempotent(), impl->get_name());
      return vid;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Runtime::find_or_create_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock tv_lock(task_variant_lock, false /*exclusive*/);
        std::map<TaskID, TaskImpl*>::const_iterator finder =
            task_table.find(task_id);
        if (finder != task_table.end())
          return finder->second;
      }
      AutoLock tv_lock(task_variant_lock);
      std::map<TaskID, TaskImpl*>::const_iterator finder =
          task_table.find(task_id);
      // Check to see if we lost the race
      if (finder == task_table.end())
      {
        TaskImpl* result = new TaskImpl(task_id);
        task_table[task_id] = result;
        return result;
      }
      else  // Lost the race as it already exists
        return finder->second;
    }

    //--------------------------------------------------------------------------
    TaskImpl* Runtime::find_task_impl(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      AutoLock tv_lock(task_variant_lock, false /*exclusive*/);
      std::map<TaskID, TaskImpl*>::const_iterator finder =
          task_table.find(task_id);
      legion_assert(finder != task_table.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    VariantImpl* Runtime::find_variant_impl(
        TaskID task_id, VariantID variant_id, bool can_fail)
    //--------------------------------------------------------------------------
    {
      TaskImpl* owner = find_or_create_task_impl(task_id);
      return owner->find_variant_impl(variant_id, can_fail);
    }

    //--------------------------------------------------------------------------
    ReductionOpID Runtime::generate_dynamic_reduction_id(
        bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_reduction_id();
      ReductionOpID result = unique_redop_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Reduction IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ReductionOpID Runtime::generate_library_reduction_ids(
        const char* name, size_t count)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (count == 0)
        return (ReductionOpID)LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibraryRedopIDs>::const_iterator finder =
            library_redop_ids.find(library_name);
        if (finder != library_redop_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ReductionOpID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibraryRedopIDs>::const_iterator finder =
            library_redop_ids.find(library_name);
        if (finder != library_redop_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "ReductionOpID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibraryRedopIDs& record = library_redop_ids[library_name];
          record.count = count;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_redop_id;
            unique_library_redop_id += count;
            legion_assert(unique_library_redop_id > unsigned(record.result));
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        RedopLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(count);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibraryRedopIDs>::const_iterator finder =
          library_redop_ids.find(library_name);
      legion_assert(finder != library_redop_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    CustomSerdezID Runtime::generate_dynamic_serdez_id(
        bool check_context /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (check_context && (implicit_context != nullptr))
        return implicit_context->generate_dynamic_serdez_id();
      CustomSerdezID result = unique_serdez_id.fetch_add(runtime_stride);
      // Check for hitting the library limit
      if (result >= LEGION_INITIAL_LIBRARY_ID_OFFSET)
      {
        Fatal fatal;
        fatal << "Dynamic Custom Serdez IDs exceeded library ID offset "
              << LEGION_INITIAL_LIBRARY_ID_OFFSET << ".";
        fatal.raise();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    CustomSerdezID Runtime::generate_library_serdez_ids(
        const char* name, size_t count)
    //--------------------------------------------------------------------------
    {
      // Easy case if the user asks for no IDs
      if (count == 0)
        return (CustomSerdezID)LEGION_AUTO_GENERATE_ID;
      const std::string library_name(name);
      // Take the lock in read only mode and see if we can find the result
      RtEvent wait_on;
      {
        AutoLock l_lock(library_lock, false /*exclusive*/);
        std::map<std::string, LibrarySerdezIDs>::const_iterator finder =
            library_serdez_ids.find(library_name);
        if (finder != library_serdez_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "CustomSerdezID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
      }
      RtUserEvent request_event;
      if (!wait_on.exists())
      {
        AutoLock l_lock(library_lock);
        // Check to make sure we didn't lose the race
        std::map<std::string, LibrarySerdezIDs>::const_iterator finder =
            library_serdez_ids.find(library_name);
        if (finder != library_serdez_ids.end())
        {
          // First do a check to see if the counts match
          if (finder->second.count != count)
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "CustomSerdezID generation counts " << finder->second.count
                  << " and " << count << " differ for library " << name << ".";
            error.raise();
          }
          if (finder->second.result_set)
            return finder->second.result;
          // This should never happen unless we are on a node other than 0
          legion_assert(address_space > 0);
          wait_on = finder->second.ready;
        }
        if (!wait_on.exists())
        {
          LibrarySerdezIDs& record = library_serdez_ids[library_name];
          record.count = count;
          if (address_space == 0)
          {
            // We're going to make the result
            record.result = unique_library_serdez_id;
            unique_library_serdez_id += count;
            legion_assert(unique_library_serdez_id > unsigned(record.result));
            record.result_set = true;
            return record.result;
          }
          else
          {
            // We're going to request the result
            request_event = Runtime::create_rt_user_event();
            record.ready = request_event;
            record.result_set = false;
            wait_on = request_event;
          }
        }
      }
      // Should only get here on nodes other than 0
      legion_assert(address_space > 0);
      legion_assert(wait_on.exists());
      if (request_event.exists())
      {
        // Include the null terminator in length
        const size_t string_length = strlen(name) + 1;
        // Send the request to node 0 for the result
        SerdezLibraryRequest rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(string_length);
          rez.serialize(name, string_length);
          rez.serialize<size_t>(count);
          rez.serialize(request_event);
        }
        rez.dispatch(0 /*target*/);
      }
      wait_on.wait();
      // When we wake up we should be able to find the result
      AutoLock l_lock(library_lock, false /*exclusive*/);
      std::map<std::string, LibrarySerdezIDs>::const_iterator finder =
          library_serdez_ids.find(library_name);
      legion_assert(finder != library_serdez_ids.end());
      legion_assert(finder->second.result_set);
      return finder->second.result;
    }

    //--------------------------------------------------------------------------
    MemoryManager* Runtime::find_memory_manager(Memory mem)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock m_lock(memory_manager_lock, false /*exclusive*/);
        std::map<Memory, MemoryManager*>::const_iterator finder =
            memory_managers.find(mem);
        if (finder != memory_managers.end())
          return finder->second;
      }
      // Not there?  Take exclusive lock and check again, create if needed
      AutoLock m_lock(memory_manager_lock);
      std::map<Memory, MemoryManager*>::const_iterator finder =
          memory_managers.find(mem);
      if (finder != memory_managers.end())
        return finder->second;
      // Really do need to create it (and put it in the map)
      MemoryManager* result = new MemoryManager(mem);
      memory_managers[mem] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    MessageManager* Runtime::find_messenger(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      legion_assert(sid < LEGION_MAX_NUM_NODES);
      legion_assert(
          sid != address_space);  // shouldn't be sending messages to ourself
      MessageManager* result = message_managers[sid].load();
      if (result != nullptr)
        return result;
      // If we made it here, then we don't have a message manager yet
      // re-take the lock and re-check to see if we don't have a manager
      // If we still don't then we need to make one
      RtEvent wait_on;
      bool send_request = false;
      {
        AutoLock m_lock(message_manager_lock);
        // Re-check to see if we lost the race, force the compiler
        // to re-load the value here
        result = message_managers[sid].load();
        if (result != nullptr)
          return result;
        // Figure out if there is an event to wait on yet
        std::map<AddressSpace, RtUserEvent>::const_iterator finder =
            pending_endpoint_requests.find(sid);
        if (finder == pending_endpoint_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          pending_endpoint_requests[sid] = done;
          wait_on = done;
          send_request = true;
        }
        else
          wait_on = finder->second;
      }
      if (send_request)
      {
        [[maybe_unused]] bool found = false;
        // Find a processor on which to send the task
        // TODO: refine search once Realm supports queries on specifc spaces
        Machine::ProcessorQuery all_procs(machine);
        for (Machine::ProcessorQuery::iterator it = all_procs.begin();
             it != all_procs.end(); it++)
        {
          if (it->address_space() != sid)
            continue;
          found = true;
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize<bool>(true);  // request
            rez.serialize(utility_group);
          }
          const Realm::ProfilingRequestSet empty_requests;
          it->spawn(
              LG_ENDPOINT_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
              empty_requests);
          break;
        }
        legion_assert(found);
      }
      legion_assert(wait_on.exists());
      if (!wait_on.has_triggered())
        wait_on.wait();
      // When we wake up there should be a result
      result = message_managers[sid].load();
      legion_assert(result != nullptr);
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_endpoint_creation(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      bool request;
      derez.deserialize(request);
      Processor remote_utility_group;
      derez.deserialize(remote_utility_group);
      if (request)
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize<bool>(false /*request*/);
          rez.serialize(utility_group);
          rez.serialize(address_space);
        }
        const Realm::ProfilingRequestSet empty_requests;
        remote_utility_group.spawn(
            LG_ENDPOINT_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
            empty_requests);
      }
      else
      {
        AddressSpaceID remote_space;
        derez.deserialize(remote_space);
        AutoLock m_lock(message_manager_lock);
        message_managers[remote_space].store(new MessageManager(
            remote_space, max_message_size, remote_utility_group));
        std::map<AddressSpaceID, RtUserEvent>::iterator finder =
            pending_endpoint_requests.find(remote_space);
        legion_assert(finder != pending_endpoint_requests.end());
        Runtime::trigger_event(finder->second);
        pending_endpoint_requests.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_message(
        Processor target, MapperID map_id, Processor source,
        const void* message, size_t message_size, unsigned message_kind)
    //--------------------------------------------------------------------------
    {
      if (is_local(target))
      {
        Mapper::MapperMessage message_args;
        message_args.sender = source;
        message_args.kind = message_kind;
        message_args.message = message;
        message_args.size = message_size;
        message_args.broadcast = false;
        MapperManager* mapper = find_mapper(target, map_id);
        mapper->invoke_handle_message(&message_args);
      }
      else
      {
        MapperMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(message_kind);
          rez.serialize(message_size);
          rez.serialize(message, message_size);
        }
        rez.dispatch(target.address_space());
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::process_mapper_broadcast(
        MapperID map_id, Processor source, const void* message,
        size_t message_size, unsigned message_kind, int radix, int index)
    //--------------------------------------------------------------------------
    {
      // First forward the message onto any remote nodes
      int base = index * radix;
      int init = source.address_space();
      // The runtime stride is the same as the number of nodes
      const int total_nodes = total_address_spaces;
      for (int r = 1; r <= radix; r++)
      {
        int offset = base + r;
        // If we've handled all of our nodes then we are done
        if (offset >= total_nodes)
          break;
        AddressSpaceID target = (init + offset) % total_nodes;
        MapperBroadcast rez;
        {
          RezCheck z(rez);
          rez.serialize(map_id);
          rez.serialize(source);
          rez.serialize(message_kind);
          rez.serialize(radix);
          rez.serialize(offset);
          rez.serialize(message_size);
          rez.serialize(message, message_size);
        }
        rez.dispatch(target);
      }
      // Then send it to all our local mappers, set will deduplicate
      std::set<MapperManager*> managers;
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
        managers.insert(it.second->find_mapper(map_id));
      Mapper::MapperMessage message_args;
      message_args.sender = source;
      message_args.kind = message_kind;
      message_args.message = message;
      message_args.size = message_size;
      message_args.broadcast = true;
      for (MapperManager* manager : managers)
        manager->invoke_handle_message(&message_args);
    }

    //--------------------------------------------------------------------------
    void Runtime::send_task(IndividualTask* task)
    //--------------------------------------------------------------------------
    {
      Processor target = task->target_proc;
      if (!target.exists())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper requested invalid NO_PROC as target processor.";
        error.raise();
      }
      // Check to see if the target processor is still local
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(target);
      if (finder != proc_managers.end())
      {
        // Update the current processor
        task->set_current_proc(target);
        finder->second->add_to_ready_queue(task);
      }
      else
      {
        TaskMessage rez;
        bool deactivate_task;
        const AddressSpaceID target_addr = target.address_space();
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(task->get_task_kind());
          deactivate_task = task->pack_task(rez, target_addr);
        }
        rez.dispatch(target_addr);
        if (deactivate_task)
          task->deactivate();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_task(SliceTask* task)
    //--------------------------------------------------------------------------
    {
      const Processor target = task->target_proc;
      legion_assert(!is_local(target));
      TaskMessage rez;
      bool deactivate_task;
      const AddressSpaceID target_addr = target.address_space();
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(task->get_task_kind());
        deactivate_task = task->pack_task(rez, target_addr);
      }
      rez.dispatch(target_addr);
      if (deactivate_task)
        task->deactivate();
    }

    //--------------------------------------------------------------------------
    void Runtime::send_tasks(Processor target, std::vector<SingleTask*>& tasks)
    //--------------------------------------------------------------------------
    {
      legion_assert(!tasks.empty());
      legion_assert(target.exists());
      // Check to see if the target processor is still local
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(target);
      if (finder != proc_managers.end())
      {
        // Still local
        for (SingleTask* const & task : tasks)
        {
          // Update the current processor
          task->set_current_proc(target);
          finder->second->add_to_ready_queue(task);
        }
      }
      else
      {
        std::sort(tasks.begin(), tasks.end());
        // Send each of these tasks, if some of the tasks share the same
        // slice they might end up getting sent together
        while (!tasks.empty())
        {
          SingleTask* task = tasks.back();
          tasks.pop_back();
          if (task->send_task(target, tasks))
            task->deactivate();
        }
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_steal_request(
        const std::multimap<Processor, MapperID>& targets, Processor thief)
    //--------------------------------------------------------------------------
    {
      for (std::multimap<Processor, MapperID>::const_iterator it =
               targets.begin();
           it != targets.end(); it++)
      {
        Processor target = it->first;
        std::map<Processor, ProcessorManager*>::const_iterator finder =
            proc_managers.find(target);
        if (finder == proc_managers.end())
        {
          // Need to send remotely
          StealTaskMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(target);
            rez.serialize(thief);
            int num_mappers = targets.count(target);
            rez.serialize(num_mappers);
            for (; it != targets.upper_bound(target); it++)
              rez.serialize(it->second);
          }
          rez.dispatch(target.address_space());
        }
        else
        {
          // Still local, so notify the processor manager
          std::vector<MapperID> thieves;
          for (; it != targets.upper_bound(target); it++)
            thieves.emplace_back(it->second);
          finder->second->process_steal_request(thief, thieves);
        }
        if (it == targets.end())
          break;
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::send_advertisements(
        const std::set<Processor>& targets, MapperID map_id, Processor source)
    //--------------------------------------------------------------------------
    {
      std::set<AddressSpaceID> already_sent;
      for (const Processor& it : targets)
      {
        std::map<Processor, ProcessorManager*>::const_iterator finder =
            proc_managers.find(it);
        if (finder != proc_managers.end())
        {
          // still local
          finder->second->process_advertisement(source, map_id);
        }
        else
        {
          // otherwise remote, check to see if we already sent it
          const AddressSpaceID target = it.address_space();
          if (already_sent.find(target) != already_sent.end())
            continue;
          AdvertiseTaskMessage rez;
          {
            RezCheck z(rez);
            rez.serialize(source);
            rez.serialize(map_id);
          }
          rez.dispatch(target);
          already_sent.insert(target);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperMessage::handle(Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor target;
      derez.deserialize(target);
      MapperID map_id;
      derez.deserialize(map_id);
      Processor source;
      derez.deserialize(source);
      unsigned message_kind;
      derez.deserialize(message_kind);
      size_t message_size;
      derez.deserialize(message_size);
      const void* message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      runtime->process_mapper_message(
          target, map_id, source, message, message_size, message_kind);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperBroadcast::handle(Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      MapperID map_id;
      derez.deserialize(map_id);
      Processor source;
      derez.deserialize(source);
      unsigned message_kind;
      derez.deserialize(message_kind);
      int radix;
      derez.deserialize(radix);
      int index;
      derez.deserialize(index);
      size_t message_size;
      derez.deserialize(message_size);
      const void* message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      runtime->process_mapper_broadcast(
          map_id, source, message, message_size, message_kind, radix, index);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_steal(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor target;
      derez.deserialize(target);
      Processor thief;
      derez.deserialize(thief);
      int num_mappers;
      derez.deserialize(num_mappers);
      std::vector<MapperID> thieves(num_mappers);
      for (int idx = 0; idx < num_mappers; idx++)
        derez.deserialize(thieves[idx]);
      legion_assert(proc_managers.find(target) != proc_managers.end());
      proc_managers[target]->process_steal_request(thief, thieves);
    }

    //--------------------------------------------------------------------------
    /*static*/ void StealTaskMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_steal(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_advertisement(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      Processor source;
      derez.deserialize(source);
      MapperID map_id;
      derez.deserialize(map_id);
      // Just advertise it to all the managers
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
        it.second->process_advertisement(source, map_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void AdvertiseTaskMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_advertisement(derez);
    }

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void Runtime::handle_registration_callback(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_context == nullptr);
      legion_assert(runtime != nullptr);
      DerezCheck z(derez);
      size_t dso_size;
      derez.deserialize(dso_size);
      const std::string dso_name((const char*)derez.get_current_pointer());
      derez.advance_pointer(dso_size);
      size_t sym_size;
      derez.deserialize(sym_size);
      const std::string sym_name((const char*)derez.get_current_pointer());
      derez.advance_pointer(sym_size);
      size_t buffer_size;
      derez.deserialize(buffer_size);
      const void* buffer = derez.get_current_pointer();
      if (buffer_size > 0)
        derez.advance_pointer(buffer_size);
      bool withargs;
      derez.deserialize<bool>(withargs);
      bool deduplicate;
      derez.deserialize(deduplicate);
      size_t dedup_tag;
      derez.deserialize(dedup_tag);
      RtEvent global_done_event;
      derez.deserialize(global_done_event);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      // Converting the DSO reference could call dlopen and might block
      // us if the constructor for that shared object requests its own
      // global registration callback, so register our guards first
      const RegistrationKey key(dedup_tag, dso_name, sym_name);
      if (deduplicate)
      {
        AutoLock c_lock(callback_lock);
        // First see if the local case has already been done in which case
        // we know that we are done also when it is done
        std::map<RegistrationKey, RtEvent>::const_iterator finder =
            global_local_done.find(key);
        if (finder != global_local_done.end())
        {
          Runtime::trigger_event(done_event, finder->second);
          return;
        }
        // No one has attempted a global registration callback here yet
        // Record that we are pending and put in a guard for all the
        // of the global registrations being done
        if (global_callbacks_done.find(key) == global_callbacks_done.end())
          global_callbacks_done[key] = global_done_event;
        pending_remote_callbacks[key].insert(done_event);
      }

      // Now we can do the translation of ourselves to get the function pointer
      Realm::DSOReferenceImplementation dso(dso_name, sym_name);
      legion_assert(callback_translator.can_translate(
          typeid(Realm::DSOReferenceImplementation),
          typeid(Realm::FunctionPointerImplementation)));
      Realm::FunctionPointerImplementation* impl =
          static_cast<Realm::FunctionPointerImplementation*>(
              callback_translator.translate(
                  &dso, typeid(Realm::FunctionPointerImplementation)));
      legion_assert(impl != nullptr);
      void* callback = impl->get_impl<void*>();
      RtEvent precondition;
      // Now take the lock and see if we need to perform anything
      if (deduplicate)
      {
        AutoLock c_lock(callback_lock);
        std::map<RegistrationKey, std::set<RtUserEvent> >::iterator finder =
            pending_remote_callbacks.find(key);
        // If someone already handled everything then we are done
        if (finder != pending_remote_callbacks.end())
        {
          // We should still be in there
          legion_assert(
              finder->second.find(done_event) != finder->second.end());
          finder->second.erase(done_event);
          if (finder->second.empty())
            pending_remote_callbacks.erase(finder);
          // Now see if anyone else has done the local registration
          std::map<void*, RtEvent>::const_iterator finder =
              local_callbacks_done.find(callback);
          if (finder != local_callbacks_done.end())
          {
            legion_assert(finder->second.exists());
            precondition = finder->second;
          }
          else
          {
            local_callbacks_done[callback] = done_event;
            global_local_done[key] = done_event;
          }
        }
        else  // We were already handled so nothing to do
          done_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (!deduplicate || done_event.exists())
      {
        // This is the signal that we need to do the callback
        if (!precondition.exists())
        {
          inside_registration_callback = GLOBAL_REGISTRATION_CALLBACK;
          if (withargs)
          {
            RegistrationWithArgsCallbackFnptr callbackwithargs =
                (RegistrationWithArgsCallbackFnptr)callback;
            RegistrationCallbackArgs args{
                machine, external, local_procs,
                UntypedBuffer(buffer, buffer_size)};
            (*callbackwithargs)(args);
          }
          else
          {
            RegistrationCallbackFnptr callbackwithoutargs =
                (RegistrationCallbackFnptr)callback;
            (*callbackwithoutargs)(machine, external, local_procs);
          }
          inside_registration_callback = NO_REGISTRATION_CALLBACK;
        }
        if (done_event.exists())
          Runtime::trigger_event(done_event, precondition);
      }
      // Delete our resources that we allocated
      delete impl;
    }
#endif  // LEGION_USE_LIBDL

    //--------------------------------------------------------------------------
    /*static*/ void RegistrationCallbackMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_USE_LIBDL
      runtime->handle_registration_callback(derez);
#else
      std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConstraintRelease::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      runtime->release_layout(layout_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_mapper_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      MapperID result = generate_library_mapper_ids(name, count);
      MapperLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_mapper_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_mapper_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      MapperID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryMapperIDs>::iterator finder =
            library_mapper_ids.find(library_name);
        legion_assert(finder != library_mapper_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_mapper_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_trace_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      TraceID result = generate_library_trace_ids(name, count);
      TraceLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TraceLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_trace_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_trace_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      TraceID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryTraceIDs>::iterator finder =
            library_trace_ids.find(library_name);
        legion_assert(finder != library_trace_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TraceLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_trace_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_projection_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      ProjectionID result = generate_library_projection_ids(name, count);
      ProjectionLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProjectionLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_projection_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_projection_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      ProjectionID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryProjectionIDs>::iterator finder =
            library_projection_ids.find(library_name);
        legion_assert(finder != library_projection_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProjectionLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_projection_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_sharding_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      ShardingID result = generate_library_sharding_ids(name, count);
      ShardingLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardingLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_sharding_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_sharding_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      ShardingID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryShardingIDs>::iterator finder =
            library_sharding_ids.find(library_name);
        legion_assert(finder != library_sharding_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardingLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_sharding_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_concurrent_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      ConcurrentID result = generate_library_concurrent_ids(name, count);
      ConcurrentLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConcurrentLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_concurrent_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_concurrent_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      ConcurrentID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryConcurrentIDs>::iterator finder =
            library_concurrent_ids.find(library_name);
        legion_assert(finder != library_concurrent_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConcurrentLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_concurrent_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_exception_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      ExceptionHandlerID result =
          generate_library_exception_handler_ids(name, count);
      ExceptionLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExceptionLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_exception_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_exception_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      ExceptionHandlerID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryExceptionIDs>::iterator finder =
            library_exception_ids.find(library_name);
        legion_assert(finder != library_exception_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExceptionLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_exception_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_task_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      TaskID result = generate_library_task_ids(name, count);
      TaskLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_task_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_task_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      TaskID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryTaskIDs>::iterator finder =
            library_task_ids.find(library_name);
        legion_assert(finder != library_task_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_task_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_redop_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      ReductionOpID result = generate_library_reduction_ids(name, count);
      RedopLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RedopLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_redop_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_redop_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      ReductionOpID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibraryRedopIDs>::iterator finder =
            library_redop_ids.find(library_name);
        legion_assert(finder != library_redop_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RedopLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_redop_response(derez);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_serdez_request(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      size_t count;
      derez.deserialize(count);
      RtUserEvent done;
      derez.deserialize(done);

      CustomSerdezID result = generate_library_serdez_ids(name, count);
      SerdezLibraryResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(string_length);
        rez.serialize(name, string_length);
        rez.serialize(result);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SerdezLibraryRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_serdez_request(derez, source);
    }

    //--------------------------------------------------------------------------
    void Runtime::handle_library_serdez_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t string_length;
      derez.deserialize(string_length);
      const char* name = (const char*)derez.get_current_pointer();
      derez.advance_pointer(string_length);
      CustomSerdezID result;
      derez.deserialize(result);
      RtUserEvent done;
      derez.deserialize(done);

      const std::string library_name(name);
      {
        AutoLock l_lock(library_lock);
        std::map<std::string, LibrarySerdezIDs>::iterator finder =
            library_serdez_ids.find(library_name);
        legion_assert(finder != library_serdez_ids.end());
        legion_assert(!finder->second.result_set);
        legion_assert(finder->second.ready == done);
        finder->second.result = result;
        finder->second.result_set = true;
      }
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SerdezLibraryResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->handle_library_serdez_response(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexSpaceDestruction::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      legion_assert(done.exists());
      std::set<RtEvent> applied;
      runtime->destroy_index_space(handle, source, applied);
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexPartitionDestruction::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexPartition handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      legion_assert(done.exists());
      std::set<RtEvent> applied;
      runtime->destroy_index_partition(handle, applied);
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FieldSpaceDestruction::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldSpace handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      legion_assert(done.exists());
      std::set<RtEvent> applied;
      runtime->destroy_field_space(handle, applied);
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalRegionDestruction::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      RtUserEvent done;
      derez.deserialize(done);
      std::set<RtEvent> applied;
      runtime->destroy_logical_region(handle, applied);
      if (done.exists())
      {
        if (!applied.empty())
          Runtime::trigger_event(done, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedIDRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      std::atomic<DistributedID>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      const DistributedID did = runtime->get_available_distributed_id();
      DistributedIDResponse rez;
      rez.serialize(did);
      rez.serialize(target);
      rez.serialize(done);
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedIDResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      std::atomic<DistributedID>* target;
      derez.deserialize(target);
      target->store(did);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TopLevelTaskComplete::handle(Deserializer&, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      runtime->decrement_outstanding_top_level_tasks();
    }

    //--------------------------------------------------------------------------
    /*static*/ void StartupBarrierMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      RtBarrier startup_barrier;
      derez.deserialize(startup_barrier);
      runtime->broadcast_startup_barrier(startup_barrier);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SharedOwnershipMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      int kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case 0:
          {
            IndexSpace handle;
            derez.deserialize(handle);
            runtime->create_shared_ownership(handle, false, true);
            break;
          }
        case 1:
          {
            IndexPartition handle;
            derez.deserialize(handle);
            runtime->create_shared_ownership(handle, false, true);
            break;
          }
        case 2:
          {
            FieldSpace handle;
            derez.deserialize(handle);
            runtime->create_shared_ownership(handle, false, true);
            break;
          }
        case 3:
          {
            LogicalRegion handle;
            derez.deserialize(handle);
            runtime->create_shared_ownership(handle, false, true);
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    bool Runtime::create_physical_instance(
        Memory target_memory, const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
        UniqueID creator_id, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      if (unsat != nullptr)
      {
        LayoutConstraintKind unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
        unsigned unsat_index = 0;
        if (!manager->create_physical_instance(
                constraints, regions, coordinates, result, processor, acquire,
                priority, tight_bounds, &unsat_kind, &unsat_index, footprint,
                safe_for_unbounded_pools, creator_id))
        {
          *unsat = constraints.convert_unsatisfied(unsat_kind, unsat_index);
          return false;
        }
        else
          return true;
      }
      else
        return manager->create_physical_instance(
            constraints, regions, coordinates, result, processor, acquire,
            priority, tight_bounds, nullptr, nullptr, footprint,
            safe_for_unbounded_pools, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::create_physical_instance(
        Memory target_memory, LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
        UniqueID creator_id, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      if (unsat != nullptr)
      {
        LayoutConstraintKind unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
        unsigned unsat_index = 0;
        if (!manager->create_physical_instance(
                constraints, regions, coordinates, result, processor, acquire,
                priority, tight_bounds, &unsat_kind, &unsat_index, footprint,
                safe_for_unbounded_pools, creator_id))
        {
          *unsat = constraints->convert_unsatisfied(unsat_kind, unsat_index);
          return false;
        }
        else
          return true;
      }
      else
        return manager->create_physical_instance(
            constraints, regions, coordinates, result, processor, acquire,
            priority, tight_bounds, nullptr, nullptr, footprint,
            safe_for_unbounded_pools, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_or_create_physical_instance(
        Memory target_memory, const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        bool& created, Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
        UniqueID creator_id, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      if (unsat != nullptr)
      {
        LayoutConstraintKind unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
        unsigned unsat_index = 0;
        if (!manager->find_or_create_physical_instance(
                constraints, regions, coordinates, result, created, processor,
                acquire, priority, tight_bounds, &unsat_kind, &unsat_index,
                footprint, safe_for_unbounded_pools, creator_id))
        {
          *unsat = constraints.convert_unsatisfied(unsat_kind, unsat_index);
          return false;
        }
        else
          return true;
      }
      else
        return manager->find_or_create_physical_instance(
            constraints, regions, coordinates, result, created, processor,
            acquire, priority, tight_bounds, nullptr, nullptr, footprint,
            safe_for_unbounded_pools, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_or_create_physical_instance(
        Memory target_memory, LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        const TaskTreeCoordinates& coordinates, MappingInstance& result,
        bool& created, Processor processor, bool acquire, GCPriority priority,
        bool tight_bounds, const LayoutConstraint** unsat, size_t* footprint,
        UniqueID creator_id, RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      if (unsat != nullptr)
      {
        LayoutConstraintKind unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
        unsigned unsat_index = 0;
        if (!manager->find_or_create_physical_instance(
                constraints, regions, coordinates, result, created, processor,
                acquire, priority, tight_bounds, &unsat_kind, &unsat_index,
                footprint, safe_for_unbounded_pools, creator_id))
        {
          *unsat = constraints->convert_unsatisfied(unsat_kind, unsat_index);
          return false;
        }
        else
          return true;
      }
      else
        return manager->find_or_create_physical_instance(
            constraints, regions, coordinates, result, created, processor,
            acquire, priority, tight_bounds, nullptr, nullptr, footprint,
            safe_for_unbounded_pools, creator_id);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_physical_instance(
        Memory target_memory, const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      return manager->find_physical_instance(
          constraints, regions, result, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    bool Runtime::find_physical_instance(
        Memory target_memory, LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions, MappingInstance& result,
        bool acquire, bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      return manager->find_physical_instance(
          constraints, regions, result, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    void Runtime::find_physical_instances(
        Memory target_memory, const LayoutConstraintSet& constraints,
        const std::vector<LogicalRegion>& regions,
        std::vector<MappingInstance>& results, bool acquire,
        bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      return manager->find_physical_instances(
          constraints, regions, results, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    void Runtime::find_physical_instances(
        Memory target_memory, LayoutConstraints* constraints,
        const std::vector<LogicalRegion>& regions,
        std::vector<MappingInstance>& results, bool acquire,
        bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      MemoryManager* manager = find_memory_manager(target_memory);
      return manager->find_physical_instances(
          constraints, regions, results, acquire, tight_region_bounds);
    }

    //--------------------------------------------------------------------------
    void Runtime::release_tree_instances(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      std::map<Memory, MemoryManager*> copy_managers;
      {
        AutoLock m_lock(memory_manager_lock, false /*exclusive*/);
        copy_managers = memory_managers;
      }
      for (const std::pair<const Memory, MemoryManager*>& it : copy_managers)
        it.second->release_tree_instances(tid);
    }

    //--------------------------------------------------------------------------
    void Runtime::activate_context(InnerContext* context)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
        it.second->activate_context(context);
    }

    //--------------------------------------------------------------------------
    void Runtime::deactivate_context(InnerContext* context)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Processor, ProcessorManager*>& it :
           proc_managers)
        it.second->deactivate_context(context);
    }

    //--------------------------------------------------------------------------
    void Runtime::add_to_ready_queue(Processor p, SingleTask* task)
    //--------------------------------------------------------------------------
    {
      legion_assert(p.kind() != Processor::UTIL_PROC);
      legion_assert(proc_managers.find(p) != proc_managers.end());
      proc_managers[p]->add_to_ready_queue(task);
    }

    //--------------------------------------------------------------------------
    Processor Runtime::find_processor_group(const std::vector<Processor>& procs)
    //--------------------------------------------------------------------------
    {
      // Compute a hash of all the processor ids to avoid testing all sets
      // Only need to worry about local IDs since all processors are
      // in this address space.
      ProcessorMask local_mask = find_processor_mask(procs);
      uint64_t hash = local_mask.get_hash_key();
      AutoLock g_lock(group_lock);
      rt::map<uint64_t, rt::deque<ProcessorGroupInfo> >::iterator finder =
          processor_groups.find(hash);
      if (finder != processor_groups.end())
      {
        for (rt::deque<ProcessorGroupInfo>::const_iterator it =
                 finder->second.begin();
             it != finder->second.end(); it++)
        {
          if (local_mask == it->processor_mask)
            return it->processor_group;
        }
      }
      // If we make it here create a new processor group and add it
      ProcessorGroup group = ProcessorGroup::create_group(procs);
      if (finder != processor_groups.end())
        finder->second.emplace_back(ProcessorGroupInfo(group, local_mask));
      else
        processor_groups[hash].emplace_back(
            ProcessorGroupInfo(group, local_mask));
      return group;
    }

    //--------------------------------------------------------------------------
    ProcessorMask Runtime::find_processor_mask(
        const std::vector<Processor>& procs)
    //--------------------------------------------------------------------------
    {
      ProcessorMask result;
      std::vector<Processor> need_allocation;
      {
        AutoLock p_lock(processor_mapping_lock, false /*exclusive*/);
        for (const Processor& proc : procs)
        {
          std::map<Processor, unsigned>::const_iterator finder =
              processor_mapping.find(proc);
          if (finder == processor_mapping.end())
          {
            need_allocation.emplace_back(proc);
            continue;
          }
          result.set_bit(finder->second);
        }
      }
      if (need_allocation.empty())
        return result;
      AutoLock p_lock(processor_mapping_lock);
      for (const Processor& proc : need_allocation)
      {
        // Check to make sure we didn't lose the race
        std::map<Processor, unsigned>::const_iterator finder =
            processor_mapping.find(proc);
        if (finder != processor_mapping.end())
        {
          result.set_bit(finder->second);
          continue;
        }
        unsigned next_index = processor_mapping.size();
        legion_assert(next_index < LEGION_MAX_NUM_PROCS);
        processor_mapping[proc] = next_index;
        result.set_bit(next_index);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::order_concurrent_task_launch(
        Processor proc, SingleTask* task, ApEvent precondition,
        ApUserEvent ready, VariantID vid)
    //--------------------------------------------------------------------------
    {
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      legion_assert(finder != proc_managers.end());
      finder->second->order_concurrent_task_launch(
          task, precondition, ready, vid);
    }

    //--------------------------------------------------------------------------
    void Runtime::end_concurrent_task(Processor proc)
    //--------------------------------------------------------------------------
    {
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      legion_assert(finder != proc_managers.end());
      finder->second->end_concurrent_task();
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_next_static_distributed_id(uint64_t& next_did)
    //--------------------------------------------------------------------------
    {
      const DistributedID result = next_did++;
      // If we're the owner we have to bump the available ones here too
      if (determine_owner(result) == address_space)
        legion_no_skip_assert(get_available_distributed_id() == result);
      return result;
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_available_distributed_id(void)
    //--------------------------------------------------------------------------
    {
      DistributedID result = unique_distributed_id.fetch_add(runtime_stride);
      // Check for overflow
      legion_assert(result < LEGION_DISTRIBUTED_ID_MASK);
      return result;
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_remote_distributed_id(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      std::atomic<DistributedID> result(0);
      const RtUserEvent done = Runtime::create_rt_user_event();
      DistributedIDRequest rez;
      rez.serialize(&result);
      rez.serialize(done);
      rez.dispatch(target);
      if (!done.has_triggered())
        done.wait();
      legion_assert(result.load() != 0);
      return result.load();
    }

    //--------------------------------------------------------------------------
    AddressSpaceID Runtime::determine_owner(DistributedID did) const
    //--------------------------------------------------------------------------
    {
      return ((did & LEGION_DISTRIBUTED_ID_MASK) % total_address_spaces);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::find_distance(AddressSpaceID src, AddressSpaceID dst) const
    //--------------------------------------------------------------------------
    {
      size_t abs_diff = (src < dst) ? (dst - src) : (src - dst);
      return (abs_diff < (total_address_spaces / 2)) ?
                 abs_diff :
                 (total_address_spaces - abs_diff);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_distributed_collectable(
        DistributedID did, DistributedCollectable* dc)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      RtUserEvent to_trigger;
      {
        AutoLock dc_lock(distributed_collectable_lock);
        // If we make it here then we have the lock
        legion_assert(dist_collectables.find(did) == dist_collectables.end());
        dist_collectables[did] = dc;
        // See if this was a pending collectable
        std::map<
            DistributedID,
            std::pair<DistributedCollectable*, RtUserEvent> >::iterator finder =
            pending_collectables.find(did);
        if (finder != pending_collectables.end())
        {
          legion_assert(
              (finder->second.first == dc) ||
              (finder->second.first == nullptr));
          to_trigger = finder->second.second;
          pending_collectables.erase(finder);
        }
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock);
      lng::map<DistributedID, DistributedCollectable*>::iterator finder =
          dist_collectables.find(did);
      legion_assert(finder != dist_collectables.end());
      dist_collectables.erase(finder);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_distributed_collectable(DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
      return (dist_collectables.find(did) != dist_collectables.end());
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::find_distributed_collectable(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      RtEvent ready;
      DistributedCollectable* result = nullptr;
      const DistributedID to_find = LEGION_DISTRIBUTED_ID_FILTER(did);
      {
        AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(to_find);
        if (finder == dist_collectables.end())
        {
          // Check to see if it is in the pending set too
          std::map<
              DistributedID,
              std::pair<DistributedCollectable*, RtUserEvent> >::const_iterator
              pending_finder = pending_collectables.find(to_find);
          if (pending_finder != pending_collectables.end())
          {
            result = pending_finder->second.first;
            ready = pending_finder->second.second;
          }
        }
        else
          return finder->second;
      }
      if (!ready.exists())
      {
        AutoLock d_lock(distributed_collectable_lock);
        // Try again to see if we lost the race
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(to_find);
        if (finder == dist_collectables.end())
        {
          // Check to see if it is in the pending set too
          std::map<
              DistributedID,
              std::pair<DistributedCollectable*, RtUserEvent> >::iterator
              pending_finder = pending_collectables.find(to_find);
          if (pending_finder == pending_collectables.end())
            pending_finder =
                pending_collectables
                    .emplace(std::make_pair(
                        to_find,
                        std::pair<DistributedCollectable*, RtUserEvent>(
                            (DistributedCollectable*)nullptr,
                            RtUserEvent::NO_RT_USER_EVENT)))
                    .first;
          result = pending_finder->second.first;
          if (!pending_finder->second.second.exists())
            pending_finder->second.second = Runtime::create_rt_user_event();
          ready = pending_finder->second.second;
        }
        else
          return finder->second;
      }
      if (!ready.has_triggered())
        ready.wait();
      if (result != nullptr)
        return result;
      AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
      lng::map<DistributedID, DistributedCollectable*>::const_iterator finder =
          dist_collectables.find(to_find);
      legion_assert(finder != dist_collectables.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    DistributedCollectable* Runtime::weak_find_distributed_collectable(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      const DistributedID to_find = LEGION_DISTRIBUTED_ID_FILTER(did);
      AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
      lng::map<DistributedID, DistributedCollectable*>::const_iterator finder =
          dist_collectables.find(to_find);
      if (finder == dist_collectables.end())
        return nullptr;
      finder->second->add_base_resource_ref(RUNTIME_REF);
      return finder->second;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void* Runtime::find_or_create_pending_collectable_location(
        DistributedID did)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      AutoLock d_lock(distributed_collectable_lock);
      legion_assert(dist_collectables.find(did) == dist_collectables.end());
      std::map<
          DistributedID,
          std::pair<DistributedCollectable*, RtUserEvent> >::iterator finder =
          pending_collectables.find(did);
      if (finder == pending_collectables.end())
        finder = pending_collectables
                     .emplace(std::make_pair(
                         did, std::pair<DistributedCollectable*, RtUserEvent>(
                                  (DistributedCollectable*)nullptr,
                                  RtUserEvent::NO_RT_USER_EVENT)))
                     .first;
      if (finder->second.first == nullptr)
        finder->second.first =
            legion_malloc<T, LONG_LIFETIME>(sizeof(T), alignof(T));
      return finder->second.first;
    }

    // Instantiate the template for types that use it
    template void* Runtime::find_or_create_pending_collectable_location<
        RemoteContext>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        PhysicalManager>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        EquivalenceSet>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        MaterializedView>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        ReductionView>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        FillView>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        PhiView>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        ReplicatedView>(DistributedID);
    template void* Runtime::find_or_create_pending_collectable_location<
        AllreduceView>(DistributedID);

    //--------------------------------------------------------------------------
    LogicalView* Runtime::find_or_request_logical_view(
        DistributedID did, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable* dc = nullptr;
      if (LogicalView::is_materialized_did(did))
        dc = find_or_request_distributed_collectable<
            MaterializedView, ViewRequestMessage>(did, ready);
      else if (LogicalView::is_reduction_did(did))
        dc = find_or_request_distributed_collectable<
            ReductionView, ViewRequestMessage>(did, ready);
      else if (LogicalView::is_fill_did(did))
        dc = find_or_request_distributed_collectable<
            FillView, ViewRequestMessage>(did, ready);
      else if (LogicalView::is_replicated_did(did))
        dc = find_or_request_distributed_collectable<
            ReplicatedView, ViewRequestMessage>(did, ready);
      else if (LogicalView::is_allreduce_did(did))
        dc = find_or_request_distributed_collectable<
            AllreduceView, ViewRequestMessage>(did, ready);
      else if (LogicalView::is_phi_did(did))
        dc = find_or_request_distributed_collectable<
            PhiView, ViewRequestMessage>(did, ready);
      else
        std::abort();
      // Have to static cast since the memory might not have been initialized
      return static_cast<LogicalView*>(dc);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* Runtime::find_or_request_instance_manager(
        DistributedID did, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      DistributedCollectable* dc = nullptr;
      if (InstanceManager::is_physical_did(did))
        dc = find_or_request_distributed_collectable<
            PhysicalManager, ManagerRequestMessage>(did, ready);
      else
        std::abort();
      // Have to static cast since the memory might not have been initialized
      return static_cast<PhysicalManager*>(dc);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* Runtime::find_or_request_equivalence_set(
        DistributedID did, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(LEGION_DISTRIBUTED_HELP_DECODE(did) == EQUIVALENCE_SET_DC);
      DistributedCollectable* dc = find_or_request_distributed_collectable<
          EquivalenceSet, EquivalenceSetRequest>(did, ready);
      // Have to static cast since the memory might not have been initialized
      return static_cast<EquivalenceSet*>(dc);
    }

    //--------------------------------------------------------------------------
    InnerContext* Runtime::find_or_request_inner_context(DistributedID did)
    //--------------------------------------------------------------------------
    {
      legion_assert(LEGION_DISTRIBUTED_HELP_DECODE(did) == INNER_CONTEXT_DC);
      RtEvent ready;
      DistributedCollectable* dc = find_or_request_distributed_collectable<
          RemoteContext, RemoteContextRequest>(did, ready);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      return static_cast<InnerContext*>(dc);
    }

    //--------------------------------------------------------------------------
    ShardManager* Runtime::find_shard_manager(DistributedID did, bool can_fail)
    //--------------------------------------------------------------------------
    {
      legion_assert(LEGION_DISTRIBUTED_HELP_DECODE(did) == SHARD_MANAGER_DC);
      if (can_fail)
      {
        const DistributedID to_find = LEGION_DISTRIBUTED_ID_FILTER(did);
        AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(to_find);
        if (finder == dist_collectables.end())
          return nullptr;
        else
          return static_cast<ShardManager*>(finder->second);
      }
      else
        return static_cast<ShardManager*>(find_distributed_collectable(did));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename AM>
    DistributedCollectable* Runtime::find_or_request_distributed_collectable(
        DistributedID to_find, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      const DistributedID did = LEGION_DISTRIBUTED_ID_FILTER(to_find);
      DistributedCollectable* result = nullptr;
      {
        AutoLock d_lock(distributed_collectable_lock);
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(did);
        // If we've already got it, then we are done
        if (finder != dist_collectables.end())
        {
          ready = RtEvent::NO_RT_EVENT;
          return finder->second;
        }
        // If it is already pending, we can just return the ready event
        std::map<
            DistributedID,
            std::pair<DistributedCollectable*, RtUserEvent> >::iterator
            pending_finder = pending_collectables.find(did);
        if (pending_finder != pending_collectables.end())
        {
          if (pending_finder->second.first == nullptr)
            pending_finder->second.first =
                legion_malloc<T, LONG_LIFETIME>(sizeof(T), alignof(T));
          if (!pending_finder->second.second.exists())
            pending_finder->second.second = Runtime::create_rt_user_event();
          ready = pending_finder->second.second;
          return pending_finder->second.first;
        }
        // This is the first request we've seen for this did, make it now
        // Allocate space for the result and type case
        result = legion_malloc<T, LONG_LIFETIME>(sizeof(T), alignof(T));
        RtUserEvent to_trigger = Runtime::create_rt_user_event();
        pending_collectables[did] =
            std::pair<DistributedCollectable*, RtUserEvent>(result, to_trigger);
        ready = to_trigger;
      }
      AddressSpaceID target = determine_owner(did);
      legion_assert(
          target != address_space);  // shouldn't be sending to ourself
      // Now send the message
      AM rez;
      {
        RezCheck z(rez);
        rez.serialize(to_find);
        rez.serialize(address_space);
      }
      rez.dispatch(target);
      return result;
    }

    //--------------------------------------------------------------------------
    FutureImpl* Runtime::find_or_create_future(
        DistributedID did, DistributedID ctx_did,
        const ContextCoordinate& coord, Provenance* provenance,
        bool has_global_reference, RtEvent& registered, Operation* op,
        GenerationID gen, UniqueID op_uid, int op_depth,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      {
        AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
          FutureImpl* result = legion_safe_cast<FutureImpl*>(finder->second);
          return result;
        }
      }
      InnerContext* context = find_or_request_inner_context(ctx_did);
      FutureImpl* result = new FutureImpl(
          context, false /*register*/, did, op, gen, coord, op_uid, op_depth,
          provenance, mapping);
      // Retake the lock and see if we lost the race
      AutoLock d_lock(distributed_collectable_lock);
      lng::map<DistributedID, DistributedCollectable*>::const_iterator finder =
          dist_collectables.find(did);
      if (finder != dist_collectables.end())
      {
        // We lost the race
        delete result;
        result = legion_safe_cast<FutureImpl*>(finder->second);
        return result;
      }
      registered = result->record_future_registered(has_global_reference);
      dist_collectables[did] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* Runtime::find_or_create_future_map(
        DistributedID did, TaskContext* ctx, uint64_t coord, IndexSpace domain,
        Provenance* provenance, const std::optional<uint64_t>& ctx_index)
    //--------------------------------------------------------------------------
    {
      did &= LEGION_DISTRIBUTED_ID_MASK;
      {
        AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
        lng::map<DistributedID, DistributedCollectable*>::const_iterator
            finder = dist_collectables.find(did);
        if (finder != dist_collectables.end())
        {
          FutureMapImpl* result =
              legion_safe_cast<FutureMapImpl*>(finder->second);
          return result;
        }
      }
      legion_assert(domain.exists());
      IndexSpaceNode* domain_node = get_node(domain);
      FutureMapImpl* result = new FutureMapImpl(
          ctx, domain_node, did, coord, ctx_index, provenance,
          false /*register now*/);
      AutoLock d_lock(distributed_collectable_lock);
      lng::map<DistributedID, DistributedCollectable*>::const_iterator finder =
          dist_collectables.find(did);
      if (finder != dist_collectables.end())
      {
        // We lost the race
        delete result;
        result = legion_safe_cast<FutureMapImpl*>(finder->second);
        return result;
      }
      result->record_future_map_registered();
      dist_collectables[did] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::find_or_create_index_slice_space(
        const Domain& domain, bool take_ownership, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const TypeTag type_tag = domain.is_type;
      legion_assert(type_tag != 0);
      // Two different paths depending on whether this domain is dense or sparse
      if (domain.dense())
      {
        {
          AutoLock is_lock(is_slice_lock, false /*exclusive*/);
          std::map<Domain, IndexSpaceNode*>::const_iterator finder =
              dense_slice_spaces.find(domain);
          if (finder != dense_slice_spaces.end())
            return finder->second->handle;
        }
        AutoLock is_lock(is_slice_lock);
        // Check to make sure we didn't lose the race
        std::map<Domain, IndexSpaceNode*>::const_iterator finder =
            dense_slice_spaces.find(domain);
        if (finder != dense_slice_spaces.end())
          return finder->second->handle;
        const IndexSpace result(
            get_unique_index_space_id(), get_unique_index_tree_id(), type_tag);
        IndexSpaceNode* node = create_node(
            result, domain, take_ownership, nullptr /*parent*/, 0 /*color*/,
            RtEvent::NO_RT_EVENT, provenance, ApEvent::NO_AP_EVENT,
            0 /*expr id*/, nullptr /*mapping*/, true /*add root reference*/);
        LegionSpy::log_top_index_space(
            result.get_id(), address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
        dense_slice_spaces.emplace(domain, node);
        return result;
      }
      else
      {
        // Create an expression from the domain and get its canonical result
        // Don't need to worry about deleting this explicitly since this will
        // create a task-local reference to it that will be removed at the
        // end of this task and will implicitly delete it if necessary
        IndexSpaceExpression* temp_expr =
            InternalExpressionCreator::create_with_domain(type_tag, domain);
        const uint64_t hash = temp_expr->get_canonical_hash();
        IndexSpaceExpression* canonical = temp_expr->get_canonical_expression();
        // If the temp expression is its own canonical expression then it
        // definitely doesn't overlap with anything else
        if (canonical != temp_expr)
        {
          // See if we can find another entry in the sparse slice spaces
          AutoLock is_lock(is_slice_lock, false /*exclusive*/);
          std::unordered_map<uint64_t, SliceSpaces>::const_iterator finder =
              sparse_slice_spaces.find(hash);
          if (finder != sparse_slice_spaces.end())
          {
            for (unsigned idx = 0; idx < finder->second.size(); idx++)
            {
              if (finder->second[idx]->get_canonical_expression() != canonical)
                continue;
              if (take_ownership)
              {
                Domain copy = domain;
                copy.destroy();
              }
              return finder->second[idx]->handle;
            }
          }
        }
        // If we get here then take the lock and try again
        AutoLock is_lock(is_slice_lock);
        SliceSpaces& spaces = sparse_slice_spaces[hash];
        // Check again to make sure we didn't lose the race
        for (unsigned idx = 0; idx < spaces.size(); idx++)
        {
          if (spaces[idx]->get_canonical_expression() != canonical)
            continue;
          if (take_ownership)
          {
            Domain copy = domain;
            copy.destroy();
          }
          return spaces[idx]->handle;
        }
        // Create a new index space node to save for us to use
        const IndexSpace result(
            get_unique_index_space_id(), get_unique_index_tree_id(), type_tag);
        IndexSpaceNode* node = temp_expr->create_node(
            result, RtEvent::NO_RT_EVENT, provenance, nullptr /*mapping*/);
        LegionSpy::log_top_index_space(
            result.get_id(), address_space,
            (provenance == nullptr) ? std::string_view() : provenance->human);
        spaces.insert(node);
        return result;
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::increment_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
      unsigned previous = outstanding_top_level_tasks.fetch_add(1);
      if (previous == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Illegal attempt to launch a top-level task after "
            << "Runtime::wait_for_shutdown has been called. All top-level "
               "tasks "
            << "must be created on a node before signaling to the runtime that "
            << "the client is ready for Legion to shutdown on that node.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_outstanding_top_level_tasks(void)
    //--------------------------------------------------------------------------
    {
      unsigned previous = outstanding_top_level_tasks.fetch_sub(1);
      legion_assert(previous > 0);
      if (previous == 1)
      {
        if (address_space > 0)
        {
          // Send a message to the next node down the tree to remove our
          // guard reference that we have there
          AddressSpaceID parent = (address_space - 1) / legion_collective_radix;
          TopLevelTaskComplete rez;
          rez.dispatch(parent);
        }
        else  // We're the owner node so start the quiesence algorithm
          issue_runtime_shutdown_attempt();
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::issue_runtime_shutdown_attempt(void)
    //--------------------------------------------------------------------------
    {
      ShutdownManager::RetryShutdownArgs args(
          ShutdownManager::CHECK_TERMINATION);
      // Issue this with a low priority so that other meta-tasks
      // have an opportunity to run
      issue_runtime_meta_task(args, LG_LOW_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void Runtime::initiate_runtime_shutdown(
        AddressSpaceID source, ShutdownManager::ShutdownPhase phase,
        ShutdownManager* owner)
    //--------------------------------------------------------------------------
    {
      log_shutdown.info(
          "Received notification on node %d for phase %d", address_space,
          phase);
      // If this is the first phase, do all our normal stuff
      if (phase == ShutdownManager::CHECK_TERMINATION)
      {
        // Get the preconditions for any outstanding operations still
        // available for garabage collection and wait on them to
        // try and get close to when there are no more outstanding tasks
        std::map<Memory, MemoryManager*> copy_managers;
        {
          AutoLock m_lock(memory_manager_lock, false /*exclusive*/);
          copy_managers = memory_managers;
        }
        std::set<ApEvent> wait_events;
        for (const std::pair<const Memory, MemoryManager*>& it : copy_managers)
          it.second->find_shutdown_preconditions(wait_events);
        if (!wait_events.empty())
        {
          RtEvent wait_on = Runtime::protect_merge_events(wait_events);
          wait_on.wait();
        }
      }
      else if (
          (phase == ShutdownManager::CHECK_SHUTDOWN) && !prepared_for_shutdown)
      {
        // First time we check for shutdown we do the prepare for shutdown
        prepare_runtime_shutdown();
      }
      ShutdownManager* shutdown_manager =
          new ShutdownManager(phase, source, LEGION_SHUTDOWN_RADIX, owner);
      if (shutdown_manager->attempt_shutdown())
        delete shutdown_manager;
    }

    //--------------------------------------------------------------------------
    void Runtime::confirm_runtime_shutdown(
        ShutdownManager* shutdown_manager, bool phase_one)
    //--------------------------------------------------------------------------
    {
      if (has_outstanding_tasks())
      {
        shutdown_manager->record_outstanding_tasks();
#ifdef LEGION_DEBUG
        const char* lg_task_descriptions[LG_LAST_TASK_ID] = {
#define META_TASK_NAMES(kind, type, name) name,
            LEGION_META_TASKS(META_TASK_NAMES)
#undef META_TASK_NAMES
        };
        AutoLock out_lock(outstanding_task_lock, false /*exclusive*/);
        for (const std::pair<const std::pair<unsigned, bool>, unsigned>& it :
             outstanding_task_counts)
        {
          if (it.second == 0)
            continue;
          if (it.first.second)
            log_shutdown.info(
                "RT %d: %d outstanding meta task(s) %s", address_space,
                it.second, lg_task_descriptions[it.first.first]);
          else
            log_shutdown.info(
                "RT %d: %d outstanding application task(s) %d", address_space,
                it.second, it.first.first);
        }
#endif
      }
      // Check all our message managers for outstanding messages
      for (unsigned idx = 0; idx < LEGION_MAX_NUM_NODES; idx++)
      {
        MessageManager* manager = message_managers[idx].load();
        if (manager != nullptr)
          manager->confirm_shutdown(shutdown_manager, phase_one);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::prepare_runtime_shutdown(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!prepared_for_shutdown);
      legion_assert(virtual_manager != nullptr);
      // Search through all our distributed collectables and find any
      // futures which are leaking and therefore need to be finalized
      std::vector<FutureImpl*> leaked_futures;
      {
        // Also have any leaking futures force delete their instances
        AutoLock d_lock(distributed_collectable_lock, false /*exclusive*/);
        for (lng::map<DistributedID, DistributedCollectable*>::const_iterator
                 it = dist_collectables.begin();
             it != dist_collectables.end(); it++)
        {
          // See if this is a future
          if (LEGION_DISTRIBUTED_HELP_DECODE(it->second->did) != FUTURE_DC)
            continue;
          FutureImpl* impl = legion_safe_cast<FutureImpl*>(it->second);
          impl->add_base_resource_ref(RUNTIME_REF);
          leaked_futures.emplace_back(impl);
        }
      }
      for (FutureImpl* const & future_ptr : leaked_futures)
      {
        future_ptr->prepare_for_shutdown();
        if (future_ptr->remove_base_resource_ref(RUNTIME_REF))
          delete future_ptr;
      }
      for (const std::pair<const Memory, MemoryManager*>& mem_pair :
           memory_managers)
        mem_pair.second->prepare_for_shutdown();
      // Do processor managers after memory managers in case we need to
      // report any deleted instances back to the mappers
      for (const std::pair<const Processor, ProcessorManager*>& proc_pair :
           proc_managers)
        proc_pair.second->prepare_for_shutdown();
      // Destroy any index slice spaces that we made during execution
      std::set<RtEvent> applied;
      for (const std::pair<const Domain, IndexSpaceNode*>& it :
           dense_slice_spaces)
        destroy_index_space(it.second->handle, address_space, applied);
      for (const std::pair<const uint64_t, SliceSpaces>& it :
           sparse_slice_spaces)
        for (unsigned idx = 0; idx < it.second.size(); idx++)
          destroy_index_space(it.second[idx]->handle, address_space, applied);
      for (const std::pair<const ProjectionID, ProjectionFunction*>& proj_pair :
           projection_functions)
        proj_pair.second->prepare_for_shutdown();
      std::vector<LayoutConstraints*> to_remove;
      {
        AutoLock l_lock(layout_constraints_lock, false /*exclusive*/);
        for (const std::pair<const LayoutConstraintID, LayoutConstraints*>& it :
             layout_constraints_table)
          if (it.second->is_owner() && !it.second->internal)
            to_remove.emplace_back(it.second);
      }
      if (!to_remove.empty())
      {
        for (LayoutConstraints* constraints : to_remove)
          if (constraints->remove_base_gc_ref(APPLICATION_REF))
            delete constraints;
      }
      if (!redop_fill_views.empty())
      {
        for (const std::pair<const ReductionOpID, FillView*>& it :
             redop_fill_views)
          if (it.second->remove_base_valid_ref(RUNTIME_REF))
            delete it.second;
        redop_fill_views.clear();
      }
      if (!empty_expressions.empty())
      {
        for (IndexSpaceExpression* empty_expr : empty_expressions)
          if (empty_expr->remove_base_expression_reference(RUNTIME_REF))
            delete empty_expr;
        empty_expressions.clear();
      }
      if (virtual_manager->remove_base_gc_ref(NEVER_GC_REF))
        delete virtual_manager;
      virtual_manager = nullptr;
      if (!applied.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(applied);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      prepared_for_shutdown = true;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_outstanding_tasks(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      AutoLock out_lock(outstanding_task_lock);
      return (total_outstanding_tasks > 0);
#else
      return total_outstanding_tasks.load();
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::increment_total_outstanding_tasks(unsigned tid, bool meta)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      AutoLock out_lock(outstanding_task_lock);
      total_outstanding_tasks++;
      std::pair<unsigned, bool> key(tid, meta);
      std::map<std::pair<unsigned, bool>, unsigned>::iterator finder =
          outstanding_task_counts.find(key);
      if (finder == outstanding_task_counts.end())
        outstanding_task_counts[key] = 1;
      else
        finder->second++;
#else
      total_outstanding_tasks.fetch_add(1);
#endif
    }

    //--------------------------------------------------------------------------
    void Runtime::decrement_total_outstanding_tasks(unsigned tid, bool meta)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      AutoLock out_lock(outstanding_task_lock);
      legion_assert(total_outstanding_tasks > 0);
      total_outstanding_tasks--;
      std::pair<unsigned, bool> key(tid, meta);
      std::map<std::pair<unsigned, bool>, unsigned>::iterator finder =
          outstanding_task_counts.find(key);
      legion_assert(finder != outstanding_task_counts.end());
      legion_assert(finder->second > 0);
      finder->second--;
#else
      total_outstanding_tasks.fetch_sub(1);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_index_space(
        IndexSpace handle, const Domain& domain, bool take_ownership,
        Provenance* provenance, CollectiveMapping* mapping,
        IndexSpaceExprID expr_id, ApEvent ready /*=ApEvent::NO_AP_EVENT*/,
        RtEvent init /*= RtEvent::NO_RT_EVENT*/)
    //--------------------------------------------------------------------------
    {
      return create_node(
          handle, domain, take_ownership, nullptr /*parent*/, 0 /*color*/, init,
          provenance, ready, expr_id, mapping, true /*add root reference*/);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_union_space(
        IndexSpace handle, Provenance* provenance,
        const std::vector<IndexSpace>& sources, RtEvent initialized,
        CollectiveMapping* collective_mapping, IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
      // Construct the set of index space expressions
      std::set<IndexSpaceExpression*> exprs;
      for (const IndexSpace& index_space : sources)
      {
        if (!index_space.exists())
          continue;
        exprs.insert(get_node(index_space));
      }
      legion_assert(!exprs.empty());
      IndexSpaceExpression* expr = union_index_spaces(exprs);
      return expr->create_node(
          handle, initialized, provenance, collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_intersection_space(
        IndexSpace handle, Provenance* provenance,
        const std::vector<IndexSpace>& sources, RtEvent initialized,
        CollectiveMapping* collective_mapping, IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
      // Construct the set of index space expressions
      std::set<IndexSpaceExpression*> exprs;
      for (const IndexSpace& index_space : sources)
      {
        if (!index_space.exists())
          continue;
        exprs.insert(get_node(index_space));
      }
      legion_assert(!exprs.empty());
      IndexSpaceExpression* expr = intersect_index_spaces(exprs);
      return expr->create_node(
          handle, initialized, provenance, collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_difference_space(
        IndexSpace handle, Provenance* provenance, IndexSpace left,
        IndexSpace right, RtEvent initialized,
        CollectiveMapping* collective_mapping, IndexSpaceExprID expr_id)
    //--------------------------------------------------------------------------
    {
      legion_assert(left.exists());
      IndexSpaceNode* lhs = get_node(left);
      if (!right.exists())
        return lhs->create_node(
            handle, initialized, provenance, collective_mapping);
      IndexSpaceNode* rhs = get_node(right);
      IndexSpaceExpression* expr = subtract_index_spaces(lhs, rhs);
      return expr->create_node(
          handle, initialized, provenance, collective_mapping, expr_id);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_space(
        IndexSpace handle, AddressSpaceID source, std::set<RtEvent>& applied,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      if (node->invalidate_root(source, applied, mapping))
        delete node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::send_index_space_destruction(
        IndexSpace handle, AddressSpaceID target, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      IndexSpaceDestruction rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        const RtUserEvent done = create_rt_user_event();
        rez.serialize(done);
        applied.insert(done);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_index_partition(
        IndexPartition handle, std::set<RtEvent>& applied,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space = IndexPartNode::get_owner_space(handle);
      if (mapping != nullptr)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == address_space)
          {
            IndexPartNode* node = get_node(handle);
            if (node->remove_base_valid_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == address_space)
            send_index_partition_destruction(handle, owner_space, applied);
        }
      }
      else
      {
        if (owner_space == address_space)
        {
          IndexPartNode* node = get_node(handle);
          if (node->remove_base_valid_ref(APPLICATION_REF))
            delete node;
        }
        else
          send_index_partition_destruction(handle, owner_space, applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::send_index_partition_destruction(
        IndexPartition handle, AddressSpaceID target,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      IndexPartitionDestruction rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        const RtUserEvent done = create_rt_user_event();
        rez.serialize(done);
        applied.insert(done);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_coordinate_size(IndexSpace handle, bool range)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->get_coordinate_size(range);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* parent_node = get_node(parent);
      IndexPartNode* child_node = parent_node->get_child(color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_subspace(
        IndexPartition parent, const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* parent_node = get_node(parent);
      return parent_node->color_space->contains_point(realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_subspace(
        IndexPartition parent, const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* parent_node = get_node(parent);
      LegionColor child_color =
          parent_node->color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode* child_node = parent_node->get_child(child_color);
      return child_node->handle;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_domain(
        IndexSpace handle, void* realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      node->get_index_space_domain(realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    Domain Runtime::get_index_partition_color_space(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* part = get_node(p);
      const IndexSpace color_space = part->color_space->handle;
      switch (NT_TemplateHelper::get_dim(color_space.get_type_tag()))
      {
#define DIMFUNC(DIM)                                      \
  case DIM:                                               \
    {                                                     \
      DomainT<DIM, coord_t> color_index_space;            \
      get_index_space_domain(                             \
          color_space, &color_index_space,                \
          NT_TemplateHelper::encode_tag<DIM, coord_t>()); \
      return Domain(color_index_space);                   \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return Domain::NO_DOMAIN;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_partition_color_space(
        IndexPartition p, void* realm_is, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* part = get_node(p);
      const IndexSpace color_space = part->color_space->handle;
      get_index_space_domain(color_space, realm_is, type_tag);
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_index_partition_color_space_name(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      return get_node(p)->color_space->handle;
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_partition_colors(
        IndexSpace sp, std::set<Color>& colors)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(sp);
      std::vector<LegionColor> temp_colors;
      node->get_colors(temp_colors);
      for (const LegionColor& color : temp_colors) colors.insert(color);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_index_space_color_point(
        IndexSpace handle, void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      if (node->parent == nullptr)
      {
        // We know the answer here
        if (type_tag != NT_TemplateHelper::encode_tag<1, coord_t>())
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'get_index_space_color'.";
          error.raise();
        }
        Realm::Point<1, coord_t>* color =
            (Realm::Point<1, coord_t>*)realm_color;
        *color = Realm::Point<1, coord_t>(0);
        return;
      }
      // Otherwise we can get the color for the partition color space
      IndexSpaceNode* color_space = node->parent->color_space;
      color_space->delinearize_color(node->color, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_index_space_color_point(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      return node->get_domain_point_color();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_index_partition_color(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = get_node(handle);
      return node->color;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::get_parent_index_space(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      return (node->parent != nullptr);
    }

    //--------------------------------------------------------------------------
    IndexPartition Runtime::get_parent_index_partition(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      if (node->parent == nullptr)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Parent index partition requested for index space " << handle
              << " with no parent. Use has_parent_index_partition to check "
              << "before requesting a parent.";
        error.raise();
      }
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_space_depth(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->depth;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_index_partition_depth(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      return get_node(handle)->depth;
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_domain_volume(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      return node->get_volume();
    }

    //--------------------------------------------------------------------------
    void Runtime::find_domain(IndexSpace handle, Domain& launch_domain)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle);
      launch_domain = node->get_tight_domain();
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_disjoint(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(p);
      return node->is_disjoint(true /*app query*/);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_index_partition_complete(IndexPartition p)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = runtime->get_node(p);
      return node->is_complete(true /*app query*/);
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_partition(IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* parent_node = runtime->get_node(parent);
      return parent_node->has_color(color);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_field_space(
        FieldSpace handle, std::set<RtEvent>& applied,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space =
          FieldSpaceNode::get_owner_space(handle);
      if (mapping != nullptr)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == address_space)
          {
            FieldSpaceNode* node = get_node(handle);
            if (node->remove_base_gc_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == address_space)
            send_field_space_destruction(handle, owner_space, applied);
        }
      }
      else
      {
        if (owner_space == address_space)
        {
          FieldSpaceNode* node = get_node(handle);
          if (node->remove_base_gc_ref(APPLICATION_REF))
            delete node;
        }
        else
          send_field_space_destruction(handle, owner_space, applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::send_field_space_destruction(
        FieldSpace handle, AddressSpaceID target, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      FieldSpaceDestruction rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        const RtUserEvent done = create_rt_user_event();
        rez.serialize(done);
        applied.insert(done);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    RtEvent Runtime::allocate_field(
        FieldSpace handle, size_t field_size, FieldID fid,
        CustomSerdezID serdez_id, Provenance* provenance,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      RtEvent ready = node->allocate_field(
          fid, field_size, serdez_id, provenance, sharded_non_owner);
      return ready;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* Runtime::allocate_field(
        FieldSpace handle, ApEvent size_ready, FieldID fid,
        CustomSerdezID serdez_id, Provenance* provenance, RtEvent& precondition,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      precondition = node->allocate_field(
          fid, size_ready, serdez_id, provenance, sharded_non_owner);
      return node;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field(
        FieldSpace handle, FieldID fid, std::set<RtEvent>& applied,
        bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->free_field(fid, address_space, applied, sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    RtEvent Runtime::allocate_fields(
        FieldSpace handle, const std::vector<size_t>& sizes,
        const std::vector<FieldID>& fields, CustomSerdezID serdez_id,
        Provenance* provenance, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      legion_assert(sizes.size() == fields.size());
      // We know that none of these field allocations are local
      FieldSpaceNode* node = get_node(handle);
      RtEvent ready = node->allocate_fields(
          sizes, fields, serdez_id, provenance, sharded_non_owner);
      return ready;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* Runtime::allocate_fields(
        FieldSpace handle, ApEvent sizes_ready,
        const std::vector<FieldID>& fields, CustomSerdezID serdez_id,
        Provenance* provenance, RtEvent& precondition, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      // We know that none of these field allocations are local
      FieldSpaceNode* node = get_node(handle);
      precondition = node->allocate_fields(
          sizes_ready, fields, serdez_id, provenance, sharded_non_owner);
      return node;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_fields(
        FieldSpace handle, const std::vector<FieldID>& to_free,
        std::set<RtEvent>& applied, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->free_fields(to_free, address_space, applied, sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_field_indexes(
        FieldSpace handle, const std::vector<FieldID>& to_free,
        RtEvent freed_event, bool sharded_non_owner)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->free_field_indexes(to_free, freed_event, sharded_non_owner);
    }

    //--------------------------------------------------------------------------
    bool Runtime::allocate_local_fields(
        FieldSpace handle, const std::vector<FieldID>& fields,
        const std::vector<size_t>& sizes, CustomSerdezID serdez_id,
        const std::set<unsigned>& current_indexes,
        std::vector<unsigned>& new_indexes, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      return node->allocate_local_fields(
          fields, sizes, serdez_id, current_indexes, new_indexes, provenance);
    }

    //--------------------------------------------------------------------------
    void Runtime::free_local_fields(
        FieldSpace handle, const std::vector<FieldID>& to_free,
        const std::vector<unsigned>& indexes, const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->free_local_fields(to_free, indexes, mapping);
    }

    //--------------------------------------------------------------------------
    void Runtime::update_local_fields(
        FieldSpace handle, const std::vector<FieldID>& fields,
        const std::vector<size_t>& sizes,
        const std::vector<CustomSerdezID>& serdez_ids,
        const std::vector<unsigned>& indexes, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->update_local_fields(fields, sizes, serdez_ids, indexes, provenance);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_local_fields(
        FieldSpace handle, const std::vector<FieldID>& to_remove)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->remove_local_fields(to_remove);
    }

    //--------------------------------------------------------------------------
    size_t Runtime::get_field_size(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      if (!node->has_field(fid))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "FieldSpace " << handle << " has no field " << fid << ".";
        error.raise();
      }
      return node->get_field_size(fid);
    }

    //--------------------------------------------------------------------------
    CustomSerdezID Runtime::get_field_serdez(FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      if (!node->has_field(fid))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "FieldSpace " << handle << " has no field " << fid << ".";
        error.raise();
      }
      return node->get_field_serdez(fid);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_field_space_fields(
        FieldSpace handle, std::vector<FieldID>& fields)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* node = get_node(handle);
      node->get_all_fields(fields);
    }

    //--------------------------------------------------------------------------
    void Runtime::destroy_logical_region(
        LogicalRegion handle, std::set<RtEvent>& applied,
        const CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID owner_space =
          RegionNode::get_owner_space(handle.get_tree_id());
      if (mapping != nullptr)
      {
        if (mapping->contains(owner_space))
        {
          // If we're the owner space node then we do the removal
          if (owner_space == address_space)
          {
            RegionNode* node = get_node(handle);
            if (node->remove_base_gc_ref(APPLICATION_REF))
              delete node;
          }
        }
        else
        {
          const AddressSpaceID nearest = mapping->find_nearest(owner_space);
          if (nearest == address_space)
            send_logical_region_destruction(handle, owner_space, applied);
        }
      }
      else
      {
        if (owner_space == address_space)
        {
          RegionNode* node = get_node(handle);
          if (node->remove_base_gc_ref(APPLICATION_REF))
            delete node;
        }
        else
          send_logical_region_destruction(handle, owner_space, applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::send_logical_region_destruction(
        LogicalRegion handle, AddressSpaceID target, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      LogicalRegionDestruction rez;
      {
        RezCheck z(rez);
        rez.serialize(handle);
        const RtUserEvent done = create_rt_user_event();
        rez.serialize(done);
        applied.insert(done);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition(
        LogicalRegion parent, IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(parent.tree_did, handle, parent.field_space);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_color(
        LogicalRegion parent, Color c)
    //--------------------------------------------------------------------------
    {
      RegionNode* parent_node = get_node(parent);
      IndexPartNode* index_node = parent_node->row_source->get_child(c);
      LogicalPartition result(
          parent.tree_did, index_node->handle, parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_partition_by_color(
        LogicalRegion parent, Color color)
    //--------------------------------------------------------------------------
    {
      RegionNode* parent_node = get_node(parent);
      return parent_node->has_color(color);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_logical_partition_by_tree(
        IndexPartition handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalPartition(
          LEGION_DISTRIBUTED_HELP_ENCODE(tid, REGION_TREE_NODE_DC), handle,
          space);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion(
        LogicalPartition parent, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(parent.tree_did, handle, parent.field_space);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_color(
        LogicalPartition parent, const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      PartitionNode* parent_node = get_node(parent);
      IndexSpaceNode* color_space = parent_node->row_source->color_space;
      if (!color_space->contains_point(realm_color, type_tag))
      {
        DomainPoint bad_point;
        switch (color_space->get_num_dims())
        {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    {                                                                          \
      RealmPointConverter<DIM, Realm::DIMTYPES>::convert_from(                 \
          realm_color, type_tag, bad_point, "get_logical_subregion_by_color"); \
      break;                                                                   \
    }
          LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
          default:
            std::abort();
        }
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Invalid color space color for child " << bad_point
              << " of logical partition (" << parent.index_partition << ", "
              << parent.field_space << ", " << parent.get_tree_id() << ").";
        error.raise();
      }
      LegionColor color = color_space->linearize_color(realm_color, type_tag);
      IndexSpaceNode* index_node = parent_node->row_source->get_child(color);
      LogicalRegion result(
          parent.tree_did, index_node->handle, parent.field_space);
      return result;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_logical_subregion_by_color(
        LogicalPartition parent, const void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      PartitionNode* parent_node = get_node(parent);
      IndexSpaceNode* color_space = parent_node->row_source->color_space;
      return color_space->contains_point(realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_logical_subregion_by_tree(
        IndexSpace handle, FieldSpace space, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // No lock needed for this one
      return LogicalRegion(
          LEGION_DISTRIBUTED_HELP_ENCODE(tid, REGION_TREE_NODE_DC), handle,
          space);
    }

    //--------------------------------------------------------------------------
    void Runtime::get_logical_region_color(
        LogicalRegion handle, void* realm_color, TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle.get_index_space());
      if (node->parent == nullptr)
      {
        // We know the answer here
        if (type_tag != NT_TemplateHelper::encode_tag<1, coord_t>())
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Dynamic type mismatch in 'get_logical_region_color'.";
          error.raise();
        }
        Realm::Point<1, coord_t>* color =
            (Realm::Point<1, coord_t>*)realm_color;
        *color = Realm::Point<1, coord_t>(0);
        return;
      }
      // Otherwise we can get the color for the partition color space
      IndexSpaceNode* color_space = node->parent->color_space;
      color_space->delinearize_color(node->color, realm_color, type_tag);
    }

    //--------------------------------------------------------------------------
    DomainPoint Runtime::get_logical_region_color_point(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* node = get_node(handle.get_index_space());
      return node->get_domain_point_color();
    }

    //--------------------------------------------------------------------------
    Color Runtime::get_logical_partition_color(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode* node = get_node(handle);
      return node->row_source->color;
    }

    //--------------------------------------------------------------------------
    LogicalRegion Runtime::get_parent_logical_region(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      PartitionNode* node = get_node(handle);
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode* node = get_node(handle);
      return (node->parent != nullptr);
    }

    //--------------------------------------------------------------------------
    LogicalPartition Runtime::get_parent_logical_partition(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      RegionNode* node = get_node(handle);
      if (node->parent == nullptr)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Parent logical partition requested for logical region ("
              << handle.index_space << ", " << handle.field_space << ", "
              << handle.get_tree_id() << ") with no parent. Use "
              << "has_parent_logical_partition to check before requesting a "
                 "parent.";
        error.raise();
      }
      return node->parent->handle;
    }

    //--------------------------------------------------------------------------
    void Runtime::invalidate_region_tree_context(
        ContextID ctx, const RegionRequirement& req,
        bool filter_specific_fields)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type == LEGION_SINGULAR_PROJECTION);
      RegionNode* region_node = get_node(req.region);
      if (filter_specific_fields)
      {
        FieldMask user_mask =
            region_node->column_source->get_field_mask(req.privilege_fields);
        DeletionInvalidator invalidator(ctx, user_mask);
        region_node->visit_node(&invalidator);
      }
      else
      {
        CurrentInvalidator invalidator(ctx);
        region_node->visit_node(&invalidator);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::check_region_tree_context(ContextID ctx)
    //--------------------------------------------------------------------------
    {
      CurrentInitializer init(ctx);
      AutoLock l_lock(lookup_lock, false /*exclusive*/);
      // Need to hold references to prevent deletion race
      for (const std::pair<const RegionTreeID, RegionNode*>& it : tree_nodes)
        it.second->visit_node(&init);
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_node(
        IndexSpace sp, const Domain& domain, bool take_ownership,
        IndexPartNode* parent, LegionColor color, RtEvent initialized,
        Provenance* provenance, ApEvent is_ready, IndexSpaceExprID expr_id,
        CollectiveMapping* mapping, const bool add_root_reference,
        unsigned depth, const bool tree_valid)
    //--------------------------------------------------------------------------
    {
      IndexSpaceCreator creator(
          sp, parent, color, expr_id, initialized, depth, provenance, mapping,
          tree_valid);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode* result = creator.result;
      legion_assert(result != nullptr);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace, IndexSpaceNode*>::const_iterator it =
            index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          // Need to remove resource reference if not owner
          delete result;
          result = it->second;
          // If the parent is nullptr then we don't need to perform a duplicate
          // set
          if (!domain.exists() || (parent == nullptr))
            return result;
        }
        else
        {
          index_nodes[sp] = result;
          index_space_requests.erase(sp);
          if (add_root_reference)
            result->add_base_valid_ref(APPLICATION_REF);
          // If we didn't give it a value add a reference to be removed once
          // the index space node has been set
          if (!domain.exists())
          {
            // Hold the reference on the parent partition to keep both it
            // and the child index space alive if there is a a parent
            if (result->parent != nullptr)
              result->parent->add_base_gc_ref(REGION_TREE_REF);
            else
              result->add_base_gc_ref(REGION_TREE_REF);
          }
          else
            result->set_domain(
                domain, is_ready, take_ownership, false /*broadcast*/,
                true /*initializing*/);
          if (parent != nullptr)
          {
            legion_assert(!add_root_reference);
            // Only do this after we've added all the references
            parent->add_child(result);
          }
          result->register_with_runtime();
          return result;
        }
      }
      legion_assert(domain.exists());
      if (result->set_domain(domain, is_ready, take_ownership))
        std::abort();  // should never hit this
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::create_node(
        IndexSpace sp, IndexPartNode& parent, LegionColor color,
        RtEvent initialized, Provenance* provenance, IndexSpaceExprID expr_id,
        CollectiveMapping* mapping, unsigned depth)
    //--------------------------------------------------------------------------
    {
      IndexSpaceCreator creator(
          sp, &parent, color, expr_id, initialized, depth, provenance, mapping,
          true /*tree valid*/);
      NT_TemplateHelper::demux<IndexSpaceCreator>(sp.get_type_tag(), &creator);
      IndexSpaceNode* result = creator.result;
      legion_assert(result != nullptr);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexSpace, IndexSpaceNode*>::const_iterator it =
            index_nodes.find(sp);
        if (it != index_nodes.end())
        {
          delete result;
          return it->second;
        }
        index_nodes[sp] = result;
        index_space_requests.erase(sp);
        // Add a reference for when we set this index space node
        // Hold the reference on the parent partition to keep both it
        // and the child index space alive
        parent.add_base_gc_ref(REGION_TREE_REF);
        // Only record this with the parent after all the references are added
        parent.add_child(result);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* Runtime::create_node(
        IndexPartition p, IndexSpaceNode* parent, IndexSpaceNode* color_space,
        LegionColor color, bool disjoint, int complete, Provenance* provenance,
        RtEvent initialized, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(
          p, parent, color_space, color, disjoint, complete, initialized,
          mapping, provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode* result = creator.result;
      legion_assert(parent != nullptr);
      legion_assert(result != nullptr);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition, IndexPartNode*>::const_iterator it =
            index_parts.find(p);
        if (it != index_parts.end())
        {
          delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted,
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF);
        parent->add_child(result);
        // Add it to the partition of our parent if it exists, otherwise
        // our parent index space is a root so we add the reference there
        if (parent->parent != nullptr)
          parent->parent->add_nested_valid_ref(result->did);
        else
          parent->add_nested_valid_ref(result->did);
        if (color_space->parent != nullptr)
          color_space->parent->add_nested_valid_ref(result->did);
        else
          color_space->add_nested_valid_ref(result->did);
        // We know if we're disjoint or not but if we're not complete we might
        // still be getting notifications to compute the complete
        if (complete < 0)
          result->initialize_disjoint_complete_notifications();
        else if ((implicit_profiler != nullptr) && result->is_owner())
          implicit_profiler->register_index_partition(
              parent->handle.get_id(), p.get_id(), disjoint, result->color);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartNode* Runtime::create_node(
        IndexPartition p, IndexSpaceNode* parent, IndexSpaceNode* color_space,
        LegionColor color, int complete, Provenance* provenance,
        RtEvent initialized, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartCreator creator(
          p, parent, color_space, color, complete, initialized, mapping,
          provenance);
      NT_TemplateHelper::demux<IndexPartCreator>(p.get_type_tag(), &creator);
      IndexPartNode* result = creator.result;
      legion_assert(parent != nullptr);
      legion_assert(result != nullptr);
      // Check to see if someone else has already made it
      {
        // Hold the lookup lock while modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition, IndexPartNode*>::const_iterator it =
            index_parts.find(p);
        if (it != index_parts.end())
        {
          // Need to remove resource reference if not owner
          delete result;
          return it->second;
        }
        index_parts[p] = result;
        index_part_requests.erase(p);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted,
        if (result->is_owner())
          result->add_base_valid_ref(APPLICATION_REF);
        parent->add_child(result);
        // Add it to the partition of our parent if it exists, otherwise
        // our parent index space is a root so we add the reference there
        if (parent->parent != nullptr)
          parent->parent->add_nested_valid_ref(result->did);
        else
          parent->add_nested_valid_ref(result->did);
        if (color_space->parent != nullptr)
          color_space->parent->add_nested_valid_ref(result->did);
        else
          color_space->add_nested_valid_ref(result->did);
        // We don't know if we're disjonit or yet not so we need to do
        // the disjoint and complete analysis
        result->initialize_disjoint_complete_notifications();
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* Runtime::create_node(
        FieldSpace space, RtEvent initialized, Provenance* provenance,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* result =
          new FieldSpaceNode(space, initialized, mapping, provenance);
      legion_assert(result != nullptr);
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace, FieldSpaceNode*>::const_iterator it =
            field_nodes.find(space);
        if (it != field_nodes.end())
        {
          delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_gc_ref(APPLICATION_REF);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* Runtime::create_node(
        FieldSpace space, RtEvent initialized, Provenance* provenance,
        CollectiveMapping* mapping, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* result =
          new FieldSpaceNode(space, initialized, mapping, provenance, derez);
      legion_assert(result != nullptr);
      // Hold the lookup lock while modifying the lookup table
      {
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace, FieldSpaceNode*>::const_iterator it =
            field_nodes.find(space);
        if (it != field_nodes.end())
        {
          delete result;
          return it->second;
        }
        field_nodes[space] = result;
        field_space_requests.erase(space);
        // If we're the owner add a valid reference that will be removed
        // when we are deleted, otherwise we're remote so we add a gc
        // reference that will be removed by the owner when we can be
        // safely collected
        if (result->is_owner())
          result->add_base_gc_ref(APPLICATION_REF);
        result->register_with_runtime();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* Runtime::create_node(
        LogicalRegion r, PartitionNode* parent, RtEvent initialized,
        DistributedID did, Provenance* provenance, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      if (parent != nullptr)
      {
        legion_assert(r.field_space == parent->handle.field_space);
        legion_assert(r.tree_did == parent->handle.tree_did);
      }
      // Special case for root nodes without dids, we better find them
      if ((parent == nullptr) && (did == 0))
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        // Check to see if it already exists
        std::map<LogicalRegion, RegionNode*>::const_iterator finder =
            region_nodes.find(r);
        legion_assert(finder != region_nodes.end());
        return finder->second;
      }
      RtEvent row_ready, col_ready;
      IndexSpaceNode* row_src = get_node(r.index_space, &row_ready);
      FieldSpaceNode* col_src = get_node(r.field_space, &col_ready);
      if (row_src == nullptr)
      {
        row_ready.wait();
        row_src = get_node(r.index_space);
        row_ready = RtEvent::NO_RT_EVENT;
      }
      if (col_src == nullptr)
      {
        col_ready.wait();
        col_src = get_node(r.field_space);
        col_ready = RtEvent::NO_RT_EVENT;
      }

      if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready);
      RegionNode* result = new RegionNode(
          r, parent, row_src, col_src, did, initialized,
          (parent == nullptr) ? initialized : parent->tree_initialized, mapping,
          provenance);
      legion_assert(result != nullptr);
      // Special case here in case multiple clients attempt to
      // make the node at the same time
      {
        // Hold the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        // Check to see if it already exists
        std::map<LogicalRegion, RegionNode*>::const_iterator it =
            region_nodes.find(r);
        if (it != region_nodes.end())
        {
          // It already exists, delete our copy and return
          // the one that has already been made
          delete result;
          return it->second;
        }
        // Now we can add it to the map
        region_nodes[r] = result;
        // If this is a top level region add it to the collection
        // of top level tree IDs
        if (parent == nullptr)
        {
          legion_assert(tree_nodes.find(r.get_tree_id()) == tree_nodes.end());
          tree_nodes[r.get_tree_id()] = result;
          region_tree_requests.erase(r.get_tree_id());
          // If we're the root we get a valid reference on the owner
          // node otherwise we get a gc ref from the owner node
          if (result->is_owner())
            result->add_base_gc_ref(APPLICATION_REF);
        }
        result->record_registered();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* Runtime::create_node(LogicalPartition p, RegionNode* parent)
    //--------------------------------------------------------------------------
    {
      legion_assert(parent != nullptr);
      legion_assert(p.field_space == parent->handle.field_space);
      legion_assert(p.tree_did == parent->handle.tree_did);
      RtEvent row_ready, col_ready;
      IndexPartNode* row_src = get_node(p.index_partition, &row_ready);
      FieldSpaceNode* col_src = get_node(p.field_space, &col_ready);
      if (row_src == nullptr)
      {
        row_ready.wait();
        row_src = get_node(p.index_partition);
        row_ready = RtEvent::NO_RT_EVENT;
      }
      if (col_src == nullptr)
      {
        col_ready.wait();
        col_src = get_node(p.field_space);
        col_ready = RtEvent::NO_RT_EVENT;
      }
      RtEvent initialized = parent->tree_initialized;
      if (row_ready.exists() || col_ready.exists())
        initialized = Runtime::merge_events(initialized, row_ready, col_ready);
      PartitionNode* result = new PartitionNode(
          p, parent, row_src, col_src, initialized, parent->tree_initialized);
      legion_assert(result != nullptr);
      // Special case here in case multiple clients attempt
      // to make the node at the same time
      {
        // Hole the lookup lock when modifying the lookup table
        AutoLock l_lock(lookup_lock);
        std::map<LogicalPartition, PartitionNode*>::const_iterator it =
            part_nodes.find(p);
        if (it != part_nodes.end())
        {
          // It already exists, delete our copy and
          // return the one that has already been made
          delete result;
          return it->second;
        }
        // Now we can put the node in the map
        part_nodes[p] = result;
        // Add gc ref that will be removed when either the root region node
        // or the index partition node has been destroyed
        result->add_base_gc_ref(REGION_TREE_REF);
        result->record_registered();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceNode* Runtime::get_node(
        IndexSpace space, RtEvent* defer /*=nullptr*/,
        const bool can_fail /*=false*/, const bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for IndexSpace NO_SPACE.";
        error.raise();
      }
      RtEvent wait_on;
      IndexSpaceNode* result = nullptr;
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<IndexSpace, IndexSpaceNode*>::const_iterator finder =
            index_nodes.find(space);
        if (finder != index_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != nullptr) &&
              !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != nullptr)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = IndexSpaceNode::get_owner_space(space);
      if (owner == address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<DistributedID, RtUserEvent>::iterator finder =
              pending_index_spaces.find(space.get_id());
          if (finder != pending_index_spaces.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != nullptr)
          {
            *defer = pending_wait;
            return nullptr;
          }
          else
          {
            pending_wait.wait();
            return get_node(space, defer, false /*can fail*/, false /*first*/);
          }
        }
        else if (can_fail)
          return nullptr;
        else
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find entry for index space " << space.get_id()
                << ".";
          error.raise();
        }
      }
      // Retake the lock and get something to wait on
      {
        AutoLock l_lock(lookup_lock);
        // Check to make sure we didn't loose the race
        std::map<IndexSpace, IndexSpaceNode*>::const_iterator finder =
            index_nodes.find(space);
        if (finder != index_nodes.end())
          return finder->second;
        // Still doesn't exists, see if we sent a request already
        std::map<IndexSpace, RtEvent>::const_iterator wait_finder =
            index_space_requests.find(space);
        if (wait_finder == index_space_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          index_space_requests[space] = done;
          IndexSpaceRequest rez;
          rez.serialize(space);
          rez.serialize(done);
          rez.serialize(address_space);
          rez.dispatch(owner);
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == nullptr)
      {
        // Wait on the event
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexSpace, IndexSpaceNode*>::iterator finder =
              index_nodes.find(space);
          if (finder != index_nodes.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else if (can_fail)
            return nullptr;
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
        {
          Fatal fatal;
          fatal << "Unable to find entry for index space " << space
                << ". This is definitely a runtime bug.";
          fatal.raise();
        }
        wait_on.wait();
        return get_node(space, nullptr, can_fail, false /*first*/);
      }
      else
      {
        *defer = wait_on;
        return nullptr;
      }
    }

    //--------------------------------------------------------------------------
    IndexPartNode* Runtime::get_node(
        IndexPartition part, RtEvent* defer /* = nullptr*/,
        const bool can_fail /* = false*/, const bool first /* = true*/,
        const bool local_only /* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!part.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for IndexPartition NO_PART.";
        error.raise();
      }
      RtEvent wait_on;
      IndexPartNode* result = nullptr;
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<IndexPartition, IndexPartNode*>::const_iterator finder =
            index_parts.find(part);
        if (finder != index_parts.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != nullptr) &&
              !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != nullptr)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexPartNode::get_owner_space(part);
      // If we only want to do the test locally then return the result too
      if ((owner == address_space) || local_only)
      {
        // See if it is in the set of pending partitions in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<DistributedID, RtUserEvent>::iterator finder =
              pending_partitions.find(part.get_id());
          if (finder != pending_partitions.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != nullptr)
          {
            *defer = pending_wait;
            return nullptr;
          }
          else
          {
            pending_wait.wait();
            return get_node(part, defer, false /*can fail*/, false /*first*/);
          }
        }
        else if (can_fail)
          return nullptr;
        else
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find entry for index partition " << part << ".";
          error.raise();
        }
      }
      {
        // Retake the lock in exclusive mode and make
        // sure we didn't loose the race
        AutoLock l_lock(lookup_lock);
        std::map<IndexPartition, IndexPartNode*>::const_iterator finder =
            index_parts.find(part);
        if (finder != index_parts.end())
          return finder->second;
        // See if we've already sent the request or not
        std::map<IndexPartition, RtEvent>::const_iterator wait_finder =
            index_part_requests.find(part);
        if (wait_finder == index_part_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          index_part_requests[part] = done;
          IndexPartitionRequest rez;
          rez.serialize(part);
          rez.serialize(done);
          rez.serialize(address_space);
          rez.dispatch(owner);
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == nullptr)
      {
        // Wait for the event
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock);
          std::map<IndexPartition, IndexPartNode*>::iterator finder =
              index_parts.find(part);
          if (finder != index_parts.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else if (can_fail)
            return nullptr;
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
        {
          Fatal fatal;
          fatal << "Unable to find entry for index partition " << part << ". "
                << "This is definitely a runtime bug.";
          fatal.raise();
        }
        wait_on.wait();
        return get_node(part, nullptr, can_fail, false /*first*/);
      }
      else
      {
        *defer = wait_on;
        return nullptr;
      }
    }

    //--------------------------------------------------------------------------
    FieldSpaceNode* Runtime::get_node(
        FieldSpace space, RtEvent* defer /*=nullptr*/, bool can_fail /*=false*/,
        bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (!space.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for FieldSpace NO_SPACE.";
        error.raise();
      }
      RtEvent wait_on;
      FieldSpaceNode* result = nullptr;
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<FieldSpace, FieldSpaceNode*>::const_iterator finder =
            field_nodes.find(space);
        if (finder != field_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          if ((defer != nullptr) &&
              !finder->second->initialized.has_triggered())
          {
            *defer = finder->second->initialized;
            return finder->second;
          }
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != nullptr)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = FieldSpaceNode::get_owner_space(space);
      if (owner == address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<DistributedID, RtUserEvent>::iterator finder =
              pending_field_spaces.find(space.get_id());
          if (finder != pending_field_spaces.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          if (defer != nullptr)
          {
            *defer = pending_wait;
            return nullptr;
          }
          else
          {
            pending_wait.wait();
            return get_node(space, defer, can_fail, false /*first*/);
          }
        }
        else if (can_fail)
          return nullptr;
        else
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find entry for field space " << space << ".";
          error.raise();
        }
      }
      {
        // Retake the lock in exclusive mode and
        // check to make sure we didn't loose the race
        AutoLock l_lock(lookup_lock);
        std::map<FieldSpace, FieldSpaceNode*>::const_iterator finder =
            field_nodes.find(space);
        if (finder != field_nodes.end())
          return finder->second;
        // Now see if we've already sent a request
        std::map<FieldSpace, RtEvent>::const_iterator wait_finder =
            field_space_requests.find(space);
        if (wait_finder == field_space_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          field_space_requests[space] = done;
          FieldSpaceRequest rez;
          rez.serialize(space);
          rez.serialize(done);
          rez.serialize(address_space);
          rez.dispatch(owner);
          wait_on = done;
        }
        else
          wait_on = wait_finder->second;
      }
      if (defer == nullptr)
      {
        // Wait for the event to be ready
        wait_on.wait();
        {
          AutoLock l_lock(lookup_lock, false /*exclusive*/);
          std::map<FieldSpace, FieldSpaceNode*>::const_iterator finder =
              field_nodes.find(space);
          if (finder != field_nodes.end())
          {
            if (finder->second->initialized.exists())
            {
              if (finder->second->initialized.has_triggered())
              {
                finder->second->initialized = RtEvent::NO_RT_EVENT;
                return finder->second;
              }
              else
                wait_on = finder->second->initialized;
            }
            else
              return finder->second;
          }
          else if (can_fail)
            return nullptr;
          else
            wait_on = RtEvent::NO_RT_EVENT;
        }
        if (!wait_on.exists())
        {
          Fatal fatal;
          fatal << "Unable to find entry for field space " << space << ". "
                << "This is definitely a runtime bug.";
          fatal.raise();
        }
        wait_on.wait();
        return get_node(space, nullptr, can_fail, false /*first*/);
      }
      else
      {
        *defer = wait_on;
        return nullptr;
      }
    }

    //--------------------------------------------------------------------------
    RegionNode* Runtime::get_node(
        LogicalRegion handle, bool need_check /* = true*/,
        bool can_fail /* = false*/, bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for LogicalRegion NO_REGION.";
        error.raise();
      }
      // Check to see if the node already exists
      RtEvent wait_on;
      RegionNode* result = nullptr;
      bool has_top_level_region = false;
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<LogicalRegion, RegionNode*>::const_iterator finder =
            region_nodes.find(handle);
        if (finder != region_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          wait_on = finder->second->initialized;
          result = finder->second;
        }
        // Check to see if we have the top level region
        else if (need_check)
          has_top_level_region =
              (tree_nodes.find(handle.get_tree_id()) != tree_nodes.end());
        else
          has_top_level_region = true;
      }
      if (result != nullptr)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // If we don't have the top-level region, we need to request it before
      // we go crawling up the tree so we know where to stop
      if (!has_top_level_region)
      {
        RegionNode* root = get_tree(handle.get_tree_id(), true /*can fail*/);
        if (root == nullptr)
        {
          if (can_fail)
            return nullptr;
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find logical region " << handle << ".";
          error.raise();
        }
        else if (root->handle == handle)
          return root;
      }
      // Otherwise it hasn't been made yet, so make it
      IndexSpaceNode* index_node =
          get_node(handle.index_space, nullptr, true /*can fail*/);
      if (index_node == nullptr)
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find logical region " << handle << ".";
        error.raise();
      }
      if (index_node->parent != nullptr)
      {
        legion_assert(index_node->parent != nullptr);
        LogicalPartition parent_handle(
            handle.tree_did, index_node->parent->handle, handle.field_space);
        // Note this request can recursively build more nodes, but we
        // are guaranteed that the top level node exists
        PartitionNode* parent = get_node(parent_handle, false /*need check*/);
        // Now make our node and then return it
        result = create_node(handle, parent, RtEvent::NO_RT_EVENT, 0 /*did*/);
      }
      else
      {
        // This better be a root node, if it's not then something requested
        // that we construct a logical reigon node after the parent partition
        // was destroyed which is very bad
        legion_assert(index_node->depth == 0);
        // Even though this is a root node, we'll discover it's already made
        result = create_node(handle, nullptr, RtEvent::NO_RT_EVENT, 0 /*did*/);
      }
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        if (!result->initialized.exists())
          return result;
        wait_on = result->initialized;
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock l_lock(lookup_lock);
      result->initialized = RtEvent::NO_RT_EVENT;
      return result;
    }

    //--------------------------------------------------------------------------
    PartitionNode* Runtime::get_node(
        LogicalPartition handle, bool need_check /* = true*/,
        bool can_fail /* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for LogicalPartition NO_PART.";
        error.raise();
      }
      RtEvent wait_on;
      PartitionNode* result = nullptr;
      // Check to see if the node already exists
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<LogicalPartition, PartitionNode*>::const_iterator it =
            part_nodes.find(handle);
        if (it != part_nodes.end())
        {
          if (it->second->initialized.exists())
          {
            wait_on = it->second->initialized;
            result = it->second;
          }
          else
            return it->second;
        }
      }
      if (result != nullptr)
      {
        if (!wait_on.has_triggered())
          wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Otherwise it hasn't been made yet so make it
      IndexPartNode* index_node =
          get_node(handle.index_partition, nullptr, true /*can fail*/);
      if (index_node == nullptr)
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find logical partition " << handle << ".";
        error.raise();
      }
      legion_assert(index_node->parent != nullptr);
      LogicalRegion parent_handle(
          handle.tree_did, index_node->parent->handle, handle.field_space);
      // Note this request can recursively build more nodes, but we
      // are guaranteed that the top level node exists
      RegionNode* parent =
          get_node(parent_handle, need_check, true /*can fail*/);
      if (parent == nullptr)
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find logical partition " << handle << ".";
        error.raise();
      }
      // Now create our node and return it
      result = create_node(handle, parent);
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        if (!result->initialized.exists())
          return result;
        wait_on = result->initialized;
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
      AutoLock l_lock(lookup_lock);
      result->initialized = RtEvent::NO_RT_EVENT;
      return result;
    }

    //--------------------------------------------------------------------------
    RegionNode* Runtime::get_tree(
        RegionTreeID tid, bool can_fail, bool first /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (tid == 0)
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Invalid request for tree ID 0 which is never a tree ID.";
        error.raise();
      }
      RtEvent wait_on;
      RegionNode* result = nullptr;
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<RegionTreeID, RegionNode*>::const_iterator finder =
            tree_nodes.find(tid);
        if (finder != tree_nodes.end())
        {
          if (!finder->second->initialized.exists())
            return finder->second;
          wait_on = finder->second->initialized;
          result = finder->second;
        }
      }
      if (result != nullptr)
      {
        wait_on.wait();
        AutoLock l_lock(lookup_lock);
        result->initialized = RtEvent::NO_RT_EVENT;
        return result;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpaceID owner = RegionTreeNode::get_owner_space(tid);
      if (owner == runtime->address_space)
      {
        // See if it is in the set of pending spaces in which case we
        // can wait for it to be recorded
        RtEvent pending_wait;
        if (first)
        {
          AutoLock l_lock(lookup_lock);
          std::map<RegionTreeID, RtUserEvent>::iterator finder =
              pending_region_trees.find(tid);
          if (finder != pending_region_trees.end())
          {
            if (!finder->second.exists())
              finder->second = Runtime::create_rt_user_event();
            pending_wait = finder->second;
          }
        }
        if (pending_wait.exists())
        {
          pending_wait.wait();
          return get_tree(tid, can_fail, false /*first*/);
        }
        else if (can_fail)
          return nullptr;
        else
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Unable to find entry for region tree ID " << tid << ".";
          error.raise();
        }
      }
      {
        // Retake the lock in exclusive mode and check to
        // make sure that we didn't lose the race
        AutoLock l_lock(lookup_lock);
        std::map<RegionTreeID, RegionNode*>::const_iterator finder =
            tree_nodes.find(tid);
        if (finder != tree_nodes.end())
          return finder->second;
        // Now see if we've already send a request
        std::map<RegionTreeID, RtEvent>::const_iterator req_finder =
            region_tree_requests.find(tid);
        if (req_finder == region_tree_requests.end())
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          region_tree_requests[tid] = done;
          TopLevelRegionRequest rez;
          rez.serialize(tid);
          rez.serialize(done);
          rez.serialize(runtime->address_space);
          rez.dispatch(owner);
          wait_on = done;
        }
        else
          wait_on = req_finder->second;
      }
      wait_on.wait();
      AutoLock l_lock(lookup_lock, false /*exclusive*/);
      std::map<RegionTreeID, RegionNode*>::const_iterator finder =
          tree_nodes.find(tid);
      if (finder == tree_nodes.end())
      {
        if (can_fail)
          return nullptr;
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find top-level tree entry for region tree " << tid
              << ". This is either a runtime bug or requires Legion fences if "
              << "names are being returned out of the context in which they "
              << "are being created.";
        error.raise();
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    RtEvent Runtime::find_or_request_node(
        IndexSpace space, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock l_lock(lookup_lock, false /*exclusive*/);
        std::map<IndexSpace, IndexSpaceNode*>::const_iterator finder =
            index_nodes.find(space);
        if (finder != index_nodes.end())
          return RtEvent::NO_RT_EVENT;
      }
      // Couldn't find it, so send a request to the owner node
      AddressSpace owner = IndexSpaceNode::get_owner_space(space);
      if (owner == address_space)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find entry for index space " << space << ".";
        error.raise();
      }
      AutoLock l_lock(lookup_lock);
      // Check to make sure we didn't loose the race
      std::map<IndexSpace, IndexSpaceNode*>::const_iterator finder =
          index_nodes.find(space);
      if (finder != index_nodes.end())
        return RtEvent::NO_RT_EVENT;
      // Still doesn't exists, see if we sent a request already
      std::map<IndexSpace, RtEvent>::const_iterator wait_finder =
          index_space_requests.find(space);
      if (wait_finder == index_space_requests.end())
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        index_space_requests[space] = done;
        IndexSpaceRequest rez;
        rez.serialize(space);
        rez.serialize(done);
        rez.serialize(address_space);
        rez.dispatch(target);
        return done;
      }
      else
        return wait_finder->second;
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_node(IndexSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      std::map<IndexSpace, IndexSpaceNode*>::iterator finder =
          index_nodes.find(space);
      legion_assert(finder != index_nodes.end());
      index_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_node(IndexPartition part)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      legion_assert(
          index_part_requests.find(part) == index_part_requests.end());
      std::map<IndexPartition, IndexPartNode*>::iterator finder =
          index_parts.find(part);
      legion_assert(finder != index_parts.end());
      index_parts.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_node(FieldSpace space)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      std::map<FieldSpace, FieldSpaceNode*>::iterator finder =
          field_nodes.find(space);
      legion_assert(finder != field_nodes.end());
      field_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_node(LogicalRegion handle, bool top)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      if (top)
      {
        std::map<RegionTreeID, RegionNode*>::iterator finder =
            tree_nodes.find(handle.get_tree_id());
        legion_assert(finder != tree_nodes.end());
        tree_nodes.erase(finder);
      }
      std::map<LogicalRegion, RegionNode*>::iterator finder =
          region_nodes.find(handle);
      legion_assert(finder != region_nodes.end());
      region_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_node(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_lock);
      std::map<LogicalPartition, PartitionNode*>::iterator finder =
          part_nodes.find(handle);
      legion_assert(finder != part_nodes.end());
      part_nodes.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::record_pending_index_space(DistributedID space)
    //--------------------------------------------------------------------------
    {
      // We should be the owner for this space
      legion_assert(
          (LEGION_DISTRIBUTED_ID_FILTER(space) % total_address_spaces) ==
          address_space);
      AutoLock l_lock(lookup_lock);
      legion_assert(
          pending_index_spaces.find(space) == pending_index_spaces.end());
      pending_index_spaces[space] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void Runtime::record_pending_partition(DistributedID pid)
    //--------------------------------------------------------------------------
    {
      // We should be the owner for this space
      legion_assert(
          (LEGION_DISTRIBUTED_ID_FILTER(pid) % total_address_spaces) ==
          address_space);
      AutoLock l_lock(lookup_lock);
      legion_assert(pending_partitions.find(pid) == pending_partitions.end());
      pending_partitions[pid] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void Runtime::record_pending_field_space(DistributedID space)
    //--------------------------------------------------------------------------
    {
      // We should be the owner for this space
      legion_assert(
          (LEGION_DISTRIBUTED_ID_FILTER(space) % total_address_spaces) ==
          address_space);
      AutoLock l_lock(lookup_lock);
      legion_assert(
          pending_field_spaces.find(space) == pending_field_spaces.end());
      pending_field_spaces[space] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void Runtime::record_pending_region_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // We should be the owner for this space
      legion_assert(
          (LEGION_DISTRIBUTED_ID_FILTER(tid) % total_address_spaces) ==
          address_space);
      AutoLock l_lock(lookup_lock);
      legion_assert(
          pending_region_trees.find(tid) == pending_region_trees.end());
      pending_region_trees[tid] = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void Runtime::revoke_pending_index_space(DistributedID space)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<DistributedID, RtUserEvent>::iterator finder =
            pending_index_spaces.find(space);
        legion_assert(finder != pending_index_spaces.end());
        to_trigger = finder->second;
        pending_index_spaces.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::revoke_pending_partition(DistributedID pid)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<DistributedID, RtUserEvent>::iterator finder =
            pending_partitions.find(pid);
        legion_assert(finder != pending_partitions.end());
        to_trigger = finder->second;
        pending_partitions.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::revoke_pending_field_space(DistributedID space)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<DistributedID, RtUserEvent>::iterator finder =
            pending_field_spaces.find(space);
        legion_assert(finder != pending_field_spaces.end());
        to_trigger = finder->second;
        pending_field_spaces.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void Runtime::revoke_pending_region_tree(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      {
        AutoLock l_lock(lookup_lock);
        std::map<RegionTreeID, RtUserEvent>::iterator finder =
            pending_region_trees.find(tid);
        legion_assert(finder != pending_region_trees.end());
        to_trigger = finder->second;
        pending_region_trees.erase(finder);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_top_level_index_space(IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      return (get_node(handle)->parent == nullptr);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_top_level_region(LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      return (get_node(handle)->parent == nullptr);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_subregion(LogicalRegion child, LogicalRegion parent)
    //--------------------------------------------------------------------------
    {
      if (child == parent)
        return true;
      if (child.get_tree_id() != parent.get_tree_id())
        return false;
      return has_index_path(parent.get_index_space(), child.get_index_space());
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_subregion(LogicalRegion child, LogicalPartition parent)
    //--------------------------------------------------------------------------
    {
      if (child.get_tree_id() != parent.get_tree_id())
        return false;
      RegionNode* child_node = get_node(child);
      PartitionNode* parent_node = get_node(parent);
      while (child_node->parent != nullptr)
      {
        if (child_node->parent == parent_node)
          return true;
        child_node = child_node->parent->parent;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_disjoint(IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* node = get_node(handle);
      return node->is_disjoint(true /*app query*/);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_disjoint(LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      return is_disjoint(handle.get_index_partition());
    }

    //--------------------------------------------------------------------------
    bool Runtime::are_disjoint(IndexSpace one, IndexSpace two)
    //--------------------------------------------------------------------------
    {
      if (one == two)
        return false;
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      // See if they intersect with each other
      IndexSpaceNode* sp_one = get_node(one);
      IndexSpaceNode* sp_two = get_node(two);
      return !sp_one->intersects_with(sp_two);
    }

    //--------------------------------------------------------------------------
    bool Runtime::are_disjoint(IndexSpace one, IndexPartition two)
    //--------------------------------------------------------------------------
    {
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      IndexSpaceNode* space_node = get_node(one);
      IndexPartNode* part_node = get_node(two);
      return !space_node->intersects_with(part_node);
    }

    //--------------------------------------------------------------------------
    bool Runtime::are_disjoint(IndexPartition one, IndexPartition two)
    //--------------------------------------------------------------------------
    {
      if (one == two)
        return false;
      if (one.get_tree_id() != two.get_tree_id())
        return true;
      IndexPartNode* part_one = get_node(one);
      IndexPartNode* part_two = get_node(two);
      return !part_one->intersects_with(part_two);
    }

    //--------------------------------------------------------------------------
    bool Runtime::are_disjoint_tree_only(IndexTreeNode* one, IndexTreeNode* two)
    //--------------------------------------------------------------------------
    {
      if (one == two)
        return false;
      // Bring them to the same depth
      while (one->depth < two->depth) two = two->get_parent();
      while (two->depth < one->depth) one = one->get_parent();
      legion_assert(one->depth == two->depth);
      // Test again
      if (one == two)
        return false;
      // Same depth, not the same node
      IndexTreeNode* parent_one = one->get_parent();
      IndexTreeNode* parent_two = two->get_parent();
      while (parent_one != parent_two)
      {
        one = parent_one;
        parent_one = one->get_parent();
        two = parent_two;
        parent_two = two->get_parent();
      }
      legion_assert(parent_one == parent_two);
      legion_assert(one != two);  // can't be the same child
      // Now we have the common ancestor, see if the two children are disjoint
      if (parent_one->is_index_space_node())
      {
        if (parent_one->as_index_space_node()->are_disjoint(
                one->color, two->color))
          return true;
      }
      else
      {
        if (parent_one->as_index_part_node()->are_disjoint(
                one->color, two->color))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::check_types(TypeTag t1, TypeTag t2, bool& diff_dims)
    //--------------------------------------------------------------------------
    {
      if (t1 == t2)
        return true;
      const int d1 = NT_TemplateHelper::get_dim(t1);
      const int d2 = NT_TemplateHelper::get_dim(t2);
      diff_dims = (d1 != d2);
      return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_dominated(IndexSpace src, IndexSpace dst)
    //--------------------------------------------------------------------------
    {
      // Check to see if dst is dominated by source
      legion_assert(src.get_type_tag() == dst.get_type_tag());
      IndexSpaceNode* src_node = get_node(src);
      IndexSpaceNode* dst_node = get_node(dst);
      return src_node->dominates(dst_node);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_dominated_tree_only(
        IndexSpace test, IndexPartition dominator)
    //--------------------------------------------------------------------------
    {
      legion_assert(test.get_tree_id() == dominator.get_tree_id());
      IndexSpaceNode* node = get_node(test);
      IndexPartNode* const dom = get_node(dominator);
      while (node->depth > (dom->depth + 1))
      {
        legion_assert(node->parent != nullptr);
        node = node->parent->parent;
      }
      if (node->parent == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_dominated_tree_only(
        IndexPartition test, IndexSpace dominator)
    //--------------------------------------------------------------------------
    {
      legion_assert(test.get_tree_id() == dominator.get_tree_id());
      IndexPartNode* node = get_node(test);
      IndexSpaceNode* const dom = get_node(dominator);
      while (node->depth > (dom->depth + 1))
      {
        legion_assert(node->parent != nullptr);
        node = node->parent->parent;
      }
      if (node->parent == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_dominated_tree_only(
        IndexPartition test, IndexPartition dominator)
    //--------------------------------------------------------------------------
    {
      legion_assert(test.get_tree_id() == dominator.get_tree_id());
      IndexPartNode* node = get_node(test);
      IndexPartNode* const dom = get_node(dominator);
      while (node->depth > dom->depth)
      {
        legion_assert(node->parent != nullptr);
        node = node->parent->parent;
      }
      if (node == dom)
        return true;
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_index_path(IndexSpace parent, IndexSpace child)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* child_node = get_node(child);
      if (parent == child)
        return true;  // Early out
      IndexSpaceNode* parent_node = get_node(parent);
      while (parent_node != child_node)
      {
        if (parent_node->depth >= child_node->depth)
          return false;
        if (child_node->parent == nullptr)
          return false;
        child_node = child_node->parent->parent;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool Runtime::has_partition_path(IndexSpace parent, IndexPartition child)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* child_node = get_node(child);
      if (child_node->parent == nullptr)
        return false;
      return has_index_path(parent, child_node->parent->handle);
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_projection_depth(
        LogicalRegion result, LogicalRegion upper)
    //--------------------------------------------------------------------------
    {
      RegionNode* start = get_node(result);
      RegionNode* finish = get_node(upper);
      unsigned depth = 0;
      while (start != finish)
      {
        legion_assert(start->get_depth() > finish->get_depth());
        start = start->parent->parent;
        depth++;
      }
      return depth;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_projection_depth(
        LogicalRegion result, LogicalPartition upper)
    //--------------------------------------------------------------------------
    {
      RegionNode* start = get_node(result);
      legion_assert(start->parent != nullptr);
      PartitionNode* finish = get_node(upper);
      unsigned depth = 0;
      while (start->parent != finish)
      {
        legion_assert(start->parent->get_depth() > finish->get_depth());
        start = start->parent->parent;
        depth++;
        legion_assert(start->parent != nullptr);
      }
      return depth;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::union_index_spaces(
        IndexSpaceExpression* lhs, IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs->type_tag == rhs->type_tag);
      legion_assert(lhs->is_valid());
      legion_assert(rhs->is_valid());
      if (lhs == rhs)
      {
        lhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(lhs);
        return lhs;
      }
      if (lhs->is_empty())
      {
        rhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(rhs);
        return rhs;
      }
      if (rhs->is_empty())
      {
        lhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(lhs);
        return lhs;
      }
      IndexSpaceExpression* result = lhs->inline_union(rhs);
      if (result == nullptr)
      {
        IndexSpaceExpression* lhs_canon = lhs->get_canonical_expression();
        IndexSpaceExpression* rhs_canon = rhs->get_canonical_expression();
        if (lhs_canon == rhs_canon)
          return lhs;
        std::vector<IndexSpaceExpression*> exprs(2);
        if (compare_expressions(lhs_canon, rhs_canon))
        {
          exprs[0] = lhs_canon;
          exprs[1] = rhs_canon;
        }
        else
        {
          exprs[0] = rhs_canon;
          exprs[1] = lhs_canon;
        }
        result = union_index_spaces(exprs);
      }
      else
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::union_index_spaces(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      legion_assert(!exprs.empty());
#ifdef LEGION_DEBUG
      for (SetView<IndexSpaceExpression*>::const_iterator it = exprs.begin();
           it != exprs.end(); it++)
        legion_assert((*it)->is_valid());
#endif
      if (exprs.size() == 1)
      {
        IndexSpaceExpression* result = *(exprs.begin());
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      IndexSpaceExpression* result = (*exprs.begin())->inline_union(exprs);
      if (result != nullptr)
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      std::vector<IndexSpaceExpression*> expressions;
      expressions.reserve(exprs.size());
      for (SetView<IndexSpaceExpression*>::const_iterator it = exprs.begin();
           it != exprs.end(); it++)
      {
        // Remove any empty expressions on the way in
        if (!(*it)->is_empty())
          expressions.emplace_back((*it)->get_canonical_expression());
      }
      if (expressions.empty())
      {
        IndexSpaceExpression* result = *(exprs.begin());
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      if (expressions.size() == 1)
      {
        IndexSpaceExpression* result = expressions.back();
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      // sort them in order by their IDs
      std::sort(expressions.begin(), expressions.end(), compare_expressions);
      // remove duplicates
      std::vector<IndexSpaceExpression*>::iterator last =
          std::unique(expressions.begin(), expressions.end());
      if (last != expressions.end())
      {
        expressions.erase(last, expressions.end());
        legion_assert(!expressions.empty());
        if (expressions.size() == 1)
        {
          IndexSpaceExpression* result = expressions.back();
          result->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(result);
          return expressions.back();
        }
      }
      bool first_pass = true;
      // this helps make sure we don't overflow our stack
      while (expressions.size() > MAX_EXPRESSION_FANOUT)
      {
        std::vector<IndexSpaceExpression*> next_expressions;
        while (!expressions.empty())
        {
          if (expressions.size() > 1)
          {
            std::vector<IndexSpaceExpression*> temp_expressions;
            temp_expressions.reserve(MAX_EXPRESSION_FANOUT);
            // Pop up to 32 expressions off the back
            for (unsigned idx = 0; idx < MAX_EXPRESSION_FANOUT; idx++)
            {
              temp_expressions.emplace_back(expressions.back());
              expressions.pop_back();
              if (expressions.empty())
                break;
            }
            IndexSpaceExpression* expr = union_index_spaces(temp_expressions);
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.emplace_back(expr);
          }
          else
          {
            IndexSpaceExpression* expr = expressions.back();
            expressions.pop_back();
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.emplace_back(expr);
          }
        }
        if (!first_pass)
        {
          // Remove the expression references on the previous set
          for (IndexSpaceExpression* expr : expressions)
            if (expr->remove_base_expression_reference(REGION_TREE_REF))
              delete expr;
        }
        else
          first_pass = false;
        expressions.swap(next_expressions);
        // canonicalize and uniquify them all again
        std::set<IndexSpaceExpression*, CompareExpressions> unique_expressions;
        for (unsigned idx = 0; idx < expressions.size(); idx++)
        {
          IndexSpaceExpression* expr = expressions[idx];
          IndexSpaceExpression* unique = expr->get_canonical_expression();
          if (unique_expressions.insert(unique).second)
            unique->add_base_expression_reference(REGION_TREE_REF);
        }
        // Remove the expression references
        for (IndexSpaceExpression* expr : expressions)
          if (expr->remove_base_expression_reference(REGION_TREE_REF))
            delete expr;
        if (unique_expressions.size() == 1)
        {
          IndexSpaceExpression* result = *(unique_expressions.begin());
          if (exprs.find(result) == exprs.end())
          {
            result->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(result);
          }
          // Remove the extra expression reference we added
          if (result->remove_base_expression_reference(REGION_TREE_REF))
            std::abort();  // should never hit this
          return result;
        }
        expressions.resize(unique_expressions.size());
        unsigned index = 0;
        for (IndexSpaceExpression* unique_expr : unique_expressions)
          expressions[index++] = unique_expr;
      }
      result = union_index_spaces(expressions);
      result->add_base_expression_reference(LIVE_EXPR_REF);
      ImplicitReferenceTracker::record_live_expression(result);
      if (!first_pass)
      {
        // Remove the extra references on the expression vector we added
        for (IndexSpaceExpression* expr : expressions)
          if (expr->remove_base_expression_reference(REGION_TREE_REF))
            delete expr;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::union_index_spaces(
        const std::vector<IndexSpaceExpression*>& expressions,
        OperationCreator* creator /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(expressions.size() >= 2);
      legion_assert(expressions.size() <= MAX_EXPRESSION_FANOUT);
      IndexSpaceExpression* first = expressions[0];
      const IndexSpaceExprID key = first->expr_id;
      // See if we can find it in read-only mode
      {
        AutoLock l_lock(lookup_is_op_lock, false /*exclusive*/);
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            union_ops.find(key);
        if (finder != union_ops.end())
        {
          IndexSpaceExpression* result = nullptr;
          ExpressionTrieNode* next = nullptr;
          if (finder->second->find_operation(expressions, result, next) &&
              result->try_add_live_reference())
            return result;
          if (creator == nullptr)
          {
            UnionOpCreator union_creator(first->type_tag, expressions);
            return next->find_or_create_operation(expressions, union_creator);
          }
          else
            return next->find_or_create_operation(expressions, *creator);
        }
      }
      ExpressionTrieNode* node = nullptr;
      if (creator == nullptr)
      {
        UnionOpCreator union_creator(first->type_tag, expressions);
        // Didn't find it, retake the lock, see if we lost the race
        // and if no make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            union_ops.find(key);
        if (finder == union_ops.end())
        {
          // Didn't lose the race, so make the node
          node = new ExpressionTrieNode(0 /*depth*/, first->expr_id);
          union_ops[key] = node;
        }
        else
          node = finder->second;
        legion_assert(node != nullptr);
        return node->find_or_create_operation(expressions, union_creator);
      }
      else
      {
        // Didn't find it, retake the lock, see if we lost the race
        // and if no make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            union_ops.find(key);
        if (finder == union_ops.end())
        {
          // Didn't lose the race, so make the node
          node = new ExpressionTrieNode(0 /*depth*/, first->expr_id);
          union_ops[key] = node;
        }
        else
          node = finder->second;
        legion_assert(node != nullptr);
        return node->find_or_create_operation(expressions, *creator);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::intersect_index_spaces(
        IndexSpaceExpression* lhs, IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs->type_tag == rhs->type_tag);
      legion_assert(lhs->is_valid());
      legion_assert(rhs->is_valid());
      if (lhs == rhs)
      {
        lhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(lhs);
        return lhs;
      }
      if (lhs->is_empty())
      {
        lhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(lhs);
        return lhs;
      }
      if (rhs->is_empty())
      {
        rhs->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(rhs);
        return rhs;
      }
      IndexSpaceExpression* result = lhs->inline_intersection(rhs);
      if (result == nullptr)
      {
        IndexSpaceExpression* lhs_canon = lhs->get_canonical_expression();
        IndexSpaceExpression* rhs_canon = rhs->get_canonical_expression();
        if (lhs_canon == rhs_canon)
          return lhs;
        std::vector<IndexSpaceExpression*> exprs(2);
        if (compare_expressions(lhs_canon, rhs_canon))
        {
          exprs[0] = lhs_canon;
          exprs[1] = rhs_canon;
        }
        else
        {
          exprs[0] = rhs_canon;
          exprs[1] = lhs_canon;
        }
        result = intersect_index_spaces(exprs);
      }
      else
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::intersect_index_spaces(
        const SetView<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      legion_assert(!exprs.empty());
#ifdef LEGION_DEBUG
      for (SetView<IndexSpaceExpression*>::const_iterator it = exprs.begin();
           it != exprs.end(); it++)
        legion_assert((*it)->is_valid());
#endif
      if (exprs.size() == 1)
      {
        IndexSpaceExpression* result = *(exprs.begin());
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      IndexSpaceExpression* result =
          (*exprs.begin())->inline_intersection(exprs);
      if (result != nullptr)
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      std::vector<IndexSpaceExpression*> expressions(
          exprs.begin(), exprs.end());
      // Do a quick pass to see if any of them are empty in which case we
      // know that the result of the whole intersection is empty
      for (unsigned idx = 0; idx < expressions.size(); idx++)
      {
        IndexSpaceExpression*& expr = expressions[idx];
        if (expr->is_empty())
        {
          expr->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(expr);
          return expr;
        }
        expr = expr->get_canonical_expression();
      }
      // sort them in order by their IDs
      std::sort(expressions.begin(), expressions.end(), compare_expressions);
      // remove duplicates
      std::vector<IndexSpaceExpression*>::iterator last =
          std::unique(expressions.begin(), expressions.end());
      if (last != expressions.end())
      {
        expressions.erase(last, expressions.end());
        legion_assert(!expressions.empty());
        if (expressions.size() == 1)
        {
          IndexSpaceExpression* result = expressions.back();
          result->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(result);
          return result;
        }
      }
      bool first_pass = true;
      // this helps make sure we don't overflow our stack
      while (expressions.size() > MAX_EXPRESSION_FANOUT)
      {
        std::vector<IndexSpaceExpression*> next_expressions;
        while (!expressions.empty())
        {
          if (expressions.size() > 1)
          {
            std::vector<IndexSpaceExpression*> temp_expressions;
            temp_expressions.reserve(MAX_EXPRESSION_FANOUT);
            // Pop up to 32 expressions off the back
            for (unsigned idx = 0; idx < MAX_EXPRESSION_FANOUT; idx++)
            {
              temp_expressions.emplace_back(expressions.back());
              expressions.pop_back();
              if (expressions.empty())
                break;
            }
            IndexSpaceExpression* expr =
                intersect_index_spaces(temp_expressions);
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.emplace_back(expr);
          }
          else
          {
            IndexSpaceExpression* expr = expressions.back();
            expressions.pop_back();
            expr->add_base_expression_reference(REGION_TREE_REF);
            next_expressions.emplace_back(expr);
          }
        }
        if (!first_pass)
        {
          // Remove the expression references on the previous set
          for (IndexSpaceExpression* expr : expressions)
            if (expr->remove_base_expression_reference(REGION_TREE_REF))
              delete expr;
        }
        else
          first_pass = false;
        expressions.swap(next_expressions);
        // canonicalize and uniquify them all again
        std::set<IndexSpaceExpression*, CompareExpressions> unique_expressions;
        for (unsigned idx = 0; idx < expressions.size(); idx++)
        {
          IndexSpaceExpression* expr = expressions[idx];
          IndexSpaceExpression* unique = expr->get_canonical_expression();
          if (unique->is_empty())
          {
            // Add a reference to the unique expression
            unique->add_base_expression_reference(LIVE_EXPR_REF);
            ImplicitReferenceTracker::record_live_expression(unique);
            // Remove references on all the things we no longer need
            for (IndexSpaceExpression* unique_expr : unique_expressions)
              if (unique_expr->remove_base_expression_reference(
                      REGION_TREE_REF))
                delete unique_expr;
            for (IndexSpaceExpression* expr : expressions)
              if (expr->remove_base_expression_reference(REGION_TREE_REF))
                delete expr;
            return unique;
          }
          if (unique_expressions.insert(unique).second)
            unique->add_base_expression_reference(REGION_TREE_REF);
        }
        // Remove the expression references
        for (IndexSpaceExpression* expr : expressions)
          if (expr->remove_base_expression_reference(REGION_TREE_REF))
            delete expr;
        if (unique_expressions.size() == 1)
        {
          IndexSpaceExpression* result = *(unique_expressions.begin());
          result->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(result);
          // Remove the extra expression reference we added
          if (result->remove_base_expression_reference(REGION_TREE_REF))
            std::abort();  // should never hit this
          return result;
        }
        expressions.resize(unique_expressions.size());
        unsigned index = 0;
        for (IndexSpaceExpression* unique_expr : unique_expressions)
          expressions[index++] = unique_expr;
      }
      result = intersect_index_spaces(expressions);
      result->add_base_expression_reference(LIVE_EXPR_REF);
      ImplicitReferenceTracker::record_live_expression(result);
      if (!first_pass)
      {
        // Remove the extra references on the expression vector we added
        for (IndexSpaceExpression* expr : expressions)
          if (expr->remove_base_expression_reference(REGION_TREE_REF))
            delete expr;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::intersect_index_spaces(
        const std::vector<IndexSpaceExpression*>& expressions,
        OperationCreator* creator /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(expressions.size() >= 2);
      legion_assert(expressions.size() <= MAX_EXPRESSION_FANOUT);
      IndexSpaceExpression* first = expressions[0];
      const IndexSpaceExprID key = first->expr_id;
      // See if we can find it in read-only mode
      {
        AutoLock l_lock(lookup_is_op_lock, false /*exclusive*/);
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            intersection_ops.find(key);
        if (finder != intersection_ops.end())
        {
          IndexSpaceExpression* result = nullptr;
          ExpressionTrieNode* next = nullptr;
          if (finder->second->find_operation(expressions, result, next) &&
              result->try_add_live_reference())
            return result;
          if (creator == nullptr)
          {
            IntersectionOpCreator inter_creator(first->type_tag, expressions);
            return next->find_or_create_operation(expressions, inter_creator);
          }
          else
            return next->find_or_create_operation(expressions, *creator);
        }
      }
      ExpressionTrieNode* node = nullptr;
      if (creator == nullptr)
      {
        IntersectionOpCreator inter_creator(first->type_tag, expressions);
        // Didn't find it, retake the lock, see if we lost the race
        // and if not make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        // See if we lost the race
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            intersection_ops.find(key);
        if (finder == intersection_ops.end())
        {
          // Didn't lose the race so make the node
          node = new ExpressionTrieNode(0 /*depth*/, first->expr_id);
          intersection_ops[key] = node;
        }
        else
          node = finder->second;
        legion_assert(node != nullptr);
        return node->find_or_create_operation(expressions, inter_creator);
      }
      else
      {
        // Didn't find it, retake the lock, see if we lost the race
        // and if not make the actual trie node
        AutoLock l_lock(lookup_is_op_lock);
        // See if we lost the race
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            intersection_ops.find(key);
        if (finder == intersection_ops.end())
        {
          // Didn't lose the race so make the node
          node = new ExpressionTrieNode(0 /*depth*/, first->expr_id);
          intersection_ops[key] = node;
        }
        else
          node = finder->second;
        legion_assert(node != nullptr);
        return node->find_or_create_operation(expressions, *creator);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::subtract_index_spaces(
        IndexSpaceExpression* lhs, IndexSpaceExpression* rhs,
        OperationCreator* creator /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs->type_tag == rhs->type_tag);
      legion_assert(lhs->is_valid());
      legion_assert(rhs->is_valid());
      // Handle a few easy cases
      if (creator == nullptr)
      {
        if (lhs->is_empty())
        {
          lhs->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(lhs);
          return lhs;
        }
        if (rhs->is_empty())
        {
          rhs->add_base_expression_reference(LIVE_EXPR_REF);
          ImplicitReferenceTracker::record_live_expression(rhs);
          return lhs;
        }
      }
      IndexSpaceExpression* result = lhs->inline_subtraction(rhs);
      if (result != nullptr)
      {
        result->add_base_expression_reference(LIVE_EXPR_REF);
        ImplicitReferenceTracker::record_live_expression(result);
        return result;
      }
      std::vector<IndexSpaceExpression*> expressions(2);
      expressions[0] = lhs->get_canonical_expression();
      expressions[1] = rhs->get_canonical_expression();
      const IndexSpaceExprID key = expressions[0]->expr_id;
      // See if we can find it in read-only mode
      {
        AutoLock l_lock(lookup_is_op_lock, false /*exclusive*/);
        std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator finder =
            difference_ops.find(key);
        if (finder != difference_ops.end())
        {
          IndexSpaceExpression* expr = nullptr;
          ExpressionTrieNode* next = nullptr;
          if (finder->second->find_operation(expressions, expr, next) &&
              expr->try_add_live_reference())
            result = expr;
          if (result == nullptr)
          {
            if (creator == nullptr)
            {
              DifferenceOpCreator diff_creator(
                  lhs->type_tag, expressions[0], expressions[1]);
              result =
                  next->find_or_create_operation(expressions, diff_creator);
            }
            else
              result = next->find_or_create_operation(expressions, *creator);
          }
        }
      }
      if (result == nullptr)
      {
        ExpressionTrieNode* node = nullptr;
        if (creator == nullptr)
        {
          DifferenceOpCreator diff_creator(
              lhs->type_tag, expressions[0], expressions[1]);
          // Didn't find it, retake the lock, see if we lost the race
          // and if not make the actual trie node
          AutoLock l_lock(lookup_is_op_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = difference_ops.find(key);
          if (finder == difference_ops.end())
          {
            // Didn't lose the race so make the node
            node = new ExpressionTrieNode(0 /*depth*/, expressions[0]->expr_id);
            difference_ops[key] = node;
          }
          else
            node = finder->second;
          legion_assert(node != nullptr);
          result = node->find_or_create_operation(expressions, diff_creator);
        }
        else
        {
          // Didn't find it, retake the lock, see if we lost the race
          // and if not make the actual trie node
          AutoLock l_lock(lookup_is_op_lock);
          // See if we lost the race
          std::map<IndexSpaceExprID, ExpressionTrieNode*>::const_iterator
              finder = difference_ops.find(key);
          if (finder == difference_ops.end())
          {
            // Didn't lose the race so make the node
            node = new ExpressionTrieNode(0 /*depth*/, expressions[0]->expr_id);
            difference_ops[key] = node;
          }
          else
            node = finder->second;
          legion_assert(node != nullptr);
          result = node->find_or_create_operation(expressions, *creator);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::find_canonical_expression(
        IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      // we'll hash expressions based on the number of dimensions and points
      // to try to get an early separation for them for testing congruence
      if (expr->is_empty())
        return expr;
      const uint64_t hash_key = expr->get_canonical_hash();
      AutoLock c_lock(congruence_lock);
      return expr->find_congruent_expression(canonical_expressions[hash_key]);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_canonical_expression(IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for empty expressions
      if (expr->is_empty())
        return;
      const uint64_t hash_key = expr->get_canonical_hash();
      AutoLock c_lock(congruence_lock);
      std::unordered_map<uint64_t, CanonicalSet>::iterator finder =
          canonical_expressions.find(hash_key);
      legion_assert(finder != canonical_expressions.end());
      legion_no_skip_assert(finder->second.erase(expr));
      if (finder->second.empty())
        canonical_expressions.erase(finder);
    }

    //--------------------------------------------------------------------------
    void Runtime::record_empty_expression(IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_is_op_lock);
      empty_expressions.emplace_back(expr);
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_union_operation(
        IndexSpaceOperation* op,
        const std::vector<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      legion_assert(op->op_kind == IndexSpaceOperation::UNION_OP_KIND);
      const IndexSpaceExprID key = exprs[0]->expr_id;
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID, ExpressionTrieNode*>::iterator finder =
          union_ops.find(key);
      legion_assert(finder != union_ops.end());
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        union_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_intersection_operation(
        IndexSpaceOperation* op,
        const std::vector<IndexSpaceExpression*>& exprs)
    //--------------------------------------------------------------------------
    {
      legion_assert(op->op_kind == IndexSpaceOperation::INTERSECT_OP_KIND);
      const IndexSpaceExprID key(exprs[0]->expr_id);
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID, ExpressionTrieNode*>::iterator finder =
          intersection_ops.find(key);
      legion_assert(finder != intersection_ops.end());
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        intersection_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::remove_subtraction_operation(
        IndexSpaceOperation* op, IndexSpaceExpression* lhs,
        IndexSpaceExpression* rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(op->op_kind == IndexSpaceOperation::DIFFERENCE_OP_KIND);
      const IndexSpaceExprID key = lhs->expr_id;
      std::vector<IndexSpaceExpression*> exprs(2);
      exprs[0] = lhs;
      exprs[1] = rhs;
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID, ExpressionTrieNode*>::iterator finder =
          difference_ops.find(key);
      legion_assert(finder != difference_ops.end());
      if (finder->second->remove_operation(exprs))
      {
        delete finder->second;
        difference_ops.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* Runtime::find_or_create_remote_expression(
        IndexSpaceExprID remote_expr_id, Deserializer& derez, bool& created)
    //--------------------------------------------------------------------------
    {
      // See if we can find it with the read-only lock first
      {
        AutoLock l_lock(lookup_is_op_lock, false /*exclusive*/);
        std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator
            finder = remote_expressions.find(remote_expr_id);
        if (finder != remote_expressions.end())
        {
          created = false;
          finder->second->skip_unpack_expression(derez);
          return finder->second;
        }
      }
      // Take the lock in exclusive mode and see if we lost the race
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID, IndexSpaceExpression*>::const_iterator finder =
          remote_expressions.find(remote_expr_id);
      if (finder != remote_expressions.end())
      {
        created = false;
        finder->second->skip_unpack_expression(derez);
        return finder->second;
      }
      // If we didn't lose the lock then we can make the instance
      created = true;
      TypeTag type_tag;
      derez.deserialize(type_tag);
      RemoteExpressionCreator creator(remote_expr_id, type_tag, derez);
      NT_TemplateHelper::demux<RemoteExpressionCreator>(type_tag, &creator);
      legion_assert(creator.operation != nullptr);
      remote_expressions[remote_expr_id] = creator.operation;
      return creator.operation;
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_remote_expression(IndexSpaceExprID remote_expr_id)
    //--------------------------------------------------------------------------
    {
      AutoLock l_lock(lookup_is_op_lock);
      std::map<IndexSpaceExprID, IndexSpaceExpression*>::iterator finder =
          remote_expressions.find(remote_expr_id);
      if (finder != remote_expressions.end())
        remote_expressions.erase(finder);
    }

    //--------------------------------------------------------------------------
    ContextID Runtime::allocate_region_tree_context(void)
    //--------------------------------------------------------------------------
    {
      // Try getting something off the list of available contexts
      AutoLock ctx_lock(context_lock);
      if (available_contexts.empty())
      {
        // Double the number of available contexts
        available_contexts.resize(total_contexts);
        total_contexts *= 2;
        for (unsigned idx = 0; idx < available_contexts.size(); idx++)
          available_contexts[idx] = total_contexts - (idx + 1);
        // Tell all the processor managers about the additional contexts
        for (const std::pair<const Processor, ProcessorManager*>& it :
             proc_managers)
          it.second->update_max_context_count(total_contexts);
      }
      ContextID result = available_contexts.back();
      available_contexts.pop_back();
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::free_region_tree_context(ContextID context)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      check_region_tree_context(context);
#endif
      AutoLock ctx_lock(context_lock);
      available_contexts.emplace_back(context);
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_local(Processor proc) const
    //--------------------------------------------------------------------------
    {
      legion_assert(proc.exists());
      return (proc.address_space() == address_space);
    }

    //--------------------------------------------------------------------------
    ProcessorManager* Runtime::find_processor_manager(Processor proc) const
    //--------------------------------------------------------------------------
    {
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      legion_assert(finder != proc_managers.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    bool Runtime::is_visible_memory(Processor proc, Memory memory)
    //--------------------------------------------------------------------------
    {
      // If we cached it locally for our processors, then just go
      // ahead and get the result
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      if (finder != proc_managers.end())
        return finder->second->is_visible_memory(memory);
      // Otherwise look up the result
      Machine::MemoryQuery visible_memories(machine);
      // Have to handle the case where this is a processor group
      if (proc.kind() == Processor::PROC_GROUP)
      {
        std::vector<Processor> group_members;
        proc.get_group_members(group_members);
        for (const Processor& member : group_members)
          visible_memories.has_affinity_to(member);
      }
      else
        visible_memories.has_affinity_to(proc);
      for (Machine::MemoryQuery::iterator it = visible_memories.begin();
           it != visible_memories.end(); it++)
        if ((*it) == memory)
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void Runtime::find_visible_memories(
        Processor proc, std::set<Memory>& visible)
    //--------------------------------------------------------------------------
    {
      // If we cached it locally for our processors, then just go
      // ahead and get the result
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      if (finder != proc_managers.end())
      {
        finder->second->find_visible_memories(visible);
        return;
      }
      // Otherwise look up the result
      Machine::MemoryQuery visible_memories(machine);
      // Have to handle the case where this is a processor group
      if (proc.kind() == Processor::PROC_GROUP)
      {
        std::vector<Processor> group_members;
        proc.get_group_members(group_members);
        for (const Processor& member : group_members)
          visible_memories.has_affinity_to(member);
      }
      else
        visible_memories.has_affinity_to(proc);
      for (Machine::MemoryQuery::iterator it = visible_memories.begin();
           it != visible_memories.end(); it++)
        visible.insert(*it);
    }

    //--------------------------------------------------------------------------
    Memory Runtime::find_local_memory(Processor proc, Memory::Kind mem_kind)
    //--------------------------------------------------------------------------
    {
      if ((mem_kind == Memory::SYSTEM_MEM) &&
          (proc.address_space() == address_space))
        return runtime_system_memory;
      // Check to see if this is a local processor in which case
      // we should be able to do this much faster
      std::map<Processor, ProcessorManager*>::const_iterator finder =
          proc_managers.find(proc);
      if (finder != proc_managers.end())
        return finder->second->find_best_visible_memory(mem_kind);
      // Otherwise look up the result
      Machine::MemoryQuery visible_memories(machine);
      // Must be of the right kind
      visible_memories.only_kind(mem_kind);
      // Must not be empty
      visible_memories.has_capacity(1 /*at least one byte*/);
      // Have to handle the case where this is a processor group
      if (proc.kind() == Processor::PROC_GROUP)
      {
        std::vector<Processor> group_members;
        proc.get_group_members(group_members);
        for (const Processor& member : group_members)
          visible_memories.best_affinity_to(member);
      }
      else
        visible_memories.best_affinity_to(proc);
      if (visible_memories.count() == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << proc.kind() << " processor " << proc << " has no " << mem_kind
              << " memory.";
        error.raise();
      }
      return visible_memories.first();
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_unique_index_space_id(void)
    //--------------------------------------------------------------------------
    {
      const DistributedID did = get_available_distributed_id();
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_SPACE_NODE_DC);
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_unique_index_partition_id(void)
    //--------------------------------------------------------------------------
    {
      const DistributedID did = get_available_distributed_id();
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, INDEX_PART_NODE_DC);
    }

    //--------------------------------------------------------------------------
    DistributedID Runtime::get_unique_field_space_id(void)
    //--------------------------------------------------------------------------
    {
      const DistributedID did = get_available_distributed_id();
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, FIELD_SPACE_DC);
    }

    //--------------------------------------------------------------------------
    IndexTreeID Runtime::get_unique_index_tree_id(void)
    //--------------------------------------------------------------------------
    {
      IndexTreeID result = unique_index_tree_id.fetch_add(runtime_stride);
      // check for overflow
      // If we have overflow on the number of region trees
      // created then we are really in a bad place.
      legion_assert(result <= unique_index_tree_id);
      return result;
    }

    //--------------------------------------------------------------------------
    RegionTreeID Runtime::get_unique_region_tree_id(void)
    //--------------------------------------------------------------------------
    {
      RegionTreeID did = get_available_distributed_id();
      return LEGION_DISTRIBUTED_HELP_ENCODE(did, REGION_TREE_NODE_DC);
    }

    //--------------------------------------------------------------------------
    UniqueID Runtime::get_unique_operation_id(void)
    //--------------------------------------------------------------------------
    {
      UniqueID result = unique_operation_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_operation_id);
      return result;
    }

    //--------------------------------------------------------------------------
    FieldID Runtime::get_unique_field_id(void)
    //--------------------------------------------------------------------------
    {
      FieldID result = unique_field_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_field_id);
      return result;
    }

    //--------------------------------------------------------------------------
    CodeDescriptorID Runtime::get_unique_code_descriptor_id(void)
    //--------------------------------------------------------------------------
    {
      CodeDescriptorID result =
          unique_code_descriptor_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_code_descriptor_id);
      return result;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID Runtime::get_unique_constraint_id(void)
    //--------------------------------------------------------------------------
    {
      LayoutConstraintID result =
          unique_constraint_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_constraint_id);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpaceExprID Runtime::get_unique_index_space_expr_id(void)
    //--------------------------------------------------------------------------
    {
      IndexSpaceExprID result = unique_is_expr_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_is_expr_id);
      return result;
    }

    //--------------------------------------------------------------------------
    uint64_t Runtime::get_unique_top_level_task_id(void)
    //--------------------------------------------------------------------------
    {
      uint64_t result = unique_top_level_task_id.fetch_add(runtime_stride);
      legion_assert(result < unique_top_level_task_id);
      return result;
    }

    //--------------------------------------------------------------------------
    uint64_t Runtime::get_unique_implicit_top_level_task_id(void)
    //--------------------------------------------------------------------------
    {
      // These count the same across all the nodes and don't need to be
      // atomic since it's up to the caller to guard againt concurrency here
      uint64_t result = unique_implicit_top_level_task_id++;
      legion_assert(result < unique_implicit_top_level_task_id);
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::get_unique_indirections_id(void)
    //--------------------------------------------------------------------------
    {
      unsigned result = unique_indirections_id.fetch_add(runtime_stride);
      // check for overflow
      legion_assert(result <= unique_indirections_id);
      return result;
    }

    //--------------------------------------------------------------------------
    Provenance* Runtime::find_or_create_provenance(
        const char* prov, size_t size)
    //--------------------------------------------------------------------------
    {
      if ((prov == nullptr) || (size == 0))
        return nullptr;
      // Compute the hash, use the murmur hasher since we can guarantee that it
      // is deterministic across the processes which is not true of std::hash
      Murmur3Hasher hasher;
      hasher.hash(prov, size);
      uint64_t hashes[2];
      hasher.finalize(hashes);
      const uint64_t hash = hashes[0] ^ hashes[1];
      // Check to see if we can find it in read-only mode first
      {
        AutoLock prov_lock(provenance_lock, false /*exclusive*/);
        std::unordered_map<uint64_t, Provenance*>::const_iterator finder =
            provenances.find(hash);
        if (finder != provenances.end())
        {
          finder->second->add_reference();
          return finder->second;
        }
      }
      // Retake the lock in exclusive mode
      AutoLock prov_lock(provenance_lock);
      // Check to make sure we didn't lose the race
      std::unordered_map<uint64_t, Provenance*>::const_iterator finder =
          provenances.find(hash);
      if (finder != provenances.end())
      {
        finder->second->add_reference();
        return finder->second;
      }
      // Generate a new provenance object
      Provenance* result = new Provenance(hash, prov, size);
      result->add_reference(2);  // one for ourself and one for the caller
      provenances.emplace(std::make_pair(hash, result));
      // If we have a profiler, then record this provenance
      if (profiler != nullptr)
        profiler->record_provenance(hash, prov, size);
      return result;
    }

    //--------------------------------------------------------------------------
    Provenance* Runtime::find_provenance(ProvenanceID pid)
    //--------------------------------------------------------------------------
    {
      AutoLock prov_lock(provenance_lock, false /*exclusive*/);
      std::unordered_map<uint64_t, Provenance*>::const_iterator finder =
          provenances.find(pid);
      if (finder == provenances.end())
        return nullptr;
      finder->second->add_reference();
      return finder->second;
    }

    //--------------------------------------------------------------------------
    IndexSpace Runtime::help_create_index_space_handle(TypeTag type_tag)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle(
          get_unique_index_space_id(), get_unique_index_tree_id(), type_tag);
      return handle;
    }

    //--------------------------------------------------------------------------
    unsigned Runtime::generate_random_integer(void)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(random_lock);
      unsigned result = nrand48(random_state);
      return result;
    }

#ifdef LEGION_TRACE_ALLOCATION
    //--------------------------------------------------------------------------
    void Runtime::trace_allocation(
        const std::type_info& info, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (prepared_for_shutdown)
        return;
      const std::size_t hash = info.hash_code();
      AutoLock a_lock(allocation_lock);
      std::unordered_map<std::size_t, AllocationTracker>::iterator finder =
          allocation_manager.find(hash);
      if (finder == allocation_manager.end())
        finder =
            allocation_manager
                .emplace(std::make_pair(hash, AllocationTracker(info.name())))
                .first;
      size_t alloc_size = size * elems;
      finder->second.total_allocations += elems;
      finder->second.total_bytes += alloc_size;
      finder->second.diff_allocations += elems;
      finder->second.diff_bytes += alloc_size;
    }

    //--------------------------------------------------------------------------
    void Runtime::trace_free(const std::type_info& info, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (prepared_for_shutdown)
        return;
      const std::size_t hash = info.hash_code();
      AutoLock a_lock(allocation_lock);
      std::unordered_map<std::size_t, AllocationTracker>::iterator finder =
          allocation_manager.find(hash);
      size_t free_size = size * elems;
      finder->second.total_allocations -= elems;
      finder->second.total_bytes -= free_size;
      finder->second.diff_allocations -= elems;
      finder->second.diff_bytes -= free_size;
    }

    //--------------------------------------------------------------------------
    void Runtime::dump_allocation_info(void)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(allocation_lock);
      for (const std::pair<const std::size_t, AllocationTracker>& alloc_pair :
           allocation_manager)
      {
        // Skip anything that is empty
        if (alloc_pair.second.total_allocations == 0)
          continue;
        // Skip anything that hasn't changed
        if (alloc_pair.second.diff_allocations == 0)
          continue;
        log_allocation.info(
            "%s on %d: "
            "total=%d total_bytes=%ld diff=%d diff_bytes=%lld",
            alloc_pair.second.name, address_space,
            alloc_pair.second.total_allocations, alloc_pair.second.total_bytes,
            alloc_pair.second.diff_allocations,
            (long long int)alloc_pair.second.diff_bytes);
        alloc_pair.second.diff_allocations = 0;
        alloc_pair.second.diff_bytes = 0;
      }
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);
      log_allocation.info("RSS: %ld", usage.ru_maxrss);
      log_allocation.info(" ");
    }
#endif  // TRACE_ALLOCATION

    //--------------------------------------------------------------------------
    LayoutConstraintID Runtime::register_layout(
        const LayoutConstraintRegistrar& registrar,
        LayoutConstraintID layout_id, DistributedID did,
        CollectiveMapping* collective_mapping)
    //--------------------------------------------------------------------------
    {
      if (layout_id == LEGION_AUTO_GENERATE_ID)
        layout_id = get_unique_constraint_id();
      // Now make our entry and then return the result
      LayoutConstraints* constraints = new LayoutConstraints(
          layout_id, registrar, false /*internal*/, did, collective_mapping);
      if (!register_layout(constraints))
        // If someone else already registered this ID then we delete our object
        delete constraints;
      return layout_id;
    }

    //--------------------------------------------------------------------------
    LayoutConstraints* Runtime::register_layout(
        FieldSpace handle, const LayoutConstraintSet& cons, bool internal)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = new LayoutConstraints(
          get_unique_constraint_id(), cons, handle, internal);
      register_layout(constraints);
      return constraints;
    }

    //--------------------------------------------------------------------------
    bool Runtime::register_layout(LayoutConstraints* new_constraints)
    //--------------------------------------------------------------------------
    {
      new_constraints->add_base_resource_ref(RUNTIME_REF);
      // If we're not internal and we're the owner then we also
      // add an application reference to prevent early collection
      if (!new_constraints->internal && new_constraints->is_owner())
        new_constraints->add_base_gc_ref(APPLICATION_REF);
      AutoLock l_lock(layout_constraints_lock);
      std::map<LayoutConstraintID, LayoutConstraints*>::const_iterator finder =
          layout_constraints_table.find(new_constraints->layout_id);
      if (finder != layout_constraints_table.end())
        return false;
      layout_constraints_table[new_constraints->layout_id] = new_constraints;
      // Remove any pending requests
      pending_constraint_requests.erase(new_constraints->layout_id);
      // Now we can do the registration with the runtime
      new_constraints->register_with_runtime();
      return true;
    }

    //--------------------------------------------------------------------------
    void Runtime::release_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = find_layout_constraints(layout_id);
      legion_assert(!constraints->internal);
      // Check to see if this is the owner
      if (constraints->is_owner())
      {
        if (constraints->remove_base_gc_ref(APPLICATION_REF))
          delete constraints;
      }
      else
      {
        // Send a message to the owner asking it to do the release
        ConstraintRelease rez;
        {
          RezCheck z(rez);
          rez.serialize(layout_id);
        }
        rez.dispatch(constraints->owner_space);
      }
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_layout(LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = nullptr;
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID, LayoutConstraints*>::iterator finder =
            layout_constraints_table.find(layout_id);
        if (finder != layout_constraints_table.end())
        {
          constraints = finder->second;
          layout_constraints_table.erase(finder);
        }
      }
      if ((constraints != nullptr) &&
          constraints->remove_base_resource_ref(RUNTIME_REF))
        delete (constraints);
    }

    //--------------------------------------------------------------------------
    /*static*/ LayoutConstraintID Runtime::preregister_layout(
        const LayoutConstraintRegistrar& registrar,
        LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_layout' after the runtime has "
                 "started.";
        error.raise();
      }
      std::map<LayoutConstraintID, LayoutConstraintRegistrar>&
          pending_constraints = get_pending_constraint_table();
      // See if we have to generate an ID
      if (layout_id == LEGION_AUTO_GENERATE_ID)
      {
        // Find the first available layout ID
        if (!pending_constraints.empty())
        {
          std::map<LayoutConstraintID, LayoutConstraintRegistrar>::
              const_reverse_iterator finder = pending_constraints.crbegin();
          if (finder->first <= LEGION_MAX_APPLICATION_LAYOUT_ID)
            layout_id = LEGION_MAX_APPLICATION_LAYOUT_ID + 1;
          else
            layout_id = finder->first + 1;
        }
        else
          layout_id = LEGION_MAX_APPLICATION_LAYOUT_ID + 1;
      }
      else
      {
        if (layout_id == 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal use of reserved constraint ID 0";
          error.raise();
        }
        else if (LEGION_MAX_APPLICATION_LAYOUT_ID < layout_id)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Illegal application-provided layout constraint ID "
                << layout_id
                << " which exceeds the LEGION_MAX_APPLICATION_LAYOUT_ID of "
                << LEGION_MAX_APPLICATION_LAYOUT_ID
                << " configured in legion_config.h.";
          error.raise();
        }
        // Check to make sure it is not already used
        std::map<LayoutConstraintID, LayoutConstraintRegistrar>::const_iterator
            finder = pending_constraints.find(layout_id);
        if (finder != pending_constraints.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Duplicate use of constraint ID " << layout_id;
          error.raise();
        }
      }
      pending_constraints[layout_id] = registrar;
      return layout_id;
    }

    //--------------------------------------------------------------------------
    FieldSpace Runtime::get_layout_constraint_field_space(
        LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = find_layout_constraints(layout_id);
      return constraints->get_field_space();
    }

    //--------------------------------------------------------------------------
    void Runtime::get_layout_constraints(
        LayoutConstraintID layout_id, LayoutConstraintSet& layout_constraints)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = find_layout_constraints(layout_id);
      layout_constraints = *constraints;
    }

    //--------------------------------------------------------------------------
    const char* Runtime::get_layout_constraints_name(
        LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      LayoutConstraints* constraints = find_layout_constraints(layout_id);
      return constraints->get_name();
    }

    //--------------------------------------------------------------------------
    LayoutConstraints* Runtime::find_layout_constraints(
        LayoutConstraintID layout_id, bool can_fail /*= false*/,
        RtEvent* wait_for /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      // See if we can find it first
      RtEvent wait_on;
      {
        AutoLock l_lock(layout_constraints_lock);
        std::map<LayoutConstraintID, LayoutConstraints*>::const_iterator
            finder = layout_constraints_table.find(layout_id);
        if (finder != layout_constraints_table.end())
        {
          return finder->second;
        }
        else
        {
          // See if a request has already been issued
          std::map<LayoutConstraintID, RtEvent>::const_iterator wait_on_finder =
              pending_constraint_requests.find(layout_id);
          if (can_fail || (wait_on_finder == pending_constraint_requests.end()))
          {
            // Ask for the constraints
            AddressSpaceID target =
                LayoutConstraints::get_owner_space(layout_id);
            if (target == address_space)
            {
              if (can_fail)
                return nullptr;
              Error error(LEGION_INTERFACE_EXCEPTION);
              error << "Unable to find layout constraint " << layout_id;
              error.raise();
            }
            RtUserEvent to_trigger = Runtime::create_rt_user_event();
            ConstraintRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(layout_id);
              rez.serialize(to_trigger);
              rez.serialize(can_fail);
            }
            // Send the message
            rez.dispatch(target);
            // Only save the event to wait on if this can't fail
            if (!can_fail)
              pending_constraint_requests[layout_id] = to_trigger;
            wait_on = to_trigger;
          }
          else
            wait_on = wait_on_finder->second;
        }
      }
      // If we want the wait event, just return
      if (wait_for != nullptr)
      {
        *wait_for = wait_on;
        return nullptr;
      }
      // If we didn't find it send a remote request for the constraints
      wait_on.wait();
      // When we wake up, the result should be there
      AutoLock l_lock(layout_constraints_lock);
      std::map<LayoutConstraintID, LayoutConstraints*>::const_iterator finder =
          layout_constraints_table.find(layout_id);
      if (finder == layout_constraints_table.end())
      {
        if (can_fail)
          return nullptr;
        legion_assert(finder != layout_constraints_table.end());
      }
      return finder->second;
    }

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::start(
        int argc, char** argv, bool background, bool supply_default_mapper,
        bool filter, bool init_network, bool configure_machine)
    //--------------------------------------------------------------------------
    {
      // Some static asserts that need to hold true for the runtime to work
      static_assert(
          LEGION_MAX_RETURN_SIZE > 0,
          "Need a positive and non-zero value for LEGION_MAX_RETURN_SIZE");
      static_assert(
          (1 << LEGION_FIELD_LOG2) == LEGION_MAX_FIELDS,
          "LEGION_MAX_FIELDS must be a pwoer of 2");
      static_assert(
          LEGION_MAX_NUM_NODES > 0,
          "Need a positive and non-zero value for LEGION_MAX_NUM_NODES");
      static_assert(
          LEGION_MAX_NUM_PROCS > 0,
          "Need a positive and non-zero value for LEGION_MAX_NUM_PROCS");
      static_assert(
          LEGION_DEFAULT_MAX_TASK_WINDOW > 0,
          "Need a positive and non-zero value for "
          "LEGION_DEFAULT_MAX_TASK_WINDOW");
      static_assert(
          LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE > 0,
          "Need a positive and non-zero value for "
          "LEGION_DEFAULT_MIN_TASKS_TO_SCHEDULE");
      static_assert(
          LEGION_DEFAULT_MAX_MESSAGE_SIZE > 0,
          "Need a positive and non-zero value for "
          "LEGION_DEFAULT_MAX_MESSAGE_SIZE");
#ifdef LEGION_SPY
      static_assert(
          Realm::Logger::REALM_LOGGING_MIN_LEVEL <= Realm::Logger::LEVEL_PRINT,
          "Legion Spy requires a COMPILE_TIME_MIN_LEVEL of at most "
          "LEVEL_PRINT.");
#endif
#ifdef LEGION_GC
      static_assert(
          Realm::Logger::REALM_LOGGING_MIN_LEVEL <= Realm::Logger::LEVEL_INFO,
          "Legion GC requires a COMPILE_TIME_MIN_LEVEL of at most LEVEL_INFO.");
#endif
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      static_assert(
          Realm::Logger::REALM_LOGGING_MIN_LEVEL <= Realm::Logger::LEVEL_INFO,
          "LEGION_DEBUG_SHUTDOWN_HANG requires a COMPILE_TIME_MIN_LEVEL "
          "of at most LEVEL_INFO.");
#endif
      // Register builtin reduction operators
      register_builtin_reduction_operators();
      // Need to pass argc and argv to low-level runtime before we can record
      // their values as they might be changed by GASNet or MPI or whatever.
      // Note that the logger isn't initialized until after this call returns
      // which means any logging that occurs before this has undefined behavior.
      const LegionConfiguration& config = initialize(
          &argc, &argv, !runtime_cmdline_parsed, filter, init_network,
          configure_machine);
      RealmRuntime realm = RealmRuntime::get_runtime();
      // Finish configuring the machine so we can start querying the machine
      // model and setting up our data structures before we start Realm
      if (configure_machine)
        realm.finish_configure();

      // Perform any waits that the user requested before starting
      if (config.delay_start > 0)
        sleep(config.delay_start);
      // Check for any slow configurations
      if (!config.slow_config_ok)
        perform_slow_config_checks(config);
      // Configure legion spy if necessary
      LegionSpy::log_legion_spy_config();
      // Construct our runtime objects
      std::set<Processor> local_procs;
      const Processor first_proc = configure_runtime(
          argc, argv, config, realm, local_procs, background,
          supply_default_mapper);
      // Startup kind should be a CPU or a Utility processor
      legion_assert(
          (first_proc.kind() == Processor::LOC_PROC) ||
          (first_proc.kind() == Processor::UTIL_PROC));
      // First processor should be on node zero
      legion_assert(first_proc.address_space() == 0);
      legion_assert(!local_procs.empty());
      // Configure MPI Interoperability
      const std::vector<LegionHandshake>& pending_handshakes =
          get_pending_handshake_table();
      if ((mpi_rank >= 0) || (!pending_handshakes.empty()))
        configure_interoperability();
      // We have to set these prior to starting Realm as once we start
      // Realm it might fork child processes so they all need to see
      // the same values for these static variables
      runtime_started = true;
      runtime_backgrounded = background;

      // Now that we have everything setup we can tell Realm to
      // start the processors. It is at this point which fork
      // can be called to spawn subprocesses.
      realm.start();

      Realm::Barrier startup_barrier = Realm::Barrier::NO_BARRIER;
      if (runtime->total_address_spaces > 1)
      {
        // First we do a collective spawn to make sure that Realm is
        // started and all of our meta-tasks have been registered
        // across all of the nodes
        // Very important, do NOT pass in any event preconditions to
        // this task and do not use the postcondition as it comes from
        // node zero and we don't want all the nodes to subscribe to
        // node zero unnecessarily.
        realm.collective_spawn(first_proc, LG_STARTUP_TASK_ID, nullptr, 0);
        // Now get the start-up barrier that will be set by the
        // start-up task as it broadcasts through the nodes
        startup_barrier = find_or_wait_for_startup_barrier();
      }
      // We also need to run a nop task on every processor to make sure
      // that Realm has finished initializing that processor. This is
      // especially important for things like Python processors which
      // might still be loading modules and we want to ensure that they
      // are completely done doing that before we try to do anything
      std::vector<RtEvent> nop_events;
      nop_events.reserve(local_procs.size());
      for (const Processor& local_proc : local_procs)
        nop_events.emplace_back(RtEvent(
            local_proc.spawn(Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0)));
      // Now we can initialize the Legion runtime(s) on this node
      TopLevelContext* top_context = runtime->initialize_runtime(first_proc);
      if (startup_barrier.exists())
      {
        // Make sure all the nodes are done
        startup_barrier.arrive(1 /*count*/, Runtime::merge_events(nop_events));
        // Wait for all the nodes to be done with the initialization
        startup_barrier.wait();
      }
      else
      {
        const RtEvent initialized = Runtime::merge_events(nop_events);
        initialized.wait();
      }

      // Launch the top-level task if we have a main set
      if (runtime->address_space == 0)
      {
        if (legion_main_set)
        {
          legion_assert(top_context != nullptr);
          TaskLauncher launcher(
              Runtime::legion_main_id,
              UntypedBuffer(&runtime->input_args, sizeof(InputArgs)),
              Predicate::TRUE_PRED, legion_main_mapper_id);
          runtime->launch_top_level_task(launcher, top_context);
        }
        // Cleanup the start-up barrier
        if (startup_barrier.exists())
          startup_barrier.destroy_barrier();
      }
      // If we are supposed to background this thread, then we wait
      // for the runtime to shutdown, otherwise we can now return
      if (background)
        return 0;
      // Decrement the total outstanding top-level tasks to reflect that
      // this node is ready to shutdown when everything is done
      runtime->decrement_outstanding_top_level_tasks();
      // Wait for Realm shutdown to be complete
      return realm.wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    bool Runtime::LegionConfiguration::parse_bool(
        const std::string& parameter, const std::string_view& flag, bool& value,
        bool polarity)
    //--------------------------------------------------------------------------
    {
      if (parameter.compare(flag) != 0)
        return false;
      value = polarity;
      return true;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool Runtime::LegionConfiguration::parse_int(
        std::vector<std::string>::const_iterator it,
        std::vector<std::string>::const_iterator end,
        const std::string_view& flag, T& value, bool& bad)
    //--------------------------------------------------------------------------
    {
      if (it->compare(flag) != 0)
        return false;
      it = std::next(it);
      if (it != end)
      {
        const int result = std::atoi(it->c_str());
        if (result == 0)
        {
          // Check to see if the string contains only zeros
          for (unsigned idx = 0; idx < it->size(); idx++)
          {
            if (it->at(idx) == '0')
              continue;
            bad = true;
            return true;
          }
          value = result;
        }
        else if (std::is_unsigned<T>::value && (result < 0))
          bad = true;
        else
          value = result;
      }
      else
        bad = true;
      return true;
    }

    //--------------------------------------------------------------------------
    bool Runtime::LegionConfiguration::parse_string(
        std::vector<std::string>::const_iterator it,
        std::vector<std::string>::const_iterator end,
        const std::string_view& flag, std::string& value, bool& bad)
    //--------------------------------------------------------------------------
    {
      if (it->compare(flag) != 0)
        return false;
      it = std::next(it);
      if (it != end)
        value = *it;
      else
        bad = true;
      return true;
    }

    //--------------------------------------------------------------------------
    size_t Runtime::LegionConfiguration::parse_option(
        std::vector<std::string>::const_iterator it,
        std::vector<std::string>::const_iterator end, bool& bad_parameter)
    //--------------------------------------------------------------------------
    {
      if (parse_bool(*it, "-lg:warn_backtrace", warnings_backtrace) ||
          parse_bool(*it, "-lg:warn", runtime_warnings) ||
          parse_bool(*it, "-lg:werror", warnings_are_errors) ||
          parse_bool(*it, "-lg:leaks", report_leaks) ||
          parse_bool(*it, "-lg:registration", record_registration) ||
          parse_bool(*it, "-lg:nosteal", stealing_disabled) ||
          parse_bool(*it, "-lg:nosteal", stealing_disabled) ||
          parse_bool(*it, "-lg:resilient", resilient_mode) ||
          parse_bool(*it, "-lg:unsafe_launch", unsafe_launch) ||
          parse_bool(*it, "-lg:unsafe_mapper", unsafe_mapper) ||
          parse_bool(*it, "-lg:safe_mapper", safe_mapper) ||
          parse_bool(*it, "-lg:safe_model", safe_model) ||
          parse_bool(*it, "-lg:safe_tracing", safe_tracing) ||
          parse_bool(*it, "-lg:inorder", program_order_execution) ||
          parse_bool(*it, "-lg:dump_physical_traces", dump_physical_traces) ||
          parse_bool(*it, "-lg:no_tracing", no_tracing) ||
          parse_bool(*it, "-lg:no_physical_tracing", no_physical_tracing) ||
          parse_bool(*it, "-lg:no_auto_tracing", no_auto_tracing) ||
          parse_bool(*it, "-lg:no_trace_optimization", no_trace_optimization) ||
          parse_bool(*it, "-lg:no_fence_elision", no_fence_elision) ||
          parse_bool(
              *it, "-lg:no_transitive_reduction", no_transitive_reduction) ||
          parse_bool(
              *it, "-lg:inline_transitive_reduction",
              inline_transitive_reduction) ||
          parse_bool(*it, "-lg:replay_on_cpus", replay_on_cpus) ||
          parse_bool(*it, "-lg:disjointness", verify_partitions) ||
          parse_bool(*it, "-lg:partcheck", verify_partitions) ||
          parse_bool(*it, "-lg:dump_free_ranges", dump_free_ranges) ||
          parse_bool(*it, "-lg:no_dyn", disable_independence_tests) ||
          parse_bool(
              *it, "-lg:enable_pointwise_analysis",
              enable_pointwise_analysis) ||
          parse_bool(*it, "-lg:verbose", verbose_logging) ||
          parse_bool(*it, "-lg:prof_self", prof_self_profile) ||
          parse_bool(
              *it, "-lg:prof_critical_paths", prof_no_critical_paths, false) ||
          parse_bool(
              *it, "-lg:prof_no_critical_paths", prof_no_critical_paths) ||
          parse_bool(
              *it, "-lg:prof_all_critical_arrivals",
              prof_all_critical_arrivals) ||
          parse_bool(*it, "-lg:debug_ok", slow_config_ok) ||
          parse_bool(*it, "-lg:test", enable_test_mapper))
        return 1;
      if (parse_int(
              it, end, "-lg:safe_ctrlrepl", safe_control_replication,
              bad_parameter) ||
          parse_int(
              it, end, "-lg:window", initial_task_window_size, bad_parameter) ||
          parse_int(
              it, end, "-lg:hysteresis", initial_task_window_hysteresis,
              bad_parameter) ||
          parse_int(
              it, end, "-lg:sched", initial_tasks_to_schedule, bad_parameter) ||
          parse_int(
              it, end, "-lg:vector", initial_meta_task_vector_width,
              bad_parameter) ||
          parse_int(it, end, "-lg:message", max_message_size, bad_parameter) ||
          parse_int(it, end, "-lg:epoch", gc_epoch_size, bad_parameter) ||
          parse_int(it, end, "-lg:local", max_local_fields, bad_parameter) ||
          parse_int(
              it, end, "-lg:parallel_replay", max_replay_parallelism,
              bad_parameter) ||
          parse_int(it, end, "-lg:spy", spy_level, bad_parameter) ||
          parse_int(it, end, "-lg:delay", delay_start, bad_parameter) ||
          parse_int(it, end, "-lg:prof", num_profiling_nodes, bad_parameter) ||
          parse_int(
              it, end, "-lg:prof_footprint", prof_footprint_threshold,
              bad_parameter) ||
          parse_int(
              it, end, "-lg:prof_latency", prof_target_latency,
              bad_parameter) ||
          parse_int(
              it, end, "-lg:prof_call_threshold", prof_call_threshold,
              bad_parameter))
        return 2;
      if (parse_string(it, end, "-lg:replay", replay_file, bad_parameter) ||
          parse_string(it, end, "-lg:ldb", ldb_file, bad_parameter) ||
          parse_string(
              it, end, "-lg:serializer", serializer_type, bad_parameter) ||
          parse_string(
              it, end, "-lg:prof_logfile", prof_logfile, bad_parameter))
        return 2;
      return 0;
    }

    //--------------------------------------------------------------------------
    /*static*/ const Runtime::LegionConfiguration& Runtime::initialize(
        int* argc, char*** argv, bool parse, bool filter, bool init_network,
        bool configure_machine)
    //--------------------------------------------------------------------------
    {
      static LegionConfiguration config;
      RealmRuntime realm = RealmRuntime::get_runtime();
      if (!runtime_initialized)
      {
        if (init_network)
          legion_no_skip_assert(realm.network_init(argc, argv));
        if (configure_machine)
          legion_no_skip_assert(realm.create_configs(*argc, *argv));
        runtime_initialized = true;
      }
      if (runtime_cmdline_parsed || !parse)
        return config;
      // Next we configure the realm runtime after which we can access the
      // machine model and make events and reservations and do reigstrations
      std::vector<std::string> cmdline;
      // Check to see if there are any Legion default args from the environment
      const char* e = getenv("LEGION_DEFAULT_ARGS");
      if (e)
      {
        // This code is borrowed from Realm for parsing default arguments
        // Prepend any default args so they can still be overridden
        // by actual flags on the command line
        while (*e)
        {
          if (isspace(*e))
          {
            e++;
            continue;
          }
          const char* starts = nullptr;
          if (*e == '\'')
          {
            // single quoted string
            e++;
            legion_assert(*e);
            starts = e;
            // read until next single quote
            while (*e && (*e != '\'')) e++;
            cmdline.emplace_back(std::string(starts, size_t(e++ - starts)));
            legion_assert(!*e || isspace(*e));
            continue;
          }
          if (*e == '\"')
          {
            // double quoted string
            e++;
            legion_assert(*e);
            starts = e;
            // read until next double quote
            while (*e && (*e != '\"')) e++;
            cmdline.emplace_back(std::string(starts, size_t(e++ - starts)));
            legion_assert(!*e || isspace(*e));
            continue;
          }
          // no quotes - just take until next whitespace
          starts = e;
          while (*e && !isspace(*e)) e++;
          cmdline.emplace_back(std::string(starts, size_t(e - starts)));
        }
      }
      size_t num_args = *argc;
      legion_assert(num_args > 0);  // should always have a binary name
      cmdline.reserve(cmdline.size() + num_args);
      for (unsigned i = 0; i < num_args; i++) cmdline.emplace_back((*argv)[i]);
      if (configure_machine)
        realm.parse_command_line(cmdline, filter);
      // We use the binary name as the demarcation between the default
      // arguments and the command line arguments, note that it's location
      // could have changed by the call realm.parse_command_line if
      // Realm pruned out default arguments
      std::optional<unsigned> binary_offset;
      constexpr std::string_view prefix("-lg:");
      for (std::vector<std::string>::iterator it = cmdline.begin();
           it != cmdline.end();
           /*nothing*/)
      {
        // First check if we have a legion prefix
        if ((it->size() < prefix.size()) ||
            (it->compare(0, prefix.size(), prefix) != 0))
        {
          // Not a legion flag
          if (!binary_offset && (it->compare((*argv)[0]) == 0))
            binary_offset = std::distance(cmdline.begin(), it);
          it++;
        }
        else
        {
          bool bad_parameter = false;
          const size_t matched =
              config.parse_option(it, cmdline.end(), bad_parameter);
          if (bad_parameter)
          {
            Error error(LEGION_STARTUP_EXCEPTION);
            std::vector<std::string>::iterator next = std::next(it);
            if (next == cmdline.end())
            {
              if (binary_offset)
                error << "Missing parameter for Legion argument '" << *it
                      << "' in command line arguments.";
              else
                error << "Missing argument for Legion option '" << *it
                      << "' in LEGION_DEFAULT_ARGS.";
            }
            else
            {
              if (binary_offset)
                error << "Invalid parameter '" << *next << "' passed to Legion "
                      << "argument '" << *it << "' in command line arguments.";
              else
                error << "Invalid parameter '" << *next << "' passed to Legion "
                      << "argument '" << *it << "' in LEGION_DEFAULT_ARGS.";
            }
            error.raise();
          }
          else if (matched == 0)
          {
            Error error(LEGION_STARTUP_EXCEPTION);
            if (binary_offset)
              error << "Detected unknown Legion option '" << *it
                    << "' in command line arguments.";
            else
              error << "Detected unknown Legion option '" << *it
                    << "' in LEGION_DEFAULT_ARGS.";
            error.raise();
          }
          else if (filter)
            it = cmdline.erase(it, it + matched);
          else
            it = it + matched;
        }
      }
      // Restore the legion spy logging level
      spy_logging_level = (SpyLoggingLevel)config.spy_level;
      // Should never have filtered out the binary name
      legion_assert(binary_offset);
      if (filter)
      {
        unsigned arg_index = 1;
        for (unsigned idx = *binary_offset + 1; idx < cmdline.size(); idx++)
        {
          // Find the location of this string in the original
          // arguments so that we can get its original pointer
          legion_assert(arg_index < num_args);
          while (cmdline[idx].compare((*argv)[arg_index]) != 0)
          {
            arg_index++;
            legion_assert(arg_index < num_args);
          }
          // Now that we've got it's original pointer we can move
          // it to the new location in the outputs
          if (arg_index == (idx - *binary_offset))
            arg_index++;  // already in the right place
          else
            (*argv)[idx - *binary_offset] = (*argv)[arg_index++];
        }
        *argc = cmdline.size() - *binary_offset;
      }
      if (config.initial_task_window_hysteresis > 100)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal task window hysteresis value of "
              << config.initial_task_window_hysteresis
              << " which is not a value between 0 and 100.";
        error.raise();
      }
      if (config.max_local_fields > LEGION_MAX_FIELDS)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal max local fields value " << config.max_local_fields
              << " which is larger than the value of LEGION_MAX_FIELDS ("
              << LEGION_MAX_FIELDS << ").";
        error.raise();
      }
      constexpr Realm::Logger::LoggingLevel compile_time_min_level =
          Realm::Logger::REALM_LOGGING_MIN_LEVEL;
      if ((spy_logging_level > NO_SPY_LOGGING) &&
          (Realm::Logger::LEVEL_PRINT < compile_time_min_level))
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Legion Spy logging requires a COMPILE_TIME_MIN_LEVEL of at "
                 "most LEVEL_PRINT, but current setting is "
              << ((compile_time_min_level == Realm::Logger::LEVEL_WARNING) ?
                      "LEVEL_WARNING" :
                  (compile_time_min_level == Realm::Logger::LEVEL_ERROR) ?
                      "LEVEL_ERROR" :
                  (compile_time_min_level == Realm::Logger::LEVEL_FATAL) ?
                      "LEVEL_FATAL" :
                      "LEVEL_NONE");
        error.raise();
      }
      if ((config.num_profiling_nodes > 0) &&
          (strcmp(config.serializer_type.c_str(), "ascii") == 0) &&
          (Realm::Logger::LEVEL_INFO < compile_time_min_level))
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error
            << "Legion Prof 'ascii' logging requires a COMPILE_TIME_MIN_LEVEL "
               "of at most LEVEL_INFO, but current setting is "
            << ((compile_time_min_level == Realm::Logger::LEVEL_PRINT) ?
                    "LEVEL_PRINT" :
                (compile_time_min_level == Realm::Logger::LEVEL_WARNING) ?
                    "LEVEL_WARNING" :
                (compile_time_min_level == Realm::Logger::LEVEL_ERROR) ?
                    "LEVEL_ERROR" :
                (compile_time_min_level == Realm::Logger::LEVEL_FATAL) ?
                    "LEVEL_FATAL" :
                    "LEVEL_NONE");
        error.raise();
      }
      if (config.record_registration &&
          (Realm::Logger::LEVEL_PRINT < compile_time_min_level))
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error
            << "Legion registration logging requires a COMPILE_TIME_MIN_LEVEL "
               "of at most LEVEL_PRINT, but current setting is "
            << ((compile_time_min_level == Realm::Logger::LEVEL_WARNING) ?
                    "LEVEL_WARNING" :
                (compile_time_min_level == Realm::Logger::LEVEL_ERROR) ?
                    "LEVEL_ERROR" :
                (compile_time_min_level == Realm::Logger::LEVEL_FATAL) ?
                    "LEVEL_FATAL" :
                    "LEVEL_NONE");
        error.raise();
      }
      if (config.dump_physical_traces &&
          (Realm::Logger::LEVEL_INFO < compile_time_min_level))
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Legion physical trace logging requires a "
                 "COMPILE_TIME_MIN_LEVEL of at most LEVEL_INFO, but current "
                 "setting is "
              << ((compile_time_min_level == Realm::Logger::LEVEL_PRINT) ?
                      "LEVEL_PRINT" :
                  (compile_time_min_level == Realm::Logger::LEVEL_WARNING) ?
                      "LEVEL_WARNING" :
                  (compile_time_min_level == Realm::Logger::LEVEL_ERROR) ?
                      "LEVEL_ERROR" :
                  (compile_time_min_level == Realm::Logger::LEVEL_FATAL) ?
                      "LEVEL_FATAL" :
                      "LEVEL_NONE");
        error.raise();
      }
      runtime_cmdline_parsed = true;
      return config;
    }

    //--------------------------------------------------------------------------
    /*static*/ unsigned Runtime::initialize_outstanding_top_level_tasks(
        AddressSpaceID local_space, size_t total_spaces, unsigned radix)
    //--------------------------------------------------------------------------
    {
      // We always have at least one top-level task in the count as a guard
      // that will be removed once we know that we aren't launching anymore
      // new top-level tasks on this node
      unsigned result = 1;
      // Now count how many notifications we expect to see and add that to
      // our count to act as an additional guard. This will allow us to do
      // a tree reduction down from each node towards node 0 which will
      // start the shutdown quiescence test
      AddressSpaceID offset = local_space * radix;
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        AddressSpaceID target = offset + idx;
        if (target < total_spaces)
          result++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future Runtime::launch_top_level_task(
        const TaskLauncher& launcher, TopLevelContext* top_context)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_procs.empty());
      // Find a target processor, we'll prefer a CPU processor for
      // backwards compatibility, but will take anything we get
      Processor target = Processor::NO_PROC;
      for (const Processor& local_proc : local_procs)
      {
        if (local_proc.kind() == Processor::LOC_PROC)
        {
          target = local_proc;
          break;
        }
        else if (!target.exists())
          target = local_proc;
      }
      legion_assert(target.exists());
      if (top_context == nullptr)
        top_context = new TopLevelContext(
            target, get_unique_top_level_task_id(), 0 /*implicit*/);
      // Save the current context if there is one and restore it later
      TaskContext* previous_implicit = implicit_context;
      // Save the context in the implicit context
      implicit_context = top_context;
      if ((profiler != nullptr) && (implicit_profiler == nullptr))
        profiler->instantiate_profiling_instance();
      // Add a reference to the top level context
      top_context->add_base_gc_ref(RUNTIME_REF);
      // Get an individual task to be the top-level task
      IndividualTask* top_task = get_operation<IndividualTask>();
      AutoProvenance provenance(launcher.provenance);
      // Mark that this task is the top-level task
      Future result = top_task->initialize_task(
          top_context, launcher, provenance, true /*top level task*/);
      // Set this to be the current processor
      top_task->set_current_proc(target);
      top_task->select_task_options(false /*prioritize*/);
      increment_outstanding_top_level_tasks();
      // Launch a task to deactivate the top-level context
      // when the top-level task is done
      TopFinishArgs args(top_context);
      RtEvent pre = top_task->get_commit_event();
      issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, pre);
      add_to_ready_queue(target, top_task);
      // Now we can restore the previous implicit context
      implicit_context = previous_implicit;
      return result;
    }

    //--------------------------------------------------------------------------
    IndividualTask* Runtime::create_implicit_top_level(
        TaskID top_task_id, MapperID top_mapper_id, Processor proxy,
        const char* task_name, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      // Get an individual task to be the top-level task
      IndividualTask* top_task = get_operation<IndividualTask>();
      // Get a remote task to serve as the top of the top-level task
      TopLevelContext* top_context = new TopLevelContext(
          proxy, 0 /*id*/, get_unique_implicit_top_level_task_id(), 0 /*did*/,
          mapping);
      // Add a reference to the top level context
      top_context->add_base_gc_ref(RUNTIME_REF);
      TaskLauncher launcher(
          top_task_id, UntypedBuffer(), Predicate::TRUE_PRED, top_mapper_id);
      // Mark that this task is the top-level task
      top_task->initialize_task(
          top_context, launcher, nullptr /*provenance*/,
          true /*top level task*/);
      increment_outstanding_top_level_tasks();
      // Launch a task to deactivate the top-level context
      // when the top-level task is done
      TopFinishArgs args(top_context);
      RtEvent pre = top_task->get_commit_event();
      issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, pre);
      return top_task;
    }

    //--------------------------------------------------------------------------
    void Runtime::TopFinishArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      if (ctx->remove_base_gc_ref(RUNTIME_REF))
        delete ctx;
      // Finally tell the runtime that we have one less top level task
      runtime->decrement_outstanding_top_level_tasks();
    }

    //--------------------------------------------------------------------------
    ImplicitShardManager* Runtime::find_implicit_shard_manager(
        TaskID top_task_id, MapperID mapper_id, Processor::Kind kind,
        unsigned shards_per_address_space)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(shard_lock);
      std::map<TaskID, ImplicitShardManager*>::iterator finder =
          implicit_shard_managers.find(top_task_id);
      if (finder != implicit_shard_managers.end())
        return finder->second;
      ImplicitShardManager* result = new ImplicitShardManager(
          top_task_id, mapper_id, kind, shards_per_address_space);
      implicit_shard_managers[top_task_id] = result;
      result->add_reference(shards_per_address_space);
      return result;
    }

    //--------------------------------------------------------------------------
    void Runtime::unregister_implicit_shard_manager(TaskID top_task_id)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(shard_lock);
      std::map<TaskID, ImplicitShardManager*>::iterator finder =
          implicit_shard_managers.find(top_task_id);
      legion_assert(finder != implicit_shard_managers.end());
      implicit_shard_managers.erase(finder);
    }

    //--------------------------------------------------------------------------
    Context Runtime::begin_implicit_task(
        TaskID top_task_id, MapperID top_mapper_id, Processor::Kind proc_kind,
        const char* task_name, bool control_replicable,
        unsigned shards_per_address_space, int shard_id,
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Implicit top-level tasks are not allowed to be started "
                 "before the Legion runtime is started.";
        error.raise();
      }
      // Check that we're not inside a task
      if (implicit_context != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Implicit top-level tasks are not allowed to be started "
                 "inside of Legion tasks. Only external computations are "
                 "permitted to create new implicit top-level tasks.";
        error.raise();
      }
      // Save the top-level task name if necessary
      // Record a fake variant if we're profiling
      if (task_name != nullptr)
      {
        attach_semantic_information(
            top_task_id, LEGION_NAME_SEMANTIC_TAG, task_name,
            strlen(task_name) + 1, true /*mutable*/, false /*send to owner*/);
        if (profiler != nullptr)
          profiler->register_task_variant(
              top_task_id, 0 /*variant ID*/, task_name);
      }
      else if (profiler != nullptr)
      {
        char implicit_name[64];
        snprintf(implicit_name, 64, "implicit_variant_%d", top_task_id);
        profiler->register_task_variant(
            top_task_id, 0 /*variant ID*/, implicit_name);
      }
      // Find a proxy processor for us to use for this task
      // We might already even be on a Realm processor
      Processor proxy = Processor::get_executing_processor();
      if (!proxy.exists())
      {
        legion_assert(!local_procs.empty());
        if (proc_kind != Processor::NO_KIND)
        {
          for (const Processor& local_proc : local_procs)
          {
            if (local_proc.kind() == proc_kind)
            {
              proxy = local_proc;
              break;
            }
          }
          if (!proxy.exists())
          {
            Error error(LEGION_RESOURCE_EXCEPTION);
            error << "Unable to find a proxy processor of kind " << proc_kind
                  << " for implicit top-level task ";
            if (task_name != nullptr)
              error << task_name;
            else
              error << top_task_id;
            error << " on node " << address_space
                  << ". If you specify that a particular kind of "
                  << "processor kind must be found for an implicit "
                  << "top-level task then you must guarantee that the "
                  << "machine was configured with at least one such "
                  << "processor on this node.";
            error.raise();
          }
        }
        else  // Processor kind doesn't matter so use any
          proxy = *local_procs.begin();
      }
      SingleTask* local_task = nullptr;
      // Now that the runtime is started we can make our context
      if (control_replicable && (total_address_spaces > 1))
      {
        // Either find or make an implicit shard manager for hooking up
        ImplicitShardManager* implicit_shard_manager =
            find_implicit_shard_manager(
                top_task_id, top_mapper_id, proc_kind,
                shards_per_address_space);
        local_task = implicit_shard_manager->create_shard(
            shard_id, point, proxy, task_name);
        if (implicit_shard_manager->remove_reference())
          delete implicit_shard_manager;
      }
      else
      {
        local_task = create_implicit_top_level(
            top_task_id, top_mapper_id, proxy, task_name);
        // Increment the pending count here
        local_task->get_context()->increment_pending();
      }
      increment_total_outstanding_tasks(top_task_id, false);
      InnerContext* execution_context = local_task->create_implicit_context();
      execution_context->begin_task(proxy);
      return execution_context;
    }

    //--------------------------------------------------------------------------
    void Runtime::unbind_implicit_task_from_external_thread(TaskContext* ctx)
    //--------------------------------------------------------------------------
    {
      if (!ctx->implicit_task)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to unbind a context for task " << *ctx
              << " that is not an implicit top-level task";
        error.raise();
      }
      if (ctx != implicit_context)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to unbind an implicit top-level task " << *ctx
              << "when it is not bound to the current external thread.";
        error.raise();
      }
      if (implicit_profiler != nullptr)
      {
        ctx->begin_wait(LgEvent::NO_LG_EVENT, true /*from application*/);
        implicit_fevent = implicit_profiler->external_fevent;
        Fatal fatal;
        fatal << "Need support for profiling unbind implicit top-level tasks";
        fatal.raise();
      }
      else
        implicit_fevent = LgEvent::NO_LG_EVENT;
      implicit_context = nullptr;
      implicit_enclosing_context = 0;
      implicit_unique_op_id = 0;
    }

    //--------------------------------------------------------------------------
    void Runtime::bind_implicit_task_to_external_thread(TaskContext* ctx)
    //--------------------------------------------------------------------------
    {
      if (!ctx->implicit_task)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to bind a context for task " << *ctx
              << " that is not an implicit top-level task";
        error.raise();
      }
      if (Processor::get_executing_processor().exists())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Attempted to bind an implicit top-level task " << *ctx
              << " on a Realm processor. Implicit top-level tasks can only "
              << "be run on external threads not managed by Realm.";
        error.raise();
      }
      if (implicit_context != nullptr)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to bind an implicit top-level task " << *ctx
              << "to an external thread that already has an implicit top-level "
              << "task associated with it. Only one implicit top-level task "
              << "can be associated with an external thread at a time.";
        error.raise();
      }
      ctx->end_wait(LgEvent::NO_LG_EVENT, true /*from application*/);
      if ((profiler != nullptr) && (implicit_profiler == nullptr))
      {
        profiler->instantiate_profiling_instance();
        Fatal fatal;
        fatal << "Need support for profiling binding implicit top-level tasks";
        fatal.raise();
      }
      implicit_context = ctx;
      implicit_enclosing_context = ctx->did;
      implicit_fevent = ctx->owner_task->get_completion_event();
      implicit_unique_op_id = ctx->owner_task->get_unique_op_id();
    }

    //--------------------------------------------------------------------------
    void Runtime::finish_implicit_task(TaskContext* ctx, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      if (implicit_context != ctx)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to finish an implicit top-level task " << *ctx
              << " which is not currently bound to the external thread.";
        error.raise();
      }
      if (!ctx->implicit_task)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to finish an implicit task for task " << *ctx
              << "that is not an implicit top-level task";
        error.raise();
      }
      // this is just a normal finish operation
      ctx->end_task(
          nullptr, 0, false /*owned*/, PhysicalInstance::NO_INST,
          nullptr /*callback functor*/, nullptr /*resource*/,
          nullptr /*freefunc*/, nullptr /*metadataptr*/, 0 /*metadatasize*/,
          effects);
      if (implicit_profiler != nullptr)
        implicit_fevent = implicit_profiler->external_fevent;
      else
        implicit_fevent = LgEvent::NO_LG_EVENT;
      implicit_context = nullptr;
      implicit_unique_op_id = 0;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::perform_slow_config_checks(
        const LegionConfiguration& config)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      if (config.num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with debug enabled
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "!!! YOU ARE PROFILING IN DEBUG MODE           !!!\n");
        fprintf(stderr, "!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr, "!!! COMPILE WITH DEBUG=0 FOR PROFILING        !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
      if ((spy_logging_level > NO_SPY_LOGGING) &&
          (config.num_profiling_nodes > 0))
      {
        // Give a massive warning about profiling with Legion Spy enabled
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "!!! YOU ARE PROFILING WITH LegionSpy ENABLED  !!!\n");
        fprintf(stderr, "!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr, "!!! RUN WITHOUT -lg:spy flag FOR PROFILING    !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#ifdef LEGION_BOUNDS_CHECKS
      if (config.num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with bounds checks enabled
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "!!! YOU ARE PROFILING WITH LEGION_BOUNDS_CHECKS!!!\n");
        fprintf(stderr, "!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR !!!\n");
        fprintf(stderr, "!!! PLEASE COMPILE WITHOUT LEGION_BOUNDS_CHECKS!!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
#ifdef LEGION_PRIVILEGE_CHECKS
      if (config.num_profiling_nodes > 0)
      {
        // Give a massive warning about profiling with privilege checks enabled
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(
            stderr, "!!!YOU ARE PROFILING WITH LEGION_PRIVILEGE_CHECKS!!\n");
        fprintf(stderr, "!!!SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(
            stderr, "!!!PLEASE COMPILE WITHOUT LEGION_PRIVILEGE_CHECKS!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
#endif
      if (config.verify_partitions && (config.num_profiling_nodes > 0))
      {
        // Give a massive warning about profiling with partition checks enabled
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "!!! YOU ARE PROFILING WITH PARTITION CHECKS ON!!!\n");
        fprintf(stderr, "!!! SERIOUS PERFORMANCE DEGRADATION WILL OCCUR!!!\n");
        fprintf(stderr, "!!! DO NOT USE -lg:partcheck WITH PROFILING   !!!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for (int i = 0; i < 4; i++)
          fprintf(
              stderr, "!WARNING WARNING WARNING WARNING WARNING WARNING!\n");
        for (int i = 0; i < 2; i++)
          fprintf(
              stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "SLEEPING FOR 5 SECONDS SO YOU READ THIS WARNING...\n");
        fflush(stderr);
        sleep(5);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_interoperability(void)
    //--------------------------------------------------------------------------
    {
      const std::vector<LegionHandshake>& pending_handshakes =
          get_pending_handshake_table();
      if (!pending_handshakes.empty())
      {
        for (const LegionHandshake& handshake : pending_handshakes)
          handshake.impl->initialize();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor Runtime::configure_runtime(
        int argc, char** argv, const LegionConfiguration& config,
        RealmRuntime& realm, std::set<Processor>& local_procs, bool background,
        bool supply_default_mapper)
    //--------------------------------------------------------------------------
    {
      Processor first_proc = Processor::NO_PROC;
      // Do some error checking in case we are running with separate instances
      Machine machine = Machine::get_machine();
      // Compute the data structures necessary for constructing a runtime
      std::set<Processor> local_util_procs;
      Processor::Kind startup_kind = Processor::NO_KIND;
      // First we find all our local processors
      {
        Machine::ProcessorQuery local_proc_query(machine);
        local_proc_query.local_address_space();
        // Check for exceeding the local number of processors
        if (local_proc_query.count() > LEGION_MAX_NUM_PROCS)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Maximum number of local processors "
                << local_proc_query.count()
                << " exceeds compile-time maximum of " << LEGION_MAX_NUM_PROCS
                << ".  Change the value LEGION_MAX_NUM_PROCS in "
                   "legion_config.h and recompile.";
          error.raise();
        }
        for (Machine::ProcessorQuery::iterator it = local_proc_query.begin();
             it != local_proc_query.end(); it++)
        {
          if (it->kind() == Processor::UTIL_PROC)
          {
            local_util_procs.insert(*it);
            // Startup can also be a utility processor if nothing else
            if (startup_kind == Processor::NO_KIND)
              startup_kind = Processor::UTIL_PROC;
          }
          else
          {
            local_procs.insert(*it);
            // Prefer CPUs for the startup kind
            if (it->kind() == Processor::LOC_PROC)
              startup_kind = Processor::LOC_PROC;
          }
        }
        if (local_procs.empty())
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error << "Machine model contains no local processors!";
          error.raise();
        }
      }
      // Check to make sure we have something to do startup
      if (startup_kind == Processor::NO_KIND)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Machine model contains no CPU processors and no utility "
                 "processors! At least one CPU or one utility processor is "
                 "required for Legion.";
        error.raise();
      }
      // Find the local system memory for our runtime
      Memory system_memory;
      {
        Machine::MemoryQuery local_sysmems(machine);
        local_sysmems.local_address_space();
        local_sysmems.only_kind(Memory::SYSTEM_MEM);
        if (local_sysmems.count() == 0)
        {
          Error error(LEGION_STARTUP_EXCEPTION);
          error << "Machine model contains no local system memories! At least "
                   "one system memory must exist in each process for Legion.";
          error.raise();
        }
        system_memory = local_sysmems.first();
      }
      // Now build the data structures for all processors
      std::set<AddressSpaceID> address_spaces;
      Machine::ProcessorQuery all_procs(machine);
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
           it != all_procs.end(); it++)
      {
        AddressSpaceID sid = it->address_space();
        address_spaces.insert(sid);
        if (!first_proc.exists() && (sid == 0) && (it->kind() == startup_kind))
          first_proc = *it;
      }
      // Make one runtime instance and record it with all the processors
      const AddressSpace local_space = local_procs.begin()->address_space();
      // Make a separate copy of the input arguments for the runtime
      InputArgs input_args;
      input_args.argc = argc;
      if (argc > 0)
      {
        input_args.argv = (char**)malloc(argc * sizeof(char*));
        for (int i = 0; i < argc; i++)
        {
          if (argv[i] != nullptr)
            input_args.argv[i] = strdup(argv[i]);
          else
            input_args.argv[i] = nullptr;
        }
      }
      else
        input_args.argv = nullptr;
      // Construct the runtime singleton
      new Runtime(
          machine, config, background, input_args, local_space, system_memory,
          local_procs, local_util_procs, address_spaces, supply_default_mapper);
      Realm::ProfilingRequestSet no_requests;
      // Keep track of all the registration events
      std::vector<RtEvent> registered_events;
      // Make the code descriptors for our tasks
      CodeDescriptor startup_task(Runtime::startup_runtime_task);
      CodeDescriptor shutdown_task(Runtime::shutdown_runtime_task);
      CodeDescriptor lg_task(Runtime::legion_runtime_task);
      CodeDescriptor rt_profiling_task(Runtime::profiling_runtime_task);
      CodeDescriptor endpoint_task(Runtime::endpoint_runtime_task);
      CodeDescriptor app_proc_task(Runtime::application_processor_runtime_task);
      for (const Processor& local_proc : local_procs)
      {
        // These tasks get registered on startup_kind processors
        if (local_proc.kind() == startup_kind)
          registered_events.emplace_back(RtEvent(local_proc.register_task(
              LG_STARTUP_TASK_ID, startup_task, no_requests)));
        // Only register runtime task on application processors if we don't
        // have any utility processors
        if (local_util_procs.empty())
        {
          registered_events.emplace_back(RtEvent(local_proc.register_task(
              LG_SHUTDOWN_TASK_ID, shutdown_task, no_requests)));
#ifdef LEGION_SEPARATE_META_TASKS
          for (unsigned idx = 0; idx < LG_LAST_TASK_ID; idx++)
          {
            if (idx == LG_MESSAGE_ID)
            {
              for (unsigned msg = 0; msg < LAST_SEND_KIND; msg++)
                registered_events.emplace_back(RtEvent(local_proc.register_task(
                    LG_TASK_ID + idx + msg, lg_task, no_requests)));
            }
            else
              registered_events.emplace_back(RtEvent(local_proc.register_task(
                  LG_TASK_ID + idx, lg_task, no_requests)));
          }
#else
          registered_events.emplace_back(RtEvent(
              local_proc.register_task(LG_TASK_ID, lg_task, no_requests)));
#endif
          registered_events.emplace_back(RtEvent(local_proc.register_task(
              LG_ENDPOINT_TASK_ID, endpoint_task, no_requests)));
        }
        // Application processor tasks get registered on all
        // processors which are not utility processors
#ifdef LEGION_SEPARATE_META_TASKS
        for (unsigned idx = 0; idx < LG_LAST_TASK_ID; idx++)
          registered_events.emplace_back(RtEvent(local_proc.register_task(
              LG_APP_PROC_TASK_ID + idx, app_proc_task, no_requests)));
#else
        registered_events.emplace_back(RtEvent(local_proc.register_task(
            LG_APP_PROC_TASK_ID, app_proc_task, no_requests)));
#endif
        // Register profiling return meta-task on all processor kinds
        registered_events.emplace_back(RtEvent(local_proc.register_task(
            LG_LEGION_PROFILING_ID, rt_profiling_task, no_requests)));
      }
      for (const Processor& local_util : local_util_procs)
      {
        // These tasks get registered on startup_kind processors
        if (local_util.kind() == startup_kind)
          registered_events.emplace_back(RtEvent(local_util.register_task(
              LG_STARTUP_TASK_ID, startup_task, no_requests)));
        registered_events.emplace_back(RtEvent(local_util.register_task(
            LG_SHUTDOWN_TASK_ID, shutdown_task, no_requests)));
#ifdef LEGION_SEPARATE_META_TASKS
        for (unsigned idx = 0; idx < LG_LAST_TASK_ID; idx++)
        {
          if (idx == LG_MESSAGE_ID)
          {
            for (unsigned msg = 0; msg < LAST_SEND_KIND; msg++)
              registered_events.emplace_back(RtEvent(local_util.register_task(
                  LG_TASK_ID + idx + msg, lg_task, no_requests)));
          }
          else
            registered_events.emplace_back(RtEvent(local_util.register_task(
                LG_TASK_ID + idx, lg_task, no_requests)));
        }
#else
        registered_events.emplace_back(RtEvent(
            local_util.register_task(LG_TASK_ID, lg_task, no_requests)));
#endif
        registered_events.emplace_back(RtEvent(local_util.register_task(
            LG_ENDPOINT_TASK_ID, endpoint_task, no_requests)));
        // Register profiling return meta-task on all processor kinds
        registered_events.emplace_back(RtEvent(local_util.register_task(
            LG_LEGION_PROFILING_ID, rt_profiling_task, no_requests)));
      }
      // Lastly do any other registrations we might have
      ReductionOpTable& red_table = get_reduction_table(true /*safe*/);
      red_table[BarrierArrivalReduction::REDOP] =
          Realm::ReductionOpUntyped::create_reduction_op<
              BarrierArrivalReduction>();
#ifdef LEGION_DEBUG_COLLECTIVES
      red_table[CollectiveCheckReduction::REDOP] =
          Realm::ReductionOpUntyped::create_reduction_op<
              CollectiveCheckReduction>();
      red_table[CloseCheckReduction::REDOP] =
          Realm::ReductionOpUntyped::create_reduction_op<CloseCheckReduction>();
#endif
      for (const std::pair<
               const Realm::ReductionOpID, Realm::ReductionOpUntyped*>&
               red_pair : red_table)
        realm.register_reduction(red_pair.first, red_pair.second);

      const SerdezOpTable& serdez_table = get_serdez_table(true /*safe*/);
      for (const std::pair<const CustomSerdezID, const SerdezOp*>& serdez_pair :
           serdez_table)
        realm.register_custom_serdez(serdez_pair.first, serdez_pair.second);

      if (config.record_registration)
      {
        log_registration.print(
            "Legion startup task has Realm ID %d", LG_STARTUP_TASK_ID);
        log_registration.print(
            "Legion runtime shutdown task has Realm ID %d",
            LG_SHUTDOWN_TASK_ID);
        log_registration.print(
            "Legion runtime meta-task has Realm ID %d", LG_TASK_ID);
        log_registration.print(
            "Legion runtime profiling task Realm ID %d",
            LG_LEGION_PROFILING_ID);
        log_registration.print(
            "Legion endpoint task has Realm ID %d", LG_ENDPOINT_TASK_ID);
#ifdef LEGION_SEPARATE_META_TASKS
        LG_TASK_DESCRIPTIONS(descs);
        for (unsigned idx = 0; idx < LG_LAST_TASK_ID; idx++)
        {
          if (idx == LG_MESSAGE_ID)
          {
            LG_MESSAGE_DESCRIPTIONS(msg_descs);
            for (unsigned msg = 0; msg < LAST_SEND_KIND; msg++)
              log_registration.print(
                  "Legion message %s meta-task has Realm ID %d", msg_descs[msg],
                  LG_TASK_ID + idx + msg);
          }
          else
          {
            log_registration.print(
                "Legion runtime %s meta-task has Realm ID %d", descs[idx],
                LG_TASK_ID + idx);
            log_registration.print(
                "Legion application %s meta-task has Realm ID %d", descs[idx],
                LG_APP_PROC_TASK_ID + idx);
          }
        }
#endif
      }
      // Make sure that we are done registering before we return
      RtEvent ready = Runtime::merge_events(registered_events);
      if (ready.exists())
        ready.wait();
      return first_proc;
    }

    //--------------------------------------------------------------------------
    /*static*/ int Runtime::wait_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      if (!runtime_backgrounded)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to wait_for_shutdown when runtime was not "
                 "launched in background mode!";
        error.raise();
      }
      // If this is the first time we've called this on this node then
      // we need to remove our reference to allow shutdown to proceed
      if (!background_wait.exchange(true))
        runtime->decrement_outstanding_top_level_tasks();
      return RealmRuntime::get_runtime().wait_for_shutdown();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_return_code(int code)
    //--------------------------------------------------------------------------
    {
      return_code = code;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_id(TaskID top_id)
    //--------------------------------------------------------------------------
    {
      legion_main_id = top_id;
      legion_main_set = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::set_top_level_task_mapper_id(MapperID mapper_id)
    //--------------------------------------------------------------------------
    {
      legion_main_mapper_id = mapper_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::configure_MPI_interoperability(int rank)
    //--------------------------------------------------------------------------
    {
      if (runtime_started)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal call to 'configure_MPI_interoperability' after the "
                 "runtime has been started!";
        error.raise();
      }
      legion_assert(rank >= 0);
      // Check to see if it was already set
      if (mpi_rank >= 0)
      {
        if (rank != mpi_rank)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "multiple calls to configure_MPI_interoperability with "
                   "different ranks "
                << mpi_rank << " and " << rank
                << " on the same Legion runtime!";
          error.raise();
        }
        else
        {
          Warning warning;
          warning
              << "duplicate calls to configure_MPI_interoperability on rank "
              << rank;
          warning.raise();
        }
      }
      mpi_rank = rank;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::register_handshake(LegionHandshake& handshake)
    //--------------------------------------------------------------------------
    {
      // See if the runtime is started or not
      if (runtime_started)
      {
        // If it's started, we can just do the initialization now
        handshake.impl->initialize();
      }
      else
      {
        std::vector<LegionHandshake>& pending_handshakes =
            get_pending_handshake_table();
        pending_handshakes.emplace_back(handshake);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ const ReductionOp* Runtime::get_reduction_op(
        ReductionOpID redop_id, bool has_lock /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (redop_id == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "ReductionOpID zero is reserved.";
        error.raise();
      }
      if (!runtime_started || has_lock)
      {
        ReductionOpTable& red_table =
            Runtime::get_reduction_table(true /*safe*/);
        if (red_table.find(redop_id) == red_table.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid ReductionOpID " << redop_id;
          error.raise();
        }
        return red_table[redop_id];
      }
      else
        return runtime->get_reduction(redop_id);
    }

    //--------------------------------------------------------------------------
    const ReductionOp* Runtime::get_reduction(ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(redop_lock);
      return get_reduction_op(redop_id, true /*has lock*/);
    }

    //--------------------------------------------------------------------------
    FillView* Runtime::find_or_create_reduction_fill_view(ReductionOpID redop)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock r_lock(redop_lock, false /*exclusive*/);
        std::map<ReductionOpID, FillView*>::const_iterator finder =
            redop_fill_views.find(redop);
        if (finder != redop_fill_views.end())
          return finder->second;
      }
      AutoLock r_lock(redop_lock);
      // Check to see if we lost the race
      std::map<ReductionOpID, FillView*>::const_iterator finder =
          redop_fill_views.find(redop);
      if (finder != redop_fill_views.end())
        return finder->second;
      const ReductionOp* reduction_op =
          get_reduction_op(redop, true /*has lock*/);
      legion_assert(reduction_op->identity != nullptr);
      FillView* fill_view = new FillView(
          get_available_distributed_id(), 0 /*no creator*/,
          reduction_op->identity, reduction_op->sizeof_rhs,
          true /*register now*/);
      fill_view->add_base_valid_ref(RUNTIME_REF);
      redop_fill_views[redop] = fill_view;
      return fill_view;
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezOp* Runtime::get_serdez_op(
        CustomSerdezID serdez_id, bool has_lock /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (serdez_id == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "CustomSerdezID zero is reserved.";
        error.raise();
      }
      if (!runtime_started || has_lock)
      {
        SerdezOpTable& serdez_table = Runtime::get_serdez_table(true /*safe*/);
        if (serdez_table.find(serdez_id) == serdez_table.end())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "Invalid CustomSerdezOpID " << serdez_id;
          error.raise();
        }
        return serdez_table[serdez_id];
      }
      else
        return runtime->get_serdez(serdez_id);
    }

    //--------------------------------------------------------------------------
    const SerdezOp* Runtime::get_serdez(CustomSerdezID serdez_id)
    //--------------------------------------------------------------------------
    {
      AutoLock s_lock(serdez_lock);
      return get_serdez_op(serdez_id, true /*has lock*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ const SerdezRedopFns* Runtime::get_serdez_redop_fns(
        ReductionOpID redop_id, bool has_lock /*= false*/)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started || has_lock)
      {
        SerdezRedopTable& serdez_table = get_serdez_redop_table(true /*safe*/);
        SerdezRedopTable::const_iterator finder = serdez_table.find(redop_id);
        if (finder != serdez_table.end())
          return &(finder->second);
        return nullptr;
      }
      else
        return runtime->get_serdez_redop(redop_id);
    }

    //--------------------------------------------------------------------------
    const SerdezRedopFns* Runtime::get_serdez_redop(ReductionOpID redop_id)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(redop_lock);
      return get_serdez_redop_fns(redop_id, true /*has lock*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::add_registration_callback(
        RegistrationCallback callback, bool deduplicate, size_t dedup_tag)
    //--------------------------------------------------------------------------
    {
      if (!callback)
      {
        Warning warning;
        warning << "Skipping request to add an empty registration callback";
        warning.raise();
        return;
      }
      if (!runtime_started)
      {
        std::vector<PendingRegistrationCallback>& registration_callbacks =
            get_pending_registration_callbacks();
        registration_callbacks.emplace_back(
            PendingRegistrationCallback(callback, deduplicate, dedup_tag));
      }
      else
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'add_registration_callback' after the "
                 "runtime has been started! Please use "
                 "'perform_registration_callback' for registration calls to be "
                 "done after the runtime has started.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::add_registration_callback(
        RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
        bool deduplicate, size_t dedup_tag)
    //--------------------------------------------------------------------------
    {
      if (!callback)
      {
        Warning warning;
        warning << "Skipping request to add an empty registration callback";
        warning.raise();
        return;
      }
      if (!runtime_started)
      {
        std::vector<PendingRegistrationCallback>& registration_callbacks =
            get_pending_registration_callbacks();
        const size_t size = buffer.get_size();
        if (size > 0)
        {
          void* copy = malloc(size);
          memcpy(copy, buffer.get_ptr(), size);
          registration_callbacks.emplace_back(PendingRegistrationCallback(
              callback, UntypedBuffer(copy, size), deduplicate, dedup_tag));
        }
        else
          registration_callbacks.emplace_back(PendingRegistrationCallback(
              callback, buffer, deduplicate, dedup_tag));
      }
      else
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'add_registration_callback' after the "
                 "runtime has been started! Please use "
                 "'perform_registration_callback' for registration calls to be "
                 "done after the runtime has started.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::perform_dynamic_registration_callback(
        RegistrationCallback callback, bool global, bool deduplicate,
        size_t dedup_tag)
    //--------------------------------------------------------------------------
    {
      if (!callback)
      {
        Warning warning;
        warning << "Skipping request to perform an empty registration callback";
        warning.raise();
        return;
      }
      if (runtime_started)
      {
        const PendingRegistrationCallback registration(
            callback, deduplicate, dedup_tag);
        const RtEvent done_event = runtime->perform_registration_callback(
            registration, global, false /*preggistered*/);
        if (done_event.exists() && !done_event.has_triggered())
        {
          // Block waiting for these to finish currently since we need
          // to guarantee that all the resources are registered before
          // we proceed any further
          if (Processor::get_executing_processor().exists())
            done_event.wait();
          else
            done_event.external_wait();
        }
      }
      else  // can safely ignore global as this call must be done everywhere
        add_registration_callback(callback, deduplicate, dedup_tag);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::perform_dynamic_registration_callback(
        RegistrationWithArgsCallback callback, const UntypedBuffer& buffer,
        bool global, bool deduplicate, size_t dedup_tag)
    //--------------------------------------------------------------------------
    {
      if (!callback)
      {
        Warning warning;
        warning << "Skipping request to perform an empty registration callback";
        warning.raise();
        return;
      }
      if (runtime_started)
      {
        const PendingRegistrationCallback registration(
            callback, buffer, deduplicate, dedup_tag);
        const RtEvent done_event = runtime->perform_registration_callback(
            registration, global, false /*preregistered*/);
        if (done_event.exists() && !done_event.has_triggered())
        {
          // Block waiting for these to finish currently since we need
          // to guarantee that all the resources are registered before
          // we proceed any further
          if (Processor::get_executing_processor().exists())
            done_event.wait();
          else
            done_event.external_wait();
        }
      }
      else  // can safely ignore global as this call must be done everywhere
        add_registration_callback(callback, buffer, deduplicate, dedup_tag);
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpTable& Runtime::get_reduction_table(bool safe)
    //--------------------------------------------------------------------------
    {
      static ReductionOpTable table;
      if (!safe && runtime_started)
        std::abort();
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezOpTable& Runtime::get_serdez_table(bool safe)
    //--------------------------------------------------------------------------
    {
      static SerdezOpTable table;
      if (!safe && runtime_started)
        std::abort();
      return table;
    }

    //--------------------------------------------------------------------------
    /*static*/ SerdezRedopTable& Runtime::get_serdez_redop_table(bool safe)
    //--------------------------------------------------------------------------
    {
      static SerdezRedopTable table;
      if (!safe && runtime_started)
        std::abort();
      return table;
    }

#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
    // Define a free function for Runtime::register_reduction_op because
    //  legion_redop.cu cannot include runtime.h
    //--------------------------------------------------------------------------
    void runtime_register_reduction_op(
        ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
        SerdezFoldFunc fold_func, bool permit_duplicates, bool has_lock = false)
    //--------------------------------------------------------------------------
    {
      Runtime::register_reduction_op(
          redop_id, redop, init_func, fold_func, permit_duplicates, has_lock);
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::register_reduction_op(
        ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
        SerdezFoldFunc fold_func, bool permit_duplicates,
        bool has_lock /*= false*/)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started || has_lock)
      {
        if (redop_id == 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "ERROR: ReductionOpID zero is reserved.";
          error.raise();
        }
        // TODO: figure out a way to make this safe with dynamic registration
#if 0
        if (redop_id >= LEGION_MAX_APPLICATION_REDOP_ID)
          REPORT_LEGION_ERROR(ERROR_RESERVED_REDOP_ID,
                         "ERROR: ReductionOpID %d is greater than or equal "
                         "to the LEGION_MAX_APPLICATION_REDOP_ID of %d "
                         "set in legion_config.h.", redop_id, 
                         LEGION_MAX_APPLICATION_REDOP_ID)
#endif
        if (redop->identity == nullptr)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "ERROR: Legion does not support reduction operators without "
                   "identity values. All reduction operators must have an "
                   "identity value to support fold operations.";
          error.raise();
        }
        ReductionOpTable& red_table =
            Runtime::get_reduction_table(true /*safe*/);
        // Check to make sure we're not overwriting a prior reduction op
        if (!permit_duplicates && (red_table.find(redop_id) != red_table.end()))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "ERROR: ReductionOpID " << redop_id
                << " has already been used in the reduction table";
          error.raise();
        }
        red_table[redop_id] = redop;
        if ((init_func != nullptr) || (fold_func != nullptr))
        {
          legion_assert((init_func != nullptr) && (fold_func != nullptr));
          SerdezRedopTable& serdez_red_table =
              Runtime::get_serdez_redop_table(true /*safe*/);
          SerdezRedopFns& fns = serdez_red_table[redop_id];
          fns.init_fn = init_func;
          fns.fold_fn = fold_func;
        }
      }
      else
        runtime->register_reduction(
            redop_id, redop, init_func, fold_func, permit_duplicates,
            false /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_reduction(
        ReductionOpID redop_id, ReductionOp* redop, SerdezInitFunc init_func,
        SerdezFoldFunc fold_func, bool permit_duplicates, bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (!preregistered && !inside_registration_callback)
      {
        Warning warning;
        warning << "Reduction operator " << redop_id
                << " was dynamically registered outside of a registration "
                   "callback invocation. In the near future this will become "
                   "an error in order to support task subprocesses. Please use "
                   "'perform_registration_callback' to generate a callback "
                   "where it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      // Dynamic registration so do it with realm too
      RealmRuntime realm = RealmRuntime::get_runtime();
      realm.register_reduction(redop_id, redop);
      AutoLock r_lock(redop_lock);
      Runtime::register_reduction_op(
          redop_id, redop, init_func, fold_func, permit_duplicates,
          true /*has locks*/);
    }

    //--------------------------------------------------------------------------
    void Runtime::register_serdez(
        CustomSerdezID serdez_id, SerdezOp* serdez_op, bool permit_duplicates,
        bool preregistered)
    //--------------------------------------------------------------------------
    {
      if (!preregistered && !inside_registration_callback)
      {
        Warning warning;
        warning << "Custom serdez operator " << serdez_id
                << " was dynamically registered outside of a registration "
                   "callback invocation. In the near future this will become "
                   "an error in order to support task subprocesses. Please use "
                   "'perform_registration_callback' to generate a callback "
                   "where it will be safe to perform dynamic registrations.";
        warning.raise();
      }
      // Dynamic registration so do it with realm too
      RealmRuntime realm = RealmRuntime::get_runtime();
      realm.register_custom_serdez(serdez_id, serdez_op);
      AutoLock s_lock(serdez_lock);
      Runtime::register_serdez_op(
          serdez_id, serdez_op, permit_duplicates, true /*has lock*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::register_serdez_op(
        CustomSerdezID serdez_id, SerdezOp* serdez_op, bool permit_duplicates,
        bool has_lock /*= false*/)
    //--------------------------------------------------------------------------
    {
      if (!runtime_started || has_lock)
      {
        if (serdez_id == 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "ERROR: Custom Serdez ID zero is reserved.";
          error.raise();
        }
        // TODO: figure out a way to make this safe with dynamic registration
#if 0
        if (serdez_id >= LEGION_MAX_APPLICATION_SERDEZ_ID)
          REPORT_LEGION_ERROR(ERROR_RESERVED_SERDEZ_ID,
                         "ERROR: ReductionOpID %d is greater than or equal "
                         "to the LEGION_MAX_APPLICATION_SERDEZ_ID of %d set "
                         "in legion_config.h.", serdez_id, 
                         LEGION_MAX_APPLICATION_SERDEZ_ID)
#endif
        SerdezOpTable& serdez_table = Runtime::get_serdez_table(true /*safe*/);
        // Check to make sure we're not overwriting a prior serdez op
        if (!permit_duplicates &&
            (serdez_table.find(serdez_id) != serdez_table.end()))
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "ERROR: CustomSerdezID " << serdez_id
                << " has already been used in the serdez operation table";
          error.raise();
        }
        serdez_table[serdez_id] = serdez_op;
      }
      else
        runtime->register_serdez(
            serdez_id, serdez_op, permit_duplicates, false /*preregistered*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ std::deque<PendingVariantRegistration*>&
        Runtime::get_pending_variant_table(void)
    //--------------------------------------------------------------------------
    {
      static std::deque<PendingVariantRegistration*> pending_variant_table;
      return pending_variant_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<LayoutConstraintID, LayoutConstraintRegistrar>&
        Runtime::get_pending_constraint_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<LayoutConstraintID, LayoutConstraintRegistrar>
          pending_constraint_table;
      return pending_constraint_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<ProjectionID, ProjectionFunctor*>&
        Runtime::get_pending_projection_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<ProjectionID, ProjectionFunctor*>
          pending_projection_table;
      return pending_projection_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<ShardingID, ShardingFunctor*>&
        Runtime::get_pending_sharding_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<ShardingID, ShardingFunctor*> pending_sharding_table;
      return pending_sharding_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<ConcurrentID, ConcurrentColoringFunctor*>&
        Runtime::get_pending_concurrent_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<ConcurrentID, ConcurrentColoringFunctor*>
          pending_concurrent_table;
      return pending_concurrent_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<ExceptionHandlerID, ExceptionHandler*>&
        Runtime::get_pending_exception_handler_table(void)
    //--------------------------------------------------------------------------
    {
      static std::map<ExceptionHandlerID, ExceptionHandler*>
          pending_exception_handler_table;
      return pending_exception_handler_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::vector<LegionHandshake>&
        Runtime::get_pending_handshake_table(void)
    //--------------------------------------------------------------------------
    {
      static std::vector<LegionHandshake> pending_handshakes_table;
      return pending_handshakes_table;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::vector<PendingRegistrationCallback>&
        Runtime::get_pending_registration_callbacks(void)
    //--------------------------------------------------------------------------
    {
      static std::vector<PendingRegistrationCallback> pending_callbacks;
      return pending_callbacks;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID& Runtime::get_current_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      static TaskID current_task_id = LEGION_MAX_APPLICATION_TASK_ID;
      return current_task_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ TaskID Runtime::generate_static_task_id(void)
    //--------------------------------------------------------------------------
    {
      TaskID& next_task = get_current_static_task_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_task_id' after the runtime "
                 "has been started!";
        error.raise();
      }
      return next_task++;
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpID& Runtime::get_current_static_reduction_id(void)
    //--------------------------------------------------------------------------
    {
      // Make sure to reserve space for the built-in reduction operators
      static ReductionOpID current_redop_id =
          LEGION_MAX_APPLICATION_REDOP_ID + LEGION_REDOP_LAST;
      return current_redop_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ ReductionOpID Runtime::generate_static_reduction_id(void)
    //--------------------------------------------------------------------------
    {
      ReductionOpID& next_redop = get_current_static_reduction_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_reduction_id' after the "
                 "runtime has been started!";
        error.raise();
      }
      return next_redop++;
    }

    //--------------------------------------------------------------------------
    /*static*/ CustomSerdezID& Runtime::get_current_static_serdez_id(void)
    //--------------------------------------------------------------------------
    {
      static CustomSerdezID current_serdez_id =
          LEGION_MAX_APPLICATION_SERDEZ_ID;
      return current_serdez_id;
    }

    //--------------------------------------------------------------------------
    /*static*/ CustomSerdezID Runtime::generate_static_serdez_id(void)
    //--------------------------------------------------------------------------
    {
      CustomSerdezID& next_serdez = get_current_static_serdez_id();
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'generate_static_serdez_id' after the "
                 "runtime has been started!";
        error.raise();
      }
      return next_serdez++;
    }

    //--------------------------------------------------------------------------
    /*static*/ VariantID Runtime::preregister_variant(
        const TaskVariantRegistrar& registrar, const void* user_data,
        size_t user_data_size, const CodeDescriptor& code_desc,
        size_t return_size, bool has_return_size, const char* task_name,
        VariantID vid, bool check_id)
    //--------------------------------------------------------------------------
    {
      // Report an error if the runtime has already started
      if (runtime_started)
      {
        Error error(LEGION_STARTUP_EXCEPTION);
        error << "Illegal call to 'preregister_task_variant' after the runtime "
                 "has been started!";
        error.raise();
      }
      if (check_id && (registrar.task_id >= get_current_static_task_id()))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error
            << "Error preregistering task with ID " << registrar.task_id
            << ". Exceeds the statically set bounds on application task IDs of "
            << LEGION_MAX_APPLICATION_TASK_ID << ". See "
            << LEGION_MACRO_TO_STRING(LEGION_MAX_APPLICATION_TASK_ID)
            << " in legion_config.h.";
        error.raise();
      }
      std::deque<PendingVariantRegistration*>& pending_table =
          get_pending_variant_table();
      // See if we need to pick a variant
      if (vid == LEGION_AUTO_GENERATE_ID)
        vid = pending_table.size() + 1;
      else if (vid == 0)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Error preregistering variant for task ID "
              << registrar.task_id
              << " with variant ID 0. Variant ID 0 is reserved for task "
                 "generators.";
        error.raise();
      }
      // Offset by the runtime tasks
      pending_table.emplace_back(new PendingVariantRegistration(
          vid, return_size, has_return_size, registrar, user_data,
          user_data_size, code_desc, task_name));
      return vid;
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::raise_warning(
        Exception&& exception, const Realm::Backtrace& backtrace)
    //--------------------------------------------------------------------------
    {
      legion_assert(exception.type == LEGION_WARNING_EXCEPTION);
      if (runtime != nullptr)
      {
        if (runtime->warnings_are_errors)
          raise_exception(std::move(exception), backtrace);
        if (runtime->warnings_backtrace)
          exception.record_backtrace(backtrace);
        Provenance* provenance = nullptr;
        if (implicit_provenance != 0)
        {
          provenance = runtime->find_provenance(implicit_provenance);
          // This might still fail if we haven't recorded this
          // provenance on this node in which case there is nothing to do
          if (provenance != nullptr)
          {
            std::ostream stream(&exception);
            stream << "\n-----------------------------------\n";
            stream << "Provenance: " << provenance->human << '\n';
          }
        }
        else if (implicit_operation != nullptr)
        {
          provenance = implicit_operation->get_provenance();
          if (provenance != nullptr)
          {
            provenance->add_reference();
            std::ostream stream(&exception);
            stream << "\n-----------------------------------\n";
            stream << "Provenance: " << provenance->human << '\n';
          }
        }
        // Check to see if the user wants to suppress this warning
        if (implicit_context != nullptr)
        {
          ExceptionHandler* handler = runtime->find_exception_handler(
              implicit_context->get_current_exception_handler());
          if (handler->can_handle(LEGION_WARNING_EXCEPTION) &&
              handler->handle_exception(
                  exception,
                  (provenance == nullptr) ? std::string_view() :
                                            provenance->full,
                  backtrace))
            return;
        }
        else if (implicit_operation != nullptr)
        {
          ExceptionHandler* handler = runtime->find_exception_handler(
              implicit_operation->get_exception_handler());
          if (handler->can_handle(LEGION_WARNING_EXCEPTION) &&
              handler->handle_exception(
                  exception,
                  (provenance == nullptr) ? std::string_view() :
                                            provenance->full,
                  backtrace))
            return;
        }
        if ((provenance != nullptr) && provenance->remove_reference())
          delete provenance;
      }
#if 0
      if (runtime == nullptr)
      {
        // Runtime has either not started or been shutdown already
        // Report the warning and then we're done

      }
      else if (implicit_context != nullptr)
      {
        // We're in an application task (maybe an external task)
        // Have the context check for a handler to modify or suppress
        // the warning message


      }
      else if (implicit_operation != nullptr)
      {

      }
      else
      {
        // Just report the warning message

      }
#endif
      log_legion.warning() << std::string_view(exception);
    }

    //--------------------------------------------------------------------------
    /*static*/ [[noreturn]] void Runtime::raise_exception(
        Exception&& exception, const Realm::Backtrace& backtrace)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (exception.type != LEGION_WARNING_EXCEPTION) ||
          ((runtime != nullptr) && runtime->warnings_are_errors));
      if (runtime != nullptr)
      {
        Provenance* provenance = nullptr;
        if (implicit_provenance != 0)
        {
          provenance = runtime->find_provenance(implicit_provenance);
          // This might still fail if we haven't recorded this
          // provenance on this node in which case there is nothing to do
          if (provenance != nullptr)
          {
            std::ostream stream(&exception);
            stream << "\n-----------------------------------\n";
            stream << "Provenance: " << provenance->human << '\n';
          }
        }
        else if (implicit_operation != nullptr)
        {
          provenance = implicit_operation->get_provenance();
          if (provenance != nullptr)
          {
            provenance->add_reference();
            std::ostream stream(&exception);
            stream << "\n-----------------------------------\n";
            stream << "Provenance: " << provenance->human << '\n';
          }
        }
        if (implicit_context != nullptr)
        {
          // Record the task tree context
          implicit_context->record_task_tree_trace(exception, nullptr);
          exception.record_backtrace(backtrace);
          ExceptionHandler* handler = runtime->find_exception_handler(
              implicit_context->get_current_exception_handler());
          if (handler->can_handle(exception.type) &&
              handler->handle_exception(
                  exception,
                  (provenance == nullptr) ? std::string_view() :
                                            provenance->full,
                  backtrace))
          {
            log_legion.warning() << "Ignoring request to handle "
                                 << "exception in " << *implicit_context
                                 << " because it is not yet supported";
          }
        }
        else if (implicit_operation != nullptr)
        {
          // Record the task tree context
          InnerContext* context = implicit_operation->get_context();
          context->record_task_tree_trace(exception, implicit_operation);
          exception.record_backtrace(backtrace);
          ExceptionHandler* handler = runtime->find_exception_handler(
              context->get_current_exception_handler());
          if (handler->can_handle(exception.type) &&
              handler->handle_exception(
                  exception,
                  (provenance == nullptr) ? std::string_view() :
                                            provenance->full,
                  backtrace))
          {
            log_legion.warning() << "Ignoring request to handle "
                                 << "exception in " << *implicit_operation
                                 << " because it is not yet supported";
          }
        }
        else
          exception.record_backtrace(backtrace);
        if ((provenance != nullptr) && provenance->remove_reference())
          delete provenance;
      }
      else
        exception.record_backtrace(backtrace);
#if 0
      if (runtime == nullptr)
      {
        // Runtime has either not started or been shutdown already
        // Report the error and then throw an exception because we 
        // can't recover from this

      }
      if (implicit_context != nullptr)
      {
        // We're in an application task (maybe an external task)
        // Report the exception back to the context

      }
      if (implicit_operation != nullptr)
      {
        // See if this operation has an exception handler that we can
        // use to modify the text of the exception

      }
      if (Processor::get_executing_processor().exists())
      {
        // We're in a meta-task but we can't throw an exception for this
        // one since we know it won't be handled. Report the exception
        // and then abort so we can know to fix this.

      }
#endif
      // If you get here you've called into Legion from an external thread
      // while it is running. The only optional here is to shutdown Legion
      // safely and then abort because we can't recover from this
      if (exception.type == LEGION_FATAL_EXCEPTION)
        log_legion.fatal() << std::string_view(exception);
      else
        log_legion.error() << std::string_view(exception);
      std::abort();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::shutdown_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      // We don't profile this task
      implicit_fevent = LgEvent::NO_LG_EVENT;
      // Finalize the runtime and then delete it
      std::vector<Realm::Event> shutdown_events;
      runtime->finalize_runtime(shutdown_events);
      delete runtime;
      // If we have any shutdown events we need to wait for them to have
      // finished before we return and end up marking ourselves finished
      if (!shutdown_events.empty())
        Realm::Event::merge_events(shutdown_events).wait();
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::legion_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      // Make sure implicit_context is a nullptr so that we know that
      // Meta-tasks only run on the same processors as application tasks
      // when there are no utility processors to use
      if (implicit_context != nullptr)
      {
        implicit_context = nullptr;
        // We should only have an implicit context if we're on an application
        // processor and that should only happen if we don't have any
        // utility processors
        legion_assert(runtime->local_utils.empty());
      }
      legion_assert(implicit_reference_tracker == nullptr);
      // We immediately bump the priority of all meta-tasks once they start
      // up to the highest level to ensure that they drain once they begin
      Processor::set_current_task_priority(LG_RUNNING_PRIORITY);
      LgTaskID tid;
      std::memcpy(&tid, args, sizeof(tid));
      if (runtime->profiler != nullptr)
      {
        implicit_fevent = LgEvent(Processor::get_current_finish_event());
        if (implicit_profiler == nullptr)
          runtime->profiler->instantiate_profiling_instance();
        // If this is a message task, then we need to initialize the
        // implicit_fevent before doing anything that can block
        if (tid == LG_MESSAGE_ID)
          runtime->profiler->increment_outstanding_message_request();
      }
      legion_assert(tid < LG_LAST_TASK_ID);
      void (*handler)(const void*, size_t) = meta_task_table[tid];
      legion_assert(handler != nullptr);
      (*handler)(args, arglen);
      if (implicit_reference_tracker != nullptr)
      {
        delete implicit_reference_tracker;
        implicit_reference_tracker = nullptr;
      }
      if (tid < LG_BEGIN_SHUTDOWN_TASK_IDS)
        runtime->decrement_total_outstanding_tasks(tid, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[tid].fetch_sub(1);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::profiling_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      // Make sure implicit_context is a nullptr so that we know that
      if (implicit_context != nullptr)
        implicit_context = nullptr;
      if (runtime->profiler != nullptr)
      {
        implicit_fevent = LgEvent(Processor::get_current_finish_event());
        if (implicit_profiler == nullptr)
          runtime->profiler->instantiate_profiling_instance();
      }
      Realm::ProfilingResponse response(args, arglen);
      const ProfilingResponseBase* base =
          static_cast<const ProfilingResponseBase*>(response.user_data());
      LgEvent fevent;
      bool failed_alloc = false;
      if (base->handler == nullptr)
      {
        // This is the remote message case
        legion_assert(runtime->profiler != nullptr);
        const long long t_start = Realm::Clock::current_time_in_nanoseconds();
        // Check to see if should report this profiling
        if (runtime->profiler->handle_profiling_response(
                response, args, arglen, fevent, failed_alloc))
        {
          const long long t_stop = Realm::Clock::current_time_in_nanoseconds();
          const LgEvent finish_event(Processor::get_current_finish_event());
          implicit_profiler->process_proc_desc(p);
          implicit_profiler->record_proftask(
              p, base->op_id, t_start, t_stop, fevent, finish_event,
              base->completion || failed_alloc);
        }
      }
      else if (runtime->profiler != nullptr)
      {
        const long long t_start = Realm::Clock::current_time_in_nanoseconds();
        // Check to see if should report this profiling
        if (base->handler->handle_profiling_response(
                response, args, arglen, fevent, failed_alloc))
        {
          const long long t_stop = Realm::Clock::current_time_in_nanoseconds();
          const LgEvent finish_event(Processor::get_current_finish_event());
          implicit_profiler->process_proc_desc(p);
          implicit_profiler->record_proftask(
              p, base->op_id, t_start, t_stop, fevent, finish_event,
              base->completion || failed_alloc);
        }
      }
      else
        base->handler->handle_profiling_response(
            response, args, arglen, fevent, failed_alloc);
    }

    //--------------------------------------------------------------------------
    void Runtime::broadcast_startup_barrier(RtBarrier startup_barrier)
    //--------------------------------------------------------------------------
    {
      legion_assert(startup_barrier.exists());
      // Make sure the representation of the barriers haven't changed
      static_assert(
          sizeof(startup_barrier) ==
              (sizeof(startup_event) + sizeof(startup_timestamp)),
          "Realm Barrier representation changed");
      // Tree broadcast it out to any downstream nodes
      AddressSpaceID offset = address_space * legion_collective_radix;
      for (int idx = 1; idx <= legion_collective_radix; idx++)
      {
        AddressSpaceID target = offset + idx;
        if (target < total_address_spaces)
        {
          StartupBarrierMessage rez;
          rez.serialize(startup_barrier);
          rez.dispatch(target);
        }
      }
      // Write the timestamp first
      startup_timestamp = startup_barrier.timestamp;
      // Then set the ID locally
      RtUserEvent to_trigger;
      to_trigger.id = startup_event.exchange(startup_barrier.id);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::startup_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      // We don't profile this task
      implicit_profiler = nullptr;
      implicit_fevent = LgEvent::NO_LG_EVENT;
      // Create the startup barrier and send it out
      // Note we don't profile this for critical paths
      RtBarrier startup_barrier(
          Realm::Barrier::create_barrier(runtime->total_address_spaces));
      runtime->broadcast_startup_barrier(startup_barrier);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::endpoint_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      Deserializer derez(args, arglen);
      // We don't profile this task
      implicit_profiler = nullptr;
      implicit_fevent = LgEvent::NO_LG_EVENT;
      runtime->handle_endpoint_creation(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void Runtime::application_processor_runtime_task(
        const void* args, size_t arglen, const void* userdata, size_t userlen,
        Processor p)
    //--------------------------------------------------------------------------
    {
      if (implicit_context != nullptr)
        implicit_context = nullptr;
      legion_assert(implicit_reference_tracker == nullptr);
      if (runtime->profiler != nullptr)
      {
        implicit_fevent = LgEvent(Processor::get_current_finish_event());
        if (implicit_profiler == nullptr)
          runtime->profiler->instantiate_profiling_instance();
      }
      // We immediately bump the priority of all meta-tasks once they start
      // up to the highest level to ensure that they drain once they begin
      Processor::set_current_task_priority(LG_RUNNING_PRIORITY);
      LgTaskID tid;
      std::memcpy(&tid, args, sizeof(tid));
      legion_assert(tid < LG_LAST_TASK_ID);
      void (*handler)(const void*, size_t) = meta_task_table[tid];
      legion_assert(handler != nullptr);
      (*handler)(args, arglen);
      if (implicit_reference_tracker != nullptr)
      {
        delete implicit_reference_tracker;
        implicit_reference_tracker = nullptr;
      }
      runtime->decrement_total_outstanding_tasks(tid, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[tid].fetch_sub(1);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ RtBarrier Runtime::find_or_wait_for_startup_barrier(void)
    //--------------------------------------------------------------------------
    {
      RtBarrier result;
      result.id = startup_event.load();
      if (result.exists())
      {
        result.timestamp = startup_timestamp;
        return result;
      }
      // Barrier isn't ready yet so make an event to wait on and try to
      // swap it into the startup event
      const RtUserEvent ready = Runtime::create_rt_user_event();
      if (startup_event.compare_exchange_strong(result.id, ready.id))
      {
        ready.wait();
        result.id = startup_event.load();
      }
      else  // Was already set
        Runtime::trigger_event(ready);
      // Get the timestamp
      result.timestamp = startup_timestamp;
      return result;
    }

#ifdef LEGION_TRACE_ALLOCATION
    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_allocation(
        const std::type_info& info, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (runtime != nullptr)
        runtime->trace_allocation(info, size, elems);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionAllocation::trace_free(
        const std::type_info& info, size_t size, int elems)
    //--------------------------------------------------------------------------
    {
      if (runtime != nullptr)
        runtime->trace_free(info, size, elems);
    }
#endif

  }  // namespace Internal
}  // namespace Legion
