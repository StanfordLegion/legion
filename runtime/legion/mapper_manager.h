/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#ifndef __MAPPER_MANAGER_H__
#define __MAPPER_MANAGER_H__

#include "legion_types.h"
#include "legion_mapping.h"

namespace Legion {
  namespace Internal {

    enum MappingCallKind {
      GET_MAPER_SYNC_MODEL_CALL,
      SELECT_TASK_OPTIONS_CALL,
      PREMAP_TASK_CALL,
      SLICE_TASK_CALL,
      MAP_TASK_CALL,
      SELECT_VARIANT_CALL,
      POSTMAP_TASK_CALL,
      TASK_SELECT_SOURCES_CALL,
      TASK_SPECULATE_CALL,
      TASK_REPORT_PROFILING_CALL,
      MAP_INLINE_CALL,
      INLINE_SELECT_SOURCES_CALL,
      INLINT_REPORT_PROFILING_CALL,
      MAP_COPY_CALL,
      COPY_SELECT_SOURCES_CALL,
      COPY_SPECULATE_CALL,
      COPY_REPORT_PROFILING_CALL,
      MAP_CLOSE_CALL,
      CLOSE_SELECT_SOURCES_CALL,
      CLOSE_REPORT_PROFILING_CALL,
      MAP_ACQUIRE_CALL,
      ACQUIRE_SPECULATE_CALL,
      ACQUIRE_REPORT_PROFILING_CALL,
      MAP_RELEASE_CALL,
      RELEASE_SELECT_SOURCES_CALL,
      RELEASE_SPECULATE_CALL,
      RELEASE_REPORT_PROFILING_CALL,
      CONFIGURE_CONTEXT_CALL,
      SELECT_TUNABLE_VALUE_CALL,
      MAP_MUST_EPOCH_CALL,
      MAP_DATAFLOW_GRAPH_CALL,
      SELECT_TASKS_TO_MAP_CALL,
      SELECT_STEAL_TARGETS_CALL,
      PERMIT_STEAL_REQUEST_CALL,
      HANDLE_MESSAGE_CALL,
      HANDLE_TASK_RESULT_CALL,
    };

    class MappingCallInfo {
    public:
      MappingCallInfo(MapperManager *man, MappingCallKind k)
        : manager(man), resume(UserEvent::NO_USER_EVENT), kind(k) { }
    public:
      MapperManager *const manager;
      UserEvent            resume;
      MappingCallKind      kind;
    };

    /**
     * \class MapperManager
     * This is the base class for a bunch different kinds of mapper
     * managers. Some calls into this manager from the mapper will
     * be handled right away, while other we may need to defer and
     * possibly preempt.  This later class of calls are the ones that
     * are made virtual so that the 
     */
    class MapperManager {
    public:
      struct MapperContinuationArgs {
      public:
        HLRTaskID hlr_id;
        MapperManager *manager;
        MappingCallKind call;
        void *arg1, *arg2, *arg3;
      };
    public:
      MapperManager(Runtime *runtime, Mapping::Mapper *mapper);
      virtual ~MapperManager(void);
    public:
      const char* get_mapper_name(void);
    public: // Task mapper calls
      void invoke_select_task_options(TaskOp *task, Mapper::TaskOptions *output,
                                      bool first_invocation = true);
      void invoke_premap_task(TaskOp *task, Mapper::PremapTaskInput *input,
                              Mapper::PremapTaskOutput *output, 
                              bool first_invocation = true);
      void invoke_slice_task(TaskOp *task, Mapper::SliceTaskInput *input,
                               Mapper::SliceTaskOutput *output, 
                               bool first_invocation = true);
      void invoke_map_task(TaskOp *task, Mapper::MapTaskInput *input,
                           Mapper::MapTaskOutput *output, 
                           bool first_invocation = true);
      void invoke_select_task_variant(TaskOp *task, 
                                      Mapper::SelectVariantInput *input,
                                      Mapper::SelectVariantOutput *output,
                                      bool first_invocation = true);
      void invoke_post_map_task(TaskOp *task, Mapper::PostMapInput *input,
                                Mapper::PostMapOutput *output,
                                bool first_invocation = true);
      void invoke_select_task_sources(TaskOp *task, 
                                      Mapper::SelectTaskSrcInput *input,
                                      Mapper::SelectTaskSrcOutput *output,
                                      bool first_invocation = true);
      void invoke_task_speculate(TaskOp *task, 
                                 Mapper::SpeculativeOutput *output,
                                 bool first_invocation = true);
      void invoke_task_report_profiling(TaskOp *task, 
                                        Mapper::TaskProfilingInfo *input,
                                        bool first_invocation = true);
    public: // Inline mapper calls
      void invoke_map_inline(MapOp *op, Mapper::MapInlineInput *input,
                             Mapper::MapInlineOutput *output, 
                             bool first_invocation = true);
      void invoke_select_inline_sources(MapOp *op, 
                                        Mapper::SelectInlineSrcInput *input,
                                        Mapper::SelectInlineSrcOutput *output,
                                        bool first_invocation = true);
      void invoke_inline_report_profiling(MapOp *op, 
                                          Mapper::InlineProfilingInfo *input,
                                          bool first_invocation = true);
    public: // Copy mapper calls
      void invoke_map_copy(CopyOp *op,
                           Mapper::MapCopyInput *input,
                           Mapper::MapCopyOutput *output,
                           bool first_invocation = true);
      void invoke_select_copy_sources(CopyOp *op,
                                      Mapper::SelectCopySrcInput *input,
                                      Mapper::SelectCopySrcOutput *output,
                                      bool first_invocation = true);
      void invoke_copy_speculate(CopyOp *op, Mapper::SpeculativeOutput *output,
                                 bool first_invocation = true);
      void invoke_copy_report_profiling(CopyOp *op,
                                        Mapper::CopyProfilingInfo *input,
                                        bool first_invocation = true);
    public: // Close mapper calls
      void invoke_map_close(CloseOp *op,
                            Mapper::MapCloseInput *input,
                            Mapper::MapCloseOutput *output,
                            bool first_invocation = true);
      void invoke_select_close_sources(CloseOp *op,
                                       Mapper::SelectCloseSrcInput *input,
                                       Mapper::SelectCloseSrcOutput *output,
                                       bool first_invocation = true);
      void invoke_close_report_profiling(CloseOp *op,
                                         Mapper::CloseProfilingInfo *input,
                                         bool first_invocation = true);
    public: // Acquire mapper calls
      void invoke_map_acquire(AcquireOp *op,
                              Mapper::MapAcquireInput *input,
                              Mapper::MapAcquireOutput *output,
                              bool first_invocation = true);
      void invoke_acquire_speculate(AcquireOp *op,
                                    Mapper::SpeculativeOutput *output,
                                    bool first_invocation = true);
      void invoke_acquire_report_profiling(AcquireOp *op,
                                           Mapper::AcquireProfilingInfo *input,
                                           bool first_invocation = true);
    public: // Release mapper calls
      void invoke_map_release(ReleaseOp *op,
                              Mapper::MapReleaseInput *input,
                              Mapper::MapReleaseOutput *output,
                              bool first_invocation = true);
      void invoke_select_release_sources(ReleaseOp *op,
                                         Mapper::SelectReleaseSrcInput *input,
                                         Mapper::SelectReleaseSrcOutput *output,
                                         bool first_invocation = true);
      void invoke_release_speculate(ReleaseOp *op,
                                    Mapper::SpeculativeOutput *output,
                                    bool first_invocation = true);
      void invoke_release_report_profiling(ReleaseOp *op,
                                           Mapper::ReleaseProfilingInfo *input,
                                           bool first_invocation = true);
    public: // Task execution mapper calls
      void invoke_configure_context(TaskOp *task,
                                    Mapper::ContextConfigOutput *output,
                                    bool first_invocation = true);
      void invoke_select_tunable_value(TaskOp *task,
                                       Mapper::SelectTunableInput *input,
                                       Mapper::SelectTunableOutput *output,
                                       bool first_invocation = true);
    public: // must epoch and graph mapper calls
      void invoke_map_must_epoch(MustEpochOp *op,
                                 Mapper::MapMustEpochInput *input,
                                 Mapper::MapMustEpochOutput *output,
                                 bool first_invocation = true);
      void invoke_map_dataflow_graph(Mapper::MapDataflowGraphInput *input,
                                     Mapper::MapDataflowGraphOutput *output,
                                     bool first_invocation = true);
    public: // scheduling and stealing mapper calls
      void invoke_select_tasks_to_map(Mapper::SelectMappingInput *input,
                                      Mapper::SelectMappingOutput *output,
                                      bool first_invocation = true);
      void invoke_select_steal_targets(Mapper::SelectStealingInput *input,
                                       Mapper::SelectStealingOutput *output,
                                       bool first_invocation = true);
      void invoke_permit_steal_request(Mapper::StealRequestInput *input,
                                       Mapper::StealRequestOutput *output,
                                       bool first_invocation = true);
    public: // handling mapper calls
      void invoke_handle_message(Mapper::MapperMessage *message,
                                 bool first_invocation = true);
      void invoke_handle_task_result(Mapper::MapperTaskResult *result,
                                     bool first_invocation = true);
    public:
      virtual bool is_locked(MappingCallInfo *info) = 0;
      virtual void lock_mapper(MappingCallInfo *info, bool read_only) = 0;
      virtual void unlock_mapper(MappingCallInfo *info) = 0;
    public:
      virtual bool is_reentrant(MappingCallInfo *info) = 0;
      virtual void enable_reentrant(MappingCallInfo *info) = 0;
      virtual void disable_reentrant(MappingCallInfo *info) = 0;
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                bool first_invocation, Event &precondition) = 0;
      virtual void pause_mapper_call(MappingCallInfo *info) = 0;
      virtual void resume_mapper_call(MappingCallInfo *info) = 0;
      virtual void finish_mapper_call(MappingCallInfo *info) = 0;
    public:
      void send_message(MappingCallInfo *info, Processor target, 
                        const void *message, size_t message_size);
      void broadcast(MappingCallInfo *info, const void *message, 
                     size_t message_size, int radix);
    public:
      IndexPartition get_index_partition(IndexSpace parent, Color color);
      IndexSpace get_index_subspace(IndexPartition p, Color c);
      IndexSpace get_index_subspace(IndexPartition p, 
                                    const DomainPoint &color);
      bool has_multiple_domains(IndexSpace handle);
      Domain get_index_space_domain(IndexSpace handle);
      void get_index_space_domains(IndexSpace handle,
                                   std::vector<Domain> &domains);
      Domain get_index_partition_color_space(IndexPartition p);
      void get_index_space_partition_colors(IndexSpace sp, 
                                            std::set<Color> &colors);
      bool is_index_partition_disjoint(IndexPartition p);
      template<unsigned DIM>
      IndexSpace get_index_subspace(IndexPartition p, 
                                    ColorPoint &color_point);
      Color get_index_space_color(IndexSpace handle);
      Color get_index_partition_color(IndexPartition handle);
      IndexSpace get_parent_index_space(IndexPartition handle);
      bool has_parent_index_partition(IndexSpace handle);
      IndexPartition get_parent_index_partition(IndexSpace handle);
    public:
      size_t get_field_size(FieldSpace handle, FieldID fid);
      void get_field_space_fields(FieldSpace handle, 
                                  std::set<FieldID> &fields);
    protected:
      // Both these must be called while holding the lock
      MappingCallInfo* allocate_call_info(MappingCallKind kind, bool need_lock);
      void free_call_info(MappingCallInfo *info, bool need_lock);
    public:
      Runtime *const runtime;
      Mapping::Mapper *const mapper;
    protected:
      const Reservation mapper_lock;
    protected:
      std::vector<MappingCallInfo*> available_infos;
    };

    /**
     * \class SerializingManager
     * In this class at most one mapper call can be running at 
     * a time. Mapper calls that invoke expensive runtime operations
     * can be pre-empted and it is up to the mapper to control
     * whether additional mapper calls when the call is blocked.
     */
    class SerializingManager : public MapperManager {
    public:
      SerializingManager(Runtime *runtime, Mapping::Mapper *mapper,
                         bool reentrant);
      SerializingManager(const SerializingManager &rhs);
      virtual ~SerializingManager(void);
    public:
      SerializingManager& operator=(const SerializingManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                bool first_invocation, Event &precondition);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      bool permit_reentrant;
    protected:
      // The one executing call if any otherwise NULL
      MappingCallInfo *executing_call; 
      // Calls yet to start running
      std::deque<MappingCallInfo*> pending_calls; 
      // Number of calls paused due to runtime work
      unsigned paused_calls;
      // Calls that are ready to resume after runtime work
      std::deque<MappingCallInfo*> ready_calls;
      // Calls that are waiting for diabling of reentrancy
      std::deque<MappingCallInfo*> non_reentrant_calls;
    };

    /**
     * \class ConcurrentManager
     * In this class many mapper calls can be running concurrently.
     * It is upper to the mapper to lock itself when necessary to 
     * protect internal state. Mappers can be locked in exclusive
     * or non-exclusive modes.
     */
    class ConcurrentManager : public MapperManager {
    public:
      enum LockState {
        UNLOCKED_STATE,
        READ_ONLY_STATE,
        EXCLUSIVE_STATE,
      };
    public:
      ConcurrentManager(Runtime *runtime, Mapping::Mapper *mapper);
      ConcurrentManager(const ConcurrentManager &rhs);
      virtual ~ConcurrentManager(void);
    public:
      ConcurrentManager& operator=(const ConcurrentManager &rhs);
    public:
      virtual bool is_locked(MappingCallInfo *info);
      virtual void lock_mapper(MappingCallInfo *info, bool read_only);
      virtual void unlock_mapper(MappingCallInfo *info);
    public:
      virtual bool is_reentrant(MappingCallInfo *info);
      virtual void enable_reentrant(MappingCallInfo *info);
      virtual void disable_reentrant(MappingCallInfo *info);
    protected:
      virtual MappingCallInfo* begin_mapper_call(MappingCallKind kind,
                bool first_invocation, Event &precondition);
      virtual void pause_mapper_call(MappingCallInfo *info);
      virtual void resume_mapper_call(MappingCallInfo *info);
      virtual void finish_mapper_call(MappingCallInfo *info);
    protected:
      // Must be called while holding the lock
      void release_lock(std::vector<UserEvent> &to_trigger);
    protected:
      LockState lock_state;
      std::set<MappingCallInfo*> current_holders;
      std::deque<MappingCallInfo*> read_only_waiters;
      std::deque<MappingCallInfo*> exclusive_waiters;
    };

    /**
     * \class MapperContinuation
     * A class for deferring mapper calls
     */
    class MapperContinuation {
    public:
      struct ContinuationArgs {
        HLRTaskID hlr_id;
        MapperContinuation *continuation;
      };
    public:
      MapperContinuation(MapperManager *manager);
    public:
      void defer(Runtime *runtime, Event precondition, Operation *op = NULL);
    public:
      virtual void execute(void) = 0;
    public:
      static void handle_continuation(const void *args);
    public:
      MapperManager *const manager;
    };

    template<typename T1,
             void (MapperManager::*CALL)(T1*, bool)>
    class MapperContinuation1 : public MapperContinuation {
    public:
      MapperContinuation1(MapperManager *man, T1 *a1)
        : MapperContinuation(man), arg1(a1) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, false/*first*/); }
    public:
      T1 *const arg1;
    };

    template<typename T1, typename T2, 
             void (MapperManager::*CALL)(T1*, T2*, bool)>
    class MapperContinuation2 : public MapperContinuation {
    public:
      MapperContinuation2(MapperManager *man, T1 *a1, T2 *a2)
        : MapperContinuation(man), arg1(a1), arg2(a2) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, arg2, false/*first*/); }
    public:
      T1 *const arg1;
      T2 *const arg2;
    };

    template<typename T1, typename T2, typename T3,
             void (MapperManager::*CALL)(T1*, T2*, T3*, bool)>
    class MapperContinuation3 : public MapperContinuation {
    public:
      MapperContinuation3(MapperManager *man, T1 *a1, T2 *a2, T3 *a3)
        : MapperContinuation(man), arg1(a1), arg2(a2), arg3(a3) { }
    public:
      virtual void execute(void)
      { (manager->*CALL)(arg1, arg2, arg3, false/*first*/); }
    public:
      T1 *const arg1;
      T2 *const arg2;
      T3 *const arg3;
    };

  };
};

#endif // __MAPPER_MANAGER_H__
