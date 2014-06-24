/* Copyright 2014 Stanford University
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



#ifndef __LEGION_LOGGING_H__
#define __LEGION_LOGGING_H__

#ifdef LEGION_LOGGING

#include "lowlevel.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"

#include <cassert>
#include <deque>
#include <sstream>
#include <string>
#include <iostream>

#define LEGION_LOGGING_CHECK_NO_EVENT

namespace LegionRuntime {

  // These need to be in this namespace so they
  // are visible to both the high- and low-level 
  // namespaces.
  enum TimingKind {
    // High-Level Timing Kinds 
    BEGIN_DEPENDENCE_ANALYSIS = 0,
    END_DEPENDENCE_ANALYSIS = 1,
    BEGIN_PRE_MAPPING = 2,
    END_PRE_MAPPING = 3,
    BEGIN_MAPPING = 4,
    END_MAPPING = 5,
    BEGIN_SLICING = 6,
    END_SLICING = 7,
    BEGIN_EXECUTION = 8,
    END_EXECUTION = 9,
    LAUNCH_TASK = 10,
    RESOLVE_SPECULATION = 11,
    COMPLETE_OPERATION = 12,
    COMMIT_OPERATION = 13,
    BEGIN_WINDOW_WAIT = 14,
    END_WINDOW_WAIT = 15,
    BEGIN_SCHEDULING = 16,
    END_SCHEDULING = 17,
    // Low-Level Timing Kinds
    COPY_INIT = 18,
    COPY_READY = 19,
    COPY_BEGIN = 20,
    COPY_END = 21,
    // High-Level Timing Kinds
    BEGIN_GC = 22,
    END_GC = 23,
    BEGIN_POST_EXEC = 24,
    END_POST_EXEC = 25,
  };
  
  namespace HighLevel { 

    /**
     * \namespace LegionLogging
     * Namespace for profiling functionality including all the necessary
     * logging infrastructure.  For performance reasons, messages get
     * buffered in memory during execution and are written to disk only
     * at the end of the program execution.
     */
    namespace LegionLogging {

      //========================================================================
      // Logging message types
      //========================================================================

      struct LogMsgProcessor {
      public:
        LogMsgProcessor(AddressSpaceID address_space, Processor p) :
            address_space(address_space), proc(p)
        {
        }
      public:
        AddressSpaceID address_space;
        Processor proc;
      };

      struct LogMsgMemory {
      public:
        LogMsgMemory(Memory mem, Memory::Kind kind, size_t capacity) :
            mem(mem), kind(kind), capacity(capacity)
        {
        }
      public:
        Memory mem;
        Memory::Kind kind;
        size_t capacity;
      };

      struct LogMsgProcMemAffinity {
      public:
        LogMsgProcMemAffinity(Processor proc, Memory mem, size_t bandwidth,
                              size_t latency) :
            proc(proc), mem(mem), bandwidth(bandwidth), latency(latency)
        {
        }
      public:
        Processor proc;
        Memory mem;
        size_t bandwidth;
        size_t latency;
      };

      struct LogMsgMemMemAffinity {
      public:
        LogMsgMemMemAffinity(Memory mem1, Memory mem2, size_t bandwidth,
                             size_t latency) :
            mem1(mem1), mem2(mem2), bandwidth(bandwidth), latency(latency)
        {
        }
      public:
        Memory mem1;
        Memory mem2;
        size_t bandwidth;
        size_t latency;
      };

      enum LogMsgOperationKind {
        MAPPING_OPERATION = 0,
        COPY_OPERATION = 1,
        FENCE_OPERATION = 2,
        DELETION_OPERATION = 3,
        CLOSE_OPERATION = 4,
      };

      struct LogMsgOperation {
      public:
        LogMsgOperation(Processor proc, LogMsgOperationKind k, UniqueID id, UniqueID context, unsigned long long time) :
            proc(proc), kind(k), unique_op_id(id), context(context), time(time)
        {
        }
      public:
        Processor proc;
        LogMsgOperationKind kind;
        UniqueID unique_op_id;
        UniqueID context;
        unsigned long long time;
      };

      struct LogMsgTaskOperation {
      public:
        LogMsgTaskOperation(Processor proc, bool i, UniqueID id, Processor::TaskFuncID tid, UniqueID context, unsigned long long time, MappingTagID tag) :
            proc(proc), isIndividual(i), unique_op_id(id), task_id(tid), context(context), time(time), tag(tag)
        {
        }
      public:
        Processor proc;
        bool isIndividual;
        UniqueID unique_op_id;
        Processor::TaskFuncID task_id;
        UniqueID context;
        unsigned long long time;
        MappingTagID tag;
      };

      struct LogMsgTaskInstanceVariant {
      public:
        LogMsgTaskInstanceVariant(UniqueID unique_op_id, VariantID vid) :
            unique_op_id(unique_op_id), vid(vid)
        {
        }
      public:
        UniqueID unique_op_id;
        VariantID vid;
      };

      struct LogMsgTaskCollection {
      public:
        LogMsgTaskCollection(Processor::TaskFuncID task_id, bool leaf,
                             bool idempotent, const char *name) :
            task_id(task_id), leaf(leaf), idempotent(idempotent), name(name)
        {
        }
      public:
        Processor::TaskFuncID task_id;
        bool leaf;
        bool idempotent;
        const char *name;
      };

      struct LogMsgTaskVariant {
      public:
        LogMsgTaskVariant(Processor::TaskFuncID task_id,
                          Processor::Kind proc_kind, bool single_task,
                          bool index_task, VariantID vid) :
            task_id(task_id), proc_kind(proc_kind), single_task(single_task), index_task(
                index_task), vid(vid)
        {
        }
      public:
        Processor::TaskFuncID task_id;
        Processor::Kind proc_kind;
        bool single_task;
        bool index_task;
        VariantID vid;
      };

      struct LogMsgTopLevelTask {
      public:
        LogMsgTopLevelTask(Processor::TaskFuncID task_id, UniqueID unique_op_id) :
            task_id(task_id), unique_op_id(unique_op_id)
        {
        }
      public:
        Processor::TaskFuncID task_id;
        UniqueID unique_op_id;
      };

      struct LogMsgIndexSlice {
      public:
        LogMsgIndexSlice(UniqueID index_id, UniqueID slice_id) :
            index_id(index_id), slice_id(slice_id)
        {
        }
      public:
        UniqueID index_id;
        UniqueID slice_id;
      };

      struct LogMsgSliceSlice {
      public:
        LogMsgSliceSlice(UniqueID slice_parent, UniqueID slice_subslice) :
            slice_parent(slice_parent), slice_subslice(slice_subslice)
        {
        }
      public:
        UniqueID slice_parent;
        UniqueID slice_subslice;
      };

      struct LogMsgPointPoint {
      public:
        LogMsgPointPoint(UniqueID orig_point, UniqueID new_point) :
            orig_point(orig_point), new_point(new_point)
        {
        }
      public:
        UniqueID orig_point;
        UniqueID new_point;
      };

      struct LogMsgSlicePoint {
      public:
        LogMsgSlicePoint(UniqueID slice_id, UniqueID point_id,
                         const DomainPoint &point) :
            slice_id(slice_id), point_id(point_id), point(point)
        {
        }
      public:
        UniqueID slice_id;
        UniqueID point_id;
        const DomainPoint &point;
      };

      struct LogMsgOperationTiming {
      public:
        LogMsgOperationTiming(Processor proc, UniqueID unique_op_id, TimingKind kind,
                              unsigned long long time) :
            proc(proc), unique_op_id(unique_op_id), kind(kind), time(time)
        {
        }
      public:
        Processor proc;
        UniqueID unique_op_id;
        TimingKind kind;
        unsigned long long time;
      };

      struct LogMsgEventTiming {
      public:
        LogMsgEventTiming(Processor proc, Event event, TimingKind kind, unsigned long long time) :
            proc(proc), event(event), kind(kind), time(time)
        {
        }
      public:
        Processor proc;
        Event event;
        TimingKind kind;
        unsigned long long time;
      };

      enum WaitKind {
        WAIT_BEGIN = 0,
        WAIT_END = 1,
        WAIT_NOWAIT = 2, // indicates that the resource being waited on is already ready
      };

      struct LogMsgFutureWait {
      public:
        LogMsgFutureWait(Processor proc, UniqueID context, UniqueID wait_on, WaitKind kind, unsigned long long time) :
            proc(proc), context(context), wait_on(wait_on), kind(kind), time(time)
        {
        }
      public:
        Processor proc;
        UniqueID context;
        UniqueID wait_on;
        WaitKind kind;
        unsigned long long time;
      };

      struct LogMsgInlineWait {
      public:
        LogMsgInlineWait(Processor proc, UniqueID context, Event wait_on, WaitKind kind, unsigned long long time) :
            proc(proc), context(context), wait_on(wait_on), kind(kind), time(time)
        {
        }
      public:
        Processor proc;
        UniqueID context;
        Event wait_on;
        WaitKind kind;
        unsigned long long time;
      };

      struct LogMsgTopIndexSpace {
      public:
        LogMsgTopIndexSpace(IndexSpace space) :
            space(space)
        {
        }
      public:
        IndexSpace space;
      };

      struct LogMsgIndexPartition {
      public:
        LogMsgIndexPartition(IndexSpace parent, IndexPartition handle,
                             bool disjoint, Color color) :
            parent(parent), handle(handle), disjoint(disjoint), color(color)
        {
        }
      public:
        IndexSpace parent;
        IndexPartition handle;
        bool disjoint;
        Color color;
      };

      struct LogMsgIndexSubspace {
      public:
        LogMsgIndexSubspace(IndexPartition parent, IndexSpace handle,
                            Color color) :
            parent(parent), handle(handle), color(color)
        {
        }
      public:
        IndexPartition parent;
        IndexSpace handle;
        Color color;
      };

      struct LogMsgFieldSpace {
      public:
        LogMsgFieldSpace(FieldSpace handle) :
            handle(handle)
        {
        }
      public:
        FieldSpace handle;
      };

      struct LogMsgFieldCreation {
      public:
        LogMsgFieldCreation(FieldSpace handle, FieldID fid, bool local) :
            handle(handle), fid(fid), local(local)
        {
        }
      public:
        FieldSpace handle;
        FieldID fid;
        bool local;
      };

      struct LogMsgTopRegion {
      public:
        LogMsgTopRegion(IndexSpace ispace, FieldSpace fspace, RegionTreeID tid) :
            ispace(ispace), fspace(fspace), tid(tid)
        {
        }
      public:
        IndexSpace ispace;
        FieldSpace fspace;
        RegionTreeID tid;
      };

      struct LogMsgLogicalRequirement {
      public:
        LogMsgLogicalRequirement(UniqueID unique_op_id, unsigned index,
                                 bool region, unsigned index_component,
                                 unsigned field_component, RegionTreeID tid,
                                 PrivilegeMode privilege,
                                 CoherenceProperty prop, ReductionOpID redop) :
            unique_op_id(unique_op_id), index(index), region(region), index_component(
                index_component), field_component(field_component), tid(tid), privilege(
                privilege), prop(prop), redop(redop)
        {
        }
      public:
        UniqueID unique_op_id;
        unsigned index;
        bool region;
        unsigned index_component;
        unsigned field_component;
        RegionTreeID tid;
        PrivilegeMode privilege;
        CoherenceProperty prop;
        ReductionOpID redop;
      };

      struct LogMsgRequirementFields {
      public:
        LogMsgRequirementFields(UniqueID unique_op_id, unsigned index,
                                const std::set<FieldID> &logical_fields) :
            unique_op_id(unique_op_id), index(index), logical_fields(
                logical_fields)
        {
        }
      public:
        UniqueID unique_op_id;
        unsigned index;
        const std::set<FieldID> &logical_fields;
      };

      struct LogMsgMappingDependence {
      public:
        LogMsgMappingDependence(UniqueID parent_context, UniqueID previous_id,
                                unsigned previous_index, UniqueID next_id,
                                unsigned next_index, DependenceType dep_type) :
            parent_context(parent_context), previous_id(previous_id), previous_index(
                previous_index), next_id(next_id), next_index(next_index), dep_type(
                dep_type)
        {
        }
      public:
        UniqueID parent_context;
        UniqueID previous_id;
        unsigned previous_index;
        UniqueID next_id;
        unsigned next_index;
        DependenceType dep_type;
      };

      struct LogMsgTaskInstanceRequirement {
      public:
        LogMsgTaskInstanceRequirement(UniqueID unique_id, unsigned index,
                                      IndexSpace handle) :
            unique_id(unique_id), index(index), handle(handle)
        {
        }
      public:
        UniqueID unique_id;
        unsigned index;
        IndexSpace handle;
      };

      struct LogMsgEventDependency {
      public:
        LogMsgEventDependency(Event one, Event two) :
            one(one), two(two)
        {
        }
      public:
        Event one;
        Event two;
      };

      struct LogMsgOperationEvents {
      public:
        LogMsgOperationEvents(UniqueID unique_op_id, Event start_event,
                              Event end_event) :
            unique_op_id(unique_op_id), start_event(start_event), end_event(
                end_event)
        {
        }
      public:
        UniqueID unique_op_id;
        Event start_event;
        Event end_event;
      };

      struct LogMsgPhysicalInstance {
      public:
        LogMsgPhysicalInstance(PhysicalInstance instance, Memory memory,
                               IndexSpace index_handle, FieldSpace field_handle,
                               RegionTreeID tree_id, ReductionOpID redop, bool fold,
                               Domain indirect_domain) :
            instance(instance), memory(memory), index_handle(index_handle), field_handle(
                field_handle), tree_id(tree_id), redop(redop), fold(fold), indirect_domain(
                indirect_domain)
        {
        }
      public:
        PhysicalInstance instance;
        Memory memory;
        IndexSpace index_handle;
        FieldSpace field_handle;
        RegionTreeID tree_id;
        ReductionOpID redop;
        bool fold;
        Domain indirect_domain;
      };

      struct LogMsgPhysicalUser {
      public:
        LogMsgPhysicalUser(PhysicalInstance instance, UniqueID unique_op_id,
                           unsigned index) :
            instance(instance), unique_op_id(unique_op_id), index(index)
        {
        }
      public:
        PhysicalInstance instance;
        UniqueID unique_op_id;
        unsigned index;
      };

      struct LogMsgLowlevelCopy {
      public:
        LogMsgLowlevelCopy(PhysicalInstance src_instance,
                           PhysicalInstance dst_instance,
                           IndexSpace index_handle, FieldSpace field_handle,
                           RegionTreeID tree_id, Event start_event,
                           Event termination_event,
                           std::string fields, ReductionOpID redop) :
            src_instance(src_instance), dst_instance(dst_instance), index_handle(
                index_handle), field_handle(field_handle), tree_id(tree_id), start_event(
                start_event), termination_event(termination_event), fields(
                fields), redop(redop)
        {
        }
      public:
        PhysicalInstance src_instance;
        PhysicalInstance dst_instance;
        IndexSpace index_handle;
        FieldSpace field_handle;
        RegionTreeID tree_id;
        Event start_event;
        Event termination_event;
        std::string fields;
        ReductionOpID redop;
      };

      struct LogMsgLowlevelCopyAssoc {
      public:
        LogMsgLowlevelCopyAssoc(Event highlevel_end_event, Event lowlevel_end_event, unsigned channel_id) :
            highlevel_end_event(highlevel_end_event), lowlevel_end_event(lowlevel_end_event), channel_id(channel_id)
        {
        }
      public:
        Event highlevel_end_event;
        Event lowlevel_end_event;
        unsigned channel_id;
      };

      struct LogMsgLowlevelCopyChannel {
      public:
        LogMsgLowlevelCopyChannel(unsigned id, const char* name) :
            id(id), name(name)
        {
        }
      public:
        unsigned id;
        const char* name;
      };

      //========================================================================
      // Profiler and in-memory buffering
      //========================================================================

      // TODO: this is no longer per processor, but per thread, so a different
      //  name might be more appropriate
      struct ProcessorProfiler {
      public:
        ProcessorProfiler() {

        }
      public:
        void add_msg(const LogMsgOperation &msg) { msgs_operation.push_back(msg); }
        void add_msg(const LogMsgTaskOperation &msg) { msgs_task_operation.push_back(msg); }
        void add_msg(const LogMsgTaskInstanceVariant &msg) { msgs_task_instance_variant.push_back(msg); }
        void add_msg(const LogMsgIndexSlice &msg) { msgs_index_slice.push_back(msg); }
        void add_msg(const LogMsgSliceSlice &msg) { msgs_slice_slice.push_back(msg); }
        void add_msg(const LogMsgPointPoint &msg) { msgs_point_point.push_back(msg); }
        void add_msg(const LogMsgSlicePoint &msg) { msgs_slice_point.push_back(msg); }
        void add_msg(const LogMsgOperationTiming &msg) { msgs_operation_timing.push_back(msg); }
        void add_msg(const LogMsgEventTiming &msg) { msgs_event_timing.push_back(msg); }
        void add_msg(const LogMsgFutureWait &msg) { msgs_future_wait.push_back(msg); }
        void add_msg(const LogMsgInlineWait &msg) { msgs_inline_wait.push_back(msg); }
        void add_msg(const LogMsgTopIndexSpace &msg) { msgs_top_index_space.push_back(msg); }
        void add_msg(const LogMsgIndexPartition &msg) { msgs_index_partition.push_back(msg); }
        void add_msg(const LogMsgIndexSubspace &msg) { msgs_index_subspace.push_back(msg); }
        void add_msg(const LogMsgFieldSpace &msg) { msgs_field_space.push_back(msg); }
        void add_msg(const LogMsgFieldCreation &msg) { msgs_field_creation.push_back(msg); }
        void add_msg(const LogMsgTopRegion &msg) { msgs_top_region.push_back(msg); }
        void add_msg(const LogMsgLogicalRequirement &msg) { msgs_logical_requirement.push_back(msg); }
        void add_msg(const LogMsgRequirementFields &msg) { msgs_requirement_fields.push_back(msg); }
        void add_msg(const LogMsgMappingDependence &msg) { msgs_mapping_dependence.push_back(msg); }
        void add_msg(const LogMsgTaskInstanceRequirement &msg) { msgs_task_instance_requirement.push_back(msg); }
        void add_msg(const LogMsgEventDependency &msg) { msgs_event_dependency.push_back(msg); }
        void add_msg(const LogMsgOperationEvents &msg) { msgs_operation_events.push_back(msg); }
        void add_msg(const LogMsgPhysicalInstance &msg) { msgs_physical_instance.push_back(msg); }
        void add_msg(const LogMsgPhysicalUser &msg) { msgs_phyiscal_user.push_back(msg); }
        void add_msg(const LogMsgLowlevelCopy &msg) { msgs_lowlevel_copy.push_back(msg); }
        void add_msg(const LogMsgLowlevelCopyAssoc &msg) { msgs_lowlevel_copy_assoc.push_back(msg); }
        void add_msg(const LogMsgLowlevelCopyChannel &msg) { msgs_lowlevel_copy_channel.push_back(msg); }
      private:
        // no copy constructor or assignment
        ProcessorProfiler(const ProcessorProfiler& copy_from) {assert(false);}
        ProcessorProfiler& operator=(const ProcessorProfiler& copy_from) {assert(false);}
      public:
        Processor proc;
        std::deque<LogMsgOperation> msgs_operation;
        std::deque<LogMsgTaskOperation> msgs_task_operation;
        std::deque<LogMsgTaskInstanceVariant> msgs_task_instance_variant;
        std::deque<LogMsgIndexSlice> msgs_index_slice;
        std::deque<LogMsgSliceSlice> msgs_slice_slice;
        std::deque<LogMsgPointPoint> msgs_point_point;  
        std::deque<LogMsgSlicePoint> msgs_slice_point;
        std::deque<LogMsgOperationTiming> msgs_operation_timing;
        std::deque<LogMsgEventTiming> msgs_event_timing;
        std::deque<LogMsgFutureWait> msgs_future_wait;
        std::deque<LogMsgInlineWait> msgs_inline_wait;
        std::deque<LogMsgTopIndexSpace> msgs_top_index_space;
        std::deque<LogMsgIndexPartition> msgs_index_partition;
        std::deque<LogMsgIndexSubspace> msgs_index_subspace;
        std::deque<LogMsgFieldSpace> msgs_field_space;
        std::deque<LogMsgFieldCreation> msgs_field_creation;
        std::deque<LogMsgTopRegion> msgs_top_region;
        std::deque<LogMsgLogicalRequirement> msgs_logical_requirement;
        std::deque<LogMsgRequirementFields> msgs_requirement_fields;
        std::deque<LogMsgMappingDependence> msgs_mapping_dependence;
        std::deque<LogMsgTaskInstanceRequirement> msgs_task_instance_requirement;
        std::deque<LogMsgEventDependency> msgs_event_dependency;
        std::deque<LogMsgOperationEvents> msgs_operation_events;
        std::deque<LogMsgPhysicalInstance> msgs_physical_instance;
        std::deque<LogMsgPhysicalUser> msgs_phyiscal_user;
        std::deque<LogMsgLowlevelCopy> msgs_lowlevel_copy;
        std::deque<LogMsgLowlevelCopyAssoc> msgs_lowlevel_copy_assoc;
        std::deque<LogMsgLowlevelCopyChannel> msgs_lowlevel_copy_channel;
      };

      extern Logger::Category log_logging;
      // all profiler objects created so far
      extern std::list<ProcessorProfiler *> processor_profilers;
      extern pthread_key_t pthread_profiler_key;
      extern pthread_mutex_t profiler_mutex;
      // all message queues for sequential logger calls
      extern std::deque<LogMsgProcessor> msgs_processor;
      extern std::deque<LogMsgMemory> msgs_memory;
      extern std::deque<LogMsgProcMemAffinity> msgs_proc_mem_affinity;
      extern std::deque<LogMsgMemMemAffinity> msgs_mem_mem_affinity;
      extern std::deque<LogMsgTaskCollection> msgs_task_collection;
      extern std::deque<LogMsgTaskVariant> msgs_task_variant;
      extern std::deque<LogMsgTopLevelTask> msgs_top_level_task;
      extern unsigned long long init_time;
      extern AddressSpaceID address_space;

      static inline ProcessorProfiler& get_profiler()
      {
        ProcessorProfiler *profiler = (ProcessorProfiler*) pthread_getspecific(pthread_profiler_key);
        if(!profiler) {
          profiler = new ProcessorProfiler();
          pthread_setspecific(pthread_profiler_key, profiler);
          pthread_mutex_lock(&profiler_mutex);
          // keep a list of all profiler objects
          processor_profilers.push_back(profiler);
          pthread_mutex_unlock(&profiler_mutex);
        }
        return *profiler;
      }

      //========================================================================
      // Logger calls for initialization 
      //========================================================================
      // Sequential
      static void initialize_legion_logging(AddressSpaceID sid,
                                            const Processor processor)
      {
        for(std::deque<LogMsgProcessor>::iterator it = msgs_processor.begin(); it != msgs_processor.end(); ++it) {
          if (it->proc.id == processor.id && it->address_space == sid)
            return; // already recorded this processor
        }
        ProcessorProfiler &p = get_profiler();
        msgs_processor.push_back(LogMsgProcessor(sid, processor));

        Processor utility_proc = processor.get_utility_processor();
        initialize_legion_logging(sid, utility_proc);
      }
      // Sequential
      // Called in Runtime constructor in runtime.cc
      static void initialize_legion_logging(AddressSpaceID sid,
                                        const std::set<Processor> &processors)
      {
        init_time = TimeStamp::get_current_time_in_micros();
        address_space = sid;
        pthread_key_create(&pthread_profiler_key, NULL);
        for(std::set<Processor>::iterator it = processors.begin(); it != processors.end(); ++it)
        {
          initialize_legion_logging(sid, *it);
        }
      }
      // Sequential
      // Called in Runtime destructor in runtime.cc
      static void finalize_legion_logging(const std::set<Processor> &processors)
      {
        log_logging(LEVEL_INFO, "nodeinfo: {\"address_space\": %u, \"init_time\":%llu}", address_space, init_time);
        // global log messages
        for (unsigned idx = 0; idx < msgs_processor.size(); idx++)
        {
          LogMsgProcessor &msg = msgs_processor[idx];
          Machine *machine = Machine::get_machine();
          log_logging(LEVEL_INFO, "processor: {\"address_space\": %d, \"id\":%u, \"utility\":%u, \"kind\":%d}", msg.address_space, msg.proc.id, msg.proc.get_utility_processor().id, machine->get_processor_kind(msg.proc));
        }
        for (unsigned idx = 0; idx < msgs_memory.size(); idx++)
        {
          LogMsgMemory &msg = msgs_memory[idx];
          log_logging(LEVEL_INFO, "memory: {\"id\":%d, \"kind\":%d, \"capacity\":%ld}", msg.mem.id, msg.kind, msg.capacity);
        }
        for (unsigned idx = 0; idx < msgs_proc_mem_affinity.size(); idx++)
        {
          LogMsgProcMemAffinity &msg = msgs_proc_mem_affinity[idx];
          log_logging(LEVEL_INFO, "proc_mem_affinity: {\"proc\":%u, \"mem\":%d, \"bandwidth\":%ld, \"latency\":%ld}", msg.proc.id, msg.mem.id, msg.bandwidth, msg.latency);
        }
        for (unsigned idx = 0; idx < msgs_mem_mem_affinity.size(); idx++)
        {
          LogMsgMemMemAffinity &msg = msgs_mem_mem_affinity[idx];
          log_logging(LEVEL_INFO, "mem_mem_affinity: {\"mem1\":%d, \"mem2\":%d, \"bandwidth\":%ld, \"latency\":%ld}", msg.mem1.id, msg.mem2.id, msg.bandwidth, msg.latency);
        }
        for (unsigned idx = 0; idx < msgs_task_collection.size(); idx++)
        {
          LogMsgTaskCollection &msg = msgs_task_collection[idx];
          log_logging(LEVEL_INFO, "task_collection: {\"task_id\":%d, \"leaf\":%d, \"idempotent\":%d, \"name\":\"%s\"}", msg.task_id, msg.leaf, msg.idempotent, msg.name);
        }
        for (unsigned idx = 0; idx < msgs_task_variant.size(); idx++)
        {
          LogMsgTaskVariant &msg = msgs_task_variant[idx];
          log_logging(LEVEL_INFO, "task_variant: {\"task_id\":%d, \"proc_kind\":%d, \"single_task\":%d, \"index_task\":%d, \"vid\":%ld}", msg.task_id, msg.proc_kind, msg.single_task, msg.index_task, msg.vid);
        }
        for (unsigned idx = 0; idx < msgs_top_level_task.size(); idx++)
        {
          LogMsgTopLevelTask &msg = msgs_top_level_task[idx];
          log_logging(LEVEL_INFO, "top_level_task: {\"task_id\":%d, \"unique_op_id\":%lld}", msg.task_id, msg.unique_op_id);
        }
        // per thread log messages
        for (std::list<ProcessorProfiler *>::iterator it=processor_profilers.begin(); it != processor_profilers.end(); ++it)
        {
          ProcessorProfiler &prof = **it;
          for (unsigned idx = 0; idx < prof.msgs_task_operation.size(); idx++)
          {
            LogMsgTaskOperation &msg = prof.msgs_task_operation[idx];
            log_logging(LEVEL_INFO, "task_operation: {\"processor\":%u, \"is_individual\":%d, \"unique_op_id\":%lld, \"task_id\":%d, \"context\":%lld, \"time\":%lld, \"tag\":%lu}", msg.proc.id, msg.isIndividual, msg.unique_op_id, msg.task_id, msg.context, msg.time, msg.tag);
          }
          for (unsigned idx = 0; idx < prof.msgs_operation.size(); idx++)
          {
            LogMsgOperation &msg = prof.msgs_operation[idx];
            const char * kind;
            switch (msg.kind)
            {
              case MAPPING_OPERATION: kind = "MAPPING_OPERATION"; break;
              case COPY_OPERATION: kind = "COPY_OPERATION"; break;
              case FENCE_OPERATION: kind = "FENCE_OPERATION"; break;
              case DELETION_OPERATION: kind = "DELETION_OPERATION"; break;
              case CLOSE_OPERATION: kind = "CLOSE_OPERATION"; break;
              default: kind = "UNKNOWN_OPERATION"; break;
            }
            log_logging(LEVEL_INFO, "operation: {\"processor\":%u, \"kind\":\"%s\", \"unique_op_id\":%lld, \"context\":%lld, \"time\":%lld}", msg.proc.id, kind, msg.unique_op_id, msg.context, msg.time);
          }
          for (unsigned idx = 0; idx < prof.msgs_task_instance_variant.size(); idx++) {
            LogMsgTaskInstanceVariant &msg = prof.msgs_task_instance_variant[idx];
            log_logging(LEVEL_INFO, "task_instance_variant: {\"unique_op_id\":%lld, \"vid\":%ld}", msg.unique_op_id, msg.vid);
          }
          for (unsigned idx = 0; idx < prof.msgs_index_slice.size(); idx++)
          {
            LogMsgIndexSlice &msg = prof.msgs_index_slice[idx];
            log_logging(LEVEL_INFO, "index_slice: {\"index_id\":%lld, \"slice_id\":%lld}", msg.index_id, msg.slice_id);
          }
          for (unsigned idx = 0; idx < prof.msgs_slice_slice.size(); idx++)
          {
            LogMsgSliceSlice &msg = prof.msgs_slice_slice[idx];
            log_logging(LEVEL_INFO, "slice_slice: {\"slice_parent\":%lld, \"slice_subslice\":%lld}", msg.slice_parent, msg.slice_subslice);
          }
          for (unsigned idx = 0; idx < prof.msgs_point_point.size(); idx++)
          {
            LogMsgPointPoint &msg = prof.msgs_point_point[idx];
            log_logging(LEVEL_INFO, "point_point: {\"orig_point\":%lld, \"new_point\":%lld}", msg.orig_point, msg.new_point);
          }
          for (unsigned idx = 0; idx < prof.msgs_slice_point.size(); idx++)
          {
            LogMsgSlicePoint &msg = prof.msgs_slice_point[idx];
            log_logging(LEVEL_INFO, "slice_point: {\"slice_id\":%lld, \"point_id\":%lld, \"dim\":%d, \"d0\":%d, \"d1\":%d, \"d2\":%d}", msg.slice_id, msg.point_id, msg.point.dim, msg.point.point_data[0], msg.point.point_data[1], msg.point.point_data[2]);
          }
          for (unsigned idx = 0; idx < prof.msgs_lowlevel_copy.size(); idx++)
          {
            LogMsgLowlevelCopy &msg = prof.msgs_lowlevel_copy[idx];
            log_logging(LEVEL_INFO, "lowlevel_copy: {\"src_instance\":%u, \"dst_instance\":%u, \"index_handle\":%u, \"field_handle\":%d, \"tree_id\":%u, \"begin_event_id\":%d, \"begin_event_gen\":%d, \"end_event_id\":%d, \"end_event_gen\":%d, \"redop\":%d, \"fields\":\"%s\"}", msg.src_instance.id, msg.dst_instance.id, msg.index_handle.id, msg.field_handle.get_id(), msg.tree_id, msg.start_event.id, msg.start_event.gen, msg.termination_event.id, msg.termination_event.gen, msg.redop, msg.fields.c_str());
          }
          for (unsigned idx = 0; idx < prof.msgs_operation_timing.size(); idx++)
          {
            LogMsgOperationTiming &msg = prof.msgs_operation_timing[idx];
            log_logging(LEVEL_INFO, "operation_timing: {\"processor\":%u, \"unique_op_id\":%lld, \"kind\":%d, \"time\":%lld}", msg.proc.id, msg.unique_op_id, msg.kind, msg.time);
          }
          for (unsigned idx = 0; idx < prof.msgs_event_timing.size(); idx++)
          {
            LogMsgEventTiming &msg = prof.msgs_event_timing[idx];
            log_logging(LEVEL_INFO, "event_timing: {\"address_space\": %u, \"processor\":%u, \"event_id\":%d, \"event_gen\":%d, \"kind\":%d, \"time\":%lld}", address_space, msg.proc.id, msg.event.id, msg.event.gen, msg.kind, msg.time);
          }
          for (unsigned idx = 0; idx < prof.msgs_future_wait.size(); idx++)
          {
            LogMsgFutureWait &msg = prof.msgs_future_wait[idx];
            log_logging(LEVEL_INFO, "future_wait: {\"processor\":%u, \"context\":%lld, \"wait_on\":%lld, \"kind\":%d, \"time\":%lld}", msg.proc.id, msg.context, msg.wait_on, msg.kind, msg.time);
          }
          for (unsigned idx = 0; idx < prof.msgs_inline_wait.size(); idx++)
          {
            LogMsgInlineWait &msg = prof.msgs_inline_wait[idx];
            log_logging(LEVEL_INFO, "inline_wait: {\"processor\":%u, \"context\":%lld, \"event_id\":%d, \"event_gen\":%d, \"kind\":%d, \"time\":%lld}", msg.proc.id, msg.context, msg.wait_on.id, msg.wait_on.gen, msg.kind, msg.time);
          }
          for (unsigned idx = 0; idx < prof.msgs_top_index_space.size(); idx++)
          {
            LogMsgTopIndexSpace &msg = prof.msgs_top_index_space[idx];
            log_logging(LEVEL_INFO, "top_index_space: {\"id\":%d}", msg.space.id);
          }
          for (unsigned idx = 0; idx < prof.msgs_index_partition.size(); idx++)
          {
            LogMsgIndexPartition &msg = prof.msgs_index_partition[idx];
            log_logging(LEVEL_INFO, "index_partition: {\"parent\":%d, \"handle\":%d, \"disjoint\":%d, \"color\":%d}", msg.parent.id, msg.handle, msg.disjoint, msg.color);
          }
          for (unsigned idx = 0; idx < prof.msgs_index_subspace.size(); idx++)
          {
            LogMsgIndexSubspace &msg = prof.msgs_index_subspace[idx];
            log_logging(LEVEL_INFO, "index_subspace: {\"parent\":%d, \"handle\":%d, \"color\":%d}", msg.parent, msg.handle.id, msg.color);
          }
          for (unsigned idx = 0; idx < prof.msgs_field_space.size(); idx++)
          {
            LogMsgFieldSpace &msg = prof.msgs_field_space[idx];
            log_logging(LEVEL_INFO, "field_space: {\"handle\":%d}", msg.handle.get_id());
          }
          for (unsigned idx = 0; idx < prof.msgs_field_creation.size(); idx++)
          {
            LogMsgFieldCreation &msg = prof.msgs_field_creation[idx];
            log_logging(LEVEL_INFO, "field_creation: {\"handle\":%d, \"fid\":%d, \"local\":%d}", msg.handle.get_id(), msg.fid, msg.local);
          }
          for (unsigned idx = 0; idx < prof.msgs_top_region.size(); idx++)
          {
            LogMsgTopRegion &msg = prof.msgs_top_region[idx];
            log_logging(LEVEL_INFO, "top_region: {\"ispace\":%d, \"fspace\":%d, \"tid\":%d}", msg.ispace.id, msg.fspace.get_id(), msg.tid);
          }
          for (unsigned idx = 0; idx < prof.msgs_logical_requirement.size(); idx++)
          {
            LogMsgLogicalRequirement &msg = prof.msgs_logical_requirement[idx];
            log_logging(LEVEL_INFO, "logical_requirement: {\"unique_op_id\":%lld, \"index\":%d, \"region\":%d, \"index_component\":%d, \"field_component\":%d, \"tid\":%d, \"privilege\":%d, \"prop\":%d, \"redop\":%d}", msg.unique_op_id, msg.index, msg.region, msg.index_component, msg.field_component, msg.tid, msg.privilege, msg.prop, msg.redop);
          }
          for (unsigned idx = 0; idx < prof.msgs_requirement_fields.size(); idx++)
          {
            LogMsgRequirementFields &msg = prof.msgs_requirement_fields[idx];
            for (std::set<FieldID>::iterator it = msg.logical_fields.begin(); it!=msg.logical_fields.end(); ++it)
              log_logging(LEVEL_INFO, "requirement_fields: {\"unique_op_id\":%lld, \"index\":%d, \"logical_field\":%d}", msg.unique_op_id, msg.index, *it);
          }
          for (unsigned idx = 0; idx < prof.msgs_mapping_dependence.size(); idx++)
          {
            LogMsgMappingDependence &msg = prof.msgs_mapping_dependence[idx];
            log_logging(LEVEL_INFO, "mapping_dependence: {\"parent_context\":%lld, \"previous_id\":%lld, \"previous_index\":%d, \"next_id\":%lld, \"next_index\":%d, \"dep_type\":%d}", msg.parent_context, msg.previous_id, msg.previous_index, msg.next_id, msg.next_index, msg.dep_type);
          }
          for (unsigned idx = 0; idx < prof.msgs_task_instance_requirement.size(); idx++)
          {
            LogMsgTaskInstanceRequirement &msg = prof.msgs_task_instance_requirement[idx];
            log_logging(LEVEL_INFO, "task_instance_requirement: {\"unique_id\":%lld, \"index\":%d, \"handle\":%d}", msg.unique_id, msg.index, msg.handle.id);
          }
          for (unsigned idx = 0; idx < prof.msgs_operation_events.size(); idx++)
          {
            LogMsgOperationEvents &msg = prof.msgs_operation_events[idx];
            log_logging(LEVEL_INFO, "operation_events: {\"unique_op_id\":%lld, \"begin_event_id\":%d, \"begin_event_gen\":%d, \"end_event_id\":%d, \"end_event_gen\":%d}", msg.unique_op_id, msg.start_event.id, msg.start_event.gen, msg.end_event.id, msg.end_event.gen);
          }
          for (unsigned idx = 0; idx < prof.msgs_event_dependency.size(); idx++)
          {
            LogMsgEventDependency &msg = prof.msgs_event_dependency[idx];
            log_logging(LEVEL_INFO, "event_dependency: {\"event1_id\":%d, \"event1_gen\":%d, \"event2_id\":%d, \"event2_gen\":%d}", msg.one.id, msg.one.gen, msg.two.id, msg.two.gen);
          }
          for (unsigned idx = 0; idx < prof.msgs_physical_instance.size(); idx++)
          {
            LogMsgPhysicalInstance &msg = prof.msgs_physical_instance[idx];
            log_logging(LEVEL_INFO, "physical_instance: {\"instance\":%d, \"memory\":%d, \"index_handle\":%d, \"field_handle\":%d, \"tree_id\":%d, \"redop\":%d, \"fold\":%d, \"indirect_domain\":%d}", msg.instance.id, msg.memory.id, msg.index_handle.id, msg.field_handle.get_id(), msg.tree_id, msg.redop, msg.fold, msg.indirect_domain.is_id);
          }
          for (unsigned idx = 0; idx < prof.msgs_phyiscal_user.size(); idx++)
          {
            LogMsgPhysicalUser &msg = prof.msgs_phyiscal_user[idx];
            log_logging(LEVEL_INFO, "physical_user: {\"instance\":%d, \"unique_op_id\":%lld, \"index\":%d}", msg.instance.id, msg.unique_op_id, msg.index);
          }
          for (unsigned idx = 0; idx < prof.msgs_lowlevel_copy_assoc.size(); idx++)
          {
            LogMsgLowlevelCopyAssoc &msg = prof.msgs_lowlevel_copy_assoc[idx];
            log_logging(LEVEL_INFO, "llcopy_assoc: {\"highlevel_id\":%d, \"highlevel_gen\":%d, \"lowlevel_id\":%d, \"lowlevel_gen\":%d, \"channel_id\":%u}", msg.highlevel_end_event.id, msg.highlevel_end_event.gen, msg.lowlevel_end_event.id, msg.lowlevel_end_event.gen, msg.channel_id);
          }
          for (unsigned idx = 0; idx < prof.msgs_lowlevel_copy_channel.size(); idx++)
          {
            LogMsgLowlevelCopyChannel &msg = prof.msgs_lowlevel_copy_channel[idx];
            log_logging(LEVEL_INFO, "llcopy_channel: {\"id\":%u, \"name\":%s}", msg.id, msg.name);
          }
        }
      }
      
      //========================================================================
      // Logger calls for the machine architecture
      //========================================================================

      // Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static void log_processor(Processor proc, Processor utility,
                                       Processor::Kind kind)
      {
      }

      // Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static void log_memory(Memory mem, Memory::Kind kind,
                                    size_t capacity)
      {
        msgs_memory.push_back(LogMsgMemory(mem, kind ,capacity));
      }

      // Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static void log_proc_mem_affinity(Processor proc, Memory mem,
                                               size_t bandwidth,
                                               size_t latency)
      {
        msgs_proc_mem_affinity.push_back(LogMsgProcMemAffinity(proc, mem, bandwidth, latency));
      }

      //Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static void log_mem_mem_affinity(Memory mem1, Memory mem2,
                                              size_t bandwidth,
                                              size_t latency)
      {
        msgs_mem_mem_affinity.push_back(LogMsgMemMemAffinity(mem1, mem2, bandwidth, latency));
      }

      //========================================================================
      // Logger calls for operations 
      //========================================================================

      // Thread-safe
      // Called in MapOp::initialize in legion_ops.cc
      static inline void log_mapping_operation(Processor p,
                                               UniqueID context,
                                               UniqueID unique_op_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperation(p, MAPPING_OPERATION, unique_op_id, context, time));
      }

      // Thread-safe
      // Called in CopyOp::initialize in legion_ops.cc
      static inline void log_copy_operation(Processor p,
                                            UniqueID context,
                                            UniqueID unique_op_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperation(p, COPY_OPERATION, unique_op_id, context, time));
      }

      // Thread-safe
      // Called in FenceOp::initialize in legion_ops.cc
      static inline void log_fence_operation(Processor p,
                                             UniqueID context,
                                             UniqueID unique_op_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperation(p, FENCE_OPERATION, unique_op_id, context, time));
      }

      // Thread-safe
      // Called in DeletionOp::initialize in legion_ops.cc
      static inline void log_deletion_operation(Processor p,
                                                UniqueID context,
                                                UniqueID unique_op_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperation(p, DELETION_OPERATION, unique_op_id, context, time));
      }

      // Thread-safe
      // Called in CloseOp::initialize in legion_ops.cc
      static inline void log_close_operation(Processor p,
                                             UniqueID context,
                                             UniqueID unique_op_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperation(p, CLOSE_OPERATION, unique_op_id, context, time));
      }

      // Thread-safe
      // Called in IndividualTask::initialize_task in legion_tasks.cc
      static inline void log_individual_task(Processor p,
                                             UniqueID context,
                                             UniqueID unique_op_id,
                                             Processor::TaskFuncID task_id,
                                             MappingTagID tag)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgTaskOperation(p, true, unique_op_id, task_id, context, time, tag));
      }

      // Thread-safe
      // Called in IndexTask::initialize_task in legion_tasks.cc
      static inline void log_index_space_task(Processor p,
                                              UniqueID context,
                                              UniqueID unique_op_id,
                                              Processor::TaskFuncID task_id,
                                              MappingTagID tag)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgTaskOperation(p, false, unique_op_id, task_id, context, time, tag));
      }

      //========================================================================
      // Logger calls for tasks
      //========================================================================

      // Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static inline void log_task_collection(Processor::TaskFuncID task_id,
                                             bool leaf, bool idempotent,
                                             const char *name)
      {
        msgs_task_collection.push_back(LogMsgTaskCollection(task_id, leaf, idempotent, name));
      }

      // Sequential
      // Called in Runtime::perform_one_time_logging in runtime.cc
      static inline void log_task_variant(Processor::TaskFuncID task_id,
                                          Processor::Kind proc_kind,
                                          bool single_task, bool index_task,
                                          VariantID vid)
      {
        msgs_task_variant.push_back(LogMsgTaskVariant(task_id, proc_kind, single_task, index_task, vid));
      }

      // Sequential
      // Called in Runtime::launch_top_level_task in runtime.cc
      static inline void log_top_level_task(Processor::TaskFuncID task_id,
                                            UniqueID unique_op_id)
      {
        msgs_top_level_task.push_back(LogMsgTopLevelTask(task_id, unique_op_id));
      }

      // Thread-safe
      // Called in IndexTask::clone_as_slice_task in legion_tasks.cc
      static inline void log_index_slice(Processor p, UniqueID index_id,
                                          UniqueID slice_id)
      {
        get_profiler().add_msg(LogMsgIndexSlice(index_id, slice_id));
      }

      // Thread-safe
      // Called in SliceTask::clone_as_slice_task in legion_tasks.cc
      // Called in SliceTask::unpack_task
      static inline void log_slice_slice(Processor p, UniqueID slice_parent,
                                          UniqueID slice_subslice)
      {
        get_profiler().add_msg(LogMsgSliceSlice(slice_parent, slice_subslice));
      }

      // Thread-safe
      // Called in SliceTask::clone_as_point_task in legion_tasks.cc
      // Called in SliceTask::unpack_task
      static inline void log_slice_point(Processor p, UniqueID slice_id,
                                          UniqueID point_id, 
                                          const DomainPoint &point)
      {
        get_profiler().add_msg(LogMsgSlicePoint(slice_id, point_id, point));
      }

      // Thread-safe
      // Called in IndividualTask::unpack_task
      static inline void log_point_point(Processor p, UniqueID orig_point,
                                         UniqueID new_point)
      {
        get_profiler().add_msg(LogMsgPointPoint(orig_point, new_point));
      }

      //========================================================================
      // Operation timing calls 
      //========================================================================
      
      // Thread-safe
      static inline void log_timing_event(Processor p, UniqueID unique_id,
                                          TimingKind kind)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgOperationTiming(p, unique_id, kind, time));
      }

      // Thread-safe
      static inline void log_timing_event(Processor p, Event event,
                                          TimingKind kind)
      {
#ifdef LEGION_LOGGING_CHECK_NO_EVENT
        assert(event.exists());
#endif
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgEventTiming(p, event, kind, time));
      }

      // Thread-safe
      static inline void log_future_wait_begin(Processor p, UniqueID context,
                                         UniqueID wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgFutureWait(p, context, wait_on, WAIT_BEGIN, time));
      }

      // Thread-safe
      // Calls to this method always follow a call to log_future_wait_begin
      // with the same context/wait_on parameter (making it possible to match
      // up the beginning and end of future waits).
      static inline void log_future_wait_end(Processor p, UniqueID context,
                                         UniqueID wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgFutureWait(p, context, wait_on, WAIT_END, time));
      }

      // Thread-safe
      static inline void log_future_nowait(Processor p, UniqueID context,
                                         UniqueID wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgFutureWait(p, context, wait_on, WAIT_NOWAIT, time));
      }

      // Thread-safe
      static inline void log_inline_wait_begin(Processor p, UniqueID context,
                                         Event wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgInlineWait(p, context, wait_on, WAIT_BEGIN, time));
      }

      // Thread-safe
      static inline void log_inline_wait_end(Processor p, UniqueID context,
                                         Event wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgInlineWait(p, context, wait_on, WAIT_END, time));
      }

      // Thread-safe
      static inline void log_inline_nowait(Processor p, UniqueID context,
                                         Event wait_on)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        get_profiler().add_msg(LogMsgInlineWait(p, context, wait_on, WAIT_NOWAIT, time));
      }

      //========================================================================
      // Logger calls for the shape of region trees
      //========================================================================

      // Thread-safe
      // Called in Runtime::create_index_space
      // Called in Runtime::create_index_space with domain in runtime.cc
      static inline void log_top_index_space(Processor p, IndexSpace space)
      {
        get_profiler().add_msg(LogMsgTopIndexSpace(space));
      }

      // Thread-safe
      // Called in Runtime::create_index_partition in runtime.cc
      static inline void log_index_partition(Processor p,
                                             IndexSpace parent, 
                                             IndexPartition handle,
                                             bool disjoint, Color color)
      {
        get_profiler().add_msg(LogMsgIndexPartition(parent, handle, disjoint, color));
      }

      // Thread-safe
      // Called in Runtime::create_index_partition in runtime.cc
      static inline void log_index_subspace(Processor p, IndexPartition parent,
                                            IndexSpace handle, Color color)
      {
        get_profiler().add_msg(LogMsgIndexSubspace(parent, handle, color));
      }

      // Thread-safe
      // Called in Runtime::create_field_space in runtime.cc
      static inline void log_field_space(Processor p, FieldSpace handle)
      {
        get_profiler().add_msg(LogMsgFieldSpace(handle));
      }

      // Thread-safe
      // Called in Runtime::allocate_field in runtime.cc
      // Called in Runtime::allocate_fields in runtime.cc
      static inline void log_field_creation(Processor p, FieldSpace handle, 
                                            FieldID fid, bool local)
      {
        get_profiler().add_msg(LogMsgFieldCreation(handle, fid, local));
      }

      // Thread-safe
      // Called in Runtime::create_logical_region in runtime.cc
      static inline void log_top_region(Processor p, IndexSpace ispace,
                                        FieldSpace fspace, RegionTreeID tid)
      {
        get_profiler().add_msg(LogMsgTopRegion(ispace, fspace, tid));
      }

      //========================================================================
      // Logger calls for mapping dependence analysis 
      //========================================================================

      // Thread-safe
      // Called in MapOp::initialize in legion_ops.cc
      // Called in CopyOp::initialize in legion_ops.cc
      // Called in CloseOp::initialize in legion_ops.cc
      // Called in TaskOp::log_requirement in legion_tasks.cc
      static inline void log_logical_requirement(Processor p, 
                                                 UniqueID unique_op_id,
                                                 unsigned index,
                                                 bool region,
                                                 unsigned index_component,
                                                 unsigned field_component,
                                                 RegionTreeID tid,
                                                 PrivilegeMode privilege,
                                                 CoherenceProperty prop,
                                                 ReductionOpID redop,
                                       const std::set<FieldID> &logical_fields)
      {
        // TODO: fields are not logged at the moment
        get_profiler().add_msg(LogMsgLogicalRequirement(unique_op_id, index, region, index_component, field_component, tid, privilege, prop, redop));
      }

      // Thread-safe
      // Called in RegionTreeNode::register_logical_dependences 
      //    in region_tree.cc
      // Called in RegionTreeNode::perform_dependence_checks in region_tree.cc
      static inline void log_mapping_dependence(Processor p,
                                                UniqueID parent_context,
                                                UniqueID previous_id,
                                                unsigned previous_index,
                                                UniqueID next_id,
                                                unsigned next_index,
                                                DependenceType dep_type)
      {
        get_profiler().add_msg(LogMsgMappingDependence(parent_context, previous_id, previous_index, next_id, next_index, dep_type));
      }

      //========================================================================
      // Logger calls for physical dependence analysis
      //========================================================================

      // Thread-safe
      // Called in SingleTask::launch_task in legion_tasks.cc
      static inline void log_task_instance_requirement(Processor p,
                                                       UniqueID unique_id,
                                                       unsigned index,
                                                       IndexSpace handle)
      {
        get_profiler().add_msg(LogMsgTaskInstanceRequirement(unique_id,index, handle));
      }

      // Thread-safe
      // Called in IndividualTask::perform_mapping in legion_tasks.cc
      // Called in SliceTask::enumerate_points in legion_tasks.cc
      static inline void log_task_instance_variant(Processor p,
                                                   UniqueID unique_op_id,
                                                   VariantID vid)
      {
        get_profiler().add_msg(LogMsgTaskInstanceVariant(unique_op_id, vid));
      }

      //========================================================================
      // Logger calls for events
      //========================================================================

      // Thread-safe
      // Called in MapOp::trigger_execution in legion_ops.cc
      // Called in CloseOp::trigger_execution in legion_ops.cc
      // Called in SingleTask::launch_task in legion_tasks.cc
      static inline void log_event_dependence(Processor p, 
                                              Event one, Event two)
      {
#ifdef LEGION_LOGGING_CHECK_NO_EVENT
        assert(one.exists());
        assert(two.exists());
#endif
        if (one != two)
          get_profiler().add_msg(LogMsgEventDependency(one, two));
      }

      // Thread-safe
      // Called in SingleTask::launch_task in legion_tasks.cc
      // Called in RegionTreeNode::issue_update_copies in region_tree.cc
      // Called in ReductionView::perform_reduction in region_tree.cc
      // Called in InstanceView::add_user in region_tree.cc
      // Called in ReductionView::add_user in region_tree.cc
      static inline void log_event_dependences(Processor p,
                           const std::set<Event> &preconditions, Event result)
      {
        for (std::set<Event>::const_iterator it = preconditions.begin();
              it != preconditions.end(); it++)
        {
          log_event_dependence(p, *it, result);
        }
      }

      // Thread-safe
      // Called in MapOp::trigger_execution in legion_ops.cc
      // Called in CloseOp::trigger_execution in legion_ops.cc
      // Called in SingleTask::launch_task in legion_tasks.cc
      static inline void log_operation_events(Processor p,
                                              UniqueID unique_op_id,
                                              Event start_event,
                                              Event end_event)
      {
        get_profiler().add_msg(LogMsgOperationEvents(unique_op_id, start_event, end_event));
      }

      //========================================================================
      // Logger calls for physical instances
      //========================================================================

      // Thread-safe
      // Called in RegionNode::create_instance in region_tree.cc
      // Called in RegionNode::create_reduction in region_tree.cc
      static inline void log_physical_instance(Processor p,
                                               PhysicalInstance instance,
                                               Memory memory, 
                                               IndexSpace index_handle,
                                               FieldSpace field_handle,
                                               RegionTreeID tree_id,
                                               ReductionOpID redop = 0,
                                               bool fold = true, 
                                   Domain indirect_domain = Domain::NO_DOMAIN)
      {
        get_profiler().add_msg(LogMsgPhysicalInstance(instance, memory, index_handle, field_handle, tree_id, redop, fold, indirect_domain));
      }

      // Thread safe
      // Called in MapOp::trigger_execution in legion_ops.cc
      // Called in CloseOp::trigger_execution in legion_ops.cc
      // Called in SingleTask::begin_task in legion_tasks.cc
      static inline void log_physical_user(Processor p,
                                           PhysicalInstance instance,
                                           UniqueID unique_op_id,
                                           unsigned index)
      {
        get_profiler().add_msg(LogMsgPhysicalUser(instance, unique_op_id, index));
      }

      //========================================================================
      // Logger calls for low-level copy operations
      //========================================================================

      // Thread-safe
      // Called in RegionTreeNode::issue_update_copies in region_tree.cc
      // Called in ReductionView::perform_reduction in region_tree.cc
      static inline void log_lowlevel_copy(Processor p,
                                           PhysicalInstance src_instance,
                                           PhysicalInstance dst_instance,
                                           IndexSpace index_handle,
                                           FieldSpace field_handle,
                                           RegionTreeID tree_id,
                                           Event start_event,
                                           Event highlevel_end_event,
                                           const std::set<FieldID> &fields,
                                           ReductionOpID redop = 0)
      {
#ifdef LEGION_LOGGING_CHECK_NO_EVENT
        assert(start_event.exists());
        assert(highlevel_end_event.exists());
#endif
        std::ostringstream os("");
        for (std::set<FieldID>::iterator it=fields.begin(); it!=fields.end(); ++it) {
          if (it != fields.begin()) os << ", ";
          os << *it;
        }
        get_profiler().add_msg(LogMsgLowlevelCopy(src_instance, dst_instance, index_handle, field_handle, tree_id, start_event, highlevel_end_event, os.str(), redop));
      }

      static inline void log_lowlevel_copy_association(Event highlevel_end_event,
                                                       Event lowlevel_end_event,
                                                       unsigned channel_id)
      {
        log_event_dependence(Processor::NO_PROC, lowlevel_end_event, highlevel_end_event);
        get_profiler().add_msg(LogMsgLowlevelCopyAssoc(highlevel_end_event, lowlevel_end_event, channel_id));
      }

      static inline void log_lowlevel_copy_channel(unsigned id, const char* name)
      {
        get_profiler().add_msg(LogMsgLowlevelCopyChannel(id, name));
      }
    }
  }
}

#endif // LEGION__LOGGING

#endif // __LEGION__LOGGING_H__

// EOF

