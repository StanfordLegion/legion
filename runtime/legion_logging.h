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


#ifndef __LEGION_LOGGING_H__
#define __LEGION_LOGGING_H__

#include "lowlevel.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"

/**
 * This file contains calls for logging that are consumed by the legion_spy tool in the tools directory.
 * To see where these statements get consumed, look in spy_parser.py
 */

namespace LegionRuntime {
  namespace HighLevel {
    namespace LegionSpy {

      extern Logger::Category log_spy;

      static int next_point_id = 0;

      // Logger calls for the machine architecture
      static inline void log_utility_processor(unsigned unique_id)
      {
        log_spy(LEVEL_INFO, "Utility %x", unique_id);
      }

      static inline void log_processor(unsigned unique_id, unsigned util_id, unsigned kind)
      {
        log_spy(LEVEL_INFO, "Processor %x %x %d", unique_id, util_id, kind);
      }

      static inline void log_memory(unsigned unique_id, size_t capacity)
      {
        log_spy(LEVEL_INFO, "Memory %x %ld", unique_id, capacity);
      }

      static inline void log_proc_mem_affinity(unsigned proc_id, unsigned mem_id, unsigned bandwidth, unsigned latency)
      {
        log_spy(LEVEL_INFO, "Processor Memory %x %x %d %d", proc_id, mem_id, bandwidth, latency);
      }

      static inline void log_mem_mem_affinity(unsigned mem1, unsigned mem2, unsigned bandwidth, unsigned latency)
      {
        log_spy(LEVEL_INFO, "Memory Memory %x %x %d %d", mem1, mem2, bandwidth, latency);
      }

      // Logger calls for the shape of region trees
      static inline void log_top_index_space(unsigned unique_id)
      {
        log_spy(LEVEL_INFO,"Index Space %x", unique_id);
      }

      static inline void log_index_partition(unsigned parent_id, unsigned unique_id, bool disjoint, unsigned color)
      {
        log_spy(LEVEL_INFO,"Index Partition %x %d %d %d", parent_id, unique_id, disjoint, color);
      }

      static inline void log_index_subspace(unsigned parent_id, unsigned unique_id, unsigned color)
      {
        log_spy(LEVEL_INFO,"Index Subspace %d %x %d", parent_id, unique_id, color);
      }

      static inline void log_field_space(unsigned unique_id)
      {
        log_spy(LEVEL_INFO,"Field Space %d", unique_id);
      }

      static inline void log_field_creation(unsigned unique_id, unsigned field_id)
      {
        log_spy(LEVEL_INFO,"Field Creation %d %d", unique_id, field_id);
      }

      static inline void log_top_region(unsigned index_space, unsigned field_space, unsigned tree_id)
      {
        log_spy(LEVEL_INFO,"Region %x %d %d", index_space, field_space, tree_id);
      }

      // Logger calls for operations 
      static inline void log_top_level_task(unsigned hid, unsigned gen, unsigned unique_id, unsigned context, unsigned top_id)
      {
        log_spy(LEVEL_INFO,"Top Task %x %d %d %d %d", hid, gen, context, unique_id, top_id);
      }

      static inline void log_task_operation(unsigned unique_id, unsigned task_id, unsigned parent_id, unsigned parent_ctx, unsigned hid, unsigned gen, bool index_space)
      {
        log_spy(LEVEL_INFO,"Task Operation %d %d %d %d %x %d %d", unique_id, task_id, parent_id, parent_ctx, hid, gen, index_space);
      }

      static inline void log_mapping_operation(unsigned unique_id, unsigned parent_id, unsigned parent_ctx, unsigned hid, unsigned gen)
      {
        log_spy(LEVEL_INFO,"Mapping Operation %d %d %d %x %d", unique_id, parent_id, parent_ctx, hid, gen);
      }

      static inline void log_deletion_operation(unsigned unique_id, unsigned parent_id, unsigned parent_ctx, unsigned hid, unsigned gen)
      {
        log_spy(LEVEL_INFO,"Deletion Operation %d %d %d %x %d", unique_id, parent_id, parent_ctx, hid, gen);
      }

      static inline void log_task_name(unsigned unique_id, const char *name)
      {
        log_spy(LEVEL_INFO,"Task Name %d %s", unique_id, name);
      }

      // Logger calls for mapping dependence analysis 
      static inline void log_logical_requirement(unsigned unique_id, unsigned index, bool region, unsigned index_component,
                            unsigned field_component, unsigned tree_id, unsigned privilege, unsigned coherence, unsigned redop)
      {
        log_spy(LEVEL_INFO,"Logical Requirement %d %d %d %x %d %d %d %d %d", unique_id, index, region, index_component,
                                                field_component, tree_id, privilege, coherence, redop);
      }

      static inline void log_requirement_fields(unsigned unique_id, unsigned index, const std::set<unsigned> &logical_fields)
      {
        for (std::set<unsigned>::const_iterator it = logical_fields.begin();
              it != logical_fields.end(); it++)
        {
          log_spy(LEVEL_INFO,"Logical Requirement Field %d %d %d", unique_id, index, *it);
        }
      }

      static inline void log_mapping_dependence(unsigned parent_id, unsigned parent_ctx, unsigned hid, unsigned gen, unsigned prev_id, 
                                                unsigned prev_idx, unsigned next_id, unsigned next_idx, unsigned dep_type)
      {
        log_spy(LEVEL_INFO,"Mapping Dependence %d %d %x %d %d %d %d %d %d", parent_id, parent_ctx, hid, gen, prev_id, prev_idx, next_id, next_idx, dep_type);
      }

      // Logger calls for physical dependence analysis
      static inline void log_task_instance_requirement(unsigned unique_id, unsigned ctx, unsigned gen, unsigned hid, unsigned idx, unsigned index)
      {
        log_spy(LEVEL_INFO,"Task Instance Requirement %d %d %d %x %d %x", unique_id, ctx, gen, hid, idx, index);
      }

      // Logger calls for events
      static inline void log_event_dependence(Event one, Event two)
      {
        if (one != two)
          log_spy(LEVEL_INFO,"Event Event %x %d %x %d", one.id, one.gen, two.id, two.gen);
      }

      static inline void log_event_dependences(const std::set<Event> &preconditions, Event result)
      {
        for (std::set<Event>::const_iterator it = preconditions.begin();
              it != preconditions.end(); it++)
        {
          if (*it != result)
            log_spy(LEVEL_INFO,"Event Event %x %d %x %d", it->id, it->gen, result.id, result.gen);
        }
      }

      static inline void log_task_events(unsigned hid, unsigned gen, unsigned ctx, unsigned unique_id, bool index_space, unsigned point, 
                                          Event start_event, Event term_event)
      {
        log_spy(LEVEL_INFO,"Task Events %x %d %d %d %d %d %x %d %x %d", hid, gen, ctx, unique_id, index_space, point, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }

      template<int DIM>
      static inline void log_task_events(unsigned hid, unsigned gen, unsigned ctx, unsigned unique_id, bool index_space, Arrays::Point<DIM> point,
                                          Event start_event, Event term_event)
      {
        int point_id = next_point_id++;
        char point_buffer[128];
        point_buffer[0] = '\0';
        for (int i = 0; i < DIM; i++)
        {
          char idx_buffer[128];
          sprintf(idx_buffer,"%d ", point[i]);
          strcat(point_buffer,idx_buffer);
        }
        log_spy(LEVEL_INFO,"Point Value %d %d %s\n", DIM, point_id, point_buffer);
        log_spy(LEVEL_INFO,"Point Task Events %x %d %d %d %d %d %x %d %x %d", hid, gen, ctx, unique_id, index_space, point_id, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }

      static inline void log_index_task_termination(unsigned unique_id, Event term_event)
      {
        log_spy(LEVEL_INFO,"Index Termination %d %x %d", unique_id, term_event.id, term_event.gen);
      }

      static inline void log_copy_operation(unsigned src_manager_id, unsigned dst_manager_id,
                                            unsigned index_handle, unsigned field_handle, unsigned tree_id,
                                            Event start_event, Event term_event, const char *mask)
      {
        log_spy(LEVEL_INFO,"Copy Events %d %d %x %d %d %x %d %x %d %s", src_manager_id, dst_manager_id,
          index_handle, field_handle, tree_id, start_event.id, start_event.gen, term_event.id, term_event.gen, mask);
      }

      static inline void log_reduction_operation(unsigned src_manager_id, unsigned dst_manager_id,
                                                 unsigned index_handle, unsigned field_handle, unsigned tree_id,
                                                 Event start_event, Event term_event, unsigned redop, const char *mask)
      {
        log_spy(LEVEL_INFO,"Reduce Events %d %d %x %d %d %x %d %x %d %d %s", src_manager_id, dst_manager_id, 
          index_handle, field_handle, tree_id, start_event.id, start_event.gen, term_event.id, term_event.gen, redop, mask);
      }

      static inline void log_map_events(unsigned unique_id, Event start_event, Event term_event)
      {
        log_spy(LEVEL_INFO, "Map Events %d %x %d %x %d", unique_id, start_event.id, start_event.gen, term_event.id, term_event.gen);
      }

      // Logger calls for physical instances
      static inline void log_physical_instance(unsigned inst_id, unsigned mem_id, unsigned index_handle, unsigned field_handle, unsigned tree_id)
      {
        log_spy(LEVEL_INFO, "Physical Instance %x %x %x %d %d", inst_id, mem_id, index_handle, field_handle, tree_id);
      }

      static inline void log_instance_manager(unsigned inst_id, unsigned manager_id)
      {
        log_spy(LEVEL_INFO, "Instance Manager %x %d", inst_id, manager_id);
      }

      static inline void log_physical_reduction(unsigned inst_id, unsigned mem_id, unsigned index_handle, unsigned field_handle, unsigned tree_id, bool fold, unsigned indirect_id = 0)
      {
        log_spy(LEVEL_INFO, "Reduction Instance %x %x %x %d %d %d %d", inst_id, mem_id, index_handle, field_handle, tree_id, fold, indirect_id);
      }

      static inline void log_reduction_manager(unsigned inst_id, unsigned manager_id)
      {
        log_spy(LEVEL_INFO, "Reduction Manager %x %d", inst_id, manager_id);
      }

      static inline void log_task_user(unsigned uid, unsigned ctx, unsigned gen, unsigned hid, unsigned idx, unsigned manager_id)
      {
        log_spy(LEVEL_INFO, "Task Instance User %d %d %d %x %d %d", uid, ctx, gen, hid, idx, manager_id);
      }

      static inline void log_mapping_user(unsigned uid, unsigned manager_id)
      {
        log_spy(LEVEL_INFO, "Mapping Instance User %d %d", uid, manager_id);
      }

      // Logger calls for timing information
      static inline void log_task_execution_information(unsigned uid, unsigned ctx, unsigned gen, unsigned hid, unsigned proc_id)
      {
        log_spy(LEVEL_INFO, "Execution Information %d %d %d %x %d", uid, ctx, gen, hid, proc_id); 
      }

      static inline void log_task_begin_timing(unsigned uid, unsigned ctx, unsigned gen, unsigned hid, unsigned long start_time)
      {
        log_spy(LEVEL_INFO, "Begin Task Timing %d %d %d %x %ld", uid, ctx, gen, hid, start_time);
      }

      static inline void log_task_end_timing(unsigned uid, unsigned ctx, unsigned gen, unsigned hid, unsigned long end_time)
      {
        log_spy(LEVEL_INFO, "End Task Timing %d %d %d %x %ld", uid, ctx, gen, hid, end_time);
      }
    };

    class TreeStateLogger {
    public:
      explicit TreeStateLogger(Processor local_proc);
      ~TreeStateLogger(void);
    public:
      void log(const char *fmt, ...);
      void down(void);
      void up(void);
      void start_block(const char *ftm, ...);
      void finish_block(void);
      unsigned get_depth(void) const { return depth; }
    public:
      static void capture_state(HighLevelRuntime *rt, unsigned idx, const char *task_name, 
                                unsigned uid, RegionNode *node, ContextID ctx, bool pack, 
                                bool send, FieldMask capture_mask, FieldMask working_mask);
      static void capture_state(HighLevelRuntime *rt, unsigned idx, const char *task_name,
                                unsigned uid, PartitionNode *node, ContextID ctx, 
                                bool pack, bool send, FieldMask capture_mask, 
                                FieldMask working_mask);
      static void capture_state(HighLevelRuntime *rt, const RegionRequirement *req, 
                                unsigned idx, const char *task_name, unsigned uid,
                                RegionNode *node, ContextID ctx, bool pre_map, 
                                bool sanitize, bool closing, 
                                FieldMask capture_mask, FieldMask working_mask);
      static void capture_state(HighLevelRuntime *rt, LogicalRegion handle, 
                                const char *task_name, unsigned uid,
                                RegionNode *node, ContextID ctx, bool pack, unsigned shift,
                                FieldMask capture_mask, FieldMask working_mask);
    private:
      void println(const char *fmt, va_list args);
    private:
      FILE *tree_state_log;
      char block_buffer[128];
      unsigned depth;
    };
  };
};

#endif // __LEGION_LOGGING_H__

