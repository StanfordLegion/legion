/* Copyright 2022 Stanford University, NVIDIA Corporation
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


#ifndef __MAPPING_UTILITIES__
#define __MAPPING_UTILITIES__

#include "legion.h"

#include <stdlib.h>
#include <assert.h>

namespace Legion {
  namespace Mapping {
    namespace Utilities {

      /**
       * A class containing methods for pulling out
       * information from the Machine structure.
       * Queries are memoized so that future calls
       * are less expensive. Static versions of each
       * of the methods are also available, but do
       * not memoize the results of their invocations.
       */
      class MachineQueryInterface {
      public:
        MachineQueryInterface(Machine m);
      public:
        /**
         * Find a memory visible to all the processors
         */
        Memory find_global_memory(void);
        static
        Memory find_global_memory(Machine machine);
        /**
         * Get the memory stack for a given processor sorted
         * by either throughput or latency.
         */
        void find_memory_stack(Processor proc,
            std::vector<Memory> &stack, bool latency);
        static void find_memory_stack(Machine machine, Processor proc,
                              std::vector<Memory> &stack, bool latency);
        /**
         * Get the memory stack for a given memory sorted by either
         * throughput or latency.
         */
        void find_memory_stack(Memory mem, std::vector<Memory> &stack,
                               bool latency);
        static void find_memory_stack(Machine machine, Memory mem,
                            std::vector<Memory> &stack, bool latency);
        /**
         * Find the memory of a given kind that is visible from
         * the specified processor.
         */
        Memory find_memory_kind(Processor proc, Memory::Kind kind);
        static Memory find_memory_kind(Machine machine, Processor proc,
                                       Memory::Kind kind);
        /**
         * Find the memory of a given kind that is visible from
         * the specified memory.
         */
        Memory find_memory_kind(Memory mem, Memory::Kind kind);
        static Memory find_memory_kind(Machine machine, Memory mem,
                                       Memory::Kind kind);
        /**
         * Find the processor of a given kind that is is visible
         * from the specified memory.
         */
        Processor find_processor_kind(Memory mem, Processor::Kind kind);
        static Processor find_processor_kind(Machine machine, Memory mem,
                                             Processor::Kind kind);
        /**
         * Return a set of processors filtered on the given type.
         * If an empty set is passed in then the set will be populated
         * with all processors of the given kind.
         */
        const std::set<Processor>& filter_processors(Processor::Kind kind);
        static void filter_processors(Machine machine, Processor::Kind kind,
                                      std::set<Processor> &procs);
        /**
         * Return a set of memories filtered on the given type.
         * If an empty set is passed in then the set will be populated
         * with all processors of the given kind.
         */
        const std::set<Memory>& filter_memories(Memory::Kind kind);
        static void filter_memories(Machine machine, Memory::Kind kind,
                                    std::set<Memory> &mems);
      protected:
        static void sort_memories(Machine machine, Processor proc,
                                  std::vector<Memory> &memories, bool latency);
        static void sort_memories(Machine machine, Memory mem,
                                  std::vector<Memory> &memories, bool latency);
      protected:
        const Machine machine;
        Memory global_memory;
        std::map<Processor,std::vector<Memory> > proc_mem_stacks;
        std::map<Memory,std::vector<Memory> > mem_mem_stacks;
        std::map<std::pair<Processor,Memory::Kind>,Memory> proc_mem_table;
        std::map<std::pair<Memory,Memory::Kind>,Memory> mem_mem_table;
        std::map<std::pair<Memory,Processor::Kind>,Processor> mem_proc_table;
        std::map<Processor::Kind,std::set<Processor> > proc_kinds;
        std::map<Memory::Kind,std::set<Memory> > mem_kinds;
      };

      /**
       * A mapping memoizer class for storing mappings for (processor,task_id)
       * pairs.  There is a two-phase process with the memoizer.  All calls
       * to record and notify mapping will update a temporary memoized mapping.
       * If the performance of the mapping is good, then the programmer can
       * commit the mapping as the permanent mapping for the pair.  Otherwise
       * the next calls to record and notify will overwrite the temporary
       * mapping.  If no temporary mapping is commited then the mapping
       * memoizer will continue to return false for calls to has_mapping.
       */
      class MappingMemoizer {
      public:
        MappingMemoizer(void);
      public:
        /**
         * Check to see if there is a memoized mapping for this task on
         * the given processor.
         */
        bool has_mapping(Processor target, const Task *task,
                         unsigned index) const;
        /**
         * Get back the mapping for this task on the given processor.
         * Returns true on success.
         */
        bool recall_mapping(Processor target, const Task *task,
            unsigned index, std::vector<Memory> &ranking) const;
        /**
         * Get the memory chosen for this mapping.  Returns NO_MEMORY
         * if it can't find it.
         */
        Memory recall_chosen(Processor target, const Task *task,
                             unsigned index) const;
        /**
         * Store a temporary mapping for this task on the given processor
         */
        void record_mapping(Processor target, const Task *task,
              unsigned index, const std::vector<Memory> &ranking);
        /**
         * Remember exactly which memory was chosen for the temporary mapping
         */
        void notify_mapping(Processor target, const Task *task,
                            unsigned index, Memory result);
        /**
         * Commit the mapping as the permanent mapping
         */
        void commit_mapping(Processor target, const Task *task);
      protected:
        struct MemoizedMapping {
        public:
          MemoizedMapping(void);
          MemoizedMapping(size_t num_elmts);
        public:
          std::vector<Memory> chosen;
          std::vector<std::vector<Memory> > rankings;
        };
      protected:
        typedef std::pair<Processor,Processor::TaskFuncID> MappingKey;
        std::map<MappingKey,MemoizedMapping> temporary_mappings;
        std::map<MappingKey,MemoizedMapping> permanent_mappings;
      };

      /**
       * The Mapping Profiler will cycle through all the different
       * variants of the task until it has reached the required
       * number of records for selecting the best execution kind
       * for the processor.
       */
      class MappingProfiler {
      public:
        MappingProfiler(void);
      public:
        /**
         * Set the required number of profiling samples required
         * for each variant before profiling can be considered
         * complete for that variant. The default number is one.
         */
        void set_needed_profiling_samples(unsigned num_samples);
        void set_needed_profiling_samples(Processor::TaskFuncID task_id,
                                          unsigned num_samples);
        /**
         * Set the maximum number of profiling samples to keep
         * around for any variant. By default it is 32.
         */
        void set_max_profiling_samples(unsigned max_samples);
        void set_max_profiling_samples(Processor::TaskFuncID task_id,
                                       unsigned max_samples);
        /**
         * Set the profiling samples to be gathered in the profiler
         * of the processor that triggered the task. By default it is false.
         */
        void set_gather_in_original_processor(Processor::TaskFuncID task_id,
                                              bool flag);

        /**
         * Check to see if profiling is complete for all the
         * variants of this task.
         */
        bool profiling_complete(const Task *task) const;
        bool profiling_complete(const Task *task, Processor::Kind kind) const;
        /**
         * Return the processor kind for the best performing
         * variant of this task.
         */
        Processor::Kind best_processor_kind(const Task *task) const;
      public:
        struct Profile {
          long long execution_time;
          Processor target_processor;
          DomainPoint index_point;
        };

        struct VariantProfile {
        public:
          VariantProfile(void);
        public:
          long long total_time;
          std::list<Profile> samples;
        };

        typedef std::map<Processor::Kind,VariantProfile> VariantMap;
        typedef std::map<Processor::TaskFuncID,VariantMap> TaskMap;

        void add_profiling_sample(Processor::TaskFuncID task_id,
                                  const Profile& sample);

        TaskMap get_task_profiles() const;
        VariantMap get_variant_profiles(Processor::TaskFuncID tid) const;
        VariantProfile get_variant_profile(Processor::TaskFuncID tid,
                                           Processor::Kind kind) const;

        void clear_samples(Processor::TaskFuncID task_id);
        void clear_samples(Processor::TaskFuncID task_id, Processor::Kind kind);

        typedef std::list<DomainPoint> PointList;
        typedef std::map<Processor, PointList> AssignmentMap;

        /**
         * Return a balanced assignment of point tasks to processors by LPT
         * (Longest-Processing Time) scheduling based on the average
         * execution times. Assume that processors of the same kind are
         * homogeneous and only the workload of point tasks can be skwed.
         */
        AssignmentMap get_balanced_assignments(Processor::TaskFuncID task_id,
                                               Processor::Kind kind) const;

        AssignmentMap get_balanced_assignments(Processor::TaskFuncID task_id)
                                                                          const;


        struct ProfilingOption {
          ProfilingOption(void);
          ProfilingOption(unsigned, unsigned);
          unsigned needed_samples;
          unsigned max_samples;
          bool gather_in_orig_proc;
        };

        typedef std::map<Processor::TaskFuncID,ProfilingOption> OptionMap;

        ProfilingOption get_profiling_option(Processor::TaskFuncID tid) const;

      protected:
        unsigned needed_samples;
        unsigned max_samples;
        TaskMap task_profiles;
        OptionMap profiling_options;
      };

      /** @{
          Functions for printing various runtime objects to strings from inside
          a mapper. */

      const char* to_string(Processor::Kind kind);

      const char* to_string(Memory::Kind kind);

      const char* to_string(PrivilegeMode priv);

      const char* to_string(CoherenceProperty prop);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Domain& dom);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            LogicalRegion lr);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            IndexSpace is);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const LayoutConstraintSet& constraints);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            FieldSpace fs,
                            const std::set<FieldID>& fields);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            PhysicalInstance inst);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const RegionRequirement& req,
                            unsigned req_idx);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Task& task,
                            bool include_index_point = true);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const InlineMapping& inline_op);

      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Copy& copy,
                            bool include_index_point = true);

      /** @} */

    }; // namespace Utilities
  }; // namespace Mapping
}; // namespace Legion

// For backwards compatbility
namespace LegionRuntime {
  namespace HighLevel {
    namespace MappingUtilities {
      typedef Legion::Mapping::Utilities::MachineQueryInterface
        MachineQueryInterface;
      typedef Legion::Mapping::Utilities::MappingMemoizer MappingMemoizer;
      typedef Legion::Mapping::Utilities::MappingProfiler MappingProfiler;
    };
  };
};

#endif // __MAPPING_UTILITIES__
