/* Copyright 2018 Stanford University, NVIDIA Corporation
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


#ifndef __SHIM_MAPPER_H__
#define __SHIM_MAPPER_H__

#include "legion.h"
#include "mappers/mapping_utilities.h"
#include "mappers/default_mapper.h"
#include <stdlib.h>
#include <assert.h>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    /**
     * \class ShimMapper
     * The shim mapper class provides some backwards compatibility to 
     * the old mapper interface for existing mappers. It is not designed
     * to be particularly fast (lots of translation by copying). The support
     * for old mapper methods is not complete. Only commonly used old
     * mapper calls are currently implemented. If there is one that you
     * used to use, but is not currently implemented please feel free
     * to add it. Note that when using the ShimMapper only old maper calls
     * should be overloaded. Overloading new mapper calls will result in
     * undefined behavior.
     */
    class ShimMapper : public DefaultMapper {
    public:
      // Our internal classes
      class Task;
      class Copy;
      class Inline;
      class TaskVariantCollection;
    public:
      // Our extended version of region requirements with the old mapping fields
      struct RegionRequirement : public Legion::RegionRequirement {
      public:
        RegionRequirement(void);
      public:
        RegionRequirement& operator=(const Legion::RegionRequirement &rhs);
      public: // inputs
        bool restricted;
        size_t max_blocking_factor;
        std::map<Memory,bool> current_instances;
      public: // outputs
        bool virtual_map;
        bool early_map;
        bool enable_WAR_optimization;
        bool reduction_list;
        bool make_persistent;
        size_t blocking_factor;
        std::vector<Memory> target_ranking;
        std::set<FieldID> additional_fields;
      public:
        bool mapping_failed;
        Memory selected_memory;
      };
    public: 
      // Our version of mappable
      class Mappable {
      public:
        enum MappableKind {
          TASK_MAPPABLE,
          INLINE_MAPPABLE,
          COPY_MAPPABLE,
          ACQUIRE_MAPPABLE,
          RELEASE_MAPPABLE,
        };
      public:
        virtual MappableKind get_mappable_kind(void) const = 0;
        virtual Task* as_mappable_task(void) const = 0;
        virtual Copy* as_mappable_copy(void) const = 0;
        virtual Inline* as_mappable_inline(void) const = 0;
        virtual UniqueID get_unique_mappable_id(void) const = 0;
      };
      // Our version of tasks
      class Task : public Mappable, public Legion::Task {
      public:
        Task(const Legion::Task &rhs, TaskVariantCollection *var);
      public:
        virtual MappableKind get_mappable_kind(void) const;
        virtual Task* as_mappable_task(void) const;
        virtual Copy* as_mappable_copy(void) const;
        virtual Inline* as_mappable_inline(void) const;
        virtual UniqueID get_unique_mappable_id(void) const;
        virtual UniqueID get_unique_id(void) const;
        virtual unsigned get_context_index(void) const;
        virtual int get_depth(void) const;
        virtual const char* get_task_name(void) const;
        virtual bool has_trace(void) const;
      public:
        inline UniqueID get_unique_task_id(void) const { return unique_id; }
      public:
        // select task options fields
        bool                            inline_task;
        bool                            map_locally;
        bool                            spawn_task;
        bool                            profile_task;
      public:
        // select task variants options
        VariantID                       selected_variant;
        TaskVariantCollection*          variants;
      public:
        // map task options
        std::set<Processor>             additional_procs;
        TaskPriority                    task_priority;
        bool                            post_map_task;
        std::vector<RegionRequirement>  regions;
      private:
        UniqueID                        unique_id;
        unsigned                        context_index;
        int                             depth;
        const char * const              task_name;
      };
      // Our version of inline mapping
      class Inline : public Mappable, public Legion::InlineMapping {
      public:
        Inline(const Legion::InlineMapping &rhs);
      public:
        virtual MappableKind get_mappable_kind(void) const;
        virtual Task* as_mappable_task(void) const;
        virtual Copy* as_mappable_copy(void) const;
        virtual Inline* as_mappable_inline(void) const;
        virtual UniqueID get_unique_mappable_id(void) const;
        virtual UniqueID get_unique_id(void) const;
        virtual unsigned get_context_index(void) const;
        virtual int get_depth(void) const;
      public:
        inline UniqueID get_unique_inline_id(void) const { return unique_id; }
      public:
        RegionRequirement               requirement;
      private:
        UniqueID                        unique_id;
        unsigned                        context_index;
        int                             depth;
      };
      // Our version of copy operations
      class Copy : public Mappable, public Legion::Copy {
      public:
        Copy(const Legion::Copy &rhs);
      public:
        virtual MappableKind get_mappable_kind(void) const;
        virtual Task* as_mappable_task(void) const;
        virtual Copy* as_mappable_copy(void) const;
        virtual Inline* as_mappable_inline(void) const;
        virtual UniqueID get_unique_mappable_id(void) const;
        virtual UniqueID get_unique_id(void) const;
        virtual unsigned get_context_index(void) const;
        virtual int get_depth(void) const;
      public:
        inline UniqueID get_unique_copy_id(void) const { return unique_id; }
      public:
        std::vector<RegionRequirement>  src_requirements;
        std::vector<RegionRequirement>  dst_requirements;
      private:
        UniqueID                        unique_id;
        unsigned                        context_index;
        int                             depth;
      };
      // Task Variant Collection
      class TaskVariantCollection {
      public:
        class Variant {
        public:
          Processor::TaskFuncID low_id;
          Processor::Kind proc_kind;
          bool single_task; /**< supports single tasks*/
          bool index_space; /**< supports index tasks*/
          bool inner;
          bool leaf;
          VariantID vid;
        public:
          Variant(void)
            : low_id(0) { }
          Variant(Processor::TaskFuncID id, Processor::Kind k, 
                  bool single, bool index, 
                  bool in, bool lf,
                  VariantID v)
            : low_id(id), proc_kind(k), 
              single_task(single), index_space(index), 
              inner(in), leaf(lf), vid(v) { }
        };
      public:
        TaskVariantCollection(Processor::TaskFuncID uid, 
                              const char *n,
                              const bool idem, size_t ret)
          : user_id(uid), name(n), 
            idempotent(idem), return_size(ret) { }
        void add_variant(Processor::TaskFuncID low_id, 
                         Processor::Kind kind, 
                         bool single, bool index,
                         bool inner, bool leaf,
                         VariantID vid);
        const Variant& select_variant(bool single, bool index, 
                                      Processor::Kind kind);
      public:
        /**
         * Check to see if a collection of variants has one
         * that meets the specific criteria.
         * @param kind the kind of processor to support
         * @param single whether the variants supports single tasks
         * @param index_space whether the variant supports index space launches
         * @return true if a variant exists, false otherwise
         */
        bool has_variant(Processor::Kind kind, 
                         bool single, bool index_space);
        /**
         * Return the variant ID for a variant that
         * meets the given qualifications
         * @param kind the kind of processor to support
         * @param single whether the variant supports single tasks
         * @param index_space whether the variant supports index space launches
         * @return the variant ID if one exists
         */
        VariantID get_variant(Processor::Kind kind, 
                              bool single, bool index_space);
        /**
         * Check to see if a collection has a variant with
         * the given ID.
         * @param vid the variant ID
         * @return true if the collection has a variant with the given ID
         */
        bool has_variant(VariantID vid);
        /**
         * Find the variant with a given ID.
         * @param vid the variant ID
         * @return a const reference to the variant if it exists
         */
        const Variant& get_variant(VariantID vid);

        const std::map<VariantID,Variant>& get_all_variants(void) const
          { return variants; }
      public:
        const Processor::TaskFuncID user_id;
        const char *name;
        const bool idempotent;
        const size_t return_size;
      protected:
        std::map<VariantID,Variant> variants;
      };
    public:
      ShimMapper(Machine machine, Runtime *rt, MapperRuntime *mrt, 
                 Processor local, const char *name = NULL);
      ShimMapper(const ShimMapper &rhs);
      virtual ~ShimMapper(void);
    public:
      ShimMapper& operator=(const ShimMapper &rhs);
    public:
      static const char* create_shim_name(Processor p);
    public:
      // New mapper calls
      virtual MapperSyncModel get_mapper_sync_model(void) const;
      virtual void select_task_options(const MapperContext    ctx,
                                       const Legion::Task&    task,
                                             TaskOptions&     output);
      // Overload this to make the compiler happy
      virtual void select_task_variant(const MapperContext          ctx,
                                       const Legion::Task&          task,
                                       const SelectVariantInput&    input,
                                             SelectVariantOutput&   output);
      virtual void map_task(const MapperContext               ctx,
                            const Legion::Task&               task,
                            const MapTaskInput&               input,
                                  MapTaskOutput&              output);
      virtual void map_copy(const MapperContext               ctx,
                            const Legion::Copy&               copy,
                            const MapCopyInput&               input,
                                  MapCopyOutput&              output);
      virtual void map_inline(const MapperContext             ctx,
                              const InlineMapping&            inline_op,
                              const MapInlineInput&           input,
                                    MapInlineOutput&          output);
      virtual void slice_task(const MapperContext             ctx,
                              const Legion::Task&             task, 
                              const SliceTaskInput&           input,
                                    SliceTaskOutput&          output);
      typedef TaskSlice DomainSplit;
      virtual void select_tunable_value(const MapperContext         ctx,
                                        const Legion::Task&         task,
                                        const SelectTunableInput&   input,
                                              SelectTunableOutput&  output);
      virtual void handle_message(const MapperContext         ctx,
                                  const MapperMessage&        message);
    public:
      // Old mapper calls
      virtual void select_task_options(Task *task);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
      virtual void notify_mapping_result(const Mappable *mappable);
      virtual void notify_mapping_failed(const Mappable *mappable);
      virtual void slice_domain(const Task *task, const Domain &domain,
                                std::vector<DomainSplit> &slices);
      virtual void handle_message(Processor source, 
                                  const void *message, size_t length);
      virtual int get_tunable_value(const Task *task, 
				    TunableID tid, MappingTagID tag);
    protected:
      Color get_logical_region_color(LogicalRegion handle);
      bool has_parent_logical_partition(LogicalRegion handle);
      LogicalPartition get_parent_logical_partition(LogicalRegion handle);
      LogicalRegion get_parent_logical_region(LogicalPartition handle);
      void get_field_space_fields(FieldSpace handle, std::set<FieldID> &fields);
      void broadcast_message(const void *message, size_t message_size);
    protected:
      TaskVariantCollection* find_task_variant_collection(MapperContext ctx,
                                        TaskID task_id, const char *task_name);
      void initialize_requirement_mapping_fields(RegionRequirement &req,
                          const std::vector<PhysicalInstance> &instances);
      bool convert_requirement_mapping(MapperContext ctx,RegionRequirement &req,
                                       std::vector<PhysicalInstance> &result); 
      void initialize_aos_constraints(LayoutConstraintSet &constraints,
                                      const std::set<FieldID> &fields,
                                      ReductionOpID redop);
      void initialize_soa_constraints(LayoutConstraintSet &constraints,
                                      const std::set<FieldID> &fields,
                                      ReductionOpID redop);
    public:
      static void decompose_index_space(const Domain &domain, 
                              const std::vector<Processor> &targets,
                              unsigned splitting_factor, 
                              std::vector<DomainSplit> &slice);
    protected:
      MapperRuntime *const mapper_runtime;
      const Processor::Kind local_kind;
      const Machine machine;
      Runtime *const runtime;
      // The maximum number of tasks a mapper will allow to be stolen at a time
      // Controlled by -dm:thefts
      unsigned max_steals_per_theft;
      // The maximum number of times that a single task is allowed to be stolen
      // Controlled by -dm:count
      unsigned max_steal_count;
      // The splitting factor for breaking index spaces across the machine
      // Mapper will try to break the space into split_factor * num_procs
      // difference pieces
      // Controlled by -dm:split
      unsigned splitting_factor;
      // Do a breadth-first traversal of the task tree, by default we do
      // a depth-first traversal to improve locality
      bool breadth_first_traversal;
      // Whether or not copies can be made to avoid Write-After-Read dependences
      // Controlled by -dm:war
      bool war_enabled;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
      // Maximum number of failed mappings for a task before error
      unsigned max_failed_mappings;
      std::map<UniqueID,unsigned> failed_mappings;
      // Utilities for use within the default mapper 
      Utilities::MachineQueryInterface machine_interface;
      Utilities::MappingMemoizer memoizer;
      Utilities::MappingProfiler profiler;
    protected:
      std::map<TaskID,TaskVariantCollection*> task_variant_collections;
    private:
      MapperContext current_ctx;
    };

  };
};

// For backwards compatibility
namespace LegionRuntime {
  namespace HighLevel {
    typedef Legion::Mapping::ShimMapper ShimMapper;
  };
};

#endif // __SHIM_MAPPER_H__

// EOF

