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

#ifndef __LEGION_MULTI_TASK_H__
#define __LEGION_MULTI_TASK_H__

#include "legion/tasks/task.h"
#include "legion/operations/collective.h"
#include "legion/operations/pointwise.h"

namespace Legion {
  namespace Internal {

    /**
     * \class MultiTask
     * This is the parent type for each of the multi-task
     * kinds of classes.
     */
    class MultiTask
      : public PointwiseAnalyzable<CollectiveViewCreator<TaskOp> > {
    public:
      struct FutureHandles : public Collectable {
      public:
        std::map<DomainPoint, DistributedID> handles;
      };
      struct ConcurrentGroup {
        ConcurrentGroup(void)
          : exchange(nullptr), group_points(0), color_points(0),
            lamport_clock(0), variant(0), poisoned(false)
        { }
        union ConcurrentPrecondition {
          ConcurrentPrecondition(void)
            : interpreted(RtUserEvent::NO_RT_USER_EVENT)
          { }
          RtUserEvent interpreted;
          RtBarrier traced;
        } precondition;
        std::vector<RtEvent> preconditions;
        std::map<Processor, DomainPoint> processors;
        std::vector<std::pair<PointTask*, ProcessorManager*> > point_tasks;
        std::vector<std::pair<SliceTask*, AddressSpace> > slice_tasks;
        std::vector<ShardID> shards;
        ConcurrentAllreduce* exchange;
        size_t group_points;  // local points for this shard
        size_t color_points;  // global points across all shards
        // This barrier is only here to help with a bug that currently
        // exists in the CUDA driver between collective kernel launches
        // and invocations of cudaMalloc, once it is fixed then we should
        // be able to remove it
        RtBarrier task_barrier;
        uint64_t lamport_clock;
        VariantID variant;
        bool poisoned;
      };
    public:
      MultiTask(void);
      virtual ~MultiTask(void);
    public:
      bool is_sliced(void) const;
      void slice_index_space(void);
      void trigger_slices(void);
      void clone_multi_from(
          MultiTask* task, IndexSpace is, Processor p, bool recurse,
          bool stealable);
      RtBarrier get_concurrent_task_barrier(Color color) const;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual bool is_reducing_future(void) const override
      {
        return (redop > 0);
      }
      virtual Domain get_slice_domain(void) const override;
      virtual ShardID get_shard_id(void) const override { return 0; }
      virtual size_t get_total_shards(void) const override { return 1; }
      virtual DomainPoint get_shard_point(void) const override
      {
        return DomainPoint(0);
      }
      virtual Domain get_shard_domain(void) const override
      {
        return Domain(DomainPoint(0), DomainPoint(0));
      }
      virtual bool is_pointwise_analyzable(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override = 0;
    public:
      virtual void predicate_false(void) override = 0;
      virtual void premap_task(void) = 0;
      virtual bool distribute_task(void) override = 0;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override = 0;
      virtual void launch_task(bool inline_task = false) override = 0;
      virtual bool is_stealable(void) const override = 0;
    public:
      virtual TaskKind get_task_kind(void) const override = 0;
    public:
      virtual void trigger_mapping(void) override;
    protected:
      virtual void trigger_task_commit(void) override = 0;
    public:
      virtual bool pack_task(
          Serializer& rez, AddressSpaceID target) override = 0;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override = 0;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override = 0;
    public:
      virtual SliceTask* clone_as_slice_task(
          IndexSpace is, Processor p, bool recurse, bool stealable) = 0;
      virtual void reduce_future(
          const DomainPoint& point, FutureInstance* instance,
          ApEvent effects) = 0;
      virtual void register_must_epoch(void) = 0;
    public:
      void pack_multi_task(Serializer& rez, AddressSpaceID target);
      void unpack_multi_task(
          Deserializer& derez, std::set<RtEvent>& ready_events);
    public:
      // Return true if it is safe to delete the future
      bool fold_reduction_future(FutureInstance* instance, ApEvent effects);
      void report_concurrent_mapping_failure(
          Processor processor, const DomainPoint& one,
          const DomainPoint& two) const;
    protected:
      std::list<SliceTask*> slices;
      bool sliced;
    protected:
      IndexSpaceNode* launch_space;  // global set of points
      IndexSpace internal_space;     // local set of points
      FutureMap future_map;
      size_t future_map_coordinate;
      FutureHandles* future_handles;
      ReductionOpID redop;
      bool deterministic_redop;
      const ReductionOp* reduction_op;
      Future redop_initial_value;
      FutureMap point_arguments;
      std::vector<FutureMap> point_futures;
      std::vector<OutputOptions> output_region_options;
      // For handling reductions of types with serdez methods
      const SerdezRedopFns* serdez_redop_fns;
      std::atomic<FutureInstance*> reduction_instance;
      ApEvent reduction_instance_precondition;
      std::vector<ApEvent> reduction_fold_effects;
      // Only for handling serdez reductions
      void* serdez_redop_state;
      size_t serdez_redop_state_size;
      // Reduction metadata
      void* reduction_metadata;
      size_t reduction_metasize;
      // Temporary storage for future results
      std::map<DomainPoint, std::pair<FutureInstance*, ApEvent> >
          temporary_futures;
      // used for detecting cases where we've already mapped a mutli task
      // on the same node but moved it to a different processor
      bool first_mapping;
    protected:
      ConcurrentID concurrent_functor;
      unsigned concurrent_points;
      std::map<Color, ConcurrentGroup> concurrent_groups;
    protected:
      uint64_t collective_lamport_clock;
      RtUserEvent collective_lamport_clock_ready;
      size_t collective_unbounded_points;
    protected:
      bool children_commit_invoked;
    protected:
      Future predicate_false_future;
      BufferManager<MultiTask, OPERATION_LIFETIME> predicate_false_result;
    protected:
      // These are the mapped events for individual point tasks in the
      // index space launch for doing pointwise mapping dependences
      std::map<DomainPoint, RtEvent> point_mapped_events;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_MULTI_TASK_H__
