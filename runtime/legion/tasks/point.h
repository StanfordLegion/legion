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

#ifndef __LEGION_POINT_TASK_H__
#define __LEGION_POINT_TASK_H__

#include "legion/tasks/single.h"
#include "legion/api/functors_impl.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PointTask
     * A point task is a single point of an index space task
     * launch.  It will primarily be managed by its enclosing
     * slice task owner.
     */
    class PointTask : public SingleTask,
                      public ProjectionPoint,
                      public Heapify<PointTask, OPERATION_LIFETIME> {
    public:
      PointTask(void);
      PointTask(const PointTask& rhs) = delete;
      virtual ~PointTask(void);
    public:
      PointTask& operator=(const PointTask& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual Operation* get_origin_operation(void) override;
      virtual SingleTask* get_origin_task(void) const override
      {
        return orig_task;
      }
      virtual Domain get_slice_domain(void) const override
      {
        return Domain(index_point, index_point);
      }
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
      virtual bool is_reducing_future(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_replay(void) override;
    public:
      virtual void predicate_false(void) override;
      virtual bool distribute_task(void) override;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override;
      virtual void handle_future_size(
          size_t return_type_size, std::set<RtEvent>& applied_events) override;
      virtual void shard_off(RtEvent mapped_precondition) override;
      virtual bool is_stealable(void) const override;
      virtual bool replicate_task(void) override;
      virtual VersionInfo& get_version_info(unsigned idx) override;
      virtual const VersionInfo& get_version_info(unsigned idx) const override;
      virtual bool is_output_global(unsigned idx) const override;
      virtual bool is_output_bounded(unsigned idx) const override;
      virtual bool is_output_grouped(unsigned idx) const override;
      virtual void record_output_extent(
          unsigned idx, const DomainPoint& color,
          const DomainPoint& extents) override;
      virtual void record_output_registered(
          RtEvent registered, std::set<RtEvent>& applied_events) override;
      virtual bool finalize_map_task_output(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner) override;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override;
    public:
      virtual TaskKind get_task_kind(void) const override;
    public:
      virtual void trigger_complete(ApEvent effects) override;
      virtual void trigger_task_commit(void) override;
    public:
      virtual bool send_task(
          Processor target, std::vector<SingleTask*>& others) override;
      virtual bool pack_task(Serializer& rez, AddressSpaceID target) override;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override;
    public:
      virtual void handle_future(
          ApEvent effects, FutureInstance* instance, const void* metadata,
          size_t metasize, FutureFunctor* functor, Processor future_proc,
          bool own_functor) override;
      virtual void handle_mispredication(void) override;
    public:
      virtual uint64_t order_collectively_mapped_unbounded_pools(
          uint64_t lamport_clock, bool need_result) override;
      virtual ApEvent order_concurrent_launch(
          ApEvent start, VariantImpl* impl) override;
      virtual void concurrent_allreduce(
          ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
          bool poisoned) override;
      virtual void perform_concurrent_task_barrier(void) override;
      bool check_concurrent_variant(VariantID vid);
    public:
      // ProjectionPoint methods
      virtual const DomainPoint& get_domain_point(void) const override;
      virtual void set_projection_result(
          unsigned idx, LogicalRegion result) override;
      virtual void record_intra_space_dependences(
          unsigned index, const std::vector<DomainPoint>& dependences) override;
      virtual void record_pointwise_dependence(
          uint64_t previous_context_index, const DomainPoint& previous_point,
          ShardID shard_id) override;
      virtual const Operation* as_operation(void) const override
      {
        return this;
      }
    public:
      void initialize_point(
          SliceTask* owner, const DomainPoint& point,
          const FutureMap& point_arguments, bool inline_task,
          const std::vector<FutureMap>& point_futures,
          bool record_future_pointwise_dependences);
      RtEvent perform_pointwise_analysis(void);
    public:
      // From MemoizableOp
      virtual void complete_replay(ApEvent pre) override;
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const override;
    public:
      virtual size_t get_collective_points(void) const override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
      virtual RtEvent convert_collective_views(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, const InstanceSet& targets,
          InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
          bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
    public:  // Collective stuff for replicated versions of this point task
      virtual void perform_replicate_collective_versioning(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, CollectiveVersioningBase::RegionVersioning>&
              to_perform) override;
      virtual void convert_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey& key,
          std::map<
              LogicalRegion, CollectiveViewCreatorBase::CollectiveRendezvous>&
              rendezvous) override;
    public:
      bool has_remaining_inlining_dependences(
          std::map<PointTask*, unsigned>& remaining,
          std::map<RtEvent, std::vector<PointTask*> >& event_deps) const;
      void complete_point_projection(void);
    protected:
      friend class SliceTask;
      PointTask* orig_task;
      SliceTask* slice_owner;
    protected:
      std::vector<RtEvent> pointwise_mapping_dependences;
    protected:
      Color concurrent_color;
      RtBarrier concurrent_task_barrier;
      // This is the concurrent precondition event that we need to signal
      // when the preconditions are met for this point task. For non-traced
      // execution this will be a user event that we signal. For traced
      // code (either recording or replaying) this will be a barrier that
      // we will arrive on for each point task.
      union ConcurrentPrecondition {
        ConcurrentPrecondition(void)
          : interpreted(RtUserEvent::NO_RT_USER_EVENT)
        { }
        RtUserEvent interpreted;
        RtBarrier traced;
      } concurrent_precondition;
      // This is the postcondition event that need to wait for before
      // doing the concurrent lamport max all-reduce. It ensures that
      // the preconditions for all the points in the concurrent task
      // have been met and therefore it's safe to do the lamport protocol
      RtEvent concurrent_postcondition;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_POINT_TASK_H__
