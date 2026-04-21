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

#ifndef __LEGION_SLICE_TASK_H__
#define __LEGION_SLICE_TASK_H__

#include "legion/tasks/multi.h"
#include "legion/utilities/resources.h"

namespace Legion {
  namespace Internal {

    /**
     * \class SliceTask
     * A slice task is a (possibly whole) fraction of an index
     * space task launch.  Once slice task object is made for
     * each slice created by the mapper when (possibly recursively)
     * slicing up the domain of the index space task launch.
     */
    class SliceTask : public MultiTask,
                      public ResourceTracker,
                      public Heapify<SliceTask, OPERATION_LIFETIME> {
    public:
      SliceTask(void);
      SliceTask(const SliceTask& rhs) = delete;
      virtual ~SliceTask(void);
    public:
      SliceTask& operator=(const SliceTask& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual Operation* get_origin_operation(void) override;
    public:
      virtual void trigger_dependence_analysis(void) override;
    public:
      virtual void predicate_false(void) override;
      virtual void premap_task(void) override;
      virtual bool distribute_task(void) override;
      virtual VersionInfo& get_version_info(unsigned idx) override;
      virtual const VersionInfo& get_version_info(unsigned idx) const override;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override;
      virtual void launch_task(bool inline_task = false) override;
      virtual bool is_stealable(void) const override;
      virtual bool is_output_global(unsigned idx) const override;
      virtual bool is_output_valid(unsigned idx) const override;
      virtual bool is_output_grouped(unsigned idx) const override;
      virtual void trigger_complete(ApEvent effects) override;
    public:
      virtual TaskKind get_task_kind(void) const override;
    public:
      bool send_task(
          Processor target, PointTask* point, std::vector<SingleTask*>& others);
      void pack_slice_task(
          Serializer& rez, AddressSpaceID target,
          const std::vector<PointTask*>& to_send);
      virtual bool pack_task(Serializer& rez, AddressSpaceID target) override;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override;
    public:
      virtual SliceTask* clone_as_slice_task(
          IndexSpace is, Processor p, bool recurse, bool stealable) override;
      virtual void reduce_future(
          const DomainPoint& point, FutureInstance* instance,
          ApEvent effects) override;
      void handle_future(
          ApEvent complete, const DomainPoint& point, FutureInstance* instance,
          const void* metadata, size_t metasize, FutureFunctor* functor,
          Processor future_proc, bool own_functor);
    public:
      virtual void register_must_epoch(void) override;
      PointTask* clone_as_point_task(
          const DomainPoint& point, bool inline_task);
      size_t enumerate_points(bool inline_task);
      void set_predicate_false_result(
          const DomainPoint& point, TaskContext* execution_context);
    public:
      void check_target_processors(void) const;
      void update_target_processor(void);
      void expand_replay_slices(std::list<SliceTask*>& slices);
    protected:
      virtual void trigger_task_commit(void) override;
    public:
      RtEvent find_intra_space_dependence(const DomainPoint& point);
      void return_privileges(
          TaskContext* point_context, std::set<RtEvent>& preconditions);
      void record_point_mapped(
          PointTask* point, RtEvent child_mapped, bool shard_off = false);
      void record_point_complete(ApEvent child_effects);
      void record_point_committed(
          RtEvent commit_precondition = RtEvent::NO_RT_EVENT);
    public:
      void handle_future_size(size_t future_size, const DomainPoint& p);
      void record_output_extent(
          unsigned index, const DomainPoint& color, const DomainPoint& extent);
      void record_output_registered(
          RtEvent registered, std::set<RtEvent>& applied_events);
      void rendezvous_concurrent_mapped(
          const DomainPoint& point, Processor target, Color color,
          RtEvent precondition);
      uint64_t collective_lamport_allreduce(
          uint64_t lamport_clock, bool need_result);
      void concurrent_allreduce(
          PointTask* point, ProcessorManager* manager, uint64_t lamport_clock,
          VariantID vid, bool poisoned);
      void finish_concurrent_allreduce(
          Color color, uint64_t lamport_clock, bool poisoned, VariantID vid,
          RtBarrier concurrent_task_barrier);
    protected:
      void send_rendezvous_concurrent_mapped(void);
      void forward_completion_effects(void);
      void pack_remote_complete(Serializer& rez, ApEvent slice_effects);
      void pack_remote_commit(Serializer& rez, RtEvent applied_condition);
    public:  // Privilege tracker methods
      virtual void receive_resources(
          uint64_t return_index,
          std::map<LogicalRegion, unsigned>& created_regions,
          std::vector<DeletedRegion>& deleted_regions,
          std::set<std::pair<FieldSpace, FieldID> >& created_fields,
          std::vector<DeletedField>& deleted_fields,
          std::map<FieldSpace, unsigned>& created_field_spaces,
          std::map<FieldSpace, std::set<LogicalRegion> >& latent_spaces,
          std::vector<DeletedFieldSpace>& deleted_field_spaces,
          std::map<IndexSpace, unsigned>& created_index_spaces,
          std::vector<DeletedIndexSpace>& deleted_index_spaces,
          std::map<IndexPartition, unsigned>& created_partitions,
          std::vector<DeletedPartition>& deleted_partitions,
          std::set<RtEvent>& preconditions) override;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
      virtual void complete_replay(ApEvent instance_ready_event) override;
    public:
      virtual size_t get_collective_points(void) const override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      void perform_replicate_collective_versioning(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform);
      void convert_replicate_collective_views(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
      virtual void finalize_collective_versioning_analysis(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform) override;
      virtual RtEvent convert_collective_views(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, const InstanceSet& targets,
          InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
          bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals) override;
      virtual void rendezvous_collective_mapping(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, RendezvousResult* result, AddressSpaceID source,
          const op::vector<std::pair<DistributedID, FieldMask> >& insts)
          override;
    protected:
      friend class IndexTask;
      friend class PointTask;
      friend class ReplMustEpochOp;
      std::vector<PointTask*> points;
    protected:
      unsigned num_unmapped_points;
      std::atomic<unsigned> num_uncompleted_points;
      unsigned num_uncommitted_points;
    protected:
      IndexTask* index_owner;
      UniqueID remote_unique_id;
      bool origin_mapped;
      DomainPoint reduction_instance_point;
    protected:
      std::set<RtEvent> commit_preconditions;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SLICE_TASK_H__
