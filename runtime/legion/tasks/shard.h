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

#ifndef __LEGION_SHARD_TASK_H__
#define __LEGION_SHARD_TASK_H__

#include "legion/tasks/single.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ShardTask
     * A shard task is copy of a single task that is used for
     * executing a single copy of a control replicated task.
     * It implements the functionality of a single task so that
     * we can use it mostly transparently for the execution of
     * a single shard.
     */
    class ShardTask : public SingleTask {
    public:
      ShardTask(
          SingleTask* source, InnerContext* parent, ShardManager* manager,
          ShardID shard_id, Processor target, VariantID chosen);
      ShardTask(
          InnerContext* parent_ctx, Deserializer& derez, ShardManager* manager,
          ShardID shard_id, Processor target, VariantID chosen);
      ShardTask(const ShardTask& rhs) = delete;
      virtual ~ShardTask(void);
    public:
      ShardTask& operator=(const ShardTask& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual Domain get_slice_domain(void) const override;
      virtual ShardID get_shard_id(void) const override { return shard_id; }
      virtual size_t get_total_shards(void) const override;
      virtual DomainPoint get_shard_point(void) const override;
      virtual Domain get_shard_domain(void) const override;
      virtual SingleTask* get_origin_task(void) const override { std::abort(); }
      virtual bool is_shard_task(void) const override { return true; }
      virtual bool is_top_level_task(void) const override;
      // Set this to true so we always eagerly evaluate future functors
      // at the end of a task to get an actual future instance to pass back
      virtual bool is_reducing_future(void) const override { return true; }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void predicate_false(void) override;
      virtual bool distribute_task(void) override;
      virtual RtEvent perform_must_epoch_version_analysis(MustEpochOp* own);
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override;
      virtual void handle_future_size(
          size_t return_type_size, std::set<RtEvent>& applied_events) override;
      virtual bool is_stealable(void) const override;
      virtual void initialize_map_task_input(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner) override;
      virtual bool finalize_map_task_output(
          Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
          MustEpochOp* must_epoch_owner) override;
    public:
      virtual TaskKind get_task_kind(void) const override;
    public:
      // Override these methods from operation class
      virtual void trigger_mapping(void) override;
      virtual void trigger_complete(ApEvent effects) override;
    protected:
      virtual void trigger_task_commit(void) override;
    public:
      virtual bool send_task(
          Processor target, std::vector<SingleTask*>& others) override;
      virtual bool pack_task(Serializer& rez, AddressSpaceID target) override;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override;
    public:
      virtual void handle_future(
          ApEvent effects, FutureInstance* instance, const void* metadata,
          size_t metasize, FutureFunctor* functor, Processor future_proc,
          bool own_functor) override;
      virtual void handle_mispredication(void) override;
    public:
      virtual uint64_t order_collectively_mapped_unbounded_pools(
          uint64_t lamport_clock, bool need_result) override;
      virtual void concurrent_allreduce(
          ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
          bool poisoned) override;
      virtual void perform_concurrent_task_barrier(void) override;
    public:
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
    protected:
      virtual TaskContext* create_execution_context(
          VariantImpl* v, std::set<ApEvent>& launch_events, bool inline_task,
          bool leaf_task) override;
    public:
      virtual InnerContext* create_implicit_context(void) override;
    public:
      void dispatch(void);
      void return_resources(
          ResourceTracker* target, std::set<RtEvent>& preconditions);
      void report_leaks_and_duplicates(std::set<RtEvent>& preconditions);
      ReplicateContext* get_replicate_context(void) const;
    public:
      void initialize_implicit_task(TaskID tid, MapperID mid, Processor proxy);
    public:
      const ShardID shard_id;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SHARD_TASK_H__
