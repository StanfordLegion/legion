/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_ALLREDUCE_H__
#define __LEGION_ALLREDUCE_H__

#include "legion/operations/memoizable.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class AllReduceOp
     * Operation for reducing future maps down to futures
     */
    class AllReduceOp : public MemoizableOp {
    public:
      AllReduceOp(void);
      AllReduceOp(const AllReduceOp& rhs) = delete;
      virtual ~AllReduceOp(void);
    public:
      AllReduceOp& operator=(const AllReduceOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, const FutureMap& future_map, ReductionOpID redop,
          bool deterministic, MapperID mapper_id, MappingTagID tag,
          Provenance* provenance, Future initial_value);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
      // AllReduceOps should never actually need this but it might get
      // called in the process of doing a mapping call
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override
      {
        return nullptr;
      }
    protected:
      void invoke_mapper(void);
      ApEvent finalize_serdez_targets(void);
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_execution(void) override;
      virtual void trigger_replay(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    protected:
      // These are virtual methods to override for control replication
      virtual void populate_sources(void);
      virtual void create_future_instances(void);
      virtual void all_reduce_serdez(void);
      virtual ApEvent all_reduce_redop(RtEvent& executed);
    protected:
      ApEvent init_redop_target(FutureInstance* target);
      void fold_serdez(FutureImpl* impl);
    private:
      void prepare_future(
          std::vector<RtEvent>& preconditions, FutureImpl* future);
      void subscribe_to_future(
          std::vector<RtEvent>& ready_events, FutureImpl* future);
      void perform_allreduce(void);
    protected:
      FutureMap future_map;
      ReductionOpID redop_id;
      const ReductionOp* redop;
      const SerdezRedopFns* serdez_redop_fns;
      Future result;
      std::map<DomainPoint, FutureImpl*> sources;
      std::vector<FutureInstance*> targets;
      std::vector<Memory> target_memories;
      std::vector<RtEvent> map_applied_conditions;
      size_t future_result_size;
      FutureInstance* serdez_redop_instance;
      void* serdez_redop_buffer;
      size_t serdez_upper_bound;
      MapperID mapper_id;
      MappingTagID tag;
      bool deterministic;
      Future initial_value;
    };

    /**
     * \class ReplAllReduceOp
     * An all-reduce operation that is aware that it is
     * being executed in a control replication context
     */
    class ReplAllReduceOp : public AllReduceOp {
    public:
      ReplAllReduceOp(void);
      ReplAllReduceOp(const ReplAllReduceOp& rhs) = delete;
      virtual ~ReplAllReduceOp(void);
    public:
      ReplAllReduceOp& operator=(const ReplAllReduceOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* ctx);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    protected:
      virtual void populate_sources(void) override;
      virtual void create_future_instances(void) override;
      virtual void all_reduce_serdez(void) override;
      virtual ApEvent all_reduce_redop(RtEvent& executed) override;
    protected:
      BufferExchange* serdez_redop_collective;
      FutureAllReduceCollective* all_reduce_collective;
      FutureReductionCollective* reduction_collective;
      FutureBroadcastCollective* broadcast_collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ALLREDUCE_H__
