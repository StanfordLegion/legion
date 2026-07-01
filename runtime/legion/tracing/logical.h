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

#ifndef __LEGION_LOGICAL_TRACE_H__
#define __LEGION_LOGICAL_TRACE_H__

#include "legion/operations/operation.h"
#include "legion/tracing/recording.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LogicalTrace
     * The logical trace class captures the tracing information
     * for the logical dependence analysis so that it can be
     * replayed without needing to perform the analysis again
     */
    class LogicalTrace : public TraceHashRecorder,
                         public Collectable {
    public:
      struct DependenceRecord {
      public:
        DependenceRecord(int idx)
          : operation_idx(idx), prev_idx(-1), next_idx(-1),
            dtype(LEGION_TRUE_DEPENDENCE)
        { }
        DependenceRecord(
            int op_idx, int pidx, int nidx, DependenceType d,
            const FieldMask& m)
          : operation_idx(op_idx), prev_idx(pidx), next_idx(nidx), dtype(d),
            dependent_mask(m)
        { }
      public:
        inline bool merge(const DependenceRecord& record)
        {
          if ((operation_idx != record.operation_idx) ||
              (prev_idx != record.prev_idx) || (next_idx != record.next_idx) ||
              (dtype != record.dtype))
            return false;
          dependent_mask |= record.dependent_mask;
          return true;
        }
      public:
        int operation_idx;
        int prev_idx;  // previous region requirement index
        int next_idx;  // next region requirement index
        DependenceType dtype;
        FieldMask dependent_mask;
      };
      struct CloseInfo {
      public:
        CloseInfo(
            MergeCloseOp* op, unsigned idx,
#ifdef LEGION_DEBUG_COLLECTIVES
            RegionTreeNode* n,
#endif
            const RegionRequirement& r)
          : close_op(op), requirement(r), creator_idx(idx)
#ifdef LEGION_DEBUG_COLLECTIVES
            ,
            node(n)
#endif
        { }
      public:
        MergeCloseOp* close_op;  // only valid during capture
        RegionRequirement requirement;
        ctx::vector<DependenceRecord> dependences;
        FieldMask close_mask;
        unsigned creator_idx;
#ifdef LEGION_DEBUG_COLLECTIVES
        RegionTreeNode* node;
#endif
      };
      struct OperationInfo {
      public:
        ctx::vector<DependenceRecord> dependences;
        // Note that in this data structure the "context_index"
        // field of PointwiseDependence data structure is actually the
        // relative offset in the trace of the prior operation
        std::map<unsigned, std::vector<PointwiseDependence>>
            pointwise_dependences;
        ctx::vector<CloseInfo> closes;
        // Only need this during trace capture
        // It records dependences for internal operations (that are not merge
        // close ops, mainly refinement ops) based on the region
        // requirement the internal operations were made for so we can forward
        // them on when later things depende on them. This data structure is
        // cleared after we're done with the trace recording
        std::map<unsigned, ctx::vector<DependenceRecord>> internal_dependences;
      };
      struct VerificationInfo {
      public:
        VerificationInfo(OpKind k, TaskID tid, unsigned r, const uint64_t h[2])
          : kind(k), task_id(tid), regions(r)
        {
          hash[0] = h[0];
          hash[1] = h[1];
        }
      public:
        OpKind kind;
        TaskID task_id;
        unsigned regions;
        uint64_t hash[2];
      };
      class StaticTranslator {
      public:
        StaticTranslator(const std::set<RegionTreeID>* trs)
        {
          if (trs != nullptr)
            trees.insert(trs->begin(), trs->end());
        }
      public:
        inline bool skip_analysis(RegionTreeID tid) const
        {
          if (trees.empty())
            return true;
          else
            return (trees.find(tid) != trees.end());
        }
        inline void push_dependences(const std::vector<StaticDependence>* deps)
        {
          AutoLock t_lock(translator_lock);
          if (deps != nullptr)
            dependences.emplace_back(*deps);
          else
            dependences.resize(dependences.size() + 1);
        }
        inline void pop_dependences(std::vector<StaticDependence>& deps)
        {
          AutoLock t_lock(translator_lock);
          legion_assert(!dependences.empty());
          deps.swap(dependences.front());
          dependences.pop_front();
        }
      public:
        LocalLock translator_lock;
        std::deque<std::vector<StaticDependence>> dependences;
        std::set<RegionTreeID> trees;
      };
    public:
      LogicalTrace(
          InnerContext* ctx, TraceID tid, bool logical_only, bool static_trace,
          Provenance* provenance, const std::set<RegionTreeID>* trees);
      ~LogicalTrace(void);
    public:  // From TraceHashRecorder
      virtual bool record_operation_hash(
          Operation* op, Murmur3Hasher& hasher, uint64_t opidx) override;
      virtual bool record_operation_noop(
          Operation* op, uint64_t opidx) override;
      virtual bool record_operation_untraceable(
          Operation* op, uint64_t opidx) override;
    public:
      inline TraceID get_trace_id(void) const { return tid; }
      inline size_t get_operation_count(void) const
      {
        return replay_info.size();
      }
    public:
      void initialize_operation(
          Operation* op, const std::vector<StaticDependence>* dependences);
      void check_operation_count(void);
      bool skip_analysis(RegionTreeID tid) const;
      size_t register_operation(Operation* op, GenerationID gen);
      void register_internal(InternalOp* op);
      void register_close(
          MergeCloseOp* op, unsigned creator_idx,
#ifdef LEGION_DEBUG_COLLECTIVES
          RegionTreeNode* node,
#endif
          const RegionRequirement& req);
      bool record_dependence(
          Operation* target, GenerationID target_gen, Operation* source,
          GenerationID source_gen);
      bool record_region_dependence(
          Operation* target, GenerationID target_gen, Operation* source,
          GenerationID source_gen, unsigned target_idx, unsigned source_idx,
          DependenceType dtype, const FieldMask& dependent_mask);
      void record_pointwise_dependence(
          Operation* target, GenerationID target_gen, Operation* source,
          GenerationID source_gen, unsigned idx,
          const PointwiseDependence& dependence);
    public:
      // Called by task execution thread
      inline bool is_fixed(void) const { return fixed; }
      void fix_trace(Provenance* provenance);
      inline void record_blocking_call(void) { blocking_call_observed = true; }
      inline bool get_and_clear_blocking_call(void)
      {
        const bool result = blocking_call_observed;
        blocking_call_observed = false;
        return result;
      }
      inline void record_intermediate_fence(void) { intermediate_fence = true; }
      inline bool has_intermediate_fence(void) const
      {
        return intermediate_fence;
      }
      inline void reset_intermediate_fence(void) { intermediate_fence = false; }
    public:
      // Called during logical dependence analysis stage
      inline bool is_recording(void) const { return recording; }
      void begin_logical_trace(FenceOp* fence_op);
      void end_logical_trace(FenceOp* fence_op);
    public:
      bool has_physical_trace(void) const
      {
        return (physical_trace != nullptr);
      }
      PhysicalTrace* get_physical_trace(void) { return physical_trace; }
      void invalidate_equivalence_sets(void) const;
    public:
      // A little bit of help for recording the set of colors for
      // control replicated concurrent index space task launches
      bool find_concurrent_colors(
          ReplIndexTask* task,
          std::map<Color, CollectiveID>& concurrent_exchange_colors);
      void record_concurrent_colors(
          ReplIndexTask* task,
          const std::map<Color, CollectiveID>& concurrent_exchange_colors);
    protected:
      void replay_operation_dependences(
          Operation* op, const ctx::vector<DependenceRecord>& dependences);
      void replay_pointwise_dependences(
          Operation* op,
          const std::map<unsigned, std::vector<PointwiseDependence>>&
              dependences);
      void translate_dependence_records(
          Operation* op, const unsigned index,
          const std::vector<StaticDependence>& dependences);
    public:
      InnerContext* const context;
      const TraceID tid;
      Provenance* const begin_provenance;
      // Set after end_trace is called
      Provenance* end_provenance;
    protected:
      // Pointer to a physical trace
      PhysicalTrace* const physical_trace;
    protected:
      // Application stage of the pipeline
      std::vector<VerificationInfo> verification_infos;
      uint64_t verification_index;
      bool blocking_call_observed;
      bool fixed;
      bool intermediate_fence;
    protected:
      struct OpInfo {
      public:
        OpInfo(Operation* o)
          : op(o), gen(op->get_generation()),
            context_index(op->get_context_index()),
            unique_id(op->get_unique_op_id())
        { }
      public:
        Operation* op;
        GenerationID gen;
        uint64_t context_index;
        UniqueID unique_id;
      };
      // Logical dependence analysis stage of the pipeline
      bool recording;
      size_t replay_index;
      std::deque<OperationInfo> replay_info;
      std::map<std::pair<Operation*, GenerationID>, UniqueID> frontiers;
      std::vector<OpInfo> operations;
      // Only need this backwards lookup for trace capture
      std::map<
          std::pair<Operation*, GenerationID>, std::pair<unsigned, unsigned>>
          op_map;
      FenceOp* trace_fence;
      GenerationID trace_fence_gen;
      StaticTranslator* static_translator;
    protected:
      // Help for control replicated concurrent index task launches
      std::map<TraceLocalID, std::vector<Color>> concurrent_colors;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_LOGICAL_TRACE_H__
