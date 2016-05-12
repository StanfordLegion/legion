/* Copyright 2016 Stanford University, NVIDIA Corporation
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


#ifndef __LEGION_TRACE__
#define __LEGION_TRACE__

#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LegionTrace
     * This class is used for memoizing the dynamic
     * dependence analysis for series of operations
     * in a given task's context.
     */
    class LegionTrace {
    public:
      static const AllocationType alloc_type = TRACE_ALLOC;
    public:
      struct OperationInfo {
      public:
        OperationInfo(Operation *op)
          : kind(op->get_operation_kind()), count(op->get_region_count()) { }
      public:
        Operation::OpKind kind;
        unsigned count;
      };
      struct DependenceRecord {
      public:
        DependenceRecord(int idx)
          : operation_idx(idx), prev_idx(-1), next_idx(-1),
            validates(false), dtype(TRUE_DEPENDENCE) { }
        DependenceRecord(int op_idx, int pidx, int nidx,
                         bool val, DependenceType d,
                         const FieldMask &m)
          : operation_idx(op_idx), prev_idx(pidx), 
            next_idx(nidx), validates(val), dtype(d),
            dependent_mask(m) { }
      public:
        int operation_idx;
        int prev_idx;
        int next_idx;
        bool validates;
        DependenceType dtype;
        FieldMask dependent_mask;
      };
    public:
      LegionTrace(TraceID tid, SingleTask *ctx);
      LegionTrace(const LegionTrace &rhs);
      ~LegionTrace(void);
    public:
      LegionTrace& operator=(const LegionTrace &rhs);
    public:
      // Called by task execution thread
      inline bool is_fixed(void) const { return fixed; }
      void fix_trace(void);
    public:
      // Called by analysis thread
      void end_trace_capture(void);
      void end_trace_execution(Operation *op);
    public:
      // Called by analysis thread
      void register_operation(Operation *op, GenerationID gen);
      void record_dependence(Operation *target, GenerationID target_gen,
                             Operation *source, GenerationID source_gen);
      void record_region_dependence(Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      void record_aliased_requirements(unsigned idx1, unsigned idx2);
    protected:
      std::vector<std::pair<Operation*,GenerationID> > operations;
      // Only need this backwards lookup for recording dependences
      std::map<std::pair<Operation*,GenerationID>,unsigned> op_map;
      // Close operations have a nasty interaction with traces because
      // we can generate different sets of close operations each time we
      // run the trace depending on the state of the logical region tree.
      // Therefore, we keep track of closes done when capturing the trace
      // record transitive dependences on the other operations in the
      // trace that we would have interfered with had the close operations
      // not been necessary.
      std::map<std::pair<Operation*,GenerationID>,
               LegionVector<DependenceRecord>::aligned> close_dependences;
    protected:
      // This is the generalized form of the dependences
      // For each operation, we remember a list of operations that
      // it dependens on and whether it is a validates the region
      std::deque<LegionVector<DependenceRecord>::aligned> dependences;
      // We also need a data structure to record when there are
      // aliased but non-interfering region requirements. This should
      // be pretty sparse so we'll make it a map
      std::map<unsigned,std::vector<std::pair<unsigned,unsigned> > > alias_reqs;
      // Metadata for checking the validity of a trace when it is replayed
      std::vector<OperationInfo> op_info;
    protected:
      const TraceID tid;
      SingleTask *const ctx;
      bool fixed;
      bool tracing;
#ifdef LEGION_SPY
    protected:
      std::vector<UniqueID> current_uids;
      std::vector<unsigned> num_regions;
#endif
    };

    /**
     * \class TraceCaptureOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when a trace capture
     * is finished so the LegionTrace object can compute the
     * dependences data structure.
     */
    class TraceCaptureOp : public Operation {
    public:
      static const AllocationType alloc_type = TRACE_CAPTURE_OP_ALLOC;
    public:
      TraceCaptureOp(Runtime *rt);
      TraceCaptureOp(const TraceCaptureOp &rhs);
      virtual ~TraceCaptureOp(void);
    public:
      TraceCaptureOp& operator=(const TraceCaptureOp &rhs);
    public:
      void initialize_capture(SingleTask *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual void trigger_dependence_analysis(void);
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public FenceOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      TraceCompleteOp(Runtime *rt);
      TraceCompleteOp(const TraceCompleteOp &rhs);
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs);
    public:
      void initialize_complete(SingleTask *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual void trigger_dependence_analysis(void);
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_TRACE__
