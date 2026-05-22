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

#ifndef __LEGION_PREDICATE_OPERATION_H__
#define __LEGION_PREDICATE_OPERATION_H__

#include "legion/api/predicate_impl.h"
#include "legion/operations/memoizable.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PredicatedOp
     * A predicated operation is an abstract class
     * that serves as the basis for operation which
     * will be executed with a predicate value.
     * Note that all speculative operations are also memoizable operations.
     */
    class PredicatedOp : public MemoizableOp {
    public:
      enum PredState {
        PENDING_PREDICATE_STATE,
        PREDICATED_TRUE_STATE,
        PREDICATED_FALSE_STATE,
      };
    public:
      PredicatedOp(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      void initialize_predication(
          InnerContext* ctx, const Predicate& p, Provenance* provenance);
      virtual bool is_predicated_op(void) const override;
      // Wait until the predicate is valid and then return
      // its value.  Give it the current processor in case it
      // needs to wait for the value
      bool get_predicate_value(size_t index);
    public:
      // This method gets invoked if a predicate for a predicated
      // operation resolves to false before we try to map the operation
      virtual void predicate_false(void) = 0;
    protected:
      PredState predication_state;
      PredicateImpl* predicate;
    public:
      // For managing predication
      PredEvent true_guard;
      PredEvent false_guard;
    };

    /**
     * \class Predicated
     * Override the logical dependence analysis to handle any kind
     * of predicated analysis or speculation
     */
    template<typename OP>
    class Predicated : public OP {
    public:
      Predicated(void) : OP() { }
      virtual ~Predicated(void) { }
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t opidx) override;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/operations/predicate.inl"

#endif  // __LEGION_PREDICATE_OPERATION_H__
