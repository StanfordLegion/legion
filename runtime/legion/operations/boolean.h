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

#ifndef __LEGION_BOOLEAN_OPERATIONS_H__
#define __LEGION_BOOLEAN_OPERATIONS_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FuturePredOp
     * A class for making predicates out of futures or vice versa.
     */
    class FuturePredOp : public Operation {
    public:
      FuturePredOp(void);
      FuturePredOp(const FuturePredOp& rhs) = delete;
      virtual ~FuturePredOp(void);
    public:
      FuturePredOp& operator=(const FuturePredOp& rhs) = delete;
    public:
      Predicate initialize(
          InnerContext* ctx, const Future& f, Provenance* provenance);
      Future initialize(
          InnerContext* ctx, const Predicate& p, Provenance* provenance);
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
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_execution(void) override;
    protected:
      Future future;
      Predicate predicate;
      bool to_predicate;
    };

    /**
     * \class NotPredOp
     * A class for negating other predicates
     */
    class NotPredOp : public Operation {
    public:
      NotPredOp(void);
      NotPredOp(const NotPredOp& rhs) = delete;
      virtual ~NotPredOp(void);
    public:
      NotPredOp& operator=(const NotPredOp& rhs) = delete;
    public:
      Predicate initialize(
          InnerContext* task, const Predicate& p, Provenance* provenance);
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
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_execution(void) override;
    protected:
      Predicate previous, to_set;
    };

    /**
     * \class AndPredOp
     * A class for and-ing other predicates
     */
    class AndPredOp : public Operation {
    public:
      AndPredOp(void);
      AndPredOp(const AndPredOp& rhs) = delete;
      virtual ~AndPredOp(void);
    public:
      AndPredOp& operator=(const AndPredOp& rhs) = delete;
    public:
      Predicate initialize(
          InnerContext* task, std::vector<Predicate>& predicates,
          Provenance* provenance);
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
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_execution(void) override;
    protected:
      std::vector<Predicate> previous;
      Predicate to_set;
    };

    /**
     * \class OrPredOp
     * A class for or-ing other predicates
     */
    class OrPredOp : public Operation {
    public:
      OrPredOp(void);
      OrPredOp(const OrPredOp& rhs) = delete;
      virtual ~OrPredOp(void);
    public:
      OrPredOp& operator=(const OrPredOp& rhs) = delete;
    public:
      Predicate initialize(
          InnerContext* task, std::vector<Predicate>& predicates,
          Provenance* provenance);
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
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_execution(void) override;
    protected:
      std::vector<Predicate> previous;
      Predicate to_set;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_BOOLEAN_OPERATIONS_H__
