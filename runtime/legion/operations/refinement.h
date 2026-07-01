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

#ifndef __LEGION_REFINEMENT_OPERATION_H__
#define __LEGION_REFINEMENT_OPERATION_H__

#include "legion/operations/internal.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RefinementOp
     * A refinement operation is an internal operation that
     * is used to update the equivalence sets being used to
     * represent logical regions.
     */
    class RefinementOp : public InternalOp {
    public:
      RefinementOp(void);
      RefinementOp(const RefinementOp& rhs) = delete;
      virtual ~RefinementOp(void);
    public:
      RefinementOp& operator=(const RefinementOp& rhs) = delete;
    public:
      void initialize(
          Operation* creator, unsigned idx, LogicalRegion parent,
          RegionTreeNode* refinement_node, unsigned parent_req_index);
      void record_refinement_mask(
          unsigned refinement_number, const FieldMask& refinement_mask);
      RegionTreeNode* get_refinement_node(void) const;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual const FieldMask& get_internal_mask(void) const override;
      // Ignore interfering requirements reports here
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2) override
      { }
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
    protected:
      FieldMask refinement_mask;
      RegionTreeNode* refinement_node;
      // The parent region requirement for the refinement to update
      unsigned parent_req_index;
      // For uniquely identify this refinement in the context of
      // its creator operation
      unsigned refinement_number;
    };

    /**
     * \class ReplRefinementOp
     * A refinement operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplRefinementOp : public RefinementOp {
    public:
      ReplRefinementOp(void);
      ReplRefinementOp(const ReplRefinementOp& rhs) = delete;
      virtual ~ReplRefinementOp(void);
    public:
      ReplRefinementOp& operator=(const ReplRefinementOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      void set_repl_refinement_info(
          RtBarrier mapped_barrier, RtBarrier refinement_barrier);
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
    protected:
      RtBarrier mapped_barrier;
      RtBarrier refinement_barrier;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REFINEMENT_OPERATION_H__
