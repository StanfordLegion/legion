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

#include "legion/operations/internal.h"
#include "legion/tracing/logical.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Internal Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InternalOp::InternalOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InternalOp::~InternalOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void InternalOp::initialize_internal(Operation* creator, int intern_idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(creator != nullptr);
      // We never track internal operations
      initialize_operation(creator->get_context(), creator->get_provenance());
      context_index = creator->get_context_index();
      exception_handler = creator->get_exception_handler();
      legion_assert(creator_req_idx == -1);
      legion_assert(create_op == nullptr);
      create_op = creator;
      create_gen = creator->get_generation();
      creator_req_idx = intern_idx;
      trace = creator->get_trace();
      if (trace != nullptr)
        // Use is_recording and not is_fixed because we're in the
        // logical dependence analysis stage of the pipeline
        tracing = trace->is_recording();
    }

    //--------------------------------------------------------------------------
    void InternalOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      creator_req_idx = -1;
      create_op = nullptr;
      create_gen = 0;
    }

    //--------------------------------------------------------------------------
    void InternalOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      Operation::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    void InternalOp::record_trace_dependence(
        Operation* target, GenerationID target_gen, int target_idx,
        int source_idx, DependenceType dtype, const FieldMask& dependent_mask)
    //--------------------------------------------------------------------------
    {
      legion_assert(creator_req_idx >= 0);
      // Check to see if the target is also our creator
      // in which case we can skip it
      if ((target == create_op) && (target_gen == create_gen))
        return;
      // Check to see if the source is our source
      if (source_idx != creator_req_idx)
        return;
      FieldMask overlap = get_internal_mask() & dependent_mask;
      // If the fields also don't overlap then we are done
      if (!overlap)
        return;
      // Otherwise do the registration
      register_region_dependence(
          0 /*idx*/, target, target_gen, target_idx, dtype, overlap);
    }

    //--------------------------------------------------------------------------
    unsigned InternalOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx == 0);
      return create_op->find_parent_index(creator_req_idx);
    }

  }  // namespace Internal
}  // namespace Legion
