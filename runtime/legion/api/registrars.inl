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

// Included from registrars.h - do not include this directly

// Useful for IDEs
#include "legion/api/registrars.h"

namespace Legion {

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const SpecializedConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const MemoryConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const OrderingConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const TilingConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const FieldConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const DimensionConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const AlignmentConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const OffsetConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const PointerConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline LayoutConstraintRegistrar& LayoutConstraintRegistrar::add_constraint(
      const PaddingConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_constraint(
      const ISAConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    execution_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_constraint(
      const ProcessorConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    execution_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_constraint(
      const ResourceConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    execution_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_constraint(
      const LaunchConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    execution_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_constraint(
      const ColocationConstraint& constraint)
  //--------------------------------------------------------------------------
  {
    execution_constraints.add_constraint(constraint);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline TaskVariantRegistrar& TaskVariantRegistrar::add_layout_constraint_set(
      unsigned index, LayoutConstraintID desc)
  //--------------------------------------------------------------------------
  {
    layout_constraints.add_layout_constraint(index, desc);
    return *this;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_leaf(bool is_leaf /*= true*/)
  //--------------------------------------------------------------------------
  {
    leaf_variant = is_leaf;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_inner(bool is_inner /*= true*/)
  //--------------------------------------------------------------------------
  {
    inner_variant = is_inner;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_idempotent(bool is_idemp /*= true*/)
  //--------------------------------------------------------------------------
  {
    idempotent_variant = is_idemp;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_replicable(bool is_repl /*= true*/)
  //--------------------------------------------------------------------------
  {
    replicable_variant = is_repl;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_concurrent(bool is_concur /*= true*/)
  //--------------------------------------------------------------------------
  {
    concurrent_variant = is_concur;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::set_concurrent_barrier(bool bar /*= true*/)
  //--------------------------------------------------------------------------
  {
    concurrent_barrier = bar;
  }

  //--------------------------------------------------------------------------
  inline void TaskVariantRegistrar::add_generator_task(TaskID tid)
  //--------------------------------------------------------------------------
  {
    generator_tasks.insert(tid);
  }

}  // namespace Legion
