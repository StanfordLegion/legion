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

#include "legion_constraint.h"

namespace LegionRuntime {
  namespace HighLevel {

    /////////////////////////////////////////////////////////////
    // ISAConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ISAConstraint::ISAConstraint(uint64_t prop /*= 0*/)
      : isa_prop(prop)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    bool ISAConstraint::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      return (isa_prop != 0);
    }

    //--------------------------------------------------------------------------
    void ISAConstraint::find_proc_kinds(std::vector<Processor::Kind> &kds) const
    //--------------------------------------------------------------------------
    {
      if (isa_prop & (X86_ISA | ARM_ISA | POW_ISA))
      {
        kds.push_back(Processor::LOC_PROC);
        kds.push_back(Processor::IO_PROC);
      }
      if (isa_prop & (PTX_ISA | CUDA_ISA))
        kds.push_back(Processor::TOC_PROC);
    }

    /////////////////////////////////////////////////////////////
    // ResourceConstraint
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ResourceConstraint::ResourceConstraint(ResourceKind resource,
                                           EqualityKind equality, size_t val)
      : resource_kind(resource), equality_kind(equality), value(val)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // LaunchConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LaunchConstraint::LaunchConstraint(LaunchKind kind, size_t value)
      : launch_kind(kind), dims(1)
    //--------------------------------------------------------------------------
    {
      values[0] = value;
    }

    //--------------------------------------------------------------------------
    LaunchConstraint::LaunchConstraint(LaunchKind kind, const size_t *vs, int d)
      : launch_kind(kind), dims(d)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(dims < 3);
#endif
      for (int i = 0; i < dims; i++)
        values[i] = vs[i];
    }

    /////////////////////////////////////////////////////////////
    // ColocationConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColocationConstraint::ColocationConstraint(unsigned index1, unsigned index2)
    //--------------------------------------------------------------------------
    {
      indexes.insert(index1);
      indexes.insert(index2);
    }

    //--------------------------------------------------------------------------
    ColocationConstraint::ColocationConstraint(const std::vector<unsigned> &idx)
      : indexes(idx.begin(), idx.end())
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // ColocationConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExecutionConstraintSet& ExecutionConstraintSet::add_constraint(
                                                const ISAConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      isa_constraint = constraint;
      return *this;
    }

    //--------------------------------------------------------------------------
    ExecutionConstraintSet& ExecutionConstraintSet::add_constraint(
                                           const ResourceConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      resource_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    ExecutionConstraintSet& ExecutionConstraintSet::add_constraint(
                                             const LaunchConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      launch_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    ExecutionConstraintSet& ExecutionConstraintSet::add_constraint(
                                         const ColocationConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      colocation_constraints.push_back(constraint);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // Specialized Constraint
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SpecializedConstraint::SpecializedConstraint(SpecializedKind k)
      : kind(k)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Memory Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MemoryConstraint::MemoryConstraint(void)
      : has_kind(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MemoryConstraint::MemoryConstraint(Memory::Kind k)
      : kind(k), has_kind(true)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Field Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldConstraint::FieldConstraint(const std::vector<FieldID> &order, bool cg)
      : ordering(order), contiguous(cg)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Ordering Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OrderingConstraint::OrderingConstraint(
                           const std::vector<DimensionKind> &order, bool contig)
      : ordering(order), contiguous(contig)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Splitting Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SplittingConstraint::SplittingConstraint(DimensionKind k, size_t v, bool ck)
      : kind(k), value(v), chunks(ck)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Dimension Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DimensionConstraint::DimensionConstraint(DimensionKind k, 
                                             EqualityKind eq, size_t val)
      : kind(k), eqk(eq), value(val)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Alignment Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AlignmentConstraint::AlignmentConstraint(FieldID f, 
                                             EqualityKind eq, size_t align)
      : fid(f), eqk(eq), alignment(align)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Offset Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OffsetConstraint::OffsetConstraint(FieldID f, size_t off)
      : fid(f), offset(off)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Pointer Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointerConstraint::PointerConstraint(void)
      : is_valid(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointerConstraint::PointerConstraint(FieldID f, uintptr_t p, Memory m)
      : is_valid(true), fid(f), ptr(p), memory(m)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Layout Constraint Set 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                        const SpecializedConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      specialized_constraint = constraint;
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                             const MemoryConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      memory_constraint = constraint;
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                           const OrderingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      ordering_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                          const SplittingConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      splitting_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                              const FieldConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      field_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                          const DimensionConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      dimension_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                          const AlignmentConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      alignment_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                             const OffsetConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      offset_constraints.push_back(constraint);
      return *this;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintSet& LayoutConstraintSet::add_constraint(
                                            const PointerConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      pointer_constraint = constraint;
      return *this;
    }

  };
};

// EOF

