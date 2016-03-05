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

#include "legion_utilities.h"
#include "legion_constraint.h"

namespace Legion {

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
    void ISAConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(isa_prop);
    }

    //--------------------------------------------------------------------------
    void ISAConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(isa_prop);
    }

    /////////////////////////////////////////////////////////////
    // Processor Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorConstraint::ProcessorConstraint(void)
      : valid(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProcessorConstraint::ProcessorConstraint(Processor::Kind k)
      : kind(k), valid(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ProcessorConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(valid);
      if (valid)
        rez.serialize(kind);
    }
    
    //--------------------------------------------------------------------------
    void ProcessorConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid);
      if (valid)
        derez.deserialize(kind);
    }

    /////////////////////////////////////////////////////////////
    // ResourceConstraint
    /////////////////////////////////////////////////////////////


    //--------------------------------------------------------------------------
    ResourceConstraint::ResourceConstraint(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceConstraint::ResourceConstraint(ResourceKind resource,
                                           EqualityKind equality, size_t val)
      : resource_kind(resource), equality_kind(equality), value(val)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ResourceConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(resource_kind);
      rez.serialize(equality_kind);
      rez.serialize(value);
    }

    //--------------------------------------------------------------------------
    void ResourceConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(resource_kind);
      derez.deserialize(equality_kind);
      derez.deserialize(value);
    }

    /////////////////////////////////////////////////////////////
    // LaunchConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LaunchConstraint::LaunchConstraint(void)
      : dims(0)
    //--------------------------------------------------------------------------
    {
    }

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

    //--------------------------------------------------------------------------
    void LaunchConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(launch_kind);
      rez.serialize(dims);
      for (int i = 0; i < dims; i++)
        rez.serialize(values[i]);
    }

    //--------------------------------------------------------------------------
    void LaunchConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(launch_kind);
      derez.deserialize(dims);
      for (int i = 0; i < dims; i++)
        derez.deserialize(values[i]);
    }

    /////////////////////////////////////////////////////////////
    // ColocationConstraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ColocationConstraint::ColocationConstraint(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ColocationConstraint::ColocationConstraint(unsigned index1, unsigned index2,
                                               const std::set<FieldID> &fids)
    //--------------------------------------------------------------------------
    {
      indexes.insert(index1);
      indexes.insert(index2);
      fields = fids;
    }

    //--------------------------------------------------------------------------
    ColocationConstraint::ColocationConstraint(const std::vector<unsigned> &idx,
                                               const std::set<FieldID> &fids)
      : fields(fids), indexes(idx.begin(), idx.end())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ColocationConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(indexes.size());
      for (std::set<unsigned>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
        rez.serialize(*it);
      rez.serialize<size_t>(fields.size());
      for (std::set<FieldID>::const_iterator it = fields.begin();
            it != fields.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void ColocationConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_indexes;
      derez.deserialize(num_indexes);
      for (unsigned idx = 0; idx < num_indexes; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        indexes.insert(index);
      }
      size_t num_fields;
      derez.deserialize(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        fields.insert(fid);
      }
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
                                          const ProcessorConstraint &constraint)
    //--------------------------------------------------------------------------
    {
      processor_constraint = constraint;
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

    //--------------------------------------------------------------------------
    void ExecutionConstraintSet::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      isa_constraint.serialize(rez);
      processor_constraint.serialize(rez);
#define PACK_CONSTRAINTS(Type, constraints)                             \
      rez.serialize<size_t>(constraints.size());                        \
      for (std::vector<Type>::const_iterator it = constraints.begin();  \
            it != constraints.end(); it++)                              \
      {                                                                 \
        it->serialize(rez);                                             \
      }
      PACK_CONSTRAINTS(ResourceConstraint, resource_constraints)
      PACK_CONSTRAINTS(LaunchConstraint, launch_constraints)
      PACK_CONSTRAINTS(ColocationConstraint, colocation_constraints)
#undef PACK_CONSTRAINTS
    }

    //--------------------------------------------------------------------------
    void ExecutionConstraintSet::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      isa_constraint.deserialize(derez);
      processor_constraint.deserialize(derez);
#define UNPACK_CONSTRAINTS(Type, constraints)                       \
      {                                                             \
        size_t constraint_size;                                     \
        derez.deserialize(constraint_size);                         \
        constraints.resize(constraint_size);                        \
        for (std::vector<Type>::iterator it = constraints.begin();  \
              it != constraints.end(); it++)                        \
        {                                                           \
          it->deserialize(derez);                                   \
        }                                                           \
      }
      UNPACK_CONSTRAINTS(ResourceConstraint, resource_constraints)
      UNPACK_CONSTRAINTS(LaunchConstraint, launch_constraints)
      UNPACK_CONSTRAINTS(ColocationConstraint, colocation_constraints)
#undef UNPACK_CONSTRAINTS
    }

    /////////////////////////////////////////////////////////////
    // Specialized Constraint
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SpecializedConstraint::SpecializedConstraint(SpecializedKind k,
                                                 ReductionOpID r)
      : kind(k), redop(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SpecializedConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(kind);
      if ((kind == REDUCTION_FOLD_SPECIALIZE) || 
          (kind == REDUCTION_LIST_SPECIALIZE))
        rez.serialize(redop);
    }

    //--------------------------------------------------------------------------
    void SpecializedConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(kind);
      if ((kind == REDUCTION_FOLD_SPECIALIZE) || 
          (kind == REDUCTION_LIST_SPECIALIZE))
        derez.deserialize(redop);
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

    //--------------------------------------------------------------------------
    void MemoryConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(has_kind);
      if (has_kind)
        rez.serialize(kind);
    }

    //--------------------------------------------------------------------------
    void MemoryConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(has_kind);
      if (has_kind)
        derez.deserialize(kind);
    }

    /////////////////////////////////////////////////////////////
    // Field Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldConstraint::FieldConstraint(void)
      : contiguous(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldConstraint::FieldConstraint(const std::vector<FieldID> &set, 
                                     bool cg, bool in)
      : field_set(set), contiguous(cg), inorder(in)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldConstraint::FieldConstraint(const std::set<FieldID> &set,
                                     bool cg, bool in)
      : field_set(set.begin(),set.end()), contiguous(cg), inorder(in)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void FieldConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(contiguous);
      rez.serialize(inorder);
      rez.serialize<size_t>(field_set.size());
      for (std::vector<FieldID>::const_iterator it = field_set.begin();
            it != field_set.end(); it++)
        rez.serialize(*it);
    }
    
    //--------------------------------------------------------------------------
    void FieldConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(contiguous);
      derez.deserialize(inorder);
      size_t num_orders;
      derez.deserialize(num_orders);
      field_set.resize(num_orders);
      for (std::vector<FieldID>::iterator it = field_set.begin();
            it != field_set.end(); it++)
        derez.deserialize(*it);
    }

    /////////////////////////////////////////////////////////////
    // Ordering Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OrderingConstraint::OrderingConstraint(void)
      : contiguous(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OrderingConstraint::OrderingConstraint(
                           const std::vector<DimensionKind> &order, bool contig)
      : ordering(order), contiguous(contig)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void OrderingConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(contiguous);
      rez.serialize<size_t>(ordering.size());
      for (std::vector<DimensionKind>::const_iterator it = ordering.begin();
            it != ordering.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void OrderingConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(contiguous);
      size_t num_orders;
      derez.deserialize(num_orders);
      ordering.resize(num_orders);
      for (std::vector<DimensionKind>::iterator it = ordering.begin();
            it != ordering.end(); it++)
        derez.deserialize(*it);
    }

    /////////////////////////////////////////////////////////////
    // Splitting Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SplittingConstraint::SplittingConstraint(void)
      : chunks(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SplittingConstraint::SplittingConstraint(DimensionKind k)
      : kind(k), chunks(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SplittingConstraint::SplittingConstraint(DimensionKind k, size_t v)
      : kind(k), value(v), chunks(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SplittingConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(kind);
      rez.serialize(chunks);
      if (!chunks)
        rez.serialize(value);
    }

    //--------------------------------------------------------------------------
    void SplittingConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(kind);
      derez.deserialize(chunks);
      if (!chunks)
        derez.deserialize(value);
    }

    /////////////////////////////////////////////////////////////
    // Dimension Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DimensionConstraint::DimensionConstraint(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DimensionConstraint::DimensionConstraint(DimensionKind k, 
                                             EqualityKind eq, size_t val)
      : kind(k), eqk(eq), value(val)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DimensionConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(kind);
      rez.serialize(eqk);
      rez.serialize(value);
    }

    //--------------------------------------------------------------------------
    void DimensionConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(kind);
      derez.deserialize(eqk);
      derez.deserialize(value);
    }

    /////////////////////////////////////////////////////////////
    // Alignment Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AlignmentConstraint::AlignmentConstraint(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AlignmentConstraint::AlignmentConstraint(FieldID f, 
                                             EqualityKind eq, size_t align)
      : fid(f), eqk(eq), alignment(align)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void AlignmentConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(fid);
      rez.serialize(eqk);
      rez.serialize(alignment);
    }

    //--------------------------------------------------------------------------
    void AlignmentConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(fid);
      derez.deserialize(eqk);
      derez.deserialize(alignment);
    }

    /////////////////////////////////////////////////////////////
    // Offset Constraint 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OffsetConstraint::OffsetConstraint(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OffsetConstraint::OffsetConstraint(FieldID f, size_t off)
      : fid(f), offset(off)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void OffsetConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(fid);
      rez.serialize(offset);
    }

    //--------------------------------------------------------------------------
    void OffsetConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(fid);
      derez.deserialize(offset);
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

    //--------------------------------------------------------------------------
    void PointerConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(is_valid);
      if (is_valid)
      {
        rez.serialize(fid);
        rez.serialize(ptr);
        rez.serialize(memory);
      }
    }

    //--------------------------------------------------------------------------
    void PointerConstraint::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(is_valid);
      if (is_valid)
      {
        derez.deserialize(fid);
        derez.deserialize(ptr);
        derez.deserialize(memory);
      }
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
      ordering_constraint = constraint;
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
      field_constraint = constraint;
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

    //--------------------------------------------------------------------------
    void LayoutConstraintSet::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      specialized_constraint.serialize(rez);
      field_constraint.serialize(rez);
      memory_constraint.serialize(rez);
      pointer_constraint.serialize(rez);
      ordering_constraint.serialize(rez);
#define PACK_CONSTRAINTS(Type, constraints)                             \
      rez.serialize<size_t>(constraints.size());                        \
      for (std::vector<Type>::const_iterator it = constraints.begin();  \
            it != constraints.end(); it++)                              \
      {                                                                 \
        it->serialize(rez);                                             \
      }
      PACK_CONSTRAINTS(SplittingConstraint, splitting_constraints)
      PACK_CONSTRAINTS(DimensionConstraint, dimension_constraints)
      PACK_CONSTRAINTS(AlignmentConstraint, alignment_constraints)
      PACK_CONSTRAINTS(OffsetConstraint, offset_constraints)
#undef PACK_CONSTRAINTS
    }

    //--------------------------------------------------------------------------
    void LayoutConstraintSet::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      specialized_constraint.deserialize(derez);
      memory_constraint.deserialize(derez);
      pointer_constraint.deserialize(derez);
#define UNPACK_CONSTRAINTS(Type, constraints)                       \
      {                                                             \
        size_t constraint_size;                                     \
        derez.deserialize(constraint_size);                         \
        constraints.resize(constraint_size);                        \
        for (std::vector<Type>::iterator it = constraints.begin();  \
              it != constraints.end(); it++)                        \
        {                                                           \
          it->deserialize(derez);                                   \
        }                                                           \
      }
      UNPACK_CONSTRAINTS(SplittingConstraint, splitting_constraints)
      UNPACK_CONSTRAINTS(DimensionConstraint, dimension_constraints)
      UNPACK_CONSTRAINTS(AlignmentConstraint, alignment_constraints)
      UNPACK_CONSTRAINTS(OffsetConstraint, offset_constraints)
#undef UNPACK_CONSTRAINTS
    }

    /////////////////////////////////////////////////////////////
    // Task Layout Constraint Set 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskLayoutConstraintSet& TaskLayoutConstraintSet::add_layout_constraint(
                                          unsigned idx, LayoutConstraintID desc)
    //--------------------------------------------------------------------------
    {
      layouts.insert(std::pair<unsigned,LayoutConstraintID>(idx, desc));
      return *this;
    }

    //--------------------------------------------------------------------------
    void TaskLayoutConstraintSet::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(layouts.size());
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layouts.begin(); it != layouts.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void TaskLayoutConstraintSet::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_layouts;
      derez.deserialize(num_layouts);
      for (unsigned idx = 0; idx < num_layouts; idx++)
      {
        std::pair<unsigned,LayoutConstraintID> pair;
        derez.deserialize(pair.first);
        derez.deserialize(pair.second);
        layouts.insert(pair);
      }
    }

}; // namespace Legion

// EOF

