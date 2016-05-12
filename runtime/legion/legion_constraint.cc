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
  
    // some helper methods

    //--------------------------------------------------------------------------
    static inline bool bound_entails(EqualityKind eq1, long v1,
                                     EqualityKind eq2, long v2)
    //--------------------------------------------------------------------------
    {
      switch (eq1)
      {
        case LT_EK: // < v1
          {
            // Can entail for <, <=, !=  
            if ((eq2 == LT_EK) && (v1 <= v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 < v2)) // <= v2
              return true;
            if ((eq2 == NE_EK) && (v1 <= v2)) // != v2
              return true;
            return false;
          }
        case LE_EK: // <= v1
          {
            // Can entail for <, <=, !=
            if ((eq2 == LT_EK) && (v1 < v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 <= v2)) // <= v2
              return true;
            if ((eq2 == NE_EK) && (v1 < v2)) // != v2
              return true;
            return false;
          }
        case GT_EK: // > v1
          {
            // Can entail for >, >=, !=
            if ((eq2 == GT_EK) && (v1 >= v2)) // > v2
              return true;
            if ((eq2 == GE_EK) && (v1 > v2)) // >= v2
              return true;
            if ((eq2 == NE_EK) && (v1 >= v2)) // != v2
              return true;
            return false;
          }
        case GE_EK: // >= v1
          {
            // Can entail for >, >=, !=
            if ((eq2 == GT_EK) && (v1 > v2)) // > v2
              return true;
            if ((eq2 == GE_EK) && (v1 >= v2)) // >= v2
              return true;
            if ((eq2 == NE_EK) && (v1 > v2)) // != v2
              return true;
            return false;
          }
        case EQ_EK: // == v1
          {
            // Can entail for <, <=, >, >=, ==, !=
            if ((eq2 == LT_EK) && (v1 < v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 <= v2)) // <= v2
              return true;
            if ((eq2 == GT_EK) && (v1 > v2)) // > v2
              return true;
            if ((eq2 == GE_EK) && (v1 >= v2)) // >= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 == v2)) // == v2
              return true;
            if ((eq2 == NE_EK) && (v1 != v2)) // != v2
              return true;
            return false;
          }
        case NE_EK: // != v1
          {
            // Can only entail for != of the same value
            if ((eq2 == NE_EK) && (v1 == v2)) // != v2
              return true;
            return false;
          }
        default:
          assert(false); // unknown
      }
      return false;
    }

    //--------------------------------------------------------------------------
    static inline bool bound_conflicts(EqualityKind eq1, long v1,
                                       EqualityKind eq2, long v2)
    //--------------------------------------------------------------------------
    {
      switch (eq1)
      {
        case LT_EK: // < v1
          {
            // conflicts with >, >=, ==
            if ((eq2 == GT_EK) && ((v1-1) <= v2)) // > v2
              return true;
            if ((eq2 == GE_EK) && (v1 <= v2)) // >= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 <= v2)) // == v2
              return true;
            return false;
          }
        case LE_EK: // <= v1
          {
            // conflicts with >, >=, == 
            if ((eq2 == GT_EK) && (v1 <= v2)) // > v2
              return true;
            if ((eq2 == GE_EK) && (v1 < v2)) // >= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 < v2)) // == v2
              return true;
            return false;
          }
        case GT_EK: // > v1
          {
            // coflicts with <, <=, ==
            if ((eq2 == LT_EK) && ((v1+1) >= v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 >= v2)) // <= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 >= v2)) // == v2
              return true;
            return false;
          }
        case GE_EK: // >= v1
          {
            // conflicts with <, <=, ==
            if ((eq2 == LT_EK) && (v1 >= v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 > v2)) // <= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 > v2)) // == v2
              return true;
            return false;
          }
        case EQ_EK: // == v1
          {
            // conflicts with <, <=, >, >=, ==, !=
            if ((eq2 == LT_EK) && (v1 >= v2)) // < v2
              return true;
            if ((eq2 == LE_EK) && (v1 > v2)) // <= v2
              return true;
            if ((eq2 == GT_EK) && (v1 <= v2)) // > v2
              return true;
            if ((eq2 == GT_EK) && (v1 < v2)) // >= v2
              return true;
            if ((eq2 == EQ_EK) && (v1 != v2)) // == v2
              return true;
            if ((eq2 == NE_EK) && (v1 == v2)) // != v2
              return true;
            return false;
          }
        case NE_EK: // != v1
          {
            // conflicts with ==
            if ((eq2 == EQ_EK) && (v1 == v2)) // == v2
              return true;
            return false;
          }
        default:
          assert(false); // unknown
      }
      return false;
    }

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
#ifdef DEBUG_LEGION
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
      if (redop == 0)
      {
        if ((kind != NORMAL_SPECIALIZE) && (kind != VIRTUAL_SPECIALIZE))
        {
          fprintf(stderr,"Illegal specialize constraint with reduction op %d."
                         "Only reduction specialized constraints are "
                         "permitted to have non-zero reduction operators.",
                         redop);
          assert(false);
        }
      }
      else
      {
        if ((kind != REDUCTION_FOLD_SPECIALIZE) && 
            (kind != REDUCTION_LIST_SPECIALIZE))
        {
          fprintf(stderr,"Illegal %s reduction specialized constraint with "
                         "reduction op '0'. Zero is a reserved reduction "
                         "operator ID and cannot be used for any reductions "
                         "and therefore is an invalid constraint.",
                         (kind == REDUCTION_FOLD_SPECIALIZE) ? "fold" : "list");
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SpecializedConstraint::entails(const SpecializedConstraint &other)const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return false;
      if (redop != other.redop)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool SpecializedConstraint::conflicts(
                                       const SpecializedConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return true;
      if (redop != other.redop)
        return true;
      return false;
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
    bool MemoryConstraint::entails(const MemoryConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (!other.has_kind)
        return true;
      if (!has_kind)
        return false;
      if (kind == other.kind)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    bool MemoryConstraint::conflicts(const MemoryConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (!has_kind || !other.has_kind)
        return false;
      if (kind != other.kind)
        return true;
      return false;
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
    bool FieldConstraint::entails(const FieldConstraint &other) const
    //--------------------------------------------------------------------------
    {
      // Handle empty field sets quickly
      if (other.field_set.empty())
        return true;
      if (field_set.empty())
        return false;
      if (field_set.size() < other.field_set.size())
        return false; // can't have all the fields
      // Find the indexes of the other fields in our set
      std::vector<unsigned> field_indexes(other.field_set.size());
      unsigned local_idx = 0;
      for (std::vector<FieldID>::const_iterator it = other.field_set.begin();
            it != other.field_set.end(); it++,local_idx++)
      {
        bool found = false;
        for (unsigned idx = 0; idx < field_set.size(); idx++)
        {
          if (field_set[idx] == (*it))
          {
            field_indexes[local_idx] = idx;
            found = true;
            break;
          }
        }
        if (!found)
          return false; // can't entail if we don't have the field
      }
      if (other.contiguous)
      {
        if (other.inorder)
        {
          // Other is both inorder and contiguous
          // If we're not both contiguous and inorder we can't entail it
          if (!contiguous || !inorder)
            return false;
          // See if our fields are in order and grow by one each time 
          for (unsigned idx = 1; idx < field_indexes.size(); idx++)
          {
            if ((field_indexes[idx-1]+1) != field_indexes[idx])
              return false;
          }
          return true;
        }
        else
        {
          // Other is contiguous but not inorder
          // If we're not contiguous we can't entail it
          if (!contiguous)
            return false;
          // See if all our indexes are continuous 
          std::set<unsigned> sorted_indexes(field_indexes.begin(),
                                            field_indexes.end());
          int previous = -1;
          for (std::set<unsigned>::const_iterator it = sorted_indexes.begin();
                it != sorted_indexes.end(); it++)
          {
            if (previous != -1)
            {
              if ((previous+1) != int(*it))
                return false;
            }
            previous = (*it); 
          }
          return true;
        }
      }
      else
      {
        if (other.inorder)
        {
          // Other is inorder but not contiguous
          // If we're not inorder we can't entail it
          if (!inorder)
            return false;
          // Must be in order but not necessarily contiguous
          // See if our indexes are monotonically increasing 
          for (unsigned idx = 1; idx < field_indexes.size(); idx++)
          {
            // Not monotonically increasing
            if (field_indexes[idx-1] > field_indexes[idx])
              return false;
          }
          return true;
        }
        else
        {
          // Other is neither inorder or contiguous
          // We already know we have all the fields so we are done 
          return true;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool FieldConstraint::conflicts(const FieldConstraint &other) const
    //--------------------------------------------------------------------------
    {
      // The only way for us to conflict is if we want fields to be inorder
      // and the other one wants them also to be inorder but a different order
      if (inorder && other.inorder)
      {
        // Both inorder, see if our fields come in the same order 
        // Do this different if they both are contigous or not
        if (contiguous && other.contiguous)
        {
          int previous_idx = -1;
          for (std::vector<FieldID>::const_iterator it = field_set.begin();
                it != field_set.end(); it++)
          {
            int next_idx = -1;
            for (unsigned idx = 0; idx < other.field_set.size(); idx++)
            {
              if ((*it) == other.field_set[idx])
              {
                next_idx = idx;
                break;
              }
            }
            if (next_idx >= 0)
            {
              // This field was in the other set, see if it was in a good place
              if (previous_idx >= 0)
              {
                if ((next_idx) != (previous_idx+1))
                  return true; // conflict!
              }
              // Record the previous idx and keep going
              previous_idx = next_idx;
            }
            else if (previous_idx >= 0)
              return true; // fields are not contiguous
          }
        }
        else
        {
          int previous_idx = -1;
          for (std::vector<FieldID>::const_iterator it = field_set.begin();
                it != field_set.end(); it++)
          {
            int next_idx = -1;
            for (unsigned idx = 0; idx < other.field_set.size(); idx++)
            {
              if ((*it) == other.field_set[idx])
              {
                next_idx = idx;
                break;
              }
            }
            // Only care if we found it
            if (next_idx >= 0)
            {
              // This field was in the other set, see if it was in a good place
              if (previous_idx >= 0)
              {
                if (next_idx < previous_idx)
                  return true; // conflict!
              }
              // Record the previous idx and keep going
              previous_idx = next_idx;
            }
          }
        }
      }
      return false;  
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
    bool OrderingConstraint::entails(const OrderingConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (other.ordering.empty())
        return true;
      // We don't even have enough fields so no way we can entail
      if (ordering.size() < other.ordering.size())
        return false;
      // See if we have all the dimensions
      std::vector<unsigned> dim_indexes(other.ordering.size());
      unsigned local_idx = 0;
      for (std::vector<DimensionKind>::const_iterator it = 
           other.ordering.begin(); it != other.ordering.end(); it++,local_idx++)
      {
        bool found = false;
        for (unsigned idx = 0; idx < ordering.size(); idx++)
        {
          if (ordering[idx] == (*it))
          {
            dim_indexes[local_idx] = idx;
            // If they aren't in the same order, it is no good
            if ((local_idx > 0) && (dim_indexes[local_idx-1] > idx))
              return false;
            found = true;
            break;
          }
        }
        if (!found)
          return false; // if we don't have the dimension can't entail
      }
      if (other.contiguous)
      {
        // If we're not contiguous we can't entail the other
        if (!contiguous)
          return false;
        // See if the indexes are contiguous
        std::set<unsigned> sorted_indexes(dim_indexes.begin(), 
                                          dim_indexes.end());
        int previous = -1;
        for (std::set<unsigned>::const_iterator it = sorted_indexes.begin();
              it != sorted_indexes.end(); it++)
        {
          if (previous != -1)
          {
            // Not contiguous
            if ((previous+1) != int(*it))
              return false;
          }
          previous = (*it);
        }
        return true;
      }
      else
      {
        // We've got all the dimensions in the right order so we are good
        return true; 
      }
    }

    //--------------------------------------------------------------------------
    bool OrderingConstraint::conflicts(const OrderingConstraint &other) const
    //--------------------------------------------------------------------------
    {
      // If they both must be contiguous there is a slightly different check      
      if (contiguous && other.contiguous)
      {
        int previous_idx = -1;
        for (std::vector<DimensionKind>::const_iterator it = ordering.begin();
              it != ordering.end(); it++)
        {
          int next_idx = -1;
          for (unsigned idx = 0; idx < other.ordering.size(); idx++)
          {
            if ((*it) == other.ordering[idx])
            {
              next_idx = idx;
              break;
            }
          }
          if (next_idx >= 0)
          {
            // This field was in the other set, see if it was in a good place
            if (previous_idx >= 0)
            {
              if (next_idx != (previous_idx+1))
                return true; // conflict
            }
            // Record the previous and keep going
            previous_idx = next_idx;
          }
          else if (previous_idx >= 0)
            return true; // fields are not contiguous
        }
      }
      else
      {
        int previous_idx = -1;
        for (std::vector<DimensionKind>::const_iterator it = ordering.begin();
              it != ordering.end(); it++)
        {
          int next_idx = -1;
          for (unsigned idx = 0; idx < other.ordering.size(); idx++)
          {
            if ((*it) == other.ordering[idx])
            {
              next_idx = idx;
              break;
            }
          }
          // Only care if we found it
          if (next_idx >= 0)
          {
            if ((previous_idx >= 0) && (next_idx < previous_idx))
              return true; // not in the right order
            // Record this as the previous
            previous_idx = next_idx;
          }
        }
      }
      return false;
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
    bool SplittingConstraint::entails(const SplittingConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return false;
      if (value != other.value)
        return false;
      if (chunks != other.value)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool SplittingConstraint::conflicts(const SplittingConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return false;
      if (value != other.value)
        return true;
      if (chunks != other.chunks)
        return true;
      return false;
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
    bool DimensionConstraint::entails(const DimensionConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return false;
      if (bound_entails(eqk, value, other.eqk, other.value))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    bool DimensionConstraint::conflicts(const DimensionConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (kind != other.kind)
        return false;
      if (bound_conflicts(eqk, value, other.eqk, other.value))
        return true;
      return false;
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
    bool AlignmentConstraint::entails(const AlignmentConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (fid != other.fid)
        return false;
      if (bound_entails(eqk, alignment, other.eqk, other.alignment))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    bool AlignmentConstraint::conflicts(const AlignmentConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (fid != other.fid)
        return false;
      if (bound_conflicts(eqk, alignment, other.eqk, other.alignment))
        return true;
      return false;
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
    bool OffsetConstraint::entails(const OffsetConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (fid != other.fid)
        return false;
      if (offset == other.offset)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    bool OffsetConstraint::conflicts(const OffsetConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (fid != other.fid)
        return false;
      if (offset != other.offset)
        return true;
      return false;
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
    PointerConstraint::PointerConstraint(Memory m, uintptr_t p)
      : is_valid(true), memory(m), ptr(p)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PointerConstraint::entails(const PointerConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (!other.is_valid)
        return true;
      if (!is_valid)
        return false;
      if (memory != other.memory)
        return false;
      if (ptr != other.ptr)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointerConstraint::conflicts(const PointerConstraint &other) const
    //--------------------------------------------------------------------------
    {
      if (!is_valid || !other.is_valid)
        return false;
      if (memory != other.memory)
        return false;
      if (ptr != other.ptr)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void PointerConstraint::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(is_valid);
      if (is_valid)
      {
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
    bool LayoutConstraintSet::entails(const LayoutConstraintSet &other) const
    //--------------------------------------------------------------------------
    {
      if (!specialized_constraint.entails(other.specialized_constraint))
        return false;
      if (!field_constraint.entails(other.field_constraint))
        return false;
      if (!memory_constraint.entails(other.memory_constraint))
        return false;
      if (!pointer_constraint.entails(other.pointer_constraint))
        return false;
      if (!ordering_constraint.entails(other.ordering_constraint))
        return false;
      for (std::vector<SplittingConstraint>::const_iterator it = 
            other.splitting_constraints.begin(); it !=
            other.splitting_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < splitting_constraints.size(); idx++)
        {
          if (splitting_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<DimensionConstraint>::const_iterator it = 
            other.dimension_constraints.begin(); it != 
            other.dimension_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < dimension_constraints.size(); idx++)
        {
          if (dimension_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<AlignmentConstraint>::const_iterator it = 
            other.alignment_constraints.begin(); it != 
            other.alignment_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < alignment_constraints.size(); idx++)
        {
          if (alignment_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      for (std::vector<OffsetConstraint>::const_iterator it = 
            other.offset_constraints.begin(); it != 
            other.offset_constraints.end(); it++)
      {
        bool entailed = false;
        for (unsigned idx = 0; idx < offset_constraints.size(); idx++)
        {
          if (offset_constraints[idx].entails(*it))
          {
            entailed = true;
            break;
          }
        }
        if (!entailed)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LayoutConstraintSet::conflicts(const LayoutConstraintSet &other) const
    //--------------------------------------------------------------------------
    {
      // Do these in order
      if (specialized_constraint.conflicts(other.specialized_constraint))
        return true;
      if (field_constraint.conflicts(other.field_constraint))
        return true;
      if (memory_constraint.conflicts(other.memory_constraint))
        return true;
      if (pointer_constraint.conflicts(other.pointer_constraint))
        return true;
      if (ordering_constraint.conflicts(other.ordering_constraint))
        return true;
      for (std::vector<SplittingConstraint>::const_iterator it = 
            splitting_constraints.begin(); it != 
            splitting_constraints.end(); it++)
      {
        for (unsigned idx = 0; idx < other.splitting_constraints.size(); idx++)
          if (it->conflicts(other.splitting_constraints[idx]))
            return true;
      }
      for (std::vector<DimensionConstraint>::const_iterator it = 
            dimension_constraints.begin(); it !=
            dimension_constraints.end(); it++)
      {
        for (unsigned idx = 0; idx < other.dimension_constraints.size(); idx++)
          if (it->conflicts(other.dimension_constraints[idx]))
            return true;
      }
      for (std::vector<AlignmentConstraint>::const_iterator it = 
            alignment_constraints.begin(); it !=
            alignment_constraints.end(); it++)
      {
        for (unsigned idx = 0; idx < other.alignment_constraints.size(); idx++)
          if (it->conflicts(other.alignment_constraints[idx]))
            return true;
      }
      for (std::vector<OffsetConstraint>::const_iterator it = 
            offset_constraints.begin(); it != 
            offset_constraints.end(); it++)
      {
        for (unsigned idx = 0; idx < other.offset_constraints.size(); idx++)
          if (it->conflicts(other.offset_constraints[idx]))
            return true;
      }
      return false;
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
      field_constraint.deserialize(derez);
      memory_constraint.deserialize(derez);
      pointer_constraint.deserialize(derez);
      ordering_constraint.deserialize(derez);
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

