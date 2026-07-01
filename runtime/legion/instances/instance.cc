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

#include "legion/instances/instance.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/field.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // InstanceManager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceManager::InstanceManager(
        DistributedID did, LayoutDescription* desc, FieldSpaceNode* node,
        IndexSpaceExpression* domain, RegionTreeID tid, bool register_now,
        CollectiveMapping* mapping)
      : DistributedCollectable(did, register_now, mapping), layout(desc),
        field_space_node(node), instance_domain(domain), tree_id(tid)
    //--------------------------------------------------------------------------
    {
      // Add a reference to the layout
      if (layout != nullptr)
        layout->add_reference();
      if (field_space_node != nullptr)
        field_space_node->add_nested_gc_ref(did);
      if (instance_domain != nullptr)
        instance_domain->add_nested_expression_reference(did);
    }

    //--------------------------------------------------------------------------
    InstanceManager::~InstanceManager(void)
    //--------------------------------------------------------------------------
    {
      if ((layout != nullptr) && layout->remove_reference())
        delete layout;
      if ((field_space_node != nullptr) &&
          field_space_node->remove_nested_gc_ref(did))
        delete field_space_node;
      if ((instance_domain != nullptr) &&
          instance_domain->remove_nested_expression_reference(did))
        delete instance_domain;
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::entails(
        LayoutConstraints* constraints,
        const LayoutConstraint** failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint& pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
        // Always test the pointer constraint locally
        if (!pointer_constraint.entails(constraints->pointer_constraint))
        {
          if (failed_constraint != nullptr)
            *failed_constraint = &pointer;
          return false;
        }
      }
      return layout->constraints->entails(
          constraints,
          (instance_domain != nullptr) ? instance_domain->get_num_dims() : 0,
          failed_constraint, false /*test pointer*/);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::entails(
        const LayoutConstraintSet& constraints,
        const LayoutConstraint** failed_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint& pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
        // Always test the pointer constraint locally
        if (!pointer_constraint.entails(constraints.pointer_constraint))
        {
          if (failed_constraint != nullptr)
            *failed_constraint = &pointer;
          return false;
        }
      }
      return layout->constraints->entails(
          constraints,
          (instance_domain != nullptr) ? instance_domain->get_num_dims() : 0,
          failed_constraint, false /*test pointer*/);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::conflicts(
        LayoutConstraints* constraints,
        const LayoutConstraint** conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint& pointer = constraints->pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
        // Always test the pointer constraint locally
        if (pointer_constraint.conflicts(constraints->pointer_constraint))
        {
          if (conflict_constraint != nullptr)
            *conflict_constraint = &pointer;
          return true;
        }
      }
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(
          constraints,
          (instance_domain != nullptr) ? instance_domain->get_num_dims() : 0,
          conflict_constraint);
    }

    //--------------------------------------------------------------------------
    bool InstanceManager::conflicts(
        const LayoutConstraintSet& constraints,
        const LayoutConstraint** conflict_constraint) const
    //--------------------------------------------------------------------------
    {
      const PointerConstraint& pointer = constraints.pointer_constraint;
      if (pointer.is_valid)
      {
        PointerConstraint pointer_constraint = get_pointer_constraint();
        // Always test the pointer constraint locally
        if (pointer_constraint.conflicts(constraints.pointer_constraint))
        {
          if (conflict_constraint != nullptr)
            *conflict_constraint = &pointer;
          return true;
        }
      }
      // We know our layouts don't have a pointer constraint so nothing special
      return layout->constraints->conflicts(
          constraints,
          (instance_domain != nullptr) ? instance_domain->get_num_dims() : 0,
          conflict_constraint);
    }

  }  // namespace Internal
}  // namespace Legion
