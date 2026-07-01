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

#ifndef __LEGION_REGISTRARS_H__
#define __LEGION_REGISTRARS_H__

#include "legion/api/constraints.h"
#include "legion/api/data.h"

namespace Legion {

  /**
   * \struct LayoutConstraintRegistrar
   * A layout description registrar is the mechanism by which application
   * can register a set of constraints with a specific ID. This ID can
   * then be used globally to refer to this set of constraints. All
   * constraint sets are associated with a specifid field space which
   * contains the FieldIDs used in the constraints. This can be a
   * NO_SPACE if there are no field constraints. All the rest of the
   * constraints can be optionally specified.
   */
  struct LayoutConstraintRegistrar {
  public:
    LayoutConstraintRegistrar(void);
    LayoutConstraintRegistrar(
        FieldSpace handle, const char* layout_name = nullptr);
  public:
    inline LayoutConstraintRegistrar& add_constraint(
        const SpecializedConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const MemoryConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const OrderingConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const TilingConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const FieldConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const DimensionConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const AlignmentConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const OffsetConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const PointerConstraint& constraint);
    inline LayoutConstraintRegistrar& add_constraint(
        const PaddingConstraint& constraint);
  public:
    FieldSpace handle;
    LayoutConstraintSet layout_constraints;
    const char* layout_name;
  };

  /**
   * \struct PoolBounds
   * A small helper class for tracking the bounds on what
   * memory pools can support when they are created
   */
  struct PoolBounds {
  public:
    PoolBounds(UnboundPoolScope u, size_t s = 0)
      : size(s), alignment(0), scope(u)
    { }
    PoolBounds(size_t s = 0, uint32_t a = 16)
      : size(s), alignment(a), scope(LEGION_BOUNDED_POOL)
    { }
    PoolBounds(const PoolBounds&) = default;
    PoolBounds(PoolBounds&&) = default;
    PoolBounds& operator=(const PoolBounds&) = default;
    PoolBounds& operator=(PoolBounds&&) = default;
    inline bool is_bounded(void) const
    {
      return (scope == LEGION_BOUNDED_POOL);
    }
  public:
    // If this is a bounded pool then size is the number of bytes in the pool
    // If it is an unbounded pool then size is how many free bytes the pool
    // is allowed to keep locally from freed instances without returning
    // them back to the Realm allocator, zero means that all freed instances
    // are immediately sent back to the Realm allocator
    size_t size;             // upper bound of the pool in bytes
    uint32_t alignment;      // maximum alignment supported
    UnboundPoolScope scope;  // scope for unbound pools
  };

  /**
   * \struct TaskVariantRegistrar
   * This structure captures all the meta-data information necessary for
   * describing a task variant including the logical task ID, the execution
   * constraints, the layout constraints, and any properties of the task.
   * This structure is used for registering task variants and is also
   * the output type for variants created by generator tasks.
   */
  struct TaskVariantRegistrar {
  public:
    TaskVariantRegistrar(void);
    TaskVariantRegistrar(
        TaskID task_id, bool global = true, const char* variant_name = nullptr);
    TaskVariantRegistrar(
        TaskID task_id, const char* variant_name, bool global = true);
  public:  // Add execution constraints
    inline TaskVariantRegistrar& add_constraint(
        const ISAConstraint& constraint);
    inline TaskVariantRegistrar& add_constraint(
        const ProcessorConstraint& constraint);
    inline TaskVariantRegistrar& add_constraint(
        const ResourceConstraint& constraint);
    inline TaskVariantRegistrar& add_constraint(
        const LaunchConstraint& constraint);
    inline TaskVariantRegistrar& add_constraint(
        const ColocationConstraint& constraint);
  public:  // Add layout constraint sets
    inline TaskVariantRegistrar& add_layout_constraint_set(
        unsigned index, LayoutConstraintID desc);
  public:  // Set properties
    inline void set_leaf(bool is_leaf = true);
    inline void set_inner(bool is_inner = true);
    inline void set_idempotent(bool is_idempotent = true);
    inline void set_replicable(bool is_replicable = true);
    inline void set_concurrent(bool is_concurrent = true);
    inline void set_concurrent_barrier(bool needs_barrier = true);
  public:  // Generator Task IDs
    inline void add_generator_task(TaskID tid);
  public:
    TaskID task_id;
    bool global_registration;
    const char* task_variant_name;
  public:  // constraints
    ExecutionConstraintSet execution_constraints;
    TaskLayoutConstraintSet layout_constraints;
  public:
    // If this is a leaf task variant then the application can
    // request that the runtime preserve a pool in the memory of
    // the corresponding kind with the closest affinity to the target
    // processor for handling dynamic memory allocations during the
    // execution of the task. Pool bounds can also be set to request
    // an unbounded pool allocation. Note that requesting an unbound
    // memory allocation will likely result in severe performance degradation.
    // All these pool bounds are escaping meaning that allocations performed
    // in the resulting pools can outlive the lifetime of the task.
    std::map<Memory::Kind, PoolBounds> leaf_pool_bounds;
    // If the leaf task wants to make pools for non-escaping allocations
    // it can specify non-escaping leaf pool bounds. These must be
    // concrete pools as unbounded pools are always escaping.
    std::map<Memory::Kind, PoolBounds> non_escaping_leaf_pool_bounds;
  public:
    // TaskIDs for which this variant can serve as a generator
    std::set<TaskID> generator_tasks;
  public:  // properties
    bool leaf_variant;
    bool inner_variant;
    bool idempotent_variant;
    bool replicable_variant;
    bool concurrent_variant;
    bool concurrent_barrier;
  };

}  // namespace Legion

#include "legion/api/registrars.inl"

#endif  // __LEGION_REGISTRARS_H__
