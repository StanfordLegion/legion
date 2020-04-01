/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_CONSTRAINT_H__
#define __LEGION_CONSTRAINT_H__

/**
 * \file legion_constraint.h
 */

#include "legion/legion_types.h"

namespace Legion {

#define LEGION_EXECUTION_CONSTRAINT_KINDS(__op__) \
  __op__(ISA_CONSTRAINT, "ISA") \
  __op__(PROCESSOR_CONSTRAINT, "Processor") \
  __op__(RESOURCE_CONSTRAINT, "Resource") \
  __op__(LAUNCH_CONSTRAINT, "Launch") \
  __op__(COLOCATION_CONSTRAINT, "Co-Location")

#define LEGION_LAYOUT_CONSTRAINT_KINDS(__op__) \
  __op__(SPECIALIZED_CONSTRAINT, "Specialized") \
  __op__(MEMORY CONSTRAINT, "Memory") \
  __op__(FIELD_CONSTRAINT, "Field") \
  __op__(ORDERING_CONSTRAINT, "Ordering") \
  __op__(SPLITTING_CONSTRAINT, "Splitting") \
  __op__(DIMENSION_CONSTRAINT, "Dimension") \
  __op__(ALIGNMENT_CONSTRAINT, "Alignment") \
  __op__(OFFSET_CONSTRAINT, "Offset") \
  __op__(POINTER_CONSTRAINT, "Pointer")

    /**
     * \class ISAConstraint
     * ISA constraints specify the kind of instruction constraints
     * a task variant requires for its execution. At a high-level this
     * will be controlling which kind of low-level runtime processor
     * we can map this onto.  Note that some constraints such as Terra
     * and LLVM do not correspond to actual low-level processor kinds, but
     * represent variants which can be JIT-ed to different target 
     * processors on demand. The LLVM and Terra kinds can also be mixed
     * with other constraints to say specifically which target processors
     * the task can be JIT-ed to.
     */
    class ISAConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                                ISA_CONSTRAINT;
    public:
      ISAConstraint(uint64_t prop = 0);
    public:
      bool entails(const ISAConstraint &other) const;
      bool conflicts(const ISAConstraint &other) const;
    public:
      void swap(ISAConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      uint64_t isa_prop;
    };

    /**
     * \class ProcessorConstraint
     * Processor constraints are used to declare that a task variant
     * should only be able to executed on processors of certain
     * kinds. This is necessary for example, to distinguish I/O tasks
     * which can run on all x86 cores by their ISA constraints, but
     * users want to restrict their execution to just I/O processors.
     */
    class ProcessorConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                            PROCESSOR_CONSTRAINT;
    public:
      ProcessorConstraint(Processor::Kind kind = Processor::NO_KIND);
    public:
      inline bool is_valid(void) const { return !valid_kinds.empty(); }
      void add_kind(Processor::Kind kind);
      bool can_use(Processor::Kind kind) const;
    public:
      bool entails(const ProcessorConstraint &other) const;
      bool conflicts(const ProcessorConstraint &other) const;
    public:
      void swap(ProcessorConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::vector<Processor::Kind> valid_kinds;
    };

    /**
     * \class ResourceConstraint
     * Resource constraints provide a way of specifying the expected
     * constraints for a task variant on a particular architecture.
     * The target low-level processor must meet the constraint in 
     * order for the invariant to run on the target processor.
     */
    class ResourceConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                            RESOURCE_CONSTRAINT;
    public:
      ResourceConstraint(void);
      ResourceConstraint(ResourceKind resource_kind, 
                         EqualityKind eq_kind, size_t value);
    public:
      bool entails(const ResourceConstraint &other) const;
      bool conflicts(const ResourceConstraint &other) const;
    public:
      void swap(ResourceConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      ResourceKind resource_kind;
      EqualityKind equality_kind;
      size_t value;
    };

    /**
     * \class LaunchConstraint
     * Constraints on the launch configuration for this task.
     * Right now this mainly applies to GPU tasks which can 
     * specify things like required CTA and grid shape. In
     * the future we can imagine extending this to other
     * processor kinds.
     */
    class LaunchConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                            LAUNCH_CONSTRAINT;
    public:
      LaunchConstraint(void);
      LaunchConstraint(LaunchKind kind, size_t value);
      LaunchConstraint(LaunchKind kind, const size_t *value, int dims);
    public:
      bool entails(const LaunchConstraint &other) const;
      bool conflicts(const LaunchConstraint &other) const;
    public:
      void swap(LaunchConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      LaunchKind launch_kind;
      size_t values[3];
      int dims;
    };

    /**
     * \class ColocationConstraint
     * Co-location constraints can be used to specify that two
     * region requirements for a task need to be located in the 
     * same physical instance for layout purposes.
     */
    class ColocationConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                            COLOCATION_CONSTRAINT;
    public:
      ColocationConstraint(void);
      ColocationConstraint(unsigned index1, unsigned index2,
                           const std::set<FieldID> &fields);
      ColocationConstraint(const std::vector<unsigned> &indexes,
                           const std::set<FieldID> &fields);
    public:
      bool entails(const ColocationConstraint &other) const;
      bool conflicts(const ColocationConstraint &other) const;
    public:
      void swap(ColocationConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::set<FieldID> fields;
      std::set<unsigned> indexes;
    };

    /**
     * \class ExecutionConstraintSet
     * Provide a class for tracking all the associated 
     * constraints for a given task variant.
     */
    class ExecutionConstraintSet {
    public:
      // Make sure these methods return ourself so we 
      // can easily chain together adding constraints
      ExecutionConstraintSet& 
        add_constraint(const ISAConstraint &constraint);
      ExecutionConstraintSet&
        add_constraint(const ProcessorConstraint &constraint);
      ExecutionConstraintSet&
        add_constraint(const ResourceConstraint &constraint);
      ExecutionConstraintSet&
        add_constraint(const LaunchConstraint &constraint);
      ExecutionConstraintSet&
        add_constraint(const ColocationConstraint &constraint);
    public:
      void swap(ExecutionConstraintSet &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      ISAConstraint                              isa_constraint;
      ProcessorConstraint                  processor_constraint;
      std::vector<ResourceConstraint>      resource_constraints;
      std::vector<LaunchConstraint>          launch_constraints;
      std::vector<ColocationConstraint>  colocation_constraints;
    };

    /**
     * \class LayoutConstraint
     * This is a base class for all layout constraints and provides
     * implementations for casting to each of the derived types
     */
    class LayoutConstraint {
    public:
      virtual ~LayoutConstraint(void) { }
    public:
      virtual LayoutConstraintKind get_constraint_kind(void) const = 0;
    public:
      virtual SpecializedConstraint* as_specialized_constraint(void) = 0;
      virtual MemoryConstraint* as_memory_constraint(void) = 0;
      virtual FieldConstraint* as_field_constraint(void) = 0;
      virtual OrderingConstraint* as_ordering_constraint(void) = 0;
      virtual SplittingConstraint* as_splitting_constraint(void) = 0;
      virtual DimensionConstraint* as_dimension_constraint(void) = 0;
      virtual AlignmentConstraint* as_alignment_constraint(void) = 0;
      virtual OffsetConstraint* as_offset_constraint(void) = 0;
      virtual PointerConstraint* as_pointer_constraint(void) = 0;
    public:
      virtual const SpecializedConstraint* 
        as_specialized_constraint(void) const = 0;
      virtual const MemoryConstraint* 
        as_memory_constraint(void) const = 0;
      virtual const FieldConstraint* 
        as_field_constraint(void) const = 0;
      virtual const OrderingConstraint* 
        as_ordering_constraint(void) const = 0;
      virtual const SplittingConstraint* 
        as_splitting_constraint(void) const = 0;
      virtual const DimensionConstraint* 
        as_dimension_constraint(void) const = 0;
      virtual const AlignmentConstraint* 
        as_alignment_constraint(void) const = 0;
      virtual const OffsetConstraint* 
        as_offset_constraint(void) const = 0;
      virtual const PointerConstraint* 
        as_pointer_constraint(void) const = 0;
    };

    /**
     * A helper class for definiting the virtual methods for all kinds
     * of layout constraint classes
     */
    template<typename T>
    class LayoutConstraintBase : public LayoutConstraint {
    public:
      virtual LayoutConstraintKind get_constraint_kind(void) const
        { return T::constraint_kind; }
    public:
      virtual SpecializedConstraint* as_specialized_constraint(void)
      { return (T::constraint_kind == SPECIALIZED_CONSTRAINT) ? 
                reinterpret_cast<SpecializedConstraint*>(this) : NULL; }
      virtual MemoryConstraint* as_memory_constraint(void)
      { return (T::constraint_kind == MEMORY_CONSTRAINT) ?
                reinterpret_cast<MemoryConstraint*>(this) : NULL; }
      virtual FieldConstraint* as_field_constraint(void)
      { return (T::constraint_kind == FIELD_CONSTRAINT) ?
                reinterpret_cast<FieldConstraint*>(this) : NULL; }
      virtual OrderingConstraint* as_ordering_constraint(void)
      { return (T::constraint_kind == ORDERING_CONSTRAINT) ?
                reinterpret_cast<OrderingConstraint*>(this) : NULL; }
      virtual SplittingConstraint* as_splitting_constraint(void)
      { return (T::constraint_kind == SPLITTING_CONSTRAINT) ?
                reinterpret_cast<SplittingConstraint*>(this) : NULL; }
      virtual DimensionConstraint* as_dimension_constraint(void)
      { return (T::constraint_kind == DIMENSION_CONSTRAINT) ?
                reinterpret_cast<DimensionConstraint*>(this) : NULL; }
      virtual AlignmentConstraint* as_alignment_constraint(void)
      { return (T::constraint_kind == ALIGNMENT_CONSTRAINT) ?
                reinterpret_cast<AlignmentConstraint*>(this) : NULL; }
      virtual OffsetConstraint* as_offset_constraint(void)
      { return (T::constraint_kind == OFFSET_CONSTRAINT) ?
                reinterpret_cast<OffsetConstraint*>(this) : NULL; }
      virtual PointerConstraint* as_pointer_constraint(void)
      { return (T::constraint_kind == POINTER_CONSTRAINT) ?
                reinterpret_cast<PointerConstraint*>(this) : NULL; }
    public:
      virtual const SpecializedConstraint* as_specialized_constraint(void) const
      { return (T::constraint_kind == SPECIALIZED_CONSTRAINT) ? 
                reinterpret_cast<const SpecializedConstraint*>(this) : NULL; }
      virtual const MemoryConstraint* as_memory_constraint(void) const
      { return (T::constraint_kind == MEMORY_CONSTRAINT) ?
                reinterpret_cast<const MemoryConstraint*>(this) : NULL; }
      virtual const FieldConstraint* as_field_constraint(void) const
      { return (T::constraint_kind == FIELD_CONSTRAINT) ?
                reinterpret_cast<const FieldConstraint*>(this) : NULL; }
      virtual const OrderingConstraint* as_ordering_constraint(void) const
      { return (T::constraint_kind == ORDERING_CONSTRAINT) ?
                reinterpret_cast<const OrderingConstraint*>(this) : NULL; }
      virtual const SplittingConstraint* as_splitting_constraint(void) const
      { return (T::constraint_kind == SPLITTING_CONSTRAINT) ?
                reinterpret_cast<const SplittingConstraint*>(this) : NULL; }
      virtual const DimensionConstraint* as_dimension_constraint(void) const
      { return (T::constraint_kind == DIMENSION_CONSTRAINT) ?
                reinterpret_cast<const DimensionConstraint*>(this) : NULL; }
      virtual const AlignmentConstraint* as_alignment_constraint(void) const
      { return (T::constraint_kind == ALIGNMENT_CONSTRAINT) ?
                reinterpret_cast<const AlignmentConstraint*>(this) : NULL; }
      virtual const OffsetConstraint* as_offset_constraint(void) const
      { return (T::constraint_kind == OFFSET_CONSTRAINT) ?
                reinterpret_cast<const OffsetConstraint*>(this) : NULL; }
      virtual const PointerConstraint* as_pointer_constraint(void) const
      { return (T::constraint_kind == POINTER_CONSTRAINT) ?
                reinterpret_cast<const PointerConstraint*>(this) : NULL; }
    };

    /**
     * \class SpecializedConstraint
     * This is a pretty basic constraint for physical instances.
     * Normal is a standard physical instance, while specialized
     * values are for indiciating the need for a custom kind of
     * physical instance like a reduction-list or a 
     * reduction fold instance. You can also use redop of 0 with 
     * either of the reduction specializations to indicate that
     * you expect a reduction instance, but it does not matter
     * which reduction instance is required. We can provide other 
     * kinds of specializations here in the future. Note the default
     * constructor will fall back to the normal case so this
     * kind of constraint won't need to be set in the default case.
     */
    class SpecializedConstraint : 
      public LayoutConstraintBase<SpecializedConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            SPECIALIZED_CONSTRAINT;
    public:
      SpecializedConstraint(SpecializedKind kind = NORMAL_SPECIALIZE,
                            ReductionOpID redop = 0, 
                            bool no_access = false,
                            bool exact = false);
    public:
      bool entails(const SpecializedConstraint &other) const;
      bool conflicts(const SpecializedConstraint &other) const;
    public:
      inline SpecializedKind get_kind(void) const { return kind; }
      inline ReductionOpID get_reduction_op(void) const { return redop; }
    public:
      void swap(SpecializedConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      bool is_affine(void) const;
      bool is_compact(void) const;
      bool is_varfield(void) const;
      bool is_virtual(void) const;
      bool is_reduction(void) const;
      bool is_file(void) const;
      inline bool is_no_access(void) const { return no_access; }
      inline bool is_exact(void) const { return exact; }
      // For backwards compatibility
      bool is_normal(void) const;
    public:
      SpecializedKind kind;
      ReductionOpID  redop;
      bool       no_access;
      bool           exact;
    };

    /**
     * \class MemoryConstraint
     * A placement constraint is used to specify where in 
     * memory to create physical instances. This is just an 
     * ordering of memories in which the runtime should attempt
     * to create a physical instance.
     */
    class MemoryConstraint : public LayoutConstraintBase<MemoryConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            MEMORY_CONSTRAINT;
    public:
      MemoryConstraint(void);
      MemoryConstraint(Memory::Kind kind);
    public:
      inline bool is_valid(void) const { return has_kind; }
      inline Memory::Kind get_kind(void) const { return kind; }
    public:
      bool entails(const MemoryConstraint &other) const;
      bool conflicts(const MemoryConstraint &other) const;
    public:
      void swap(MemoryConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      Memory::Kind kind;
      bool has_kind;
    };

    /**
     * \class FieldConstraint
     * A field constraint controls the ordering of fields in the
     * layout. Multiple field constraints can be specified 
     * an instance layout so that partial orders on fields can
     * be established. Users can also say with the contiguous
     * flag whether they want the fields to be contiguous in
     * the physical instance layout.
     */
    class FieldConstraint : public LayoutConstraintBase<FieldConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            FIELD_CONSTRAINT;
    public:
      FieldConstraint(bool contiguous = false, bool inorder = false);
      FieldConstraint(const std::vector<FieldID> &field_set,
                      bool contiguous, bool inorder = true);
      FieldConstraint(const std::set<FieldID> &field_set,
                      bool contiguous, bool inorder = true);
    public:
      inline bool is_contiguous(void) const { return contiguous; }
      inline bool is_inorder(void) const { return inorder; }
      inline const std::vector<FieldID>& get_field_set(void) const 
        { return field_set; }
    public:
      bool entails(const FieldConstraint &other) const;
      bool conflicts(const FieldConstraint &other) const;
    public:
      void swap(FieldConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::vector<FieldID> field_set;
      bool contiguous;
      bool inorder;
    };

    /**
     * \class OrderingConstraint
     * An ordering constraint specifies an ordering on 
     * index space dimensions as well as the special 'field'
     * dimension. The earliest dimensions are assumed to
     * be the most-rapidly changing in memory, therefore:
     *
     * Array-of-structs (AOS): DIM_F is listed first
     * Struct-of-arrays (SOA): DIM_F is listed last
     *
     * Again users can specify multiple of these constraints 
     * to describe a partial order, but the common case will
     * almost certainly be to specify a total order. Users
     * can also specify whether or not the related constraints
     * are contiguous with each other.
     *
     * It is important to note that users cannot mix full
     * dimensions with split dimensions. For example, either
     * 'DIM_X' can be used or 'INNER_DIM_X' and 'OUTER_DIM_X'
     * but never both at the same time. The 'INNER' and 
     * 'OUTER' dims may only be specified in a dimension
     * constraint if there is an associated split constraint
     * saying how to split the logical dimension.
     */
    class OrderingConstraint : public LayoutConstraintBase<OrderingConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            ORDERING_CONSTRAINT;
    public:
      OrderingConstraint(bool contiguous = false);
      OrderingConstraint(const std::vector<DimensionKind> &ordering,
                         bool contiguous);
    public:
      bool entails(const OrderingConstraint &other, unsigned total_dims) const;
      bool conflicts(const OrderingConstraint &other,unsigned total_dims) const;
    public:
      void swap(OrderingConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      static bool is_skip_dimension(DimensionKind dim, unsigned total_dims);
    public:
      std::vector<DimensionKind> ordering;
      bool contiguous;
    };

    /**
     * \class SplittingConstraint
     * Specify how to split a normal index space dimension into
     * an inner and outer dimension. The split can be specified
     * in one of two ways: either by saying how many chunks to
     * break the dimension into, or by specifying a splitting
     * factor that will create as many chunks as necessary.
     * These two constructors provide both top-down and bottom-up
     * ways of saying how to break a dimension apart.
     */
    class SplittingConstraint : 
      public LayoutConstraintBase<SplittingConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            SPLITTING_CONSTRAINT;
    public:
      SplittingConstraint(void);
      SplittingConstraint(DimensionKind dim); // chunks
      SplittingConstraint(DimensionKind dim, size_t value);
    public:
      bool entails(const SplittingConstraint &other) const;
      bool conflicts(const SplittingConstraint &other) const;
    public:
      void swap(SplittingConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      DimensionKind kind;
      size_t value;
      bool chunks;
    };

    /**
     * \class DimensionConstraint
     * Dimension constraints specify the minimum or maximum 
     * necessary size of a given dimension.
     */
    class DimensionConstraint : 
      public LayoutConstraintBase<DimensionConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                          DIMENSION_CONSTRAINT;
    public:
      DimensionConstraint(void);
      DimensionConstraint(DimensionKind dim, EqualityKind eq, size_t value);
    public:
      bool entails(const DimensionConstraint &other) const;
      bool conflicts(const DimensionConstraint &other) const;
    public:
      void swap(DimensionConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      DimensionKind kind;
      EqualityKind eqk;
      size_t value;
    };

    /**
     * \class AlignmentConstraint
     * Specify the alignment constraint for a field. Users can
     * set lower or upper bounds or equality for a the 
     * byte-alignment of a given field.
     */
    class AlignmentConstraint : 
      public LayoutConstraintBase<AlignmentConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            ALIGNMENT_CONSTRAINT;
    public:
      AlignmentConstraint(void);
      AlignmentConstraint(FieldID fid, EqualityKind kind, 
                          size_t byte_boundary);
    public:
      bool entails(const AlignmentConstraint &other) const;
      bool conflicts(const AlignmentConstraint &other) const;
    public:
      void swap(AlignmentConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      FieldID fid;
      EqualityKind eqk;
      size_t alignment;
    };

    /**
     * \class OffsetConstraint
     * Specify an offset constraint for a given field. In
     * the case of this constraint equality is implied.
     */
    class OffsetConstraint : public LayoutConstraintBase<OffsetConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            OFFSET_CONSTRAINT;
    public:
      OffsetConstraint(void);
      OffsetConstraint(FieldID fid, size_t offset);
    public:
      bool entails(const OffsetConstraint &other) const;
      bool conflicts(const OffsetConstraint &other) const;
    public:
      void swap(OffsetConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      FieldID fid;
      off_t offset;
    };

    /**
     * \class PointerConstraint
     * Specify the assumed pointer for a given field in
     * the physical instance.
     */
    class PointerConstraint : public LayoutConstraintBase<PointerConstraint> {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            POINTER_CONSTRAINT;
    public:
      PointerConstraint(void);
      PointerConstraint(Memory memory, uintptr_t ptr);
    public:
      bool entails(const PointerConstraint &other) const;
      bool conflicts(const PointerConstraint &other) const;
    public:
      void swap(PointerConstraint &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      bool is_valid;
      Memory memory;
      uintptr_t ptr;
    };

    /**
     * \class LayoutConstraintSet
     * Provide a class for tracking all the associated 
     * layout constraints for a given region requirement
     * of a task.
     */
    class LayoutConstraintSet {
    public: 
      LayoutConstraintSet&
        add_constraint(const SpecializedConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const FieldConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const MemoryConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const OrderingConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const SplittingConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const DimensionConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const AlignmentConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const OffsetConstraint &constraint);
      LayoutConstraintSet&
        add_constraint(const PointerConstraint &constraint);
    public:
      // failed_constraint will be the one from 'other' that wasn't entailed
      bool entails(const LayoutConstraintSet &other, 
                   unsigned total_dims = 0,
                   const LayoutConstraint **failed_constraint = NULL) const;
      // conflict_constraint will be the one from 'this' that conficted
      // with a constraint from 'other'
      bool conflicts(const LayoutConstraintSet &other,
                     unsigned total_dims = 0,
                     const LayoutConstraint **conflict_constraint = NULL) const;
    public:
      void swap(LayoutConstraintSet &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      SpecializedConstraint            specialized_constraint;
      FieldConstraint                  field_constraint;
      MemoryConstraint                 memory_constraint;
      PointerConstraint                pointer_constraint;
      OrderingConstraint               ordering_constraint;
      std::vector<SplittingConstraint> splitting_constraints;
      std::vector<DimensionConstraint> dimension_constraints;
      std::vector<AlignmentConstraint> alignment_constraints; 
      std::vector<OffsetConstraint>    offset_constraints;
    };

    /**
     * \class TaskLayoutConstraintSet
     * Provide a class to describe the layout descriptions for
     * all the regions in a task. Since a region requirement
     * can be satisfied by more than one instance, we allow
     * multiple layout descriptions to be specified for the
     * same region requirement.  The desriptions for a region
     * requirement should not describe any of the same fields.
     */
    class TaskLayoutConstraintSet {
    public:
      TaskLayoutConstraintSet&
        add_layout_constraint(unsigned idx, LayoutConstraintID desc);
    public:
      void swap(TaskLayoutConstraintSet &rhs);
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::multimap<unsigned,LayoutConstraintID> layouts;
    };

}; // namespace Legion

#endif // __LEGION_CONSTRAINT_H__

