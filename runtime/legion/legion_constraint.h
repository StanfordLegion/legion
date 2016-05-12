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

#ifndef __LEGION_CONSTRAINT_H__
#define __LEGION_CONSTRAINT_H__

/**
 * \file legion_constraint.h
 */

#include "legion_types.h"

namespace Legion {

    // These are the constraint kinds for describing execution 
    enum ExecutionConstraintKind {
      ISA_CONSTRAINT, // instruction set architecture
      PROCESSOR_CONSTRAINT, // processor kind constraint
      RESOURCE_CONSTRAINT, // physical resources
      LAUNCH_CONSTRAINT, // launch configuration
      COLOCATION_CONSTRAINT, // region requirements in same instance
    };

    // These are the constraint kinds for describing data layout 
    enum LayoutConstraintKind {
      SPECIALIZED_CONSTRAINT, // normal or speicalized (e.g. reduction-fold)
      MEMORY_CONSTRAINT, // constraint on the kind of memory
      FIELD_CONSTRAINT, // ordering of fields
      ORDERING_CONSTRAINT, // ordering of dimensions
      SPLITTING_CONSTRAINT, // splitting of dimensions 
      DIMENSION_CONSTRAINT, // dimension size constraint
      ALIGNMENT_CONSTRAINT, // alignment of a field
      OFFSET_CONSTRAINT, // offset of a field
      POINTER_CONSTRAINT, // pointer of a field
    };

    enum EqualityKind {
      LT_EK, // <
      LE_EK, // <=
      GT_EK, // >
      GE_EK, // >=
      EQ_EK, // ==
      NE_EK, // !=
    };

    enum DimensionKind {
      DIM_X, // first logical index space dimension
      DIM_Y, // second logical index space dimension
      DIM_Z, // ...
      DIM_F, // field dimension
      INNER_DIM_X, // inner dimension for tiling X
      OUTER_DIM_X, // outer dimension for tiling X
      INNER_DIM_Y, // ...
      OUTER_DIM_Y,
      INNER_DIM_Z,
      OUTER_DIM_Z,
    };

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
      // Make all flags 1-hot encoding so we can logically-or them together
      enum ISAKind {
        // Top-level ISA Kinds
        X86_ISA   = 0x00000001,
        ARM_ISA   = 0x00000002,
        POW_ISA   = 0x00000004, // Power PC
        PTX_ISA   = 0x00000008, // auto-launch by runtime
        CUDA_ISA  = 0x00000010, // run on CPU thread bound to CUDA context
        LUA_ISA   = 0x00000020, // run on Lua processor
        TERRA_ISA = 0x00000040, // JIT to target processor kind
        LLVM_ISA  = 0x00000080, // JIT to target processor kind
        GL_ISA    = 0x00000100, // run on CPU thread with OpenGL context
        // x86 Vector Instructions
        SSE_ISA   = 0x00000200,
        SSE2_ISA  = 0x00000400,
        SSE3_ISA  = 0x00000800,
        SSE4_ISA  = 0x00001000,
        AVX_ISA   = 0x00002000,
        AVX2_ISA  = 0x00004000,
        FMA_ISA   = 0x00008000,
        MIC_ISA   = 0x00010000,
        // GPU variants
        SM_10_ISA = 0x00020000,
        SM_20_ISA = 0x00040000,
        SM_30_ISA = 0x00080000,
        SM_35_ISA = 0x00100000,
        // ARM Vector Instructions
        NEON_ISA  = 0x00200000,
      };
    public:
      ISAConstraint(uint64_t prop = 0);
    public:
      bool entaisl(const ISAConstraint &other) const;
      bool conflicts(const ISAConstraint &other) const;
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      uint64_t isa_prop;
    };

    /**
     * \class ProcessorConstraint
     * Processor constraints are used to declare that a task variant
     * should only be able to executed on processors of a certain
     * kind. This is necessary for example, to distinguish I/O tasks
     * which can run on all x86 cores by their ISA constraints, but
     * users want to restrict their execution to just I/O processors.
     */
    class ProcessorConstraint {
    public:
      static const ExecutionConstraintKind constraint_kind = 
                                            PROCESSOR_CONSTRAINT;
    public:
      ProcessorConstraint(void);
      ProcessorConstraint(Processor::Kind kind);
    public:
      inline bool is_valid(void) const { return valid; }
      inline Processor::Kind get_kind(void) const { return kind; }
    public:
      bool entails(const ProcessorConstraint &other) const;
      bool conflicts(const ProcessorConstraint &other) const;
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      Processor::Kind kind;
      bool valid;
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
      enum ResourceKind {
        L1_CACHE_SIZE,
        L2_CACHE_SIZE,
        L3_CACHE_SIZE,
        L1_CACHE_ASSOCIATIVITY,
        L2_CACHE_ASSOCIATIVITY,
        L3_CACHE_ASSOCIATIVITY,
        REGISTER_FILE_SIZE,
        SHARED_MEMORY_SIZE,
        TEXTURE_CACHE_SIZE,
        CONSTANT_CACHE_SIZE,
        NAMED_BARRIERS,
        SM_COUNT, // total SMs on the device
        MAX_OCCUPANCY, // max warps per SM
      };
    public:
      ResourceConstraint(void);
      ResourceConstraint(ResourceKind resource_kind, 
                         EqualityKind eq_kind, size_t value);
    public:
      bool entails(const ResourceConstraint &other) const;
      bool conflicts(const ResourceConstraint &other) const;
    public:
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
      enum LaunchKind {
        CTA_SHAPE,
        GRID_SHAPE,
        DYNAMIC_SHARED_MEMORY,
        REGISTERS_PER_THREAD,
        CTAS_PER_SM,
        NAMED_BARRIERS_PER_CTA,
      };
    public:
      LaunchConstraint(void);
      LaunchConstraint(LaunchKind kind, size_t value);
      LaunchConstraint(LaunchKind kind, const size_t *value, int dims);
    public:
      bool entails(const LaunchConstraint &other) const;
      bool conflicts(const LaunchConstraint &other) const;
    public:
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
     * \class SpecializedConstraint
     * This is a pretty basic constraint for physical instances.
     * Normal is a standard physical instance, while specialized
     * values are for indiciating the need for a custom kind of
     * physical instance like a reduction-list or a 
     * reduction fold instance. We can provide other kinds of 
     * specializations here in the future. Note the default
     * constructor will fall back to the normal case so this
     * kind of constraint won't need to be set in the default case.
     */
    class SpecializedConstraint {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            SPECIALIZED_CONSTRAINT;
    public:
      enum SpecializedKind {
        NORMAL_SPECIALIZE,
        REDUCTION_FOLD_SPECIALIZE,
        REDUCTION_LIST_SPECIALIZE,
        VIRTUAL_SPECIALIZE,
      };
    public:
      SpecializedConstraint(SpecializedKind kind = NORMAL_SPECIALIZE,
                            ReductionOpID redop = 0);
    public:
      bool entails(const SpecializedConstraint &other) const;
      bool conflicts(const SpecializedConstraint &other) const;
    public:
      inline SpecializedKind get_kind(void) const { return kind; }
      inline ReductionOpID get_reduction_op(void) const { return redop; }
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      SpecializedKind kind;
      ReductionOpID  redop;
    };

    /**
     * \class MemoryConstraint
     * A placement constraint is used to specify where in 
     * memory to create physical instances. This is just an 
     * ordering of memories in which the runtime should attempt
     * to create a physical instance.
     */
    class MemoryConstraint {
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
    class FieldConstraint {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            FIELD_CONSTRAINT;
    public:
      FieldConstraint(void);
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
    class OrderingConstraint {
    public:
      static const LayoutConstraintKind constraint_kind = 
                                            ORDERING_CONSTRAINT;
    public:
      OrderingConstraint(void);
      OrderingConstraint(const std::vector<DimensionKind> &ordering,
                         bool contiguous);
    public:
      bool entails(const OrderingConstraint &other) const;
      bool conflicts(const OrderingConstraint &other) const;
    public:
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
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
    class SplittingConstraint {
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
    class DimensionConstraint {
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
    class AlignmentConstraint {
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
    class OffsetConstraint {
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
    class PointerConstraint  {
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
      bool entails(const LayoutConstraintSet &other) const;
      bool conflicts(const LayoutConstraintSet &other) const;
    public:
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
      void serialize(Serializer &rez) const;
      void deserialize(Deserializer &derez);
    public:
      std::multimap<unsigned,LayoutConstraintID> layouts;
    };

}; // namespace Legion

#endif // __LEGION_CONSTRAINT_H__

