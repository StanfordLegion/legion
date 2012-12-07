/* Copyright 2012 Stanford University
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


#ifndef __LEGION_TYPES_H__
#define __LEGION_TYPES_H__

#include "lowlevel.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <stdint.h>

#include "limits.h"

#include <map>
#include <set>
#include <list>
#include <vector>

#define AUTO_GENERATE_ID   UINT_MAX

#define MAX_FIELDS         128
// The folowing fields are used in the FieldMask instantiation of BitMask
// If you change one you probably have to change the others too
#define FIELD_TYPE          uint64_t 
#define FIELD_SHIFT         6
#define FIELD_MASK          0x3F
#define FIELD_ALL_ONES      0xFFFFFFFFFFFFFFFF


namespace LegionRuntime {
  namespace HighLevel {
    
    enum LegionErrorType {
      NO_ERROR = 0,
      ERROR_RESERVED_REDOP_ID = 1,
      ERROR_DUPLICATE_REDOP_ID = 2,
      ERROR_RESERVED_TYPE_HANDLE = 3,
      ERROR_DUPLICATE_TYPE_HANDLE = 4,
      ERROR_DUPLICATE_FIELD_ID = 5,
      ERROR_PARENT_TYPE_HANDLE_NONEXISTENT = 6,
      ERROR_MISSING_PARENT_FIELD_ID = 7,
      ERROR_RESERVED_PROJECTION_ID = 8,
      ERROR_DUPLICATE_PROJECTION_ID = 9,
      ERROR_UNREGISTERED_VARIANT = 10,
      ERROR_USE_REDUCTION_REGION_REQ = 11,
      ERROR_INVALID_ACCESSOR_REQUESTED = 12,
      ERROR_PHYSICAL_REGION_UNMAPPED = 13,
      ERROR_RESERVED_TASK_ID = 14,
      ERROR_INVALID_ARG_MAP_DESTRUCTION = 15,
      ERROR_RESERVED_MAPPING_ID = 16,
      ERROR_BAD_INDEX_PRIVILEGES = 17,
      ERROR_BAD_FIELD_PRIVILEGES = 18,
      ERROR_BAD_REGION_PRIVILEGES = 19,
      ERROR_BAD_PARTITION_PRIVILEGES = 20,
      ERROR_BAD_PARENT_INDEX = 21,
      ERROR_BAD_INDEX_PATH = 22,
      ERROR_BAD_PARENT_REGION = 23,
      ERROR_BAD_REGION_PATH = 24,
      ERROR_BAD_PARTITION_PATH = 25,
      ERROR_BAD_FIELD = 26,
      ERROR_BAD_REGION_TYPE = 27,
      ERROR_INVALID_TYPE_HANDLE = 28,
      ERROR_LEAF_TASK_VIOLATION = 29,
      ERROR_INVALID_REDOP_ID = 30,
      ERROR_REDUCTION_INITIAL_VALUE_MISMATCH = 31,
      ERROR_INVALID_UNMAP_OP = 32,
      ERROR_INVALID_DUPLICATE_MAPPING = 33,
      ERROR_INVALID_REGION_ARGUMENT_INDEX = 34,
      ERROR_INVALID_MAPPING_ACCESS = 35,
      ERROR_STALE_INLINE_MAPPING_ACCESS = 36,
      ERROR_INVALID_INDEX_SPACE_PARENT = 37,
      ERROR_INVALID_INDEX_PART_PARENT = 38,
      ERROR_INVALID_INDEX_SPACE_COLOR = 39,
      ERROR_INVALID_INDEX_PART_COLOR = 40,
      ERROR_INVALID_INDEX_SPACE_HANDLE = 41,
      ERROR_INVALID_INDEX_PART_HANDLE = 42,
      ERROR_FIELD_SPACE_FIELD_MISMATCH = 43,
      ERROR_INVALID_INSTANCE_FIELD = 44,
      ERROR_DUPLICATE_INSTANCE_FIELD = 45,
      ERROR_TYPE_INST_MISMATCH = 46,
      ERROR_TYPE_INST_MISSIZE = 47,
      ERROR_INVALID_INDEX_SPACE_ENTRY = 48,
      ERROR_INVALID_INDEX_PART_ENTRY = 49,
      ERROR_INVALID_FIELD_SPACE_ENTRY = 50,
      ERROR_INVALID_REGION_ENTRY = 51,
      ERROR_INVALID_PARTITION_ENTRY = 52,
      ERROR_ALIASED_INTRA_TASK_REGIONS = 53,
      ERROR_MAX_FIELD_OVERFLOW = 54,
      ERROR_MISSING_TASK_COLLECTION = 55,
      ERROR_INVALID_IDENTITY_PROJECTION_USE = 56,
      ERROR_INVALID_PROJECTION_ID = 57,
      ERROR_NON_DISJOINT_PARTITION = 58,
      ERROR_BAD_PROJECTION_USE = 59,
      ERROR_INDEPENDENT_SLICES_VIOLATION = 60,
      ERROR_INVALID_REGION_HANDLE = 61,
      ERROR_INVALID_PARTITION_HANDLE = 62,
      ERROR_VIRTUAL_MAP_IN_LEAF_TASK = 63,
      ERROR_LEAF_MISMATCH = 64,
      ERROR_INVALID_PROCESSOR_SELECTION = 65,
      ERROR_INVALID_VARIANT_SELECTION = 66,
      ERROR_INVALID_MAPPER_OUTPUT = 67,
      ERROR_UNINITIALIZED_REDUCTION = 68,
    };

    // enum and namepsaces don't really get along well
    enum AccessorType {
      AccessorGeneric            = 0x00000001, //LowLevel::AccessorGeneric,
      AccessorArray              = 0x00000002, //LowLevel::AccessorArray,
      AccessorArrayReductionFold = 0x00000004, //LowLevel::AccessorArrayReductionFold,
      AccessorGPU                = 0x00000008, //LowLevel::AccessorGPU,
      AccessorGPUReductionFold   = 0x00000010, //LowLevel::AccessorGPUReductionFold,
      AccessorReductionList      = 0x00000020, //LowLevel::AccessorReductionList,
    };

    enum PrivilegeMode {
      NO_ACCESS  = 0x00000000,
      READ_ONLY  = 0x00000001,
      READ_WRITE = 0x00000111,
      WRITE_ONLY = 0x00000010,
      REDUCE     = 0x00000100,
    };

    enum AllocateMode {
      NO_MEMORY       = 0x00000000,
      ALLOCABLE       = 0x00000001,
      FREEABLE        = 0x00000010,
      MUTABLE         = 0x00000011,
      REGION_CREATION = 0x00000100,
      REGION_DELETION = 0x00001000,
      ALL_MEMORY      = 0x00001111,
    };

    enum CoherenceProperty {
      EXCLUSIVE    = 0,
      ATOMIC       = 1,
      SIMULTANEOUS = 2,
      RELAXED      = 3,
    };

    enum HandleType {
      SINGULAR,
      PROJECTION,
    };

    enum DependenceType {
      NO_DEPENDENCE = 0,
      TRUE_DEPENDENCE = 1,
      ANTI_DEPENDENCE = 2, // Write-After-Read or Write-After-Write with Write-Only privilege
      ATOMIC_DEPENDENCE = 3,
      SIMULTANEOUS_DEPENDENCE = 4,
    };

    // Runtime task numbering 
    enum {
      INIT_FUNC_ID       = LowLevel::Processor::TASK_ID_PROCESSOR_INIT,
      SHUTDOWN_FUNC_ID   = LowLevel::Processor::TASK_ID_PROCESSOR_SHUTDOWN,
      SCHEDULER_ID       = LowLevel::Processor::TASK_ID_PROCESSOR_IDLE,
      ENQUEUE_TASK_ID    = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+0),
      STEAL_TASK_ID      = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+1),
      CHILDREN_MAPPED_ID = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+2),
      FINISH_ID          = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+3),
      NOTIFY_START_ID    = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+4),
      NOTIFY_MAPPED_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+5),
      NOTIFY_FINISH_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+6),
      ADVERTISEMENT_ID   = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+7),
      TERMINATION_ID     = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+8),
      CUSTOM_PREDICATE_ID= (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+9),
      TASK_ID_AVAILABLE  = (LowLevel::Processor::TASK_ID_FIRST_AVAILABLE+10),
    };

    // Forward declarations for user level objects
    // legion.h
    class LogicalRegion;
    class LogicalPartition;
    class FieldSpace;
    class Task;
    class TaskVariantCollection;
    class Future;
    class FutureMap;
    class IndexSpaceRequirement;
    class FieldSpaceRequirement;
    class RegionRequirement;
    template<int NUM_PRIV, int NUM_INST> class SizedRegionRequirement;
    class TaskArgument;
    class ArgumentMap;
    class FutureMap;
    class PhysicalRegion;
    class IndexAllocator;
    class FieldAllocator;
    class IndexIterator;
    template<typename T> struct ColoredPoints; 
    class Structure;
    class HighLevelRuntime;
    class Mapper;

    // Forward declarations for runtime level objects
    // legion.h
    class Lockable;
    class Collectable;
    class Notifiable;
    class Waitable;
    class Mappable; // For things that need to have a mapper and lock selelcted 
    class PredicateImpl;
    class FutureImpl;
    class FutureMapImpl;
    class ArgumentMapImpl;
    class ArgumentMapStore;
    class PhysicalRegionImpl;
    class IndexAllocatorImpl;
    class AnyPoint;

    // Forward declarations for runtime level predicates
    // in legion.cc
    class PredicateAnd;
    class PredicateOr;
    class PredicateNot;
    class PredicateFuture;
    class PredicateCustom;

    // legion_ops.h
    class GeneralizedOperation;
    class TaskContext;
    class MappingOperation;
    class DeletionOperation;
    class SingleTask;
    class MultiTask;
    class IndexTask;
    class SliceTask;
    class PointTask;
    class IndividualTask;

    // region_tree.h
    class RegionTreeForest;
    class IndexSpaceNode;
    class IndexPartNode;
    class FieldSpaceNode;
    class RegionTreeNode;
    class RegionNode;
    class PartitionNode;
    class PhysicalManager; // base class for instance and reduction
    class PhysicalView; // base class for instance and reduciton
    class InstanceManager;
    class InstanceKey;
    class InstanceView;
    class InstanceRef;
    class ReductionManager;
    class ListReductionManager;
    class FoldReductionManager;
    class ReductionView;
    class RegionAnalyzer;
    class RegionMapper;

    struct EscapedUser;
    struct EscapedCopy;
    struct RegionUsage;
    struct GenericUser;
    struct LogicalUser;
    struct PhysicalUser;
    class TreeCloser;
    class LogicalCloser;
    class PhysicalCloser;
    class ReductionCloser;

    // legion_utilities.h
    class Serializer;
    class Deserializer;
    template<typename T> class Fraction;
    template<typename T, unsigned int MAX> class BitMask;

    // legion_logging.h
    class TreeStateLogger;

    typedef LowLevel::Machine Machine;
    typedef LowLevel::IndexSpace IndexSpace;
    typedef LowLevel::IndexSpaceAllocator IndexSpaceAllocator;
    typedef LowLevel::RegionInstance PhysicalInstance;
    typedef LowLevel::Memory Memory;
    typedef LowLevel::Processor Processor;
    typedef LowLevel::Event Event;
    typedef LowLevel::UserEvent UserEvent;
    typedef LowLevel::Lock Lock;
    typedef LowLevel::Barrier Barrier;
    typedef LowLevel::ReductionOpID ReductionOpID;
    typedef LowLevel::ReductionOpUntyped ReductionOp;
    typedef LowLevel::Machine::ProcessorMemoryAffinity ProcessorMemoryAffinity;
    typedef LowLevel::Machine::MemoryMemoryAffinity MemoryMemoryAffinity;
    typedef LowLevel::ElementMask::Enumerator Enumerator;
    typedef unsigned int Color;
    typedef unsigned int IndexPartition;
    typedef unsigned int FieldID;
    typedef unsigned int MapperID;
    typedef unsigned int UniqueID;
    typedef unsigned int ContextID;
    typedef unsigned int InstanceID;
    typedef unsigned int FieldSpaceID;
    typedef unsigned int GenerationID;
    typedef unsigned int TypeHandle;
    typedef unsigned int ProjectionID;
    typedef unsigned int RegionTreeID;
    typedef unsigned int UniqueManagerID;
    typedef unsigned long MappingTagID;
    typedef unsigned long VariantID;
    typedef Processor::TaskFuncID TaskID;
    typedef SingleTask* Context;
    typedef std::map<Color,ColoredPoints<ptr_t> > Coloring;
    typedef void (*RegistrationCallbackFnptr)(Machine *machine, HighLevelRuntime *rt, const std::set<Processor> &local_procs);
    typedef Color (*ProjectionFnptr)(const void*,size_t,unsigned);
    typedef bool (*PredicateFnptr)(const void*, size_t, const std::vector<Future> futures);
    typedef std::map<TypeHandle,Structure> TypeTable;
    typedef std::map<ProjectionID,ProjectionFnptr> ProjectionTable;
    typedef BitMask<FIELD_TYPE, MAX_FIELDS> FieldMask;
    typedef Fraction<unsigned long> InstFrac;

#define FRIEND_ALL_RUNTIME_CLASSES                \
    friend class HighLevelRuntime;                \
    friend class GeneralizedOperation;            \
    friend class MappingOperation;                \
    friend class DeletionOperation;               \
    friend class TaskContext;                     \
    friend class SingleTask;                      \
    friend class MultiTask;                       \
    friend class IndividualTask;                  \
    friend class PointTask;                       \
    friend class IndexTask;                       \
    friend class SliceTask;                       \
    friend class RegionRequirement;               \
    friend class RegionTreeForest;                \
    friend class IndexSpaceNode;                  \
    friend class IndexPartNode;                   \
    friend class FieldSpaceNode;                  \
    friend class RegionTreeNode;                  \
    friend class RegionNode;                      \
    friend class PartitionNode;                   \
    friend class PhysicalView;                    \
    friend class InstanceView;                    \
    friend class ReductionView;                   \
    friend class PhysicalManager;                 \
    friend class InstanceManager;                 \
    friend class ReductionManager;                \
    friend class ListReductionManager;            \
    friend class FoldReductionManager;            \
    friend class TreeStateLogger;

    // Timing events
    enum {
#ifdef PRECISE_HIGH_LEVEL_TIMING
      TIME_HIGH_LEVEL_CREATE_REGION = 100,
      TIME_HIGH_LEVEL_DESTROY_REGION = 101,
      TIME_HIGH_LEVEL_SMASH_REGION = 102
      TIME_HIGH_LEVEL_JOIN_REGION = 103
      TIME_HIGH_LEVEL_CREATE_PARTITION = 104,
      TIME_HIGH_LEVEL_DESTROY_PARTITION = 105,
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = 106,
      TIME_HIGH_LEVEL_STEAL_REQUEST = 107,
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = 108,
      TIME_HIGH_LEVEL_FINISH_TASK = 109,
      TIME_HIGH_LEVEL_NOTIFY_START = 110,
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = 111,
      TIME_HIGH_LEVEL_NOTIFY_FINISH = 112,
      TIME_HIGH_LEVEL_EXECUTE_TASK = 113,
      TIME_HIGH_LEVEL_SCHEDULER = 114,
      TIME_HIGH_LEVEL_ISSUE_STEAL = 115,
      TIME_HIGH_LEVEL_GET_SUBREGION = 116,
      TIME_HIGH_LEVEL_INLINE_MAP = 117,
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = 118,
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = 119,
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = 120,
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = 121,
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = 122,
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = 123,
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = 124,
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = 125,
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = 126,
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = 127,
      TIME_HIGH_LEVEL_ALLOCATE_FIELD = 128,
      TIME_HIGH_LEVEL_FREE_FIELD = 129,
#else
      TIME_HIGH_LEVEL_CREATE_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SMASH_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_JOIN_REGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_ENQUEUE_TASKS = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_STEAL_REQUEST = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CHILDREN_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_FINISH_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_START = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_MAPPED = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_NOTIFY_FINISH = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_EXECUTE_TASK = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_SCHEDULER = TIME_HIGH_LEVEL,
      TIME_HIGH_LEVEL_ISSUE_STEAL = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_INLINE_MAP = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_INDEX_SUBSPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_CREATE_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_DESTROY_FIELD_SPACE = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_PARTITION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_GET_LOGICAL_SUBREGION = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_ALLOCATE_FIELD = TIME_HIGH_LEVEL, 
      TIME_HIGH_LEVEL_FREE_FIELD = TIME_HIGH_LEVEL, 
#endif
    };


  }; // HighLevel namespace
}; // LegionRuntime namespace

#endif // __LEGION_TYPES_H__
