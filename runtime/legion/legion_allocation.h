/* Copyright 2017 Stanford University, NVIDIA Corporation
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


#ifndef __LEGION_ALLOCATION__
#define __LEGION_ALLOCATION__

#include <set>
#include <map>
#include <new>
#include <list>
#include <deque>
#include <vector>
#include <limits>
#include <cstddef>
#include <functional>
#include <stdlib.h>
#ifndef __MACH__
#include <malloc.h>
#endif
#include "legion_template_help.h" // StaticAssert
#if __cplusplus == 201103L
#include <utility>
#endif

namespace Legion {
  namespace Internal {

    enum AllocationType {
      ARGUMENT_MAP_ALLOC,
      ARGUMENT_MAP_STORE_ALLOC,
      STORE_ARGUMENT_ALLOC,
      MPI_HANDSHAKE_ALLOC,
      GRANT_ALLOC,
      FUTURE_ALLOC,
      FUTURE_MAP_ALLOC,
      PHYSICAL_REGION_ALLOC,
      STATIC_TRACE_ALLOC,
      DYNAMIC_TRACE_ALLOC,
      ALLOC_MANAGER_ALLOC,
      ALLOC_INTERNAL_ALLOC,
      TASK_ARGS_ALLOC,
      LOCAL_ARGS_ALLOC,
      REDUCTION_ALLOC,
      PREDICATE_ALLOC,
      FUTURE_RESULT_ALLOC,
      INSTANCE_MANAGER_ALLOC,
      LIST_MANAGER_ALLOC,
      FOLD_MANAGER_ALLOC,
      COMPOSITE_NODE_ALLOC,
      TREE_CLOSE_ALLOC,
      TREE_CLOSE_IMPL_ALLOC,
      MATERIALIZED_VIEW_ALLOC,
      REDUCTION_VIEW_ALLOC,
      COMPOSITE_VIEW_ALLOC,
      FILL_VIEW_ALLOC,
      INDIVIDUAL_TASK_ALLOC,
      POINT_TASK_ALLOC,
      INDEX_TASK_ALLOC,
      SLICE_TASK_ALLOC,
      TOP_TASK_ALLOC,
      REMOTE_TASK_ALLOC,
      INLINE_TASK_ALLOC,
      MAP_OP_ALLOC,
      COPY_OP_ALLOC,
      FENCE_OP_ALLOC,
      FRAME_OP_ALLOC,
      DELETION_OP_ALLOC,
      OPEN_OP_ALLOC,
      ADVANCE_OP_ALLOC,
      CLOSE_OP_ALLOC,
      DYNAMIC_COLLECTIVE_OP_ALLOC,
      FUTURE_PRED_OP_ALLOC,
      NOT_PRED_OP_ALLOC,
      AND_PRED_OP_ALLOC,
      OR_PRED_OP_ALLOC,
      ACQUIRE_OP_ALLOC,
      RELEASE_OP_ALLOC,
      TRACE_CAPTURE_OP_ALLOC,
      TRACE_COMPLETE_OP_ALLOC,
      MUST_EPOCH_OP_ALLOC,
      PENDING_PARTITION_OP_ALLOC,
      DEPENDENT_PARTITION_OP_ALLOC,
      FILL_OP_ALLOC,
      ATTACH_OP_ALLOC,
      DETACH_OP_ALLOC,
      MESSAGE_BUFFER_ALLOC,
      EXECUTING_CHILD_ALLOC,
      EXECUTED_CHILD_ALLOC,
      COMPLETE_CHILD_ALLOC,
      PHYSICAL_MANAGER_ALLOC,
      LOGICAL_VIEW_ALLOC,
      LOGICAL_FIELD_VERSIONS_ALLOC,
      LOGICAL_FIELD_STATE_ALLOC,
      CURR_LOGICAL_ALLOC,
      PREV_LOGICAL_ALLOC,
      VERSION_ID_ALLOC,
      LOGICAL_REC_ALLOC,
      CLOSE_LOGICAL_ALLOC,
      VALID_VIEW_ALLOC,
      VALID_REDUCTION_ALLOC,
      PENDING_UPDATES_ALLOC,
      LAYOUT_DESCRIPTION_ALLOC,
      PHYSICAL_USER_ALLOC,
      PHYSICAL_VERSION_ALLOC,
      MEMORY_INSTANCES_ALLOC,
      MEMORY_GARBAGE_ALLOC,
      PROCESSOR_GROUP_ALLOC,
      RUNTIME_DISTRIBUTED_ALLOC,
      RUNTIME_DIST_COLLECT_ALLOC,
      RUNTIME_GC_EPOCH_ALLOC,
      RUNTIME_FUTURE_ALLOC,
      RUNTIME_REMOTE_ALLOC,
      TASK_INLINE_REGION_ALLOC,
      TASK_TRACES_ALLOC,
      TASK_RESERVATION_ALLOC,
      TASK_BARRIER_ALLOC,
      TASK_LOCAL_FIELD_ALLOC,
      SEMANTIC_INFO_ALLOC,
      DIRECTORY_ALLOC,
      DENSE_INDEX_ALLOC,
      CURRENT_STATE_ALLOC,
      VERSION_MANAGER_ALLOC,
      PHYSICAL_STATE_ALLOC,
      VERSION_STATE_ALLOC,
      AGGREGATE_VERSION_ALLOC,
      TASK_IMPL_ALLOC,
      VARIANT_IMPL_ALLOC,
      LAYOUT_CONSTRAINTS_ALLOC,
      LAST_ALLOC, // must be last
    };

    /**
     * A helper class for determining alignment of types
     */
    template<typename T>
    class AlignmentTrait {
    public:
      struct AlignmentFinder {
        char a;
        T b;
      };
      enum { AlignmentOf = sizeof(AlignmentFinder) - sizeof(T) };
    };

    //--------------------------------------------------------------------------
    template<size_t SIZE, size_t ALIGNMENT, bool BYTES>
    inline void* legion_alloc_aligned(size_t cnt)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT((SIZE % ALIGNMENT) == 0);
      size_t alloc_size = cnt;
      if (!BYTES)
        alloc_size *= SIZE;
      void *result;
      if (ALIGNMENT > LEGION_MAX_ALIGNMENT)
      {
#ifdef DEBUG_LEGION
        assert((alloc_size % ALIGNMENT) == 0);
#endif
      // memalign is faster than posix_memalign so use it if we have it
#ifdef __MACH__
#ifndef NDEBUG
        int error = 
#endif
          posix_memalign(&result, ALIGNMENT, alloc_size);
        assert(error == 0);
#else
        result = memalign(ALIGNMENT, alloc_size);
#endif
      }
      else
        result = malloc(alloc_size);

      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool BYTES>
    inline void* legion_alloc_aligned(size_t cnt)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<sizeof(T),
              AlignmentTrait<T>::AlignmentOf,BYTES>(cnt);
    }

#ifdef TRACE_ALLOCATION
    // forward declaration of runtime
    class Runtime;

    // Implementations in runtime.cc
    struct LegionAllocation {
    public:
      static void trace_allocation(AllocationType a, size_t size, int elems=1);
      static void trace_free(AllocationType a, size_t size, int elems=1);
      static Runtime* find_runtime(void);
      static void trace_allocation(Runtime *&rt, AllocationType a, 
                                   size_t size, int elems=1);
      static void trace_free(Runtime *&rt, AllocationType a, 
                             size_t size, int elems=1);
    };

    // A Helper class for determining if we have an allocation type
    template<typename T>
    struct HasAllocType {
      typedef char no[1];
      typedef char yes[2];

      struct Fallback { int alloc_type; };
      struct Derived : T, Fallback { };

      template<typename U, U> struct Check;

      template<typename U>
      static no& test_for_alloc_type(
          Check<int (Fallback::*), &U::alloc_type> *);

      template<typename U>
      static yes& test_for_alloc_type(...);

      static const bool value = 
        (sizeof(test_for_alloc_type<Derived>(0)) == sizeof(yes));
    };

    template<typename T, bool HAS_ALLOC_TYPE>
    struct HandleAllocation {
      static inline void trace_allocation(void)
      {
        LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
      }
      static inline void trace_free(void)
      {
        LegionAllocation::trace_free(T::alloc_type, sizeof(T));
      }
    };

    template<typename T>
    struct HandleAllocation<T,false> {
      static inline void trace_allocation(void) { /*nothing*/ }
      static inline void trace_free(void) { /*nothing*/ }
    };
#endif

    // Helper methods for doing tracing of memory allocations
    //--------------------------------------------------------------------------
    inline void* legion_malloc(AllocationType a, size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(a, size);
#endif
      return malloc(size);
    }

    //--------------------------------------------------------------------------
    inline void* legion_realloc(AllocationType a, void *ptr, 
                                size_t old_size, size_t new_size)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      Runtime *rt = LegionAllocation::find_runtime(); 
      LegionAllocation::trace_free(rt, a, old_size);
      LegionAllocation::trace_allocation(rt, a, new_size);
#endif
      return realloc(ptr, new_size);
    }

    //--------------------------------------------------------------------------
    inline void legion_free(AllocationType a, void *ptr, size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_free(a, size);
#endif
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T* legion_new(void)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T();
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T* legion_new_in_place(void *buffer)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T();
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1>
    inline T* legion_new(const T1 &arg1)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2>
    inline T* legion_new(const T1 &arg1, const T2 &arg2)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3>
    inline T* legion_new(const T1 &arg1, const T2 &arg2, const T3 &arg3)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, 
                                  const T2 &arg2, const T3 &arg3)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, typename T4>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, typename T4>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, const T5 &arg5)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                 const T3 &arg3, const T4 &arg4, const T5 &arg5)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6, const T7 &arg7)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                 const T3 &arg3, const T4 &arg4, 
                                 const T5 &arg5, const T6 &arg6, const T7 &arg7)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6, 
                                   arg7, arg8);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6, 
                                   arg7, arg8);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, const T9 &arg9)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                 const T3 &arg3, const T4 &arg4, 
                                 const T5 &arg5, const T6 &arg6,
                                 const T7 &arg7, const T8 &arg8, const T9 &arg9)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, const T11 &arg11)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                const T3 &arg3, const T4 &arg4, 
                                const T5 &arg5, const T6 &arg6,
                                const T7 &arg7, const T8 &arg8, 
                                const T9 &arg9, const T10 &arg10, 
                                const T11 &arg11)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12,
                         const T13 &arg13)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12, 
                                   arg13);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12,
                                  const T13 &arg13)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12, 
                                   arg13);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12,
                         const T13 &arg13, const T14 &arg14)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12,
                                  const T13 &arg13, const T14 &arg14)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12,
                         const T13 &arg13, const T14 &arg14,
                         const T15 &arg15)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12,
                                  const T13 &arg13, const T14 &arg14,
                                  const T15 &arg15)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15,
             typename T16>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12,
                         const T13 &arg13, const T14 &arg14,
                         const T15 &arg15, const T16 &arg16)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15, arg16);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15,
             typename T16>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12,
                                  const T13 &arg13, const T14 &arg14,
                                  const T15 &arg15, const T16 &arg16)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15, arg16);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15,
             typename T16, typename T17>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, 
                         const T11 &arg11, const T12 &arg12,
                         const T13 &arg13, const T14 &arg14,
                         const T15 &arg15, const T16 &arg16,
                         const T17 &arg17)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      void *buffer = legion_alloc_aligned<T,false/*bytes*/>(1/*count*/);
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15, arg16, arg17);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11,
             typename T12, typename T13, typename T14, typename T15,
             typename T16, typename T17>
    inline T* legion_new_in_place(void *buffer, const T1 &arg1, const T2 &arg2,
                                  const T3 &arg3, const T4 &arg4, 
                                  const T5 &arg5, const T6 &arg6,
                                  const T7 &arg7, const T8 &arg8, 
                                  const T9 &arg9, const T10 &arg10, 
                                  const T11 &arg11, const T12 &arg12,
                                  const T13 &arg13, const T14 &arg14,
                                  const T15 &arg15, const T16 &arg16,
                                  const T17 &arg17)
    //--------------------------------------------------------------------------
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      T *result = ::new (buffer) T(arg1, arg2, arg3, arg4, arg5, arg6,
                                   arg7, arg8, arg9, arg10, arg11, arg12,
                                   arg13, arg14, arg15, arg16, arg17);
      return result;
    }

    template<typename T>
    inline void legion_delete(T *to_free)
    {
#ifdef TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
      to_free->~T();
      free(to_free);
    }

    /**
     * \class AlignedAllocator
     * A class for doing aligned allocation of memory for
     * STL data structures containing data types that require
     * larger alignments than the default malloc provides.
     */
    template<typename T>
    class AlignedAllocator {
    public:
      typedef size_t          size_type;
      typedef ptrdiff_t difference_type;
      typedef T*                pointer;
      typedef const T*    const_pointer;
      typedef T&              reference;
      typedef const T&  const_reference;
      typedef T              value_type;
    public:
      template<typename U>
      struct rebind {
        typedef AlignedAllocator<U> other;
      };
    public:
      inline explicit AlignedAllocator(void) { }
      inline ~AlignedAllocator(void) { }
      inline AlignedAllocator(const AlignedAllocator<T> &rhs) { }
      template<typename U>
      inline AlignedAllocator(const AlignedAllocator<U> &rhs) { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
      inline pointer allocate(size_type cnt,
                              typename std::allocator<void>::const_pointer = 0)
      {
        void *result = legion_alloc_aligned<T, false/*bytes*/>(cnt);
        return reinterpret_cast<pointer>(result);
      }
      inline void deallocate(pointer p, size_type size) {
        free(p);
      }
    public:
      inline size_type max_size(void) const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
      }
    public:
      inline void construct(pointer p, const T &t) { new(p) T(t); }
      inline void destroy(pointer p) { p->~T(); }
    public:
      inline bool operator==(AlignedAllocator const&) const { return true; }
      inline bool operator!=(AlignedAllocator const& a) const
                                           { return !operator==(a); }
    };

    /**
     * \class LegionAllocator
     * A custom Legion allocator for tracing memory usage in STL
     * data structures. When tracing is disabled, it defaults back
     * to using the standard malloc/free and new/delete operations.
     */
    template<typename T, AllocationType A, bool ALIGNED>
    class LegionAllocator {
    public:
      typedef size_t          size_type;
      typedef ptrdiff_t difference_type;
      typedef T*                pointer;
      typedef const T*    const_pointer;
      typedef T&              reference;
      typedef const T&  const_reference;
      typedef T              value_type;
    public:
      template<typename U>
      struct rebind {
        typedef LegionAllocator<U, A, ALIGNED> other;
      };
    public:
      inline explicit LegionAllocator(void) 
#ifdef TRACE_ALLOCATION
        : runtime(LegionAllocation::find_runtime()) 
#endif
      { }
      inline ~LegionAllocator(void) { }
      inline LegionAllocator(const LegionAllocator<T, A, ALIGNED> &rhs)
#ifdef TRACE_ALLOCATION
        : runtime(rhs.runtime) 
#endif
      { }
      template<typename U>
      inline LegionAllocator(const LegionAllocator<U, A, ALIGNED> &rhs) 
#ifdef TRACE_ALLOCATION
        : runtime(rhs.runtime) 
#endif  
      { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
      inline pointer allocate(size_type cnt,
                      typename std::allocator<void>::const_pointer = 0) {
#ifdef TRACE_ALLOCATION
        LegionAllocation::trace_allocation(runtime, A, sizeof(T), cnt);
#endif
        void *result;
        if (ALIGNED)
          result = legion_alloc_aligned<T, false/*bytes*/>(cnt);
        else
          result = malloc(cnt * sizeof(T));
        return reinterpret_cast<pointer>(result);
      }
      inline void deallocate(pointer p, size_type size) {
#ifdef TRACE_ALLOCATION
        LegionAllocation::trace_free(runtime, A, sizeof(T), size);
#endif
        free(p);
      }
    public:
      inline size_type max_size(void) const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
      }
    public:
#if __cplusplus == 201103L
      template<class U, class... Args>
      inline void construct(U *p, Args&&... args) 
        { ::new((void*)p) U(std::forward<Args>(args)...); }
      template<class U>
      inline void destroy(U *p) { p->~U(); }
#else
      inline void construct(pointer p, const T &t) { new(p) T(t); }
      inline void destroy(pointer p) { p->~T(); }
#endif
    public:
      inline bool operator==(LegionAllocator const&) const { return true; }
      inline bool operator!=(LegionAllocator const& a) const
                                           { return !operator==(a); }
    public:
#ifdef TRACE_ALLOCATION
      Runtime *runtime;
#endif
    };

    template<typename T, AllocationType A = LAST_ALLOC,
             typename COMPARATOR = std::less<T> >
    struct LegionSet {
      typedef std::set<T, COMPARATOR, 
                       LegionAllocator<T, A, false/*aligned*/> > tracked;
      typedef std::set<T, COMPARATOR, AlignedAllocator<T> > aligned;
      typedef std::set<T, COMPARATOR, 
                       LegionAllocator<T, A, true/*aligned*/> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionList {
      typedef std::list<T, LegionAllocator<T, A, false/*aligned*/> > tracked;
      typedef std::list<T, AlignedAllocator<T> > aligned;
      typedef std::list<T, 
                        LegionAllocator<T, A, true/*aligned*/> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionDeque {
      typedef std::deque<T, LegionAllocator<T, A, false/*aligned*/> > tracked;
      typedef std::deque<T, AlignedAllocator<T> > aligned;
      typedef std::deque<T, 
                         LegionAllocator<T, A, true/*aligned*/> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionVector {
      typedef std::vector<T, LegionAllocator<T, A, false/*aligned*/> > tracked;
      typedef std::vector<T, AlignedAllocator<T> > aligned;
      typedef std::vector<T, 
                         LegionAllocator<T, A, true/*aligned*/> > track_aligned;
    };

    template<typename T1, typename T2, 
             AllocationType A = LAST_ALLOC, typename COMPARATOR = std::less<T1> >
    struct LegionMap {
      typedef std::map<T1, T2, COMPARATOR,
        LegionAllocator<std::pair<const T1, T2>, A, false/*aligned*/> > tracked;
      typedef std::map<T1, T2, COMPARATOR,
                           AlignedAllocator<std::pair<const T1, T2> > > aligned;
      typedef std::map<T1, T2, COMPARATOR, LegionAllocator<
                   std::pair<const T1, T2>, A, true/*aligned*/> > track_aligned;
    };
  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_ALLOCATION__
