/* Copyright 2015 Stanford University
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
#include <list>
#include <deque>
#include <vector>
#include <limits>
#include <cstddef>
#include <functional>
#include <stdlib.h>
#include "legion_types.h" // StaticAssert

namespace LegionRuntime {
  namespace HighLevel {

    enum AllocationType {
      ARGUMENT_MAP_ALLOC,
      ARGUMENT_MAP_STORE_ALLOC,
      STORE_ARGUMENT_ALLOC,
      MPI_HANDSHAKE_ALLOC,
      GRANT_ALLOC,
      FUTURE_ALLOC,
      FUTURE_MAP_ALLOC,
      PHYSICAL_REGION_ALLOC,
      TRACE_ALLOC,
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
      INDIVIDUAL_TASK_ALLOC,
      POINT_TASK_ALLOC,
      INDEX_TASK_ALLOC,
      SLICE_TASK_ALLOC,
      REMOTE_TASK_ALLOC,
      INLINE_TASK_ALLOC,
      MAP_OP_ALLOC,
      COPY_OP_ALLOC,
      FENCE_OP_ALLOC,
      FRAME_OP_ALLOC,
      DELETION_OP_ALLOC,
      CLOSE_OP_ALLOC,
      FUTURE_PRED_OP_ALLOC,
      NOT_PRED_OP_ALLOC,
      AND_PRED_OP_ALLOC,
      OR_PRED_OP_ALLOC,
      ACQUIRE_OP_ALLOC,
      RELEASE_OP_ALLOC,
      TRACE_CAPTURE_OP_ALLOC,
      TRACE_COMPLETE_OP_ALLOC,
      MUST_EPOCH_OP_ALLOC,
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
      VALID_VIEW_ALLOC,
      VALID_REDUCTION_ALLOC,
      PENDING_UPDATES_ALLOC,
      LAYOUT_DESCRIPTION_ALLOC,
      CURR_PHYSICAL_ALLOC,
      PREV_PHYSICAL_ALLOC,
      EVENT_REFERENCE_ALLOC,
      PHYSICAL_VERSION_ALLOC,
      MEMORY_INSTANCES_ALLOC,
      MEMORY_REDUCTION_ALLOC,
      MEMORY_AVAILABLE_ALLOC,
      PROCESSOR_GROUP_ALLOC,
      RUNTIME_DISTRIBUTED_ALLOC,
      RUNTIME_DIST_COLLECT_ALLOC,
      RUNTIME_HIER_COLLECT_ALLOC,
      RUNTIME_GC_EPOCH_ALLOC,
      RUNTIME_FUTURE_ALLOC,
      RUNTIME_REMOTE_ALLOC,
      TASK_INSTANCE_REGION_ALLOC,
      TASK_LOCAL_REGION_ALLOC,
      TASK_INLINE_REGION_ALLOC,
      TASK_TRACES_ALLOC,
      TASK_RESERVATION_ALLOC,
      TASK_BARRIER_ALLOC,
      TASK_LOCAL_FIELD_ALLOC,
      TASK_INLINE_ALLOC,
      SEMANTIC_INFO_ALLOC,
      DIRECTORY_ALLOC,
      LAST_ALLOC, // must be last
    };

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
#endif

    // Helper methods for doing tracing of memory allocations
    inline void* legion_malloc(AllocationType a, size_t size)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(a, size);
#endif
      return malloc(size);
    }

    inline void* legion_realloc(AllocationType a, void *ptr, 
                                size_t old_size, size_t new_size)
    {
#ifdef TRACE_ALLOCATION
      Runtime *rt = LegionAllocation::find_runtime(); 
      LegionAllocation::trace_free(rt, a, old_size);
      LegionAllocation::trace_allocation(rt, a, new_size);
#endif
      return realloc(ptr, new_size);
    }

    inline void legion_free(AllocationType a, void *ptr, size_t size)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_free(a, size);
#endif
      free(ptr);
    }

    template<typename T>
    inline T* legion_new(void)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T();
    }

    template<typename T, typename T1>
    inline T* legion_new(const T1 &arg1)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1);
    }

    template<typename T, typename T1, typename T2>
    inline T* legion_new(const T1 &arg1, const T2 &arg2)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2);
    }

    template<typename T, typename T1, typename T2, typename T3>
    inline T* legion_new(const T1 &arg1, const T2 &arg2, const T3 &arg3)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3);
    }

    template<typename T, typename T1, typename T2, typename T3, typename T4>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, const T5 &arg5)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6, const T7 &arg7)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, const T9 &arg9)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
    }

    template<typename T, typename T1, typename T2, typename T3, 
             typename T4, typename T5, typename T6, typename T7,
             typename T8, typename T9, typename T10, typename T11>
    inline T* legion_new(const T1 &arg1, const T2 &arg2,
                         const T3 &arg3, const T4 &arg4, 
                         const T5 &arg5, const T6 &arg6,
                         const T7 &arg7, const T8 &arg8, 
                         const T9 &arg9, const T10 &arg10, const T11 &arg11)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_allocation(T::alloc_type, sizeof(T));
#endif
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, 
                   arg9, arg10, arg11);
    }

    template<typename T>
    inline void legion_delete(T *to_free)
    {
#ifdef TRACE_ALLOCATION
      LegionAllocation::trace_free(T::alloc_type, sizeof(T));
#endif
      delete to_free;
    }

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

    /**
     * \class AlignedAllocator
     * A class for doing aligned allocation of memory for
     * STL data structures containing data types that require
     * larger alignments than the default malloc provides.
     */
    template<typename T, size_t N = 8>
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
        typedef AlignedAllocator<U, N> other;
      };
    public:
      inline explicit AlignedAllocator(void) { }
      inline ~AlignedAllocator(void) { }
      inline AlignedAllocator(const AlignedAllocator<T, N> &rhs) { }
      template<typename U>
      inline AlignedAllocator(const AlignedAllocator<U, N> &rhs) { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
      inline pointer allocate(size_type cnt,
                              typename std::allocator<void>::const_pointer = 0)
      {
        size_type alloc_size = cnt * sizeof(T);
        if (AlignmentTrait<T>::AlignmentOf >= N)
        {
          void *result;
          int error = 
            posix_memalign(&result, AlignmentTrait<T>::AlignmentOf, alloc_size);
          assert(error == 0);
          return reinterpret_cast<pointer>(result);
        }
        return reinterpret_cast<pointer>(malloc(alloc_size));
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
    template<typename T, AllocationType A, size_t N = 8>
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
        typedef LegionAllocator<U, A, N> other;
      };
    public:
      inline explicit LegionAllocator(void) 
#ifdef TRACE_ALLOCATION
        : runtime(LegionAllocation::find_runtime()) 
#endif
      { }
      inline ~LegionAllocator(void) { }
      inline LegionAllocator(const LegionAllocator<T, A, N> &rhs)
#ifdef TRACE_ALLOCATION
        : runtime(rhs.runtime) 
#endif
      { }
      template<typename U>
      inline LegionAllocator(const LegionAllocator<U, A, N> &rhs) 
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
        size_type alloc_size = cnt * sizeof(T);
#ifdef TRACE_ALLOCATION
        LegionAllocation::trace_allocation(runtime, A, sizeof(T), cnt);
#endif
        if (AlignmentTrait<T>::AlignmentOf >= N)
        {
          void *result;
          int error = 
            posix_memalign(&result, AlignmentTrait<T>::AlignmentOf, alloc_size);
          assert(error == 0);
          return reinterpret_cast<pointer>(result);
        }
        return reinterpret_cast<pointer>(malloc(alloc_size));
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
      inline void construct(pointer p, const T &t) { new(p) T(t); }
      inline void destroy(pointer p) { p->~T(); }
    public:
      inline bool operator==(LegionAllocator const&) const { return true; }
      inline bool operator!=(LegionAllocator const& a) const
                                           { return !operator==(a); }
    public:
#ifdef TRACE_ALLOCATION
      Runtime *runtime;
#endif
    };

#if defined(__AVX__)
#define MAX_ALIGN 32
#elif defined(__SSE2__)
#define MAX_ALIGN 16
#else
#define MAX_ALIGN 8
#endif
    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionSet {
      typedef std::set<T, std::less<T>, LegionAllocator<T, A> > tracked;
      typedef std::set<T, std::less<T>, 
                       AlignedAllocator<T, MAX_ALIGN> > aligned;
      typedef std::set<T, std::less<T>,
                       LegionAllocator<T, A, MAX_ALIGN> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionList {
      typedef std::list<T, LegionAllocator<T, A> > tracked;
      typedef std::list<T, AlignedAllocator<T, MAX_ALIGN> > aligned;
      typedef std::list<T, LegionAllocator<T, A, MAX_ALIGN> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionDeque {
      typedef std::deque<T, LegionAllocator<T, A> > tracked;
      typedef std::deque<T, AlignedAllocator<T, MAX_ALIGN> > aligned;
      typedef std::deque<T, LegionAllocator<T, A, MAX_ALIGN> > track_aligned;
    };

    template<typename T, AllocationType A = LAST_ALLOC>
    struct LegionVector {
      typedef std::vector<T, LegionAllocator<T, A> > tracked;
      typedef std::vector<T, AlignedAllocator<T, MAX_ALIGN> > aligned;
      typedef std::vector<T, LegionAllocator<T, A, MAX_ALIGN> > track_aligned;
    };

    template<typename T1, typename T2, AllocationType A = LAST_ALLOC>
    struct LegionMap {
      typedef std::map<T1, T2, std::less<T1>,
               LegionAllocator<std::pair<const T1, T2>, A> > tracked;
      typedef std::map<T1, T2, std::less<T1>,
               AlignedAllocator<std::pair<const T1, T2>, MAX_ALIGN> > aligned;
      typedef std::map<T1, T2, std::less<T1>,
       LegionAllocator<std::pair<const T1, T2>, A, MAX_ALIGN> > track_aligned;
    };
#undef MAX_ALIGN

#if 0
    template<typename T, AllocationType A>
    struct LegionContainer {
      typedef std::set<T, std::less<T>, LegionAllocator<T, A> > set;
      typedef std::list<T, LegionAllocator<T, A> > list;
      typedef std::deque<T, LegionAllocator<T, A> > deque;
      typedef std::vector<T, LegionAllocator<T, A> > vector;
    };

    template<typename T1, typename T2, AllocationType A>
    struct LegionKeyValue {
      typedef std::map<T1, T2, std::less<T1>, 
                       LegionAllocator<std::pair<const T1, T2>, A> > map;
    };
#endif
  };
};

#endif // __LEGION_ALLOCATION__
