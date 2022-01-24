/* Copyright 2022 Stanford University, NVIDIA Corporation
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
#include <stddef.h>
#include <functional>
#include <stdlib.h>
#include "legion/legion_config.h"
#include "legion/legion_template_help.h" // StaticAssert
#include <utility>

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
      EXTERNAL_RESOURCES_ALLOC,
      STATIC_TRACE_ALLOC,
      DYNAMIC_TRACE_ALLOC,
      ALLOC_MANAGER_ALLOC,
      ALLOC_INTERNAL_ALLOC,
      TASK_ARGS_ALLOC,
      REDUCTION_ALLOC,
      PREDICATE_ALLOC,
      FUTURE_RESULT_ALLOC,
      INDIVIDUAL_INST_MANAGER_ALLOC,
      COLLECTIVE_INST_MANAGER_ALLOC,
      TREE_CLOSE_ALLOC,
      TREE_CLOSE_IMPL_ALLOC,
      MATERIALIZED_VIEW_ALLOC,
      REDUCTION_VIEW_ALLOC,
      FILL_VIEW_ALLOC,
      PHI_VIEW_ALLOC,
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
      CREATION_OP_ALLOC,
      DELETION_OP_ALLOC,
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
      TRACE_REPLAY_OP_ALLOC,
      TRACE_BEGIN_OP_ALLOC,
      TRACE_SUMMARY_OP_ALLOC,
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
      EQUIVALENCE_SET_ALLOC,
      AGGREGATE_VERSION_ALLOC,
      TASK_IMPL_ALLOC,
      VARIANT_IMPL_ALLOC,
      LAYOUT_CONSTRAINTS_ALLOC,
      COPY_FILL_AGGREGATOR_ALLOC,
      UNION_EXPR_ALLOC,
      INTERSECTION_EXPR_ALLOC,
      DIFFERENCE_EXPR_ALLOC,
      INSTANCE_EXPR_ALLOC,
      REMOTE_EXPR_ALLOC,
      UNTRACKED_ALLOC, // must be last
    };

    //--------------------------------------------------------------------------
    template<size_t SIZE, size_t ALIGNMENT, bool BYTES>
    inline void* legion_alloc_aligned(size_t cnt)
    //--------------------------------------------------------------------------
    {
      LEGION_STATIC_ASSERT((SIZE % ALIGNMENT) == 0, 
          "SIZE must be evenly divisible by ALIGNMENT");
      size_t alloc_size = cnt;
      if (!BYTES)
        alloc_size *= SIZE;
      void *result = NULL;
      if (ALIGNMENT > LEGION_MAX_ALIGNMENT)
      {
#ifdef DEBUG_LEGION
        assert((alloc_size % ALIGNMENT) == 0);
#endif
#if defined(DEBUG_LEGION) && !defined(NDEBUG)
        int error = 
#else
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#endif
#endif
          posix_memalign(&result, ALIGNMENT, alloc_size);
#if defined(DEBUG_LEGION) && !defined(NDEBUG)
        assert(error == 0);
#else
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
      }
      else
        result = malloc(alloc_size);

#ifdef DEBUG_LEGION
      assert(result != NULL);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool BYTES>
    inline void* legion_alloc_aligned(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<sizeof(T), alignof(T), BYTES>(count);
    }

#ifdef LEGION_TRACE_ALLOCATION
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
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_allocation(a, size);
#endif
      return malloc(size);
    }

    //--------------------------------------------------------------------------
    inline void* legion_realloc(AllocationType a, void *ptr, 
                                size_t old_size, size_t new_size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
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
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_free(a, size);
#endif
      free(ptr);
    }

    // A class for Legion objects to inherit from to have their dynamic
    // memory allocations managed for alignment and tracing
    template<typename T>
    class LegionHeapify {
    public:
      static inline void* operator new(size_t count);
      static inline void* operator new[](size_t count);
    public:
      static inline void* operator new(size_t count, void *ptr);
      static inline void* operator new[](size_t count, void *ptr);
    public:
      static inline void operator delete(void *ptr);
      static inline void operator delete[](void *ptr);
    public:
      static inline void operator delete(void *ptr, void *place);
      static inline void operator delete[](void *ptr, void *place);
    };

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* LegionHeapify<T>::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      return legion_alloc_aligned<T,true/*bytes*/>(count);  
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* LegionHeapify<T>::operator new[](size_t count)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      return legion_alloc_aligned<T,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* LegionHeapify<T>::operator new(size_t count, 
                                                           void *ptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void* LegionHeapify<T>::operator new[](size_t count,
                                                             void *ptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_allocation();
#endif
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void LegionHeapify<T>::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void LegionHeapify<T>::operator delete[](void *ptr)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void LegionHeapify<T>::operator delete(void *ptr, 
                                                             void *place)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
      free(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    /*static*/ inline void LegionHeapify<T>::operator delete[](void *ptr,
                                                               void *place)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      HandleAllocation<T,HasAllocType<T>::value>::trace_free();
#endif
      free(ptr);
    }

    /**
     * \class LegionAllocator
     * A custom Legion allocator for tracing memory usage in STL
     * data structures. When tracing is disabled, it defaults back
     * to using the standard malloc/free and new/delete operations.
     */
    template<typename T, AllocationType A = UNTRACKED_ALLOC>
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
        typedef LegionAllocator<U, A> other;
      };
    public:
      inline explicit LegionAllocator(void) 
#ifdef LEGION_TRACE_ALLOCATION
        : runtime(LegionAllocation::find_runtime()) 
#endif
      { }
      inline ~LegionAllocator(void) { }
      inline LegionAllocator(const LegionAllocator<T, A> &rhs)
#ifdef LEGION_TRACE_ALLOCATION
        : runtime(rhs.runtime) 
#endif
      { }
      template<typename U>
      inline LegionAllocator(const LegionAllocator<U, A> &rhs) 
#ifdef LEGION_TRACE_ALLOCATION
        : runtime(rhs.runtime) 
#endif  
      { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
#if __cplusplus > 201402L
      inline T* allocate(std::size_t cnt) { 
        void *ptr = legion_alloc_aligned<T, false/*bytes*/>(cnt);
        pointer result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "Fuck c++");
        memcpy(&result, &ptr, sizeof(result));
#ifdef LEGION_TRACE_ALLOCATION
        if (A != UNTRACKED_ALLOC)
          LegionAllocation::trace_allocation(runtime, A, sizeof(T), cnt);
#endif
        return result;
      }
      inline void deallocate(T *ptr, std::size_t size) { 
#ifdef LEGION_TRACE_ALLOCATION
        if (A != UNTRACKED_ALLOC)
          LegionAllocation::trace_free(runtime, A, sizeof(T), size);
#endif
        free(ptr);
      }
#else
      inline pointer allocate(size_type cnt,
                      typename std::allocator<void>::const_pointer = 0) {
#ifdef LEGION_TRACE_ALLOCATION
        if (A != UNTRACKED_ALLOC)
          LegionAllocation::trace_allocation(runtime, A, sizeof(T), cnt);
#endif
        void *ptr = legion_alloc_aligned<T, false/*bytes*/>(cnt);
        pointer result = NULL;
        static_assert(sizeof(result) == sizeof(ptr), "Fuck c++");
        memcpy(&result, &ptr, sizeof(result));
        return result;
      }
      inline void deallocate(pointer p, size_type size) {
#ifdef LEGION_TRACE_ALLOCATION
        if (A != UNTRACKED_ALLOC)
          LegionAllocation::trace_free(runtime, A, sizeof(T), size);
#endif
        free(p);
      }
#endif // After/Before C++17
    public:
      inline size_type max_size(void) const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
      }
    public:
#if __cplusplus > 201703L
      template<class U, class... Args>
      inline constexpr U* construct_at( U* p, Args&&... args ) 
        { return ::new (const_cast<void*>(static_cast<const volatile void*>(p)))
                        U(std::forward<Args>(args)...); } 

#else
      template<class U, class... Args>
      inline void construct(U* p, Args&&... args) 
        { ::new((void*)p) U(std::forward<Args>(args)...); }
#endif
#if __cplusplus > 201703L
      template<class U>
      inline constexpr void destroy_at(U* p) { p->~U(); }
#elif __cplusplus == 201703L
      template<class U>
      inline void destroy_at(U* p) { p->~U(); }
#else
      template<class U>
      inline void destroy(U* p) { p->~U(); }
#endif
    public:
      inline bool operator==(LegionAllocator const&) const { return true; }
      inline bool operator!=(LegionAllocator const& a) const
                                           { return !operator==(a); }
    public:
#ifdef LEGION_TRACE_ALLOCATION
      Runtime *runtime;
#endif
    };

    template<typename T, AllocationType A = UNTRACKED_ALLOC,
             typename COMPARATOR = std::less<T> >
    using LegionSet = std::set<T, COMPARATOR, LegionAllocator<T,A> >;

    template<typename T, AllocationType A = UNTRACKED_ALLOC>
    using LegionList = std::list<T, LegionAllocator<T, A> >;

    template<typename T, AllocationType A = UNTRACKED_ALLOC>
    using LegionDeque = std::deque<T, LegionAllocator<T, A> >;

    template<typename T, AllocationType A = UNTRACKED_ALLOC>
    using LegionVector = std::vector<T, LegionAllocator<T, A> >;

    template<typename T1, typename T2, AllocationType A = UNTRACKED_ALLOC,
             typename COMPARATOR = std::less<T1> >
    using LegionMap = std::map<T1, T2, COMPARATOR,
            LegionAllocator<std::pair<const T1,T2>, A> >;
  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_ALLOCATION__
