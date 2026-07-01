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

#ifndef __LEGION_ALLOCATION__
#define __LEGION_ALLOCATION__

#if __cplusplus >= 202002L
#include <bit>
#endif
#include <set>
#include <map>
#include <new>
#include <list>
#include <queue>
#include <deque>
#include <vector>
#include <limits>
#include <cstring>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#ifdef LEGION_TRACE_ALLOCATION
#include <typeinfo>
#endif
#include "legion/api/types.h"

// A bit tricky here, we ban use of the auto keyword in Legion.
// Types in a library like Legion should always be written
// out in full because types are for other humans to read so they
// can better understand the code. We put this declaration here
// because this file is included in almost all Legion internal
// code but is not included in user-facing code and users should
// still be allowed to use the auto keyword in their applications.
#define auto static_assert(false, "The 'auto' keyword is banned in Legion.");

namespace Legion {
  namespace Internal {

    enum AllocationLifetime {
      TASK_LOCAL_LIFETIME,  // limited to the life of this Realm task
      OPERATION_LIFETIME,   // lifetime of the operation but across tasks
      CONTEXT_LIFETIME,     // lifetime of the enclosing task context
      SHORT_LIFETIME,       // lives for a few operations but not many
      LONG_LIFETIME,        // lives for potentially a long set of operations
      RUNTIME_LIFETIME,     // lives for the duration of the Legion runtime
      TODO_LIFETIME,        // still need to figure out the lifetime
    };

#ifdef LEGION_TRACE_ALLOCATION
    // Implementations in runtime.cc
    struct LegionAllocation {
    public:
      static void trace_allocation(
          const std::type_info& info, size_t size, int elems = 1);
      static void trace_free(
          const std::type_info& info, size_t size, int elems = 1);
    };
#endif

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME, typename TRACE_TYPE = T>
    inline T* legion_malloc(std::size_t size, std::size_t alignment)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_allocation(typeid(TRACE_TYPE), size);
#endif
      if (alignment <= alignof(std::max_align_t))
        return static_cast<T*>(std::malloc(size));
      else
        return static_cast<T*>(std::aligned_alloc(alignment, size));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME, typename TRACE_TYPE = T>
    inline T* legion_calloc(std::size_t num)
    //--------------------------------------------------------------------------
    {
      if (num == 0)
        return nullptr;
      constexpr std::size_t SIZE = sizeof(T);
      constexpr std::size_t ALIGNMENT = alignof(T);
      if (num == 1)
      {
        // Need to zero-initialize this to comply with semantics of calloc
        void* ptr = static_cast<void*>(
            legion_malloc<T, LIFETIME, TRACE_TYPE>(SIZE, ALIGNMENT));
        std::memset(ptr, 0 /*value*/, SIZE);
        return static_cast<T*>(ptr);
      }
      // Compute the padding required between the elements
      constexpr std::size_t PADDING =
          (ALIGNMENT - (SIZE % ALIGNMENT)) % ALIGNMENT;
      // Has to hold for aligned alloc to work
      static_assert(((SIZE + PADDING) % ALIGNMENT) == 0);
      // Can subtract any padding off the last element
      const std::size_t bytes = num * (SIZE + PADDING);
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_allocation(typeid(TRACE_TYPE), bytes);
#endif
      void* ptr = std::aligned_alloc(ALIGNMENT, bytes);
      // Need to zero-initialize this to comply with semantics of calloc
      std::memset(ptr, 0 /*value*/, bytes);
      return static_cast<T*>(ptr);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime LIFETIME, typename TRACE_TYPE = T>
    inline T* legion_realloc(T* ptr, std::size_t old_size, std::size_t new_size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      const std::type_info& info = typeid(TRACE_TYPE);
      LegionAllocation::trace_free(info, old_size);
      LegionAllocation::trace_allocation(info, new_size);
#endif
      return static_cast<T*>(std::realloc(static_cast<void*>(ptr), new_size));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename TRACE_TYPE = T>
    inline void legion_free(T* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_TRACE_ALLOCATION
      LegionAllocation::trace_free(typeid(TRACE_TYPE), size);
#endif
      std::free(ptr);
    }

    // A class for Legion objects to inherit from to have their dynamic
    // memory allocations managed for alignment and tracing
    template<typename T, AllocationLifetime L>
    class Heapify {
    public:
      static inline void* operator new(std::size_t size);
      static inline void* operator new[](std::size_t size);
      static inline void* operator new(
          std::size_t size, std::align_val_t alignment);
      static inline void* operator new[](
          std::size_t size, std::align_val_t alignment);
    public:
      static inline void* operator new(std::size_t size, void* ptr);
      static inline void* operator new[](std::size_t size, void* ptr);
    public:
      static inline void operator delete(void* ptr, std::size_t size);
      static inline void operator delete[](void* ptr, std::size_t size);
      static inline void operator delete(void* ptr, void* place);
      static inline void operator delete[](void* ptr, void* place);
    };

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new(
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* Heapify<T, L>::operator new[](
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete(
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete[](
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete(
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void Heapify<T, L>::operator delete[](
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    // Same as Heapify but for overriding a base class definitions to keep
    // the compiler happy so it knows which definitions of operator new/delete
    // to use without getting confused
    template<typename T, typename B, AllocationLifetime L>
    class HeapifyMixin : public B {
    public:
      template<typename... Args>
      HeapifyMixin(Args&&... args) : B(std::forward<Args>(args)...)
      { }
    public:
      static inline void* operator new(std::size_t size);
      static inline void* operator new[](std::size_t size);
      static inline void* operator new(
          std::size_t size, std::align_val_t alignment);
      static inline void* operator new[](
          std::size_t size, std::align_val_t alignment);
    public:
      static inline void* operator new(std::size_t size, void* ptr);
      static inline void* operator new[](std::size_t size, void* ptr);
    public:
      static inline void operator delete(void* ptr, std::size_t size);
      static inline void operator delete[](void* ptr, std::size_t size);
      static inline void operator delete(void* ptr, void* place);
      static inline void operator delete[](void* ptr, void* place);
    };

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new(
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void* HeapifyMixin<T, B, L>::operator new[](
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete(
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete[](
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete(
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, typename B, AllocationLifetime L>
    /*static*/ inline void HeapifyMixin<T, B, L>::operator delete[](
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    // Heapify box is used for providing a box for wrapping things that
    // normally aren't allowed to be allocated on the heap
    template<typename T, AllocationLifetime L>
    class HeapifyBox : public T {
    public:
      template<typename... Args>
      HeapifyBox(Args&&... args) : T(std::forward<Args>(args)...)
      { }
    public:
      static inline void* operator new(std::size_t size);
      static inline void* operator new[](std::size_t size);
      static inline void* operator new(
          std::size_t size, std::align_val_t alignment);
      static inline void* operator new[](
          std::size_t size, std::align_val_t alignment);
    public:
      static inline void* operator new(std::size_t size, void* ptr);
      static inline void* operator new[](std::size_t size, void* ptr);
    public:
      static inline void operator delete(void* ptr, std::size_t size);
      static inline void operator delete[](void* ptr, std::size_t size);
      static inline void operator delete(void* ptr, void* place);
      static inline void operator delete[](void* ptr, void* place);
    };

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new(std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new[](std::size_t size)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(legion_malloc<T, L>(size, alignof(T)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new(
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new[](
        std::size_t size, std::align_val_t alignment)
    //--------------------------------------------------------------------------
    {
      return static_cast<void*>(
          legion_malloc<T, L>(size, static_cast<std::size_t>(alignment)));
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new(
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void* HeapifyBox<T, L>::operator new[](
        std::size_t size, void* ptr)
    //--------------------------------------------------------------------------
    {
      // No need to do tracing of allocations, that is handled when
      // legion_malloc is called for the type
      return ptr;
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void HeapifyBox<T, L>::operator delete(
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void HeapifyBox<T, L>::operator delete[](
        void* ptr, std::size_t size)
    //--------------------------------------------------------------------------
    {
      legion_free<T>(static_cast<T*>(ptr), size);
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void HeapifyBox<T, L>::operator delete(
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, AllocationLifetime L>
    /*static*/ inline void HeapifyBox<T, L>::operator delete[](
        void* ptr, void* place)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    // A class to ensure that a type is never dynamically allocated
    class NoHeapify {
    public:
      static inline void* operator new(std::size_t) = delete;
      static inline void* operator new[](std::size_t) = delete;
      static inline void* operator new(std::size_t, std::align_val_t) = delete;
      static inline void* operator new[](std::size_t, std::align_val_t) =
          delete;
    public:
      static inline void* operator new(std::size_t, void*) = delete;
      static inline void* operator new[](std::size_t, void*) = delete;
      static inline void* operator new(std::size_t, std::align_val_t, void*) =
          delete;
      static inline void* operator new[](std::size_t, std::align_val_t, void*) =
          delete;
    public:
      static inline void operator delete(void* ptr, std::size_t size) = delete;
      static inline void operator delete[](void* ptr, std::size_t size) =
          delete;
      static inline void operator delete(void* ptr, void* place) = delete;
      static inline void operator delete[](void* ptr, void* place) = delete;
    };

    /**
     * \class LegionAllocator
     * A custom Legion allocator for tracing memory usage in STL
     * data structures. When tracing is disabled, it defaults back
     * to using the standard malloc/free and new/delete operations.
     */
    template<typename T, AllocationLifetime L>
    class LegionAllocator {
    public:
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;
      typedef T* pointer;
      typedef const T* const_pointer;
      typedef T& reference;
      typedef const T& const_reference;
      typedef T value_type;
    public:
      template<typename U>
      struct rebind {
        typedef LegionAllocator<U, L> other;
      };
    public:
      inline explicit LegionAllocator(void) { }
      inline ~LegionAllocator(void) { }
      inline LegionAllocator(const LegionAllocator<T, L>& rhs) { }
      template<typename U>
      inline LegionAllocator(const LegionAllocator<U, L>& rhs)
      { }
    public:
      inline pointer address(reference r) { return &r; }
      inline const_pointer address(const_reference r) { return &r; }
    public:
      inline T* allocate(std::size_t num)
      {
        return static_cast<T*>(legion_calloc<T, L>(num));
      }
      inline void deallocate(T* ptr, std::size_t num)
      {
#ifdef LEGION_TRACE_ALLOCATION
        constexpr std::size_t SIZE = sizeof(T);
        if (num == 1)
        {
          legion_free<T>(ptr, SIZE);
        }
        else if (num > 1)
        {
          constexpr std::size_t ALIGNMENT = alignof(T);
          // Compute the padding required between the elements
          constexpr std::size_t PADDING =
              (ALIGNMENT - (SIZE % ALIGNMENT)) % ALIGNMENT;
          // Can subtract any padding off the last element
          const std::size_t bytes = num * (SIZE + PADDING);
          legion_free<T>(ptr, bytes);
        }
#else
        legion_free<T>(ptr, 0 /*bogus size*/);
#endif
      }
    public:
      inline size_type max_size(void) const
      {
        return std::numeric_limits<size_type>::max() / sizeof(T);
      }
    public:
#if __cplusplus > 201703L
      template<class U, class... Args>
      inline constexpr U* construct_at(U* p, Args&&... args)
      {
        return ::new (const_cast<void*>(static_cast<const void volatile *>(p)))
            U(std::forward<Args>(args)...);
      }

#else
      template<class U, class... Args>
      inline void construct(U* p, Args&&... args)
      {
        ::new ((void*)p) U(std::forward<Args>(args)...);
      }
#endif
#if __cplusplus > 201703L
      template<class U>
      inline constexpr void destroy_at(U* p)
      {
        p->~U();
      }
#else
      template<class U>
      inline void destroy_at(U* p)
      {
        p->~U();
      }
#endif
    public:
      inline bool operator==(const LegionAllocator&) const { return true; }
      inline bool operator!=(const LegionAllocator& a) const
      {
        return !operator==(a);
      }
    };

    // namespaces for different lifetime data structures
    namespace local {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set =
          std::set<T, COMPARATOR, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, TASK_LOCAL_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set = std::unordered_set<
          T, HASH, KEY, LegionAllocator<T, TASK_LOCAL_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, TASK_LOCAL_LIFETIME> >;
    }  // namespace local
    namespace op {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set =
          std::set<T, COMPARATOR, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, OPERATION_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set = std::unordered_set<
          T, HASH, KEY, LegionAllocator<T, OPERATION_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, OPERATION_LIFETIME> >;
    }  // namespace op
    namespace ctx {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set =
          std::set<T, COMPARATOR, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, CONTEXT_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set = std::unordered_set<
          T, HASH, KEY, LegionAllocator<T, CONTEXT_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, CONTEXT_LIFETIME> >;
    }  // namespace ctx
    namespace shrt {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set = std::set<T, COMPARATOR, LegionAllocator<T, SHORT_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, SHORT_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, SHORT_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, SHORT_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, SHORT_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, SHORT_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set =
          std::unordered_set<T, HASH, KEY, LegionAllocator<T, SHORT_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, SHORT_LIFETIME> >;
    }  // namespace shrt
    namespace lng {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set = std::set<T, COMPARATOR, LegionAllocator<T, LONG_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, LONG_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, LONG_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, LONG_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, LONG_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, LONG_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set =
          std::unordered_set<T, HASH, KEY, LegionAllocator<T, LONG_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, LONG_LIFETIME> >;
    }  // namespace lng
    namespace rt {
      template<typename T, typename COMPARATOR = std::less<T> >
      using set =
          std::set<T, COMPARATOR, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
      using map = std::map<
          T1, T2, COMPARATOR,
          LegionAllocator<std::pair<const T1, T2>, RUNTIME_LIFETIME> >;
      template<typename T>
      using list = std::list<T, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<typename T>
      using queue = std::queue<T, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<typename T>
      using deque = std::deque<T, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<typename T>
      using vector = std::vector<T, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<
          typename T, typename HASH = std::hash<T>,
          typename KEY = std::equal_to<T> >
      using unordered_set = std::unordered_set<
          T, HASH, KEY, LegionAllocator<T, RUNTIME_LIFETIME> >;
      template<
          typename T1, typename T2, typename HASH = std::hash<T1>,
          typename KEY = std::equal_to<T1> >
      using unordered_map = std::unordered_map<
          T1, T2, HASH, KEY,
          LegionAllocator<std::pair<const T1, T2>, RUNTIME_LIFETIME> >;
    }  // namespace rt

    // Some helper classes to provide views so that we can deal with underlying
    // data structures with different allocators
    template<typename T, typename COMPARATOR = std::less<T> >
    class SetView {
    public:
      using const_iterator = typename std::set<T, COMPARATOR>::const_iterator;
    public:
      template<typename ALLOCATOR>
      inline SetView(const std::set<T, COMPARATOR, ALLOCATOR>& set)
        : start(set.cbegin()), stop(set.cend()), full_size(set.size())
      { }
    public:
      inline size_t size(void) const { return full_size; }
      inline bool empty(void) const { return (start == stop); }
      inline const_iterator begin(void) const { return start; }
      inline const_iterator end(void) const { return stop; }
      inline const_iterator cbegin(void) const { return start; }
      inline const_iterator cend(void) const { return stop; }
      inline const_iterator find(T value) const
      {
        const_iterator it = std::lower_bound(start, stop, value);
        if ((it != stop) && !COMPARATOR()(value, *it))
          return it;
        else
          return stop;
      }
    private:
      const const_iterator start, stop;
      const size_t full_size;
    };

    template<typename T1, typename T2, typename COMPARATOR = std::less<T1> >
    class MapView {
    public:
      using const_iterator =
          typename std::map<T1, T2, COMPARATOR>::const_iterator;
    public:
      template<typename ALLOCATOR>
      inline MapView(const std::map<T1, T2, COMPARATOR, ALLOCATOR>& map)
        : start(map.cbegin()), stop(map.cend()), full_size(map.size())
      { }
    public:
      inline size_t size(void) const { return full_size; }
      inline bool empty(void) const { return (start == stop); }
      inline const_iterator begin(void) const { return start; }
      inline const_iterator end(void) const { return stop; }
      inline const_iterator cbegin(void) const { return start; }
      inline const_iterator cend(void) const { return stop; }
      inline const_iterator find(T1 key) const
      {
        const_iterator it = std::lower_bound(
            start, stop, key,
            [](const std::pair<const T1, T2>& pair, const T1& k) -> bool {
              return COMPARATOR()(pair.first, k);
            });
        if ((it != stop) && !COMPARATOR()(key, it->first))
          return it;
        else
          return stop;
      }
    private:
      const const_iterator start, stop;
      const size_t full_size;
    };

    template<typename T>
    class VectorView {
    public:
      using const_iterator = typename std::vector<T>::const_iterator;
    public:
      template<typename ALLOCATOR>
      inline VectorView(const std::vector<T, ALLOCATOR>& vector)
        : length(vector.size()), ptr((length > 0) ? &vector.front() : nullptr)
      { }
    public:
      inline size_t size(void) const { return length; }
      inline bool empty(void) const { return (length > 0); }
      inline const_iterator begin(void) const { return ptr; }
      inline const_iterator end(void) const { return ptr + length; }
      inline const_iterator cbegin(void) const { return ptr; }
      inline const_iterator cend(void) const { return ptr + length; }
      inline const T& operator[](unsigned idx) const { return ptr[idx]; }
    private:
      const size_t length;
      const T* const ptr;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ALLOCATION__
