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

#ifndef __LEGION_FIELDMASK_MAP_H__
#define __LEGION_FIELDMASK_MAP_H__

#include "legion/utilities/bitmask.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct FieldSet
     * A helper template class for the method below for describing
     * sets of members that all contain the same fields
     */
    template<typename T>
    struct FieldSet {
    public:
      FieldSet(void) { }
      FieldSet(const FieldMask& m) : set_mask(m) { }
    public:
      FieldMask set_mask;
      local::set<T> elements;
    };

    /**
     * \class FieldMaskMap
     * A template helper class for tracking collections of
     * objects associated with different sets of fields
     */
    template<
        typename T, AllocationLifetime L,
        typename COMPARATOR = std::less<const T*> >
    class FieldMaskMap : public Heapify<FieldMaskMap<T, L, COMPARATOR>, L> {
    private:
      using FMMap = std::map<
          T*, FieldMask, COMPARATOR,
          LegionAllocator<std::pair<T* const, FieldMask>, L> >;
    public:
      // forward declaration
      class const_iterator;
      class iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T* const, FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T* const, FieldMask>* pointer;
        typedef std::pair<T* const, FieldMask>& reference;

        iterator(FieldMaskMap* _map, std::pair<T* const, FieldMask>* _result)
          : map(_map), result(_result), single(true)
        { }
        iterator(
            FieldMaskMap* _map, typename FMMap::iterator _it, bool end = false)
          : map(_map), result(end ? nullptr : &(*_it)), it(_it), single(false)
        { }
      public:
        iterator(const iterator& rhs)
          : map(rhs.map), result(rhs.result), it(rhs.it), single(rhs.single)
        { }
        ~iterator(void) { }
      public:
        inline iterator& operator=(const iterator& rhs)
        {
          map = rhs.map;
          result = rhs.result;
          it = rhs.it;
          single = rhs.single;
          return *this;
        }
      public:
        inline bool operator==(const iterator& rhs) const
        {
          legion_assert(map == rhs.map);
          if (single)
            return (result == rhs.result);
          else
            return (it == rhs.it);
        }
        inline bool operator!=(const iterator& rhs) const
        {
          legion_assert(map == rhs.map);
          if (single)
            return (result != rhs.result);
          else
            return (it != rhs.it);
        }
      public:
        inline const std::pair<T* const, FieldMask> operator*(void)
        {
          return *result;
        }
        inline const std::pair<T* const, FieldMask>* operator->(void)
        {
          return result;
        }
        inline iterator& operator++(/*prefix*/ void)
        {
          if (!single)
          {
            ++it;
            if ((*this) != map->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return *this;
        }
        inline iterator operator++(/*postfix*/ int)
        {
          iterator copy(*this);
          if (!single)
          {
            ++it;
            if ((*this) != map->end())
              result = &(*it);
            else
              result = nullptr;
          }
          else
            result = nullptr;
          return copy;
        }
      public:
        inline operator bool(void) const { return (result != nullptr); }
      public:
        inline void merge(const FieldMask& mask)
        {
          result->second |= mask;
          if (!single)
            map->valid_fields |= mask;
        }
        inline void filter(const FieldMask& mask)
        {
          result->second -= mask;
          // Don't filter valid fields since its unsound
        }
        inline void clear(void) { result->second.clear(); }
      public:
        inline void erase(FMMap& target)
        {
          legion_assert(!single);
          // Erase it from the target
          target.erase(it);
          // Invalidate the iterator
          it = target.end();
          result = nullptr;
        }
      private:
        friend class const_iterator;
        FieldMaskMap* map;
        std::pair<T* const, FieldMask>* result;
        typename FMMap::iterator it;
        bool single;
      };
    public:
      class const_iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T* const, FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T* const, FieldMask>* pointer;
        typedef std::pair<T* const, FieldMask>& reference;

        const_iterator(const std::pair<T* const, FieldMask>* _result)
          : result(_result), it(typename FMMap::const_iterator()), single(true)
        { }
        const_iterator(typename FMMap::const_iterator _it)
          : result(nullptr), it(_it), single(false)
        { }
      public:
        const_iterator(const const_iterator& rhs)
          : result(nullptr), it(typename FMMap::const_iterator()),
            single(rhs.single)
        {
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
        }
        // We can also make a const_iterator from a normal iterator
        const_iterator(const iterator& rhs)
          : result(nullptr), it(typename FMMap::const_iterator()),
            single(rhs.single)
        {
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
        }
        ~const_iterator(void) { }
      public:
        inline const_iterator& operator=(const const_iterator& rhs)
        {
          single = rhs.single;
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
          return *this;
        }
        inline const_iterator& operator=(const iterator& rhs)
        {
          single = rhs.single;
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
          return *this;
        }
      public:
        inline bool operator==(const const_iterator& rhs) const
        {
          if (single)
            return (result == rhs.result);
          else
            return (it == rhs.it);
        }
        inline bool operator!=(const const_iterator& rhs) const
        {
          if (single)
            return (result != rhs.result);
          else
            return (it != rhs.it);
        }
      public:
        inline const std::pair<T* const, FieldMask> operator*(void)
        {
          if (single)
            return *result;
          else
            return *it;
        }
        inline const std::pair<T* const, FieldMask>* operator->(void)
        {
          if (single)
            return result;
          else
            return &(*it);
        }
        inline const_iterator& operator++(/*prefix*/ void)
        {
          if (single)
            result = nullptr;
          else
            ++it;
          return *this;
        }
        inline const_iterator operator++(/*postfix*/ int)
        {
          const_iterator copy(*this);
          if (single)
            result = nullptr;
          else
            ++it;
          return copy;
        }
      private:
        const std::pair<T* const, FieldMask>* result;
        typename FMMap::const_iterator it;
        bool single;
      };
    public:
      FieldMaskMap(void) : single(true) { entries.single_entry = nullptr; }
      inline FieldMaskMap(T* init, const FieldMask& m, bool no_null = true);
      inline FieldMaskMap(const FieldMaskMap<T, L, COMPARATOR>& rhs);
      template<AllocationLifetime L2>
      inline FieldMaskMap(const FieldMaskMap<T, L2, COMPARATOR>& rhs);
      inline FieldMaskMap(FieldMaskMap<T, L, COMPARATOR>&& rhs) noexcept;
      // If copy is set to false then this is a move constructor
      inline FieldMaskMap(FieldMaskMap<T, L, COMPARATOR>& rhs, bool copy);
      ~FieldMaskMap(void) { clear(); }
    public:
      inline FieldMaskMap& operator=(const FieldMaskMap<T, L, COMPARATOR>& rh);
      inline FieldMaskMap& operator=(
          FieldMaskMap<T, L, COMPARATOR>&& rhs) noexcept;
    public:
      inline bool empty(void) const
      {
        return single && (entries.single_entry == nullptr);
      }
      inline const FieldMask& get_valid_mask(void) const
      {
        return valid_fields;
      }
      inline const FieldMask& tighten_valid_mask(void);
      inline void relax_valid_mask(const FieldMask& m);
      inline void filter_valid_mask(const FieldMask& m);
      inline void restrict_valid_mask(const FieldMask& m);
    public:
      inline const FieldMask& operator[](T* entry) const;
    public:
      // Return true if we actually added the entry, false if it already existed
      inline bool insert(T* entry, const FieldMask& mask);
      inline void filter(const FieldMask& filter, bool tighten = true);
      inline void erase(T* to_erase);
      inline void clear(void);
      inline size_t size(void) const;
    public:
      inline void swap(FieldMaskMap& other);
    public:
      inline iterator begin(void);
      inline iterator find(T* entry);
      template<typename T2>
      inline iterator find(const T2& key);
      inline void erase(iterator& it);
      inline iterator end(void);
    public:
      inline const_iterator begin(void) const;
      inline const_iterator cbegin(void) const;
      inline const_iterator find(T* entry) const;
      template<typename T2>
      inline const_iterator find(const T2& key) const;
      inline const_iterator end(void) const;
      inline const_iterator cend(void) const;
    public:
      inline void compute_field_sets(
          FieldMask universe_mask,
          local::list<FieldSet<T*> >& output_sets) const;
    protected:
      template<typename T2, AllocationLifetime L2, typename C2>
      friend class FieldMaskMap;
      template<typename T2, typename C2>
      friend class FieldMapView;

      // Fun with C, keep these two fields first and in this order
      // so that a FieldMaskMap of size 1 looks the same as an entry
      // in the STL Map in the multi-entries case,
      // provides goodness for the iterator
      union {
        T* single_entry;
        FMMap* multi_entries;
      } entries;
      // This can be an overapproximation if we have multiple entries
      FieldMask valid_fields;
      bool single;
    };

    template<typename T, typename COMPARATOR = std::less<const T*> >
    class FieldMapView {
    public:
      using FMMap = std::map<T*, FieldMask, COMPARATOR>;
      class const_iterator {
      public:
        // explicitly set iterator traits
        typedef std::input_iterator_tag iterator_category;
        typedef std::pair<T* const, FieldMask> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<T* const, FieldMask>* pointer;
        typedef std::pair<T* const, FieldMask>& reference;

        const_iterator(const std::pair<T* const, FieldMask>* _result)
          : result(_result), it(typename FMMap::const_iterator()), single(true)
        { }
        const_iterator(
            typename std::map<T*, FieldMask, COMPARATOR>::const_iterator _it)
          : result(nullptr), it(_it), single(false)
        { }
      public:
        const_iterator(const const_iterator& rhs)
          : result(nullptr), it(typename FMMap::const_iterator()),
            single(rhs.single)
        {
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
        }
        ~const_iterator(void) { }
      public:
        inline const_iterator& operator=(const const_iterator& rhs)
        {
          single = rhs.single;
          if (single)
            result = rhs.result;
          else
            it = rhs.it;
          return *this;
        }
      public:
        inline bool operator==(const const_iterator& rhs) const
        {
          if (single)
            return (result == rhs.result);
          else
            return (it == rhs.it);
        }
        inline bool operator!=(const const_iterator& rhs) const
        {
          if (single)
            return (result != rhs.result);
          else
            return (it != rhs.it);
        }
      public:
        inline const std::pair<T* const, FieldMask> operator*(void) const
        {
          if (single)
            return *result;
          else
            return *it;
        }
        inline const std::pair<T* const, FieldMask>* operator->(void) const
        {
          if (single)
            return result;
          else
            return &(*it);
        }
        inline const_iterator& operator++(/*prefix*/ void)
        {
          if (single)
            result = nullptr;
          else
            ++it;
          return *this;
        }
        inline const_iterator operator++(/*postfix*/ int)
        {
          const_iterator copy(*this);
          if (single)
            result = nullptr;
          else
            ++it;
          return copy;
        }
      private:
        const std::pair<T* const, FieldMask>* result;
        typename FMMap::const_iterator it;
        bool single;
      };
    public:
      template<AllocationLifetime LIFETIME>
      FieldMapView(const FieldMaskMap<T, LIFETIME, COMPARATOR>& map)
        : start(nullptr), stop(nullptr), full_size(map.size()),
          valid_fields(map.get_valid_mask())
      {
        if (full_size == 1)
        {
          const FieldMaskMap<T, LIFETIME, COMPARATOR>* ptr = &map;
          std::pair<T* const, FieldMask>* result = nullptr;
          static_assert(sizeof(ptr) == sizeof(result));
          memcpy(&result, &ptr, sizeof(result));
          start = const_iterator(result);
          stop = const_iterator(nullptr);
        }
        else if (full_size > 1)
        {
          start = const_iterator(map.entries.multi_entries->begin());
          stop = const_iterator(map.entries.multi_entries->end());
        }
      }
      FieldMapView(const FieldMapView& rhs) = default;
      FieldMapView& operator=(const FieldMapView& rhs) = default;
    public:
      inline size_t size(void) const { return full_size; }
      inline bool empty(void) const { return (start == stop); }
      inline const_iterator begin(void) const { return start; }
      inline const_iterator end(void) const { return stop; }
      inline const_iterator cbegin(void) const { return start; }
      inline const_iterator cend(void) const { return stop; }
      inline const_iterator find(T* key) const
      {
        if (full_size == 0)
          return stop;
        if (full_size == 1)
        {
          if (start->first == key)
            return start;
          else
            return stop;
        }
        const_iterator it = std::lower_bound(
            start, stop, key,
            [](const std::pair<const T*, FieldMask>& pair, const T* k) -> bool {
              return COMPARATOR()(pair.first, k);
            });
        if ((it != stop) && !COMPARATOR()(key, it->first))
          return it;
        else
          return stop;
      }
      inline const FieldMask& get_valid_mask(void) const
      {
        return valid_fields;
      }
      inline void compute_field_sets(
          FieldMask universe_mask,
          local::list<FieldSet<T*> >& output_sets) const;
    private:
      const_iterator start, stop;
      const size_t full_size;
      const FieldMask& valid_fields;
    };

    // Create some insantiations of these templates in the lifetime namespaces
    namespace local {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, TASK_LOCAL_LIFETIME, COMPARATOR>;
    }  // namespace local
    namespace op {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, OPERATION_LIFETIME, COMPARATOR>;
    }  // namespace op
    namespace ctx {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, CONTEXT_LIFETIME, COMPARATOR>;
    }  // namespace ctx
    namespace shrt {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, SHORT_LIFETIME, COMPARATOR>;
    }  // namespace shrt
    namespace lng {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, LONG_LIFETIME, COMPARATOR>;
    }  // namespace lng
    namespace rt {
      template<typename T, typename COMPARATOR = std::less<const T*> >
      using FieldMaskMap =
          Legion::Internal::FieldMaskMap<T, RUNTIME_LIFETIME, COMPARATOR>;
    }  // namespace rt

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/fieldmask_map.inl"

#endif  // __LEGION_FIELDMASK_MAP_H__
