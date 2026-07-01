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

// Included from small_vector.h - do not include this directly

// Useful for IDEs
#include "legion/utilities/small_vector.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline SmallPointerVector<T, SORTED>::SmallPointerVector(void) : ptr(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline SmallPointerVector<T, SORTED>::SmallPointerVector(
        SmallPointerVector&& rhs) noexcept
      : ptr(rhs.ptr)
    //--------------------------------------------------------------------------
    {
      rhs.ptr = 0;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline SmallPointerVector<T, SORTED>::~SmallPointerVector(void)
    //--------------------------------------------------------------------------
    {
      if (ptr & 0x1)
        delete &get_vector();
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline SmallPointerVector<T, SORTED>&
        SmallPointerVector<T, SORTED>::operator=(
            SmallPointerVector&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if (ptr & 0x1)
        delete &get_vector();
      ptr = rhs.ptr;
      rhs.ptr = 0;
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline bool SmallPointerVector<T, SORTED>::empty(void) const
    //--------------------------------------------------------------------------
    {
      return (ptr == 0);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline size_t SmallPointerVector<T, SORTED>::size(void) const
    //--------------------------------------------------------------------------
    {
      if (ptr == 0)
        return 0;
      else if (ptr & 0x1)
        return get_vector().size();
      else
        return 1;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline bool SmallPointerVector<T, SORTED>::contains(T* value) const
    //--------------------------------------------------------------------------
    {
      if (ptr == 0)
        return false;
      else if (ptr & 0x1)
      {
        const std::vector<T*>& vector = get_vector();
        if (!SORTED)
        {
          for (T* const & element : vector)
            if (element == value)
              return true;
          return false;
        }
        else
          return std::binary_search(vector.begin(), vector.end(), value);
      }
      else
        return (ptr == reinterpret_cast<uintptr_t>(value));
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline void SmallPointerVector<T, SORTED>::insert(T* value)
    //--------------------------------------------------------------------------
    {
      if (ptr == 0)
      {
        ptr = reinterpret_cast<uintptr_t>(value);
        legion_assert(!(ptr & 0x1));
      }
      else if (ptr & 0x1)
      {
        std::vector<T*>& vector = get_vector();
        vector.emplace_back(value);
        if (SORTED)
          std::sort(vector.begin(), vector.end());
      }
      else
      {
        std::vector<T*>* new_vector = new std::vector<T*>(2);
        new_vector->at(0) = reinterpret_cast<T*>(ptr);
        new_vector->at(1) = value;
        if (SORTED)
          std::sort(new_vector->begin(), new_vector->end());
        ptr = reinterpret_cast<uintptr_t>(new_vector);
        ptr |= 0x1;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline bool SmallPointerVector<T, SORTED>::erase(T* value)
    //--------------------------------------------------------------------------
    {
      if (ptr == 0)
        return false;
      else if (ptr & 0x1)
      {
        std::vector<T*>& vector = get_vector();
        if (SORTED)
        {
          typename std::vector<T*>::iterator finder =
              std::lower_bound(vector.begin(), vector.end(), value);
          if ((finder != vector.end()) && (*finder == value))
          {
            vector.erase(finder);
            if (vector.size() == 1)
            {
              ptr = reinterpret_cast<uintptr_t>(vector.back());
              legion_assert(!(ptr & 0x1));
              delete &vector;
            }
            return true;
          }
          else
            return false;
        }
        else
        {
          for (typename std::vector<T*>::iterator it = vector.begin();
               it != vector.end(); it++)
          {
            if ((*it) != value)
              continue;
            vector.erase(it);
            if (vector.size() == 1)
            {
              ptr = reinterpret_cast<uintptr_t>(vector.back());
              legion_assert(!(ptr & 0x1));
              delete &vector;
            }
            return true;
          }
          return false;
        }
      }
      else
      {
        if (ptr == reinterpret_cast<uintptr_t>(value))
        {
          ptr = 0;
          return true;
        }
        else
          return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline T* SmallPointerVector<T, SORTED>::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (ptr & 0x1)
        return get_vector().at(idx);
      else if ((idx == 0) && (ptr != 0))
        return reinterpret_cast<T*>(ptr);
      else
        std::abort();
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline std::vector<T*>& SmallPointerVector<T, SORTED>::get_vector(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(ptr & 0x1);
      return *reinterpret_cast<std::vector<T*>*>(ptr ^ 0x1);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool SORTED>
    inline const std::vector<T*>& SmallPointerVector<T, SORTED>::get_vector(
        void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(ptr & 0x1);
      return *reinterpret_cast<const std::vector<T*>*>(ptr ^ 0x1);
    }

  }  // namespace Internal
}  // namespace Legion
