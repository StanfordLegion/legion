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

#ifndef __LEGION_SMALL_VECTOR_H__
#define __LEGION_SMALL_VECTOR_H__

namespace Legion {
  namespace Internal {

    // A small pointer vector is a vector that stores 0 or 1 pointers
    // efficiently without any heap allocation but then expands to a
    // full vector when there are multiple elements
    template<typename T, bool SORTED>
    class SmallPointerVector {
    public:
      inline SmallPointerVector(void);
      SmallPointerVector(const SmallPointerVector& rhs) = delete;
      inline SmallPointerVector(SmallPointerVector&& rhs) noexcept;
      inline ~SmallPointerVector(void);
    public:
      SmallPointerVector& operator=(const SmallPointerVector& rhs) = delete;
      inline SmallPointerVector& operator=(SmallPointerVector&& rhs) noexcept;
    public:
      inline bool empty(void) const;
      inline size_t size(void) const;
      inline bool contains(T* value) const;
      inline void insert(T* value);
      inline bool erase(T* value);
      inline T* operator[](unsigned idx) const;
    private:
      inline std::vector<T*>& get_vector(void);
      inline const std::vector<T*>& get_vector(void) const;
    private:
      uintptr_t ptr;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/utilities/small_vector.inl"

#endif  // __LEGION_SMALL_VECTOR_H__
