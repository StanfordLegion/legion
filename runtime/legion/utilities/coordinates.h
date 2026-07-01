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

#ifndef __LEGION_COORDINATES_H__
#define __LEGION_COORDINATES_H__

#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /**
     * \struct ContextCoordinate
     * A struct that can uniquely identify an operation inside
     * the context of a parent task by the context_index which
     * is the number of the operation in the context, and the
     * index_point specifying which point in the case of an
     * index space operation
     */
    struct ContextCoordinate {
      inline ContextCoordinate(void) : context_index(SIZE_MAX) { }
      // Prevent trivally copying for serialize/deserialize
      inline ContextCoordinate(const ContextCoordinate& rhs)
        : context_index(rhs.context_index), index_point(rhs.index_point)
      { }
      inline ContextCoordinate(ContextCoordinate&& rhs) noexcept
        : context_index(rhs.context_index), index_point(rhs.index_point)
      { }
      inline ContextCoordinate(uint64_t index) : context_index(index) { }
      inline ContextCoordinate(uint64_t index, const DomainPoint& p)
        : context_index(index), index_point(p)
      { }
      inline ContextCoordinate& operator=(const ContextCoordinate& rhs)
      {
        context_index = rhs.context_index;
        index_point = rhs.index_point;
        return *this;
      }
      inline ContextCoordinate& operator=(ContextCoordinate&& rhs) noexcept
      {
        context_index = rhs.context_index;
        index_point = rhs.index_point;
        return *this;
      }
      inline bool operator==(const ContextCoordinate& rhs) const
      {
        return (
            (context_index == rhs.context_index) &&
            (index_point == rhs.index_point));
      }
      inline bool operator!=(const ContextCoordinate& rhs) const
      {
        return !((*this) == rhs);
      }
      inline bool operator<(const ContextCoordinate& rhs) const
      {
        if (context_index < rhs.context_index)
          return true;
        if (context_index > rhs.context_index)
          return false;
        return index_point < rhs.index_point;
      }
      inline void serialize(Serializer& rez) const
      {
        rez.serialize(context_index);
        rez.serialize(index_point);
      }
      inline void deserialize(Deserializer& derez)
      {
        derez.deserialize(context_index);
        derez.deserialize(index_point);
      }
      uint64_t context_index;
      DomainPoint index_point;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(
        std::ostream& os, const ContextCoordinate& coordinate)
    //--------------------------------------------------------------------------
    {
      os << '<' << coordinate.context_index << ',' << coordinate.index_point
         << '>';
      return os;
    }

    /**
     * \class TaskTreeCoordinates
     * This represents a stack of context coordinates at every level of the
     * task tree from the root down to the current task or operation
     */
    class TaskTreeCoordinates {
    public:
      bool operator==(const TaskTreeCoordinates& rhs) const;
      bool operator!=(const TaskTreeCoordinates& rhs) const;
      bool same_index_space(const TaskTreeCoordinates& rhs) const;
    public:
      inline void clear(void) { coordinates.clear(); }
      inline bool empty(void) const { return coordinates.empty(); }
      inline size_t size(void) const { return coordinates.size(); }
      inline ContextCoordinate& back(void) { return coordinates.back(); }
      inline const ContextCoordinate& back(void) const
      {
        return coordinates.back();
      }
      inline ContextCoordinate& operator[](unsigned idx)
      {
        return coordinates[idx];
      }
      inline const ContextCoordinate& operator[](unsigned idx) const
      {
        return coordinates[idx];
      }
      inline void emplace_back(ContextCoordinate&& coordinate)
      {
        coordinates.emplace_back(coordinate);
      }
      inline void reserve(size_t size) { coordinates.reserve(size); }
      inline void swap(TaskTreeCoordinates& coords)
      {
        coordinates.swap(coords.coordinates);
      }
    public:
      void serialize(Serializer& rez) const;
      void deserialize(Deserializer& derez);
    private:
      std::vector<ContextCoordinate> coordinates;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COORDINATES_H__
