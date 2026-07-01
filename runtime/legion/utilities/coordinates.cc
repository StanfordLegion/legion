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

#include "legion/utilities/coordinates.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Task Tree Coordinates
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    bool TaskTreeCoordinates::operator==(const TaskTreeCoordinates& rhs) const
    //--------------------------------------------------------------------------
    {
      if (coordinates.size() != rhs.size())
        return false;
      for (unsigned idx = 0; idx < coordinates.size(); idx++)
        if (coordinates[idx] != rhs[idx])
          return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool TaskTreeCoordinates::operator!=(const TaskTreeCoordinates& rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    bool TaskTreeCoordinates::same_index_space(
        const TaskTreeCoordinates& rhs) const
    //--------------------------------------------------------------------------
    {
      if (coordinates.size() != rhs.size())
        return false;
      // Must the same coordinates for all but the last level
      for (unsigned idx = 0; idx < (coordinates.size() - 1); idx++)
        if (coordinates[idx] != rhs[idx])
          return false;
      // Last leve just needs to have the same context index
      if (coordinates.back().context_index != rhs.back().context_index)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskTreeCoordinates::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(coordinates.size());
      for (const ContextCoordinate& coordinate : coordinates)
        coordinate.serialize(rez);
    }

    //--------------------------------------------------------------------------
    void TaskTreeCoordinates::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_coordinates;
      derez.deserialize(num_coordinates);
      coordinates.resize(num_coordinates);
      for (ContextCoordinate& coordinate : coordinates)
        coordinate.deserialize(derez);
    }

  }  // namespace Internal
}  // namespace Legion
