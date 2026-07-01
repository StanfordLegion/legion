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

// Included from individual.h - do not include this directly

// Useful for IDEs
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    inline IndividualView* LogicalView::as_individual_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_individual_view());
      return static_cast<IndividualView*>(const_cast<LogicalView*>(this));
    }

    //--------------------------------------------------------------------------
    inline bool SpaceView::has_local_precondition(
        PhysicalUser* user, const RegionUsage& next_user,
        IndexSpaceExpression* expr, const UniqueID op_id, const unsigned index,
        const bool next_covers, const bool copy_user, bool* dominates) const
    //--------------------------------------------------------------------------
    {
      // We order these tests in a entirely based on cost

      // Different region requirements of the same operation
      // Copies from different region requirements though still
      // need to wait on each other correctly
      if ((op_id == user->op_id) && (index != user->index) &&
          (!copy_user || !user->copy_user))
        return false;
      // Now do a dependence test for privilege non-interference
      // Only reductions here are copy reductions which we know do not interfere
      DependenceType dt =
          check_dependence_type<false, false>(user->usage, next_user);
      switch (dt)
      {
        case LEGION_NO_DEPENDENCE:
        case LEGION_ATOMIC_DEPENDENCE:
        case LEGION_SIMULTANEOUS_DEPENDENCE:
          return false;
        case LEGION_TRUE_DEPENDENCE:
        case LEGION_ANTI_DEPENDENCE:
          break;
        default:
          std::abort();  // should never get here
      }
      if (!next_covers)
      {
        if (!user->covers)
        {
          // Neither one covers so we actually need to do the
          // full intersection test and see if next covers
          IndexSpaceExpression* overlap =
              runtime->intersect_index_spaces(expr, user->expr);
          if (overlap->is_empty())
            return false;
        }
        // We don't allow any user that doesn't fully cover the
        // expression to dominate anything. It's hard to guarantee
        // correctness without this. Think very carefully if you
        // plan to change this!
        if (dominates != nullptr)
          *dominates = false;
      }
      return true;
    }

  }  // namespace Internal
}  // namespace Legion
