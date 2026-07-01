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

// Included from physical.h - do not include this directly

// Useful for IDEs
#include "legion/tracing/physical.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Utility functions
    /////////////////////////////////////////////////////////////

    inline std::ostream& operator<<(std::ostream& out, ReplayableStatus status)
    {
      switch (status)
      {
        case REPLAYABLE:
          {
            out << "Yes";
            break;
          }
        case NOT_REPLAYABLE_BLOCKING:
          {
            out << "No (Blocking Call)";
            break;
          }
        case NOT_REPLAYABLE_CONSENSUS:
          {
            out << "No (Mapper Consensus)";
            break;
          }
        case NOT_REPLAYABLE_VIRTUAL:
          {
            out << "No (Virtual Mapping)";
            break;
          }
        case NOT_REPLAYABLE_REMOTE_SHARD:
          {
            out << "No (Remote Shard)";
            break;
          }
        case NOT_REPLAYABLE_NON_LEAF:
          {
            out << "No (Non-Leaf Task Variant)";
            break;
          }
        case NOT_REPLAYABLE_VARIABLE_RETURN:
          {
            out << "No (Variable Task Return Size)";
            break;
          }
        default:
          std::abort();
      }
      return out;
    }

    inline std::ostream& operator<<(std::ostream& out, IdempotencyStatus status)
    {
      switch (status)
      {
        case IDEMPOTENT:
          {
            out << "Yes";
            break;
          }
        case NOT_IDEMPOTENT_SUBSUMPTION:
          {
            out << "No (Preconditions Not Subsumed by Postconditions)";
            break;
          }
        case NOT_IDEMPOTENT_ANTIDEPENDENT:
          {
            out << "No (Postcondition Anti Dependent)";
            break;
          }
        case NOT_IDEMPOTENT_REMOTE_SHARD:
          {
            out << "No (Remote Shard)";
            break;
          }
        default:
          std::abort();
      }
      return out;
    }

  }  // namespace Internal
}  // namespace Legion
