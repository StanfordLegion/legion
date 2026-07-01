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

#ifndef __LEGION_PRIVILEGES_H__
#define __LEGION_PRIVILEGES_H__

#include "legion/api/requirements.h"

// Useful macros
#define IS_NO_ACCESS(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_NO_ACCESS)
#define IS_READ_ONLY(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_READ_PRIV)
#define HAS_READ(req) ((req).privilege & (LEGION_READ_PRIV | LEGION_REDUCE))
#define HAS_WRITE(req) ((req).privilege & (LEGION_WRITE_PRIV | LEGION_REDUCE))
#define IS_WRITE(req) ((req).privilege & LEGION_WRITE_PRIV)
#define IS_WRITE_ONLY(req) \
  (((req).privilege & LEGION_READ_WRITE) == LEGION_WRITE_PRIV)
#define IS_WRITE_DISCARD(req)                                             \
  (((req).privilege & (LEGION_WRITE_ONLY | LEGION_DISCARD_INPUT_MASK)) == \
   (LEGION_WRITE_PRIV | LEGION_DISCARD_INPUT_MASK))
#define IS_OUTPUT_DISCARD(req) \
  (((req).privilege & LEGION_DISCARD_OUTPUT_MASK) == LEGION_DISCARD_OUTPUT_MASK)
#define FILTER_DISCARD(req) \
  ((req).privilege & ~(LEGION_DISCARD_INPUT_MASK | LEGION_DISCARD_OUTPUT_MASK))
#define IS_COLLECTIVE(req) \
  (((req).prop & LEGION_COLLECTIVE_MASK) == LEGION_COLLECTIVE_MASK)
#define PRIV_ONLY(req) ((req).privilege & LEGION_READ_WRITE)
#define IS_REDUCE(req) (((req).privilege & LEGION_READ_WRITE) == LEGION_REDUCE)
#define IS_EXCLUSIVE(req) (((req).prop & LEGION_RELAXED) == LEGION_EXCLUSIVE)
#define IS_ATOMIC(req) (((req).prop & LEGION_RELAXED) == LEGION_ATOMIC)
#define IS_SIMULT(req) (((req).prop & LEGION_RELAXED) == LEGION_SIMULTANEOUS)
#define IS_RELAXED(req) (((req).prop & LEGION_RELAXED) == LEGION_RELAXED)

namespace Legion {
  namespace Internal {

    /**
     * \struct RegionUsage
     * A minimal structure for performing dependence analysis.
     */
    struct RegionUsage {
    public:
      RegionUsage(void)
        : privilege(LEGION_NO_ACCESS), prop(LEGION_EXCLUSIVE), redop(0)
      { }
      RegionUsage(PrivilegeMode p, CoherenceProperty c, ReductionOpID r)
        : privilege(p), prop(c), redop(r)
      { }
      RegionUsage(const RegionRequirement& req)
        : privilege(req.privilege), prop(req.prop), redop(req.redop)
      { }
    public:
      inline bool operator==(const RegionUsage& rhs) const
      {
        return (
            (privilege == rhs.privilege) && (prop == rhs.prop) &&
            (redop == rhs.redop));
      }
      inline bool operator!=(const RegionUsage& rhs) const
      {
        return !((*this) == rhs);
      }
    public:
      PrivilegeMode privilege;
      CoherenceProperty prop;
      ReductionOpID redop;
    };

    // The following two methods define the dependence analysis
    // for all of Legion.  Modifying them can have enormous
    // consequences on how programs execute.

    //--------------------------------------------------------------------------
    static inline DependenceType check_for_anti_dependence(
        const RegionUsage& u1, const RegionUsage& u2, DependenceType actual)
    //--------------------------------------------------------------------------
    {
      // Check for WAR or WAW with write-only
      if (IS_READ_ONLY(u1))
      {
        // We know at least req1 or req2 is a writers, so if req1 is not...
        legion_assert(HAS_WRITE(u2));
        return LEGION_ANTI_DEPENDENCE;
      }
      else
      {
        if (IS_WRITE_DISCARD(u2))
        {
          // WAW with a write-only
          return LEGION_ANTI_DEPENDENCE;
        }
        else
        {
          // This defaults to whatever the actual dependence is
          return actual;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<bool READ_DISCARD_EXCLUSIVE, bool REDUCTIONS_INTERFERE>
    static inline DependenceType check_dependence_type(
        const RegionUsage& u1, const RegionUsage& u2)
    //--------------------------------------------------------------------------
    {
      if (IS_READ_ONLY(u1) && IS_READ_ONLY(u2))
      {
        // Two readers are never a dependence unless the second is discarding
        if (READ_DISCARD_EXCLUSIVE && IS_OUTPUT_DISCARD(u2))
          return LEGION_TRUE_DEPENDENCE;
        else
          return LEGION_NO_DEPENDENCE;
      }
      else if (!REDUCTIONS_INTERFERE && IS_REDUCE(u1) && IS_REDUCE(u2))
      {
        // If they are the same kind of reduction, no dependence,
        // otherwise true dependence
        if (u1.redop == u2.redop)
        {
          // Exclusive and atomic coherence are effectively the same
          // thing in these contexts. Similarly simultaneous/relaxed
          // are also effectively the same thing for reductions.
          // However, mixing one of those "group modes" with the other
          // can result in races, so we don't allow that
          if (u1.prop != u2.prop)
          {
            const bool atomic1 = IS_EXCLUSIVE(u1) || IS_ATOMIC(u1);
            const bool atomic2 = IS_EXCLUSIVE(u2) || IS_ATOMIC(u2);
            if (atomic1 != atomic2)
              return LEGION_TRUE_DEPENDENCE;
          }
          return LEGION_NO_DEPENDENCE;
        }
        else
          return LEGION_TRUE_DEPENDENCE;
      }
      else
      {
        // Everything in here has at least one write
        legion_assert(HAS_WRITE(u1) || HAS_WRITE(u2));
        // If anything exclusive
        if (IS_EXCLUSIVE(u1) || IS_EXCLUSIVE(u2))
        {
          return check_for_anti_dependence(u1, u2, LEGION_TRUE_DEPENDENCE);
        }
        // Anything atomic (at least one is a write)
        else if (IS_ATOMIC(u1) || IS_ATOMIC(u2))
        {
          // If they're both atomics, return an atomic dependence
          if (IS_ATOMIC(u1) && IS_ATOMIC(u2))
          {
            return check_for_anti_dependence(u1, u2, LEGION_ATOMIC_DEPENDENCE);
          }
          // If the one that is not an atomic is a read, we're also ok
          // We still need a simultaneous dependence if we don't have an
          // actual dependence
          else if (
              (!IS_ATOMIC(u1) && IS_READ_ONLY(u1)) ||
              (!IS_ATOMIC(u2) && IS_READ_ONLY(u2)))
          {
            return LEGION_SIMULTANEOUS_DEPENDENCE;
          }
          // Everything else is a dependence
          return check_for_anti_dependence(u1, u2, LEGION_TRUE_DEPENDENCE);
        }
        // If either is simultaneous we have a simultaneous dependence
        else if (IS_SIMULT(u1) || IS_SIMULT(u2))
        {
          return LEGION_SIMULTANEOUS_DEPENDENCE;
        }
        else if (IS_RELAXED(u1) && IS_RELAXED(u2))
        {
          // TODO: Make this truly relaxed, right now it is the
          // same as simultaneous
          return LEGION_SIMULTANEOUS_DEPENDENCE;
          // This is what it should be: return NO_DEPENDENCE;
          // What needs to be done:
          // - RegionNode::update_valid_instances needs to allow multiple
          //               outstanding writers
          // - RegionNode needs to detect relaxed case and make copies from all
          //              relaxed instances to non-relaxed instance
        }
        // We should never make it here
        std::abort();
      }
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PRIVILEGES_H__
