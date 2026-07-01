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

#ifndef __LEGION_PREDICATE_H__
#define __LEGION_PREDICATE_H__

#include "legion/api/types.h"

namespace Legion {

  /**
   * \class Predicate
   * Predicate values are used for performing speculative
   * execution within an application.  They are lightweight handles
   * that can be passed around by value and stored in data
   * structures.  However, they should not escape the context of
   * the task in which they are created as they will be garbage
   * collected by the runtime.  Except for predicates with constant
   * value, all other predicates should be created by the runtime.
   */
  class Predicate : public Unserializable {
  public:
    static const Predicate TRUE_PRED;
    static const Predicate FALSE_PRED;
  public:
    Predicate(void);
    Predicate(const Predicate& p);
    Predicate(Predicate&& p) noexcept;
    explicit Predicate(bool value);
    ~Predicate(void);
  protected:
    FRIEND_ALL_RUNTIME_CLASSES
    Internal::PredicateImpl* impl;
    // Only the runtime should be allowed to make these
    explicit Predicate(Internal::PredicateImpl* impl);
  public:
    Predicate& operator=(const Predicate& p);
    Predicate& operator=(Predicate&& p) noexcept;
    inline bool operator==(const Predicate& p) const;
    inline bool operator<(const Predicate& p) const;
    inline bool operator!=(const Predicate& p) const;
    inline bool exists(void) const { return (impl != nullptr); }
  private:
    bool const_value;
  };

}  // namespace Legion

#include "legion/api/predicate.inl"

#endif  // __LEGION_PREDICATE_H__
