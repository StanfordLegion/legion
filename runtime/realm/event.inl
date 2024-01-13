/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// events for Realm

// nop, but helps IDEs
#include "realm/event.h"

#include "realm/serialize.h"
TYPE_IS_SERIALIZABLE(Realm::Event);
TYPE_IS_SERIALIZABLE(Realm::UserEvent);
TYPE_IS_SERIALIZABLE(Realm::Barrier);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Event

  inline bool Event::operator<(const Event& rhs) const
  {
    return id < rhs.id;
  }

  inline bool Event::operator==(const Event& rhs) const
  {
    return id == rhs.id;
  }

  inline bool Event::operator!=(const Event& rhs) const
  {
    return id != rhs.id;
  }

  inline bool Event::exists(void) const
  {
    return id != 0;
  }

  inline std::ostream& operator<<(std::ostream& os, Event e)
  {
    return os << std::hex << e.id << std::dec;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CompletionQueue

  inline bool CompletionQueue::operator<(const CompletionQueue& rhs) const
  {
    return id < rhs.id;
  }

  inline bool CompletionQueue::operator==(const CompletionQueue& rhs) const
  {
    return id == rhs.id;
  }

  inline bool CompletionQueue::operator!=(const CompletionQueue& rhs) const
  {
    return id != rhs.id;
  }

  inline bool CompletionQueue::exists(void) const
  {
    return id != 0;
  }

  inline std::ostream& operator<<(std::ostream& os, CompletionQueue e)
  {
    return os << std::hex << e.id << std::dec;
  }

  /*static*/ inline Event Event::merge_events(const std::set<Event> &wait_for) {
    // Highly inefficient, only left here for compatibility, consider using
    // std::vector or array instead
    std::vector<Event> events(wait_for.begin(), wait_for.end());
    return merge_events(span<const Event>(events));
  }
  /*static*/ inline Event
  Event::merge_events(const span<const Event> &wait_for) {
    return merge_events(wait_for.data(), wait_for.size());
  }
  /*static*/ inline Event
  Event::merge_events_ignorefaults(const span<const Event> &wait_for) {
    return merge_events_ignorefaults(wait_for.data(), wait_for.size());
  }
  /*static*/ inline Event
  Event::merge_events_ignorefaults(const std::set<Event> &wait_for) {
    // Highly inefficient, only left here for compatibility, consider using
    // std::vector or array instead
    std::vector<Event> events(wait_for.begin(), wait_for.end());
    return merge_events_ignorefaults(span<const Event>(events));
  }
  /*static*/ inline void
  Event::advise_event_ordering(const span<Event> &happens_before,
                               Event happens_after,
                               bool all_must_trigger /*= true */) {
    advise_event_ordering(happens_before.data(), happens_before.size(),
                          happens_after, all_must_trigger);
  }

}; // namespace Realm  
