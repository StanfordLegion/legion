/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// templated intrusive linked lists

// nop, but helps IDEs
#include "realm/lists.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class IntrusiveListLink<T>

  template <typename T>
  IntrusiveListLink<T>::IntrusiveListLink(void)
    : next(0)
#ifdef DEBUG_REALM_LISTS
    , current_list(0)
#endif
  {}

  template <typename T>
  IntrusiveListLink<T>::~IntrusiveListLink(void)
  {
#ifdef DEBUG_REALM_LISTS
    // should not be deleted while in a list
    assert(next == 0);
    assert(current_list == 0);
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntrusiveList<T, LINK, LT>

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  inline IntrusiveList<T, LINK, LT>::IntrusiveList(void)
  {
    head.next = 0;
    lastlink = &head;
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  inline IntrusiveList<T, LINK, LT>::~IntrusiveList(void)
  {
#ifdef DEBUG_REALM_LISTS
    lock.lock();
    // lists should be empty when deleted
    assert(head.next == 0);
#endif
  }

  // "copying" a list is only allowed if both lists are empty (this is most
  //  useful when creating lists inside of containers)
  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  IntrusiveList<T, LINK, LT>::IntrusiveList(const IntrusiveList<T, LINK, LT>& copy_from)
  {
    assert(copy_from.empty());
    head.next = 0;
    lastlink = &head;
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  IntrusiveList<T, LINK, LT>& IntrusiveList<T, LINK, LT>::operator=(const IntrusiveList<T, LINK, LT>& copy_from)
  {
    assert(empty());
    assert(copy_from.empty());
    return *this;
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  template <typename LT2>
  inline void IntrusiveList<T, LINK, LT>::swap(IntrusiveList<T, LINK, LT2>& swap_with)
  {
    lock.lock();
    swap_with.lock.lock();
    std::swap(head.next, swap_with.head.next);
    // can't just swap the lastlinks because empty lists still need to link to
    //  themselves
    std::swap(lastlink, swap_with.lastlink);
    if(!head.next) lastlink = &head;
    if(!swap_with.head.next) swap_with.lastlink = &swap_with.head;
#ifdef DEBUG_REALM_LISTS
    // fix current_list references
    for(T *pos = head.next; pos; pos = (pos->*LINK).next) {
      assert((pos->*LINK).current_list == &swap_with);
      (pos->*LINK).current_list = this;
    }
    for(T *pos = swap_with.head.next; pos; pos = (pos->*LINK).next) {
      assert((pos->*LINK).current_list == this);
      (pos->*LINK).current_list = &swap_with;
    }
#endif
    swap_with.lock.unlock();
    lock.unlock();
  }

  // sucks the contents of 'take_from' into the end of the current list
  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  template <typename LT2>
  inline void IntrusiveList<T, LINK, LT>::absorb_append(IntrusiveList<T, LINK, LT2>& take_from)
  {
    lock.lock();
    take_from.lock.lock();
#ifdef DEBUG_REALM_LISTS
    for(T *pos = take_from.head.next; pos; pos = (pos->*LINK).next) {
      assert((pos->*LINK).current_list == &take_from);
      (pos->*LINK).current_list = this;
    }
#endif
    if(take_from.head.next != 0) {
      lastlink->next = take_from.head.next;
      lastlink = take_from.lastlink;
      take_from.head.next = 0;
      take_from.lastlink = &take_from.head;
    }
    take_from.lock.unlock();
    lock.unlock();
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  inline bool IntrusiveList<T, LINK, LT>::empty(void) const
  {
    // no lock taken here because it's not thread-safe even with a lock
    return(head.next == 0);
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  inline void IntrusiveList<T, LINK, LT>::push_back(T *new_entry)
  {
    lock.lock();
#ifdef DEBUG_REALM_LISTS
    assert((new_entry->*LINK).current_list == 0);
    (new_entry->*LINK).current_list = this;
    assert((new_entry->*LINK).next == 0);
#endif
    lastlink->next = new_entry;
    lastlink = &(new_entry->*LINK);
    lock.unlock();
  }

  template <typename T, IntrusiveListLink<T> T::*LINK, typename LT>
  inline T *IntrusiveList<T, LINK, LT>::pop_front(void)
  {
    T *popped = 0;
    lock.lock();
    if(head.next) {
      popped = head.next;
#ifdef DEBUG_REALM_LISTS
      assert((popped->*LINK).current_list == this);
      (popped->*LINK).current_list = 0;
#endif
      head.next = (popped->*LINK).next;
      (popped->*LINK).next = 0;
      if(!head.next) lastlink = &head;
    }
    lock.unlock();
    return popped;
  }

#if 0
  class IntrusiveList {
  public:
    typedef T ITEMTYPE;
    typedef LT LOCKTYPE;

    IntrusiveList(void);
    ~IntrusiveList(void);

    template <typename LT2>
    void swap(IntrusiveList<T, LINK, LT2>& swap_with);

    void append(T *new_entry);
    void prepend(T *new_entry);

    bool empty(void) const;

    T *front(void) const;
    T *pop_front(void);

    mutable LT lock;
    IntrusiveListLink<T> head;
    IntrusiveListLink<T> *lastlink;
  };
#endif
}; // namespace Realm
