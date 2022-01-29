/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include <iostream>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class IntrusiveListLink<T>

  template <typename T>
  inline IntrusiveListLink<T>::IntrusiveListLink(void)
    : next(0)
#ifdef DEBUG_REALM_LISTS
    , current_list(0)
#endif
  {}

  template <typename T>
  inline IntrusiveListLink<T>::~IntrusiveListLink(void)
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

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline IntrusiveList<T, LINK, LT>::IntrusiveList(void)
  {
    head.next = 0;
    lastlink = &head;
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
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
  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline IntrusiveList<T, LINK, LT>::IntrusiveList(const IntrusiveList<T, LINK, LT>& copy_from)
  {
    assert(copy_from.empty());
    head.next = 0;
    lastlink = &head;
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline IntrusiveList<T, LINK, LT>& IntrusiveList<T, LINK, LT>::operator=(const IntrusiveList<T, LINK, LT>& copy_from)
  {
    assert(empty());
    assert(copy_from.empty());
    return *this;
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
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
    for(T *pos = head.next; pos; pos = REALM_PMTA_DEREF(pos,LINK).next) {
      assert(REALM_PMTA_DEREF(pos,LINK).current_list == &swap_with);
      REALM_PMTA_DEREF(pos,LINK).current_list = this;
    }
    for(T *pos = swap_with.head.next; pos; pos = REALM_PMTA_DEREF(pos,LINK).next) {
      assert(REALM_PMTA_DEREF(pos,LINK).current_list == this);
      REALM_PMTA_DEREF(pos,LINK).current_list = &swap_with;
    }
#endif
    swap_with.lock.unlock();
    lock.unlock();
  }

  // sucks the contents of 'take_from' into the end of the current list
  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  template <typename LT2>
  inline void IntrusiveList<T, LINK, LT>::absorb_append(IntrusiveList<T, LINK, LT2>& take_from)
  {
    lock.lock();
    take_from.lock.lock();
#ifdef DEBUG_REALM_LISTS
    for(T *pos = take_from.head.next; pos; pos = REALM_PMTA_DEREF(pos,LINK).next) {
      assert(REALM_PMTA_DEREF(pos,LINK).current_list == &take_from);
      REALM_PMTA_DEREF(pos,LINK).current_list = this;
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

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline bool IntrusiveList<T, LINK, LT>::empty(void) const
  {
#ifndef TSAN_ENABLED
    // no lock taken here because it's not thread-safe even with a lock
    return(head.next == 0);
#else
    // with thread-sanitizer, we have to take the lock to suppress the warning
    lock.lock();
    bool retval = (head.next == 0);
    lock.unlock();
    return retval;
#endif
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline void IntrusiveList<T, LINK, LT>::push_back(T *new_entry)
  {
    lock.lock();
#ifdef DEBUG_REALM_LISTS
    assert(REALM_PMTA_DEREF(new_entry,LINK).current_list == 0);
    REALM_PMTA_DEREF(new_entry,LINK).current_list = this;
    assert(REALM_PMTA_DEREF(new_entry,LINK).next == 0);
#endif
    lastlink->next = new_entry;
    lastlink = &REALM_PMTA_DEREF(new_entry,LINK);
    lock.unlock();
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline T *IntrusiveList<T, LINK, LT>::pop_front(void)
  {
    T *popped = 0;
    lock.lock();
    if(head.next) {
      popped = head.next;
#ifdef DEBUG_REALM_LISTS
      assert(REALM_PMTA_DEREF(popped,LINK).current_list == this);
      REALM_PMTA_DEREF(popped,LINK).current_list = 0;
#endif
      head.next = REALM_PMTA_DEREF(popped,LINK).next;
      REALM_PMTA_DEREF(popped,LINK).next = 0;
      if(!head.next) lastlink = &head;
    }
    lock.unlock();
    return popped;
  }

  template <typename T, REALM_PMTA_DECL(T,IntrusiveListLink<T>,LINK), typename LT>
  inline size_t IntrusiveList<T, LINK, LT>::erase(T *entry)
  {
    size_t count = 0;
    lock.lock();
    IntrusiveListLink<T> *pos = &head;
    while(pos->next) {
      if(pos->next == entry) {
	// remove this from the list
	count++;
	T *nextnext = REALM_PMTA_DEREF(pos->next,LINK).next;
#ifdef DEBUG_REALM_LISTS
	REALM_PMTA_DEREF(pos->next,LINK).next = 0;
	REALM_PMTA_DEREF(pos->next,LINK).current_list = 0;
#endif
	pos->next = nextnext;
      } else {
	// move to next entry
	pos = &REALM_PMTA_DEREF(pos->next,LINK);
      }
    }
    // make sure the lastlink is still right
    lastlink = pos;
    lock.unlock();
    return count;
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntrusivePriorityListLink<T>

  template <typename T>
  inline IntrusivePriorityListLink<T>::IntrusivePriorityListLink(void)
    : next_within_pri(0)
    , lastlink_within_pri(0)
    , next_lower_pri(0)
#ifdef DEBUG_REALM_LISTS
    , current_list(0)
#endif
  {}

  template <typename T>
  inline IntrusivePriorityListLink<T>::~IntrusivePriorityListLink(void)
  {
#ifdef DEBUG_REALM_LISTS
    // should not be deleted while in a list
    assert(next_within_pri == 0);
    assert(next_lower_pri == 0);
    assert(current_list == 0);
#endif
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class IntrusivePriorityList<T, PT, LINK, PRI, LT>

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  IntrusivePriorityList<T, PT, LINK, PRI, LT>::IntrusivePriorityList(void)
    : head(0)
  {}

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  IntrusivePriorityList<T, PT, LINK, PRI, LT>::~IntrusivePriorityList(void)
  {
#ifdef DEBUG_REALM_LISTS
    lock.lock();
    // list should be empty when deleted
    assert(head == 0);
#endif
  }

  // "copying" a list is only allowed if both lists are empty (this is most
  //  useful when creating lists inside of containers)
  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  IntrusivePriorityList<T, PT, LINK, PRI, LT>::IntrusivePriorityList(const IntrusivePriorityList<T, PT, LINK, PRI, LT>& copy_from)
    : head(0)
  {
    assert(copy_from.head == 0);
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  IntrusivePriorityList<T, PT, LINK, PRI, LT>& IntrusivePriorityList<T, PT, LINK, PRI, LT>::operator=(const IntrusivePriorityList<T, PT, LINK, PRI, LT>& copy_from)
  {
    assert(head == 0);
    assert(copy_from.head == 0);
    return *this;
  } 

#ifdef DEBUG_REALM_LISTS
  template <typename T, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK)>
  static void fixup_current_list_pointers(T *head, void *from, void *to)
  {
    T *nextp = head;
    while(nextp != 0) {
      T *cur = nextp;
      nextp = REALM_PMTA_DEREF(nextp,LINK).next_lower_pri;
      while(cur != 0) {
	assert(REALM_PMTA_DEREF(cur,LINK).current_list == from);
	REALM_PMTA_DEREF(cur,LINK).current_list = to;
	cur = REALM_PMTA_DEREF(cur,LINK).next_within_pri;
      }
    }
  }
#endif

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  template <typename LT2>
  void IntrusivePriorityList<T, PT, LINK, PRI, LT>::swap(IntrusivePriorityList<T, PT, LINK, PRI, LT2>& swap_with)
  {
    lock.lock();
    swap_with.lock.lock();
    std::swap(head, swap_with.head);
#ifdef DEBUG_REALM_LISTS
    // fix current_list pointers
    fixup_current_list_pointers<T, LINK>(head, &swap_with, this);
    fixup_current_list_pointers<T, LINK>(swap_with.head, this, &swap_with);
#endif
    swap_with.lock.unlock();
    lock.unlock();
  }

    // sucks the contents of 'take_from' into the end of the current list
  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  template <typename LT2>
  void IntrusivePriorityList<T, PT, LINK, PRI, LT>::absorb_append(IntrusivePriorityList<T, PT, LINK, PRI, LT2>& take_from)
  {
    lock.lock();
    take_from.lock.lock();
#ifdef DEBUG_REALM_LISTS
    size_t exp_size = size() + take_from.size();
    // fix current_list pointers before we tear lists apart
    fixup_current_list_pointers<T, LINK>(take_from.head, &take_from, this);
#endif
    T **curdst = &head;
    T *cursrc = take_from.head;
    while(cursrc) {
      // have something to merge

      // first, skip over any tiers that are higher priority in destination
      while(*curdst && (REALM_PMTA_DEREF(*curdst,PRI) > REALM_PMTA_DEREF(cursrc,PRI)))
	curdst = &(REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri);

      // if no equal/lower priority stuff left in destination, absorb rest
      //  of entries in one go
      if(!*curdst) {
	*curdst = cursrc;
	break;
      }

      if(REALM_PMTA_DEREF(*curdst,PRI) == REALM_PMTA_DEREF(cursrc,PRI)) {
	// equal priority - merge tiers
	T *nextsrc = REALM_PMTA_DEREF(cursrc,LINK).next_lower_pri;
	*(REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri) = cursrc;
	REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri = REALM_PMTA_DEREF(cursrc,LINK).lastlink_within_pri;
#ifdef DEBUG_REALM_LISTS
	// clean up now-unused pointers to make debugging easier
	REALM_PMTA_DEREF(cursrc,LINK).next_lower_pri = 0;
	REALM_PMTA_DEREF(cursrc,LINK).lastlink_within_pri = 0;
#endif
	curdst = &(REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri);
	cursrc = nextsrc;
      } else {
	// new priority level for destination - insert tier as is
	T *nextsrc = REALM_PMTA_DEREF(cursrc,LINK).next_lower_pri;
	REALM_PMTA_DEREF(cursrc,LINK).next_lower_pri = *curdst;
	*curdst = cursrc;
	curdst = &(REALM_PMTA_DEREF(cursrc,LINK).next_lower_pri);
	cursrc = nextsrc;
      }
    }
    take_from.head = 0;
#ifdef DEBUG_REALM_LISTS
    size_t act_size = size();
    assert(exp_size == act_size);
#endif
    take_from.lock.unlock();
    lock.unlock();
  }

  // places new item at front or back of its priority level
  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  void IntrusivePriorityList<T, PT, LINK, PRI, LT>::push_back(T *new_entry)
  {
    lock.lock();
    lock.unlock();
#ifdef DEBUG_REALM_LISTS
    // entry being added should be unentangled
    assert(REALM_PMTA_DEREF(new_entry,LINK).next_within_pri == 0);
    assert(REALM_PMTA_DEREF(new_entry,LINK).lastlink_within_pri == 0);
    assert(REALM_PMTA_DEREF(new_entry,LINK).next_lower_pri == 0);
    size_t exp_size = size() + 1;
#endif
    // scan ahead to find right priority level to insert at
    T **curdst = &head;
    while(*curdst && (REALM_PMTA_DEREF(*curdst,PRI) > REALM_PMTA_DEREF(new_entry,PRI)))
      curdst = &(REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri);

    if(*curdst && (REALM_PMTA_DEREF(*curdst,PRI) == REALM_PMTA_DEREF(new_entry,PRI))) {
      // insert at back of current tier
      REALM_PMTA_DEREF(new_entry,LINK).next_within_pri = 0;
      // lastlink_with_pri and next_lower_pri are don't cares - we are not the first in our tier
      *(REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri) = new_entry;
      REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri = &(REALM_PMTA_DEREF(new_entry,LINK).next_within_pri);
    } else {
      // start new tier ahead of whatever's left (if anything)
      REALM_PMTA_DEREF(new_entry,LINK).next_within_pri = 0;
      REALM_PMTA_DEREF(new_entry,LINK).lastlink_within_pri = &(REALM_PMTA_DEREF(new_entry,LINK).next_within_pri);
      REALM_PMTA_DEREF(new_entry,LINK).next_lower_pri = *curdst;
      *curdst = new_entry;
    }
#ifdef DEBUG_REALM_LISTS
    assert(REALM_PMTA_DEREF(new_entry,LINK).current_list == 0);
    REALM_PMTA_DEREF(new_entry,LINK).current_list = this;
    size_t act_size = size();
    assert(exp_size == act_size);
#endif
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  void IntrusivePriorityList<T, PT, LINK, PRI, LT>::push_front(T *new_entry)
  {
    lock.lock();
    lock.unlock();
#ifdef DEBUG_REALM_LISTS
    // entry being added should be unentangled
    assert(REALM_PMTA_DEREF(new_entry,LINK).next_within_pri == 0);
    assert(REALM_PMTA_DEREF(new_entry,LINK).lastlink_within_pri == 0);
    assert(REALM_PMTA_DEREF(new_entry,LINK).next_lower_pri == 0);
    size_t exp_size = size() + 1;
#endif
    // scan ahead to find right priority level to insert at
    T **curdst = &head;
    while(*curdst && (REALM_PMTA_DEREF(*curdst,PRI) > REALM_PMTA_DEREF(new_entry,PRI)))
      curdst = &(REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri);

    if(*curdst && (REALM_PMTA_DEREF(*curdst,PRI) == REALM_PMTA_DEREF(new_entry,PRI))) {
      // insert at front of current tier
      REALM_PMTA_DEREF(new_entry,LINK).next_within_pri = *curdst;
      REALM_PMTA_DEREF(new_entry,LINK).lastlink_within_pri = REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri;
      REALM_PMTA_DEREF(new_entry,LINK).next_lower_pri = REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri;
#ifdef DEBUG_REALM_LISTS
      // clean up now-unused pointers to make debugging easier
      REALM_PMTA_DEREF(*curdst,LINK).lastlink_within_pri = 0;
      REALM_PMTA_DEREF(*curdst,LINK).next_lower_pri = 0;
#endif
      *curdst = new_entry;
    } else {
      // start new tier ahead of whatever's left (if anything)
      REALM_PMTA_DEREF(new_entry,LINK).next_within_pri = 0;
      REALM_PMTA_DEREF(new_entry,LINK).lastlink_within_pri = &(REALM_PMTA_DEREF(new_entry,LINK).next_within_pri);
      REALM_PMTA_DEREF(new_entry,LINK).next_lower_pri = *curdst;
      *curdst = new_entry;
    }
#ifdef DEBUG_REALM_LISTS
    assert(REALM_PMTA_DEREF(new_entry,LINK).current_list == 0);
    REALM_PMTA_DEREF(new_entry,LINK).current_list = this;
    size_t act_size = size();
    assert(exp_size == act_size);
#endif
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  inline bool IntrusivePriorityList<T, PT, LINK, PRI, LT>::empty(void) const
  {
#ifndef TSAN_ENABLED
    // no lock taken here because it's not thread-safe even with a lock
    return(head == 0);
#else
    // with thread-sanitizer, we have to take the lock to suppress the warning
    lock.lock();
    bool retval = (head == 0);
    lock.unlock();
    return retval;
#endif
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  inline bool IntrusivePriorityList<T, PT, LINK, PRI, LT>::empty(PT min_priority) const
  {
    // have to take lock here in order to test priority safely
    lock.lock();
    bool found = (head != 0) && (REALM_PMTA_DEREF(head,PRI) >= min_priority);
    lock.unlock();
    return !found;
  }

  // this call isn't thread safe - the pointer returned may be accessed only
  //  if the caller can guarantee no concurrent pops have occurred
  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  inline T *IntrusivePriorityList<T, PT, LINK, PRI, LT>::front(void) const
  {
    return head;
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  T *IntrusivePriorityList<T, PT, LINK, PRI, LT>::pop_front(void)
  {
    lock.lock();
#ifdef DEBUG_REALM_LISTS
    size_t exp_size = size();
#endif
    T *popped = head;
    if(popped) {
#ifdef DEBUG_REALM_LISTS
      exp_size--;
#endif
      if(REALM_PMTA_DEREF(popped,LINK).next_within_pri != 0) {
	// others in tier - next one becomes head
        head = REALM_PMTA_DEREF(popped,LINK).next_within_pri;
	REALM_PMTA_DEREF(head,LINK).lastlink_within_pri = REALM_PMTA_DEREF(popped,LINK).lastlink_within_pri;
        REALM_PMTA_DEREF(head,LINK).next_lower_pri = REALM_PMTA_DEREF(popped,LINK).next_lower_pri;
      } else {
	// was only one in tier - point head at next priority tier
        head = REALM_PMTA_DEREF(popped,LINK).next_lower_pri;
      }
#ifdef DEBUG_REALM_LISTS
      assert(REALM_PMTA_DEREF(popped,LINK).current_list == this);
      REALM_PMTA_DEREF(popped,LINK).current_list = 0;
      // clean up now-unused pointers to make debugging easier
      REALM_PMTA_DEREF(popped,LINK).next_within_pri = 0;
      REALM_PMTA_DEREF(popped,LINK).lastlink_within_pri = 0;
      REALM_PMTA_DEREF(popped,LINK).next_lower_pri = 0;
#endif
    }
#ifdef DEBUG_REALM_LISTS
    size_t act_size = size();
    assert(exp_size == act_size);
#endif
    lock.unlock();
    return popped;
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  T *IntrusivePriorityList<T, PT, LINK, PRI, LT>::pop_front(PT min_priority)
  {
    lock.lock();
#ifdef DEBUG_REALM_LISTS
    size_t exp_size = size();
#endif
    T *popped = ((head && (REALM_PMTA_DEREF(head,PRI) >= min_priority)) ? head : 0);
    if(popped) {
#ifdef DEBUG_REALM_LISTS
      exp_size--;
#endif
      if(REALM_PMTA_DEREF(popped,LINK).next_within_pri != 0) {
	// others in tier - next one becomes head
        head = REALM_PMTA_DEREF(popped,LINK).next_within_pri;
	REALM_PMTA_DEREF(head,LINK).lastlink_within_pri = REALM_PMTA_DEREF(popped,LINK).lastlink_within_pri;
        REALM_PMTA_DEREF(head,LINK).next_lower_pri = REALM_PMTA_DEREF(popped,LINK).next_lower_pri;
      } else {
	// was only one in tier - point head at next priority tier
        head = REALM_PMTA_DEREF(popped,LINK).next_lower_pri;
      }
#ifdef DEBUG_REALM_LISTS
      assert(REALM_PMTA_DEREF(popped,LINK).current_list == this);
      REALM_PMTA_DEREF(popped,LINK).current_list = 0;
      // clean up now-unused pointers to make debugging easier
      REALM_PMTA_DEREF(popped,LINK).next_within_pri = 0;
      REALM_PMTA_DEREF(popped,LINK).lastlink_within_pri = 0;
      REALM_PMTA_DEREF(popped,LINK).next_lower_pri = 0;
#endif
    }
#ifdef DEBUG_REALM_LISTS
    size_t act_size = size();
    assert(exp_size == act_size);
#endif
    lock.unlock();
    return popped;
  }

  // calls callback for each element in list
  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  template <typename CALLABLE>
  inline void IntrusivePriorityList<T, PT, LINK, PRI, LT>::foreach(CALLABLE& cb)
  {
    lock.lock();
    T *cur = head;
    while(cur) {
      cb(cur);
      T *cur2 = cur;
      while(REALM_PMTA_DEREF(cur2,LINK).next_within_pri != 0) {
        cur2 = REALM_PMTA_DEREF(cur2,LINK).next_within_pri;
	cb(cur2);
      }
      cur = REALM_PMTA_DEREF(cur,LINK).next_lower_pri;
    }
    lock.unlock();
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  size_t IntrusivePriorityList<T, PT, LINK, PRI, LT>::size(void) const
  {
    size_t count = 0;
    lock.lock();
    const T *cur = head;
    while(cur) {
      count++;
      const T *cur2 = cur;
      while(REALM_PMTA_DEREF(cur2,LINK).next_within_pri != 0) {
        cur2 = REALM_PMTA_DEREF(cur2,LINK).next_within_pri;
#ifdef DEBUG_REALM_LISTS
	assert(REALM_PMTA_DEREF(cur2,PRI) == REALM_PMTA_DEREF(cur,PRI));
	assert(REALM_PMTA_DEREF(cur2,LINK).lastlink_within_pri == 0);
	assert(REALM_PMTA_DEREF(cur2,LINK).next_lower_pri == 0);
#endif
	count++;
      }
#ifdef DEBUG_REALM_LISTS
      assert(REALM_PMTA_DEREF(cur,LINK).lastlink_within_pri == &(REALM_PMTA_DEREF(cur2,LINK).next_within_pri));
#endif
      cur = REALM_PMTA_DEREF(cur,LINK).next_lower_pri;
    }
    lock.unlock();
    return count;
  }

  template <typename T, typename PT, REALM_PMTA_DECL(T,IntrusivePriorityListLink<T>,LINK), REALM_PMTA_DECL(T,PT,PRI), typename LT>
  std::ostream& operator<<(std::ostream& os, const IntrusivePriorityList<T, PT, LINK, PRI, LT>& to_print)
  {
    to_print.lock.lock();
    os << "PriList(" << ((void *)&to_print) << ") {\n";
    const T *cur = to_print.head;
    while(cur) {
      os << "  [" << (REALM_PMTA_DEREF(cur,PRI)) << "]: " << ((void *)cur);
      const T *cur2 = REALM_PMTA_DEREF(cur,LINK).next_within_pri;
      while(cur2) {
	os << ", " << ((void *)cur2);
	cur2 = REALM_PMTA_DEREF(cur2,LINK).next_within_pri;
      }
      os << "\n";
      cur = REALM_PMTA_DEREF(cur,LINK).next_lower_pri;
    }
    os << "}\n";
    to_print.lock.unlock();
    return os;
  }


}; // namespace Realm
