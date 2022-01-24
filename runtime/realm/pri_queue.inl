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

// templated priority queue

// nop, but helps IDEs
#include "realm/pri_queue.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class PriorityQueue<T, LT>

  template <typename T, typename LT>
  inline PriorityQueue<T, LT>::PriorityQueue(void)
    : highest_priority (PRI_NEG_INF)
    , entries_in_queue (0)
  {
  }

  template <typename T, typename LT>
  inline PriorityQueue<T, LT>::~PriorityQueue(void)
  {
    // STL cleans everything up
  }

  // two ways to add an item -
  //  1) add it to the end of the list at that priority (i.e. FIFO order)
  //  2) "unget" adds it to the front of the list (i.e. LIFO order)
  template <typename T, typename LT>
  inline void PriorityQueue<T, LT>::put(T item, 
					priority_t priority, 
					bool add_to_back /*= true*/)
  {
    // step 1: clamp the priority to the "finite" range
    if(priority > PRI_MAX_FINITE)
      priority = PRI_MAX_FINITE;
    else if(priority < PRI_MIN_FINITE)
      priority = PRI_MIN_FINITE;

    // increase the entry count, if we care
    if(entries_in_queue)
      (*entries_in_queue) += 1;

    // step 2: take the lock
    lock.lock();

    // step 3: if this is higher priority than anybody else, do notification callbacks
    if(priority > highest_priority) {
      // remember the original one in case this gets immediately grabbed
      priority_t orig_highest = highest_priority;

      // now update the highest before we do notifications
      highest_priority = priority;

      // now the notifications, if any
      if(perform_notifications(item, priority)) {
	// item was taken, so restore original highest priority and exit
	highest_priority = orig_highest;
	lock.unlock();
	if(entries_in_queue)
	  (*entries_in_queue) -= 1;
	return;
      }
    }

    // get the right deque (this will create one if needed)
    std::deque<T>& dq = queue[-priority]; // remember negation...

    // and add the item
    if(add_to_back)
      dq.push_back(item);
    else
      dq.push_front(item);

    // all done
    lock.unlock();
  }

  // getting an item is always from the front of the list and can be filtered to
  //  ignore things that aren't above a specified priority
  // the priority of the retrieved item (if any) is returned in *item_priority
  template <typename T, typename LT>
  inline T PriorityQueue<T, LT>::get(priority_t *item_priority, 
				     priority_t higher_than /*= PRI_NEG_INF*/)
  {
    // body is protected by lock
    lock.lock();

    // empty queue - early out
    if(queue.empty()) {
      lock.unlock();
      return 0; // TODO - EMPTY_VAL
    }

    typename std::map<priority_t, std::deque<T> >::iterator it = queue.begin();
    priority_t priority = -(it->first);

    // not interesting enough?
    if(priority <= higher_than) {
      lock.unlock();
      return 0; // TODO - EMPTY_VAL
    };

    // take item off front
    T item = it->second.front();
    it->second.pop_front();

    // if list is now empty, remove from the queue and adjust highest_priority
    if(it->second.empty()) {
      queue.erase(it);
      highest_priority = (queue.empty() ?
			    PRI_NEG_INF :
			    -(queue.begin()->first));
    }

    // release lock and then return result
    lock.unlock();

    // decrease the entry count, if we care
    if(entries_in_queue)
      (*entries_in_queue) -= 1;

    if(item_priority)
      *item_priority = priority;
    return item;
  }

  // peek is like get, but doesn't remove the element (this is only really useful when
  //  you don't have multiple getters)
  template <typename T, typename LT>
  inline T PriorityQueue<T, LT>::peek(priority_t *item_priority,
				      priority_t higher_than /*= PRI_NEG_INF*/) const
  {
    // body is protected by lock
    lock.lock();

    // empty queue - early out
    if(queue.empty()) {
      lock.unlock();
      return 0; // TODO - EMPTY_VAL
    }

    typename std::map<priority_t, std::deque<T> >::const_iterator it = queue.begin();
    priority_t priority = -(it->first);

    // not interesting enough?
    if(priority <= higher_than) {
      lock.unlock();
      return 0; // TODO - EMPTY_VAL
    };

    // peek at item on front
    T item = it->second.front();

    // release lock and then return result
    lock.unlock();

    if(item_priority)
      *item_priority = priority;
    return item;
  }

  // similarly, the empty-ness query can also ignore things below a certain priority
  // this call is lock-free (and is again of questionable utility with multiple readers)
  template <typename T, typename LT>
  inline bool PriorityQueue<T, LT>::empty(priority_t higher_than /*= PRI_NEG_INF*/) const
  {
    return(highest_priority <= higher_than);
  }

  // adds (or modifies) a subscription - only items above the specified priority will
  //  result in callbacks
  template <typename T, typename LT>
  inline void PriorityQueue<T, LT>::add_subscription(NotificationCallback *callback,
						     priority_t higher_than /*= PRI_NEG_INF*/)
  {
    // just take lock and update subscription map
    lock.lock();
    subscriptions[callback] = higher_than;
    lock.unlock();
  }
  
  template <typename T, typename LT>
  inline void PriorityQueue<T, LT>::remove_subscription(NotificationCallback *callback)
  {
    // just take lock and erase subscription map entry (if present)
    lock.lock();
    subscriptions.erase(callback);
    lock.unlock();
  }

  // helper that performs notifications for a new item - returns true if a callback
  //  consumes the item
  template <typename T, typename LT>
  inline bool PriorityQueue<T, LT>::perform_notifications(T item, priority_t item_priority)
  {
    // lock already held by caller

    for(typename std::map<NotificationCallback *, priority_t>::const_iterator it = subscriptions.begin();
	it != subscriptions.end();
	it++) {
      // skip if this isn't interesting to this callback
      if(item_priority <= it->second)
	continue;

      // do callback - a return of true means the item was consumed (so we shouldn't do
      //  any more callbacks
      if(it->first->item_available(item, item_priority))
	return true;
    }

    // got through all the callbacks and nobody wanted it
    return false;
  }

  template <typename T, typename LT>
  inline void PriorityQueue<T, LT>::set_gauge(ProfilingGauges::AbsoluteRangeGauge<int> *new_gauge)
  {
    entries_in_queue = new_gauge;
  }

}; // namespace Realm
