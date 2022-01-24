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

#ifndef REALM_PRI_QUEUE_H
#define REALM_PRI_QUEUE_H

#include <deque>
#include <map>

#include "realm/sampling.h"

namespace Realm {

  // a priority queue is templated on the type of thing stored in the queue
  // and on the kind of lock used to protect it (e.g. you can use a DummyLock
  // if mutual exclusion is provided outside of these calls)
  template <typename T, typename LT>
  class PriorityQueue {
  public:
    PriorityQueue(void);
    ~PriorityQueue(void);

    typedef T ITEMTYPE;

    // we used most of the signed integer range for priorities - we do borrow a 
    //  few of the extreme values to make sure we have "infinity" and "negative infinity"
    //  and that we don't run into problems with -INT_MIN
    // attempts to enqueue an item with a priority outside the "finite" range will result
    //  in the priority being silently clamped to the finite range
    typedef int priority_t;
    static const priority_t PRI_MAX_FINITE = INT_MAX - 1;
    static const priority_t PRI_MIN_FINITE = -(INT_MAX - 1);
    static const priority_t PRI_POS_INF = PRI_MAX_FINITE + 1;
    static const priority_t PRI_NEG_INF = PRI_MIN_FINITE - 1;

    // two ways to add an item -
    //  1) add it to the end of the list at that priority (i.e. FIFO order)
    //  2) "unget" adds it to the front of the list (i.e. LIFO order)
    void put(T item, priority_t priority, bool add_to_back = true);

    // getting an item is always from the front of the list and can be filtered to
    //  ignore things that aren't above a specified priority
    // the priority of the retrieved item (if any) is returned in *item_priority
    T get(priority_t *item_priority, priority_t higher_than = PRI_NEG_INF);

    // peek is like get, but doesn't remove the element (this is only really useful when
    //  you don't have multiple getters)
    T peek(priority_t *item_priority, priority_t higher_than = PRI_NEG_INF) const;

    // similarly, the empty-ness query can also ignore things below a certain priority
    // this call is lock-free (and is again of questionable utility with multiple readers)
    bool empty(priority_t higher_than = PRI_NEG_INF) const;

    // it is possible to subscribe to queue updates - notifications are sent when
    //  a new item arrives at a higher priority level than what is already available
    //  and offers the item for immediate retrieval - if a callback returns true, the
    //  item is considered to have been consumed
    // note that these callbacks are performed with the queue's lock HELD - watch out for
    //  deadlock scenarios
    class NotificationCallback {
    public:
      virtual bool item_available(T item, priority_t item_priority) = 0;
    };

    // adds (or modifies) a subscription - only items above the specified priority will
    //  result in callbacks
    void add_subscription(NotificationCallback *callback, priority_t higher_than = PRI_NEG_INF);
    void remove_subscription(NotificationCallback *callback);

    void set_gauge(ProfilingGauges::AbsoluteRangeGauge<int> *new_gauge);

  protected:
    // helper that performs notifications for a new item - returns true if a callback
    //  consumes the item
    bool perform_notifications(T item, priority_t item_priority);

    // 'highest_priority' may be read without the lock held, but only written with the lock
    priority_t highest_priority;

    // this lock protects everything else
    mutable LT lock;

    // the actual queue - priorities are negated here to that queue.begin() gives us the
    //  "highest" priority
    std::map<priority_t, std::deque<T> > queue;

    // notification subscriptions
    std::map<NotificationCallback *, priority_t> subscriptions;
    
    ProfilingGauges::AbsoluteRangeGauge<int> *entries_in_queue;

    template <typename T2, typename LT2>
    friend std::ostream& operator<<(std::ostream& os, const PriorityQueue<T2, LT2>& pq);
  };

  template <typename T, typename LT>
  std::ostream& operator<<(std::ostream& os, const PriorityQueue<T, LT>& pq)
  {
    pq.lock.lock();
    os << "PQ{\n";
    for(typename std::map<typename PriorityQueue<T, LT>::priority_t, std::deque<T> >::const_iterator it = pq.queue.begin();
	it != pq.queue.end();
	++it) {
      os << "  [" << -(it->first) << "]: ";
      typename std::deque<T>::const_iterator it2 = it->second.begin();
      assert(it2 != it->second.end());
      os << ((const void *)(*it2));
      while((++it2) != it->second.end())
	os << ", " << ((const void *)(*it2));
      os << "\n";
    }
    os << "}\n";
    pq.lock.unlock();
    return os;
  }

}; // namespace Realm

#include "realm/pri_queue.inl"

#endif // ifndef REALM_PRI_QUEUE_H

