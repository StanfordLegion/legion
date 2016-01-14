/* Copyright 2016 Stanford University, NVIDIA Corporation
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

// templated circular queue

#ifndef REALM_CIRC_QUEUE_H
#define REALM_CIRC_QUEUE_H

#include <vector>

namespace Realm {

  // a circular queue is similar to a deque, except that it tries to avoid
  //  new/delete during normal operation by reusing entries - allocation is
  //  only needed if the current capacity is exhausted
  //
  // currently this used a std::vector for the internal storage, which only works
  //  properly if T is a POD type
  // TODO: switch to "raw" storage and call constructors/destructors of T appropriately
  template <typename T>
  class CircularQueue {
  public:
    // default is to allocate just a few entries and then double whenever space runs out
    CircularQueue(size_t init_capacity = 16, int _growth_factor = -2);
    ~CircularQueue(void);

    typedef T ITEMTYPE;

    // using the standard STL contain methoder names and semantics
    bool empty(void) const;
    size_t size(void) const;
    size_t capacity(void) const;

    void reserve(size_t new_capacity);
    void clear(void);

    T& front(void);
    const T& front(void) const;
    void push_front(const T& val);
    void pop_front(void);

    T& back(void);
    const T& back(void) const;
    void push_back(const T& val);
    void pop_back(void);

  protected:
    size_t current_size;  // number of elements currently in queue
    size_t max_size;      // size of underlying storage
    size_t head;          // index of first valid element (i.e. front)
    size_t tail;          // index of last valid element (i.e. back)
                          //  (when empty, tail = head - 1 (mod capacity) )
    int growth_factor;    // how to grow when more space is needed
                          // if > 0, an additive increase on current capacity
                          // if < 0, a multiplicative increase (i.e. new_cap = cap * abs(growth) )

    std::vector<T> storage;
  };

}; // namespace Realm

#include "circ_queue.inl"

#endif // ifndef REALM_CIRC_QUEUE_H

