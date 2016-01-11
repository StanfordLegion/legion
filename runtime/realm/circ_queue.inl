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

// nop, but helps IDEs
#include "circ_queue.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CircularQueue<T>

  template <typename T>
  inline CircularQueue<T>::CircularQueue(size_t init_capacity /*= 16*/,
					 int _growth_factor /*= -2*/)
    : current_size(0), max_size(init_capacity), head(1), tail(0)
    , growth_factor(_growth_factor)
    , storage(init_capacity)
  {}

  template <typename T>
  inline CircularQueue<T>::~CircularQueue(void)
  {}

  // using the standard STL contain methoder names and semantics

  template <typename T>
  inline bool CircularQueue<T>::empty(void) const
  {
    return (current_size == 0);
  }
   
  template <typename T>
  inline size_t CircularQueue<T>::size(void) const
  {
    return current_size;
  }
  
  template <typename T>
  inline size_t CircularQueue<T>::capacity(void) const
  {
    return max_size;
  }
  
  template <typename T>
  inline void CircularQueue<T>::reserve(size_t new_capacity)
  {
    if(new_capacity <= max_size)
      return;

    // easy case 1: queue is empty - just resize array and move head and tail back to beginning
    if(current_size == 0) {
      storage.resize(new_capacity);
      max_size = new_capacity;
      head = 1;
      tail = 0;
      return;
    }

    // easy case 2: data doesn't wrap around the end - can just resize the array, leave head
    //  and tail alone
    if(head <= tail) {
      storage.resize(new_capacity);
      max_size = new_capacity;
      return;
    }

    // uglier case 3: current data wraps around, so we have to copy things
    // a resize would result in two copies, so we'll just allocate a whole new
    // vector, copy stuff to it, and nuke the old one      
    std::vector<T> new_storage(new_capacity);

    // copy from head to end of old storage
    size_t count1 = max_size - head;
    for(size_t i = 0; i < count1; i++) new_storage[i] = storage[head + i];

    // now from beginning of storage to tail
    size_t count2 = tail + 1;
    assert((count1 + count2) == current_size);
    for(size_t i = 0; i < count2; i++) new_storage[i + count1] = storage[i];

    head = 0;
    tail = current_size - 1;
    max_size = new_capacity;
    storage.swap(new_storage);
  }

  template <typename T>
  inline void CircularQueue<T>::clear(void)
  {
    current_size = 0;
    head = 1;
    tail = 0;
  }

  template <typename T>
  inline T& CircularQueue<T>::front(void)
  {
    return storage[head];
  }

  template <typename T>
  inline const T& CircularQueue<T>::front(void) const
  {
    return storage[head];
  }

  template <typename T>
  inline void CircularQueue<T>::push_front(const T& val)
  {
    // check for full-ness
    if(current_size == max_size) {
      assert(growth_factor != 0);
      if(growth_factor > 0) 
	reserve(max_size + growth_factor); else
	reserve(max_size * -growth_factor);
    }

    if(head == 0)
      head = max_size - 1;
    else
      head -= 1;

    current_size += 1;
    storage[head] = val;
  }

  template <typename T>
  inline void CircularQueue<T>::pop_front(void)
  {
    assert(current_size > 0);

    if(head == (max_size - 1))
      head = 0;
    else
      head += 1;

    current_size -= 1;
  }

  template <typename T>
  inline T& CircularQueue<T>::back(void)
  {
    return storage[tail];
  }

  template <typename T>
  inline const T& CircularQueue<T>::back(void) const
  {
    return storage[tail];
  }

  template <typename T>
  inline void CircularQueue<T>::push_back(const T& val)
  {
    // check for full-ness
    if(current_size == max_size) {
      assert(growth_factor != 0);
      if(growth_factor > 0) 
	reserve(max_size + growth_factor); else
	reserve(max_size * -growth_factor);
    }

    if(tail == (max_size - 1))
      tail = 0;
    else
      tail += 1;

    current_size += 1;
    storage[tail] = val;
  }

  template <typename T>
  inline void CircularQueue<T>::pop_back(void)
  {
    assert(current_size > 0);

    if(tail == 0)
      tail = max_size - 1;
    else
      tail -= 1;

    current_size -= 1;
  }

}; // namespace Realm
