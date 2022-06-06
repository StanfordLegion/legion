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

// templated circular queue

#ifndef REALM_CIRC_QUEUE_H
#define REALM_CIRC_QUEUE_H

#include <iterator>

namespace Realm {

  // a circular queue is similar to a deque, except that it tries to avoid
  //  new/delete during construction and in normal operation by reusing
  //  entries - allocation is only needed if the current capacity is exhausted

  template <typename T, unsigned INTSIZE>
  class CircularQueueIterator;

  template <typename T, unsigned INTSIZE = 4>
  class CircularQueue {
  public:
    // default is to allocate just a few entries and then double whenever space runs out
    CircularQueue(size_t init_capacity = 0, int _growth_factor = -2);
    ~CircularQueue(void);

    typedef T ITEMTYPE;
    static const size_t ITEMSIZE = sizeof(T);

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

    void swap(CircularQueue<T, INTSIZE>& swap_with);

    template <unsigned INTSIZE2>
    void swap(CircularQueue<T, INTSIZE2>& swap_with);

    typedef CircularQueueIterator<T,INTSIZE> iterator;
    typedef CircularQueueIterator<T const,INTSIZE> const_iterator;

    iterator begin(void);
    iterator end(void);

    const_iterator begin(void) const;
    const_iterator end(void) const;

  protected:
    friend class CircularQueueIterator<T,INTSIZE>;

    T *item_ptr(char *base, size_t idx) const;
    const T *item_ptr(const char *base, size_t idx) const;

    // put this first for alignment goodness
    char   internal_buffer[ITEMSIZE * INTSIZE];
    char  *external_buffer;
    size_t current_size;  // number of elements currently in queue
    size_t max_size;      // size of underlying storage
    size_t head;          // index of first valid element (i.e. front)
    size_t tail;          // index of last valid element (i.e. back)
                          //  (when empty, tail = head - 1 (mod capacity) )
    int growth_factor;    // how to grow when more space is needed
                          // if > 0, an additive increase on current capacity
                          // if < 0, a multiplicative increase (i.e. new_cap = cap * abs(growth) )
  };

  template <typename T, unsigned INTSIZE>
  class CircularQueueIterator {
  public:
    // explicitly set iterator traits
    typedef std::forward_iterator_tag iterator_category;
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef T *pointer;
    typedef T& reference;

  protected:
    friend class CircularQueue<T,INTSIZE>;

    CircularQueueIterator(CircularQueue<T,INTSIZE> *_cq, size_t _pos, bool _at_end);

  public:
    CircularQueueIterator(void);
    CircularQueueIterator(const CircularQueueIterator<T,INTSIZE>& copy_from);

    CircularQueueIterator<T,INTSIZE>& operator=(const CircularQueueIterator<T,INTSIZE>& copy_from);

    bool operator==(const CircularQueueIterator<T,INTSIZE>& compare_to) const;
    bool operator!=(const CircularQueueIterator<T,INTSIZE>& compare_to) const;

    T operator*(void);
    const T *operator->(void);

    CircularQueueIterator<T,INTSIZE>& operator++(/*prefix*/);
    CircularQueueIterator<T,INTSIZE> operator++(int /*postfix*/);

  protected:
    CircularQueue<T,INTSIZE> *cq;
    size_t pos;
    bool at_end;
  };

}; // namespace Realm

#include "realm/circ_queue.inl"

#endif // ifndef REALM_CIRC_QUEUE_H

