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

// nop, but helps IDEs
#include "realm/circ_queue.h"

#include <cassert>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class CircularQueue<T, INTSIZE>

  template <typename T, unsigned INTSIZE>
  inline CircularQueue<T,INTSIZE>::CircularQueue(size_t init_capacity /*= 0*/,
					 int _growth_factor /*= -2*/)
    : external_buffer(0), current_size(0), max_size(INTSIZE), head(1), tail(0)
    , growth_factor(_growth_factor)
  {
    if(init_capacity > INTSIZE)
      external_buffer = new char[ITEMSIZE * init_capacity];
  }

  template <typename T, unsigned INTSIZE>
  inline CircularQueue<T,INTSIZE>::~CircularQueue(void)
  {
    clear();
  }

  // using the standard STL contain methoder names and semantics

  template <typename T, unsigned INTSIZE>
  inline bool CircularQueue<T,INTSIZE>::empty(void) const
  {
    return (current_size == 0);
  }
   
  template <typename T, unsigned INTSIZE>
  inline size_t CircularQueue<T,INTSIZE>::size(void) const
  {
    return current_size;
  }
  
  template <typename T, unsigned INTSIZE>
  inline size_t CircularQueue<T,INTSIZE>::capacity(void) const
  {
    return max_size;
  }
  
  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::reserve(size_t new_capacity)
  {
    // only allow growth for now
    if(new_capacity <= max_size)
      return;

    // easy case 1: queue is empty - just resize array and move head and tail back to beginning
    if(current_size == 0) {
      if(external_buffer)
	delete[] external_buffer;

      max_size = new_capacity;
      external_buffer = new char[ITEMSIZE * new_capacity];

      head = 1;
      tail = 0;
      return;
    }

    // easy case 2: data doesn't wrap around the end - a single copy will do 
    if(head <= tail) {
      char *new_buffer = new char[ITEMSIZE * new_capacity];
      if(external_buffer) {
	std::copy(item_ptr(external_buffer, head), item_ptr(external_buffer, tail + 1),
		  item_ptr(new_buffer, head));
	delete[] external_buffer;
      } else {
	std::copy(item_ptr(internal_buffer, head), item_ptr(internal_buffer, tail + 1),
		  item_ptr(new_buffer, head));
      }
      external_buffer = new_buffer;
      max_size = new_capacity;
      return;
    }

    // uglier case 3: current data wraps around, so we have to copy things
    //  in two pieces
    char *new_buffer = new char[ITEMSIZE * new_capacity];
    
    if(external_buffer) {
      std::copy(item_ptr(external_buffer, head), item_ptr(external_buffer, max_size),
		item_ptr(new_buffer, 0));
      std::copy(item_ptr(external_buffer, 0), item_ptr(external_buffer, tail + 1),
		item_ptr(new_buffer, max_size - head));
      delete[] external_buffer;
    } else {
      std::copy(item_ptr(internal_buffer, head), item_ptr(internal_buffer, max_size),
		item_ptr(new_buffer, 0));
      std::copy(item_ptr(internal_buffer, 0), item_ptr(internal_buffer, tail + 1),
		item_ptr(new_buffer, max_size - head));
    }
    external_buffer = new_buffer;
    
    head = 0;
    tail = current_size - 1;
    max_size = new_capacity;
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::clear(void)
  {
    // destruct any valid entries
    if(current_size) {
      char *old_buffer = external_buffer ? external_buffer : internal_buffer;
      if(head <= tail) {
	for(size_t i = head; i <= tail; i++)
	  item_ptr(old_buffer, i)->~T();
      } else {
	for(size_t i = head; i < max_size; i++)
	  item_ptr(old_buffer, i)->~T();
	for(size_t i = 0; i <= tail; i++)
	  item_ptr(old_buffer, i)->~T();
      }
    }
    if(external_buffer) {
      delete[] external_buffer;
      external_buffer = 0;
    }

    current_size = INTSIZE;
    head = 1;
    tail = 0;
  }

  template <typename T, unsigned INTSIZE>
  inline T& CircularQueue<T,INTSIZE>::front(void)
  {
    return *(item_ptr(external_buffer ? external_buffer : internal_buffer,
		      head));
  }

  template <typename T, unsigned INTSIZE>
  inline const T& CircularQueue<T,INTSIZE>::front(void) const
  {
    return *(item_ptr(external_buffer ? external_buffer : internal_buffer,
		      head));
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::push_front(const T& val)
  {
    // check for full-ness
    if(current_size == max_size) {
      assert(growth_factor != 0);
      if(growth_factor > 0)
	reserve(max_size + growth_factor);
      else if(max_size > 0)
	reserve(max_size * -growth_factor);
      else
	reserve(8);  // gotta start somewhere
    }

    if(head == 0)
      head = max_size - 1;
    else
      head -= 1;

    current_size += 1;
    // use (placement) copy constructor
    T *ptr = item_ptr(external_buffer ? external_buffer : internal_buffer,
		      head);
    new(ptr) T(val);
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::pop_front(void)
  {
    assert(current_size > 0);

    // destruct existing entry
    T *ptr = item_ptr(external_buffer ? external_buffer : internal_buffer,
		      head);
    ptr->~T();

    if(head == (max_size - 1))
      head = 0;
    else
      head += 1;

    current_size -= 1;
  }

  template <typename T, unsigned INTSIZE>
  inline T& CircularQueue<T,INTSIZE>::back(void)
  {
    return *(item_ptr(external_buffer ? external_buffer : internal_buffer,
		      tail));
  }

  template <typename T, unsigned INTSIZE>
  inline const T& CircularQueue<T,INTSIZE>::back(void) const
  {
    return *(item_ptr(external_buffer ? external_buffer : internal_buffer,
		      tail));
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::push_back(const T& val)
  {
    // check for full-ness
    if(current_size == max_size) {
      assert(growth_factor != 0);
      if(growth_factor > 0)
	reserve(max_size + growth_factor);
      else if(max_size > 0)
	reserve(max_size * -growth_factor);
      else
	reserve(8);  // gotta start somewhere
    }

    if(tail == (max_size - 1))
      tail = 0;
    else
      tail += 1;

    current_size += 1;
    // use (placement) copy constructor
    T *ptr = item_ptr(external_buffer ? external_buffer : internal_buffer,
		      tail);
    new(ptr) T(val);
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::pop_back(void)
  {
    assert(current_size > 0);

    // destruct existing entry
    T *ptr = item_ptr(external_buffer ? external_buffer : internal_buffer,
		      tail);
    ptr->~T();

    if(tail == 0)
      tail = max_size - 1;
    else
      tail -= 1;

    current_size -= 1;
  }

  template <typename T, unsigned INTSIZE>
  inline void CircularQueue<T,INTSIZE>::swap(CircularQueue<T, INTSIZE>& swap_with)
  {
    // most things can be swapped, but valid data in the internal_buffer adds complexity
    if((current_size > 0) && (external_buffer == 0)) {
      if((swap_with.current_size > 0) && (swap_with.external_buffer == 0)) {
	// both sides have valid data - yuck
	assert(0);
      } else {
	// copy items from *this to swap_with
	size_t p = head;
	while(true) {
	  // copy construct and then destroy old instance
	  T *to_ptr = swap_with.item_ptr(swap_with.internal_buffer, p);
	  T *from_ptr = item_ptr(internal_buffer, p);
	  new(to_ptr) T(*from_ptr);
	  from_ptr->~T();
	  if(p == tail) break;
	  p++;
	  if(p == max_size) p = 0;
	}
      }
    } else {
      if((swap_with.current_size > 0) && (swap_with.external_buffer == 0)) {
	// copy items from swap_with to *this
	size_t p = swap_with.head;
	while(true) {
	  // copy construct and then destroy old instance
	  T *to_ptr = item_ptr(internal_buffer, p);
	  T *from_ptr = swap_with.item_ptr(swap_with.internal_buffer, p);
	  new(to_ptr) T(*from_ptr);
	  from_ptr->~T();
	  if(p == swap_with.tail) break;
	  p++;
	  if(p == swap_with.max_size) p = 0;
	}
      } else {
	// do nothing
      }
    }

    std::swap(external_buffer, swap_with.external_buffer);
    std::swap(current_size, swap_with.current_size);
    std::swap(max_size, swap_with.max_size);
    std::swap(head, swap_with.head);
    std::swap(tail, swap_with.tail);
    // don't swap growth factor
  }
    
  template <typename T, unsigned INTSIZE>
  template <unsigned INTSIZE2>
  inline void CircularQueue<T,INTSIZE>::swap(CircularQueue<T, INTSIZE2>& swap_with) 
  {
    // not yet implemented
    assert(0);
  }

  template <typename T, unsigned INTSIZE>
  inline typename CircularQueue<T,INTSIZE>::iterator CircularQueue<T,INTSIZE>::begin(void)
  {
    return iterator(this, head, (current_size == 0));
  }

  template <typename T, unsigned INTSIZE>
  inline typename CircularQueue<T,INTSIZE>::iterator CircularQueue<T,INTSIZE>::end(void)
  {
    return iterator(this, 0, true);
  }

  template <typename T, unsigned INTSIZE>
  inline typename CircularQueue<T,INTSIZE>::const_iterator CircularQueue<T,INTSIZE>::begin(void) const
  {
    return const_iterator(this, head, (current_size == 0));
  }

  template <typename T, unsigned INTSIZE>
  inline typename CircularQueue<T,INTSIZE>::const_iterator CircularQueue<T,INTSIZE>::end(void) const
  {
    return const_iterator(this, 0, true);
  }

  template <typename T, unsigned INTSIZE>
  T *CircularQueue<T,INTSIZE>::item_ptr(char *base, size_t idx) const
  {
    return reinterpret_cast<T *>(base + (idx * ITEMSIZE));
  }

  template <typename T, unsigned INTSIZE>
  const T *CircularQueue<T,INTSIZE>::item_ptr(const char *base, size_t idx) const
  {
    return reinterpret_cast<const T *>(base + (idx * ITEMSIZE));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class CircularQueueIterator<T, INTSIZE>

  template <typename T, unsigned INTSIZE>
  inline CircularQueueIterator<T,INTSIZE>::CircularQueueIterator(CircularQueue<T,INTSIZE> *_cq, size_t _pos, bool _at_end)
    : cq(_cq)
    , pos(_pos)
    , at_end(_at_end)
  {}

  template <typename T, unsigned INTSIZE>
  inline CircularQueueIterator<T,INTSIZE>::CircularQueueIterator(void)
    : cq(0)
    , pos(0)
    , at_end(true)
  {}

  template <typename T, unsigned INTSIZE>
  inline CircularQueueIterator<T,INTSIZE>::CircularQueueIterator(const CircularQueueIterator<T,INTSIZE>& copy_from)
    : cq(copy_from.cq)
    , pos(copy_from.pos)
    , at_end(copy_from.at_end)
  {}

  template <typename T, unsigned INTSIZE>
  inline CircularQueueIterator<T,INTSIZE>& CircularQueueIterator<T,INTSIZE>::operator=(const CircularQueueIterator<T,INTSIZE>& copy_from)
  {
    cq = copy_from.cq;
    pos = copy_from.pos;
    at_end = copy_from.at_end;
    return *this;
  }

  template <typename T, unsigned INTSIZE>
  inline bool CircularQueueIterator<T,INTSIZE>::operator==(const CircularQueueIterator<T,INTSIZE>& compare_to) const
  {
    // two at-end iterators always match
    if(at_end && compare_to.at_end) return true;
    // otherwise all fields must match
    return((cq == compare_to.cq) &&
	   (pos == compare_to.pos) &&
	   (at_end == compare_to.at_end));
  }

  template <typename T, unsigned INTSIZE>
  inline bool CircularQueueIterator<T,INTSIZE>::operator!=(const CircularQueueIterator<T,INTSIZE>& compare_to) const
  {
    return !(*this == compare_to);
  }
	
  template <typename T, unsigned INTSIZE>
  inline T CircularQueueIterator<T,INTSIZE>::operator*(void)
  {
    assert(cq && !at_end);
    return *(cq->item_ptr(cq->external_buffer ? cq->external_buffer : cq->internal_buffer,
			  pos));
  }

  template <typename T, unsigned INTSIZE>
  inline const T *CircularQueueIterator<T,INTSIZE>::operator->(void)
  {
    assert(cq && !at_end);
    return (cq->item_ptr(cq->external_buffer ? cq->external_buffer : cq->internal_buffer,
			 pos));
  }
	
  template <typename T, unsigned INTSIZE>
  CircularQueueIterator<T,INTSIZE>& CircularQueueIterator<T,INTSIZE>::operator++(/*prefix*/)
  {
    assert(cq && !at_end);
    if(pos == cq->tail) {
      at_end = true;
    } else {
      pos++;
      // handle wrap-around case
      if(pos == cq->max_size)
	pos = 0;
    }
    return *this;
  }

  template <typename T, unsigned INTSIZE>
  CircularQueueIterator<T,INTSIZE> CircularQueueIterator<T,INTSIZE>::operator++(int /*postfix*/)
  {
    CircularQueueIterator<T,INTSIZE> orig(*this);
    assert(cq && !at_end);
    if(pos == cq->tail) {
      at_end = true;
    } else {
      pos++;
      // handle wrap-around case
      if(pos == cq->max_size)
	pos = 0;
    }
    return orig;
  }


}; // namespace Realm
