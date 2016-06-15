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


#ifndef __RUNTIME_UTILITIES_H__
#define __RUNTIME_UTILITIES_H__

#include <iostream>
#include <string>

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdarg>

#include <map>
#include <vector>
#include <pthread.h>
#include <time.h>

#include <fcntl.h>

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

#include "realm/logging.h"
#include "realm/timers.h"

//#define DETAILED_TIMING

namespace LegionRuntime {
  // map old Logger::Category to new Realm::Logger
  namespace Logger {
    typedef Realm::Logger Category;
  };

#ifdef DEBUG_REALM
#define PTHREAD_SAFE_CALL(cmd)			\
	{					\
		int ret = (cmd);		\
		if (ret != 0) {			\
			fprintf(stderr,"PTHREAD error: %s = %d (%s)\n", #cmd, ret, strerror(ret));	\
			assert(false);		\
		}				\
	}
#else
#define PTHREAD_SAFE_CALL(cmd)			\
	(cmd);
#endif
  /**
   * A blocking lock that cannot be moved between nodes
   */
  class ImmovableLock {
  public:
    ImmovableLock(bool initialize = false) : mutex(NULL) { 
      if (initialize)
       init(); 
    }
    ImmovableLock(const ImmovableLock &other) : mutex(other.mutex) { }
  public:
    void init(void) {
      mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
      PTHREAD_SAFE_CALL(pthread_mutex_init(mutex,NULL));
    }
    void destroy(void) { 
      PTHREAD_SAFE_CALL(pthread_mutex_destroy(mutex));
      free(mutex);
      mutex = NULL;
    }
    void clear(void) {
      mutex = NULL;
    }
  public:
    void lock(void) {
      PTHREAD_SAFE_CALL(pthread_mutex_lock(mutex));
    }
    void unlock(void) {
      PTHREAD_SAFE_CALL(pthread_mutex_unlock(mutex));
    }
  public:
    bool operator<(const ImmovableLock &rhs) const
    { return (mutex < rhs.mutex); }
    bool operator==(const ImmovableLock &rhs) const
    { return (mutex == rhs.mutex); }
    ImmovableLock& operator=(const ImmovableLock &rhs)
    { mutex = rhs.mutex; return *this; }
  private:
    pthread_mutex_t *mutex;
  };

  // For backwards compatibility only
  class AutoLock {
  public:
    AutoLock(ImmovableLock &lock)
      : low_lock(lock)
    {
      low_lock.lock();
    }
  public:
    AutoLock(const AutoLock &rhs)
      : low_lock(rhs.low_lock)
    {
      // should never be called
      assert(false);
    }
    ~AutoLock(void)
    {
      low_lock.unlock();
    }
  public:
    AutoLock& operator=(const AutoLock &rhs)
    {
      // should never be called
      assert(false);
      return *this;
    }
  private:
    ImmovableLock &low_lock;
  };

  /**
   * A timer class for doing detailed timing analysis of applications
   * Implementation is specific to low level runtimes
   */
  namespace LowLevel {

    class UtilityBarrier {
    public:
      explicit UtilityBarrier(unsigned expected_arrivals)
        : remaining_arrivals(expected_arrivals),
          remaining_leaves(expected_arrivals)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_init(&inner_lock, NULL));
        PTHREAD_SAFE_CALL(pthread_cond_init(&inner_cond, NULL));
      }
      ~UtilityBarrier(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_destroy(&inner_lock));
        PTHREAD_SAFE_CALL(pthread_cond_destroy(&inner_cond));
      }
    public:
      bool arrive(void)
      {
        PTHREAD_SAFE_CALL(pthread_mutex_lock(&inner_lock));
        assert(remaining_arrivals > 0); 
        remaining_arrivals--;
        if (remaining_arrivals == 0)
        {
          PTHREAD_SAFE_CALL(pthread_cond_broadcast(&inner_cond));
        }
        else
        {
          PTHREAD_SAFE_CALL(pthread_cond_wait(&inner_cond, &inner_lock));
        }
        assert(remaining_leaves > 0);
        remaining_leaves--;
        bool last_owner = false;
        if (remaining_leaves == 0)
          last_owner = true;
        PTHREAD_SAFE_CALL(pthread_mutex_unlock(&inner_lock));
        return last_owner;
      }
    private:
      unsigned remaining_arrivals;
      unsigned remaining_leaves;
      pthread_mutex_t inner_lock;
      pthread_cond_t inner_cond;
    };

    template<typename ITEM>
    class Tracer {
    public:
      class TraceBlock {
      public:
        explicit TraceBlock(const TraceBlock &refblk)
	  : max_size(refblk.max_size), cur_size(0),
	    time_mult(refblk.time_mult), next(0)
	{
	  PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	  start_time = Realm::Clock::current_time();
	  items = new ITEM[max_size];
	}

	TraceBlock(size_t block_size, double exp_arrv_rate)
	  : max_size(block_size), cur_size(0), next(0)
	{
	  PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	  start_time = Realm::Clock::current_time();
	  items = new ITEM[max_size];

	  // set the time multiplier such that there'll likely be 2^31
	  //  time units by the time we fill the block, i.e.:
	  // (block_size / exp_arrv_rate) * mult = 2^31  -->
	  // mult = 2^31 * exp_arrv_rate / block_size
	  time_mult = 2147483648.0 * exp_arrv_rate / block_size;
	}
        pthread_mutex_t mutex;
	size_t max_size, cur_size;
	double start_time, time_mult;
	ITEM *items;
	TraceBlock *next;
      };

      static TraceBlock*& get_first_block(void)
      {
        static TraceBlock *first_block = NULL;
        return first_block;
      }

      static TraceBlock*& get_last_block(void)
      {
        static TraceBlock *last_block = NULL;
        return last_block;
      }

      static void init_trace(size_t block_size, double exp_arrv_rate)
      {
        get_first_block() = get_last_block() = new TraceBlock(block_size, exp_arrv_rate);
        assert(get_first_block() != NULL);
        assert(get_last_block() != NULL);
      }

      static inline ITEM& trace_item(void)
      {
        TraceBlock *block = get_last_block();
        assert(block != NULL);

	// loop until we can manage to reserve a real entry
	size_t my_index;
	while(1) {
	  my_index = __sync_fetch_and_add(&(block->cur_size), 1);
	  if(my_index < block->max_size) break; // victory!

	  // no space - case 1: another block has already been chained on -
	  //  try that one
	  if(block->next) {
	    block = block->next;
	    continue;
	  }

	  // case 2: try to add a block on ourselves - take the current block's
	  //  mutex and make sure somebody else didn't do it while we waited
	  pthread_mutex_lock(&(block->mutex));
	  if(block->next) {
	    pthread_mutex_unlock(&(block->mutex));
	    block = block->next;
	    continue;
	  }

	  // the next pointer is still null, and we hold the lock, so we
	  //  do the allocation of the next block
	  TraceBlock *newblock = new TraceBlock(*block);
	  
	  // nobody else knows about the block, so we can have index 0
	  my_index = 0;
	  newblock->cur_size++;

	  // now update the "last_block" static value (while we still hold
	  //  the previous block's lock)
	  get_last_block() = newblock;
	  block->next = newblock;
	  pthread_mutex_unlock(&(block->mutex));
	  
	  // new block has been added, and we already grabbed our location,
	  //  so we're done
	  block = newblock;
	  break;
	}

	double time_units = ((Realm::Clock::current_time() - block->start_time) * 
			     block->time_mult);
	assert(time_units >= 0);
	assert(time_units < 4294967296.0);

	block->items[my_index].time_units = (unsigned)time_units;
        return block->items[my_index];
      }

      // Implementation specific to low level runtime because of synchronization
      static void dump_trace(const char *filename, bool append);
    };

    // Example Item types in lowlevel_impl.h 

  }; // LegionRuntime::LowLevel namespace

}; // LegionRuntime namespace

#undef PTHREAD_SAFE_CALL

#endif // __RUNTIME_UTILITIES_H__
