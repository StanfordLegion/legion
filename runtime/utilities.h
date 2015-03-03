/* Copyright 2015 Stanford University
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

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifndef __GNUC__
#include "atomics.h" // for __sync_fetch_and_add
#endif

// outside of namespace because 50-letter-long enums are annoying
enum {
  TIME_NONE,
  TIME_KERNEL,
  TIME_COPY,
  TIME_HIGH_LEVEL,
  TIME_LOW_LEVEL,
  TIME_MAPPER,
  TIME_SYSTEM,
  TIME_AM,
};

// widget for generating debug/info messages
  enum LogLevel {
    LEVEL_SPEW,
    LEVEL_DEBUG,
    LEVEL_INFO,
    LEVEL_PRINT,
    LEVEL_WARNING,
    LEVEL_ERROR,
    LEVEL_NONE,
  };
#ifndef COMPILE_TIME_MIN_LEVEL
#define COMPILE_TIME_MIN_LEVEL LEVEL_SPEW
#endif

//#define DETAILED_TIMING

namespace LegionRuntime {
  /**
   * A logger class for tracking everything from debug messages
   * to error messages.
   */
  class Logger {
  public:
#ifdef NODE_LOGGING
    static FILE* get_log_file(void); 
    static void finalize(void);
#endif
    // Use the -DORDERED_LOGGING flag whenever you need to have ordered
    // output.  The posix standard guarantees that all writes to files
    // open for appending will be atomic, which is necessary for safely
    // merging multiple streams of output when running across many nodes.
    // Note that the user is responsible for deleting the log file
    // between execution runs or risk accumulating messages from multiple runs.
    // WARNING: apparently this is not true on NFS so instead we use file
    // locks in distributed environments to serialize access to the file
    // for safe logging.
#ifdef ORDERED_LOGGING 
    // Make the buffer 1 MB
    static const int logging_buffer_size = (1 << 20);
    static inline int get_log_file(void)
    {
      static int log_file = -1;
      if (log_file == -1)
      {
        log_file = open("legion.log",O_WRONLY | O_CREAT | O_APPEND,
                        S_IRUSR | S_IWUSR /*no other permissions so locks work*/);
        assert(log_file != -1);
      }
      return log_file;
    }
    static inline char* get_logging_buffer(void)
    {
      static char *log_buffer = NULL;
      if (log_buffer == NULL)
      {
        log_buffer = (char*)malloc(logging_buffer_size);
        assert(log_buffer != NULL);
      }
      return log_buffer;
    }
    // Make this a long long so it can never roll over
    static long long* get_logging_location(void)
    {
      static long long current_location = 0;
      return &current_location;
    }
    static int* get_written_location(void)
    {
      static int current_writing = 0;
      return &current_writing;
    }
    // Need a finalize if we have a log file
    static void finalize(void);
#endif
    static void init(int argc, const char *argv[])
    {
#ifdef ORDERED_LOGGING 
      get_log_file(); // Initializes the log file
      get_logging_buffer(); // Initialize the buffer
      get_logging_location(); // Initialize the current location
      get_written_location(); // Initialize the written location
#endif
#ifdef NODE_LOGGING
      get_log_file(); // Initialize the log file
#endif
      for(std::vector<bool>::iterator it = Logger::get_log_cats_enabled().begin();
          it != Logger::get_log_cats_enabled().end();
          it++)
        (*it) = true;

      for(int i = 1; i < argc; i++) {
        if(!strcmp(argv[i], "-level")) {
          Logger::get_log_level() = (LogLevel)atoi(argv[++i]);
          continue;
        }

        if(!strcmp(argv[i], "-cat")) {
          const char *p = argv[++i];

          if(*p == '*') {
            p++;
          } else {
            // turn off all the bits and then we'll turn on only what's requested
            for(std::vector<bool>::iterator it = Logger::get_log_cats_enabled().begin();
                it != Logger::get_log_cats_enabled().end();
                it++)
              (*it) = false;
          }

          while(*p == ',') p++;
          while(*p) {
            bool enable = true;
            if(*p == '-') {
              enable = false;
              p++;
            }
            const char *p2 = p; while(*p2 && (*p2 != ',')) p2++;
            std::string name(p, p2);
            std::map<std::string, int>::iterator it = Logger::get_categories_by_name().find(name);
            if(it == Logger::get_categories_by_name().end()) {
              fprintf(stderr, "unknown log category '%s'!\n", name.c_str());
              exit(1);
            }

            Logger::get_log_cats_enabled()[it->second] = enable;

            p = p2;
            while(*p == ',') p++;
          }
        }
        continue;
      }

#ifdef DEBUG_LOGGER_CATEGORIES
      printf("logger settings: level=%d cats=", log_level);
      bool first = true;
      for(unsigned i = 0; i < log_cats_enabled.size(); i++)
        if(log_cats_enabled[i]) {
          if(!first) printf(",");
          first = false;
          printf("%s", categories_by_id[i].c_str());
        }
      printf("\n");
#endif
    } 

    static inline void log(LogLevel level, int category, const char *fmt, ...)
    {
      if(level >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
        if(level >= Logger::get_log_level()) {             // dynamic opt-out
          if(Logger::get_log_cats_enabled()[category]) {   // category filter
            va_list args;
            va_start(args, fmt);
            logvprintf(level, category, fmt, args);
            va_end(args);
          }
        }
      }
    }

    class Category {
    public:
      Category(const std::string& name)
      {
        index = Logger::add_category(name);
      }

      operator int(void) const { return index; }

      int index;

      inline void operator()(int level, const char *fmt, ...) __attribute__((format (printf, 3, 4)))
      {
        if(level >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(level >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf((LogLevel)level, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void spew(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_SPEW >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_SPEW >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_SPEW, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void debug(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_DEBUG >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_DEBUG >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_DEBUG, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void info(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_INFO >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_INFO >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_INFO, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void print(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_PRINT >= COMPILE_TIME_MIN_LEVEL) { // static opt-out
          if (LEVEL_PRINT >= Logger::get_log_level()) { // dynamic opt-out
            if (Logger::get_log_cats_enabled()[index]) {
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_PRINT, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void warning(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_WARNING >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_WARNING >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_WARNING, index, fmt, args);
              va_end(args);
            }
          }
        }
      }

      inline void error(const char *fmt, ...) __attribute__((format (printf, 2, 3)))
      {
        if(LEVEL_ERROR >= COMPILE_TIME_MIN_LEVEL) {  // static opt-out
          if(LEVEL_ERROR >= Logger::get_log_level()) {             // dynamic opt-out
            if(Logger::get_log_cats_enabled()[index]) {      // category filter
              va_list args;
              va_start(args, fmt);
              Logger::logvprintf(LEVEL_ERROR, index, fmt, args);
              va_end(args);
            }
          }
        }
      }
    };

  protected:
    // Implementations specific to low level runtime
    static void logvprintf(LogLevel level, int category, const char *fmt, va_list args);
    static int add_category(const std::string& name)
    {
      int index;
      std::map<std::string, int>::iterator it = Logger::get_categories_by_name().find(name);
      if(it == Logger::get_categories_by_name().end()) {
        index = Logger::get_categories_by_id().size();
        Logger::get_categories_by_id().push_back(name);
        Logger::get_categories_by_name()[name] = index;
        Logger::get_log_cats_enabled().resize(index + 1);
      } else {
        index = it->second;
      }
      return index;
    }
    // Use static methods with static variables to avoid initialization ordering problem
    static LogLevel& get_log_level(void)
    {
      // default level for now is INFO
      static LogLevel log_level = LEVEL_INFO;
      return log_level;
    }
    static std::vector<bool>& get_log_cats_enabled(void)
    {
      static std::vector<bool> log_cats_enabled;
      return log_cats_enabled;
    }
    static std::map<std::string,int>& get_categories_by_name(void)
    {
      static std::map<std::string,int> categories_by_name;
      return categories_by_name;
    }
    static std::vector<std::string>& get_categories_by_id(void)
    {
      static std::vector<std::string> categories_by_id;
      return categories_by_id;
    }
    // Conversion from a log level to a string representation
    static const char* stringify(LogLevel level)
    {
      switch (level)
      {
        case LEVEL_SPEW:
          return "SPEW";
        case LEVEL_DEBUG:
          return "DEBUG";
        case LEVEL_INFO:
          return "INFO";
        case LEVEL_PRINT:
          return "PRINT";
        case LEVEL_WARNING:
          return "WARNING";
        case LEVEL_ERROR:
          return "ERROR";
        case LEVEL_NONE:
          return "NONE";
      }
      assert(false);
      return NULL;
    }
  };

  /**
   * Have a way of making timesteps for global timing
   */
#ifndef __MACH__
  class TimeStamp {
  private:
    const char *print_message;
    struct timespec spec;
    bool diff;
  public:
    TimeStamp(const char *message, bool difference)
      : print_message(message), diff(difference)
    { 
      if (difference)
      {
        clock_gettime(CLOCK_MONOTONIC, &spec); 
      }
      else
      {
        clock_gettime(CLOCK_MONOTONIC, &spec); 
        // Give the time in s
        double time = 1e6 * (spec.tv_sec) + 1e-3 * (spec.tv_nsec);
        printf("%s %7.6f\n", message, time); //spec.tv_nsec);
      }
    }
    ~TimeStamp(void)
    {
      if (diff)
      {
        struct timespec stop;
        clock_gettime(CLOCK_MONOTONIC, &stop);
        double time = get_diff_us(spec, stop);
        printf("%s %7.3f us\n", print_message, time);
      }
    }
  public:
    static inline unsigned long long get_current_time_in_micros(void)
    {
      struct timespec spec;
      clock_gettime(CLOCK_MONOTONIC, &spec);
      unsigned long long result = (((unsigned long long)spec.tv_sec) * 1000000)
        + (((unsigned long long)spec.tv_nsec)/1000);
      return result;
    }
    static inline unsigned long long get_current_time_in_nanos(void)
    {
      struct timespec spec;
      clock_gettime(CLOCK_MONOTONIC, &spec);
      unsigned long long result = (((unsigned long long)spec.tv_sec) * 1000000000)
        + ((unsigned long long)spec.tv_nsec);
      return result;
    }
  private:
    double get_diff_us(struct timespec &start, struct timespec &stop)
    {
      return (1e6 * (stop.tv_sec - spec.tv_sec) + 
              1e-3 * (stop.tv_nsec - spec.tv_nsec));
    }
  };
#else
  class TimeStamp {
  private:
    const char *print_message;
    mach_timespec_t spec;
    bool diff;
  public:
    TimeStamp(const char *message, bool difference)
      : print_message(message), diff(difference)
    {
      clock_serv_t cclock;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
      if (difference)
        clock_get_time(cclock, &spec);
      else
      {
        clock_get_time(cclock, &spec);
        // Give the time in s
        double time = 1e6 * (spec.tv_sec) + 1e-3 * (spec.tv_nsec);
        printf("%s %7.6f\n", message, time);
      }
      mach_port_deallocate(mach_task_self(), cclock);
    }
    ~TimeStamp(void)
    {
      if (diff)
      {
        mach_timespec_t stop;
        clock_serv_t cclock;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &stop);
        mach_port_deallocate(mach_task_self(), cclock);
        double time = get_diff_us(spec, stop);
        printf("%s %7.3f us\n", print_message, time);
      } 
    }
  public:
    static inline unsigned long long get_current_time_in_micros(void)
    {
      mach_timespec_t spec;
      clock_serv_t cclock;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
      clock_get_time(cclock, &spec);
      mach_port_deallocate(mach_task_self(), cclock);
      unsigned long long result = (((unsigned long long) spec.tv_sec) * 1000000) + (((unsigned long long)spec.tv_nsec) / 1000);
      return result;
    }
    static inline unsigned long long get_current_time_in_nanos(void)
    {
      mach_timespec_t spec;
      clock_serv_t cclock;
      host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
      clock_get_time(cclock, &spec);
      mach_port_deallocate(mach_task_self(), cclock);
      unsigned long long result = (((unsigned long long) spec.tv_sec) * 1000000000) + ((unsigned long long)spec.tv_nsec);
      return result;
    }
  private:
    double get_diff_us(mach_timespec_t &start, mach_timespec_t &stop)
    {
      return (1e6 * (stop.tv_sec - spec.tv_sec) + 
              1e-3 * (stop.tv_nsec - spec.tv_nsec));
    }
  };
#endif

#ifdef DEBUG_LOW_LEVEL
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

  /**
   * A timer class for doing detailed timing analysis of applications
   * Implementation is specific to low level runtimes
   */
  namespace LowLevel {
    /* clock that is (mostly-)synchronized across whole machine */
    class Clock {
    protected:
      static double zero_time;

    public:
      static double abs_time(void)
      {
#ifndef __MACH__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
#else
        mach_timespec_t tp;
        clock_serv_t cclock;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &tp);
        mach_port_deallocate(mach_task_self(), cclock);
#endif
	return((1.0 * tp.tv_sec) + (1e-9 * tp.tv_nsec));
      }
      
      static double abs_to_rel(double abs_time)
      {
	return(abs_time - zero_time);
      }

      static double rel_time(void)
      {
	return abs_to_rel(abs_time());
      }

      static void synchronize(void);
    };

    class DetailedTimer {
    public:
#ifdef DETAILED_TIMING
      static void clear_timers(bool all_nodes = true);
      static void push_timer(int timer_kind);
      static void pop_timer(void);
      static void roll_up_timers(std::map<int, double>& timers, bool local_only);
      static void report_timers(bool local_only = false);
#else
      static void clear_timers(bool all_nodes = true) {}
      static void push_timer(int timer_kind) {}
      static void pop_timer(void) {}
      static void roll_up_timers(std::map<int, double>& timers, bool local_only) {}
      static void report_timers(bool local_only = false) {}
#endif
      class ScopedPush {
      public:
        ScopedPush(int timer_kind) { push_timer(timer_kind); }
        ~ScopedPush(void) { pop_timer(); }
      };

      static const char* stringify(int level)
      {
        switch (level)
        {
          case TIME_NONE:
            return "NONE";
          case TIME_KERNEL:
            return "KERNEL";
          case TIME_COPY:
            return "COPY";
          case TIME_HIGH_LEVEL:
            return "HIGH-LEVEL";
          case TIME_LOW_LEVEL:
            return "LOW-LEVEL";
          case TIME_MAPPER:
            return "MAPPER";
          case TIME_SYSTEM:
            return "SYSTEM";
          case TIME_AM:
            return "ACTV_MESG";
          default:
            break;
        }
        // We only call this at the end of the run so leaking a little memory isn't too bad
        char *result = new char[16];
        sprintf(result,"%d",level);
        return result;
      }
    };

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
	  start_time = Clock::rel_time();
	  items = new ITEM[max_size];
	}

	TraceBlock(size_t block_size, double exp_arrv_rate)
	  : max_size(block_size), cur_size(0), next(0)
	{
	  PTHREAD_SAFE_CALL(pthread_mutex_init(&mutex,NULL));
	  start_time = Clock::rel_time();
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

	double time_units = ((Clock::rel_time() - block->start_time) * 
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

  // typedef so we can use detailed timers anywhere in the runtime
  typedef LowLevel::DetailedTimer DetailedTimer;

}; // LegionRuntime namespace

#undef PTHREAD_SAFE_CALL

#endif // __RUNTIME_UTILITIES_H__
