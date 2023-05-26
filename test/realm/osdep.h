// OS-dependent functionality used by various Realm unit tests

#ifndef OSDEP_H
#define OSDEP_H

#ifndef _MSC_VER
#include <csignal>
#include <unistd.h>
#include <sys/resource.h>
#endif

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#include <stdint.h>

static void usleep(long microseconds) { Sleep(microseconds / 1000); }
static void sleep(long seconds) { Sleep(seconds * 1000); }

static int32_t __sync_fetch_and_add(int32_t *dst, int32_t amt) { return InterlockedAdd((LONG *)dst, amt) - amt; }
static int32_t __sync_add_and_fetch(int32_t *dst, int32_t amt) { return InterlockedAdd((LONG *)dst, amt); }
static int32_t __sync_sub_and_fetch(int32_t *dst, int32_t amt) { return InterlockedAdd((LONG *)dst, -amt); }
static bool __sync_bool_compare_and_swap(volatile uint64_t *tgtptr, uint64_t oldval, uint64_t newval)
{
  uint64_t actval = InterlockedCompareExchange64((LONG64 *)tgtptr, newval, oldval);
  return (actval == oldval);
}

static void alarm(int seconds) {}
#define SIGALRM 0.0  /* float so we get our overload below */
static void signal(float signum, void (*handler)(int)) {}

static long lrand48(void) { return rand(); }
static void srand48(long seed) { srand(seed); }

#define __alignof__(T) __alignof(T)

#endif

#define USE_ACCURATE_SLEEP

void accurate_sleep(long long microseconds) {
  // Sleep can be inaccurate depending on system configuration and load,
  // at least track that here so that we know when the sleep is accurate.

  long long init_time = Realm::Clock::current_time_in_microseconds();
  long long current_time = init_time;

#ifdef USE_ACCURATE_SLEEP
  // Attempt to do a more accurate sleep by breaking the interval into
  // smaller pieces of size `granule` microseconds.

  long long final_target_time = init_time + microseconds;

  const long long granule = 100000; // 100 ms
  while (final_target_time - current_time > granule) {
    usleep(granule);
    current_time = Realm::Clock::current_time_in_microseconds();
  }
#else
  usleep(microseconds);
  current_time = Realm::Clock::current_time_in_microseconds();
#endif

  double relative = ((double)(current_time - init_time))/microseconds;
  if (relative > 1.2) {
    std::cerr << "sleep took too long - goal: " << microseconds <<
      " us, actual: " << current_time - init_time << " us, relative: " <<
      relative << std::endl;
  }
}

#endif
