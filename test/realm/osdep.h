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
static void usleep(long microseconds) { Sleep(microseconds / 1000); }
static void sleep(long seconds) { Sleep(seconds * 1000); }

static long __sync_fetch_and_add(long *dst, long amt) { return InterlockedAdd(dst, amt) - amt; }
static long __sync_add_and_fetch(long *dst, long amt) { return InterlockedAdd(dst, amt); }

static void alarm(int seconds) {}
#define SIGALRM 0.0  /* float so we get our overload below */
static void signal(float signum, void (*handler)(int)) {}

static long lrand48(void) { return rand(); }
static void srand48(long seed) { srand(seed); }

#define __alignof__(T) __alignof(T)

#endif

#endif
