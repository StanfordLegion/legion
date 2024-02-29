/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/utsname.h>
#include <sys/time.h>
// This is a modifed version of the code originally from:
//
// http://stackoverflow.com/a/10008252/379568
//
// improved by:
//
// https://github.com/jtolio/malloc_instrumentation
//
// and then further updated for use here.
//
// This LD_PRELOAD library instruments malloc, calloc, realloc, memalign,
// valloc, posix_memalign, and free by outputting all calls to a logfile
// that can then be parsed by mem_trace.py.
//
// Unfortunately, it's not quite as straightforward as it sounds, as fprintf
// and dlsym both use heap-based memory allocation. During initialization, we
// use a dummy implementation of malloc that uses a small buffer. Once dlsym
// loading is done, then we switch to our real implementation, which, unless
// a recursive mutex is already held, first outputs the call arguments, makes
// the call, and then outputs the return value. If the recursive mutex is
// already held, then the call was due to some call made while outputting
// arguments, so we just forward the call along to the real call.
//

// Build with:
// g++ -shared -fPIC -o mem_trace.so mem_trace.cc faults.cc -lpthread -ldl -rdynamic -fpermissive -g

// Use this to determine how many alloc/free calls you want to
// skip before logging is started
#define SKIP_COUNT 0

// Set this to be the path to the filename string you want to 
// use to name the logging file, it must contain a "%d" which 
// will be filled in the the ID of the process that generated
// the logfile.
#ifdef CSV
#define LOGFILE_FORMAT "./mem_trace_%s_%d.csv"
#else
#define LOGFILE_FORMAT "./mem_trace_%s_%d.bin"
#endif
#define BACKTRACE_FILE_FORMAT "./backtrace_%s_%d.txt"

#include "faults.h"
#include <execinfo.h>
#include <fstream>
#include <map>
#include "mem_trace.h"

static void* (*real_malloc)(size_t size);
static void* (*real_calloc)(size_t nmemb, size_t size);
static void* (*real_realloc)(void *ptr, size_t size);
static void* (*real_memalign)(size_t blocksize, size_t bytes);
static void* (*real_valloc)(size_t size);
static int   (*real_posix_memalign)(void** memptr, size_t alignment,
                          size_t size);
static void  (*real_free)(void *ptr);
static void* (*temp_malloc)(size_t size);
static void* (*temp_calloc)(size_t nmemb, size_t size);
static void* (*temp_realloc)(void *ptr, size_t size);
static void* (*temp_memalign)(size_t blocksize, size_t bytes);
static void* (*temp_valloc)(size_t size);
static int   (*temp_posix_memalign)(void** memptr, size_t alignment,
                                     size_t size);
static void  (*temp_free)(void *ptr);

pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t internal_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t internal_mutex_attr;
int initializing = 0;
int initialized = 0;
int internal = 0;

char tmpbuf[1024];
unsigned long tmppos = 0;
unsigned long tmpallocs = 0;

size_t alloc_count = 0;
size_t print_count = 0;
size_t free_count = 0;
FILE *outfile = NULL;
std::ofstream *fs = NULL;
typedef std::map<uintptr_t, Backtrace*> Backtraces;
Backtraces *backtraces;
struct timeval start, end;

static double elapsed_time()
{
  gettimeofday(&end, 0);
  return end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec)*1e-6;
}

void* dummy_malloc(size_t size) {
    if (tmppos + size >= sizeof(tmpbuf)) exit(1);
    void *retptr = tmpbuf + tmppos;
    tmppos += size;
    ++tmpallocs;
    return retptr;
}

void* dummy_calloc(size_t nmemb, size_t size) {
    void *ptr = dummy_malloc(nmemb * size);
    unsigned int i = 0;
    for (; i < nmemb * size; ++i)
        *((char*)(ptr + i)) = '\0';
    return ptr;
}

void dummy_free(void *ptr) {
}

static pthread_mutex_t b_lock = PTHREAD_MUTEX_INITIALIZER;

static uintptr_t record_backtrace(char* nme, size_t size, void* rv)
{
  if(!initialized)
    return 0;

  pthread_mutex_lock(&b_lock);
  Backtrace *bt = new Backtrace;
  bt->capture_backtrace(2);
  uintptr_t hash = bt->hash();

  if(backtraces->find(hash) == backtraces->end())
  {
    (*fs) << nme << "(" << size << ") = " << rv << " hash: " << hash << std::endl;
    bt->lookup_symbols();
    (*fs) << (*bt) << std::endl;
    backtraces->insert(std::pair<uintptr_t, Backtrace*>(hash, bt));
  }
  else
  {
    delete bt;
  }
  pthread_mutex_unlock(&b_lock);
  return hash;
}


int start_call() {
    pthread_mutex_lock(&init_mutex);
    if (!initializing) {
      gettimeofday(&start, 0);
      initializing = 1;
      pthread_mutexattr_init(&internal_mutex_attr);
      pthread_mutexattr_settype(&internal_mutex_attr,
                                PTHREAD_MUTEX_RECURSIVE);
      pthread_mutex_init(&internal_mutex, &internal_mutex_attr);
      pthread_mutex_lock(&internal_mutex);
      pthread_mutex_unlock(&init_mutex);
      real_malloc         = dummy_malloc;
      real_calloc         = dummy_calloc;
      real_realloc        = NULL;
      real_free           = dummy_free;
      real_memalign       = NULL;
      real_valloc         = NULL;
      real_posix_memalign = NULL;

      temp_malloc         = dlsym(RTLD_NEXT, "malloc");
      temp_calloc         = dlsym(RTLD_NEXT, "calloc");
      temp_realloc        = dlsym(RTLD_NEXT, "realloc");
      temp_free           = dlsym(RTLD_NEXT, "free");
      temp_memalign       = dlsym(RTLD_NEXT, "memalign");
      temp_valloc         = dlsym(RTLD_NEXT, "valloc");
      temp_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");

      struct utsname buf;
      uname(&buf);
      char filename[1024];
      sprintf(filename, LOGFILE_FORMAT, buf.nodename, getpid());
      #ifdef CSV
      outfile = fopen(filename, "w");
      #else
      outfile = fopen(filename, "wb");
      #endif

      if (!temp_malloc || !temp_calloc || !temp_realloc || !temp_memalign ||
          !temp_valloc || !temp_posix_memalign || !temp_free)
      {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
        exit(1);
      }

      real_malloc         = temp_malloc;
      real_calloc         = temp_calloc;
      real_realloc        = temp_realloc;
      real_free           = temp_free;
      real_memalign       = temp_memalign;
      real_valloc         = temp_valloc;
      real_posix_memalign = temp_posix_memalign;

      pthread_mutex_lock(&b_lock);
      char backtrace_filename[1024];
      sprintf(backtrace_filename, BACKTRACE_FILE_FORMAT, buf.nodename, getpid());
      fs = new std::ofstream(backtrace_filename);
      backtraces = new Backtraces();
      pthread_mutex_unlock(&b_lock);

      #ifdef CSV
      fprintf(outfile, "time,type,size,pointer,hash\n");
      #endif

      initialized = 1;
    } else {
      pthread_mutex_unlock(&init_mutex);
      pthread_mutex_lock(&internal_mutex);
    }

    if (!initialized || internal) {
      pthread_mutex_unlock(&internal_mutex);
      return 1;
    }

    internal = 1;
    return 0;
}

void end_call() {
    internal = 0;
    pthread_mutex_unlock(&internal_mutex);
}

void* malloc(size_t size) {
    if (start_call()) return real_malloc(size);
    void *ptr = NULL;
    ptr = real_malloc(size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "malloc";
      uintptr_t hash = record_backtrace(nme, size, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::MALLOC_KIND, size, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::MALLOC_KIND, size, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return ptr;
}

void* calloc(size_t nmemb, size_t size) {
    if (start_call()) return real_calloc(nmemb, size);
    void *ptr = NULL;
    ptr = real_calloc(nmemb, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "calloc";
      uintptr_t hash = record_backtrace(nme, size, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::CALLOC_KIND, size, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::CALLOC_KIND, size, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return ptr;
}

void* realloc(void *optr, size_t size) {
    if (start_call()) return real_realloc(optr, size);
    void *ptr = NULL;
    ptr = real_realloc(optr, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "realloc";
      uintptr_t hash = record_backtrace(nme, size, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::REALLOC_KIND, size, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::REALLOC_KIND, size, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return ptr;
}

void free(void *ptr) {
    if (start_call()) {
        real_free(ptr);
        return;
    }
    if (__sync_fetch_and_add(&free_count,1) >= SKIP_COUNT) {
      char nme[] = "free";
      uintptr_t hash = record_backtrace(nme, 0, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::FREE_KIND, 0, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::FREE_KIND, 0, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    real_free(ptr);
    end_call();
    return;
}

void* memalign(size_t blocksize, size_t bytes) {
    if (start_call()) return real_memalign(blocksize, bytes);

    void *ptr = NULL;
    ptr = real_memalign(blocksize, bytes);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "memalign";
      uintptr_t hash = record_backtrace(nme, bytes, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::MEMALIGN_KIND, bytes, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::MEMALIGN_KIND, bytes, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return ptr;
}

int posix_memalign(void** memptr, size_t alignment, size_t size) {
    if (start_call()) return real_posix_memalign(memptr, alignment, size);

    int rv = 0;
    rv = real_posix_memalign(memptr, alignment, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "posix_memalign";
      uintptr_t hash = record_backtrace(nme, size, *memptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::POSIXMEMALIGN_KIND, size, *memptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::POSIXMEMALIGN_KIND, size, memptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return rv;
}

void* valloc(size_t size) {
    if (start_call()) return real_valloc(size);

    void *ptr = NULL;
    ptr = real_valloc(size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT) {
      char nme[] = "valloc";
      uintptr_t hash = record_backtrace(nme, size, ptr);
      #ifdef CSV
      fprintf(outfile, "%f,%d,%zu,%p,%zd\n", elapsed_time(), AllocKind::VALLOC_KIND, size, ptr, hash);
      #else
      Alloc alloc(elapsed_time(), AllocKind::VALLOC_KIND, size, ptr, hash);
      fwrite(&alloc, 1, sizeof(alloc), outfile);
      #endif
    }
    end_call();
    return ptr;
}
