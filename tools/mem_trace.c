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

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

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
// gcc -shared -fPIC -o mem_trace.so mem_trace.c -lpthread -ldl

// Use this to determine how many alloc/free calls you want to
// skip before logging is started
#define SKIP_COUNT 1000000
// Set this to be the path to the filename string you want to 
// use to name the logging file, it must contain a "%d" which 
// will be filled in the the ID of the process that generated
// the logfile.
#define LOGFILE_FORMAT "./mem_trace_%d.txt"

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
size_t free_count = 0;
FILE *outfile = NULL;

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

int start_call() {
    pthread_mutex_lock(&init_mutex);
    if (!initializing) {
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

        char filename[128];
        sprintf(filename, LOGFILE_FORMAT, getpid());
        outfile = fopen(filename, "w");

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
    void *rv = NULL;
    rv = real_malloc(size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "malloc(%zu) = %p\n", size, rv);
    end_call();
    return rv;
}

void* calloc(size_t nmemb, size_t size) {
    if (start_call()) return real_calloc(nmemb, size);
    void *p = NULL;
    p = real_calloc(nmemb, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "calloc(%zu, %zu) = %p\n", nmemb, size, p);
    end_call();
    return p;
}

void* realloc(void *ptr, size_t size) {
    if (start_call()) return real_realloc(ptr, size);
    void *p = NULL;
    p = real_realloc(ptr, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "realloc(%p, %zu) = %p\n", ptr, size, p);
    end_call();
    return p;
}

void free(void *ptr) {
    if (start_call()) {
        real_free(ptr);
        return;
    }
    if (__sync_fetch_and_add(&free_count,1) >= SKIP_COUNT)
      fprintf(outfile, "free(%p)\n", ptr);
    real_free(ptr);
    end_call();
    return;
}

void* memalign(size_t blocksize, size_t bytes) {
    if (start_call()) return real_memalign(blocksize, bytes);

    void *p = NULL;
    p = real_memalign(blocksize, bytes);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "memalign(%zu, %zu) = %p\n", blocksize, bytes, p);
    end_call();
    return p;
}

int posix_memalign(void** memptr, size_t alignment, size_t size) {
    if (start_call()) return real_posix_memalign(memptr, alignment, size);

    int rv = 0;
    rv = real_posix_memalign(memptr, alignment, size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "posix_memalign(%p, %zu, %zu) = %p\n", memptr, alignment, size, *memptr);
    end_call();
    return rv;
}

void* valloc(size_t size) {
    if (start_call()) return real_valloc(size);

    void *p = NULL;
    p = real_valloc(size);
    if (__sync_fetch_and_add(&alloc_count,1) >= SKIP_COUNT)
      fprintf(outfile, "valloc(%zu) = %p\n", size, p);
    end_call();
    return p;
}
