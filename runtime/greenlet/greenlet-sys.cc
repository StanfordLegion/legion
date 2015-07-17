/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */

#include <stdlib.h>
#include <unistd.h>

#include "greenlet.h"
#include "greenlet-int.h"


#ifdef TLS_USE_PTHREAD

#include <pthread.h>

static pthread_key_t _root_greenlet;
static pthread_key_t _current_greenlet;

void _greenlet_pthread_init(void)
{
    if (pthread_key_create(&_root_greenlet, NULL) != 0)
        return;
    if (pthread_key_create(&_current_greenlet, NULL) != 0)
        return;
}

void _greenlet_tls_init(void)
{
    greenlet_t *root;
  
    root = (greenlet_t*)calloc(1, sizeof(greenlet_t));
    if (root != NULL)
    {
        root->gr_flags = GREENLET_STARTED;
        pthread_setspecific(_root_greenlet, root);
    }
}

greenlet_t *_greenlet_get_root()
{
    return (greenlet_t*)pthread_getspecific(_root_greenlet);
}

greenlet_t *_greenlet_get_current()
{
    return (greenlet_t*)pthread_getspecific(_current_greenlet);
}   

void _greenlet_set_current(greenlet_t *current)
{
    pthread_setspecific(_current_greenlet, current);
}

#endif  /* TLS_USE_PTHREAD */


/* Stack allocation. */

#ifdef STACK_USE_MMAP

#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>

void *_greenlet_alloc_stack(long *size)
{
    long stacksize, pagesize;
    void *stack;
    struct rlimit rlim;

    stacksize = *size;
    if (stacksize == 0) {
        if (getrlimit(RLIMIT_STACK, &rlim) < 0)
	    return NULL;
        stacksize = rlim.rlim_cur;
        if (stacksize > _greenlet_def_max_stacksize)
            stacksize = _greenlet_def_max_stacksize;
    }
    pagesize = sysconf(_SC_PAGESIZE);
    if ((pagesize < 0) || (stacksize < 2*pagesize))
        return NULL;
    stacksize = (stacksize + pagesize - 1) & ~(pagesize - 1);
    stack = mmap(NULL, stacksize, PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
    if (stack == NULL)
        return NULL;
    if (mprotect(stack, pagesize, PROT_NONE) < 0) {
	munmap(stack, stacksize);
	return NULL;
    }
    *size = stacksize;
    return stack;
}

void _greenlet_dealloc_stack(void *stack, long size)
{
    munmap(stack, size);
}

#endif  /* STACK_USE_MMAP */

#ifdef STACK_USE_VIRTUAL_ALLOC

#include <windows.h>

void *_greenlet_alloc_stack(long *size)
{
    long stacksize, pagesize;
    unsigned long old;
    void *stack;
    SYSTEM_INFO si;

    stacksize = *size;
    GetSystemInfo(&si);
    pagesize = si.dwAllocationGranularity;
    if ((stacksize == 0) || (stacksize > _greenlet_def_max_stacksize))
	stacksize = _greenlet_def_max_stacksize;
    stacksize = (stacksize + pagesize - 1) & ~(pagesize - 1);
    stack = VirtualAlloc(NULL, stacksize, MEM_COMMIT|MEM_RESERVE,
			 PAGE_READWRITE);
    if (stack == NULL)
	return NULL;
    if (VirtualProtect(stack, pagesize, PAGE_NOACCESS, &old) == 0)
    {
	VirtualFree(stack, stacksize, MEM_DECOMMIT);
	return NULL;
    }
    *size = stacksize;
    return stack;
}

void _greenlet_dealloc_stack(void *stack, long size)
{
    VirtualFree(stack, size, MEM_DECOMMIT);
}

#endif  /* STACK_USE_VIRTUAL_ALLOC */
