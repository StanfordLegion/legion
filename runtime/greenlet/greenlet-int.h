/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */

#ifndef GREENLET_INT_H_INCLUDED
#define GREENLET_INT_H_INCLUDED

#if defined(__linux__) || defined(__WIN32__)
# define TLS_USE___THREAD
#elif defined(__APPLE__)
# define TLS_USE_PTHREAD
#else
# error "Do not know how to use thread-local storage for this platform"
#endif

#if defined(__linux__) || defined(__APPLE__)
# define STACK_USE_MMAP
#elif defined(__WIN32__)
# define STACK_USE_VIRTUAL_ALLOC
#else
# error "Do not know how to allocate stacks for this platform"
#endif

#if defined(__GXX_EXPERIMENTAL_CXX0X__)
# define HAVE_CXX11
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define _greenlet_def_max_stacksize 2*1024*1024

int _greenlet_savecontext(void *frame);
void _greenlet_switchcontext(void *frame, void (*inject)(void *), void *arg)
        __attribute__((noreturn));
void _greenlet_newstack(void *stack, void (*main)(void *), void *arg)
        __attribute__((noreturn));

#ifdef TLS_USE_PTHREAD

/* Systems without __thread a library version of these. */
void _greenlet_pthread_init();
void _greenlet_tls_init();
greenlet_t *_greenlet_get_root();
greenlet_t *_greenlet_get_current();
void _greenlet_set_current(greenlet_t *current);

#endif

void *_greenlet_alloc_stack(long *size);
void _greenlet_dealloc_stack(void *stack, long size);

#ifdef __cplusplus
}
#endif

#endif /* GREENLET_INT_H_INCLUDED */
