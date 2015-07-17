/*
 * This file is part of cgreenlet. CGreenlet is free software available
 * under the terms of the MIT license. Consult the file LICENSE that was
 * shipped together with this source file for the exact licensing terms.
 *
 * Copyright (c) 2012 by the cgreenlet authors. See the file AUTHORS for a
 * full list.
 */

#include "greenlet"
#include "greenlet-int.h"

#include <new>
#include <cassert>
#include <exception>


struct _greenlet_data
{
#ifdef HAVE_CXX11
    std::exception_ptr exception;
#else
    greenlet_exit *exception;
#endif
};

greenlet::greenlet(greenlet_start_func_t start_func, 
                   void *stack, long *stacksize,
                   greenlet *parent)
{
    greenlet_t *c_parent = parent ? parent->_greenlet->gr_parent : 0L;

    _greenlet = greenlet_new(_run, c_parent, stack, stacksize);
    if (_greenlet == 0L)
        throw std::bad_alloc();
    _greenlet->gr_instance = this;
    _start_func = start_func;
    _data = new _greenlet_data;
}

greenlet::greenlet(greenlet_t *greenlet)
{
    _greenlet = greenlet;
    _greenlet->gr_instance = this;
    _start_func = greenlet->gr_start;
    greenlet->gr_start = _run;
    _data = new _greenlet_data;
}

greenlet::greenlet(const greenlet &rhs)
{
    // No copying
    assert(false);
}

greenlet::~greenlet()
{
    greenlet_destroy(_greenlet);
    delete _data;
}

greenlet& greenlet::operator=(const greenlet &rhs)
{
    // No copying
    assert(false);
    return *this;
}

greenlet *greenlet::root()
{
    greenlet_t *c_root = greenlet_root();
    greenlet *root = (greenlet *) c_root->gr_instance;
    if (root == 0L)
        root = new greenlet(c_root);
    return root;
}

greenlet *greenlet::current()
{
    greenlet_t *c_current = greenlet_current();
    greenlet *current = (greenlet *) c_current->gr_instance;
    if (current == 0L)
        current = new greenlet(c_current);
    return current;
}

greenlet *greenlet::parent()
{
    greenlet_t *c_parent = _greenlet->gr_parent;
    greenlet *parent = (greenlet *) c_parent->gr_instance;
    if (parent == 0L)
        parent = new greenlet(c_parent);
    return parent;
}

void *greenlet::switch_to(void *arg)
{
    return greenlet_switch_to(_greenlet, arg);
}

void greenlet::inject(greenlet_inject_func_t inject_func)
{
    _greenlet->gr_inject = inject_func;
}

void greenlet::reset()
{
    _greenlet->gr_flags = 0;
}

bool greenlet::isstarted()
{
    return (_greenlet->gr_flags & GREENLET_STARTED) > 0;
}

bool greenlet::isdead()
{
    return (_greenlet->gr_flags & GREENLET_DEAD) > 0;
}

void *greenlet::release_stack(long *stacksize)
{
    return greenlet_release_stack(_greenlet, stacksize);
}

void greenlet::init_greenlet_library(void)
{
    greenlet_init();
}

void greenlet::init_greenlet_thread(void)
{
    greenlet_init_thread();
}

void *greenlet::alloc_greenlet_stack(long *stacksize)
{
    return greenlet_alloc_stack(stacksize);
}

void greenlet::dealloc_greenlet_stack(void *stack, long stacksize)
{
    greenlet_dealloc_stack(stack, stacksize);
}

void *greenlet::run(void *arg)
{
    if (_start_func == 0L)
        return 0L;
    return _start_func(arg);
}

void *greenlet::_run(void *arg)
{
    greenlet *greenlet = current();
    void *result = 0L;
    try {
        result = greenlet->run(arg);
    } catch (...) {
        greenlet = greenlet->parent();
        while (greenlet->isdead())
            greenlet = greenlet->parent();
        greenlet->_greenlet->gr_inject = _inject_exception;
#ifdef HAVE_CXX11
        greenlet->_data->exception = std::current_exception();
#else
        greenlet->_data->exception = new greenlet_exit();
#endif
    }
    return result;
}

void greenlet::_inject_exception(void *arg)
{
#ifdef HAVE_CXX11
    std::rethrow_exception(current()->_data->exception);
#else
    throw *current()->_data->exception;
#endif
}
