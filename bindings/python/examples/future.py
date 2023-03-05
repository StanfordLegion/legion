#!/usr/bin/env python3

# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import codecs

import pygion
from pygion import task, Future

@task
def show(f):
    print(f.get())

@task
def show_buffer(f):
    value = codecs.decode(f.get_buffer(), 'utf-8')
    print(value)

@task
def show_nested(f):
    print("nested: %s" % f)
    show(f)

@task
def show_index(i, f):
    print("at index %s: %s" % (i, f.get()))

@task
def return_void():
    # this task doesn't literally return void, but we can ignore the
    # value and pretend it does
    pass

# the same, but typed at the task level
@task(return_type=pygion.void)
def return_void():
    pass

@task
def main():
    # No explicit type specified, value is pickled
    f = Future([1, 2, 3])
    print("value of f.get() is %s" % f.get())
    assert f.get() == [1, 2, 3]

    # Explicit type specified, value not pickled
    g = Future(123, pygion.int64)
    print("value of g.get() is %s" % g.get())
    assert g.get() == 123

    # Using a buffer object to pass raw bytes
    h = Future.from_buffer('asdf'.encode('utf-8'))
    h_value = codecs.decode(h.get_buffer(), 'utf-8')
    print("value of h.get_raw() is %s" % h_value)
    assert h_value == 'asdf'

    i = Future(return_void(), value_type=pygion.void)
    print("value of i.get() is %s" % i.get())

    j = return_void()
    print("value of j.get() is %s" % j.get())
    assert j.is_ready()

    show(f)
    show(g)
    show_buffer(h)

    show_nested(f)

    for i in pygion.IndexLaunch([3]):
        show_index(i, f)

if __name__ == '__main__':
    main()
