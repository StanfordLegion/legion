#!/usr/bin/env python

# Copyright 2018 Stanford University
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

import legion
from legion import task, Future

@task
def show(f):
    print(f.get())

@task
def show_buffer(f):
    value = codecs.decode(f.get_buffer(), 'utf-8')
    print(value)

@task(top_level=True)
def main():
    # No explicit type specified, value is pickled
    f = Future([1, 2, 3])
    print("value of f.get() is %s" % f.get())
    assert f.get() == [1, 2, 3]

    # Explicit type specified, value not pickled
    g = Future(123, legion.int64)
    print("value of f.get() is %s" % g.get())
    assert g.get() == 123

    # Using a buffer object to pass raw bytes
    h = Future.from_buffer('asdf'.encode('utf-8'))
    h_value = codecs.decode(h.get_buffer(), 'utf-8')
    print("value of h.get_raw() is %s" % h_value)
    assert h_value == 'asdf'

    show(f)
    show(g)
    show_buffer(h)
