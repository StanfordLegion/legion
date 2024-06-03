#!/usr/bin/env python3

# Copyright 2024 Stanford University
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

import pygion
from pygion import task

@task
def hi(i):
    print("hello %s" % i)
    return i

@task
def main():
    futures = []
    for i in range(10):
        # When calling a task, it's possible to assign a task to a
        # specific point using the `point` keyword argument:
        futures.append(hi(i, point=i))
    for i, future in enumerate(futures):
        print("got %s" % future.get())
        assert future.get() == i

if __name__ == '__main__':
    main()
