#!/usr/bin/env python

# Copyright 2019 Stanford University
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

import legion
from legion import task

global_var = 123 # Kids: Don't do this at home

@task
def hi(i):
    global global_var
    global_var = global_var + 4000

    print("hello %s" % i)
    return i

@task
def main():
    global global_var

    global_procs = legion.Tunable.select(legion.Tunable.GLOBAL_PYS).get()

    with legion.MustEpochLaunch(): # implicit launch domain
        for i in range(global_procs):
            hi(i, point=i)
    legion.execution_fence(block=True)

    assert global_var == 4123

    global_var = 456

    with legion.MustEpochLaunch([global_procs]):
        for i in range(global_procs):
            hi(i, point=i)
    legion.execution_fence(block=True)

    assert global_var == 4456

if __name__ == '__legion_main__':
    main()
