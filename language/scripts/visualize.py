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
import subprocess, sys

def process(graph, output_filename):
    proc = subprocess.Popen(['dot', '-Kdot', '-Tpdf', '-o', output_filename], stdin = subprocess.PIPE)
    proc.communicate(graph)
    if proc.returncode != 0:
        assert False

def driver(prefix, suffix):
    run = []
    acc = False
    num = 1
    for line in sys.stdin.read().split('\n'):
        if line == 'digraph {':
            acc = True
        if acc:
            run.append(line)
        if acc and line == '}':
            acc = False
            process('\n'.join(run), '%s%s%s' % (prefix, num, suffix))
            run = []
            num += 1

if __name__ == '__main__':
    driver('test_', '.pdf')
