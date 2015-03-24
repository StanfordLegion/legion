#!/usr/bin/env python

# Copyright 2015 Stanford University
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
import os, platform, subprocess, sys

os_name = platform.system()

root_dir = os.path.realpath(os.path.dirname(__file__))
legion_dir = os.path.dirname(root_dir)

terra_dir = os.path.join(root_dir, 'terra')

runtime_dir = os.path.join(legion_dir, 'runtime')
bindings_dir = os.path.join(legion_dir, 'bindings', 'terra')

terra_path = [
    '?.t',
    os.path.join(root_dir, 'src', '?.t'),
    os.path.join(terra_dir, 'tests', 'lib', '?.t'),
    os.path.join(terra_dir, 'release', 'include', '?.t'),
    os.path.join(bindings_dir, '?.t'),
]

include_path = [
    bindings_dir,
    runtime_dir,
]

lib_path = [
    os.path.join(terra_dir, 'build'),
    bindings_dir,
]

LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
if os_name == 'Darwin':
    LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'

terra_exe = os.path.join(terra_dir, 'terra')
terra_env = dict(os.environ.items() + [
    ('TERRA_PATH', ';'.join(terra_path)),
    (LD_LIBRARY_PATH, ':'.join(lib_path)),
    ('INCLUDE_PATH', ';'.join(include_path)),
])

def legion(args, **kwargs):
    cmd = []
    if 'USE_MPIRUN' in os.environ and int(os.environ['USE_MPIRUN']) == 1:
        mpirun_flags = (
            (os.environ['MPIRUN_FLAGS'].split()
             if 'MPIRUN_FLAGS' in os.environ else []) +
            ['-x', 'TERRA_PATH', '-x', LD_LIBRARY_PATH, '-x', 'INCLUDE_PATH'])
        cmd = cmd + ['mpirun'] + mpirun_flags
    cmd = cmd + [terra_exe] + args
    return subprocess.Popen(
        cmd, env = terra_env, **kwargs)

if __name__ == '__main__':
    sys.exit(legion(sys.argv[1:]).wait())
