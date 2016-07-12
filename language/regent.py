#!/usr/bin/env python

# Copyright 2016 Stanford University
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
import json, os, platform, subprocess, sys

def load_json_config(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except IOError:
        return None

os_name = platform.system()

# Find Regent.
regent_dir = os.path.dirname(os.path.realpath(__file__))
terra_dir = os.path.join(regent_dir, 'terra')

# Find Legion (in the environment, or relative to Regent).
if 'LG_RT_DIR' in os.environ:
    runtime_dir = os.path.realpath(os.environ['LG_RT_DIR'])
else:
    runtime_dir = os.path.join(os.path.dirname(regent_dir), 'runtime')
realm_dir = os.path.join(runtime_dir, 'realm')
mapper_dir = os.path.join(runtime_dir, 'mappers')
legion_runtime_dir = os.path.join(runtime_dir, 'legion')
bindings_dir = os.path.join(os.path.dirname(runtime_dir), 'bindings', 'terra')

# Find CUDA.
if 'CUDA' in os.environ:
    cuda_dir = os.path.realpath(os.environ['CUDA'])
elif 'CUDATOOLKIT_HOME' in os.environ:
    cuda_dir = os.path.realpath(os.environ['CUDATOOLKIT_HOME'])
else:
    cuda_dir = None
cuda_include_dir = os.path.join(cuda_dir, 'include') if cuda_dir is not None else None

# Find RDIR.
rdir_config_filename = os.path.join(regent_dir, '.rdir.json')
rdir = load_json_config(rdir_config_filename)
use_rdir = '1' if rdir in ['auto', 'manual'] else '0'

include_path = [
    bindings_dir,
    runtime_dir,
    realm_dir,
    mapper_dir,
    legion_runtime_dir,
]
if cuda_include_dir is not None:
    include_path.append(cuda_include_dir)

LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
if os_name == 'Darwin':
    LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'

lib_path = (
    (os.environ[LD_LIBRARY_PATH].split(':')
     if LD_LIBRARY_PATH in os.environ else []) +
    [os.path.join(terra_dir, 'build'),
     bindings_dir,
 ])

def root_dir():
    return os.path.dirname(runtime_dir)

def regent(args, env = {}, **kwargs):
    terra_exe = os.path.join(terra_dir, 'terra')
    if not os.path.exists(terra_exe):
        terra_exe = os.path.join(terra_dir, 'bin', 'terra')

    if 'TERRA_PATH' in os.environ:
        terra_path = [os.path.realpath(os.environ['TERRA_PATH'])]
    else:
        terra_path = []

    normal_args = [arg for arg in args if not arg.startswith('-')]
    terra_path += (
        ['?.t', '?.rg'] +
        ([os.path.join(os.path.dirname(os.path.realpath(normal_args[0])), '?.t'),
          os.path.join(os.path.dirname(os.path.realpath(normal_args[0])), '?.rg')]
          if len(normal_args) >= 1 and os.path.exists(normal_args[0]) else []) +
        [os.path.join(regent_dir, 'src', '?.t'),
         os.path.join(regent_dir, 'src', 'rdir', 'plugin', 'src', '?.t'),
         os.path.join(terra_dir, 'tests', 'lib', '?.t'),
         os.path.join(terra_dir, 'release', 'include', '?.t'),
         os.path.join(bindings_dir, '?.t')])

    terra_env = {
        'TERRA_PATH': ';'.join(terra_path),
        LD_LIBRARY_PATH: ':'.join(lib_path),
        'INCLUDE_PATH': ';'.join(include_path),
        'LG_RT_DIR': runtime_dir,
        'USE_RDIR': use_rdir,
    }

    cmd = []
    if 'LAUNCHER' in os.environ:
        cmd = cmd + (os.environ['LAUNCHER'].split()
                     if 'LAUNCHER' in os.environ else [])
    cmd = cmd + [terra_exe] + args
    cmd_env = dict(os.environ.items())
    cmd_env.update(terra_env)
    cmd_env.update(env)
    return subprocess.Popen(
        cmd, env = cmd_env, **kwargs)

if __name__ == '__main__':
    sys.exit(regent(sys.argv[1:]).wait())
