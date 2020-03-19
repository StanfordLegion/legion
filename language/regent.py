#!/usr/bin/env python

# Copyright 2020 Stanford University
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
bindings_dir = os.path.join(os.path.dirname(runtime_dir), 'bindings', 'regent')
python_dir = os.path.join(os.path.dirname(runtime_dir), 'bindings', 'python')

# Find CUDA.
cuda_dir = os.environ.get('CUDA') or os.environ.get('CUDA_HOME') or os.environ.get('CUDATOOLKIT_HOME')
if cuda_dir:
    cuda_dir = os.path.realpath(cuda_dir)
cuda_include_dir = os.path.join(cuda_dir, 'include') if cuda_dir is not None else None

# Find RDIR.
if 'USE_RDIR' in os.environ:
    use_rdir = os.environ['USE_RDIR']
else:
    rdir_config_filename = os.path.join(regent_dir, '.rdir.json')
    rdir = load_json_config(rdir_config_filename)
    use_rdir = '1' if rdir in ['auto', 'manual'] else '0'

# Detect use of CMake.
if 'USE_CMAKE' in os.environ:
    cmake = os.environ['USE_CMAKE'] == '1'
else:
    cmake_config_filename = os.path.join(regent_dir, '.cmake.json')
    cmake = load_json_config(cmake_config_filename)
cmake_build_dir = os.path.join(regent_dir, 'build')

include_path = (
    (os.environ['INCLUDE_PATH'].split(';')
     if 'INCLUDE_PATH' in os.environ else []) +
    [bindings_dir,
     runtime_dir,
    ] +
    ([os.path.join(cmake_build_dir, 'runtime')] if cmake else []))
if cuda_include_dir is not None:
    include_path.append(cuda_include_dir)

LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
if os_name == 'Darwin':
    LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'

lib_path = (
    (os.environ[LD_LIBRARY_PATH].split(':')
     if LD_LIBRARY_PATH in os.environ else []) +
    [os.path.join(terra_dir, 'build'),
     (os.path.join(cmake_build_dir, 'lib') if cmake else bindings_dir)])

def root_dir():
    return os.path.dirname(runtime_dir)

def regent(args, env={}, cwd=None, **kwargs):
    terra_exe = os.path.join(terra_dir, 'terra')
    if not os.path.exists(terra_exe):
        terra_exe = os.path.join(terra_dir, 'bin', 'terra')
    if not os.path.exists(terra_exe):
        terra_exe = os.path.join(terra_dir, 'release', 'bin', 'terra')

    if 'TERRA_PATH' in os.environ:
        terra_path = os.environ['TERRA_PATH'].split(';')
    else:
        terra_path = []

    if 'PYTHONPATH' in os.environ:
        python_path = os.environ['PYTHONPATH'].split(':')
    else:
        python_path = []

    normal_args = [arg for arg in args if not arg.startswith('-')]
    first_arg = None
    if len(normal_args) >= 1:
        first_arg = os.path.realpath(
            os.path.join(cwd, normal_args[0]) if cwd is not None else normal_args[0])
    terra_path += (
        ['?.t', '?.rg'] +
        ([os.path.join(os.path.dirname(first_arg), '?.t'),
          os.path.join(os.path.dirname(first_arg), '?.rg')]
          if first_arg is not None and os.path.exists(first_arg) else []) +
        [os.path.join(regent_dir, 'src', '?.t'),
         os.path.join(regent_dir, 'src', 'rdir', 'plugin', 'src', '?.t'),
         os.path.join(terra_dir, 'tests', 'lib', '?.t'),
         os.path.join(terra_dir, 'release', 'include', '?.t'),
         os.path.join(bindings_dir, '?.t')])

    if first_arg is not None:
        python_path.append(os.path.dirname(first_arg))
    python_path.append(python_dir)

    terra_env = {
        'TERRA_PATH': ';'.join(terra_path),
        LD_LIBRARY_PATH: ':'.join(lib_path),
        'INCLUDE_PATH': ';'.join(include_path),
        'PYTHONPATH': ':'.join(python_path),
        'LG_RT_DIR': runtime_dir,
        'USE_CMAKE': '1' if cmake else '0',
        'CMAKE_BUILD_DIR': cmake_build_dir,
        'USE_RDIR': use_rdir,
    }

    if cuda_dir is not None:
        terra_env['CUDA_HOME'] = cuda_dir

    cmd = []
    if 'LAUNCHER' in os.environ:
        cmd = cmd + (os.environ['LAUNCHER'].split()
                     if 'LAUNCHER' in os.environ else [])
    cmd = cmd + [terra_exe] + args
    cmd_env = dict(os.environ.items())
    cmd_env.update(terra_env)
    cmd_env.update(env)
    try:
        return subprocess.Popen(
            cmd, env=cmd_env, cwd=cwd, **kwargs)
    except OSError:
        print('Command failed: %s' % cmd, file=sys.stderr)
        sys.stderr.flush()
        raise

if __name__ == '__main__':
    sys.exit(regent(sys.argv[1:]).wait())
