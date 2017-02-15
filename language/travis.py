#!/usr/bin/env python

# Copyright 2017 Stanford University
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
import argparse, os, platform, subprocess

def test(root_dir, install_only, debug, spy, hdf5, env):
    threads = ['-j', '2'] if 'TRAVIS' in env else []
    terra = ['--with-terra', env['TERRA_DIR']] if 'TERRA_DIR' in env else []
    debug_flag = ['--debug'] if debug else []
    inner_flag = ['--extra=-flegion-inner', '--extra=0'] if 'DISABLE_INNER' in env else []
    if 'USE_RDIR' in env:
        regent_dir = os.path.dirname(os.path.realpath(__file__))
        rdir_config = os.path.join(regent_dir, '.rdir.json')
        if env['USE_RDIR'] == '1' and not os.path.exists(rdir_config):
            rdir = 'auto'
        else:
            rdir = 'skip'
    else:
        rdir = 'auto'

    subprocess.check_call(
        ['time', './install.py', '--rdir=%s' % rdir] + threads + terra + debug_flag,
        env = env,
        cwd = root_dir)
    if not install_only:
        if spy:
            subprocess.check_call(
                ['time', './test.py', '-q', '--spy'] + threads + inner_flag,
                env = env,
                cwd = root_dir)
        elif hdf5:
            subprocess.check_call(
                ['time', './test.py', '-q', '--hdf5'] + threads + inner_flag,
                env = env,
                cwd = root_dir)
        else:
            subprocess.check_call(
                ['time', './test.py', '-q'] + threads + debug_flag + inner_flag,
                env = env,
                cwd = root_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regent compiler test suite')
    parser.add_argument('--install-only',
                        action='store_true',
                        help='Only install, do not run tests',
                        dest='install_only')
    args = parser.parse_args()

    root_dir = os.path.realpath(os.path.dirname(__file__))
    legion_dir = os.path.dirname(root_dir)
    runtime_dir = os.path.join(legion_dir, 'runtime')

    env = dict(os.environ.iteritems())
    env.update({
        'LG_RT_DIR': runtime_dir,
        'LUAJIT_URL': 'http://legion.stanford.edu/~eslaught/mirror/LuaJIT-2.0.4.tar.gz',
    })

    debug = env['DEBUG'] == '1'
    spy = 'TEST_SPY' in env and env['TEST_SPY'] == '1'
    hdf5 = 'TEST_HDF' in env and env['TEST_HDF'] == '1'
    test(root_dir, args.install_only, debug, spy, hdf5, env)
