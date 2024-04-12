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
import argparse, os, platform, subprocess, sys

def test(root_dir, install_only, debug, max_dim, short, no_pretty,
         spy, prof, gcov, hdf5, cuda, hip, openmp, python, jobs, env):
    if 'TRAVIS' in env:
        install_threads = ['-j', '2']
        test_threads = ['-j', '2']
    else:
        # the multiprocessing pool in test.py will use os.cpu_count() which
        # assumes we only need one core/test (it's really 2+) and that there's
        # no cpu core restrictions (which can happen if multiple test runners share
        # a single physical node), so do the math ourselves
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # macos doesn't have sched_getaffinity
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
        install_threads = ['-j', str(num_cores)]
        # assume a non-empty LAUNCHER means we're running 2 processes/test
        if env.get('LAUNCHER'):
            cores_per_test = 4
        else:
            cores_per_test = 2
        num_tests = 1 + ((num_cores - 1) // cores_per_test)
        test_threads = ['-j', str(num_tests)]
        
    terra = ['--with-terra', env['TERRA_DIR']] if 'TERRA_DIR' in env else []
    build = (['--with-cmake-build', env['CMAKE_BUILD_DIR']]
             if env.get('USE_CMAKE') == '1' and 'CMAKE_BUILD_DIR' in env
             else [])
    debug_flag = ['--debug'] if debug else []
    max_dim_flag = ['--max-dim=%s' % max_dim]
    short_flag = ['--short'] if short else []
    no_pretty_flag = ['--no-pretty'] if no_pretty else []
    inner_flag = ['--extra=-flegion-inner', '--extra=0'] if 'DISABLE_INNER' in env else []
    out_dir_flag = ['--output=%s' % os.path.join(root_dir, 'test_output')]

    subprocess.check_call(
        [sys.executable, './install.py] + install_threads + terra + build + debug_flag,
        env = env,
        cwd = root_dir)
    if not install_only:
        extra_flags = []
        if spy: extra_flags.append('--spy')
        if prof: extra_flags.append('--prof')
        if gcov: extra_flags.append('--run')
        if hdf5: extra_flags.append('--hdf5')
        if cuda:
            extra_flags.append('--gpu=cuda')
            test_threads = ['-j', '1']  # do not oversubscribe GPU
        if hip:
            extra_flags.append('--gpu=hip')
            test_threads = ['-j', '1']  # do not oversubscribe GPU
        if openmp: extra_flags.append('--openmp')
        if python: extra_flags.append('--python')
        extra_flags.extend(['--extra=-fjobs', '--extra=%s' % jobs])
        # FIXME: this breaks on newer versions of Terra
        # if not spy and not prof and not gcov and not hdf5 and not openmp and not cuda:
        #     extra_flags.append('--debug')

        if (prof or spy) and 'TMP_BIN_DIR' in env:
            extra_flags.append('--legion-prof-rs=%s' % (
                os.path.join(env['TMP_BIN_DIR'], 'legion_prof')))

        subprocess.check_call(
            [sys.executable, './test.py', '-q'] + test_threads + max_dim_flag + short_flag + no_pretty_flag + extra_flags + inner_flag + out_dir_flag,
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

    env = dict(os.environ.items())
    env.update({
        'LG_RT_DIR': runtime_dir,
        # 'LUAJIT_URL': 'http://legion.stanford.edu/~eslaught/mirror/LuaJIT-2.0.4.tar.gz',
    })

    debug = env['DEBUG'] == '1'
    max_dim = int(env.get('MAX_DIM', 3))
    short = env.get('SHORT') == '1'
    no_pretty = env.get('NO_PRETTY') == '1'
    spy = env.get('TEST_SPY') == '1'
    prof = env.get('TEST_PROF') == '1'
    gcov = env.get('TEST_GCOV') == '1'
    hdf5 = env.get('TEST_HDF') == '1'
    cuda = env.get('TEST_CUDA') == '1'
    hip = env.get('TEST_HIP') == '1'
    openmp = env.get('TEST_OPENMP') == '1'
    python = env.get('TEST_PYTHON') == '1'
    jobs = int(env['REGENT_JOBS']) if 'REGENT_JOBS' in env else 1
    test(root_dir, args.install_only, debug, max_dim, short, no_pretty,
         spy, prof, gcov, hdf5, cuda, hip, openmp, python, jobs, env)
